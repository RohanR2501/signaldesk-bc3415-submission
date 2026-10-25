import json
import logging
from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from signaldesk.contracts import CustomerReview
from signaldesk.main import JsonFormatter, logger


def test_health_is_truthful_liveness(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.1",
        "status": "ok",
        "service": "signaldesk",
        "version": "0.1.0",
        "api_version": "v1",
        "phase": 4,
        "model_status": "not_checked",
    }
    UUID(response.headers["X-Request-ID"])


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("GET", "/api/v1/missing", 404), ("POST", "/api/v1/health", 405)],
)
def test_unimplemented_and_invalid_methods_are_errors(
    client: TestClient, method: str, path: str, status: int
) -> None:
    response = client.request(method, path)
    assert response.status_code == status
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["error"]["code"] == f"http_{status}"
    assert payload["request_id"] == response.headers["X-Request-ID"]
    if status == 405:
        assert "GET" in response.headers["allow"]


def test_request_id_is_generated_and_unique(client: TestClient) -> None:
    first = client.get("/api/v1/health", headers={"X-Request-ID": "untrusted"})
    second = client.get("/api/v1/health")
    assert first.headers["X-Request-ID"] != "untrusted"
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


def test_validation_never_echoes_review_text(app: FastAPI, client: TestClient) -> None:
    @app.post("/_test/validation")
    async def validate_review(review: CustomerReview) -> CustomerReview:
        return review

    secret = "private-customer-content"
    response = client.post("/_test/validation", json={"text": secret, "rating": 999})
    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["details"]
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_internal_error_is_sanitized_and_cors_survives(app: FastAPI, client: TestClient) -> None:
    @app.get("/_test/error")
    async def fail() -> None:
        raise RuntimeError("private-customer-content")

    response = client.get("/_test/error", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private-customer-content" not in response.text
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    allowed = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    denied = client.get("/api/v1/health", headers={"Origin": "https://other.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in denied.headers


def test_cors_preflight(client: TestClient) -> None:
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_openapi_advertises_implemented_routes(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert set(response.json()["paths"]) == {
        "/api/v1/health",
        "/api/v1/reviews/analyze",
        "/api/v1/summary",
        "/api/v1/products",
        "/api/v1/products/{product_id}",
        "/api/v1/products/{product_id}/timeline",
        "/api/v1/investigations",
        "/api/v1/investigations/{signal_id}",
        "/api/v1/reviews/{review_id}",
    }


@pytest.fixture
def log_records() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_logs_exclude_raw_paths_queries_bodies_and_exception_messages(
    app: FastAPI, client: TestClient, log_records: list[logging.LogRecord]
) -> None:
    secret = "private-customer-content"

    @app.get("/_test/log-error")
    async def fail() -> None:
        raise RuntimeError(secret)

    client.post(f"/missing/{secret}?text={secret}", json={"text": secret})
    client.get("/_test/log-error")
    formatted = [json.loads(JsonFormatter().format(record)) for record in log_records]
    assert formatted
    assert secret not in json.dumps(formatted)
    events = {record["event"] for record in formatted}
    assert {"http_request", "unhandled_error"} <= events
    assert any(record["route"] == "<unmatched>" for record in formatted)
    assert all(record["request_id"] for record in formatted)
