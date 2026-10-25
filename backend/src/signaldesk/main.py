"""Application composition with lazy local analysis and historical intelligence."""

import json
import logging
from collections.abc import Mapping
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from signaldesk import __version__
from signaldesk.analyzer import ReviewAnalyzer
from signaldesk.config import Settings
from signaldesk.contracts import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorBody,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
)
from signaldesk.history_api import HistoricalStore
from signaldesk.history_api import router as history_router

logger = logging.getLogger("signaldesk")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "event": record.getMessage(),
                "request_id": getattr(record, "request_id", None),
                "method": getattr(record, "method", None),
                "route": getattr(record, "route", None),
                "status": getattr(record, "status", None),
                "duration_ms": getattr(record, "duration_ms", None),
                "exception_type": getattr(record, "exception_type", None),
            }
        )


def error_response(
    request: Request,
    status: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details or []),
        request_id=request.state.request_id,
    )
    return JSONResponse(payload.model_dump(mode="json"), status_code=status, headers=headers)


class RequestContextMiddleware:
    """Keep error responses and request IDs consistent without logging review content."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())  # Generate locally; do not trust inbound correlation headers.
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        status = 500
        response_started = False

        async def send_with_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception as exc:
            logger.error(
                "unhandled_error",
                extra={"request_id": request_id, "exception_type": type(exc).__name__},
            )
            if response_started:
                raise
            response = error_response(
                Request(scope), 500, "internal_error", "An unexpected error occurred."
            )
            await response(scope, receive, send_with_id)
        finally:
            route = scope.get("route")
            logger.info(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "route": getattr(route, "path", "<unmatched>"),
                    "status": status,
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(config.log_level)
    logger.propagate = False

    app = FastAPI(
        title="SignalDesk API",
        version=__version__,
        description=(
            "Local review analysis and historical product intelligence. "
            "Signals are not verified incidents."
        ),
        responses={status: {"model": ErrorResponse} for status in (404, 405, 422, 500, 503)},
    )
    app.state.settings = config
    app.state.analyzer = ReviewAnalyzer(config.transformer_directory, config.baseline_directory)
    app.state.history = HistoricalStore()
    # CORS wraps request context so even handled 500 responses include allowed-origin headers.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
        messages = {
            404: "Endpoint or resource not found.",
            405: "Method not allowed.",
            503: "Historical data is missing, incompatible or failed integrity checks.",
        }
        return error_response(
            request,
            exc.status_code,
            f"http_{exc.status_code}",
            messages.get(exc.status_code, "The request could not be completed."),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Never echo the raw input, exception message or validation context into logs/responses.
        details = [ErrorDetail(location=list(e["loc"]), type=e["type"]) for e in exc.errors()]
        return error_response(
            request, 422, "validation_error", "Request validation failed.", details
        )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Liveness only: this does not attest to data, database or model readiness."""
        return HealthResponse(version=__version__)

    @app.post("/api/v1/reviews/analyze", response_model=AnalyzeResponse, tags=["review analysis"])
    def analyze_review(payload: AnalyzeRequest, response: Response) -> AnalyzeResponse:
        """Use local saved sentiment models and independent rules without storing submitted text."""
        response.headers["Cache-Control"] = "no-store"
        analyzer: ReviewAnalyzer = app.state.analyzer
        return analyzer.analyze(payload.text)

    app.include_router(history_router)

    return app


app = create_app()
