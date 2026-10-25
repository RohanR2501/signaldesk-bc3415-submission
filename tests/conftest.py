from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from signaldesk.config import Settings
from signaldesk.main import create_app


@pytest.fixture
def app() -> FastAPI:
    return create_app(Settings(_env_file=None, environment="test"))


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
