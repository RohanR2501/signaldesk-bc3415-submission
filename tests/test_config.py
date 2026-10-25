from pathlib import Path

import pytest
from pydantic import ValidationError

from signaldesk.config import PROJECT_ROOT, Settings


def test_environment_overrides_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALDESK_ENVIRONMENT", "test")
    monkeypatch.setenv("SIGNALDESK_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SIGNALDESK_CORS_ORIGINS", '["http://localhost:9999"]')
    settings = Settings(_env_file=None)
    assert settings.environment == "test"
    assert settings.log_level == "WARNING"
    assert settings.cors_origins == ["http://localhost:9999"]


def test_invalid_environment_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALDESK_ENVIRONMENT", "invalid")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_relative_database_path_is_independent_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, database_path=Path(".runtime/test.sqlite3"))
    assert settings.database_path == PROJECT_ROOT / ".runtime/test.sqlite3"
    assert not settings.database_path.exists()


def test_absolute_database_path_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite3"
    assert Settings(_env_file=None, database_path=path).database_path == path


def test_dotenv_loads_without_secrets(tmp_path: Path) -> None:
    path = tmp_path / "settings.env"
    path.write_text("SIGNALDESK_LOG_LEVEL=ERROR\n", encoding="utf-8")
    assert Settings(_env_file=path).log_level == "ERROR"
