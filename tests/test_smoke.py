from __future__ import annotations

import pytest
from typer.testing import CliRunner

from formuloom import __version__
from formuloom.cli import app
from formuloom.constants import (
    GROUP_WEIGHT_FINAL,
    GROUP_WEIGHT_INTERMEDIATE,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MAX_CONCURRENT_LLM_CALLS,
    SECTION_SPLIT_ROW_THRESHOLD,
    SPEC_VERSION,
    TASK_TOLERANCE,
)
from formuloom.settings import Settings, get_settings

runner = CliRunner()


def test_version_string() -> None:
    assert __version__ == "0.1.0"


def test_cli_version_flag_exits_zero() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_no_args_does_not_crash() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)


def test_diff_policy_constants() -> None:
    assert SPEC_VERSION == 2
    assert TASK_TOLERANCE == 0.01
    assert GROUP_WEIGHT_INTERMEDIATE == 0.6
    assert GROUP_WEIGHT_FINAL == 0.4
    assert MAX_CONCURRENT_LLM_CALLS == 24
    assert LLM_TIMEOUT_SECONDS == 120
    assert LLM_MAX_RETRIES == 3
    assert SECTION_SPLIT_ROW_THRESHOLD == 150


def test_get_settings_returns_settings_instance() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.model == "gpt-5.4-mini"
    assert settings.reasoning_effort == "low"
    assert settings.max_concurrent_llm_calls == MAX_CONCURRENT_LLM_CALLS
    assert settings.llm_timeout_seconds == LLM_TIMEOUT_SECONDS


def test_settings_constructs_without_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]  # pydantic-settings runtime kwarg, untyped in stubs
    assert settings.openai_api_key is None


def test_settings_loads_openai_api_key_from_env_without_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-marker-not-a-real-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]  # pydantic-settings runtime kwarg, untyped in stubs
    assert settings.openai_api_key == "sk-test-marker-not-a-real-key"
