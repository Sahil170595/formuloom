from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, RateLimitError
from openai.types.responses import Response

from formuloom import adjudicate, classify
from formuloom.adjudicate import adjudicate_sheet
from formuloom.settings import Settings


def _settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "model": "gpt-5.4-mini",
        "reasoning_effort": "low",
        "service_tier": None,
        "max_concurrent_llm_calls": 4,
        "llm_timeout_seconds": 5,
        "_env_file": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _usage_payload(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _verdicts(**row_to_verdict: str) -> str:
    rows = [
        {"row": int(r), "label": f"label {r}", "verdict": v, "reason": "because"} for r, v in row_to_verdict.items()
    ]
    return json.dumps({"rows": rows})


def _message_response(text: str, *, usage: dict[str, Any] | None = None) -> Response:
    payload = {
        "id": "resp_msg",
        "created_at": 1.0,
        "model": "gpt-5.4-mini",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": usage if usage is not None else _usage_payload(200, 20),
    }
    return Response.model_validate(payload)


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code=429, request=request, json={"error": {"message": "rate limited"}})
    return RateLimitError("rate limited", response=response, body=None)


def _fake_client(steps: list[Exception | Response]) -> tuple[AsyncOpenAI, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    remaining = list(steps)

    async def fake_create(**kwargs: Any) -> Response:
        calls.append(kwargs)
        step = remaining.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    client = AsyncOpenAI(api_key="test-key", max_retries=0)
    client.responses.create = fake_create  # type: ignore[assignment]
    return client, calls


def _speed_up_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classify, "RETRY_WAIT_MULTIPLIER_SECONDS", 0.001)
    monkeypatch.setattr(classify, "RETRY_WAIT_MAX_SECONDS", 0.01)


def _run(client: AsyncOpenAI, settings: Settings, tmp_path: Path, *, proposed: list[int], **kw: Any) -> Any:
    async def run() -> Any:
        return await adjudicate_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V7",
            prompt_version="v4",
            input_text="ADJUDICATE ... row 16 ... row 40 ...",
            proposed_rows=proposed,
            run_dir=tmp_path,
            **kw,
        )

    return asyncio.run(run())


def test_adjudicate_keeps_and_drops(tmp_path: Path) -> None:
    client, calls = _fake_client([_message_response(_verdicts(**{"16": "keep", "40": "drop"}))])
    result = _run(client, _settings(), tmp_path, proposed=[16, 40])
    assert result.keep_rows == (16,)
    assert result.dropped_rows == (40,)
    assert result.degraded is False
    assert result.from_cache is False
    assert len(calls) == 1
    fmt = calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["name"] == adjudicate.ADJUDICATE_SCHEMA_NAME


def test_adjudicate_membership_mismatch_retries_then_uses_correction(tmp_path: Path) -> None:
    first = _message_response(_verdicts(**{"16": "keep"}))
    second = _message_response(_verdicts(**{"16": "keep", "40": "drop"}))
    client, calls = _fake_client([first, second])
    result = _run(client, _settings(), tmp_path, proposed=[16, 40])
    assert result.keep_rows == (16,)
    assert result.dropped_rows == (40,)
    assert result.degraded is False
    assert len(calls) == 2
    assert "CORRECTION" in calls[1]["input"]


def test_adjudicate_membership_mismatch_twice_degrades_keep_all(tmp_path: Path) -> None:
    bad = _message_response(_verdicts(**{"16": "keep"}))
    client, calls = _fake_client([bad, bad])
    result = _run(client, _settings(), tmp_path, proposed=[16, 40])
    assert result.degraded is True
    assert result.keep_rows == (16, 40)
    assert result.dropped_rows == ()
    assert len(calls) == 2

    cache_dir = tmp_path / classify.CACHE_SUBDIR
    assert not (cache_dir.exists() and list(cache_dir.glob("*.json")))


def test_adjudicate_transient_failure_degrades_keep_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_retries(monkeypatch)
    client, _calls = _fake_client([_rate_limit_error(), _rate_limit_error(), _rate_limit_error()])
    result = _run(client, _settings(), tmp_path, proposed=[16, 40], max_retries=3)
    assert result.degraded is True
    assert result.keep_rows == (16, 40)


def test_adjudicate_malformed_json_degrades_keep_all(tmp_path: Path) -> None:
    client, _calls = _fake_client([_message_response("{not valid")])
    result = _run(client, _settings(), tmp_path, proposed=[16])
    assert result.degraded is True
    assert result.keep_rows == (16,)


def test_adjudicate_empty_proposed_makes_no_call(tmp_path: Path) -> None:
    client, calls = _fake_client([])
    result = _run(client, _settings(), tmp_path, proposed=[])
    assert result.keep_rows == ()
    assert result.degraded is False
    assert len(calls) == 0


def test_adjudicate_cache_hit_skips_api_call(tmp_path: Path) -> None:
    client, calls = _fake_client([_message_response(_verdicts(**{"16": "keep", "40": "drop"}))])
    first = _run(client, _settings(), tmp_path, proposed=[16, 40])
    assert first.from_cache is False
    second = _run(client, _settings(), tmp_path, proposed=[16, 40])
    assert second.from_cache is True
    assert second.keep_rows == (16,)
    assert len(calls) == 1


def test_adjudicate_uses_adjudicator_model_override(tmp_path: Path) -> None:
    client, calls = _fake_client([_message_response(_verdicts(**{"16": "keep"}))])
    _run(client, _settings(), tmp_path, proposed=[16], model="gpt-5.5")
    assert calls[0]["model"] == "gpt-5.5"
