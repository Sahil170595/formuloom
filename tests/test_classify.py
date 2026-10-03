from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI, RateLimitError
from openai.types.responses import Response

from formuloom import classify
from formuloom.classify import (
    ClassifyError,
    InvalidJSONError,
    RefusalError,
    RowFeatureView,
    SheetClassifyError,
    TaskProfileError,
    TruncatedResponseError,
    UnknownReferenceError,
    classify_sheet,
    classify_sheet_voted,
    classify_v0,
    classify_v0_row,
    compute_cost_usd,
    fetch_task_profile,
    get_llm_semaphore,
    model_price,
)
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


def _usage_payload(
    input_tokens: int, output_tokens: int, cached_tokens: int = 0, reasoning_tokens: int = 0
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _message_response(
    text: str, *, status: str = "completed", usage: dict[str, Any] | None = None, response_id: str = "resp_msg"
) -> Response:
    payload = {
        "id": response_id,
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
        "status": status,
        "usage": usage if usage is not None else _usage_payload(100, 20),
    }
    return Response.model_validate(payload)


def _refusal_response(refusal_text: str) -> Response:
    payload = {
        "id": "resp_refusal",
        "created_at": 1.0,
        "model": "gpt-5.4-mini",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": refusal_text}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": _usage_payload(50, 5),
    }
    return Response.model_validate(payload)


def _incomplete_response(reason: str = "max_output_tokens") -> Response:
    payload = {
        "id": "resp_incomplete",
        "created_at": 1.0,
        "model": "gpt-5.4-mini",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "incomplete",
        "incomplete_details": {"reason": reason},
        "usage": _usage_payload(50, 5),
    }
    return Response.model_validate(payload)


def _empty_content_response() -> Response:
    payload = {
        "id": "resp_empty",
        "created_at": 1.0,
        "model": "gpt-5.4-mini",
        "object": "response",
        "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": []}],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": _usage_payload(10, 0),
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
    client.responses.create = fake_create  # type: ignore[assignment]  # queue-driven test double for the SDK's overloaded method
    return client, calls


def _speed_up_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(classify, "RETRY_WAIT_MULTIPLIER_SECONDS", 0.001)
    monkeypatch.setattr(classify, "RETRY_WAIT_MAX_SECONDS", 0.01)


@dataclass
class _Row:

    n_dependents_outside_row: int
    aggregates_range: bool
    lexicon_hits: list[str]
    input_colored_fraction: float
    bold_or_bordered: bool
    label: str | None
    has_formula: bool


def _protocol_accepts(row: _Row) -> RowFeatureView:
    return row


def test_classify_sheet_row_mode_happy_path(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "rows 2 and 4 are checkpoints", "final_rows": [2, 4]})
        client, calls = _fake_client([_message_response(text, usage=_usage_payload(500, 50, cached_tokens=100))])
        settings = _settings()
        result = await classify_sheet(
            client,
            settings,
            task="T1",
            sheet="S1",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="row 1 | A | =X | A1:A1 | currency\nrow 2 | B | =Y | A2:A2 | currency",
            candidates=[1, 2, 3, 4],
            run_dir=tmp_path,
        )
        assert result.final_refs == (2, 4)
        assert result.notes == "rows 2 and 4 are checkpoints"
        assert result.from_cache is False
        assert result.mode == "row"
        assert len(calls) == 1
        call = calls[0]
        assert call["model"] == settings.model
        assert "row 1 | A" in call["input"]
        assert call["reasoning"] == {"effort": "low"}
        assert call["text"]["format"]["type"] == "json_schema"
        assert call["text"]["format"]["strict"] is True
        assert call["text"]["format"]["name"] == "row_classification"
        assert call["text"]["format"]["schema"] == classify.ROW_JSON_SCHEMA
        assert "service_tier" not in call

    asyncio.run(run())


def test_classify_sheet_cell_mode_happy_path(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "D16 is the checkpoint", "final_cells": ["D16"]})
        client, calls = _fake_client([_message_response(text)])
        settings = _settings()
        result = await classify_sheet(
            client,
            settings,
            task="T1",
            sheet="S1",
            variant_name="V1",
            prompt_version="v1",
            mode="cell",
            context="D14 | Average debt balance\nD16 | Interest expense",
            candidates=["D14", "D15", "D16"],
            run_dir=tmp_path,
        )
        assert result.final_refs == ("D16",)
        assert result.mode == "cell"
        assert calls[0]["text"]["format"]["name"] == "cell_classification"
        assert calls[0]["text"]["format"]["schema"] == classify.CELL_JSON_SCHEMA

    asyncio.run(run())


def test_classify_sheet_passes_service_tier_when_set(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "n", "final_rows": [1]})
        client, calls = _fake_client([_message_response(text)])
        settings = _settings(service_tier="flex")
        await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1],
            run_dir=tmp_path,
        )
        assert calls[0]["service_tier"] == "flex"

    asyncio.run(run())


def test_classify_sheet_retries_on_rate_limit_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_retries(monkeypatch)

    async def run() -> None:
        text = json.dumps({"notes": "n", "final_rows": [1]})
        client, calls = _fake_client([_rate_limit_error(), _message_response(text)])
        settings = _settings()
        result = await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1],
            run_dir=tmp_path,
            max_retries=3,
        )
        assert result.final_refs == (1,)
        assert len(calls) == 2

    asyncio.run(run())


def test_classify_sheet_exhausts_retries_raises_sheet_classify_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _speed_up_retries(monkeypatch)

    async def run() -> None:
        client, calls = _fake_client([_rate_limit_error(), _rate_limit_error(), _rate_limit_error()])
        settings = _settings()
        with pytest.raises(SheetClassifyError) as excinfo:
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1],
                run_dir=tmp_path,
                max_retries=3,
            )
        assert excinfo.value.sheet == "S"
        assert len(calls) == 3

        cache_path = classify._sheet_cache_path(tmp_path, task="T", sheet="S", variant_name="V1", prompt_version="v1")
        assert not cache_path.exists()

    asyncio.run(run())


def test_classify_sheet_refusal_raises_typed_error(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_refusal_response("cannot classify this content")])
        settings = _settings()
        with pytest.raises(RefusalError) as excinfo:
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1],
                run_dir=tmp_path,
            )
        assert excinfo.value.sheet == "S"
        assert "cannot classify" in excinfo.value.refusal_text

    asyncio.run(run())


def test_classify_sheet_truncated_status_raises_typed_error(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_incomplete_response("max_output_tokens")])
        settings = _settings()
        with pytest.raises(TruncatedResponseError) as excinfo:
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="cell",
                context="ctx",
                candidates=["A1"],
                run_dir=tmp_path,
            )
        assert excinfo.value.reason == "max_output_tokens"

    asyncio.run(run())


def test_classify_sheet_empty_output_text_raises_truncated_error(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_empty_content_response()])
        settings = _settings()
        with pytest.raises(TruncatedResponseError):
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1],
                run_dir=tmp_path,
            )

    asyncio.run(run())


def test_classify_sheet_invalid_json_raises_typed_error(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_message_response("not json at all")])
        settings = _settings()
        with pytest.raises(InvalidJSONError) as excinfo:
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1],
                run_dir=tmp_path,
            )
        assert excinfo.value.raw_text == "not json at all"

    asyncio.run(run())


def test_classify_sheet_wrong_shape_json_raises_typed_error(tmp_path: Path) -> None:
    async def run() -> None:

        client, _calls = _fake_client([_message_response(json.dumps({"notes": "n", "final_rows": ["1"]}))])
        settings = _settings()
        with pytest.raises(InvalidJSONError):
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1],
                run_dir=tmp_path,
            )

    asyncio.run(run())


def test_classify_sheet_unknown_ref_corrective_retry_succeeds(tmp_path: Path) -> None:
    async def run() -> None:
        bad = json.dumps({"notes": "n", "final_rows": [1, 99]})
        good = json.dumps({"notes": "corrected", "final_rows": [1]})
        client, calls = _fake_client([_message_response(bad), _message_response(good)])
        settings = _settings()
        result = await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1, 2],
            run_dir=tmp_path,
        )
        assert result.final_refs == (1,)
        assert result.notes == "corrected"
        assert len(calls) == 2
        assert "CORRECTION" in calls[1]["input"]
        assert "99" in calls[1]["input"]

    asyncio.run(run())


def test_classify_sheet_unknown_ref_after_correction_raises(tmp_path: Path) -> None:
    async def run() -> None:
        bad = json.dumps({"notes": "n", "final_rows": [99]})
        still_bad = json.dumps({"notes": "n2", "final_rows": [100]})
        client, calls = _fake_client([_message_response(bad), _message_response(still_bad)])
        settings = _settings()
        with pytest.raises(UnknownReferenceError) as excinfo:
            await classify_sheet(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1, 2],
                run_dir=tmp_path,
            )
        assert "100" in {r for r in excinfo.value.unknown_refs}

        assert len(calls) == 2

    asyncio.run(run())


def test_classify_sheet_cache_hit_skips_api_call(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "n", "final_rows": [1]})
        client, calls = _fake_client([_message_response(text)])
        settings = _settings()
        first = await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1],
            run_dir=tmp_path,
        )
        assert first.from_cache is False
        assert len(calls) == 1

        second = await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1],
            run_dir=tmp_path,
        )
        assert second.from_cache is True
        assert second.final_refs == (1,)
        assert second.notes == "n"
        assert len(calls) == 1

    asyncio.run(run())


def test_classify_sheet_writes_cache_file(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "n", "final_rows": [1]})
        client, _calls = _fake_client([_message_response(text)])
        settings = _settings()
        await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1],
            run_dir=tmp_path,
        )
        cache_path = classify._sheet_cache_path(tmp_path, task="T", sheet="S", variant_name="V1", prompt_version="v1")
        assert cache_path.is_file()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert payload["final_refs"] == [1]
        assert payload["mode"] == "row"
        assert list(cache_path.parent.glob("*.tmp")) == []

    asyncio.run(run())


def test_classify_sheet_voted_uses_distinct_cache_per_sample(tmp_path: Path) -> None:
    async def run() -> None:
        samples = [json.dumps({"notes": f"n{i}", "final_rows": [1]}) for i in range(3)]
        client, calls = _fake_client([_message_response(t) for t in samples])
        settings = _settings()
        voted = await classify_sheet_voted(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1, 2],
            run_dir=tmp_path,
            k=3,
        )
        assert len(calls) == 3
        assert voted.k == 3
        cache_dir = tmp_path / classify.CACHE_SUBDIR
        assert len(list(cache_dir.glob("*.json"))) == 3

    asyncio.run(run())


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "cache.json"
    classify._atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
    assert path.is_file()
    assert classify._read_cache_json(path) == {"a": 1, "b": [1, 2, 3]}
    assert list(path.parent.glob("*.tmp")) == []


def test_atomic_write_json_no_partial_file_on_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "sub" / "cache.json"

    def boom(_src: str, _dst: str) -> None:
        raise OSError("simulated crash between temp write and replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        classify._atomic_write_json(target, {"a": 1})

    assert not target.exists()
    assert list(target.parent.glob(f"{target.name}.*.tmp")) == []


def test_atomic_write_json_preserves_prior_good_version_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "cache.json"
    classify._atomic_write_json(target, {"version": "good"})

    def boom(_src: str, _dst: str) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        classify._atomic_write_json(target, {"version": "bad"})

    assert classify._read_cache_json(target) == {"version": "good"}


def test_read_cache_json_missing_file_is_none(tmp_path: Path) -> None:
    assert classify._read_cache_json(tmp_path / "nope.json") is None


def test_read_cache_json_corrupt_file_is_none(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert classify._read_cache_json(path) is None


def test_usage_and_cost_accounting_single_call(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "n", "final_rows": [1]})
        usage_payload = _usage_payload(1000, 200, cached_tokens=400)
        client, _calls = _fake_client([_message_response(text, usage=usage_payload)])
        settings = _settings(model="gpt-5.4-mini")
        result = await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1],
            run_dir=tmp_path,
        )
        assert result.usage.input_tokens == 1000
        assert result.usage.cached_input_tokens == 400
        assert result.usage.output_tokens == 200
        assert result.usage.api_calls == 1
        expected_cost = compute_cost_usd("gpt-5.4-mini", input_tokens=1000, cached_input_tokens=400, output_tokens=200)
        assert result.usage.cost_usd == pytest.approx(expected_cost)
        assert result.usage.cost_usd > 0

    asyncio.run(run())


def test_usage_and_cost_accounting_includes_corrective_retry(tmp_path: Path) -> None:
    async def run() -> None:
        bad = json.dumps({"notes": "n", "final_rows": [99]})
        good = json.dumps({"notes": "n2", "final_rows": [1]})
        client, _calls = _fake_client(
            [
                _message_response(bad, usage=_usage_payload(500, 50)),
                _message_response(good, usage=_usage_payload(600, 60)),
            ]
        )
        settings = _settings()
        result = await classify_sheet(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1, 2],
            run_dir=tmp_path,
        )
        assert result.usage.api_calls == 2
        assert result.usage.input_tokens == 1100
        assert result.usage.output_tokens == 110
        assert result.usage.cost_usd > 0

    asyncio.run(run())


def test_compute_cost_usd_standard_input_and_output() -> None:
    cost = compute_cost_usd("gpt-5.4-nano", input_tokens=1_000_000, cached_input_tokens=0, output_tokens=1_000_000)
    assert cost == pytest.approx(0.20 + 1.25)


def test_compute_cost_usd_uses_cached_rate() -> None:
    cost = compute_cost_usd("gpt-5.4", input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(0.25)


def test_compute_cost_usd_mixed_cached_and_uncached() -> None:

    cost = compute_cost_usd("gpt-5.4-mini", input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=0)
    assert cost == pytest.approx(0.5 * 0.75 + 0.5 * 0.075)


def test_model_price_unknown_model_raises() -> None:
    with pytest.raises(ClassifyError, match="no price table entry"):
        model_price("gpt-4o")


V0_CASES: list[tuple[str, _Row, bool]] = [
    (
        "sink_row_with_lexicon_hit_is_final",
        _Row(
            n_dependents_outside_row=0,
            aggregates_range=False,
            lexicon_hits=["total"],
            input_colored_fraction=0.0,
            bold_or_bordered=False,
            label="Total liabilities",
            has_formula=True,
        ),
        True,
    ),
    (
        "aggregating_row_with_bold_is_final",
        _Row(
            n_dependents_outside_row=3,
            aggregates_range=True,
            lexicon_hits=[],
            input_colored_fraction=0.1,
            bold_or_bordered=True,
            label="Subtotal",
            has_formula=True,
        ),
        True,
    ),
    (
        "text_only_final_row_no_formula",
        _Row(
            n_dependents_outside_row=0,
            aggregates_range=False,
            lexicon_hits=["conclusion"],
            input_colored_fraction=0.0,
            bold_or_bordered=False,
            label="Recommendation",
            has_formula=False,
        ),
        True,
    ),
    (
        "input_colored_majority_blocks_finality",
        _Row(
            n_dependents_outside_row=0,
            aggregates_range=False,
            lexicon_hits=["total"],
            input_colored_fraction=0.9,
            bold_or_bordered=False,
            label="Assumption",
            has_formula=False,
        ),
        False,
    ),
    (
        "not_surfaced_stays_intermediate",
        _Row(
            n_dependents_outside_row=0,
            aggregates_range=False,
            lexicon_hits=[],
            input_colored_fraction=0.0,
            bold_or_bordered=False,
            label="Build-up step",
            has_formula=True,
        ),
        False,
    ),
    (
        "has_outside_dependents_and_no_aggregate_stays_intermediate",
        _Row(
            n_dependents_outside_row=2,
            aggregates_range=False,
            lexicon_hits=["total"],
            input_colored_fraction=0.0,
            bold_or_bordered=True,
            label="Interim calc",
            has_formula=True,
        ),
        False,
    ),
    (
        "boundary_input_colored_fraction_exactly_threshold_is_blocked",
        _Row(
            n_dependents_outside_row=0,
            aggregates_range=False,
            lexicon_hits=["total"],
            input_colored_fraction=classify.V0_INPUT_COLORED_FRACTION_THRESHOLD,
            bold_or_bordered=False,
            label="Edge case",
            has_formula=True,
        ),
        False,
    ),
]


@pytest.mark.parametrize("name,row,expected", V0_CASES, ids=[c[0] for c in V0_CASES])
def test_classify_v0_row_table_driven(name: str, row: _Row, expected: bool) -> None:
    assert classify_v0_row(_protocol_accepts(row)) is expected


def test_classify_v0_returns_sorted_final_rows() -> None:
    rows: dict[int, _Row] = {
        7: _Row(0, False, ["total"], 0.0, False, "Total", True),
        3: _Row(2, False, [], 0.0, False, "Build-up", True),
        1: _Row(0, True, [], 0.0, True, "Subtotal", True),
    }
    assert classify_v0(rows) == [1, 7]


def test_classify_v0_empty_rows_returns_empty_list() -> None:
    assert classify_v0({}) == []


def test_fetch_task_profile_happy_path_and_cache(tmp_path: Path) -> None:
    async def run() -> None:
        payload = {
            "workbook_purpose": "DCF valuation of Acme Corp",
            "model_type": "DCF",
            "expected_deliverables": ["implied share price"],
            "likely_capstone_outputs": ["Equity value", "Implied value per share"],
        }
        client, calls = _fake_client([_message_response(json.dumps(payload))])
        settings = _settings()
        profile = await fetch_task_profile(
            client,
            settings,
            task="T",
            instructions="Build a DCF.",
            sheet_names=["DCF", "Assumptions"],
            prompt_version="v1",
            run_dir=tmp_path,
        )
        assert profile.workbook_purpose == payload["workbook_purpose"]
        assert profile.model_type == "DCF"
        assert profile.expected_deliverables == ("implied share price",)
        assert profile.likely_capstone_outputs == ("Equity value", "Implied value per share")
        assert len(calls) == 1

        profile2 = await fetch_task_profile(
            client,
            settings,
            task="T",
            instructions="Build a DCF.",
            sheet_names=["DCF", "Assumptions"],
            prompt_version="v1",
            run_dir=tmp_path,
        )
        assert profile2 == profile
        assert len(calls) == 1

    asyncio.run(run())


def test_fetch_task_profile_different_instructions_miss_cache(tmp_path: Path) -> None:
    async def run() -> None:
        payload = {
            "workbook_purpose": "p",
            "model_type": "DCF",
            "expected_deliverables": [],
            "likely_capstone_outputs": [],
        }
        client, calls = _fake_client([_message_response(json.dumps(payload)), _message_response(json.dumps(payload))])
        settings = _settings()
        await fetch_task_profile(
            client, settings, task="T", instructions="A", sheet_names=["S1"], prompt_version="v1", run_dir=tmp_path
        )
        await fetch_task_profile(
            client, settings, task="T", instructions="B", sheet_names=["S1"], prompt_version="v1", run_dir=tmp_path
        )
        assert len(calls) == 2

    asyncio.run(run())


def test_fetch_task_profile_malformed_json_raises(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_message_response("{not valid")])
        settings = _settings()
        with pytest.raises(TaskProfileError):
            await fetch_task_profile(
                client, settings, task="T", instructions="x", sheet_names=["S1"], prompt_version="v1", run_dir=tmp_path
            )

    asyncio.run(run())


def test_fetch_task_profile_missing_field_raises(tmp_path: Path) -> None:
    async def run() -> None:
        payload = {"workbook_purpose": "x", "model_type": "DCF"}
        client, _calls = _fake_client([_message_response(json.dumps(payload))])
        settings = _settings()
        with pytest.raises(TaskProfileError):
            await fetch_task_profile(
                client, settings, task="T", instructions="x", sheet_names=["S1"], prompt_version="v1", run_dir=tmp_path
            )

    asyncio.run(run())


def test_classify_sheet_voted_majority_and_agreement(tmp_path: Path) -> None:
    async def run() -> None:
        r1 = json.dumps({"notes": "a", "final_rows": [1, 2]})
        r2 = json.dumps({"notes": "b", "final_rows": [1]})
        r3 = json.dumps({"notes": "c", "final_rows": [1, 2]})
        client, calls = _fake_client([_message_response(r1), _message_response(r2), _message_response(r3)])
        settings = _settings()
        voted = await classify_sheet_voted(
            client,
            settings,
            task="T",
            sheet="S",
            variant_name="V1",
            prompt_version="v1",
            mode="row",
            context="ctx",
            candidates=[1, 2, 3],
            run_dir=tmp_path,
            k=3,
        )
        assert voted.majority_refs == (1, 2)
        assert voted.k == 3
        assert len(voted.per_sample) == 3
        assert len(calls) == 3
        assert voted.agreement_rate == pytest.approx(2 / 3)
        assert voted.usage.api_calls == 3

    asyncio.run(run())


def test_classify_sheet_voted_rejects_k_less_than_one(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([])
        settings = _settings()
        with pytest.raises(ValueError, match="k must be >= 1"):
            await classify_sheet_voted(
                client,
                settings,
                task="T",
                sheet="S",
                variant_name="V1",
                prompt_version="v1",
                mode="row",
                context="ctx",
                candidates=[1],
                run_dir=tmp_path,
                k=0,
            )

    asyncio.run(run())


def test_get_llm_semaphore_shares_by_value() -> None:

    classify.reset_llm_semaphores()

    async def check() -> None:
        a = get_llm_semaphore(4)
        b = get_llm_semaphore(4)
        c = get_llm_semaphore(7)
        assert a is b
        assert a is not c

    try:
        asyncio.run(check())
    finally:
        classify.reset_llm_semaphores()


def test_build_async_client_requires_api_key() -> None:
    settings = _settings()
    assert settings.openai_api_key is None
    with pytest.raises(ClassifyError, match="OPENAI_API_KEY"):
        classify.build_async_client(settings)


def test_build_async_client_constructs_with_key() -> None:
    settings = _settings(OPENAI_API_KEY="sk-test-key")
    client = classify.build_async_client(settings)
    assert isinstance(client, AsyncOpenAI)
    assert client.max_retries == 0


def test_build_async_client_passes_base_url() -> None:
    settings = _settings(OPENAI_API_KEY="sk-test-key", openai_base_url="https://example.invalid/v1")
    client = classify.build_async_client(settings)
    assert str(client.base_url).startswith("https://example.invalid/v1")
