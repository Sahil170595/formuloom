from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response

from formuloom import scout
from formuloom.bundle import TaskBundle
from formuloom.classify import ROW_SCHEMA_NAME
from formuloom.cli import _build_pipeline
from formuloom.schema import VariantConfig
from formuloom.settings import Settings
from formuloom.variants import get_variant
from tests.test_cli import _install_fake_client, _message_response, _write_task_bundle


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


def _usage_payload(input_tokens: int = 100, output_tokens: int = 20) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _fake_response(text: str) -> Response:
    payload = {
        "id": "resp_scout",
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
        "usage": _usage_payload(),
    }
    return Response.model_validate(payload)


def _disposition_response(
    domain: str = "investment_banking",
    model_kind: str = "DCF",
    grading_disposition: str = "unknown",
    notes: str = "n",
) -> Response:
    return _fake_response(
        json.dumps(
            {"notes": notes, "domain": domain, "model_kind": model_kind, "grading_disposition": grading_disposition}
        )
    )


def _fake_client(steps: list[Response]) -> tuple[AsyncOpenAI, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    remaining = list(steps)

    async def fake_create(**kwargs: Any) -> Response:
        calls.append(kwargs)
        return remaining.pop(0)

    client = AsyncOpenAI(api_key="test-key", max_retries=0)
    client.responses.create = fake_create  # type: ignore[assignment]
    return client, calls


def _row_response(rows: list[int]) -> Response:
    return _message_response(json.dumps({"notes": "test", "final_rows": rows}))


def test_parse_disposition_happy_path() -> None:
    text = json.dumps(
        {
            "notes": "rationale",
            "domain": "real_estate",
            "model_kind": "proforma",
            "grading_disposition": "capstones_only",
        }
    )
    notes, domain, model_kind, grading = scout._parse_disposition("T", text)
    assert notes == "rationale"
    assert domain == "real_estate"
    assert model_kind == "proforma"
    assert grading == "capstones_only"


def test_parse_disposition_invalid_json_raises() -> None:
    with pytest.raises(scout.ScoutError, match="not valid JSON"):
        scout._parse_disposition("T", "not json")


def test_parse_disposition_non_object_top_level_raises() -> None:
    with pytest.raises(scout.ScoutError, match="not an object"):
        scout._parse_disposition("T", json.dumps([1, 2, 3]))


def test_parse_disposition_bad_domain_raises() -> None:
    text = json.dumps({"notes": "n", "domain": "banking", "model_kind": "DCF", "grading_disposition": "unknown"})
    with pytest.raises(scout.ScoutError, match="domain"):
        scout._parse_disposition("T", text)


def test_parse_disposition_bad_grading_disposition_raises() -> None:
    text = json.dumps({"notes": "n", "domain": "other", "model_kind": "DCF", "grading_disposition": "all_cells"})
    with pytest.raises(scout.ScoutError, match="grading_disposition"):
        scout._parse_disposition("T", text)


def test_parse_disposition_empty_model_kind_raises() -> None:
    text = json.dumps({"notes": "n", "domain": "other", "model_kind": "", "grading_disposition": "unknown"})
    with pytest.raises(scout.ScoutError, match="model_kind"):
        scout._parse_disposition("T", text)


def test_routing_map_matches_measured_per_domain_winners() -> None:
    assert scout.ROUTING_MAP == {
        "investment_banking": "V9",
        "corporate_finance": "V11",
        "real_estate": "V13",
        "other": "V9",
    }


@pytest.mark.parametrize(
    "domain,expected",
    [("investment_banking", "V9"), ("corporate_finance", "V11"), ("real_estate", "V13"), ("other", "V9")],
)
def test_resolve_route_uses_default_routing_map(domain: str, expected: str) -> None:
    assert scout.resolve_route(domain) == expected


def test_resolve_route_unknown_domain_falls_back_to_default() -> None:
    assert scout.resolve_route("something_not_in_the_map") == scout.DEFAULT_ROUTE_VARIANT


def test_resolve_route_falls_back_when_mapped_variant_is_not_registered() -> None:

    routing_map = {"real_estate": "V999_NOT_REGISTERED"}
    assert scout.resolve_route("real_estate", routing_map=routing_map) == scout.DEFAULT_ROUTE_VARIANT


def test_resolve_route_custom_routing_map_overrides_default() -> None:
    assert scout.resolve_route("other", routing_map={"other": "V1"}) == "V1"


def test_domain_and_grading_disposition_values_match_json_schema_enums() -> None:
    assert scout.SCOUT_JSON_SCHEMA["properties"]["domain"]["enum"] == sorted(scout.DOMAIN_VALUES)
    assert scout.SCOUT_JSON_SCHEMA["properties"]["grading_disposition"]["enum"] == sorted(
        scout.GRADING_DISPOSITION_VALUES
    )
    assert {"investment_banking", "corporate_finance", "real_estate", "other"} == scout.DOMAIN_VALUES
    assert {"capstones_only", "subtotals_count", "unknown"} == scout.GRADING_DISPOSITION_VALUES


def test_fetch_task_disposition_happy_path_and_cache(tmp_path: Path) -> None:
    async def run() -> None:
        client, calls = _fake_client([_disposition_response(domain="corporate_finance", model_kind="LBO")])
        settings = _settings()
        disposition = await scout.fetch_task_disposition(
            client,
            settings,
            task="T",
            instructions="Build an LBO model.",
            workbook_map_lines=["- S1 | used A1:B2 (2x2) | candidates 2"],
            run_dir=tmp_path,
        )
        assert disposition.domain == "corporate_finance"
        assert disposition.model_kind == "LBO"
        assert disposition.from_cache is False
        assert len(calls) == 1
        assert calls[0]["model"] == scout.SCOUT_DEFAULT_MODEL
        assert calls[0]["instructions"] == scout.SCOUT_SYSTEM_PROMPT
        assert calls[0]["text"]["format"]["name"] == scout.SCOUT_SCHEMA_NAME
        assert calls[0]["text"]["format"]["schema"] == scout.SCOUT_JSON_SCHEMA
        assert calls[0]["text"]["format"]["strict"] is True

        second = await scout.fetch_task_disposition(
            client,
            settings,
            task="T",
            instructions="Build an LBO model.",
            workbook_map_lines=["- S1 | used A1:B2 (2x2) | candidates 2"],
            run_dir=tmp_path,
        )
        assert second.from_cache is True
        assert second.domain == "corporate_finance"
        assert len(calls) == 1

    asyncio.run(run())


def test_fetch_task_disposition_different_workbook_map_misses_cache(tmp_path: Path) -> None:
    async def run() -> None:
        client, calls = _fake_client([_disposition_response(), _disposition_response()])
        settings = _settings()
        await scout.fetch_task_disposition(
            client, settings, task="T", instructions="x", workbook_map_lines=["- A"], run_dir=tmp_path
        )
        await scout.fetch_task_disposition(
            client, settings, task="T", instructions="x", workbook_map_lines=["- B"], run_dir=tmp_path
        )
        assert len(calls) == 2

    asyncio.run(run())


def test_fetch_task_disposition_malformed_json_raises(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_fake_response("{not valid")])
        settings = _settings()
        with pytest.raises(scout.ScoutError):
            await scout.fetch_task_disposition(
                client, settings, task="T", instructions="x", workbook_map_lines=["- A"], run_dir=tmp_path
            )

    asyncio.run(run())


def test_fetch_task_disposition_writes_atomic_cache_file(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_disposition_response()])
        settings = _settings()
        await scout.fetch_task_disposition(
            client, settings, task="T", instructions="x", workbook_map_lines=["- A"], run_dir=tmp_path
        )
        cache_dir = tmp_path / scout.SCOUT_CACHE_SUBDIR
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1
        assert list(cache_dir.glob("*.tmp")) == []

    asyncio.run(run())


def test_fetch_task_disposition_defaults_to_fast_model_and_low_effort_regardless_of_settings(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        client, calls = _fake_client([_disposition_response()])
        settings = _settings(model="gpt-5.4", reasoning_effort="medium")
        await scout.fetch_task_disposition(
            client, settings, task="T", instructions="x", workbook_map_lines=["- A"], run_dir=tmp_path
        )
        assert calls[0]["model"] == "gpt-5.4-mini"
        assert calls[0]["reasoning"] == {"effort": "low"}

    asyncio.run(run())


def test_build_workbook_map_lines_reuses_encode_sheet_map_line(tmp_path: Path) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")
    lines = scout.build_workbook_map_lines(bundle)
    assert len(lines) >= 1
    assert all(line.startswith("- ") for line in lines)
    assert any("S1" in line for line in lines)


def test_scout_routed_pipeline_rejects_nested_scout_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        client, _calls = _fake_client([_disposition_response(domain="other")])
        nested_cfg = VariantConfig(name="Nested16", scout_route=True)
        monkeypatch.setattr(scout, "get_variant", lambda _name: nested_cfg)

        def _factory_should_not_be_called(_config: VariantConfig) -> Any:
            raise AssertionError("pipeline_factory must never run for a nested scout route")

        settings = _settings()
        pipeline = scout.ScoutRoutedPipeline(settings, _factory_should_not_be_called, client_factory=lambda _s: client)
        bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
        bundle = TaskBundle.load(bundle_dir, "predict")
        with pytest.raises(ValueError, match="nested scout routing"):
            await pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=False)

    asyncio.run(run())


def test_v16_routes_investment_banking_to_v9_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")

    marker_model_kind = "SCOUT_MARKER_MODEL_KIND_XYZ"
    marker_notes = "SCOUT_MARKER_NOTES_XYZ"
    calls = _install_fake_client(
        monkeypatch,
        [
            _disposition_response(domain="investment_banking", model_kind=marker_model_kind, notes=marker_notes),
            _row_response([1]),
            _row_response([1]),
            _row_response([1]),
        ],
    )
    config = get_variant("V16")
    pipeline = _build_pipeline(config)
    result = asyncio.run(pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=False))

    assert len(calls) == 4
    assert calls[0]["text"]["format"]["name"] == scout.SCOUT_SCHEMA_NAME
    assert result.diff.final_refs("S1") == {"B1"}
    assert result.failures == {}
    assert result.usage.api_calls == 4

    routed_calls = calls[1:]
    assert routed_calls
    for call in routed_calls:
        assert call["text"]["format"]["name"] == ROW_SCHEMA_NAME
        assert marker_model_kind not in call["input"]
        assert marker_notes not in call["input"]
        assert "investment_banking" not in call["input"]
        assert "domain" not in call["instructions"]


def test_v16_falls_back_to_v9_when_domain_is_other(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")
    calls = _install_fake_client(
        monkeypatch,
        [
            _disposition_response(domain="other"),
            _row_response([1]),
            _row_response([1]),
            _row_response([1]),
        ],
    )
    config = get_variant("V16")
    pipeline = _build_pipeline(config)
    result = asyncio.run(pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=False))
    assert len(calls) == 4
    assert result.diff.final_refs("S1") == {"B1"}
