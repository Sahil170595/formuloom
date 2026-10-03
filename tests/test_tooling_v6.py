from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response

from formuloom import classify, tooling_v6
from formuloom.classify import (
    InvalidJSONError,
    RefusalError,
    TruncatedResponseError,
    UnknownReferenceError,
)
from formuloom.schema import CellEntry
from formuloom.settings import Settings
from formuloom.tooling_v6 import (
    CONTAINER_SESSION_USD,
    ToolingError,
    build_v6_sheet_input,
    candidate_ref_strings,
    classify_sheet_v6,
    code_interpreter_tool,
    container_cost_usd,
    open_task_container,
    parse_v6_response,
    v6_instructions,
)


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


def _tooling_response(text: str, *, usage: dict[str, Any] | None = None) -> Response:
    payload = {
        "id": "resp_v6",
        "created_at": 1.0,
        "model": "gpt-5.4-mini",
        "object": "response",
        "output": [
            {
                "id": "ci_1",
                "type": "code_interpreter_call",
                "status": "completed",
                "container_id": "cntr_abc",
                "code": "import openpyxl; wb = openpyxl.load_workbook('/mnt/data/complete.xlsx')",
                "outputs": [],
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": usage if usage is not None else _usage_payload(2000, 300),
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


class _FakeFile:
    def __init__(self, file_id: str) -> None:
        self.id = file_id


class _FakeContainer:
    def __init__(self, container_id: str) -> None:
        self.id = container_id


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
    client.responses.create = fake_create  # type: ignore[assignment]  # queue-driven test double
    return client, calls


def test_v6_instructions_imports_v1_rubric_not_forked() -> None:
    from formuloom.prompts.v1 import CELL_SCHEMA_DESCRIPTION, SYSTEM_PROMPT

    text = v6_instructions()

    assert SYSTEM_PROMPT in text
    assert CELL_SCHEMA_DESCRIPTION in text

    assert "code-interpreter" in text or "code interpreter" in text
    assert "/mnt/data/init.xlsx" in text
    assert "/mnt/data/complete.xlsx" in text


def test_build_v6_sheet_input_lists_candidates_and_sheet() -> None:
    text = build_v6_sheet_input("Acquirer", ["D16", "D17", "E20"], "Build a DCF of Acme.")
    assert "'Acquirer'" in text
    assert "D16" in text and "D17" in text and "E20" in text
    assert "Build a DCF of Acme." in text

    assert "3 A1 cell references" in text


def test_code_interpreter_tool_shape() -> None:
    assert code_interpreter_tool("cntr_xyz") == {"type": "code_interpreter", "container": "cntr_xyz"}


def test_candidate_ref_strings_extracts_a1_refs() -> None:
    entries = [CellEntry(cell="D16", cell_type="currency"), CellEntry(cell="E20", cell_type="number")]
    assert candidate_ref_strings(entries) == ["D16", "E20"]


def test_container_cost_usd_one_gig_default() -> None:
    assert container_cost_usd(5) == pytest.approx(5 * 0.03)
    assert container_cost_usd(0) == 0.0


def test_container_cost_usd_higher_tiers() -> None:
    assert container_cost_usd(2, "4g") == pytest.approx(2 * CONTAINER_SESSION_USD["4g"])
    assert container_cost_usd(1, "64g") == pytest.approx(1.92)


def test_container_cost_usd_unknown_tier_raises() -> None:
    with pytest.raises(ToolingError, match="no container price"):
        container_cost_usd(1, "999g")


def test_container_cost_usd_negative_sessions_raises() -> None:
    with pytest.raises(ToolingError, match="n_sessions must be >= 0"):
        container_cost_usd(-1)


def test_parse_v6_response_happy_path_skips_tool_call_items() -> None:
    text = json.dumps({"notes": "D16 is the section output", "final_cells": ["D16"]})
    response = _tooling_response(text, usage=_usage_payload(2000, 300, cached_tokens=500))
    notes, refs, usage = parse_v6_response("Acquirer", response, ["D14", "D15", "D16"], "gpt-5.4-mini")
    assert notes == "D16 is the section output"
    assert refs == ["D16"]
    assert usage.input_tokens == 2000
    assert usage.cached_input_tokens == 500
    assert usage.output_tokens == 300
    assert usage.cost_usd > 0
    assert usage.api_calls == 1


def test_parse_v6_response_refusal_raises() -> None:
    with pytest.raises(RefusalError) as excinfo:
        parse_v6_response("S", _refusal_response("cannot process this workbook"), ["A1"], "gpt-5.4-mini")
    assert excinfo.value.sheet == "S"
    assert "cannot process" in excinfo.value.refusal_text


def test_parse_v6_response_truncated_raises() -> None:
    with pytest.raises(TruncatedResponseError) as excinfo:
        parse_v6_response("S", _incomplete_response("max_output_tokens"), ["A1"], "gpt-5.4-mini")
    assert excinfo.value.reason == "max_output_tokens"


def test_parse_v6_response_invalid_json_raises() -> None:
    with pytest.raises(InvalidJSONError):
        parse_v6_response("S", _tooling_response("not json"), ["A1"], "gpt-5.4-mini")


def test_parse_v6_response_wrong_shape_raises() -> None:

    bad = json.dumps({"notes": "n", "final_cells": [16]})
    with pytest.raises(InvalidJSONError):
        parse_v6_response("S", _tooling_response(bad), ["A1"], "gpt-5.4-mini")


def test_classify_sheet_v6_happy_path_sends_code_interpreter_tool(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "D16 final", "final_cells": ["D16"]})
        client, calls = _fake_client([_tooling_response(text)])
        settings = _settings()
        result = await classify_sheet_v6(
            client,
            settings,
            task="synthetic-summary",
            sheet="Acquirer",
            container_id="cntr_abc",
            candidate_refs=["D14", "D15", "D16"],
            instructions="Build the acquirer model.",
            run_dir=tmp_path,
        )
        assert result.mode == "cell"
        assert result.final_refs == ("D16",)
        assert result.from_cache is False
        assert len(calls) == 1
        call = calls[0]
        assert call["tools"] == [{"type": "code_interpreter", "container": "cntr_abc"}]
        assert call["text"]["format"]["type"] == "json_schema"
        assert call["text"]["format"]["strict"] is True
        assert call["text"]["format"]["name"] == classify.CELL_SCHEMA_NAME
        assert "Acquirer" in call["input"]
        assert call["reasoning"] == {"effort": "low"}
        assert "service_tier" not in call

    asyncio.run(run())


def test_classify_sheet_v6_unknown_ref_corrective_retry(tmp_path: Path) -> None:
    async def run() -> None:
        bad = json.dumps({"notes": "n", "final_cells": ["D16", "Z99"]})
        good = json.dumps({"notes": "corrected", "final_cells": ["D16"]})
        client, calls = _fake_client([_tooling_response(bad), _tooling_response(good)])
        settings = _settings()
        result = await classify_sheet_v6(
            client,
            settings,
            task="T",
            sheet="S",
            container_id="cntr_1",
            candidate_refs=["D14", "D16"],
            instructions="x",
            run_dir=tmp_path,
        )
        assert result.final_refs == ("D16",)
        assert result.notes == "corrected"
        assert len(calls) == 2
        assert "CORRECTION" in calls[1]["input"]
        assert "Z99" in calls[1]["input"]

        assert result.usage.api_calls == 2

    asyncio.run(run())


def test_classify_sheet_v6_unknown_ref_after_correction_raises(tmp_path: Path) -> None:
    async def run() -> None:
        bad = json.dumps({"notes": "n", "final_cells": ["Z99"]})
        still_bad = json.dumps({"notes": "n2", "final_cells": ["Z100"]})
        client, calls = _fake_client([_tooling_response(bad), _tooling_response(still_bad)])
        settings = _settings()
        with pytest.raises(UnknownReferenceError) as excinfo:
            await classify_sheet_v6(
                client,
                settings,
                task="T",
                sheet="S",
                container_id="cntr_1",
                candidate_refs=["D14", "D16"],
                instructions="x",
                run_dir=tmp_path,
            )
        assert "Z100" in set(excinfo.value.unknown_refs)
        assert len(calls) == 2

    asyncio.run(run())


def test_classify_sheet_v6_cache_hit_skips_api_call(tmp_path: Path) -> None:
    async def run() -> None:
        text = json.dumps({"notes": "n", "final_cells": ["D16"]})
        client, calls = _fake_client([_tooling_response(text)])
        settings = _settings()
        first = await classify_sheet_v6(
            client,
            settings,
            task="T",
            sheet="S",
            container_id="cntr_1",
            candidate_refs=["D16"],
            instructions="x",
            run_dir=tmp_path,
        )
        assert first.from_cache is False
        assert len(calls) == 1
        second = await classify_sheet_v6(
            client,
            settings,
            task="T",
            sheet="S",
            container_id="cntr_1",
            candidate_refs=["D16"],
            instructions="x",
            run_dir=tmp_path,
        )
        assert second.from_cache is True
        assert second.final_refs == ("D16",)
        assert len(calls) == 1

    asyncio.run(run())


def test_open_task_container_uploads_both_and_creates_container(tmp_path: Path) -> None:
    async def run() -> None:
        init_path = tmp_path / "init.xlsx"
        complete_path = tmp_path / "complete.xlsx"
        init_path.write_bytes(b"INIT-BYTES")
        complete_path.write_bytes(b"COMPLETE-BYTES")

        files_calls: list[dict[str, Any]] = []
        containers_calls: list[dict[str, Any]] = []

        async def fake_files_create(**kwargs: Any) -> _FakeFile:
            files_calls.append(kwargs)
            return _FakeFile(f"file_{len(files_calls)}")

        async def fake_containers_create(**kwargs: Any) -> _FakeContainer:
            containers_calls.append(kwargs)
            return _FakeContainer("cntr_created")

        client = AsyncOpenAI(api_key="test-key", max_retries=0)
        client.files.create = fake_files_create  # type: ignore[assignment]  # test double
        client.containers.create = fake_containers_create  # type: ignore[assignment]  # test double

        handle = await open_task_container(
            client, task="synthetic-summary", init_path=init_path, complete_path=complete_path
        )
        assert handle.container_id == "cntr_created"
        assert handle.init_file_id == "file_1"
        assert handle.complete_file_id == "file_2"
        assert len(files_calls) == 2
        assert files_calls[0]["purpose"] == tooling_v6.FILE_UPLOAD_PURPOSE

        assert containers_calls[0]["file_ids"] == ["file_1", "file_2"]
        assert containers_calls[0]["memory_limit"] == tooling_v6.CONTAINER_MEMORY_LIMIT

    asyncio.run(run())


def test_open_task_container_wraps_failure_in_typed_error(tmp_path: Path) -> None:
    async def run() -> None:
        init_path = tmp_path / "init.xlsx"
        complete_path = tmp_path / "complete.xlsx"
        init_path.write_bytes(b"x")
        complete_path.write_bytes(b"y")

        async def boom(**kwargs: Any) -> _FakeFile:
            raise RuntimeError("upload exploded")

        client = AsyncOpenAI(api_key="test-key", max_retries=0)
        client.files.create = boom  # type: ignore[assignment]  # test double

        with pytest.raises(tooling_v6.ContainerSetupError) as excinfo:
            await open_task_container(client, task="T", init_path=init_path, complete_path=complete_path)
        assert excinfo.value.task == "T"

    asyncio.run(run())
