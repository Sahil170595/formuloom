from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response

from formuloom.adjudicate import ADJUDICATE_SCHEMA_NAME
from formuloom.cascade import (
    SheetCascadeResult,
    assign_tiers,
    cascade_sheet,
    resolve_band,
    row_vote_counts,
)
from formuloom.classify import ROW_SCHEMA_NAME
from formuloom.encode import SheetContext
from formuloom.ensemble import PromptVoteRecord
from formuloom.prompts import get_prompt_version
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


def _message_response(text: str) -> Response:
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
        "usage": _usage_payload(200, 20),
    }
    return Response.model_validate(payload)


def _row_response(final_rows: list[int]) -> Response:
    return _message_response(json.dumps({"notes": "n", "final_rows": final_rows}))


def _verdicts(**row_to_verdict: str) -> Response:
    rows = [
        {"row": int(r), "label": f"label {r}", "verdict": v, "reason": "because"} for r, v in row_to_verdict.items()
    ]
    return _message_response(json.dumps({"rows": rows}))


def _context(rows: tuple[int, ...] = (1, 2, 3, 4), grouping: str = "row") -> SheetContext:
    row_lines = tuple((r, f"row {r} | label{r} | - | A{r}=x | number") for r in rows)
    return SheetContext(
        task="T",
        sheet="S",
        grouping=grouping,  # type: ignore[arg-type]  # test may pass an intentionally-bad mode
        preamble="PREAMBLE\n",
        data_header="HEADER\n",
        row_lines=row_lines,
        sections=(),
        candidate_refs=tuple(f"A{r}" for r in rows),
    )


def _row_to_cells(rows: tuple[int, ...] = (1, 2, 3, 4)) -> dict[int, list[str]]:
    return {r: [f"A{r}"] for r in rows}


def _cascade_fake(
    votes_by_prompt: dict[str, Response | Exception],
    adjudicate_steps: list[Response | Exception] | None = None,
) -> tuple[AsyncOpenAI, list[dict[str, Any]]]:
    prefix_to_version = {get_prompt_version(v).static_prefix("row"): v for v in votes_by_prompt}
    if len(prefix_to_version) != len(votes_by_prompt):
        raise AssertionError("two prompt versions produced identical static prefixes; test cannot key on them")
    calls: list[dict[str, Any]] = []
    adj_remaining = list(adjudicate_steps or [])

    async def fake_create(**kwargs: Any) -> Response:
        calls.append(kwargs)
        schema_name = kwargs["text"]["format"]["name"]
        if schema_name == ADJUDICATE_SCHEMA_NAME:
            step = adj_remaining.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        if schema_name == ROW_SCHEMA_NAME:
            version = prefix_to_version.get(kwargs["instructions"])
            if version is None:
                raise AssertionError(f"vote with unrecognized instructions (not one of {sorted(votes_by_prompt)})")
            step = votes_by_prompt[version]
            if isinstance(step, Exception):
                raise step
            return step
        raise AssertionError(f"unexpected schema name {schema_name!r}")

    client = AsyncOpenAI(api_key="test-key", max_retries=0)
    client.responses.create = fake_create  # type: ignore[assignment]  # queue-driven test double
    return client, calls


def _adjudicate_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in calls if c["text"]["format"]["name"] == ADJUDICATE_SCHEMA_NAME]


def _run_cascade(client: AsyncOpenAI, tmp_path: Path, **kw: Any) -> SheetCascadeResult:
    async def run() -> SheetCascadeResult:
        return await cascade_sheet(
            client,
            _settings(),
            task="T",
            sheet="S",
            variant_name="V14",
            context=kw.pop("context", _context()),
            row_to_cells=kw.pop("row_to_cells", _row_to_cells()),
            workbook_map_line="- S | used A1:D4 (4x4) | candidates 4",
            run_dir=tmp_path,
            prompt_versions=kw.pop("prompt_versions", ("v1", "v2", "v3")),
            samples_per_prompt=kw.pop("samples_per_prompt", 1),
            **kw,
        )

    return asyncio.run(run())


def test_row_vote_counts_tallies_surviving_votes() -> None:
    per_vote = [
        PromptVoteRecord("v3", 0, (1, 2, 3), None),  # type: ignore[arg-type]  # usage unused here
        PromptVoteRecord("v4", 0, (1, 2), None),  # type: ignore[arg-type]
        PromptVoteRecord("v6", 0, (1,), None),  # type: ignore[arg-type]
    ]
    counts = row_vote_counts([1, 2, 3, 4], per_vote)
    assert counts == {1: 3, 2: 2, 3: 1, 4: 0}


def test_assign_tiers_high_low_band_boundaries() -> None:

    candidates = [10, 20, 30, 40, 50]
    vote_counts = {10: 7, 20: 8, 30: 2, 40: 6, 50: 3}
    final, intermediate, band = assign_tiers(candidates, vote_counts, high=7, low=2)
    assert final == [10, 20]
    assert intermediate == [30]
    assert band == [40, 50]


def test_assign_tiers_requires_high_above_low() -> None:
    with pytest.raises(ValueError, match="high .* > low"):
        assign_tiers([1], {1: 3}, high=3, low=3)


def test_resolve_band_judge_keep_and_vote_rescue() -> None:

    vote_counts = {2: 6, 3: 3, 4: 3}
    kept = resolve_band([2, 3, 4], vote_counts, n_votes=9, judge_keep={3}, degraded=False, band_keep_votes=5)

    assert kept == {2, 3}


def test_resolve_band_degraded_uses_vote_majority_tie_to_final() -> None:

    vote_counts = {1: 5, 2: 4, 3: 2}
    kept = resolve_band([1, 2, 3], vote_counts, n_votes=9, judge_keep=set(), degraded=True, band_keep_votes=5)
    assert kept == {1}


def test_resolve_band_degraded_even_pool_tie_breaks_to_final() -> None:

    kept = resolve_band([1, 2], {1: 2, 2: 1}, n_votes=4, judge_keep=set(), degraded=True, band_keep_votes=3)
    assert kept == {1}


def test_cascade_routes_tiers_and_adjudicates_only_the_band(tmp_path: Path) -> None:

    client, calls = _cascade_fake(
        {"v1": _row_response([1, 2, 3]), "v2": _row_response([1, 2]), "v3": _row_response([1])},
        adjudicate_steps=[_verdicts(**{"2": "drop", "3": "keep"})],
    )
    result = _run_cascade(
        client,
        tmp_path,
        high_consensus=3,
        low_consensus=0,
        band_keep_votes=2,
    )

    assert result.n_high == 1 and result.n_low == 1 and result.n_band == 2

    assert result.band_kept == 2 and result.band_dropped == 0
    assert result.band_degraded is False

    assert result.final_cells == frozenset({"A1", "A2", "A3"})

    adj = _adjudicate_calls(calls)
    assert len(adj) == 1
    adj_input = adj[0]["input"]
    assert "row 2 |" in adj_input and "row 3 |" in adj_input
    assert "row 1 |" not in adj_input and "row 4 |" not in adj_input
    assert adj[0]["model"] == "gpt-5.4"


def test_cascade_no_band_makes_no_judge_call(tmp_path: Path) -> None:

    client, calls = _cascade_fake(
        {"v1": _row_response([1, 2]), "v2": _row_response([1, 2]), "v3": _row_response([1, 2])}
    )
    result = _run_cascade(client, tmp_path, high_consensus=3, low_consensus=0, band_keep_votes=2)
    assert result.n_band == 0
    assert result.final_cells == frozenset({"A1", "A2"})
    assert _adjudicate_calls(calls) == []


def test_cascade_degraded_judge_resolves_band_by_vote_majority(tmp_path: Path) -> None:

    client, _calls = _cascade_fake(
        {"v1": _row_response([1, 2, 3]), "v2": _row_response([1, 2]), "v3": _row_response([1])},
        adjudicate_steps=[_message_response("{not valid json")],
    )
    result = _run_cascade(client, tmp_path, high_consensus=3, low_consensus=0, band_keep_votes=2)
    assert result.band_degraded is True

    assert result.final_cells == frozenset({"A1", "A2"})
    assert result.band_kept == 1 and result.band_dropped == 1


def test_cascade_all_high_consensus_needs_no_judge(tmp_path: Path) -> None:
    client, calls = _cascade_fake(
        {"v1": _row_response([1, 2, 3, 4]), "v2": _row_response([1, 2, 3, 4]), "v3": _row_response([1, 2, 3, 4])}
    )
    result = _run_cascade(client, tmp_path, high_consensus=3, low_consensus=0, band_keep_votes=2)
    assert result.n_high == 4 and result.n_band == 0
    assert result.final_cells == frozenset({"A1", "A2", "A3", "A4"})
    assert _adjudicate_calls(calls) == []


def test_cascade_requires_row_mode_context(tmp_path: Path) -> None:
    client, _calls = _cascade_fake({"v1": _row_response([1])})
    with pytest.raises(ValueError, match="requires row-mode context"):
        _run_cascade(client, tmp_path, context=_context(grouping="cell"))


def test_cascade_empty_candidates_returns_empty(tmp_path: Path) -> None:
    client, calls = _cascade_fake({"v1": _row_response([])})
    result = _run_cascade(client, tmp_path, context=_context(rows=()), row_to_cells={})
    assert result.final_cells == frozenset()
    assert result.n_votes == 0
    assert calls == []
