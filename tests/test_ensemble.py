from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response

from formuloom.classify import CACHE_SUBDIR, ClassifyError, RefusalError
from formuloom.ensemble import DiverseEnsembleResult, PromptVoteRecord, classify_sheet_diverse
from formuloom.prompts import ClassifyMode, get_prompt_version
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


def _row_response(final_rows: list[int], *, usage: dict[str, Any] | None = None) -> Response:
    payload = {
        "id": "resp",
        "created_at": 1.0,
        "model": "gpt-5.4-mini",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"notes": "n", "final_rows": final_rows}),
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": usage if usage is not None else _usage_payload(100, 20),
    }
    return Response.model_validate(payload)


def _refusal_response() -> Response:
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
                "content": [{"type": "refusal", "refusal": "cannot classify"}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": _usage_payload(50, 5),
    }
    return Response.model_validate(payload)


def _prompt_keyed_fake(
    by_prompt: dict[str, Response | Exception], mode: ClassifyMode = "row"
) -> tuple[AsyncOpenAI, list[dict[str, Any]]]:
    prefix_to_version = {get_prompt_version(v).static_prefix(mode): v for v in by_prompt}
    if len(prefix_to_version) != len(by_prompt):
        raise AssertionError("two prompt versions produced identical static prefixes; test cannot key on them")
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> Response:
        calls.append(kwargs)
        version = prefix_to_version.get(kwargs["instructions"])
        if version is None:
            raise AssertionError(f"call with unrecognized instructions (not one of {sorted(by_prompt)})")
        step = by_prompt[version]
        if isinstance(step, Exception):
            raise step
        return step

    client = AsyncOpenAI(api_key="test-key", max_retries=0)
    client.responses.create = fake_create  # type: ignore[assignment]  # queue-driven test double
    return client, calls


def _ordered_fake(steps: list[Response | Exception]) -> tuple[AsyncOpenAI, list[dict[str, Any]]]:
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


async def _run_diverse(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    run_dir: Path,
    prompt_versions: tuple[str, ...],
    candidates: list[int],
    samples_per_prompt: int = 1,
) -> DiverseEnsembleResult:
    return await classify_sheet_diverse(
        client,
        settings,
        task="T",
        sheet="S",
        variant_name="V12",
        prompt_versions=prompt_versions,
        mode="row",
        context="ctx",
        candidates=candidates,
        run_dir=run_dir,
        samples_per_prompt=samples_per_prompt,
    )


def test_majority_2_of_3_keeps_and_minority_drops(tmp_path: Path) -> None:
    async def run() -> None:
        client, calls = _prompt_keyed_fake(
            {
                "v1": _row_response([1, 2]),
                "v2": _row_response([1, 2]),
                "v3": _row_response([1, 3]),
            }
        )
        res = await _run_diverse(
            client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2", "v3"), candidates=[1, 2, 3]
        )

        assert res.majority_refs == (1, 2)
        assert res.n_votes == 3
        assert res.n_failed_votes == 0
        assert len(calls) == 3

        assert res.agreement_rate == pytest.approx(1 / 3)

    asyncio.run(run())


def test_dissenting_prompt_is_identifiable(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _prompt_keyed_fake(
            {"v1": _row_response([1, 2]), "v2": _row_response([1, 2]), "v3": _row_response([1, 3])}
        )
        res = await _run_diverse(
            client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2", "v3"), candidates=[1, 2, 3]
        )
        majority = frozenset(str(r) for r in res.majority_refs)
        by_prompt = {rec.prompt_version: rec for rec in res.per_vote}
        assert by_prompt["v1"].dissent_count(majority) == 0
        assert by_prompt["v2"].dissent_count(majority) == 0
        assert by_prompt["v3"].dissent_count(majority) == 2

    asyncio.run(run())


def test_tie_breaks_toward_final(tmp_path: Path) -> None:
    async def run() -> None:

        client, _calls = _prompt_keyed_fake({"v1": _row_response([1, 2]), "v2": _row_response([1])})
        res = await _run_diverse(
            client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2"), candidates=[1, 2, 3]
        )
        assert res.majority_refs == (1, 2)
        assert res.n_votes == 2

    asyncio.run(run())


def test_distinct_prompts_write_distinct_cache_files(tmp_path: Path) -> None:
    async def run() -> None:
        client, calls = _prompt_keyed_fake({"v1": _row_response([1]), "v2": _row_response([1])})
        await _run_diverse(client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2"), candidates=[1])
        assert len(calls) == 2
        assert len(list((tmp_path / CACHE_SUBDIR).glob("*.json"))) == 2

    asyncio.run(run())


def test_repeated_prompt_votes_are_independent_not_collapsed(tmp_path: Path) -> None:
    async def run() -> None:

        client, calls = _ordered_fake([_row_response([1]), _row_response([1, 2])])
        res = await _run_diverse(client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v1"), candidates=[1, 2])
        assert len(calls) == 2
        assert len(list((tmp_path / CACHE_SUBDIR).glob("*.json"))) == 2

        assert res.majority_refs == (1, 2)

    asyncio.run(run())


def test_samples_per_prompt_fans_out_independent_draws(tmp_path: Path) -> None:
    async def run() -> None:
        client, calls = _ordered_fake([_row_response([1]), _row_response([1]), _row_response([1])])
        res = await _run_diverse(
            client, _settings(), run_dir=tmp_path, prompt_versions=("v1",), candidates=[1], samples_per_prompt=3
        )
        assert len(calls) == 3
        assert len(list((tmp_path / CACHE_SUBDIR).glob("*.json"))) == 3
        assert res.n_votes == 3
        assert res.majority_refs == (1,)

    asyncio.run(run())


def test_one_failed_vote_is_dropped_survivors_decide(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _prompt_keyed_fake(
            {"v1": _row_response([1]), "v2": _refusal_response(), "v3": _row_response([1])}
        )
        res = await _run_diverse(
            client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2", "v3"), candidates=[1, 2]
        )
        assert res.n_failed_votes == 1
        assert res.n_votes == 2
        assert res.majority_refs == (1,)
        assert {rec.prompt_version for rec in res.per_vote} == {"v1", "v3"}

    asyncio.run(run())


def test_usage_is_summed_over_surviving_votes(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _prompt_keyed_fake(
            {
                "v1": _row_response([1], usage=_usage_payload(100, 10)),
                "v2": _row_response([1], usage=_usage_payload(200, 20)),
            }
        )
        res = await _run_diverse(client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2"), candidates=[1])
        assert res.usage.api_calls == 2
        assert res.usage.input_tokens == 300
        assert res.usage.output_tokens == 30
        assert res.usage.cost_usd > 0

    asyncio.run(run())


def test_all_votes_fail_raises_typed_error(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _prompt_keyed_fake({"v1": _refusal_response(), "v2": _refusal_response()})
        with pytest.raises(RefusalError):
            await _run_diverse(client, _settings(), run_dir=tmp_path, prompt_versions=("v1", "v2"), candidates=[1])

    asyncio.run(run())


def test_empty_prompt_versions_raises(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _ordered_fake([])
        with pytest.raises(ValueError, match="prompt_versions must be non-empty"):
            await _run_diverse(client, _settings(), run_dir=tmp_path, prompt_versions=(), candidates=[1])

    asyncio.run(run())


def test_samples_per_prompt_below_one_raises(tmp_path: Path) -> None:
    async def run() -> None:
        client, _calls = _ordered_fake([])
        with pytest.raises(ValueError, match="samples_per_prompt must be >= 1"):
            await _run_diverse(
                client, _settings(), run_dir=tmp_path, prompt_versions=("v1",), candidates=[1], samples_per_prompt=0
            )

    asyncio.run(run())


def test_result_types_are_exported() -> None:
    assert DiverseEnsembleResult.__name__ == "DiverseEnsembleResult"
    assert PromptVoteRecord.__name__ == "PromptVoteRecord"
    assert issubclass(RefusalError, ClassifyError)
