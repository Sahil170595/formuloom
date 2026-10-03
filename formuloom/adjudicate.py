from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from formuloom.classify import (
    ClassifyError,
    UsageRecord,
    _atomic_write_json,
    _call_llm_with_retry,
    _CallSpec,
    _coerce_reasoning_effort,
    _extract_valid_output_text,
    _read_cache_json,
    _sheet_cache_path,
    get_llm_semaphore,
    usage_from_response,
)
from formuloom.constants import LLM_MAX_RETRIES
from formuloom.prompts.v5_adjudicate import (
    ADJUDICATE_JSON_SCHEMA,
    ADJUDICATE_SCHEMA_NAME,
    ADJUDICATE_SYSTEM_PROMPT,
    build_adjudication_input,
)
from formuloom.schema import ReasoningEffort
from formuloom.settings import Settings

logger = logging.getLogger(__name__)

CACHE_SUFFIX_PREFIX = "adjudicate"

_INPUT_HASH_LEN = 16

CORRECTIVE_RETRY_LIMIT = 1

__all__ = [
    "AdjudicateError",
    "AdjudicateResult",
    "adjudicate_sheet",
    "build_adjudication_input",
]


class AdjudicateError(ClassifyError):
    pass


@dataclass(frozen=True)
class AdjudicateResult:

    sheet: str
    notes: str
    keep_rows: tuple[int, ...]
    dropped_rows: tuple[int, ...]
    usage: UsageRecord
    from_cache: bool
    degraded: bool = False


def _parse_verdicts(sheet: str, text: str) -> dict[int, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdjudicateError(f"sheet {sheet!r}: adjudicator output was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdjudicateError(f"sheet {sheet!r}: adjudicator output top-level value is not an object")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise AdjudicateError(f"sheet {sheet!r}: adjudicator 'rows' missing or not a list")
    verdicts: dict[int, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise AdjudicateError(f"sheet {sheet!r}: adjudicator row entry is not an object")
        row = item.get("row")
        verdict = item.get("verdict")
        if not isinstance(row, int) or isinstance(row, bool):
            raise AdjudicateError(f"sheet {sheet!r}: adjudicator row number missing or not an int")
        if verdict not in ("keep", "drop"):
            raise AdjudicateError(f"sheet {sheet!r}: adjudicator verdict {verdict!r} not in keep/drop")
        if row in verdicts:
            raise AdjudicateError(f"sheet {sheet!r}: adjudicator returned duplicate row {row}")
        verdicts[row] = verdict
    return verdicts


def _cache_suffix(input_text: str) -> str:
    digest = hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:_INPUT_HASH_LEN]
    return f"{CACHE_SUFFIX_PREFIX}_{digest}"


def _usage_to_payload(usage: UsageRecord) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cost_usd": usage.cost_usd,
        "api_calls": usage.api_calls,
    }


def _usage_from_payload(payload: Mapping[str, Any]) -> UsageRecord:
    return UsageRecord(
        input_tokens=payload["input_tokens"],
        cached_input_tokens=payload["cached_input_tokens"],
        output_tokens=payload["output_tokens"],
        reasoning_tokens=payload["reasoning_tokens"],
        cost_usd=payload["cost_usd"],
        api_calls=payload["api_calls"],
    )


def _keep_all(
    sheet: str, proposed: Sequence[int], usage: UsageRecord, *, degraded: bool, note: str
) -> AdjudicateResult:
    return AdjudicateResult(
        sheet=sheet,
        notes=note,
        keep_rows=tuple(proposed),
        dropped_rows=(),
        usage=usage,
        from_cache=False,
        degraded=degraded,
    )


def _append_correction(input_text: str, proposed: Sequence[int], got: Sequence[int]) -> str:
    return (
        f"{input_text}\n\n[CORRECTION] Your previous answer returned verdicts for rows {sorted(got)}, but the "
        f"proposed rows are EXACTLY {sorted(proposed)}. Answer again with one verdict per proposed row — the same "
        f"rows, no extras, no omissions."
    )


async def _call_and_parse(
    client: AsyncOpenAI,
    spec: _CallSpec,
    input_text: str,
    *,
    semaphore: asyncio.Semaphore,
    sheet: str,
    max_retries: int,
) -> tuple[dict[int, str], UsageRecord]:
    response, _attempts = await _call_llm_with_retry(
        client, spec, input_text, semaphore=semaphore, sheet=f"{sheet}::adjudicate", max_retries=max_retries
    )
    text = _extract_valid_output_text(f"{sheet}::adjudicate", response)
    verdicts = _parse_verdicts(sheet, text)
    usage = usage_from_response(spec.model, response)
    return verdicts, usage


async def adjudicate_sheet(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    sheet: str,
    variant_name: str,
    prompt_version: str,
    input_text: str,
    proposed_rows: Sequence[int],
    run_dir: Path,
    model: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    service_tier: str | None = None,
    max_retries: int = LLM_MAX_RETRIES,
) -> AdjudicateResult:
    proposed_unique = tuple(dict.fromkeys(proposed_rows))
    if not proposed_unique:
        return _keep_all(sheet, (), UsageRecord(), degraded=False, note="(no proposed rows)")
    proposed_set = set(proposed_unique)

    cache_path = _sheet_cache_path(
        run_dir,
        task=task,
        sheet=sheet,
        variant_name=variant_name,
        prompt_version=prompt_version,
        cache_suffix=_cache_suffix(input_text),
    )
    cached = _read_cache_json(cache_path)
    if cached is not None:
        cached_keep = set(cached["keep_rows"])
        return AdjudicateResult(
            sheet=sheet,
            notes=cached["notes"],
            keep_rows=tuple(r for r in proposed_unique if r in cached_keep),
            dropped_rows=tuple(r for r in proposed_unique if r not in cached_keep),
            usage=_usage_from_payload(cached["usage"]),
            from_cache=True,
            degraded=False,
        )

    resolved_model = model if model is not None else settings.model
    resolved_effort = (
        reasoning_effort if reasoning_effort is not None else _coerce_reasoning_effort(settings.reasoning_effort)
    )
    resolved_tier = service_tier if service_tier is not None else settings.service_tier
    spec = _CallSpec(
        model=resolved_model,
        instructions=ADJUDICATE_SYSTEM_PROMPT,
        reasoning_effort=resolved_effort,
        service_tier=resolved_tier,
        schema_name=ADJUDICATE_SCHEMA_NAME,
        json_schema=ADJUDICATE_JSON_SCHEMA,
    )
    semaphore = get_llm_semaphore(settings.max_concurrent_llm_calls)

    try:
        verdicts, usage = await _call_and_parse(
            client, spec, input_text, semaphore=semaphore, sheet=sheet, max_retries=max_retries
        )
        if set(verdicts) != proposed_set:
            corrected = _append_correction(input_text, proposed_unique, list(verdicts))
            verdicts2, usage2 = await _call_and_parse(
                client, spec, corrected, semaphore=semaphore, sheet=sheet, max_retries=max_retries
            )
            usage = usage.combined_with(usage2)
            if set(verdicts2) != proposed_set:
                logger.warning(
                    "adjudicator row set mismatch for sheet %r after retry (proposed %d, got %d); keeping all",
                    sheet,
                    len(proposed_set),
                    len(verdicts2),
                )
                return _keep_all(sheet, proposed_unique, usage, degraded=True, note="membership mismatch; kept all")
            verdicts = verdicts2
    except ClassifyError as exc:
        logger.warning("adjudication failed for sheet %r; keeping all (pass-1 behaviour): %s", sheet, exc)
        return _keep_all(sheet, proposed_unique, UsageRecord(), degraded=True, note=f"call failed; kept all ({exc})")

    kept = tuple(r for r in proposed_unique if verdicts[r] == "keep")
    dropped = tuple(r for r in proposed_unique if verdicts[r] == "drop")
    note = f"kept {len(kept)}/{len(proposed_unique)} proposed rows"
    _atomic_write_json(cache_path, {"notes": note, "keep_rows": list(kept), "usage": _usage_to_payload(usage)})
    return AdjudicateResult(
        sheet=sheet, notes=note, keep_rows=kept, dropped_rows=dropped, usage=usage, from_cache=False, degraded=False
    )
