from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import re
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.responses import Response
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from formuloom.constants import LLM_MAX_RETRIES
from formuloom.prompts import ClassifyMode, get_prompt_version
from formuloom.schema import ReasoningEffort
from formuloom.settings import Settings

logger = logging.getLogger(__name__)

RETRY_WAIT_MULTIPLIER_SECONDS = 1.0

RETRY_WAIT_MAX_SECONDS = 20.0

CORRECTIVE_RETRY_LIMIT = 1

CACHE_SUBDIR = "sheets"

TASK_PROFILE_CACHE_SUBDIR = "task_profile"

ROW_SCHEMA_NAME = "row_classification"
CELL_SCHEMA_NAME = "cell_classification"
TASK_PROFILE_SCHEMA_NAME = "task_profile"

V0_INPUT_COLORED_FRACTION_THRESHOLD = 0.5

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)

ROW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "notes": {"type": "string", "description": "1-3 sentence rationale, written before the row list."},
        "final_rows": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Row numbers classified FINAL; omit all intermediates.",
        },
    },
    "required": ["notes", "final_rows"],
    "additionalProperties": False,
}
CELL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "notes": {"type": "string", "description": "1-3 sentence rationale, written before the cell list."},
        "final_cells": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A1-style cell refs classified FINAL; omit all intermediates.",
        },
    },
    "required": ["notes", "final_cells"],
    "additionalProperties": False,
}
TASK_PROFILE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workbook_purpose": {"type": "string"},
        "model_type": {"type": "string"},
        "expected_deliverables": {"type": "array", "items": {"type": "string"}},
        "likely_capstone_outputs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["workbook_purpose", "model_type", "expected_deliverables", "likely_capstone_outputs"],
    "additionalProperties": False,
}


def _schema_for(mode: ClassifyMode) -> tuple[str, dict[str, Any]]:
    return (ROW_SCHEMA_NAME, ROW_JSON_SCHEMA) if mode == "row" else (CELL_SCHEMA_NAME, CELL_JSON_SCHEMA)


class ClassifyError(Exception):
    pass


class RefusalError(ClassifyError):

    def __init__(self, sheet: str, refusal_text: str) -> None:
        self.sheet = sheet
        self.refusal_text = refusal_text
        super().__init__(f"model refused sheet {sheet!r}: {refusal_text}")


class TruncatedResponseError(ClassifyError):

    def __init__(self, sheet: str, reason: str | None) -> None:
        self.sheet = sheet
        self.reason = reason
        super().__init__(f"response for sheet {sheet!r} was truncated (reason={reason!r})")


class InvalidJSONError(ClassifyError):

    def __init__(self, sheet: str, raw_text: str, cause: Exception) -> None:
        self.sheet = sheet
        self.raw_text = raw_text
        super().__init__(f"sheet {sheet!r}: model output was not valid/well-shaped JSON: {cause}")
        self.__cause__ = cause


class UnknownReferenceError(ClassifyError):

    def __init__(self, sheet: str, unknown_refs: frozenset[str]) -> None:
        self.sheet = sheet
        self.unknown_refs = unknown_refs
        super().__init__(f"sheet {sheet!r}: model returned refs outside the candidate set: {sorted(unknown_refs)}")


class SheetClassifyError(ClassifyError):

    def __init__(self, sheet: str, cause: Exception) -> None:
        self.sheet = sheet
        super().__init__(f"sheet {sheet!r} failed after exhausting retries: {cause}")
        self.__cause__ = cause


class TaskProfileError(ClassifyError):
    pass


@dataclass(frozen=True)
class ModelPrice:

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5.4": ModelPrice(input_per_million=2.50, cached_input_per_million=0.25, output_per_million=15.00),
    "gpt-5.4-mini": ModelPrice(input_per_million=0.75, cached_input_per_million=0.075, output_per_million=4.50),
    "gpt-5.4-nano": ModelPrice(input_per_million=0.20, cached_input_per_million=0.02, output_per_million=1.25),
    "gpt-5.5": ModelPrice(input_per_million=5.00, cached_input_per_million=0.50, output_per_million=30.00),
    "gpt-5.5-2026-04-23": ModelPrice(input_per_million=5.00, cached_input_per_million=0.50, output_per_million=30.00),
}


def model_price(model: str) -> ModelPrice:
    try:
        return MODEL_PRICES[model]
    except KeyError as exc:
        raise ClassifyError(f"no price table entry for model {model!r} (known: {sorted(MODEL_PRICES)})") from exc


def compute_cost_usd(model: str, *, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    price = model_price(model)
    uncached_input = max(0, input_tokens - cached_input_tokens)
    cost = (
        uncached_input * price.input_per_million
        + cached_input_tokens * price.cached_input_per_million
        + output_tokens * price.output_per_million
    )
    return cost / 1_000_000


@dataclass(frozen=True)
class UsageRecord:

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0

    def combined_with(self, other: UsageRecord) -> UsageRecord:
        return UsageRecord(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            api_calls=self.api_calls + other.api_calls,
        )


def usage_from_response(model: str, response: Response) -> UsageRecord:
    usage = response.usage
    if usage is None:
        logger.warning("response %s for model %s has no usage block; recording zero usage", response.id, model)
        return UsageRecord(api_calls=1)
    cost = compute_cost_usd(
        model,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.input_tokens_details.cached_tokens,
        output_tokens=usage.output_tokens,
    )
    return UsageRecord(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.input_tokens_details.cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        cost_usd=cost,
        api_calls=1,
    )


def build_async_client(settings: Settings) -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise ClassifyError("OPENAI_API_KEY is not set; classify() requires it (offline paths never call this)")
    kwargs: dict[str, Any] = {
        "api_key": settings.openai_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 0,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return AsyncOpenAI(**kwargs)


_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


def get_llm_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    per_loop = _semaphores.get(loop)
    if per_loop is None:
        per_loop = {}
        _semaphores[loop] = per_loop
    sem = per_loop.get(max_concurrent)
    if sem is None:
        sem = asyncio.Semaphore(max_concurrent)
        per_loop[max_concurrent] = sem
    return sem


def reset_llm_semaphores() -> None:
    _semaphores.clear()


_VALID_REASONING_EFFORTS: frozenset[str] = frozenset({"none", "low", "medium"})


def _coerce_reasoning_effort(value: str) -> ReasoningEffort:
    if value not in _VALID_REASONING_EFFORTS:
        raise ClassifyError(f"invalid reasoning_effort {value!r} (expected one of {sorted(_VALID_REASONING_EFFORTS)})")
    return cast(ReasoningEffort, value)


_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_key_part(part: str) -> str:
    return _SAFE_KEY_RE.sub("_", part)


def _sheet_cache_path(
    run_dir: Path, *, task: str, sheet: str, variant_name: str, prompt_version: str, cache_suffix: str = ""
) -> Path:
    parts = [task, sheet, variant_name, prompt_version]
    if cache_suffix:
        parts.append(cache_suffix)
    key = "__".join(_sanitize_key_part(p) for p in parts)
    return run_dir / CACHE_SUBDIR / f"{key}.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _read_cache_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable classify cache %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("ignoring classify cache %s: top-level value is not an object", path)
        return None
    return payload


@dataclass(frozen=True)
class ClassifySheetResult:

    sheet: str
    mode: ClassifyMode
    notes: str
    final_refs: tuple[int, ...] | tuple[str, ...]
    usage: UsageRecord
    from_cache: bool


def _typed_refs(mode: ClassifyMode, refs: Sequence[int] | Sequence[str]) -> tuple[int, ...] | tuple[str, ...]:
    if mode == "row":
        return tuple(int(r) for r in refs)
    return tuple(str(r) for r in refs)


def _result_to_cache_payload(result: ClassifySheetResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "notes": result.notes,
        "final_refs": list(result.final_refs),
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
            "output_tokens": result.usage.output_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
            "cost_usd": result.usage.cost_usd,
            "api_calls": result.usage.api_calls,
        },
    }


def _cache_payload_to_result(sheet: str, mode: ClassifyMode, payload: Mapping[str, Any]) -> ClassifySheetResult:
    usage_payload = payload["usage"]
    usage = UsageRecord(
        input_tokens=usage_payload["input_tokens"],
        cached_input_tokens=usage_payload["cached_input_tokens"],
        output_tokens=usage_payload["output_tokens"],
        reasoning_tokens=usage_payload["reasoning_tokens"],
        cost_usd=usage_payload["cost_usd"],
        api_calls=usage_payload["api_calls"],
    )
    final_refs = _typed_refs(mode, payload["final_refs"])
    return ClassifySheetResult(
        sheet=sheet, mode=mode, notes=payload["notes"], final_refs=final_refs, usage=usage, from_cache=True
    )


def _extract_valid_output_text(sheet: str, response: Response) -> str:
    if response.status == "incomplete":
        reason = response.incomplete_details.reason if response.incomplete_details else None
        raise TruncatedResponseError(sheet, reason)
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        if getattr(item, "status", None) == "incomplete":
            raise TruncatedResponseError(sheet, "message output item incomplete")
        for block in getattr(item, "content", []):
            if getattr(block, "type", None) == "refusal":
                raise RefusalError(sheet, block.refusal)
    text = response.output_text
    if not text:
        raise TruncatedResponseError(sheet, "empty output_text")
    return text


def _parse_output(sheet: str, mode: ClassifyMode, text: str) -> tuple[str, list[int] | list[str]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidJSONError(sheet, text, exc) from exc
    if not isinstance(payload, dict):
        raise InvalidJSONError(sheet, text, ValueError("top-level JSON value is not an object"))
    notes = payload.get("notes")
    if not isinstance(notes, str):
        raise InvalidJSONError(sheet, text, ValueError("'notes' missing or not a string"))
    key = "final_rows" if mode == "row" else "final_cells"
    refs = payload.get(key)
    if not isinstance(refs, list):
        raise InvalidJSONError(sheet, text, ValueError(f"{key!r} missing or not a list"))
    expected_type: type = int if mode == "row" else str
    if not all(isinstance(r, expected_type) for r in refs):
        raise InvalidJSONError(sheet, text, ValueError(f"{key!r} contains a non-{expected_type.__name__} element"))
    return notes, cast("list[int] | list[str]", refs)


def _unknown_refs(
    mode: ClassifyMode, refs: Sequence[int] | Sequence[str], candidates: Sequence[int] | Sequence[str]
) -> frozenset[str]:
    candidate_set = {str(c) for c in candidates}
    return frozenset(str(r) for r in refs if str(r) not in candidate_set)


def _append_correction(input_text: str, mode: ClassifyMode, unknown: frozenset[str]) -> str:
    key = "final_rows" if mode == "row" else "final_cells"
    return (
        f"{input_text}\n\n"
        f"[CORRECTION] Your previous answer's {key!r} included value(s) not present in "
        f"the candidate list above: {sorted(unknown)}. These are invalid. Answer again "
        f"using ONLY values drawn from the candidate list; every entry in {key!r} must be "
        f"a member of the candidates."
    )


@dataclass(frozen=True)
class _CallSpec:

    model: str
    instructions: str
    reasoning_effort: ReasoningEffort
    service_tier: str | None
    schema_name: str
    json_schema: dict[str, Any]


async def _call_llm_once(client: AsyncOpenAI, spec: _CallSpec, input_text: str) -> Response:
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "instructions": spec.instructions,
        "input": input_text,
        "reasoning": {"effort": spec.reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": spec.schema_name,
                "schema": spec.json_schema,
                "strict": True,
            }
        },
    }
    if spec.service_tier is not None:
        kwargs["service_tier"] = spec.service_tier

    response = await client.responses.create(**kwargs)
    return cast(Response, response)


async def _call_llm_with_retry(
    client: AsyncOpenAI,
    spec: _CallSpec,
    input_text: str,
    *,
    semaphore: asyncio.Semaphore,
    sheet: str,
    max_retries: int,
) -> tuple[Response, int]:
    attempts = 0
    response: Response | None = None
    retrying = AsyncRetrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_random_exponential(multiplier=RETRY_WAIT_MULTIPLIER_SECONDS, max=RETRY_WAIT_MAX_SECONDS),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    try:
        async for attempt in retrying:
            with attempt:
                attempts += 1
                async with semaphore:
                    response = await _call_llm_once(client, spec, input_text)
    except RETRYABLE_EXCEPTIONS as exc:
        raise SheetClassifyError(sheet, exc) from exc
    assert response is not None
    return response, attempts


async def _call_and_parse(
    client: AsyncOpenAI,
    spec: _CallSpec,
    input_text: str,
    *,
    semaphore: asyncio.Semaphore,
    sheet: str,
    mode: ClassifyMode,
    max_retries: int,
) -> tuple[str, list[int] | list[str], UsageRecord]:
    response, _attempts = await _call_llm_with_retry(
        client, spec, input_text, semaphore=semaphore, sheet=sheet, max_retries=max_retries
    )
    text = _extract_valid_output_text(sheet, response)
    notes, refs = _parse_output(sheet, mode, text)
    usage = usage_from_response(spec.model, response)
    return notes, refs, usage


async def _classify_with_correction(
    client: AsyncOpenAI,
    spec: _CallSpec,
    *,
    input_text: str,
    semaphore: asyncio.Semaphore,
    sheet: str,
    mode: ClassifyMode,
    candidates: Sequence[int] | Sequence[str],
    max_retries: int,
) -> ClassifySheetResult:
    notes, refs, usage = await _call_and_parse(
        client, spec, input_text, semaphore=semaphore, sheet=sheet, mode=mode, max_retries=max_retries
    )
    unknown = _unknown_refs(mode, refs, candidates)
    if unknown:
        corrected_input = _append_correction(input_text, mode, unknown)
        notes2, refs2, usage2 = await _call_and_parse(
            client, spec, corrected_input, semaphore=semaphore, sheet=sheet, mode=mode, max_retries=max_retries
        )
        usage = usage.combined_with(usage2)
        unknown2 = _unknown_refs(mode, refs2, candidates)
        if unknown2:
            raise UnknownReferenceError(sheet, unknown2)
        notes, refs = notes2, refs2
    return ClassifySheetResult(
        sheet=sheet, mode=mode, notes=notes, final_refs=_typed_refs(mode, refs), usage=usage, from_cache=False
    )


async def classify_sheet(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    sheet: str,
    variant_name: str,
    prompt_version: str,
    mode: ClassifyMode,
    context: str,
    candidates: Sequence[int] | Sequence[str],
    run_dir: Path,
    model: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    service_tier: str | None = None,
    max_retries: int = LLM_MAX_RETRIES,
    cache_suffix: str = "",
) -> ClassifySheetResult:
    cache_path = _sheet_cache_path(
        run_dir,
        task=task,
        sheet=sheet,
        variant_name=variant_name,
        prompt_version=prompt_version,
        cache_suffix=cache_suffix,
    )
    cached = _read_cache_json(cache_path)
    if cached is not None:
        return _cache_payload_to_result(sheet, mode, cached)

    prompt_v = get_prompt_version(prompt_version)
    schema_name, json_schema = _schema_for(mode)
    resolved_model = model if model is not None else settings.model
    resolved_effort = (
        reasoning_effort if reasoning_effort is not None else _coerce_reasoning_effort(settings.reasoning_effort)
    )
    resolved_tier = service_tier if service_tier is not None else settings.service_tier
    spec = _CallSpec(
        model=resolved_model,
        instructions=prompt_v.static_prefix(mode),
        reasoning_effort=resolved_effort,
        service_tier=resolved_tier,
        schema_name=schema_name,
        json_schema=json_schema,
    )
    semaphore = get_llm_semaphore(settings.max_concurrent_llm_calls)

    result = await _classify_with_correction(
        client,
        spec,
        input_text=context,
        semaphore=semaphore,
        sheet=sheet,
        mode=mode,
        candidates=candidates,
        max_retries=max_retries,
    )
    _atomic_write_json(cache_path, _result_to_cache_payload(result))
    return result


@dataclass(frozen=True)
class TaskProfile:

    workbook_purpose: str
    model_type: str
    expected_deliverables: tuple[str, ...]
    likely_capstone_outputs: tuple[str, ...]


def _task_profile_cache_key(instructions: str, sheet_names: Sequence[str], prompt_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(instructions.encode("utf-8"))
    digest.update(b"\0")
    digest.update("\0".join(sheet_names).encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_version.encode("utf-8"))
    return digest.hexdigest()


def _profile_to_payload(profile: TaskProfile) -> dict[str, Any]:
    return {
        "workbook_purpose": profile.workbook_purpose,
        "model_type": profile.model_type,
        "expected_deliverables": list(profile.expected_deliverables),
        "likely_capstone_outputs": list(profile.likely_capstone_outputs),
    }


def _profile_from_payload(payload: Mapping[str, Any]) -> TaskProfile:
    return TaskProfile(
        workbook_purpose=payload["workbook_purpose"],
        model_type=payload["model_type"],
        expected_deliverables=tuple(payload["expected_deliverables"]),
        likely_capstone_outputs=tuple(payload["likely_capstone_outputs"]),
    )


def _parse_task_profile(task: str, text: str) -> TaskProfile:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TaskProfileError(f"task profile for {task!r}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TaskProfileError(f"task profile for {task!r}: top-level JSON value is not an object")
    try:
        return TaskProfile(
            workbook_purpose=str(payload["workbook_purpose"]),
            model_type=str(payload["model_type"]),
            expected_deliverables=tuple(str(x) for x in payload["expected_deliverables"]),
            likely_capstone_outputs=tuple(str(x) for x in payload["likely_capstone_outputs"]),
        )
    except (KeyError, TypeError) as exc:
        raise TaskProfileError(f"task profile for {task!r}: missing/malformed field: {exc}") from exc


async def fetch_task_profile(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    instructions: str,
    sheet_names: Sequence[str],
    prompt_version: str,
    run_dir: Path,
    model: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    service_tier: str | None = None,
    max_retries: int = LLM_MAX_RETRIES,
) -> TaskProfile:
    key = _task_profile_cache_key(instructions, sheet_names, prompt_version)
    cache_path = run_dir / TASK_PROFILE_CACHE_SUBDIR / f"{key}.json"
    cached = _read_cache_json(cache_path)
    if cached is not None:
        return _profile_from_payload(cached)

    prompt_v = get_prompt_version(prompt_version)
    resolved_model = model if model is not None else settings.model
    resolved_effort = (
        reasoning_effort if reasoning_effort is not None else _coerce_reasoning_effort(settings.reasoning_effort)
    )
    resolved_tier = service_tier if service_tier is not None else settings.service_tier
    spec = _CallSpec(
        model=resolved_model,
        instructions=prompt_v.task_profile_system_prompt,
        reasoning_effort=resolved_effort,
        service_tier=resolved_tier,
        schema_name=TASK_PROFILE_SCHEMA_NAME,
        json_schema=TASK_PROFILE_JSON_SCHEMA,
    )
    semaphore = get_llm_semaphore(settings.max_concurrent_llm_calls)
    input_text = f"Sheet names: {', '.join(sheet_names)}\n\nInstructions:\n{instructions}"

    response, _attempts = await _call_llm_with_retry(
        client, spec, input_text, semaphore=semaphore, sheet=f"{task}::task_profile", max_retries=max_retries
    )
    text = _extract_valid_output_text(f"{task}::task_profile", response)
    profile = _parse_task_profile(task, text)
    _atomic_write_json(cache_path, _profile_to_payload(profile))
    return profile


class RowFeatureView(Protocol):

    n_dependents_outside_row: int
    aggregates_range: bool
    lexicon_hits: list[str]
    input_colored_fraction: float
    bold_or_bordered: bool
    label: str | None
    has_formula: bool


def classify_v0_row(features: RowFeatureView) -> bool:
    sink_or_aggregate = features.aggregates_range or features.n_dependents_outside_row == 0
    surfaced = bool(features.lexicon_hits) or features.bold_or_bordered
    not_input_block = features.input_colored_fraction < V0_INPUT_COLORED_FRACTION_THRESHOLD
    return sink_or_aggregate and surfaced and not_input_block


def classify_v0(rows: Mapping[int, RowFeatureView]) -> list[int]:
    return sorted(row for row, features in rows.items() if classify_v0_row(features))


@dataclass(frozen=True)
class VotedClassifySheetResult:

    sheet: str
    mode: ClassifyMode
    k: int
    majority_refs: tuple[int, ...] | tuple[str, ...]
    agreement_rate: float
    per_sample: tuple[ClassifySheetResult, ...]
    usage: UsageRecord


def _majority_vote(
    mode: ClassifyMode, candidates: Sequence[int] | Sequence[str], samples: Sequence[ClassifySheetResult]
) -> tuple[tuple[int, ...] | tuple[str, ...], float]:
    k = len(samples)
    votes: dict[str, int] = {str(c): 0 for c in candidates}
    for sample in samples:
        for ref in sample.final_refs:
            key = str(ref)
            if key in votes:
                votes[key] += 1
    majority_str = [c for c in votes if votes[c] * 2 > k]
    unanimous = sum(1 for count in votes.values() if count == 0 or count == k)
    agreement_rate = unanimous / len(votes) if votes else 1.0
    majority: tuple[int, ...] | tuple[str, ...] = (
        tuple(sorted(int(c) for c in majority_str)) if mode == "row" else tuple(sorted(majority_str))
    )
    return majority, agreement_rate


async def classify_sheet_voted(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    sheet: str,
    variant_name: str,
    prompt_version: str,
    mode: ClassifyMode,
    context: str,
    candidates: Sequence[int] | Sequence[str],
    run_dir: Path,
    k: int,
    model: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    service_tier: str | None = None,
    max_retries: int = LLM_MAX_RETRIES,
) -> VotedClassifySheetResult:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    samples = await asyncio.gather(
        *(
            classify_sheet(
                client,
                settings,
                task=task,
                sheet=sheet,
                variant_name=variant_name,
                prompt_version=prompt_version,
                mode=mode,
                context=context,
                candidates=candidates,
                run_dir=run_dir,
                model=model,
                reasoning_effort=reasoning_effort,
                service_tier=service_tier,
                max_retries=max_retries,
                cache_suffix=f"vote{i}",
            )
            for i in range(k)
        )
    )
    majority_refs, agreement_rate = _majority_vote(mode, candidates, samples)
    total_usage = functools.reduce(lambda acc, sample: acc.combined_with(sample.usage), samples, UsageRecord())
    return VotedClassifySheetResult(
        sheet=sheet,
        mode=mode,
        k=k,
        majority_refs=majority_refs,
        agreement_rate=agreement_rate,
        per_sample=tuple(samples),
        usage=total_usage,
    )
