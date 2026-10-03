from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.responses import Response
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from formuloom.classify import (
    CELL_JSON_SCHEMA,
    CELL_SCHEMA_NAME,
    RETRY_WAIT_MAX_SECONDS,
    RETRY_WAIT_MULTIPLIER_SECONDS,
    RETRYABLE_EXCEPTIONS,
    ClassifyError,
    ClassifySheetResult,
    SheetClassifyError,
    UnknownReferenceError,
    UsageRecord,
    _append_correction,
    _atomic_write_json,
    _cache_payload_to_result,
    _coerce_reasoning_effort,
    _extract_valid_output_text,
    _parse_output,
    _read_cache_json,
    _result_to_cache_payload,
    _sheet_cache_path,
    _typed_refs,
    _unknown_refs,
    get_llm_semaphore,
    usage_from_response,
)
from formuloom.constants import LLM_MAX_RETRIES
from formuloom.prompts.v1 import CELL_SCHEMA_DESCRIPTION, SYSTEM_PROMPT
from formuloom.schema import CellEntry, ReasoningEffort
from formuloom.settings import Settings

V6_VARIANT_NAME = "V6"

V6_MODE = "cell"

CONTAINER_MEMORY_LIMIT = "1g"

CONTAINER_SESSION_USD: dict[str, float] = {"1g": 0.03, "4g": 0.12, "16g": 0.48, "64g": 1.92}

FILE_UPLOAD_PURPOSE: Literal["assistants"] = "assistants"

INIT_CONTAINER_FILENAME = "init.xlsx"
COMPLETE_CONTAINER_FILENAME = "complete.xlsx"
CONTAINER_MOUNT = "/mnt/data"


class ToolingError(ClassifyError):
    pass


class ContainerSetupError(ToolingError):

    def __init__(self, task: str, cause: Exception) -> None:
        self.task = task
        super().__init__(f"failed to set up code-interpreter container for task {task!r}: {cause}")
        self.__cause__ = cause


def container_cost_usd(n_sessions: int, memory_limit: str = CONTAINER_MEMORY_LIMIT) -> float:
    try:
        per_session = CONTAINER_SESSION_USD[memory_limit]
    except KeyError as exc:
        raise ToolingError(
            f"no container price for memory tier {memory_limit!r} (known: {sorted(CONTAINER_SESSION_USD)})"
        ) from exc
    if n_sessions < 0:
        raise ToolingError(f"n_sessions must be >= 0, got {n_sessions}")
    return n_sessions * per_session


def v6_instructions() -> str:
    tooling_directive = (
        "You have a Python code-interpreter tool with two Excel workbooks already mounted "
        f"in the container: {CONTAINER_MOUNT}/{INIT_CONTAINER_FILENAME} (the BEFORE state) "
        f"and {CONTAINER_MOUNT}/{COMPLETE_CONTAINER_FILENAME} (the AFTER state). Use the "
        "tool to open BOTH workbooks with openpyxl (load once with data_only=True for "
        "cached values and once with data_only=False for formulas) and/or pandas, and "
        "examine the specific changed cells you are asked about — their formulas, their "
        "computed values, their row/column labels, and how they feed other cells. Base your "
        "classification on what you actually observe in the workbook, not on guesswork. Do "
        "the file inspection with the tool BEFORE writing your final answer."
    )
    return f"{SYSTEM_PROMPT}\n{tooling_directive}\n{CELL_SCHEMA_DESCRIPTION}"


def build_v6_sheet_input(sheet: str, candidate_refs: Sequence[str], instructions: str) -> str:
    refs = ", ".join(candidate_refs)
    return (
        f"Classify the changed cells of ONE sheet of this financial model.\n\n"
        f"Target sheet name (exact): {sheet!r}\n\n"
        f"The following {len(candidate_refs)} A1 cell references are the changed candidate "
        f"cells on that sheet (these are the ONLY cells you may classify; every reference in "
        f"your `final_cells` must be one of these, and you list only the ones you judge "
        f"FINAL — omit all intermediates):\n<candidates>\n{refs}\n</candidates>\n\n"
        f"Open {CONTAINER_MOUNT}/{INIT_CONTAINER_FILENAME} and "
        f"{CONTAINER_MOUNT}/{COMPLETE_CONTAINER_FILENAME} in the code interpreter, select "
        f"sheet {sheet!r}, and inspect these candidate cells (values, formulas, labels, "
        f"what they roll up into) before deciding.\n\n"
        f"<task_instructions>\n{instructions}\n</task_instructions>"
    )


def code_interpreter_tool(container_id: str) -> dict[str, Any]:
    return {"type": "code_interpreter", "container": container_id}


def _cell_text_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": CELL_SCHEMA_NAME,
            "schema": CELL_JSON_SCHEMA,
            "strict": True,
        }
    }


@dataclass(frozen=True)
class TaskContainer:

    task: str
    container_id: str
    init_file_id: str
    complete_file_id: str
    memory_limit: str


async def open_task_container(
    client: AsyncOpenAI, *, task: str, init_path: Path, complete_path: Path, memory_limit: str = CONTAINER_MEMORY_LIMIT
) -> TaskContainer:
    try:
        init_bytes = init_path.read_bytes()
        complete_bytes = complete_path.read_bytes()
        init_file = await client.files.create(file=(INIT_CONTAINER_FILENAME, init_bytes), purpose=FILE_UPLOAD_PURPOSE)
        complete_file = await client.files.create(
            file=(COMPLETE_CONTAINER_FILENAME, complete_bytes), purpose=FILE_UPLOAD_PURPOSE
        )
        container = await client.containers.create(
            name=f"v6-{task}",
            file_ids=[init_file.id, complete_file.id],
            memory_limit=cast(Any, memory_limit),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error with context, never swallowed
        raise ContainerSetupError(task, exc) from exc
    return TaskContainer(
        task=task,
        container_id=container.id,
        init_file_id=init_file.id,
        complete_file_id=complete_file.id,
        memory_limit=memory_limit,
    )


def parse_v6_response(
    sheet: str, response: Response, candidates: Sequence[str], model: str
) -> tuple[str, list[str], UsageRecord]:
    text = _extract_valid_output_text(sheet, response)
    notes, refs = _parse_output(sheet, "cell", text)
    usage = usage_from_response(model, response)
    return notes, cast("list[str]", refs), usage


async def _call_v6_once(
    client: AsyncOpenAI,
    *,
    model: str,
    instructions: str,
    input_text: str,
    container_id: str,
    reasoning_effort: ReasoningEffort,
    service_tier: str | None,
) -> Response:
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "reasoning": {"effort": reasoning_effort},
        "tools": [code_interpreter_tool(container_id)],
        "text": _cell_text_format(),
    }
    if service_tier is not None:
        kwargs["service_tier"] = service_tier
    response = await client.responses.create(**kwargs)
    return cast(Response, response)


async def _call_v6_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    instructions: str,
    input_text: str,
    container_id: str,
    reasoning_effort: ReasoningEffort,
    service_tier: str | None,
    semaphore: asyncio.Semaphore,
    sheet: str,
    max_retries: int,
) -> Response:
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
                async with semaphore:
                    response = await _call_v6_once(
                        client,
                        model=model,
                        instructions=instructions,
                        input_text=input_text,
                        container_id=container_id,
                        reasoning_effort=reasoning_effort,
                        service_tier=service_tier,
                    )
    except RETRYABLE_EXCEPTIONS as exc:
        raise SheetClassifyError(sheet, exc) from exc
    assert response is not None
    return response


async def classify_sheet_v6(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    sheet: str,
    container_id: str,
    candidate_refs: Sequence[str],
    instructions: str,
    run_dir: Path,
    prompt_version: str = "v1",
    model: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    service_tier: str | None = None,
    max_retries: int = LLM_MAX_RETRIES,
) -> ClassifySheetResult:
    cache_path = _sheet_cache_path(
        run_dir, task=task, sheet=sheet, variant_name=V6_VARIANT_NAME, prompt_version=prompt_version
    )
    cached = _read_cache_json(cache_path)
    if cached is not None:
        return _cache_payload_to_result(sheet, "cell", cached)

    resolved_model = model if model is not None else settings.model
    resolved_effort = (
        reasoning_effort if reasoning_effort is not None else _coerce_reasoning_effort(settings.reasoning_effort)
    )
    resolved_tier = service_tier if service_tier is not None else settings.service_tier
    prefix = v6_instructions()
    input_text = build_v6_sheet_input(sheet, candidate_refs, instructions)
    semaphore = get_llm_semaphore(settings.max_concurrent_llm_calls)

    response = await _call_v6_with_retry(
        client,
        model=resolved_model,
        instructions=prefix,
        input_text=input_text,
        container_id=container_id,
        reasoning_effort=resolved_effort,
        service_tier=resolved_tier,
        semaphore=semaphore,
        sheet=sheet,
        max_retries=max_retries,
    )
    notes, refs, usage = parse_v6_response(sheet, response, candidate_refs, resolved_model)

    unknown = _unknown_refs("cell", refs, candidate_refs)
    if unknown:
        corrected = _append_correction(input_text, "cell", unknown)
        response2 = await _call_v6_with_retry(
            client,
            model=resolved_model,
            instructions=prefix,
            input_text=corrected,
            container_id=container_id,
            reasoning_effort=resolved_effort,
            service_tier=resolved_tier,
            semaphore=semaphore,
            sheet=sheet,
            max_retries=max_retries,
        )
        notes2, refs2, usage2 = parse_v6_response(sheet, response2, candidate_refs, resolved_model)
        usage = usage.combined_with(usage2)
        still_unknown = _unknown_refs("cell", refs2, candidate_refs)
        if still_unknown:
            raise UnknownReferenceError(sheet, still_unknown)
        notes, refs = notes2, refs2

    result = ClassifySheetResult(
        sheet=sheet,
        mode="cell",
        notes=notes,
        final_refs=_typed_refs("cell", refs),
        usage=usage,
        from_cache=False,
    )
    _atomic_write_json(cache_path, _result_to_cache_payload(result))
    return result


def candidate_ref_strings(candidates: Sequence[CellEntry]) -> list[str]:
    return [entry.cell for entry in candidates]
