from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import openai
import typer
from openai import AsyncOpenAI

from formuloom import __version__
from formuloom.adjudicate import adjudicate_sheet, build_adjudication_input
from formuloom.assemble import AssembleError, assemble_diff_file
from formuloom.bundle import BundleError, TaskBundle
from formuloom.classify import (
    ClassifyError,
    UsageRecord,
    build_async_client,
    classify_sheet,
    classify_sheet_voted,
    classify_v0,
    classify_v0_row,
    fetch_task_profile,
)
from formuloom.compose import ComposePipeline
from formuloom.compose import Pipeline as _Pipeline
from formuloom.compose import PipelineResult as _PipelineResult
from formuloom.constants import SECTION_SPLIT_ROW_THRESHOLD
from formuloom.encode import Grouping, SheetContext, sheet_map_line, split_by_sections
from formuloom.encode import TaskProfile as EncodeTaskProfile
from formuloom.ensemble import classify_sheet_diverse
from formuloom.features import FeatureError, RowFeatures, SheetFeatures, TaskFeatures, build_task_features
from formuloom.schema import DiffFile, ScoreMode, TaskScoreDetail, VariantConfig, cell_row
from formuloom.score import (
    aggregate,
    build_error_report,
    build_eval_results,
    score_task,
)
from formuloom.settings import Settings, get_settings
from formuloom.variants import get_variant, is_offline
from formuloom.weak import predict_v8
from formuloom.workbook import WorkbookData, load_workbook_data

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Formuloom: classify changed workbook cells as outputs or intermediates.")

RUNS_ROOT = Path("runs")

ERROR_REPORT_SUBDIR = "errors"
PREDICTIONS_SUBDIR = "predictions"

RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%f"

FAILURES_FILE = "failures.json"

SELECTIVE_VOTING_DISAGREEMENT_THRESHOLD = 0.2

V11_GRANULAR_SUPPORT_MIN_ROWS = 120

V11_GRANULAR_SUPPORT_MIN_CANDIDATES = 1000

V11_DENSE_SOURCE_MIN_CANDIDATES = 500

V11_DENSE_SOURCE_MIN_DENSITY = 0.25

V11_ADJUDICATE_WIDE_PROPOSAL_MIN_CELLS = 500

V11_ADJUDICATE_DENSE_STATEMENT_MIN_DENSITY = 0.24

V11_ADJUDICATE_MAX_INPUT_COLOR_FRACTION = 0.90

_COMMAND_ERRORS: tuple[type[Exception], ...] = (
    BundleError,
    AssembleError,
    FeatureError,
    ClassifyError,
    ValueError,
    NotImplementedError,
    FileNotFoundError,
    OSError,
)

_MODE_ALIASES: dict[str, tuple[ScoreMode, ...]] = {
    "strict": ("strict",),
    "compat": ("annotated",),
    "annotated": ("annotated",),
    "both": ("annotated", "strict"),
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    pass


def run_cli() -> None:
    app()


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_diff_file(path: Path, diff: DiffFile) -> None:
    _atomic_write_text(path, json.dumps(diff.model_dump(), indent=2, ensure_ascii=False) + "\n")


def _default_run_dir(variant: str) -> Path:
    timestamp = datetime.now(UTC).strftime(RUN_DIR_TIMESTAMP_FORMAT)
    return RUNS_ROOT / f"{timestamp}-{variant}"


def _resolve_modes(mode: str) -> tuple[ScoreMode, ...]:
    try:
        return _MODE_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(f"unknown --mode {mode!r} (expected one of {sorted(_MODE_ALIASES)})") from exc


def _discover_tasks(data_root: Path) -> list[tuple[str, Path]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    tasks = sorted(
        (child.name, child) for child in data_root.iterdir() if child.is_dir() and (child / "raw_diff.json").is_file()
    )
    if not tasks:
        raise FileNotFoundError(f"no task bundles found under {data_root} (expected subdirs with raw_diff.json)")
    return tasks


@dataclass
class _V0RowView:

    n_dependents_outside_row: int
    aggregates_range: bool
    lexicon_hits: list[str]
    input_colored_fraction: float
    bold_or_bordered: bool
    label: str | None
    has_formula: bool


def _v0_row_view(row: RowFeatures) -> _V0RowView:
    return _V0RowView(
        n_dependents_outside_row=row.n_dependents_outside_row,
        aggregates_range=row.aggregates_range,
        lexicon_hits=row.lexicon_hits,
        input_colored_fraction=row.input_colored_fraction,
        bold_or_bordered=row.bold_or_bordered,
        label=row.label,
        has_formula=row.pattern_kind != "none",
    )


def predict_v0(bundle: TaskBundle, *, cache_dir: Path | None = None) -> DiffFile:
    complete = load_workbook_data(bundle.complete_path, cache_dir)
    task_features = build_task_features(bundle, complete)
    predicted_final: dict[str, set[str]] = {}
    for sheet_name in bundle.raw_diff.sheets:
        sheet_features = task_features.sheets.get(sheet_name)
        if sheet_features is None or not sheet_features.rows:
            predicted_final[sheet_name] = set()
            continue
        rows = {rf.row: _v0_row_view(rf) for rf in sheet_features.rows}
        final_rows = set(classify_v0(rows))
        cells: set[str] = set()
        for rf in sheet_features.rows:
            if rf.row in final_rows:
                cells.update(rf.candidate_refs)
        predicted_final[sheet_name] = cells
    return assemble_diff_file(bundle.raw_diff, predicted_final)


@dataclass(frozen=True)
class _PartResult:

    mode: Grouping
    final_refs: tuple[int, ...] | tuple[str, ...]
    usage: UsageRecord


def _selective_voting_should_trigger(
    mode: Grouping,
    probe_refs: tuple[int, ...] | tuple[str, ...],
    part: SheetContext,
    sheet_rows: Sequence[RowFeatures],
) -> bool:
    if mode != "row":
        return False
    part_rows = set(part.rows)
    relevant = [rf for rf in sheet_rows if rf.row in part_rows]
    if not relevant:
        return False
    v0_final = {rf.row for rf in relevant if classify_v0_row(_v0_row_view(rf))}
    llm_final = {int(r) for r in probe_refs}
    disagreement = len(v0_final.symmetric_difference(llm_final)) / len(relevant)
    return disagreement > SELECTIVE_VOTING_DISAGREEMENT_THRESHOLD


async def _classify_part(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    bundle: TaskBundle,
    variant: VariantConfig,
    part: SheetContext,
    run_dir: Path,
    section_split_applied: bool,
    sheet_rows: Sequence[RowFeatures],
) -> _PartResult:
    mode = part.grouping
    candidates: Sequence[int] | Sequence[str] = part.rows if mode == "row" else list(part.candidate_refs)

    if variant.ensemble_prompts:
        ensemble_result = await classify_sheet_diverse(
            client,
            settings,
            task=bundle.name,
            sheet=part.sheet,
            variant_name=variant.name,
            prompt_versions=variant.ensemble_prompts,
            mode=mode,
            context=part.text,
            candidates=candidates,
            run_dir=run_dir,
            model=variant.model,
            reasoning_effort=variant.reasoning_effort,
            samples_per_prompt=variant.voting_k,
        )
        return _PartResult(mode=mode, final_refs=ensemble_result.majority_refs, usage=ensemble_result.usage)

    if variant.voting_k <= 1:
        single_result = await classify_sheet(
            client,
            settings,
            task=bundle.name,
            sheet=part.sheet,
            variant_name=variant.name,
            prompt_version=variant.prompt_version,
            mode=mode,
            context=part.text,
            candidates=candidates,
            run_dir=run_dir,
            model=variant.model,
            reasoning_effort=variant.reasoning_effort,
        )
        return _PartResult(mode=mode, final_refs=single_result.final_refs, usage=single_result.usage)

    if not variant.selective_voting:
        voted = await classify_sheet_voted(
            client,
            settings,
            task=bundle.name,
            sheet=part.sheet,
            variant_name=variant.name,
            prompt_version=variant.prompt_version,
            mode=mode,
            context=part.text,
            candidates=candidates,
            run_dir=run_dir,
            k=variant.voting_k,
            model=variant.model,
            reasoning_effort=variant.reasoning_effort,
        )
        return _PartResult(mode=mode, final_refs=voted.majority_refs, usage=voted.usage)

    probe = await classify_sheet(
        client,
        settings,
        task=bundle.name,
        sheet=part.sheet,
        variant_name=variant.name,
        prompt_version=variant.prompt_version,
        mode=mode,
        context=part.text,
        candidates=candidates,
        run_dir=run_dir,
        model=variant.model,
        reasoning_effort=variant.reasoning_effort,
        cache_suffix="probe",
    )
    trigger = section_split_applied or _selective_voting_should_trigger(mode, probe.final_refs, part, sheet_rows)
    if not trigger:
        return _PartResult(mode=mode, final_refs=probe.final_refs, usage=probe.usage)

    voted = await classify_sheet_voted(
        client,
        settings,
        task=bundle.name,
        sheet=part.sheet,
        variant_name=variant.name,
        prompt_version=variant.prompt_version,
        mode=mode,
        context=part.text,
        candidates=candidates,
        run_dir=run_dir,
        k=variant.voting_k,
        model=variant.model,
        reasoning_effort=variant.reasoning_effort,
    )
    return _PartResult(mode=mode, final_refs=voted.majority_refs, usage=probe.usage.combined_with(voted.usage))


def _row_to_cells_map(rows: Sequence[RowFeatures]) -> dict[int, list[str]]:
    return {rf.row: rf.candidate_refs for rf in rows}


def _apply_v11_precision_guard(sheet: SheetFeatures, final_cells: set[str]) -> set[str]:
    if not final_cells:
        return final_cells
    large_granular_support = (
        len(sheet.rows) >= V11_GRANULAR_SUPPORT_MIN_ROWS
        and sheet.n_candidates >= V11_GRANULAR_SUPPORT_MIN_CANDIDATES
        and sheet.n_cells_with_cross_sheet_dependents > 0
    )
    dense_source_table = (
        sheet.n_candidates >= V11_DENSE_SOURCE_MIN_CANDIDATES
        and sheet.candidate_density >= V11_DENSE_SOURCE_MIN_DENSITY
        and sheet.n_cells_with_cross_sheet_precedents == 0
        and sheet.n_cells_with_cross_sheet_dependents > 0
    )
    if not (large_granular_support or dense_source_table):
        return final_cells
    logger.info(
        "V11 precision guard dropped %d proposed final cells on %s "
        "(rows=%d candidates=%d density=%.3f precedents=%d dependents=%d)",
        len(final_cells),
        sheet.sheet,
        len(sheet.rows),
        sheet.n_candidates,
        sheet.candidate_density,
        sheet.n_cells_with_cross_sheet_precedents,
        sheet.n_cells_with_cross_sheet_dependents,
    )
    return set()


def _should_v11_adjudicate(sheet: SheetFeatures, final_cells: set[str]) -> bool:
    if not final_cells:
        return False
    if sheet.input_colored_fraction >= V11_ADJUDICATE_MAX_INPUT_COLOR_FRACTION:
        return False

    standalone_sink_sheet = sheet.n_cells_with_cross_sheet_dependents == 0
    wide_projection_proposal = len(final_cells) >= V11_ADJUDICATE_WIDE_PROPOSAL_MIN_CELLS
    dense_statement_sheet = (
        sheet.candidate_density >= V11_ADJUDICATE_DENSE_STATEMENT_MIN_DENSITY
        and sheet.n_cells_with_cross_sheet_precedents > 0
    )
    return standalone_sink_sheet or wide_projection_proposal or dense_statement_sheet


async def _adjudicate_sheet(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    bundle: TaskBundle,
    variant: VariantConfig,
    sheet_name: str,
    context: SheetContext,
    task_features: TaskFeatures,
    profile: EncodeTaskProfile | None,
    final_cells: set[str],
    run_dir: Path,
) -> tuple[set[str], UsageRecord]:
    if not final_cells:
        return final_cells, UsageRecord()
    proposed_rows = sorted({cell_row(ref) for ref in final_cells})
    proposed_set = set(proposed_rows)
    proposed_row_lines = [(r, line) for r, line in context.row_lines if r in proposed_set]
    section_titles = [s.title for s in context.sections if s.title]
    capstones = list(profile.likely_capstone_outputs) if profile is not None else []
    adj_input = build_adjudication_input(
        sheet=sheet_name,
        workbook_map_line=sheet_map_line(sheet_name, task_features.sheets[sheet_name]),
        section_titles=section_titles,
        likely_capstone_outputs=capstones,
        proposed_row_lines=proposed_row_lines,
    )
    result = await adjudicate_sheet(
        client,
        settings,
        task=bundle.name,
        sheet=sheet_name,
        variant_name=variant.name,
        prompt_version=variant.prompt_version,
        input_text=adj_input,
        proposed_rows=proposed_rows,
        run_dir=run_dir,
        model=variant.adjudicator_model if variant.adjudicator_model is not None else variant.model,
        reasoning_effort=variant.reasoning_effort,
    )
    kept = set(result.keep_rows)
    pruned = {ref for ref in final_cells if cell_row(ref) in kept}
    return pruned, result.usage


async def _classify_sheet_llm(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    bundle: TaskBundle,
    variant: VariantConfig,
    sheet_name: str,
    task_features: TaskFeatures,
    complete: WorkbookData,
    profile: EncodeTaskProfile | None,
    run_dir: Path,
) -> tuple[set[str], UsageRecord]:
    context = SheetContext.build(
        bundle=bundle,
        complete=complete,
        features=task_features,
        sheet_name=sheet_name,
        variant=variant,
        profile=profile,
    )
    if variant.section_split and len(context.rows) > SECTION_SPLIT_ROW_THRESHOLD:
        parts = split_by_sections(context, SECTION_SPLIT_ROW_THRESHOLD)
    else:
        parts = [context]
    section_split_applied = len(parts) > 1
    sheet_rows = task_features.sheets[sheet_name].rows
    row_to_cells = _row_to_cells_map(sheet_rows)

    part_results = await asyncio.gather(
        *(
            _classify_part(
                client,
                settings,
                bundle=bundle,
                variant=variant,
                part=part,
                run_dir=run_dir,
                section_split_applied=section_split_applied,
                sheet_rows=sheet_rows,
            )
            for part in parts
        )
    )

    final_cells: set[str] = set()
    usage = UsageRecord()
    for part_result in part_results:
        usage = usage.combined_with(part_result.usage)
        if part_result.mode == "row":
            for row in part_result.final_refs:
                final_cells.update(row_to_cells.get(int(row), []))
        else:
            final_cells.update(str(ref) for ref in part_result.final_refs)

    should_adjudicate = variant.adjudicate
    if variant.name == "V11":
        sheet_features = task_features.sheets[sheet_name]
        final_cells = _apply_v11_precision_guard(sheet_features, final_cells)
        should_adjudicate = should_adjudicate and _should_v11_adjudicate(sheet_features, final_cells)

    if should_adjudicate:
        final_cells, adj_usage = await _adjudicate_sheet(
            client,
            settings,
            bundle=bundle,
            variant=variant,
            sheet_name=sheet_name,
            context=context,
            task_features=task_features,
            profile=profile,
            final_cells=final_cells,
            run_dir=run_dir,
        )
        usage = usage.combined_with(adj_usage)
    return final_cells, usage


class _LLMPipeline:

    def __init__(self, config: VariantConfig, settings: Settings) -> None:
        self._config = config
        self._settings = settings

    async def predict(self, bundle: TaskBundle, *, run_dir: Path, best_effort: bool) -> _PipelineResult:
        client = build_async_client(self._settings)
        try:
            complete = load_workbook_data(bundle.complete_path)
            task_features = build_task_features(bundle, complete)
            candidates_by_sheet = bundle.candidate_cells()

            profile: EncodeTaskProfile | None = None
            if self._config.use_task_profile:
                raw_profile = await fetch_task_profile(
                    client,
                    self._settings,
                    task=bundle.name,
                    instructions=bundle.instructions,
                    sheet_names=list(complete.sheets),
                    prompt_version=self._config.prompt_version,
                    run_dir=run_dir,
                    model=self._config.model,
                    reasoning_effort=self._config.reasoning_effort,
                )
                profile = EncodeTaskProfile(
                    workbook_purpose=raw_profile.workbook_purpose,
                    model_type=raw_profile.model_type,
                    expected_deliverables=list(raw_profile.expected_deliverables),
                    likely_capstone_outputs=list(raw_profile.likely_capstone_outputs),
                )

            sheet_names = [s for s in bundle.raw_diff.sheets if candidates_by_sheet.get(s)]

            async def _one_sheet(sheet_name: str) -> tuple[str, set[str], UsageRecord, str | None]:
                try:
                    cells, usage = await _classify_sheet_llm(
                        client,
                        self._settings,
                        bundle=bundle,
                        variant=self._config,
                        sheet_name=sheet_name,
                        task_features=task_features,
                        complete=complete,
                        profile=profile,
                        run_dir=run_dir,
                    )
                    return sheet_name, cells, usage, None
                except (ClassifyError, openai.APIError) as exc:

                    if not best_effort:
                        raise
                    logger.error("sheet %r failed (--best-effort: degrading to all-intermediate): %s", sheet_name, exc)
                    return sheet_name, set(), UsageRecord(), f"{type(exc).__name__}: {exc}"

            outcomes = await asyncio.gather(*(_one_sheet(s) for s in sheet_names))

            predicted_final: dict[str, set[str]] = {}
            failures: dict[str, str] = {}
            total_usage = UsageRecord()
            for sheet_name, cells, usage, error in outcomes:
                total_usage = total_usage.combined_with(usage)
                predicted_final[sheet_name] = cells
                if error is not None:
                    failures[sheet_name] = error

            diff = assemble_diff_file(bundle.raw_diff, predicted_final)
            return _PipelineResult(diff=diff, failures=failures, usage=total_usage)
        finally:
            await client.close()


def _build_pipeline(config: VariantConfig) -> _Pipeline:
    settings = get_settings()
    if config.cascade:
        from formuloom.cascade import CascadePipeline

        return CascadePipeline(config, settings)
    if config.route_variants:
        from formuloom.v15_router import V15RouterPipeline

        constituents = [get_variant(name) for name in config.route_variants]
        return V15RouterPipeline(constituents, lambda constituent: _LLMPipeline(constituent, settings))
    if config.compose_intersect:
        constituents = [get_variant(name) for name in config.compose_intersect]
        return ComposePipeline(constituents, lambda constituent: _LLMPipeline(constituent, settings))
    if config.scout_route:
        from formuloom.scout import ScoutRoutedPipeline

        return ScoutRoutedPipeline(settings, _build_pipeline, client_factory=build_async_client)
    return _LLMPipeline(config, settings)


@app.command("predict")
def predict_cmd(
    task_dir: Path = typer.Argument(  # noqa: B008 - typer's documented pattern; Path isn't ruff's "immutable" allowlist
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Task bundle directory (init.xlsx/complete.xlsx/raw_diff.json/instructions.md).",
    ),
    variant: str = typer.Option("V8", "--variant", help="Recipe: V0/V8 are offline; V15 uses optional providers."),
    output: Path | None = typer.Option(  # noqa: B008 - typer's documented pattern
        None, "--output", "-o", help="Where to write generated.json (default: <run-dir>/generated.json)."
    ),
    best_effort: bool = typer.Option(
        False,
        "--best-effort",
        help="Degrade a failed sheet to all-intermediate instead of hard-failing (provider variants only).",
    ),
    run_dir: Path | None = typer.Option(  # noqa: B008 - typer's documented pattern
        None, "--run-dir", help="Run dir for prompt/response caching (default: runs/<timestamp>-<variant>)."
    ),
) -> None:
    try:
        config = get_variant(variant)
        bundle = TaskBundle.load(task_dir, mode="predict")
        resolved_run_dir = run_dir if run_dir is not None else _default_run_dir(variant)

        failures: dict[str, str] = {}
        if is_offline(variant):
            if variant == "V8":
                result, diagnostics = predict_v8(bundle)
                _atomic_write_json(resolved_run_dir / "weak-supervision.json", diagnostics)
            else:
                result = predict_v0(bundle)
        else:
            pipeline = _build_pipeline(config)
            pipeline_result = asyncio.run(pipeline.predict(bundle, run_dir=resolved_run_dir, best_effort=best_effort))
            result = pipeline_result.diff
            failures = pipeline_result.failures

        out_path = output if output is not None else resolved_run_dir / "generated.json"
        _atomic_write_diff_file(out_path, result)
        typer.echo(f"wrote {out_path}")

        if failures:
            _atomic_write_json(resolved_run_dir / FAILURES_FILE, failures)
            _fail(f"{len(failures)} sheet(s) degraded to all-intermediate (--best-effort): {sorted(failures)}")
    except _COMMAND_ERRORS as exc:
        _fail(str(exc))


def _format_score_line(mode: ScoreMode, detail: TaskScoreDetail, *, primary: bool) -> str:

    tag = "PRIMARY" if primary else "diagnostic"
    return (
        f"  [{mode}] ({tag}) recall={detail.recall:.4f} precision={detail.precision:.4f} "
        f"f1={detail.f1:.4f}  (tp={detail.tp} fp={detail.fp} fn={detail.fn})"
    )


@app.command("score")
def score_cmd(
    task_dir: Path = typer.Argument(  # noqa: B008 - typer's documented pattern
        ..., exists=True, file_okay=False, dir_okay=True, help="Task bundle directory (must contain subset.json)."
    ),
    pred: Path | None = typer.Option(  # noqa: B008 - typer's documented pattern
        None, "--pred", help="Prediction generated.json (default: <task_dir>/generated.json)."
    ),
    mode: str = typer.Option("both", "--mode", help="strict | annotated | both (default: both)."),
) -> None:
    try:
        bundle = TaskBundle.load(task_dir, mode="score")
        assert bundle.golden is not None
        pred_path = pred if pred is not None else task_dir / "generated.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"prediction file not found: {pred_path}")
        prediction = DiffFile.model_validate(json.loads(pred_path.read_text(encoding="utf-8")))

        modes = _resolve_modes(mode)
        typer.echo(f"Task: {bundle.name}")
        for m in modes:
            detail = score_task(bundle.name, prediction, bundle.golden, m)
            typer.echo(_format_score_line(m, detail, primary=(m == "annotated")))
    except _COMMAND_ERRORS as exc:
        _fail(str(exc))


def _micro_f1(details: Mapping[str, TaskScoreDetail]) -> float:
    return aggregate(details).micro.f1


def _repeat_meta(runs: list[dict[str, dict[str, TaskScoreDetail]]]) -> dict[str, Any]:
    f1s = [_micro_f1(run["annotated"]) for run in runs]
    mean_f1 = sum(f1s) / len(f1s)
    variance = sum((x - mean_f1) ** 2 for x in f1s) / len(f1s)
    return {"n": len(runs), "compat_micro_f1_per_run": f1s, "mean_compat_micro_f1": mean_f1, "stddev": variance**0.5}


@app.command("eval")
def eval_cmd(
    data_root: Path = typer.Argument(  # noqa: B008 - typer's documented pattern
        ..., exists=True, file_okay=False, dir_okay=True, help="Directory of task bundle subdirectories."
    ),
    variant: str = typer.Option(..., "--variant", help="Recipe to evaluate; V0 and V8 are offline."),
    repeat: int = typer.Option(
        1, "--repeat", help="Repeat predict+score N times (cached provider runs are not fresh draws)."
    ),
    out: Path | None = typer.Option(  # noqa: B008 - typer's documented pattern
        None, "--out", help="Write eval_results.json here (default: <run-dir>/eval_results.json)."
    ),
    offline_only: bool = typer.Option(
        False, "--offline-only", help="Refuse any LLM call; fails loudly if the variant needs one."
    ),
    run_dir: Path | None = typer.Option(  # noqa: B008 - typer's documented pattern
        None, "--run-dir", help="Run dir for artifacts (default: runs/<timestamp>-<variant>)."
    ),
) -> None:
    try:
        if repeat < 1:
            raise ValueError(f"--repeat must be >= 1, got {repeat}")
        config = get_variant(variant)
        pipeline: _Pipeline | None = None
        if not is_offline(variant):
            if offline_only:
                raise ValueError(
                    f"variant {variant!r} requires LLM calls; --offline-only forbids them (V0 and V8 are offline)"
                )
            pipeline = _build_pipeline(config)

        resolved_run_dir = run_dir if run_dir is not None else _default_run_dir(variant)
        tasks = _discover_tasks(data_root)

        started = time.perf_counter()
        run_records: list[dict[str, dict[str, TaskScoreDetail]]] = []
        total_usage = UsageRecord()
        all_failures: dict[str, list[str]] = {}
        for run_idx in range(repeat):
            compat: dict[str, TaskScoreDetail] = {}
            strict: dict[str, TaskScoreDetail] = {}
            for task_name, task_dir in tasks:
                bundle = TaskBundle.load(task_dir, mode="score")
                assert bundle.golden is not None
                if pipeline is None:
                    prediction = predict_v8(bundle)[0] if variant == "V8" else predict_v0(bundle)
                else:

                    pipeline_result = asyncio.run(pipeline.predict(bundle, run_dir=resolved_run_dir, best_effort=True))
                    prediction = pipeline_result.diff
                    total_usage = total_usage.combined_with(pipeline_result.usage)
                    if pipeline_result.failures:
                        all_failures.setdefault(task_name, []).extend(sorted(pipeline_result.failures))
                compat[task_name] = score_task(task_name, prediction, bundle.golden, "annotated")
                strict[task_name] = score_task(task_name, prediction, bundle.golden, "strict")
                if run_idx == repeat - 1:
                    _atomic_write_json(
                        resolved_run_dir / ERROR_REPORT_SUBDIR / f"{task_name}.json",
                        build_error_report(task_name, prediction, bundle.golden, "annotated"),
                    )
                    _atomic_write_json(
                        resolved_run_dir / ERROR_REPORT_SUBDIR / f"{task_name}.strict.json",
                        build_error_report(task_name, prediction, bundle.golden, "strict"),
                    )
                    _atomic_write_diff_file(
                        resolved_run_dir / PREDICTIONS_SUBDIR / f"{task_name}.generated.json", prediction
                    )
            run_records.append({"annotated": compat, "strict": strict})
        elapsed = time.perf_counter() - started

        final_compat = run_records[-1]["annotated"]
        final_strict = run_records[-1]["strict"]
        extra_meta: dict[str, Any] = {
            "strict": {
                "aggregates": aggregate(final_strict).model_dump(),
                "per_task": {t: d.model_dump() for t, d in final_strict.items()},
            },
            "offline_only": offline_only,
        }
        if repeat > 1:
            extra_meta["repeat"] = _repeat_meta(run_records)
        if all_failures:
            extra_meta["failures"] = all_failures
            _atomic_write_json(resolved_run_dir / FAILURES_FILE, all_failures)

        results = build_eval_results(
            final_compat,
            variant=variant,
            model=config.model,
            prompt_version=config.prompt_version,
            tokens=total_usage.input_tokens + total_usage.output_tokens,
            cost_usd=total_usage.cost_usd,
            wall_clock_seconds=elapsed,
            extra_meta=extra_meta,
        )
        out_path = out if out is not None else resolved_run_dir / "eval_results.json"
        _atomic_write_json(out_path, results)
        typer.echo(f"wrote {out_path}")
        if all_failures:
            typer.echo(
                f"warning: {sum(len(v) for v in all_failures.values())} sheet(s) degraded (--best-effort)", err=True
            )
    except _COMMAND_ERRORS as exc:
        _fail(str(exc))


@app.command("compare")
def compare_cmd(
    run_dirs: list[Path] = typer.Argument(  # noqa: B008 - typer's documented pattern
        ..., exists=True, file_okay=False, dir_okay=True, help="Run directories, each containing an eval_results.json."
    ),
) -> None:
    try:
        if not run_dirs:
            raise ValueError("compare requires at least one run directory")
        header = f"{'run':32s} {'task':22s} {'recall':>8s} {'precision':>9s} {'f1':>8s}"
        typer.echo(header)
        typer.echo("-" * len(header))
        for run_dir in run_dirs:
            results_path = run_dir / "eval_results.json"
            if not results_path.is_file():
                raise FileNotFoundError(f"no eval_results.json in {run_dir}")
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            meta = payload.get("meta", {})
            strict_per_task = (meta.get("strict") or {}).get("per_task") or {}
            for task, scores in payload.items():
                if task == "meta":
                    continue
                typer.echo(
                    f"{run_dir.name:32s} {task:22s} {scores['recall']:8.4f} "
                    f"{scores['precision']:9.4f} {scores['f1']:8.4f}"
                )
                strict = strict_per_task.get(task)
                if strict:
                    typer.echo(
                        f"{'':32s} {'(strict)':22s} {strict['recall']:8.4f} "
                        f"{strict['precision']:9.4f} {strict['f1']:8.4f}"
                    )
            typer.echo(
                f"{run_dir.name:32s} variant={meta.get('variant')} model={meta.get('model')} "
                f"cost_usd={meta.get('cost_usd')} wall_clock_s={meta.get('wall_clock_seconds')}"
            )
            typer.echo("")
    except _COMMAND_ERRORS as exc:
        _fail(str(exc))


if __name__ == "__main__":
    run_cli()
