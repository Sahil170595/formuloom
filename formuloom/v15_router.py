from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from formuloom.assemble import assemble_diff_file
from formuloom.bundle import TaskBundle
from formuloom.compose import Pipeline, PipelineResult
from formuloom.features import SheetFeatures, build_task_features
from formuloom.schema import DiffFile, VariantConfig, cell_row
from formuloom.workbook import load_workbook_data

V15_DEFAULT_VARIANT = "V11"

V15_HIGH_RECALL_VARIANT = "V9"

V15_DENSE_STANDALONE_MIN_CANDIDATES = 1000

V15_DENSE_STANDALONE_MIN_DENSITY = 0.25

V15_WIDE_STANDALONE_MIN_PRECEDENTS = 500
V15_WIDE_STANDALONE_MIN_CANDIDATES = 1000
V15_WIDE_STANDALONE_MAX_DENSITY = 0.04
V15_WIDE_STANDALONE_MIN_CELLS_PER_ROW = 50.0

V15_LOW_DENSITY_OUTPUT_MAX_DENSITY = 0.04

V15_LOW_DENSITY_OUTPUT_MIN_PRECEDENTS = 250

V15_SOURCE_INPUT_MIN_COLOR = 0.95

V15_SOURCE_INPUT_MIN_ROWS = 30
V15_SOURCE_INPUT_MAX_ROWS = 80

V15_EMPTY_REF_MIN_CANDIDATES = 300
V15_EMPTY_REF_MAX_CANDIDATES = 800
V15_EMPTY_REF_MIN_ROWS = 25
V15_EMPTY_REF_MAX_CELLS_PER_ROW = 20.0
V15_EMPTY_REF_MAX_DENSITY = 0.05

V15_EMPTY_SMALL_MAX_CANDIDATES = 200
V15_EMPTY_SMALL_MAX_PRECEDENTS = 20
V15_EMPTY_SMALL_MAX_INPUT_COLOR = 0.50

V15_FUNDING_MIN_MULTI_SKETCH = 0.85
V15_FUNDING_MIN_PRECEDENTS = 500
V15_FUNDING_MIN_CANDIDATES = 100
V15_FUNDING_MAX_CANDIDATES = 500


@dataclass(frozen=True)
class V15SheetDecision:

    source_variant: str
    emptied: bool
    low_density_output_route: bool


def _all_labels(sheet: SheetFeatures) -> str:
    return "\n".join((row.label or "").lower() for row in sheet.rows)


def _cells_per_row(sheet: SheetFeatures) -> float:
    return sheet.n_candidates / max(len(sheet.rows), 1)


def is_v15_low_density_output_route(sheet: SheetFeatures) -> bool:
    return (
        sheet.candidate_density < V15_LOW_DENSITY_OUTPUT_MAX_DENSITY
        and sheet.n_cells_with_cross_sheet_precedents >= V15_LOW_DENSITY_OUTPUT_MIN_PRECEDENTS
        and sheet.n_cells_with_cross_sheet_dependents > 0
    )


def should_v15_route_high_recall(sheet: SheetFeatures) -> bool:
    dense_standalone_output = (
        sheet.n_cells_with_cross_sheet_precedents == 0
        and sheet.n_cells_with_cross_sheet_dependents == 0
        and sheet.n_candidates >= V15_DENSE_STANDALONE_MIN_CANDIDATES
        and sheet.candidate_density >= V15_DENSE_STANDALONE_MIN_DENSITY
    )
    source_statement_input = (
        sheet.input_colored_fraction >= V15_SOURCE_INPUT_MIN_COLOR
        and sheet.n_cells_with_cross_sheet_precedents == 0
        and sheet.n_cells_with_cross_sheet_dependents > 0
        and V15_SOURCE_INPUT_MIN_ROWS <= len(sheet.rows) <= V15_SOURCE_INPUT_MAX_ROWS
    )
    wide_standalone_projection = (
        sheet.n_cells_with_cross_sheet_dependents == 0
        and sheet.n_cells_with_cross_sheet_precedents >= V15_WIDE_STANDALONE_MIN_PRECEDENTS
        and sheet.n_candidates >= V15_WIDE_STANDALONE_MIN_CANDIDATES
        and sheet.candidate_density < V15_WIDE_STANDALONE_MAX_DENSITY
        and _cells_per_row(sheet) >= V15_WIDE_STANDALONE_MIN_CELLS_PER_ROW
    )
    return (
        dense_standalone_output
        or is_v15_low_density_output_route(sheet)
        or source_statement_input
        or wide_standalone_projection
    )


def should_v15_empty_sheet(sheet: SheetFeatures) -> bool:
    standalone_reference_table = (
        V15_EMPTY_REF_MIN_CANDIDATES <= sheet.n_candidates <= V15_EMPTY_REF_MAX_CANDIDATES
        and len(sheet.rows) >= V15_EMPTY_REF_MIN_ROWS
        and _cells_per_row(sheet) <= V15_EMPTY_REF_MAX_CELLS_PER_ROW
        and sheet.candidate_density < V15_EMPTY_REF_MAX_DENSITY
        and sheet.n_cells_with_cross_sheet_precedents > 0
        and sheet.n_cells_with_cross_sheet_dependents == 0
    )
    small_feeder_reference = (
        sheet.n_candidates <= V15_EMPTY_SMALL_MAX_CANDIDATES
        and sheet.n_cells_with_cross_sheet_dependents > sheet.n_cells_with_cross_sheet_precedents
        and sheet.n_cells_with_cross_sheet_precedents <= V15_EMPTY_SMALL_MAX_PRECEDENTS
        and sheet.input_colored_fraction < V15_EMPTY_SMALL_MAX_INPUT_COLOR
    )
    return standalone_reference_table or small_feeder_reference


def prune_v15_funding_schedule(sheet: SheetFeatures, final_cells: set[str]) -> set[str]:
    labels = _all_labels(sheet)
    is_source_use_schedule = (
        sheet.frac_rows_multi_sketch >= V15_FUNDING_MIN_MULTI_SKETCH
        and sheet.n_cells_with_cross_sheet_precedents >= V15_FUNDING_MIN_PRECEDENTS
        and V15_FUNDING_MIN_CANDIDATES <= sheet.n_candidates <= V15_FUNDING_MAX_CANDIDATES
        and "total sources" in labels
        and "total uses" in labels
    )
    if not is_source_use_schedule:
        return final_cells
    keep_rows = {row.row for row in sheet.rows if "total sources" in (row.label or "").lower()}
    return {ref for ref in final_cells if cell_row(ref) in keep_rows}


def prune_v15_margin_rows(sheet: SheetFeatures, final_cells: set[str], *, enabled: bool) -> set[str]:
    if not enabled:
        return final_cells
    drop_rows: set[int] = set()
    for row in sheet.rows:
        label = (row.label or "").lower()
        if not ("margin" in label or "growth" in label) or row.n_dependents_outside_row != 0:
            continue
        if row.candidate_refs and all(sheet.cells[ref].cell_type == "percentage" for ref in row.candidate_refs):
            drop_rows.add(row.row)
    return {ref for ref in final_cells if cell_row(ref) not in drop_rows}


def route_v15_sheet(
    sheet: SheetFeatures,
    *,
    default_cells: set[str],
    high_recall_cells: set[str],
) -> tuple[set[str], V15SheetDecision]:
    low_density_route = is_v15_low_density_output_route(sheet)
    use_high_recall = should_v15_route_high_recall(sheet)
    cells = set(high_recall_cells if use_high_recall else default_cells)
    emptied = should_v15_empty_sheet(sheet)
    if emptied:
        cells = set()
    else:
        cells = prune_v15_funding_schedule(sheet, cells)
        cells = prune_v15_margin_rows(sheet, cells, enabled=low_density_route)
    decision = V15SheetDecision(
        source_variant=V15_HIGH_RECALL_VARIANT if use_high_recall else V15_DEFAULT_VARIANT,
        emptied=emptied,
        low_density_output_route=low_density_route,
    )
    return cells, decision


def merge_v15_predictions(
    sheets: Mapping[str, SheetFeatures],
    *,
    default_diff: DiffFile,
    high_recall_diff: DiffFile,
) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for sheet_name, sheet in sheets.items():
        cells, _decision = route_v15_sheet(
            sheet,
            default_cells=set(default_diff.final_refs(sheet_name)),
            high_recall_cells=set(high_recall_diff.final_refs(sheet_name)),
        )
        merged[sheet_name] = cells
    return merged


def _merge_failures(named_failures: Sequence[tuple[str, Mapping[str, str]]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for variant_name, failures in named_failures:
        for sheet, message in failures.items():
            prefixed = f"{variant_name}: {message}"
            merged[sheet] = f"{merged[sheet]} | {prefixed}" if sheet in merged else prefixed
    return merged


class V15RouterPipeline:

    def __init__(
        self,
        constituents: Sequence[VariantConfig],
        pipeline_factory: Callable[[VariantConfig], Pipeline],
    ) -> None:
        by_name = {variant.name: variant for variant in constituents}
        missing = {V15_DEFAULT_VARIANT, V15_HIGH_RECALL_VARIANT} - set(by_name)
        if missing:
            raise ValueError(f"V15RouterPipeline missing constituent(s): {sorted(missing)}")
        self._default = by_name[V15_DEFAULT_VARIANT]
        self._high_recall = by_name[V15_HIGH_RECALL_VARIANT]
        self._factory = pipeline_factory

    async def predict(self, bundle: TaskBundle, *, run_dir: Path, best_effort: bool) -> PipelineResult:
        default_result = await self._factory(self._default).predict(bundle, run_dir=run_dir, best_effort=best_effort)
        high_recall_result = await self._factory(self._high_recall).predict(
            bundle, run_dir=run_dir, best_effort=best_effort
        )

        complete = load_workbook_data(bundle.complete_path)
        task_features = build_task_features(bundle, complete)
        raw_sheets = {sheet_name: task_features.sheets[sheet_name] for sheet_name in bundle.raw_diff.sheets}
        predicted_final = merge_v15_predictions(
            raw_sheets,
            default_diff=default_result.diff,
            high_recall_diff=high_recall_result.diff,
        )
        diff = assemble_diff_file(bundle.raw_diff, predicted_final)

        usage = default_result.usage.combined_with(high_recall_result.usage)
        failures = _merge_failures(
            [
                (self._default.name, default_result.failures),
                (self._high_recall.name, high_recall_result.failures),
            ]
        )
        return PipelineResult(diff=diff, failures=failures, usage=usage)
