from __future__ import annotations

from formuloom.features import CellFeatures, RowFeatures, SheetFeatures
from formuloom.v15_router import (
    prune_v15_funding_schedule,
    prune_v15_margin_rows,
    route_v15_sheet,
    should_v15_empty_sheet,
    should_v15_route_high_recall,
)


def _row(
    row: int,
    *,
    label: str | None = None,
    refs: list[str] | None = None,
    deps_out: int = 0,
) -> RowFeatures:
    candidate_refs = refs or [f"B{row}"]
    return RowFeatures(
        sheet="S",
        row=row,
        label=label,
        section_title=None,
        n_candidates=len(candidate_refs),
        candidate_refs=candidate_refs,
        n_dependents_outside_row=deps_out,
        distinct_source_sheets=0,
        distinct_source_sections=0,
        aggregates_range=False,
        agg_function=None,
        lexicon_hits=[],
        input_colored_fraction=0.0,
        bold_or_bordered=False,
        formula_pattern="=RC[-1]",
        pattern_kind="r1c1",
        in_chart_source=False,
        in_print_area=False,
        in_named_range=False,
        n_distinct_labels_left=1,
        n_distinct_sketches=1,
    )


def _cell(ref: str, *, cell_type: str = "number") -> CellFeatures:
    row = int("".join(ch for ch in ref if ch.isdigit()))
    return CellFeatures(
        sheet="S",
        ref=ref,
        row=row,
        col=1,
        cell_type=cell_type,
        label=None,
        label_ref=None,
        value=1,
        formula="=1",
        sketch=None,
        r1c1=None,
        is_input_colored=False,
        bold_or_bordered=False,
        n_dependents=0,
    )


def _sheet(
    *,
    row_count: int,
    n_candidates: int,
    density: float,
    input_colored_fraction: float = 0.0,
    cross_sheet_precedents: int,
    cross_sheet_dependents: int,
    frac_rows_multi_sketch: float = 0.0,
    rows: list[RowFeatures] | None = None,
) -> SheetFeatures:
    sheet_rows = rows or [_row(i) for i in range(1, row_count + 1)]
    cells = {ref: _cell(ref) for row in sheet_rows for ref in row.candidate_refs}
    return SheetFeatures(
        sheet="S",
        sections=[],
        rows=sheet_rows,
        cells=cells,
        n_candidates=n_candidates,
        used_range="A1:N100",
        max_row=100,
        max_col=14,
        candidate_density=density,
        frac_rows_multi_label=0.0,
        frac_rows_multi_sketch=frac_rows_multi_sketch,
        label_value_alternation=1.0,
        input_colored_fraction=input_colored_fraction,
        zero_formula=False,
        n_cells_with_cross_sheet_precedents=cross_sheet_precedents,
        n_cells_with_cross_sheet_dependents=cross_sheet_dependents,
    )


def test_v15_routes_dense_standalone_output_to_high_recall() -> None:
    sheet = _sheet(row_count=153, n_candidates=1260, density=0.316, cross_sheet_precedents=0, cross_sheet_dependents=0)
    assert should_v15_route_high_recall(sheet) is True


def test_v15_routes_low_density_cross_sheet_output_to_high_recall() -> None:
    sheet = _sheet(
        row_count=41,
        n_candidates=479,
        density=0.034,
        cross_sheet_precedents=303,
        cross_sheet_dependents=144,
    )
    assert should_v15_route_high_recall(sheet) is True


def test_v15_routes_wide_standalone_projection_to_high_recall() -> None:
    sheet = _sheet(row_count=16, n_candidates=1344, density=0.016, cross_sheet_precedents=952, cross_sheet_dependents=0)
    assert should_v15_route_high_recall(sheet) is True


def test_v15_routes_source_statement_input_to_high_recall() -> None:
    sheet = _sheet(
        row_count=52,
        n_candidates=156,
        density=0.013,
        input_colored_fraction=1.0,
        cross_sheet_precedents=0,
        cross_sheet_dependents=99,
    )
    assert should_v15_route_high_recall(sheet) is True


def test_v15_empty_guard_flags_reference_table() -> None:
    sheet = _sheet(row_count=40, n_candidates=488, density=0.027, cross_sheet_precedents=40, cross_sheet_dependents=0)
    assert should_v15_empty_sheet(sheet) is True


def test_v15_empty_guard_flags_small_operational_feeder() -> None:
    sheet = _sheet(
        row_count=30,
        n_candidates=150,
        density=0.208,
        input_colored_fraction=0.36,
        cross_sheet_precedents=5,
        cross_sheet_dependents=47,
    )
    assert should_v15_empty_sheet(sheet) is True


def test_v15_funding_prune_keeps_only_total_sources_rows() -> None:
    rows = [
        _row(10, label="Total Sources", refs=["B10", "C10"]),
        _row(11, label="Total Uses", refs=["B11", "C11"]),
        _row(12, label="Other", refs=["B12", "C12"]),
    ]
    sheet = _sheet(
        row_count=115,
        n_candidates=321,
        density=0.044,
        cross_sheet_precedents=2230,
        cross_sheet_dependents=204,
        frac_rows_multi_sketch=0.90,
        rows=rows,
    )
    assert prune_v15_funding_schedule(sheet, {"B10", "C10", "B11", "C11", "B12"}) == {"B10", "C10"}


def test_v15_margin_prune_drops_percentage_sink_growth_rows() -> None:
    rows = [
        _row(20, label="Growth", refs=["B20", "C20"], deps_out=0),
        _row(21, label="Revenue", refs=["B21", "C21"], deps_out=0),
    ]
    sheet = _sheet(
        row_count=25,
        n_candidates=459,
        density=0.024,
        cross_sheet_precedents=280,
        cross_sheet_dependents=40,
        rows=rows,
    )
    for ref in ("B20", "C20"):
        sheet.cells[ref] = _cell(ref, cell_type="percentage")
    assert prune_v15_margin_rows(sheet, {"B20", "C20", "B21", "C21"}, enabled=True) == {"B21", "C21"}


def test_v15_route_uses_high_recall_then_prunes_empty_sheet() -> None:
    sheet = _sheet(row_count=40, n_candidates=488, density=0.027, cross_sheet_precedents=40, cross_sheet_dependents=0)
    cells, decision = route_v15_sheet(sheet, default_cells={"B1"}, high_recall_cells={"B1", "B2"})
    assert cells == set()
    assert decision.emptied is True
