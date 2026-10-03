from __future__ import annotations

from formuloom.cli import _apply_v11_precision_guard, _should_v11_adjudicate
from formuloom.features import SheetFeatures


def _sheet(
    *,
    row_count: int,
    n_candidates: int,
    density: float,
    input_colored_fraction: float = 0.0,
    cross_sheet_precedents: int,
    cross_sheet_dependents: int,
) -> SheetFeatures:
    return SheetFeatures(
        sheet="S",
        sections=[],
        rows=[object()] * row_count,  # type: ignore[list-item]
        cells={},
        n_candidates=n_candidates,
        used_range="A1:N100",
        max_row=100,
        max_col=14,
        candidate_density=density,
        frac_rows_multi_label=0.0,
        frac_rows_multi_sketch=0.0,
        label_value_alternation=1.0,
        input_colored_fraction=input_colored_fraction,
        zero_formula=False,
        n_cells_with_cross_sheet_precedents=cross_sheet_precedents,
        n_cells_with_cross_sheet_dependents=cross_sheet_dependents,
    )


def test_v11_guard_drops_large_granular_support_sheet() -> None:
    sheet = _sheet(
        row_count=540,
        n_candidates=2684,
        density=0.299,
        cross_sheet_precedents=0,
        cross_sheet_dependents=691,
    )
    assert _apply_v11_precision_guard(sheet, {"J8", "K8"}) == set()


def test_v11_guard_drops_dense_source_scenario_sheet() -> None:
    sheet = _sheet(row_count=12, n_candidates=1005, density=0.449, cross_sheet_precedents=0, cross_sheet_dependents=172)
    assert _apply_v11_precision_guard(sheet, {"D24", "E24"}) == set()


def test_v11_guard_keeps_small_source_statement_input_sheet() -> None:
    sheet = _sheet(row_count=52, n_candidates=156, density=0.013, cross_sheet_precedents=0, cross_sheet_dependents=99)
    assert _apply_v11_precision_guard(sheet, {"F10", "G10"}) == {"F10", "G10"}


def test_v11_guard_keeps_standalone_large_output_sheet() -> None:
    sheet = _sheet(row_count=153, n_candidates=1260, density=0.316, cross_sheet_precedents=0, cross_sheet_dependents=0)
    assert _apply_v11_precision_guard(sheet, {"K14", "L14"}) == {"K14", "L14"}


def test_v11_adjudication_gate_skips_empty_proposal() -> None:
    sheet = _sheet(row_count=16, n_candidates=24, density=0.100, cross_sheet_precedents=103, cross_sheet_dependents=0)
    assert _should_v11_adjudicate(sheet, set()) is False


def test_v11_adjudication_gate_skips_input_source_feeder() -> None:
    sheet = _sheet(
        row_count=52,
        n_candidates=156,
        density=0.013,
        input_colored_fraction=1.0,
        cross_sheet_precedents=0,
        cross_sheet_dependents=99,
    )
    assert _should_v11_adjudicate(sheet, {"F10", "G10"}) is False


def test_v11_adjudication_gate_skips_non_dense_feeder_statement() -> None:
    sheet = _sheet(
        row_count=41,
        n_candidates=479,
        density=0.034,
        cross_sheet_precedents=303,
        cross_sheet_dependents=144,
    )
    assert _should_v11_adjudicate(sheet, {f"F{row}" for row in range(1, 145)}) is False


def test_v11_adjudication_gate_reviews_standalone_sink_sheet() -> None:
    sheet = _sheet(row_count=153, n_candidates=1260, density=0.316, cross_sheet_precedents=0, cross_sheet_dependents=0)
    assert _should_v11_adjudicate(sheet, {"K14", "L14"}) is True


def test_v11_adjudication_gate_reviews_wide_projection_proposal() -> None:
    sheet = _sheet(
        row_count=26,
        n_candidates=2181,
        density=0.026,
        cross_sheet_precedents=1548,
        cross_sheet_dependents=85,
    )
    assert _should_v11_adjudicate(sheet, {f"A{idx}" for idx in range(500)}) is True


def test_v11_adjudication_gate_reviews_dense_statement_with_precedents() -> None:
    sheet = _sheet(
        row_count=70,
        n_candidates=345,
        density=0.250,
        input_colored_fraction=0.284,
        cross_sheet_precedents=127,
        cross_sheet_dependents=12,
    )
    assert _should_v11_adjudicate(sheet, {"F10", "G10"}) is True


def test_v11_adjudication_gate_skips_high_input_color_dense_sheet() -> None:
    sheet = _sheet(
        row_count=70,
        n_candidates=345,
        density=0.250,
        input_colored_fraction=0.95,
        cross_sheet_precedents=127,
        cross_sheet_dependents=12,
    )
    assert _should_v11_adjudicate(sheet, {"F10", "G10"}) is False
