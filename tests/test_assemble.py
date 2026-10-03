from __future__ import annotations

import json
from pathlib import Path

import pytest

from formuloom.assemble import (
    AssembleError,
    assemble_diff_file,
    assemble_sheet_diff,
    build_sheet_universe,
)
from formuloom.constants import GROUP_WEIGHT_FINAL, GROUP_WEIGHT_INTERMEDIATE, SPEC_VERSION, TASK_TOLERANCE
from formuloom.schema import CellEntry, DiffFile

FIXTURES = Path(__file__).parent / "fixtures"


def _diff(payload: dict[str, object]) -> DiffFile:
    return DiffFile.model_validate(payload)


_SIMPLE_RAW = {
    "spec_version": 2,
    "task_tolerance": 0.01,
    "sheets": {
        "S1": {
            "sheet_weight": 0.5,
            "groups": {
                "intermediate": {
                    "weight": 1,
                    "cells": [
                        {"cell": "A1", "cell_type": "currency"},
                        {"cell": "A2", "cell_type": "number"},
                        {"cell": "A3", "cell_type": "text"},
                    ],
                }
            },
        }
    },
}


def test_assemble_sheet_diff_partitions_final_from_intermediate() -> None:
    raw = _diff(_SIMPLE_RAW)
    sheet_diff = assemble_sheet_diff(raw, "S1", {"A2"})
    assert sheet_diff.sheet_weight == 0.5
    assert sheet_diff.groups.final is not None and sheet_diff.groups.intermediate is not None
    assert {c.cell for c in sheet_diff.groups.final.cells} == {"A2"}
    assert {c.cell for c in sheet_diff.groups.intermediate.cells} == {"A1", "A3"}


def test_assemble_sheet_diff_cell_type_passthrough_verbatim() -> None:
    raw = _diff(_SIMPLE_RAW)
    sheet_diff = assemble_sheet_diff(raw, "S1", {"A2"})
    final_types = {c.cell: c.cell_type for c in sheet_diff.groups.final.cells}  # type: ignore[union-attr]
    inter_types = {c.cell: c.cell_type for c in sheet_diff.groups.intermediate.cells}  # type: ignore[union-attr]
    assert final_types == {"A2": "number"}
    assert inter_types == {"A1": "currency", "A3": "text"}


def test_weights_are_06_04_when_finals_exist() -> None:
    raw = _diff(_SIMPLE_RAW)
    sheet_diff = assemble_sheet_diff(raw, "S1", {"A2"})
    assert sheet_diff.groups.intermediate.weight == GROUP_WEIGHT_INTERMEDIATE  # type: ignore[union-attr]
    assert sheet_diff.groups.final.weight == GROUP_WEIGHT_FINAL  # type: ignore[union-attr]


def test_weights_are_10_00_when_final_group_empty() -> None:
    raw = _diff(_SIMPLE_RAW)
    sheet_diff = assemble_sheet_diff(raw, "S1", set())
    assert sheet_diff.groups.intermediate.weight == 1.0  # type: ignore[union-attr]
    assert sheet_diff.groups.final.weight == 0.0  # type: ignore[union-attr]
    assert sheet_diff.groups.final.cells == []  # type: ignore[union-attr]


def test_all_final_sheet_unobserved_edge_still_uses_06_04() -> None:

    raw = _diff(_SIMPLE_RAW)
    sheet_diff = assemble_sheet_diff(raw, "S1", {"A1", "A2", "A3"})
    assert sheet_diff.groups.intermediate.weight == GROUP_WEIGHT_INTERMEDIATE  # type: ignore[union-attr]
    assert sheet_diff.groups.final.weight == GROUP_WEIGHT_FINAL  # type: ignore[union-attr]
    assert sheet_diff.groups.intermediate.cells == []  # type: ignore[union-attr]
    assert {c.cell for c in sheet_diff.groups.final.cells} == {"A1", "A2", "A3"}  # type: ignore[union-attr]


def test_all_empty_sheet_zero_candidates_never_crashes() -> None:
    raw = _diff(
        {
            "spec_version": 2,
            "task_tolerance": 0.01,
            "sheets": {"Empty": {"sheet_weight": 0.0, "groups": {"intermediate": {"weight": 1, "cells": []}}}},
        }
    )
    sheet_diff = assemble_sheet_diff(raw, "Empty", set())
    assert sheet_diff.groups.final.cells == []  # type: ignore[union-attr]
    assert sheet_diff.groups.intermediate.cells == []  # type: ignore[union-attr]
    assert sheet_diff.groups.intermediate.weight == 1.0  # type: ignore[union-attr]


def test_multigroup_raw_diff_every_candidate_appears_exactly_once() -> None:
    raw = DiffFile.model_validate(json.loads((FIXTURES / "multigroup_raw_diff.json").read_text(encoding="utf-8")))
    sheet_diff = assemble_sheet_diff(raw, "Statement", {"B10"})
    all_cells = [c.cell for c in sheet_diff.groups.intermediate.cells] + [  # type: ignore[union-attr]
        c.cell for c in sheet_diff.groups.final.cells  # type: ignore[union-attr]
    ]

    assert sorted(all_cells) == ["B10", "B2", "B3", "C5"]
    assert len(all_cells) == len(set(all_cells))


def test_multigroup_dedup_first_occurrence_wins_type() -> None:

    raw = DiffFile.model_validate(json.loads((FIXTURES / "multigroup_raw_diff.json").read_text(encoding="utf-8")))
    sheet_diff = assemble_sheet_diff(raw, "Statement", set())
    types = {c.cell: c.cell_type for c in sheet_diff.groups.intermediate.cells}  # type: ignore[union-attr]
    assert types["B3"] == "currency"


def test_build_sheet_universe_raw_only_preserves_raw_diff_order() -> None:
    raw = _diff(_SIMPLE_RAW)
    universe = build_sheet_universe(raw, "S1")
    assert [c.cell for c in universe] == ["A1", "A2", "A3"]


def test_build_sheet_universe_appends_extras_sorted_by_row_col_after_raw() -> None:
    raw = _diff(_SIMPLE_RAW)
    extras = [
        CellEntry(cell="B5", cell_type="number"),
        CellEntry(cell="A10", cell_type="text"),
        CellEntry(cell="C1", cell_type="number"),
    ]
    universe = build_sheet_universe(raw, "S1", extras)

    assert [c.cell for c in universe] == ["A1", "A2", "A3", "C1", "B5", "A10"]


def test_build_sheet_universe_drops_extras_already_present_in_raw() -> None:
    raw = _diff(_SIMPLE_RAW)
    extras = [CellEntry(cell="A2", cell_type="number"), CellEntry(cell="B1", cell_type="text")]
    universe = build_sheet_universe(raw, "S1", extras)
    assert [c.cell for c in universe] == ["A1", "A2", "A3", "B1"]


def test_build_sheet_universe_dedupes_duplicate_extras() -> None:
    raw = _diff(_SIMPLE_RAW)
    extras = [CellEntry(cell="B1", cell_type="text"), CellEntry(cell="B1", cell_type="text")]
    universe = build_sheet_universe(raw, "S1", extras)
    assert [c.cell for c in universe] == ["A1", "A2", "A3", "B1"]


def test_assemble_sheet_diff_honors_extra_cells_in_partition() -> None:
    raw = _diff(_SIMPLE_RAW)
    extras = [CellEntry(cell="B1", cell_type="text")]
    sheet_diff = assemble_sheet_diff(raw, "S1", {"B1"}, extra_cells=extras)
    assert {c.cell for c in sheet_diff.groups.final.cells} == {"B1"}  # type: ignore[union-attr]
    assert {c.cell for c in sheet_diff.groups.intermediate.cells} == {"A1", "A2", "A3"}  # type: ignore[union-attr]


def test_assemble_diff_file_unknown_sheet_in_predicted_final_raises() -> None:
    raw = _diff(_SIMPLE_RAW)
    with pytest.raises(AssembleError, match="S1|not.*in raw_diff|unknown sheet"):
        assemble_diff_file(raw, {"NoSuchSheet": {"A1"}})


def test_assemble_sheet_diff_ref_outside_universe_raises() -> None:
    raw = _diff(_SIMPLE_RAW)
    with pytest.raises(AssembleError, match="Z99"):
        assemble_sheet_diff(raw, "S1", {"Z99"})


def test_assemble_diff_file_covers_every_raw_diff_sheet() -> None:
    raw = _diff(_SIMPLE_RAW)
    out = assemble_diff_file(raw, {})
    assert set(out.sheets) == set(raw.sheets)
    assert out.intermediate_refs("S1") == {"A1", "A2", "A3"}
    assert out.final_refs("S1") == set()


def test_assemble_diff_file_sets_spec_version_and_tolerance_from_constants() -> None:
    raw = _diff(_SIMPLE_RAW)
    out = assemble_diff_file(raw, {"S1": {"A2"}})
    assert out.spec_version == SPEC_VERSION == 2
    assert out.task_tolerance == TASK_TOLERANCE == 0.01


def test_assemble_diff_file_round_trips_through_schema() -> None:
    raw = _diff(_SIMPLE_RAW)
    out = assemble_diff_file(raw, {"S1": {"A2"}})
    dumped = out.model_dump()
    reloaded = DiffFile.model_validate(dumped)
    assert reloaded == out


def test_assemble_diff_file_multi_sheet() -> None:
    raw = DiffFile.model_validate(json.loads((FIXTURES / "multigroup_raw_diff.json").read_text(encoding="utf-8")))
    out = assemble_diff_file(raw, {"Statement": {"B10", "B3"}})
    assert out.final_refs("Statement") == {"B10", "B3"}
    assert out.intermediate_refs("Statement") == {"B2", "C5"}

    assert out.intermediate_refs("Inputs") == {"A1", "A2"}
    assert out.final_refs("Inputs") == set()
