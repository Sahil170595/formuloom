from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import openpyxl  # type: ignore[import-untyped]
import pytest

from formuloom.bundle import TaskBundle
from formuloom.perturb import (
    ANCHOR_MARGIN_ROWS,
    PerturbError,
    RegionCollisionError,
    UnsupportedFormulaError,
    _below_anchor,
    _classify_and_shift_range_token,
    _extract_sheet_prefix,
    _needs_quoting,
    _quote_sheet_name,
    _rewrite_formula_sheet_refs,
    _rewrite_ref_string_sheet,
    _shift_ref_body,
    add_scratch_block,
    duplicate_region,
    insert_blank_rows,
    load_bundle,
    pad_bottom_rows,
    perturb_suite,
    rename_sheet,
    reorder_sheet_tabs,
    shift_cell_ref,
    write_bundle,
)
from formuloom.schema import DiffFile
from formuloom.score import score_task
from formuloom.workbook import WorkbookPair, value_diff

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SCORE_MODES = ("strict", "annotated")


def _diff_file(sheets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"spec_version": 2, "task_tolerance": 0.01, "sheets": sheets}


def _sheet_diff(weight: float, intermediate: list[str], final: list[str] | None = None) -> dict[str, Any]:
    groups: dict[str, Any] = {
        "intermediate": {
            "weight": 0.6 if final else 1.0,
            "cells": [{"cell": ref, "cell_type": "number"} for ref in intermediate],
        }
    }
    if final is not None:
        groups["final"] = {"weight": 0.4, "cells": [{"cell": ref, "cell_type": "number"} for ref in final]}
    return {"sheet_weight": weight, "groups": groups}


def _write_dir(
    root: Path,
    *,
    init_wb: Any,
    complete_wb: Any,
    raw_diff: dict[str, Any],
    subset: dict[str, Any],
    instructions: str = "Build the model.",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    init_wb.save(root / "init.xlsx")
    complete_wb.save(root / "complete.xlsx")
    (root / "raw_diff.json").write_text(json.dumps(raw_diff), encoding="utf-8")
    (root / "subset.json").write_text(json.dumps(subset), encoding="utf-8")
    (root / "instructions.md").write_text(instructions, encoding="utf-8")
    return root


def _basic_bundle_dir(tmp_path: Path, name: str = "B1") -> Path:

    def _make(b2: int) -> openpyxl.Workbook:
        wb = openpyxl.Workbook()
        inputs = wb.active
        inputs.title = "Inputs"
        inputs["A1"] = "Growth Rate"
        inputs["A2"] = 0.05
        stmt = wb.create_sheet("Statement")
        stmt["A2"] = "Revenue"
        stmt["B2"] = b2
        stmt["A3"] = "COGS"
        stmt["B3"] = 40
        stmt["A4"] = "Gross Profit"
        stmt["B4"] = "=B2-B3"
        stmt["A5"] = "Adjusted"
        stmt["B5"] = "=Inputs!A2*B4"
        return wb

    raw_diff = _diff_file({"Statement": _sheet_diff(1.0, ["B2"])})
    subset = _diff_file({"Statement": _sheet_diff(1.0, ["B2"], ["B4"])})
    return _write_dir(tmp_path / name, init_wb=_make(100), complete_wb=_make(120), raw_diff=raw_diff, subset=subset)


def _rename_bundle_dir(tmp_path: Path, name: str = "RN1") -> Path:

    def _make() -> openpyxl.Workbook:
        wb = openpyxl.Workbook()
        old = wb.active
        old.title = "Old Name"
        old["A1"] = 10
        old["A2"] = "=A1+1"
        old["A3"] = "='Old Name'!A1+2"
        ref = wb.create_sheet("Ref")
        ref["A1"] = "='Old Name'!A1+1"
        ref["A2"] = '=IF(A1>0,"Old Name is great",0)'
        from openpyxl.workbook.defined_name import DefinedName  # type: ignore[import-untyped]

        wb.defined_names.add(DefinedName("MyRef", attr_text="'Old Name'!$A$1"))
        return wb

    raw_diff = _diff_file({"Old Name": _sheet_diff(1.0, ["A1"])})
    subset = _diff_file({"Old Name": _sheet_diff(1.0, ["A1"], ["A3"])})
    return _write_dir(tmp_path / name, init_wb=_make(), complete_wb=_make(), raw_diff=raw_diff, subset=subset)


def _insert_bundle_dir(tmp_path: Path, name: str = "IB1") -> Path:

    def _make() -> openpyxl.Workbook:
        wb = openpyxl.Workbook()
        s1 = wb.active
        s1.title = "S1"
        s1["A1"] = "label"
        s1["A2"] = 10
        s1["A3"] = 20
        s1["A4"] = "=A2+A3"
        s1["A5"] = "=SUM(A2:A4)"
        s1["A6"] = "=$A$3*2"
        s2 = wb.create_sheet("S2")
        s2["A1"] = "=S1!A4+1"
        s2["A2"] = "='S1'!A5*2"
        s2["A3"] = "=A1+1"
        return wb

    raw_diff = _diff_file({"S1": _sheet_diff(1.0, ["A2", "A3"])})
    subset = _diff_file({"S1": _sheet_diff(1.0, ["A2"], ["A3"])})
    return _write_dir(tmp_path / name, init_wb=_make(), complete_wb=_make(), raw_diff=raw_diff, subset=subset)


def _score_diff_file(path: Path) -> DiffFile:
    return DiffFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _assert_self_score_is_perfect(diff: DiffFile, task: str = "t") -> None:
    for mode in SCORE_MODES:
        detail = score_task(task, diff, diff, mode)  # type: ignore[arg-type]
        assert detail.precision == 1.0
        assert detail.recall == 1.0
        assert detail.f1 == 1.0


def _assert_acceptance_ab(out_dir: Path) -> DiffFile:
    predict_bundle = TaskBundle.load(out_dir, "predict")
    assert predict_bundle.golden is None
    score_bundle = TaskBundle.load(out_dir, "score")
    assert score_bundle.golden is not None
    _assert_self_score_is_perfect(score_bundle.golden)
    return score_bundle.golden


def test_needs_quoting() -> None:
    assert _needs_quoting("Statement") is False
    assert _needs_quoting("Old Name") is True
    assert _needs_quoting("Input-OpEx") is True
    assert _needs_quoting("_Sheet1") is False


def test_quote_sheet_name() -> None:
    assert _quote_sheet_name("Statement") == "Statement"
    assert _quote_sheet_name("Old Name") == "'Old Name'"
    assert _quote_sheet_name("O'Brien") == "'O''Brien'"


def test_extract_sheet_prefix_unquoted() -> None:
    assert _extract_sheet_prefix("OldName!A1") == ("OldName", len("OldName!"))


def test_extract_sheet_prefix_quoted() -> None:
    assert _extract_sheet_prefix("'Old Name'!A1") == ("Old Name", len("'Old Name'!"))


def test_extract_sheet_prefix_none() -> None:
    assert _extract_sheet_prefix("A1") == (None, 0)
    assert _extract_sheet_prefix("SUM(") == (None, 0)


def test_rewrite_formula_sheet_refs_unquoted() -> None:
    assert _rewrite_formula_sheet_refs("=OldName!A1+1", "OldName", "NewName") == "=NewName!A1+1"


def test_rewrite_formula_sheet_refs_quoted() -> None:
    assert _rewrite_formula_sheet_refs("='Old Name'!A1+1", "Old Name", "New Name") == "='New Name'!A1+1"


def test_rewrite_formula_sheet_refs_quotes_new_name_when_needed() -> None:
    assert _rewrite_formula_sheet_refs("=OldName!A1", "OldName", "New Name") == "='New Name'!A1"


def test_rewrite_formula_sheet_refs_unquotes_when_not_needed() -> None:
    assert _rewrite_formula_sheet_refs("='Old Name'!A1", "Old Name", "NewName") == "=NewName!A1"


def test_rewrite_formula_sheet_refs_range() -> None:
    assert _rewrite_formula_sheet_refs("=SUM(OldName!A1:B10)", "OldName", "NewName") == "=SUM(NewName!A1:B10)"


def test_rewrite_formula_sheet_refs_string_literal_untouched() -> None:
    f = '=IF(A1>0,"OldName is great",0)'
    assert _rewrite_formula_sheet_refs(f, "OldName", "NewName") == f


def test_rewrite_formula_sheet_refs_prefix_boundary_not_matched() -> None:
    assert _rewrite_formula_sheet_refs("=OldNameX!A1", "OldName", "NewName") == "=OldNameX!A1"


def test_rewrite_formula_sheet_refs_noop_when_absent() -> None:
    assert _rewrite_formula_sheet_refs("=A1+1", "OldName", "NewName") == "=A1+1"


def test_rewrite_ref_string_sheet_defined_name_target() -> None:
    assert _rewrite_ref_string_sheet("'Old Name'!$A$1", "Old Name", "New Name") == "'New Name'!$A$1"


def test_shift_cell_ref_below_threshold_unchanged() -> None:
    assert shift_cell_ref("B2", 5, 3) == "B2"


def test_shift_cell_ref_at_threshold_shifts() -> None:
    assert shift_cell_ref("B5", 5, 3) == "B8"


def test_shift_cell_ref_above_threshold_shifts() -> None:
    assert shift_cell_ref("B10", 5, 3) == "B13"


def test_shift_cell_ref_inverse_round_trips() -> None:
    forward = shift_cell_ref("B10", 5, 3)
    assert shift_cell_ref(forward, 5 + 3, -3) == "B10"
    unshifted = shift_cell_ref("B2", 5, 3)
    assert shift_cell_ref(unshifted, 5 + 3, -3) == "B2"


def test_shift_ref_body_single_cell_absolute_relative_mixed() -> None:
    assert _shift_ref_body("A5", 3, 2) == "A7"
    assert _shift_ref_body("$A$5", 3, 2) == "$A$7"
    assert _shift_ref_body("A$5", 3, 2) == "A$7"
    assert _shift_ref_body("$A5", 3, 2) == "$A7"
    assert _shift_ref_body("A2", 3, 2) == "A2"


def test_shift_ref_body_range() -> None:
    assert _shift_ref_body("A5:B10", 3, 2) == "A7:B12"
    assert _shift_ref_body("A1:B2", 3, 2) == "A1:B2"


def test_shift_ref_body_whole_column_unaffected() -> None:
    assert _shift_ref_body("A:A", 3, 2) == "A:A"
    assert _shift_ref_body("$A:$C", 3, 2) == "$A:$C"


def test_shift_ref_body_whole_row_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017 - internal marker, checked via classify below
        _shift_ref_body("3:3", 3, 2)


def test_shift_ref_body_bare_identifier_is_not_a_ref() -> None:
    from formuloom.perturb import _NotARef

    with pytest.raises(_NotARef):
        _shift_ref_body("TaxRate", 3, 2)


def test_classify_and_shift_range_token_same_sheet_implicit() -> None:
    new_val, offender = _classify_and_shift_range_token("A5", "S1", "S1", 3, 2)
    assert new_val == "A7" and offender is False


def test_classify_and_shift_range_token_wrong_sheet_untouched() -> None:
    new_val, offender = _classify_and_shift_range_token("A5", "S1", "S2", 3, 2)
    assert new_val == "A5" and offender is False


def test_classify_and_shift_range_token_cross_sheet_qualified() -> None:
    new_val, offender = _classify_and_shift_range_token("S1!A5", "S1", "S2", 3, 2)
    assert new_val == "S1!A7" and offender is False


def test_classify_and_shift_range_token_wrong_target_untouched() -> None:
    new_val, offender = _classify_and_shift_range_token("S3!A5", "S1", "S2", 3, 2)
    assert new_val == "S3!A5" and offender is False


def test_classify_and_shift_range_token_whole_row_is_offender() -> None:
    _, offender = _classify_and_shift_range_token("3:3", "S1", "S1", 3, 2)
    assert offender is True


def test_classify_and_shift_range_token_structured_ref_is_offender() -> None:
    _, offender = _classify_and_shift_range_token("Table1[Col]", "S1", "S1", 3, 2)
    assert offender is True


def test_classify_and_shift_range_token_defined_name_not_offender() -> None:
    new_val, offender = _classify_and_shift_range_token("TaxRate", "S1", "S1", 3, 2)
    assert new_val == "TaxRate" and offender is False


def test_load_bundle_reads_subset_even_though_it_is_a_perturbation_tool(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    assert bundle.subset is not None
    assert "Statement" in bundle.raw_diff["sheets"]
    assert bundle.complete_wb["Statement"]["B2"].value == 120
    assert bundle.complete_values_wb["Statement"]["B3"].value == 40


def test_load_bundle_missing_file_raises(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    (root / "raw_diff.json").unlink()
    from formuloom.bundle import MissingBundleFileError

    with pytest.raises(MissingBundleFileError):
        load_bundle(root)


def test_write_bundle_round_trips(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path, "SRC"))
    out = tmp_path / "OUT"
    write_bundle(bundle, out)
    for f in ("init.xlsx", "complete.xlsx", "raw_diff.json", "subset.json", "instructions.md"):
        assert (out / f).is_file()
    reloaded = TaskBundle.load(out, "score")
    assert reloaded.golden is not None


def test_write_bundle_is_atomic_no_leftover_tmp_files(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path, "SRC2"))
    out = tmp_path / "OUT2"
    write_bundle(bundle, out)
    leftovers = [p for p in out.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_add_scratch_block_writes_junk_and_leaves_json_untouched(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    raw_diff_before = json.loads((root / "raw_diff.json").read_text(encoding="utf-8"))
    subset_before = json.loads((root / "subset.json").read_text(encoding="utf-8"))

    bundle = load_bundle(root)
    add_scratch_block(bundle, "Statement", "A50")
    assert bundle.raw_diff == raw_diff_before
    assert bundle.subset == subset_before

    out = tmp_path / "OUT"
    write_bundle(bundle, out)
    reloaded_complete = openpyxl.load_workbook(out / "complete.xlsx")
    ws = reloaded_complete["Statement"]
    assert ws["A50"].value == 101
    assert str(ws["A52"].value).startswith("=SUM(")

    reloaded_init = openpyxl.load_workbook(out / "init.xlsx")
    assert reloaded_init["Statement"]["A50"].value is None


def test_add_scratch_block_collision_raises(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path))
    with pytest.raises(RegionCollisionError):
        add_scratch_block(bundle, "Statement", "A2")


def test_add_scratch_block_acceptance_and_delta_is_exactly_the_block(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    add_scratch_block(bundle, "Statement", "A50")
    out = tmp_path / "OUT"
    write_bundle(bundle, out)

    _assert_acceptance_ab(out)

    orig_pair = WorkbookPair.load(root / "init.xlsx", root / "complete.xlsx", cache_dir=tmp_path / "c1")
    pert_pair = WorkbookPair.load(out / "init.xlsx", out / "complete.xlsx", cache_dir=tmp_path / "c2")
    orig_diff = value_diff(orig_pair)["Statement"]
    pert_diff = value_diff(pert_pair)["Statement"]

    delta = pert_diff - orig_diff

    assert delta == {"A50", "B50", "C50", "A51", "B51", "C51"}
    assert orig_diff - pert_diff == set()


def test_duplicate_region_copies_values_not_formulas(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path))
    duplicate_region(bundle, "Statement", "A2:B3", "A60")
    ws = bundle.complete_wb["Statement"]
    assert ws["A60"].value == "Revenue"
    assert ws["B60"].value == 120
    assert ws["A61"].value == "COGS"
    assert ws["B61"].value == 40

    iws = bundle.init_wb["Statement"]
    assert iws["B60"].value == 120


def test_duplicate_region_of_a_formula_cell_copies_its_cached_value(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path))
    duplicate_region(bundle, "Statement", "B4:B4", "D70")
    assert bundle.complete_wb["Statement"]["D70"].value is None


def test_duplicate_region_collision_raises(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path))
    with pytest.raises(RegionCollisionError):
        duplicate_region(bundle, "Statement", "A2:B3", "A3")


def test_duplicate_region_acceptance_no_delta_needed(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    duplicate_region(bundle, "Statement", "A2:B3", "A60")
    out = tmp_path / "OUT"
    write_bundle(bundle, out)

    _assert_acceptance_ab(out)

    orig_pair = WorkbookPair.load(root / "init.xlsx", root / "complete.xlsx", cache_dir=tmp_path / "c1")
    pert_pair = WorkbookPair.load(out / "init.xlsx", out / "complete.xlsx", cache_dir=tmp_path / "c2")
    assert value_diff(pert_pair) == value_diff(orig_pair)


def test_reorder_sheet_tabs_reverses_order(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    before = list(bundle.complete_wb.sheetnames)
    reorder_sheet_tabs(bundle)
    after = list(bundle.complete_wb.sheetnames)
    assert after == list(reversed(before))
    assert list(bundle.init_wb.sheetnames) == list(reversed(before))


def test_reorder_sheet_tabs_acceptance_no_content_or_json_change(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    raw_diff_before = json.loads((root / "raw_diff.json").read_text(encoding="utf-8"))
    bundle = load_bundle(root)
    reorder_sheet_tabs(bundle)
    assert bundle.raw_diff == raw_diff_before
    out = tmp_path / "OUT"
    write_bundle(bundle, out)

    _assert_acceptance_ab(out)
    orig_pair = WorkbookPair.load(root / "init.xlsx", root / "complete.xlsx", cache_dir=tmp_path / "c1")
    pert_pair = WorkbookPair.load(out / "init.xlsx", out / "complete.xlsx", cache_dir=tmp_path / "c2")
    assert value_diff(pert_pair) == value_diff(orig_pair)


def test_pad_bottom_rows_grows_row_dimensions_only(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path))
    bottom_before = bundle.complete_wb["Statement"].max_row
    pad_bottom_rows(bundle, "Statement", 4)
    out_dir = tmp_path / "OUT"
    write_bundle(bundle, out_dir)
    reloaded = openpyxl.load_workbook(out_dir / "complete.xlsx")
    ws = reloaded["Statement"]
    assert (bottom_before + 4) in ws.row_dimensions

    for r in range(bottom_before + 1, bottom_before + 5):
        assert ws.cell(row=r, column=1).value is None


def test_pad_bottom_rows_rejects_non_positive_n(tmp_path: Path) -> None:
    bundle = load_bundle(_basic_bundle_dir(tmp_path))
    with pytest.raises(PerturbError):
        pad_bottom_rows(bundle, "Statement", 0)


def test_pad_bottom_rows_acceptance_no_delta(tmp_path: Path) -> None:
    root = _basic_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    pad_bottom_rows(bundle, "Statement", 5)
    out = tmp_path / "OUT"
    write_bundle(bundle, out)

    _assert_acceptance_ab(out)
    orig_pair = WorkbookPair.load(root / "init.xlsx", root / "complete.xlsx", cache_dir=tmp_path / "c1")
    pert_pair = WorkbookPair.load(out / "init.xlsx", out / "complete.xlsx", cache_dir=tmp_path / "c2")
    assert value_diff(pert_pair) == value_diff(orig_pair)


def test_rename_sheet_rewrites_unquoted_and_quoted_and_skips_string_literal(tmp_path: Path) -> None:
    root = _rename_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    rename_sheet(bundle, "Old Name", "New Name")

    ref = bundle.complete_wb["Ref"]
    assert ref["A1"].value == "='New Name'!A1+1"
    assert ref["A2"].value == '=IF(A1>0,"Old Name is great",0)'

    renamed = bundle.complete_wb["New Name"]
    assert renamed["A2"].value == "=A1+1"
    assert renamed["A3"].value == "='New Name'!A1+2"


def test_rename_sheet_rewrites_defined_names(tmp_path: Path) -> None:
    bundle = load_bundle(_rename_bundle_dir(tmp_path))
    rename_sheet(bundle, "Old Name", "New Name")
    assert bundle.complete_wb.defined_names["MyRef"].value == "'New Name'!$A$1"


def test_rename_sheet_renames_json_sheet_keys(tmp_path: Path) -> None:
    bundle = load_bundle(_rename_bundle_dir(tmp_path))
    rename_sheet(bundle, "Old Name", "New Name")
    assert "New Name" in bundle.raw_diff["sheets"]
    assert "Old Name" not in bundle.raw_diff["sheets"]
    assert "New Name" in bundle.subset["sheets"]  # type: ignore[index]


def test_rename_sheet_missing_old_raises(tmp_path: Path) -> None:
    bundle = load_bundle(_rename_bundle_dir(tmp_path))
    with pytest.raises(PerturbError):
        rename_sheet(bundle, "Nope", "New Name")


def test_rename_sheet_collision_raises(tmp_path: Path) -> None:
    bundle = load_bundle(_rename_bundle_dir(tmp_path))
    with pytest.raises(PerturbError):
        rename_sheet(bundle, "Old Name", "Ref")


def test_rename_sheet_acceptance_round_trip(tmp_path: Path) -> None:
    root = _rename_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    rename_sheet(bundle, "Old Name", "New Name")
    out = tmp_path / "OUT"
    write_bundle(bundle, out)

    golden = _assert_acceptance_ab(out)
    assert set(golden.sheets) == {"New Name"}

    orig_pair = WorkbookPair.load(root / "init.xlsx", root / "complete.xlsx", cache_dir=tmp_path / "c1")
    pert_pair = WorkbookPair.load(out / "init.xlsx", out / "complete.xlsx", cache_dir=tmp_path / "c2")
    orig_diff = value_diff(orig_pair)
    pert_diff = value_diff(pert_pair)

    remapped = {("New Name" if s == "Old Name" else s): v for s, v in orig_diff.items()}
    assert pert_diff == remapped


def test_insert_blank_rows_rejects_non_positive_args(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    with pytest.raises(PerturbError):
        insert_blank_rows(bundle, "S1", 3, 0)
    with pytest.raises(PerturbError):
        insert_blank_rows(bundle, "S1", 0, 2)


def test_insert_blank_rows_missing_sheet_raises(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    with pytest.raises(PerturbError):
        insert_blank_rows(bundle, "Nope", 3, 2)


def test_insert_blank_rows_shifts_same_sheet_refs(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    insert_blank_rows(bundle, "S1", 3, 2)
    ws = bundle.complete_wb["S1"]

    assert ws["A1"].value == "label"
    assert ws["A2"].value == 10

    assert ws["A5"].value == 20
    assert ws["A6"].value == "=A2+A5"

    assert ws["A7"].value == "=SUM(A2:A6)"

    assert ws["A8"].value == "=$A$5*2"


def test_insert_blank_rows_shifts_cross_sheet_refs_quoted_and_unquoted(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    insert_blank_rows(bundle, "S1", 3, 2)
    s2 = bundle.complete_wb["S2"]
    assert s2["A1"].value == "=S1!A6+1"
    assert s2["A2"].value == "='S1'!A7*2"
    assert s2["A3"].value == "=A1+1"


def test_insert_blank_rows_applies_to_both_workbooks(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    insert_blank_rows(bundle, "S1", 3, 2)
    assert bundle.init_wb["S1"]["A5"].value == 20
    assert bundle.init_wb["S2"]["A1"].value == "=S1!A6+1"


def test_insert_blank_rows_remaps_raw_diff_and_subset(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    insert_blank_rows(bundle, "S1", 3, 2)
    inter = {c["cell"] for c in bundle.raw_diff["sheets"]["S1"]["groups"]["intermediate"]["cells"]}
    assert inter == {"A2", "A5"}
    gold_final = {c["cell"] for c in bundle.subset["sheets"]["S1"]["groups"]["final"]["cells"]}  # type: ignore[index]
    assert gold_final == {"A5"}


def test_insert_blank_rows_unsupported_whole_row_range_raises_and_bundle_untouched(tmp_path: Path) -> None:
    root = _insert_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    bundle.complete_wb["S1"]["A7"] = "=SUM(3:3)"
    with pytest.raises(UnsupportedFormulaError) as excinfo:
        insert_blank_rows(bundle, "S1", 3, 2)
    assert any(ref == "A7" for _, ref, _ in excinfo.value.offenders)

    assert bundle.complete_wb["S1"]["A7"].value == "=SUM(3:3)"
    assert bundle.complete_wb["S1"]["A4"].value == "=A2+A3"


def test_insert_blank_rows_unsupported_structured_ref_raises(tmp_path: Path) -> None:
    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    bundle.complete_wb["S1"]["A7"] = "=SUM(Table1[Col])"
    with pytest.raises(UnsupportedFormulaError) as excinfo:
        insert_blank_rows(bundle, "S1", 3, 2)
    assert any(ref == "A7" for _, ref, _ in excinfo.value.offenders)


def test_insert_blank_rows_unsupported_array_formula_raises(tmp_path: Path) -> None:
    from openpyxl.worksheet.formula import ArrayFormula  # type: ignore[import-untyped]

    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    bundle.complete_wb["S1"]["A7"] = ArrayFormula(ref="A7", text="=SUM(A2:A4*B2:B4)")
    with pytest.raises(UnsupportedFormulaError) as excinfo:
        insert_blank_rows(bundle, "S1", 3, 2)
    assert any(ref == "A7" for _, ref, _ in excinfo.value.offenders)


def test_insert_blank_rows_leaves_unrelated_defined_names_alone(tmp_path: Path) -> None:
    from openpyxl.workbook.defined_name import DefinedName

    bundle = load_bundle(_insert_bundle_dir(tmp_path))
    bundle.complete_wb.defined_names.add(DefinedName("Foo", attr_text="S1!$A$5"))
    insert_blank_rows(bundle, "S1", 3, 2)
    assert bundle.complete_wb.defined_names["Foo"].value == "S1!$A$5"


def test_insert_blank_rows_acceptance_round_trip_with_inverse_remap(tmp_path: Path) -> None:
    root = _insert_bundle_dir(tmp_path)
    bundle = load_bundle(root)
    at_row, n = 3, 2
    insert_blank_rows(bundle, "S1", at_row, n)
    out = tmp_path / "OUT"
    write_bundle(bundle, out)

    golden = _assert_acceptance_ab(out)

    original_golden = _score_diff_file(root / "subset.json")
    remapped_back = {
        ref: shift_cell_ref(ref, at_row + n, -n) for ref in golden.final_refs("S1") | golden.intermediate_refs("S1")
    }
    perturbed_finals_back = {remapped_back[r] for r in golden.final_refs("S1")}
    perturbed_inters_back = {remapped_back[r] for r in golden.intermediate_refs("S1")}
    assert perturbed_finals_back == original_golden.final_refs("S1")
    assert perturbed_inters_back == original_golden.intermediate_refs("S1")

    orig_pair = WorkbookPair.load(root / "init.xlsx", root / "complete.xlsx", cache_dir=tmp_path / "c1")
    pert_pair = WorkbookPair.load(out / "init.xlsx", out / "complete.xlsx", cache_dir=tmp_path / "c2")
    pert_s1_back = {shift_cell_ref(ref, at_row + n, -n) for ref in value_diff(pert_pair)["S1"]}
    assert pert_s1_back == value_diff(orig_pair)["S1"]
    assert value_diff(pert_pair)["S2"] == value_diff(orig_pair)["S2"]


def test_perturb_suite_all_ok_on_clean_synthetic_bundle(tmp_path: Path) -> None:
    root = _insert_bundle_dir(tmp_path, "SUITE")
    out_root = tmp_path / "SUITE_OUT"
    manifest = perturb_suite(root, out_root)

    assert manifest["perturbations"].keys() == {
        "add_scratch_block",
        "duplicate_region",
        "reorder_sheet_tabs",
        "pad_bottom_rows",
        "rename_sheet",
        "insert_blank_rows",
    }
    for name, entry in manifest["perturbations"].items():
        assert entry["status"] == "ok", f"{name}: {entry}"
        out_dir = Path(entry["out_dir"])
        assert TaskBundle.load(out_dir, "score").golden is not None


def test_perturb_suite_records_unsupported_without_crashing_other_jobs(tmp_path: Path) -> None:
    root = _insert_bundle_dir(tmp_path, "SUITE2")

    wb = openpyxl.load_workbook(root / "complete.xlsx")
    wb["S1"]["A7"] = "=SUM(3:3)"
    wb.save(root / "complete.xlsx")
    wb2 = openpyxl.load_workbook(root / "init.xlsx")
    wb2["S1"]["A7"] = "=SUM(3:3)"
    wb2.save(root / "init.xlsx")

    manifest = perturb_suite(root, tmp_path / "SUITE2_OUT")
    assert manifest["perturbations"]["insert_blank_rows"]["status"] == "unsupported"
    assert manifest["perturbations"]["insert_blank_rows"]["offenders"]

    for name in ("add_scratch_block", "duplicate_region", "reorder_sheet_tabs", "pad_bottom_rows", "rename_sheet"):
        assert manifest["perturbations"][name]["status"] == "ok"


def test_below_anchor_helper() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A5"] = "x"
    assert _below_anchor(ws, ANCHOR_MARGIN_ROWS) == f"A{5 + ANCHOR_MARGIN_ROWS}"


REAL_TASKS = ["synthetic-budget", "synthetic-summary"]
_REQUIRES_DATA = pytest.mark.skipif(not DATA_ROOT.exists(), reason="data/ not present")
