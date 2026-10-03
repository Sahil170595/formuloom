from __future__ import annotations

import json
from pathlib import Path

import pytest

from formuloom.schema import DiffFile
from formuloom.score import (
    build_error_report,
    build_eval_results,
    precision_recall_f1,
    score_task,
    write_error_report,
    write_eval_results,
)

DATA = Path(__file__).parent.parent / "data"
FIXTURES = Path(__file__).parent / "fixtures"
SCORE_FX = FIXTURES / "score"
TASKS = ["synthetic-statement", "synthetic-budget", "synthetic-rollup", "synthetic-projection", "synthetic-summary"]

TWO_THIRDS = 2.0 / 3.0
ONE_THIRD = 1.0 / 3.0

requires_data = pytest.mark.skipif(not DATA.is_dir(), reason="data/ bundles not present")


def _diff(path: Path) -> DiffFile:
    return DiffFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_prf_empty_task_is_perfect() -> None:
    assert precision_recall_f1(0, 0, 0) == (1.0, 1.0, 1.0)


def test_prf_no_predictions_but_golden_finals() -> None:

    assert precision_recall_f1(0, 0, 5) == (0.0, 0.0, 0.0)


def test_prf_phantom_finals_score_zero() -> None:

    assert precision_recall_f1(0, 5, 0) == (0.0, 0.0, 0.0)


def test_prf_standard() -> None:
    p, r, f1 = precision_recall_f1(tp=1, fp=1, fn=1)
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_combined_divergence() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")

    strict = score_task("combined", pred, golden, "strict")

    assert (strict.tp, strict.fp, strict.fn) == (1, 2, 2)
    assert strict.precision == pytest.approx(ONE_THIRD)
    assert strict.recall == pytest.approx(ONE_THIRD)
    assert strict.f1 == pytest.approx(ONE_THIRD)

    compat = score_task("combined", pred, golden, "annotated")

    assert (compat.tp, compat.fp, compat.fn) == (1, 1, 1)
    assert compat.precision == pytest.approx(0.5)
    assert compat.recall == pytest.approx(0.5)
    assert compat.f1 == pytest.approx(0.5)


def test_precision_divergence_pred_final_absent_from_golden() -> None:
    golden = _diff(SCORE_FX / "golden_precision.json")
    pred = _diff(SCORE_FX / "pred_precision.json")

    strict = score_task("prec", pred, golden, "strict")
    assert (strict.tp, strict.fp, strict.fn) == (1, 1, 0)
    assert strict.precision == pytest.approx(0.5)
    assert strict.recall == pytest.approx(1.0)
    assert strict.f1 == pytest.approx(TWO_THIRDS)

    compat = score_task("prec", pred, golden, "annotated")

    assert (compat.tp, compat.fp, compat.fn) == (1, 0, 0)
    assert (compat.precision, compat.recall, compat.f1) == (1.0, 1.0, 1.0)


def test_recall_divergence_golden_final_absent_from_prediction() -> None:
    golden = _diff(SCORE_FX / "golden_recall.json")
    pred = _diff(SCORE_FX / "pred_recall.json")

    strict = score_task("rec", pred, golden, "strict")
    assert (strict.tp, strict.fp, strict.fn) == (1, 0, 1)
    assert strict.precision == pytest.approx(1.0)
    assert strict.recall == pytest.approx(0.5)
    assert strict.f1 == pytest.approx(TWO_THIRDS)

    compat = score_task("rec", pred, golden, "annotated")

    assert (compat.tp, compat.fp, compat.fn) == (1, 0, 0)
    assert (compat.precision, compat.recall, compat.f1) == (1.0, 1.0, 1.0)


def test_zero_final_task_pools_without_error() -> None:
    z = _diff(SCORE_FX / "zero_final.json")
    for mode in ("strict", "annotated"):
        detail = score_task("zero", z, z, mode)
        assert (detail.tp, detail.fp, detail.fn) == (0, 0, 0)
        assert (detail.precision, detail.recall, detail.f1) == (1.0, 1.0, 1.0)


def test_per_sheet_zero_final_does_not_break_task_pool() -> None:

    payload = {
        "spec_version": 2,
        "task_tolerance": 0.01,
        "sheets": {
            "S1": {
                "sheet_weight": 0.5,
                "groups": {"final": {"weight": 0.4, "cells": [{"cell": "A1", "cell_type": "currency"}]}},
            },
            "S2": {
                "sheet_weight": 0.5,
                "groups": {"intermediate": {"weight": 1, "cells": [{"cell": "B1", "cell_type": "number"}]}},
            },
        },
    }
    diff = DiffFile.model_validate(payload)
    detail = score_task("mixed", diff, diff, "strict")
    assert (detail.tp, detail.fp, detail.fn) == (1, 0, 0)
    assert (detail.precision, detail.recall, detail.f1) == (1.0, 1.0, 1.0)
    by_sheet = {sb.sheet: (sb.tp, sb.fp, sb.fn) for sb in detail.per_sheet}
    assert by_sheet == {"S1": (1, 0, 0), "S2": (0, 0, 0)}


def test_eval_results_top_level_shape() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")
    details = {"combined": score_task("combined", pred, golden, "strict")}
    results = build_eval_results(details, variant="V1", model="gpt-5.4-mini", timestamp="2026-07-07T00:00:00Z")

    assert set(results.keys()) == {"combined", "meta"}
    assert set(results["combined"].keys()) == {"precision", "recall", "f1"}
    meta = results["meta"]
    assert meta["mode"] == "strict"
    assert meta["variant"] == "V1"
    assert meta["timestamp"] == "2026-07-07T00:00:00Z"
    assert "aggregates" in meta and "micro" in meta["aggregates"] and "macro" in meta["aggregates"]
    assert "per_sheet" in meta


def test_eval_results_rejects_task_named_meta() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    details = {"meta": score_task("meta", golden, golden, "strict")}
    with pytest.raises(ValueError):
        build_eval_results(details)


def test_eval_results_rejects_mixed_modes() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    details = {
        "a": score_task("a", golden, golden, "strict"),
        "b": score_task("b", golden, golden, "annotated"),
    }
    with pytest.raises(ValueError):
        build_eval_results(details)


def test_sheet_breakdown_carries_golden_intermediate_count() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    detail = score_task("combined", golden, golden, "annotated")

    by_sheet = {sb.sheet: sb.golden_intermediate for sb in detail.per_sheet}
    assert by_sheet == {"S1": 2}


def test_eval_results_extended_golden_vs_itself_near_perfect() -> None:

    golden = _diff(SCORE_FX / "golden_combined.json")
    details = {"combined": score_task("combined", golden, golden, "annotated")}
    results = build_eval_results(details, timestamp="2026-07-07T00:00:00Z")
    extended = results["meta"]["extended"]
    assert extended["pooled"] == {
        "tp": 3,
        "fp": 0,
        "fn": 0,
        "tn": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert extended["mcc"] == pytest.approx(1.0)
    assert extended["balanced_accuracy"] == pytest.approx(1.0)
    assert extended["youdens_j"] == pytest.approx(1.0)
    assert "mcc_note" not in extended


def test_eval_results_extended_hand_computed_mcc() -> None:

    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")
    details = {"combined": score_task("combined", pred, golden, "annotated")}
    results = build_eval_results(details, timestamp="2026-07-07T00:00:00Z")
    extended = results["meta"]["extended"]
    assert extended["pooled"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert extended["mcc"] == pytest.approx(0.0)
    assert extended["balanced_accuracy"] == pytest.approx(0.5)


def test_eval_results_extended_absent_for_strict_mode() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")
    details = {"combined": score_task("combined", pred, golden, "strict")}
    results = build_eval_results(details, timestamp="2026-07-07T00:00:00Z")
    assert "extended" not in results["meta"]


def test_write_eval_results_roundtrip(tmp_path: Path) -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")
    details = {"combined": score_task("combined", pred, golden, "strict")}
    results = build_eval_results(details, timestamp="2026-07-07T00:00:00Z")
    out = tmp_path / "eval_results.json"
    write_eval_results(out, results)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["combined"]["f1"] == pytest.approx(ONE_THIRD)
    assert reloaded["meta"]["mode"] == "strict"


def test_error_report_groups_fp_fn_by_sheet_and_row() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")
    report = build_error_report("combined", pred, golden, "strict")
    assert report["task"] == "combined"
    assert report["mode"] == "strict"

    fps = report["false_positives"]["S1"]
    assert set(fps.keys()) == {"1"}
    assert {c["cell"] for c in fps["1"]} == {"B1", "C1"}

    assert fps["1"][0]["label"] is None
    assert fps["1"][0]["notes"] is None
    fns = report["false_negatives"]["S1"]
    assert set(fns.keys()) == {"2", "3"}
    assert fns["2"][0]["cell"] == "A2"
    assert fns["2"][0]["cell_type"] == "currency"


def test_error_report_compat_uses_intermediate_labeled_cells() -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    pred = _diff(SCORE_FX / "pred_combined.json")
    report = build_error_report("combined", pred, golden, "annotated")

    assert {c["cell"] for row in report["false_positives"]["S1"].values() for c in row} == {"B1"}
    assert {c["cell"] for row in report["false_negatives"]["S1"].values() for c in row} == {"A2"}


def test_write_error_report_roundtrip(tmp_path: Path) -> None:
    golden = _diff(SCORE_FX / "golden_recall.json")
    pred = _diff(SCORE_FX / "pred_recall.json")
    report = build_error_report("rec", pred, golden, "strict")
    out = tmp_path / "errors.json"
    write_error_report(out, report)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["false_negatives"]["S1"]["2"][0]["cell"] == "A2"


@pytest.mark.parametrize("cost_usd", [0.0, 5.0])
@pytest.mark.parametrize("perfect", [False, True])
def test_eval_results_cost_diagnostics_are_strict_json(tmp_path: Path, cost_usd: float, perfect: bool) -> None:
    golden = _diff(SCORE_FX / "golden_combined.json")
    prediction = golden.model_copy(deep=True)
    if not perfect:
        for sheet in prediction.sheets.values():
            sheet.groups.intermediate.cells.extend(sheet.groups.final.cells)
            sheet.groups.final.cells.clear()
    detail = score_task("synthetic", prediction, golden, "annotated")
    results = build_eval_results({"synthetic": detail}, cost_usd=cost_usd)
    encoded = json.dumps(results, allow_nan=False)
    extended = results["meta"]["extended"]
    undefined = extended.get("undefined_cost_metrics", {})
    for metric in ("f1", "recall"):
        if not perfect:
            assert extended["cost_of_pass"][metric] is None
            assert undefined[f"cost_of_pass.{metric}"] == "Metric is zero; cost per pass is undefined."
        else:
            assert extended["cost_of_pass"][metric] == cost_usd
        if cost_usd == 0 and perfect:
            assert extended["per_dollar"][metric] is None
            assert undefined[f"per_dollar.{metric}"] == "Cost is zero; positive metric per dollar is undefined."
        else:
            assert extended["per_dollar"][metric] == (1 / cost_usd if perfect else 0)
    output = tmp_path / "scores.json"
    write_eval_results(output, results)
    assert json.loads(output.read_text()) == json.loads(encoded)


@pytest.mark.parametrize("writer", [write_eval_results, write_error_report])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_json_writers_reject_nonfinite_without_overwriting(tmp_path: Path, writer, value: float) -> None:
    output = tmp_path / "existing.json"
    output.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float"):
        writer(output, {"unexpected": {"diagnostic": value}})
    assert output.read_text() == '{"previous": true}'
