from __future__ import annotations

from pathlib import Path

import pytest

from formuloom.bundle import (
    PREDICT_ALLOWLIST,
    SCORE_ALLOWLIST,
    LeakageError,
    MalformedBundleError,
    MissingBundleFileError,
    TaskBundle,
    _read_text,
)

DATA = Path(__file__).parent.parent / "data"
REAL_TASK = "synthetic-budget"
REAL_DIR = DATA / REAL_TASK

requires_data = pytest.mark.skipif(not REAL_DIR.is_dir(), reason="data/ bundles not present")


def _make_predict_bundle(root: Path, raw_diff_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "init.xlsx").write_bytes(b"")
    (root / "complete.xlsx").write_bytes(b"")
    (root / "instructions.md").write_text("do the thing", encoding="utf-8")
    (root / "raw_diff.json").write_text(raw_diff_text, encoding="utf-8")
    return root


_VALID_RAW_DIFF = (
    '{"spec_version": 2, "task_tolerance": 0.01, "sheets": '
    '{"S1": {"sheet_weight": 1.0, "groups": {"intermediate": '
    '{"weight": 1, "cells": [{"cell": "A1", "cell_type": "text"}]}}}}}'
)


def test_candidate_cells_union_dedup_order(tmp_path: Path) -> None:

    fixture = Path(__file__).parent / "fixtures" / "multigroup_raw_diff.json"
    root = _make_predict_bundle(tmp_path / "MG", fixture.read_text(encoding="utf-8"))
    bundle = TaskBundle.load(root, "predict")
    cands = bundle.candidate_cells()
    stmt = [c.cell for c in cands["Statement"]]

    assert stmt == ["B2", "B3", "C5", "B10"]
    assert [c.cell for c in cands["Inputs"]] == ["A1", "A2"]


def test_predict_works_when_subset_absent(tmp_path: Path) -> None:
    dest = tmp_path / REAL_TASK
    _make_predict_bundle(dest, _VALID_RAW_DIFF)
    (dest / "subset.json").write_text(_VALID_RAW_DIFF, encoding="utf-8")
    (dest / "subset.json").unlink()
    assert not (dest / "subset.json").exists()

    bundle = TaskBundle.load(dest, "predict")
    assert bundle.golden is None
    assert bundle.candidate_cells()

    with pytest.raises(MissingBundleFileError):
        TaskBundle.load(dest, "score")


def test_read_text_allowlist_is_a_real_gate(tmp_path: Path) -> None:
    subset = tmp_path / "subset.json"
    subset.write_text(_VALID_RAW_DIFF, encoding="utf-8")
    with pytest.raises(LeakageError):
        _read_text(subset, PREDICT_ALLOWLIST)

    assert _read_text(subset, SCORE_ALLOWLIST).strip().startswith("{")


def test_missing_file_raises(tmp_path: Path) -> None:
    root = _make_predict_bundle(tmp_path / "MISS", _VALID_RAW_DIFF)
    (root / "raw_diff.json").unlink()
    with pytest.raises(MissingBundleFileError):
        TaskBundle.load(root, "predict")


def test_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingBundleFileError):
        TaskBundle.load(tmp_path / "nope", "predict")


def test_malformed_json_raises(tmp_path: Path) -> None:
    root = _make_predict_bundle(tmp_path / "BAD", "{ not valid json <<<")
    with pytest.raises(MalformedBundleError):
        TaskBundle.load(root, "predict")


def test_schema_invalid_json_raises(tmp_path: Path) -> None:
    root = _make_predict_bundle(tmp_path / "WRONG", '{"unexpected": true}')
    with pytest.raises(MalformedBundleError):
        TaskBundle.load(root, "predict")
