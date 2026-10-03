from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from formuloom.schema import (
    SPEC_VERSION,
    TASK_TOLERANCE,
    Aggregates,
    CellEntry,
    DiffFile,
    MetricTriple,
    SheetBreakdown,
    TaskScore,
    TaskScoreDetail,
    VariantConfig,
    cell_column_index,
    cell_column_letters,
    cell_row,
    cell_sort_key,
)

DATA = Path(__file__).parent.parent / "data"
FIXTURES = Path(__file__).parent / "fixtures"
TASKS = ["synthetic-statement", "synthetic-budget", "synthetic-rollup", "synthetic-projection", "synthetic-summary"]

requires_data = pytest.mark.skipif(not DATA.is_dir(), reason="data/ bundles not present")


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_cell_row_and_column() -> None:
    assert cell_row("D5") == 5
    assert cell_column_letters("D5") == "D"
    assert cell_column_index("A1") == 1
    assert cell_column_index("Z9") == 26
    assert cell_column_index("AA1") == 27
    assert cell_column_index("AB100") == 28


def test_cell_sort_key_orders_by_row_then_column() -> None:
    refs = ["B2", "A2", "A10", "A1"]
    assert sorted(refs, key=cell_sort_key) == ["A1", "A2", "B2", "A10"]


def test_cell_row_rejects_non_a1() -> None:
    with pytest.raises(ValueError):
        cell_row("D5:E5")


def test_diff_policy_constants() -> None:
    assert SPEC_VERSION == 2
    assert TASK_TOLERANCE == 0.01


def test_extra_keys_are_rejected() -> None:
    payload = {
        "spec_version": 2,
        "task_tolerance": 0.01,
        "sheets": {"S1": {"sheet_weight": 1.0, "groups": {"intermediate": {"weight": 1, "cells": []}}}},
        "surprise": True,
    }
    with pytest.raises(ValidationError):
        DiffFile.model_validate(payload)


def test_unknown_cell_type_is_rejected() -> None:
    payload = {
        "spec_version": 2,
        "task_tolerance": 0.01,
        "sheets": {
            "S1": {
                "sheet_weight": 1.0,
                "groups": {"intermediate": {"weight": 1, "cells": [{"cell": "A1", "cell_type": "boolean"}]}},
            }
        },
    }
    with pytest.raises(ValidationError):
        DiffFile.model_validate(payload)


def test_multigroup_fixture_final_and_intermediate_refs() -> None:
    diff = DiffFile.model_validate(_load_json(FIXTURES / "multigroup_raw_diff.json"))
    assert diff.intermediate_refs("Statement") == {"B2", "B3", "C5"}
    assert diff.final_refs("Statement") == {"B10", "B3"}
    assert diff.final_types("Statement")["B10"] == "currency"
    assert diff.intermediate_refs("Inputs") == {"A1", "A2"}
    assert diff.final_refs("Inputs") == set()


def _roundtrip_equal(payload: dict[str, Any]) -> None:
    m1 = DiffFile.model_validate(payload)
    m2 = DiffFile.model_validate(json.loads(json.dumps(m1.model_dump(mode="json"))))
    assert m1 == m2


def test_roundtrip_fixture() -> None:
    _roundtrip_equal(_load_json(FIXTURES / "multigroup_raw_diff.json"))


def test_variant_config_defaults() -> None:
    v = VariantConfig(name="V1")
    assert v.grouping == "auto"
    assert v.universe == "raw"
    assert v.compression is True
    assert v.features_in_prompt is False
    assert v.reasoning_effort == "low"
    assert v.voting_k == 1

    assert "scorer_mode" not in VariantConfig.model_fields


def test_variant_config_rejects_bad_grouping() -> None:
    with pytest.raises(ValidationError):
        VariantConfig(name="bad", grouping="diagonal")  # type: ignore[arg-type]


def test_task_score_shape_is_exactly_three_fields() -> None:
    ts = TaskScore(precision=0.5, recall=0.5, f1=0.5)
    assert set(ts.model_dump().keys()) == {"precision", "recall", "f1"}


def test_eval_models_construct() -> None:
    detail = TaskScoreDetail(
        task="t",
        mode="strict",
        tp=1,
        fp=0,
        fn=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        per_sheet=[SheetBreakdown(sheet="S1", tp=1, fp=0, fn=0)],
    )
    assert detail.per_sheet[0].sheet == "S1"
    agg = Aggregates(
        micro=MetricTriple(precision=1.0, recall=1.0, f1=1.0),
        macro=MetricTriple(precision=1.0, recall=1.0, f1=1.0),
    )
    assert agg.micro.f1 == 1.0


def test_cell_entry_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        CellEntry.model_validate({"cell": "A1", "cell_type": "text", "note": "x"})
