from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from formuloom.bundle import TaskBundle
from formuloom.classify import ClassifyError
from formuloom.cli import _LLMPipeline
from formuloom.compose import (
    ComposePipeline,
    final_map_from_diff,
    intersect_predicted_final,
)
from formuloom.schema import VariantConfig
from formuloom.settings import get_settings
from tests.test_cli import (
    _incomplete_response,
    _install_fake_client,
    _message_response,
    _write_task_bundle,
)


def test_intersect_keeps_only_cells_all_constituents_mark_final() -> None:
    a = {"S1": {"A1", "B1", "C1"}}
    b = {"S1": {"B1", "C1", "D1"}}
    assert intersect_predicted_final([a, b]) == {"S1": {"B1", "C1"}}


def test_intersect_sheet_present_in_one_constituent_only_is_empty() -> None:

    a = {"S1": {"A1"}, "S2": {"X9"}}
    b = {"S1": {"A1"}}
    merged = intersect_predicted_final([a, b])
    assert merged == {"S1": {"A1"}, "S2": set()}


def test_intersect_three_constituents() -> None:
    a = {"S1": {"A1", "B1"}}
    b = {"S1": {"A1", "B1", "C1"}}
    c = {"S1": {"A1"}}
    assert intersect_predicted_final([a, b, c]) == {"S1": {"A1"}}


def test_intersect_empty_constituent_list_is_empty_map() -> None:
    assert intersect_predicted_final([]) == {}


def test_intersect_single_constituent_is_passthrough() -> None:
    a = {"S1": {"A1", "B1"}, "S2": set()}
    assert intersect_predicted_final([a]) == {"S1": {"A1", "B1"}, "S2": set()}


def test_compose_pipeline_rejects_empty_constituents() -> None:
    with pytest.raises(ValueError, match="at least one constituent"):
        ComposePipeline([], lambda config: _LLMPipeline(config, get_settings()))


def test_compose_pipeline_rejects_nested_composite() -> None:
    nested = VariantConfig(name="Nested", compose_intersect=("V9",))
    with pytest.raises(ValueError, match="nested composition is not supported"):
        ComposePipeline([nested], lambda config: _LLMPipeline(config, get_settings()))


_CFG_A = VariantConfig(name="CA", prompt_version="v1", use_task_profile=False, voting_k=1)
_CFG_B = VariantConfig(name="CB", prompt_version="v1", use_task_profile=False, voting_k=1)


def _row_response(rows: list[int]) -> str:
    return json.dumps({"notes": "test", "final_rows": rows})


def _compose(constituents: list[VariantConfig]) -> ComposePipeline:
    settings = get_settings()
    return ComposePipeline(constituents, lambda config: _LLMPipeline(config, settings))


def test_compose_intersects_two_constituents_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")

    calls = _install_fake_client(
        monkeypatch, [_message_response(_row_response([1, 2])), _message_response(_row_response([1]))]
    )

    pipeline = _compose([_CFG_A, _CFG_B])
    result = asyncio.run(pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=False))

    assert len(calls) == 2
    assert result.diff.final_refs("S1") == {"B1"}
    assert result.diff.intermediate_refs("S1") == {"B2", "B3"}
    assert result.failures == {}
    assert result.usage.cost_usd > 0


def test_compose_respects_best_effort_degrading_a_failed_constituent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")

    _install_fake_client(monkeypatch, [_message_response(_row_response([1])), _incomplete_response()])

    pipeline = _compose([_CFG_A, _CFG_B])
    result = asyncio.run(pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=True))

    assert result.diff.final_refs("S1") == set()
    assert result.diff.intermediate_refs("S1") == {"B1", "B2", "B3"}
    assert set(result.failures) == {"S1"}
    assert result.failures["S1"].startswith("CB: ")


def test_compose_without_best_effort_hard_fails_on_constituent_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")
    _install_fake_client(monkeypatch, [_message_response(_row_response([1])), _incomplete_response()])

    pipeline = _compose([_CFG_A, _CFG_B])
    with pytest.raises(ClassifyError):
        asyncio.run(pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=False))


def test_compose_caches_are_isolated_per_constituent_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")
    run_dir = tmp_path / "run"
    calls = _install_fake_client(
        monkeypatch, [_message_response(_row_response([1])), _message_response(_row_response([1]))]
    )

    pipeline = _compose([_CFG_A, _CFG_B])
    first = asyncio.run(pipeline.predict(bundle, run_dir=run_dir, best_effort=False))
    assert len(calls) == 2

    cache_files = sorted(p.name for p in (run_dir / "sheets").glob("*.json"))
    assert len(cache_files) == 2
    assert any("CA" in name for name in cache_files)
    assert any("CB" in name for name in cache_files)

    second = asyncio.run(pipeline.predict(bundle, run_dir=run_dir, best_effort=False))
    assert len(calls) == 2
    assert second.diff.final_refs("S1") == first.diff.final_refs("S1") == {"B1"}


def test_final_map_from_diff_reads_every_sheet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle_dir = _write_task_bundle(tmp_path / "Fixture", with_subset=False)
    bundle = TaskBundle.load(bundle_dir, "predict")
    calls = _install_fake_client(monkeypatch, [_message_response(_row_response([1]))])

    pipeline = _LLMPipeline(_CFG_A, get_settings())
    result = asyncio.run(pipeline.predict(bundle, run_dir=tmp_path / "run", best_effort=False))
    assert len(calls) == 1
    assert final_map_from_diff(result.diff) == {"S1": {"B1"}}
