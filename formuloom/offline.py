"""Provider-free inspection and explicitly substituted V15 mechanical reproduction."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from formuloom.assemble import assemble_diff_file
from formuloom.bundle import TaskBundle
from formuloom.classify import UsageRecord
from formuloom.cli import predict_v0
from formuloom.compose import PipelineResult
from formuloom.features import build_task_features
from formuloom.fixtures import FIXTURE_MARKER
from formuloom.schema import VariantConfig
from formuloom.v15_router import V15RouterPipeline, route_v15_sheet
from formuloom.variants import get_variant
from formuloom.weak import predict_v8
from formuloom.workbook import load_workbook_data


class MechanicalArm:
    """Deterministic stand-in; this is not the corresponding provider variant."""

    def __init__(self, config: VariantConfig) -> None:
        self.config = config

    async def predict(self, bundle: TaskBundle, *, run_dir: Path, best_effort: bool) -> PipelineResult:
        rules = predict_v0(bundle)
        if self.config.name == "V11":
            diff = rules
        else:
            weak, _ = predict_v8(bundle)
            diff = assemble_diff_file(
                bundle.raw_diff,
                {sheet: rules.final_refs(sheet) | weak.final_refs(sheet) for sheet in bundle.raw_diff.sheets},
            )
        return PipelineResult(diff=diff, failures={}, usage=UsageRecord())


async def run_mechanical_v15(
    bundle: TaskBundle,
    *,
    run_dir: Path,
) -> tuple[PipelineResult, dict[str, dict[str, Any]]]:
    """Execute the original V15 router with explicitly replaced deterministic arms."""
    pipeline = V15RouterPipeline([get_variant("V11"), get_variant("V9")], MechanicalArm)
    result = await pipeline.predict(bundle, run_dir=run_dir, best_effort=False)
    features = build_task_features(bundle, load_workbook_data(bundle.complete_path))
    default = await MechanicalArm(get_variant("V11")).predict(bundle, run_dir=run_dir, best_effort=False)
    recall = await MechanicalArm(get_variant("V9")).predict(bundle, run_dir=run_dir, best_effort=False)
    routes = {}
    for name in bundle.raw_diff.sheets:
        _, decision = route_v15_sheet(
            features.sheets[name],
            default_cells=default.diff.final_refs(name),
            high_recall_cells=recall.diff.final_refs(name),
        )
        routes[name] = asdict(decision)
    return result, routes


def inspect_bundle(bundle: TaskBundle, *, cache_dir: Path | None = None) -> dict[str, Any]:
    complete = load_workbook_data(bundle.complete_path, cache_dir)
    features = build_task_features(bundle, complete)
    _, weak = predict_v8(bundle, cache_dir=cache_dir)
    return {
        "schema_version": 1,
        "synthetic": bundle.instructions.startswith(FIXTURE_MARKER),
        "sheets": {
            name: {
                "candidates": sheet.n_candidates,
                "density": sheet.candidate_density,
                "cross_sheet_precedents": sheet.n_cells_with_cross_sheet_precedents,
                "cross_sheet_dependents": sheet.n_cells_with_cross_sheet_dependents,
                "rows": [asdict(row) for row in sheet.rows],
            }
            for name, sheet in features.sheets.items()
        },
        "graph_warnings": [
            {"sheet": key[0], "ref": key[1], "warnings": warnings} for key, warnings in features.graph.warnings.items()
        ],
        "weak_supervision": weak,
    }
