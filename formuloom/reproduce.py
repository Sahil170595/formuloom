"""Run a fresh synthetic workbook through real, provider-free implementations."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from formuloom.bundle import TaskBundle
from formuloom.cli import predict_v0
from formuloom.fixtures import generate_fixture
from formuloom.offline import inspect_bundle, run_mechanical_v15
from formuloom.score import SCORE_MODES, build_error_report, score_task
from formuloom.weak import predict_v8


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reproduce(out: Path) -> dict[str, Any]:
    """Refuse overwrite and score only independently authored synthetic labels."""
    out.mkdir(parents=True, exist_ok=False)
    root = generate_fixture(out / "fixture")
    prediction_bundle = TaskBundle.load(root, "predict")
    weak, _ = predict_v8(prediction_bundle, cache_dir=out / "cache")
    mechanical, routes = asyncio.run(run_mechanical_v15(prediction_bundle, run_dir=out / "mechanical"))
    predictions = {"V0": predict_v0(prediction_bundle), "V8": weak, "V15-mechanical": mechanical.diff}
    # Labels are loaded only after every prediction has been computed.
    reference = TaskBundle.load(root, "score").golden
    if reference is None:
        raise ValueError("Synthetic fixture must contain a reference policy")
    inspection = inspect_bundle(prediction_bundle, cache_dir=out / "cache")
    _write(out / "inspection.json", inspection)
    _write(out / "routes.json", routes)
    report: dict[str, Any] = {
        "schema_version": 1,
        "synthetic": True,
        "fixture": "operations-rollup-v1",
        "provider_calls": mechanical.usage.api_calls,
        "candidate_cells": sum(len(cells) for cells in prediction_bundle.candidate_cells().values()),
        "reference_final_cells": sum(len(reference.final_refs(sheet)) for sheet in reference.sheets),
        "scope": "Synthetic integration example, not held-out evidence or original provider results",
        "mechanical_substitutions": {"V11": "V0 rules", "V9": "union of V0 and V8"},
        "predictions": {},
    }
    for name, prediction in predictions.items():
        _write(out / f"{name}.json", prediction.model_dump(mode="json"))
        report["predictions"][name] = {
            mode: score_task("synthetic", prediction, reference, mode).model_dump() for mode in SCORE_MODES
        }
        _write(out / f"{name}-errors.json", build_error_report("synthetic", prediction, reference, "strict"))
    _write(out / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="New output directory; existing paths are not overwritten"
    )
    args = parser.parse_args()
    report = reproduce(args.out)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
