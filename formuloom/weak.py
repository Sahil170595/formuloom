"""V8 offline adapter over the retained labeling-function and Dawid-Skene implementation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from formuloom.assemble import assemble_diff_file
from formuloom.bundle import TaskBundle
from formuloom.features import build_task_features
from formuloom.labelmodel import DOMAIN_FINAL_PRIOR, apply_labeling_functions, fit_label_model
from formuloom.schema import DiffFile
from formuloom.workbook import load_workbook_data

DEFAULT_V8_THRESHOLD = 0.5


def predict_v8(
    bundle: TaskBundle,
    *,
    threshold: float = DEFAULT_V8_THRESHOLD,
    prior: float = DOMAIN_FINAL_PRIOR,
    cache_dir: Path | None = None,
) -> tuple[DiffFile, dict[str, Any]]:
    """Pool candidate rows across a workbook, fit without reference labels, then threshold."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if not math.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("prior must be finite and in (0, 1)")
    complete = load_workbook_data(bundle.complete_path, cache_dir)
    features = build_task_features(bundle, complete)
    rows = [row for name in bundle.raw_diff.sheets for row in features.sheets[name].rows]
    votes = apply_labeling_functions(rows)
    model = fit_label_model(votes.votes, votes.lf_names, prior_anchor=prior, prior_floor=prior, prior_ceiling=prior)
    finals: dict[str, set[str]] = {name: set() for name in bundle.raw_diff.sheets}
    for row, probability in zip(rows, model.probabilities, strict=True):
        if probability >= threshold:
            finals[row.sheet].update(row.candidate_refs)
    diagnostics = {
        "prior": model.class_prior,
        "threshold": threshold,
        "lf_names": list(votes.lf_names),
        "rows": [{"sheet": row.sheet, "row": row.row} for row in rows],
        "votes": votes.votes,
        "probabilities": list(model.probabilities),
        "lf_accuracies": {name: asdict(value) for name, value in model.lf_accuracies.items()},
        "em": asdict(model.diagnostics),
    }
    return assemble_diff_file(bundle.raw_diff, finals), diagnostics
