from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from formuloom.schema import (
    Aggregates,
    CellType,
    DiffFile,
    MetricTriple,
    ScoreMode,
    SheetBreakdown,
    TaskScore,
    TaskScoreDetail,
    cell_row,
    cell_sort_key,
)

SCORE_MODES: tuple[ScoreMode, ...] = ("strict", "annotated")

META_KEY = "meta"


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    predicted_positive = tp + fp
    actual_positive = tp + fn
    if predicted_positive == 0 and actual_positive == 0:
        return 1.0, 1.0, 1.0
    precision = tp / predicted_positive if predicted_positive > 0 else 0.0
    recall = tp / actual_positive if actual_positive > 0 else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _sheet_counts(prediction: DiffFile, golden: DiffFile, sheet: str, mode: ScoreMode) -> SheetBreakdown:
    pred_final = prediction.final_refs(sheet)
    pred_inter = prediction.intermediate_refs(sheet)
    gold_final = golden.final_refs(sheet)
    gold_inter = golden.intermediate_refs(sheet)

    tp = len(pred_final & gold_final)
    if mode == "strict":
        fp = len(pred_final - gold_final)
        fn = len(gold_final - pred_final)
    else:
        fp = len(pred_final & gold_inter)
        fn = len(pred_inter & gold_final)
    return SheetBreakdown(sheet=sheet, tp=tp, fp=fp, fn=fn, golden_intermediate=len(gold_inter))


def score_task(task: str, prediction: DiffFile, golden: DiffFile, mode: ScoreMode) -> TaskScoreDetail:
    sheets = sorted(set(prediction.sheets) | set(golden.sheets))
    per_sheet = [_sheet_counts(prediction, golden, sheet, mode) for sheet in sheets]
    tp = sum(sb.tp for sb in per_sheet)
    fp = sum(sb.fp for sb in per_sheet)
    fn = sum(sb.fn for sb in per_sheet)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return TaskScoreDetail(
        task=task,
        mode=mode,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        per_sheet=per_sheet,
    )


def aggregate(details: Mapping[str, TaskScoreDetail]) -> Aggregates:
    tp = sum(d.tp for d in details.values())
    fp = sum(d.fp for d in details.values())
    fn = sum(d.fn for d in details.values())
    micro_p, micro_r, micro_f1 = precision_recall_f1(tp, fp, fn)
    micro = MetricTriple(precision=micro_p, recall=micro_r, f1=micro_f1)

    n = len(details)
    if n == 0:
        macro = MetricTriple(precision=0.0, recall=0.0, f1=0.0)
    else:
        macro = MetricTriple(
            precision=sum(d.precision for d in details.values()) / n,
            recall=sum(d.recall for d in details.values()) / n,
            f1=sum(d.f1 for d in details.values()) / n,
        )
    return Aggregates(micro=micro, macro=macro)


def build_eval_results(
    details: Mapping[str, TaskScoreDetail],
    *,
    variant: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    tokens: int | None = None,
    cost_usd: float | None = None,
    wall_clock_seconds: float | None = None,
    timestamp: str | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if META_KEY in details:
        raise ValueError(f"a task may not be named {META_KEY!r}: it collides with the reserved meta key")
    modes = {d.mode for d in details.values()}
    if len(modes) > 1:
        raise ValueError(f"cannot mix scorer modes in one eval_results: {sorted(modes)}")
    mode = next(iter(modes)) if modes else None

    result: dict[str, Any] = {}
    for task, detail in details.items():
        result[task] = TaskScore(precision=detail.precision, recall=detail.recall, f1=detail.f1).model_dump()

    meta: dict[str, Any] = {
        "mode": mode,
        "aggregates": aggregate(details).model_dump(),
        "per_task": {task: {"tp": d.tp, "fp": d.fp, "fn": d.fn} for task, d in details.items()},
        "per_sheet": {task: [sb.model_dump() for sb in d.per_sheet] for task, d in details.items()},
        "tokens": tokens,
        "cost_usd": cost_usd,
        "wall_clock_seconds": wall_clock_seconds,
        "variant": variant,
        "model": model,
        "prompt_version": prompt_version,
        "timestamp": timestamp if timestamp is not None else datetime.now(UTC).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)
    if mode == "annotated":

        from formuloom.metrics_extra import ClusterCounts, extended_metrics

        clusters = [ClusterCounts(sb.sheet, sb.tp, sb.fp, sb.fn) for d in details.values() for sb in d.per_sheet]
        pooled_tn = sum(sb.golden_intermediate - sb.fp for d in details.values() for sb in d.per_sheet)
        extended = extended_metrics(clusters, tn=pooled_tn, cost_usd=cost_usd)
        # Keep library ratio conventions, but represent undefined export diagnostics as JSON null.
        undefined: dict[str, str] = {}
        for name, reason in (
            ("cost_of_pass", "Metric is zero; cost per pass is undefined."),
            ("per_dollar", "Cost is zero; positive metric per dollar is undefined."),
        ):
            block = extended.get(name)
            if isinstance(block, dict):
                for metric, value in block.items():
                    if value == float("inf"):
                        block[metric] = None
                        undefined[f"{name}.{metric}"] = reason
        if undefined:
            extended["undefined_cost_metrics"] = undefined
        meta["extended"] = extended
    result[META_KEY] = meta
    return result


def write_eval_results(path: Path | str, results: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _error_cells(
    prediction: DiffFile, golden: DiffFile, sheet: str, mode: ScoreMode
) -> tuple[dict[str, CellType], dict[str, CellType]]:
    pred_final = prediction.final_types(sheet)
    pred_inter = prediction.intermediate_types(sheet)
    gold_final = golden.final_types(sheet)
    gold_inter = golden.intermediate_types(sheet)

    if mode == "strict":
        fp_refs = set(pred_final) - set(gold_final)
        fn_refs = set(gold_final) - set(pred_final)
    else:
        fp_refs = set(pred_final) & set(gold_inter)
        fn_refs = set(pred_inter) & set(gold_final)

    fps = {ref: pred_final[ref] for ref in fp_refs}
    fns = {ref: gold_final[ref] for ref in fn_refs}
    return fps, fns


def _group_by_row(typed: Mapping[str, str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in sorted(typed, key=cell_sort_key):
        row_key = str(cell_row(ref))
        grouped.setdefault(row_key, []).append({"cell": ref, "cell_type": typed[ref], "label": None, "notes": None})
    return grouped


def build_error_report(task: str, prediction: DiffFile, golden: DiffFile, mode: ScoreMode) -> dict[str, Any]:
    sheets = sorted(set(prediction.sheets) | set(golden.sheets))
    false_positives: dict[str, dict[str, list[dict[str, Any]]]] = {}
    false_negatives: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for sheet in sheets:
        fps, fns = _error_cells(prediction, golden, sheet, mode)
        if fps:
            false_positives[sheet] = _group_by_row(fps)
        if fns:
            false_negatives[sheet] = _group_by_row(fns)
    return {
        "task": task,
        "mode": mode,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def write_error_report(path: Path | str, report: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
