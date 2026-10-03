from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from formuloom.score import precision_recall_f1

WILSON_Z_95: float = 1.96

DEFAULT_BOOTSTRAP_RESAMPLES: int = 1000

DEFAULT_BOOTSTRAP_ALPHA: float = 0.05

DEFAULT_BOOTSTRAP_SEED: int = 13

DEFAULT_PERMUTATION_RESAMPLES: int = 2000

DEFAULT_PERMUTATION_SEED: int = 13

MCNEMAR_EXACT_MAX_DISCORDANT: int = 25

F2_BETA: float = 2.0

DEGENERATE_DENOMINATOR_EPSILON: float = 1e-12

PERMUTATION_SMOOTHING: int = 1

FLEISS_KAPPA_VACUOUS_AGREEMENT: float = 1.0

COST_OF_PASS_CITATION: str = "arXiv:2504.13359"


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    margins = (tp + fp, tp + fn, tn + fp, tn + fn)
    if any(margin == 0 for margin in margins):
        return 0.0
    numerator = tp * tn - fp * fn
    denominator = math.sqrt(margins[0] * margins[1] * margins[2] * margins[3])
    return numerator / denominator


def fbeta(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def f2(precision: float, recall: float) -> float:
    return fbeta(precision, recall, F2_BETA)


def _sensitivity(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _specificity(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def balanced_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    return (_sensitivity(tp, fn) + _specificity(tn, fp)) / 2.0


def youdens_j(tp: int, fp: int, fn: int, tn: int) -> float:
    return _sensitivity(tp, fn) + _specificity(tn, fp) - 1.0


def wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = phat + z2 / (2.0 * total)
    adjustment = z * math.sqrt(phat * (1.0 - phat) / total + z2 / (4.0 * total * total))
    lo = (center - adjustment) / denominator
    hi = (center + adjustment) / denominator
    return (max(0.0, lo), min(1.0, hi))


class ClusterCounts(NamedTuple):

    sheet: str
    tp: int
    fp: int
    fn: int


def _pooled_prf(clusters: Sequence[ClusterCounts]) -> tuple[float, float, float]:
    tp = sum(c.tp for c in clusters)
    fp = sum(c.fp for c in clusters)
    fn = sum(c.fn for c in clusters)
    return precision_recall_f1(tp, fp, fn)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = q * (n - 1)
    lo_idx = math.floor(pos)
    hi_idx = math.ceil(pos)
    if lo_idx == hi_idx:
        return ordered[int(pos)]
    frac = pos - lo_idx
    return ordered[lo_idx] + (ordered[hi_idx] - ordered[lo_idx]) * frac


def cluster_bootstrap_ci(
    clusters: list[ClusterCounts],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    point_p, point_r, point_f1 = _pooled_prf(clusters)
    n = len(clusters)
    if n == 0:
        return {
            "n_clusters": 0,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
            "precision": {"point": point_p, "ci_lo": point_p, "ci_hi": point_p},
            "recall": {"point": point_r, "ci_lo": point_r, "ci_hi": point_r},
            "f1": {"point": point_f1, "ci_lo": point_f1, "ci_hi": point_f1},
        }
    rng = random.Random(seed)
    ps: list[float] = []
    rs: list[float] = []
    f1s: list[float] = []
    for _ in range(n_resamples):
        sample = [clusters[rng.randrange(n)] for _ in range(n)]
        p, r, resample_f1 = _pooled_prf(sample)
        ps.append(p)
        rs.append(r)
        f1s.append(resample_f1)
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    return {
        "n_clusters": n,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "seed": seed,
        "precision": {"point": point_p, "ci_lo": _percentile(ps, lo_q), "ci_hi": _percentile(ps, hi_q)},
        "recall": {"point": point_r, "ci_lo": _percentile(rs, lo_q), "ci_hi": _percentile(rs, hi_q)},
        "f1": {"point": point_f1, "ci_lo": _percentile(f1s, lo_q), "ci_hi": _percentile(f1s, hi_q)},
    }


def flip_rate(runs: list[dict[str, bool]]) -> float:
    if not runs:
        return 0.0
    all_refs: set[str] = set()
    for run in runs:
        all_refs |= run.keys()
    if not all_refs:
        return 0.0
    n_runs = len(runs)
    flips = 0
    for ref in all_refs:
        values = [run[ref] for run in runs if ref in run]
        if len(values) != n_runs:

            flips += 1
            continue
        if len(set(values)) > 1:
            flips += 1
    return flips / len(all_refs)


def fleiss_kappa_binary(runs: list[dict[str, bool]]) -> float:
    if len(runs) < 2:
        raise ValueError(f"fleiss_kappa_binary requires at least 2 runs (raters), got {len(runs)}")
    common_refs: set[str] = set(runs[0])
    for run in runs[1:]:
        common_refs &= set(run)
    if not common_refs:
        return FLEISS_KAPPA_VACUOUS_AGREEMENT
    n = len(runs)
    subjects = sorted(common_refs)
    n_subjects = len(subjects)
    total_true = 0
    p_i_sum = 0.0
    for ref in subjects:
        n_true = sum(1 for run in runs if run[ref])
        n_false = n - n_true
        total_true += n_true
        p_i_sum += (n_true * n_true + n_false * n_false - n) / (n * (n - 1))
    p_bar = p_i_sum / n_subjects
    p_true = total_true / (n_subjects * n)
    p_false = 1.0 - p_true
    p_bar_e = p_true * p_true + p_false * p_false
    denominator = 1.0 - p_bar_e
    if abs(denominator) < DEGENERATE_DENOMINATOR_EPSILON:
        return FLEISS_KAPPA_VACUOUS_AGREEMENT
    return (p_bar - p_bar_e) / denominator


def _binomial_cdf_le(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if n <= MCNEMAR_EXACT_MAX_DISCORDANT:
        return min(1.0, 2.0 * _binomial_cdf_le(min(b, c), n))
    chi2 = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2.0))


def paired_permutation_pvalue(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    n_resamples: int = DEFAULT_PERMUTATION_RESAMPLES,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> float:
    n = len(a_correct)
    if n != len(b_correct):
        raise ValueError(f"a_correct and b_correct must be the same length, got {n} and {len(b_correct)}")
    if n == 0:
        return 1.0
    observed = sum(a_correct) / n - sum(b_correct) / n
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        stat = 0.0
        for a_val, b_val in zip(a_correct, b_correct, strict=True):
            if rng.random() < 0.5:
                a_val, b_val = b_val, a_val
            stat += (1.0 if a_val else 0.0) - (1.0 if b_val else 0.0)
        stat /= n
        if abs(stat) >= abs(observed) - DEGENERATE_DENOMINATOR_EPSILON:
            at_least_as_extreme += 1
    return (at_least_as_extreme + PERMUTATION_SMOOTHING) / (n_resamples + PERMUTATION_SMOOTHING)


def bootstrap_delta_f1_ci(
    clusters_a: Sequence[ClusterCounts],
    clusters_b: Sequence[ClusterCounts],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    if len(clusters_a) != len(clusters_b):
        raise ValueError(
            "clusters_a and clusters_b must be sheet-index-aligned (same length), "
            f"got {len(clusters_a)} and {len(clusters_b)}"
        )
    n = len(clusters_a)
    _, _, f1_a = _pooled_prf(clusters_a)
    _, _, f1_b = _pooled_prf(clusters_b)
    point_delta = f1_a - f1_b
    if n == 0:
        return {
            "n_clusters": 0,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
            "delta_f1": {"point": point_delta, "ci_lo": point_delta, "ci_hi": point_delta},
        }
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_a = [clusters_a[i] for i in idx]
        sample_b = [clusters_b[i] for i in idx]
        _, _, fa = _pooled_prf(sample_a)
        _, _, fb = _pooled_prf(sample_b)
        deltas.append(fa - fb)
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    return {
        "n_clusters": n,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "seed": seed,
        "delta_f1": {"point": point_delta, "ci_lo": _percentile(deltas, lo_q), "ci_hi": _percentile(deltas, hi_q)},
    }


def cost_of_pass(cost_usd: float, metric_value: float) -> float:
    if cost_usd < 0:
        raise ValueError(f"cost_usd must be non-negative, got {cost_usd}")
    if metric_value <= 0:
        return math.inf
    return cost_usd / metric_value


def per_dollar(metric_value: float, cost_usd: float) -> float:
    if cost_usd < 0:
        raise ValueError(f"cost_usd must be non-negative, got {cost_usd}")
    if cost_usd == 0:
        return math.inf if metric_value > 0 else 0.0
    return metric_value / cost_usd


def extended_metrics(
    per_sheet_counts: Sequence[ClusterCounts],
    *,
    tn: int | None = None,
    runs: Sequence[Mapping[str, bool]] | None = None,
    cost_usd: float | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    clusters = list(per_sheet_counts)
    tp = sum(c.tp for c in clusters)
    fp = sum(c.fp for c in clusters)
    fn = sum(c.fn for c in clusters)
    precision, recall, f1_value = precision_recall_f1(tp, fp, fn)

    result: dict[str, object] = {
        "pooled": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1_value},
        "f2": f2(precision, recall),
        "wilson_recall": dict(zip(("lo", "hi"), wilson_interval(tp, tp + fn), strict=True)),
        "wilson_precision": dict(zip(("lo", "hi"), wilson_interval(tp, tp + fp), strict=True)),
        "cluster_bootstrap": cluster_bootstrap_ci(clusters, n_resamples=n_resamples, alpha=alpha, seed=seed),
        "cost_of_pass_citation": COST_OF_PASS_CITATION,
    }

    if tn is not None:
        result["mcc"] = mcc(tp, fp, fn, tn)
        result["balanced_accuracy"] = balanced_accuracy(tp, fp, fn, tn)
        result["youdens_j"] = youdens_j(tp, fp, fn, tn)
    else:
        result["mcc"] = None
        result["balanced_accuracy"] = None
        result["youdens_j"] = None
        result["mcc_note"] = (
            "tn not supplied; MCC/balanced_accuracy/youdens_j need a pooled true-negative "
            "count that score.py's per-sheet tp/fp/fn breakdown does not carry"
        )

    if runs is not None:
        runs_list = [dict(run) for run in runs]
        result["flip_rate"] = flip_rate(runs_list)
        result["fleiss_kappa"] = fleiss_kappa_binary(runs_list)

    if cost_usd is not None:
        result["cost_of_pass"] = {
            "f1": cost_of_pass(cost_usd, f1_value),
            "recall": cost_of_pass(cost_usd, recall),
        }
        result["per_dollar"] = {
            "f1": per_dollar(f1_value, cost_usd),
            "recall": per_dollar(recall, cost_usd),
        }

    return result
