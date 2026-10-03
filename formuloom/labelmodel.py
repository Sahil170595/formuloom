from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

VOTE_FINAL = 1

VOTE_INTERMEDIATE = -1

ABSTAIN = 0

LF_INPUT_COLORED_FRACTION_THRESHOLD = 0.5

LF_ACCURACY_PRIOR_MEAN = 0.7

LF_ACCURACY_PRIOR_STRENGTH = 5.0

CLASS_PRIOR_PSEUDO_STRENGTH = 2.0

DEFAULT_PRIOR_ANCHOR = 0.5

DOMAIN_FINAL_PRIOR = 0.30

MAX_EM_ITERATIONS = 100

EM_CONVERGENCE_EPSILON = 1e-4

PROB_FLOOR = 1e-6

INIT_PRIOR_CLAMP = 0.05


class LabelableRow(Protocol):

    @property
    def n_dependents_outside_row(self) -> int: ...
    @property
    def aggregates_range(self) -> bool: ...
    @property
    def lexicon_hits(self) -> Sequence[str]: ...
    @property
    def input_colored_fraction(self) -> float: ...
    @property
    def bold_or_bordered(self) -> bool: ...
    @property
    def formula_pattern(self) -> str | None: ...
    @property
    def pattern_kind(self) -> str: ...


LabelingFunction = Callable[[LabelableRow], int]


def _row_has_formula(row: LabelableRow) -> bool:
    return row.pattern_kind != "none"


def lf_graph_sink(row: LabelableRow) -> int:
    if _row_has_formula(row) and row.n_dependents_outside_row == 0:
        return VOTE_FINAL
    return ABSTAIN


def lf_aggregation(row: LabelableRow) -> int:
    return VOTE_FINAL if row.aggregates_range else ABSTAIN


def lf_lexicon(row: LabelableRow) -> int:
    return VOTE_FINAL if len(row.lexicon_hits) > 0 else ABSTAIN


def lf_input_colored(row: LabelableRow) -> int:
    if row.input_colored_fraction > LF_INPUT_COLORED_FRACTION_THRESHOLD:
        return VOTE_INTERMEDIATE
    return ABSTAIN


def lf_no_formula_no_lexicon(row: LabelableRow) -> int:
    if not _row_has_formula(row) and len(row.lexicon_hits) == 0 and not row.bold_or_bordered:
        return VOTE_INTERMEDIATE
    return ABSTAIN


def lf_bold_bordered(row: LabelableRow) -> int:
    if row.bold_or_bordered and (len(row.lexicon_hits) > 0 or row.aggregates_range):
        return VOTE_FINAL
    return ABSTAIN


_SKETCH_SINGLE_REF = "REF"
_R1C1_SINGLE_CELL_RE = re.compile(r"^=(?:[^!]+!)?R(?:\d+|\[-?\d+\])?C(?:\d+|\[-?\d+\])?$")


def lf_pass_through(row: LabelableRow) -> int:
    pattern = row.formula_pattern
    if pattern is None:
        return ABSTAIN
    kind = row.pattern_kind
    if kind == "sketch" and pattern.strip().lstrip("=") == _SKETCH_SINGLE_REF:
        return VOTE_INTERMEDIATE
    if kind == "r1c1" and _R1C1_SINGLE_CELL_RE.match(pattern) is not None:
        return VOTE_INTERMEDIATE
    return ABSTAIN


LABELING_FUNCTIONS: dict[str, LabelingFunction] = {
    "lf_graph_sink": lf_graph_sink,
    "lf_aggregation": lf_aggregation,
    "lf_lexicon": lf_lexicon,
    "lf_input_colored": lf_input_colored,
    "lf_no_formula_no_lexicon": lf_no_formula_no_lexicon,
    "lf_bold_bordered": lf_bold_bordered,
    "lf_pass_through": lf_pass_through,
}


@dataclass(frozen=True)
class VoteMatrix:

    lf_names: tuple[str, ...]
    votes: list[list[int]]


def apply_labeling_functions(
    rows: Sequence[LabelableRow], lfs: Mapping[str, LabelingFunction] | None = None
) -> VoteMatrix:
    functions = LABELING_FUNCTIONS if lfs is None else lfs
    names = tuple(functions)
    fns = [functions[name] for name in names]
    votes = [[fn(row) for fn in fns] for row in rows]
    return VoteMatrix(lf_names=names, votes=votes)


def majority_vote_label(row_votes: Sequence[int]) -> int | None:
    finals = sum(1 for v in row_votes if v == VOTE_FINAL)
    inters = sum(1 for v in row_votes if v == VOTE_INTERMEDIATE)
    if finals == 0 and inters == 0:
        return None
    if finals > inters:
        return 1
    return 0


def majority_vote_labels(votes: Sequence[Sequence[int]]) -> list[int | None]:
    return [majority_vote_label(row) for row in votes]


@dataclass(frozen=True)
class LFAccuracy:

    final_accuracy: float
    intermediate_accuracy: float
    n_votes: int
    n_final_votes: int
    n_intermediate_votes: int


@dataclass(frozen=True)
class EMDiagnostics:

    n_rows: int
    n_lfs: int
    n_conflicts: int
    n_abstain_all: int
    iterations: int
    converged: bool
    flipped: bool
    max_delta: float


@dataclass(frozen=True)
class LabelModel:

    probabilities: tuple[float, ...]
    class_prior: float
    lf_accuracies: dict[str, LFAccuracy]
    diagnostics: EMDiagnostics


def _clamp_prob(value: float) -> float:
    return min(max(value, PROB_FLOOR), 1.0 - PROB_FLOOR)


def _logsumexp2(a: float, b: float) -> float:
    if a == -math.inf and b == -math.inf:
        return -math.inf
    hi = a if a > b else b
    lo = b if a > b else a
    return hi + math.log1p(math.exp(lo - hi))


def _validate_matrix(votes: Sequence[Sequence[int]]) -> int:
    if not votes:
        return 0
    width = len(votes[0])
    for idx, row in enumerate(votes):
        if len(row) != width:
            raise ValueError(f"ragged vote matrix: row 0 has {width} LFs but row {idx} has {len(row)}")
    return width


def _estep(
    votes: Sequence[Sequence[int]], alphas: Sequence[float], betas: Sequence[float], prior: float
) -> list[float]:
    log_prior1 = math.log(_clamp_prob(prior))
    log_prior0 = math.log(_clamp_prob(1.0 - prior))
    log_alpha = [math.log(_clamp_prob(a)) for a in alphas]
    log_not_alpha = [math.log(_clamp_prob(1.0 - a)) for a in alphas]
    log_beta = [math.log(_clamp_prob(b)) for b in betas]
    log_not_beta = [math.log(_clamp_prob(1.0 - b)) for b in betas]

    probabilities: list[float] = []
    for row in votes:
        ll_final = log_prior1
        ll_inter = log_prior0
        for j, vote in enumerate(row):
            if vote == VOTE_FINAL:
                ll_final += log_alpha[j]
                ll_inter += log_not_beta[j]
            elif vote == VOTE_INTERMEDIATE:
                ll_final += log_not_alpha[j]
                ll_inter += log_beta[j]
        norm = _logsumexp2(ll_inter, ll_final)
        probabilities.append(math.exp(ll_final - norm))
    return probabilities


def _mstep(
    votes: Sequence[Sequence[int]],
    probs: Sequence[float],
    n_lfs: int,
    *,
    prior_anchor: float,
    prior_floor: float,
    prior_ceiling: float,
) -> tuple[float, list[float], list[float]]:
    a0 = LF_ACCURACY_PRIOR_MEAN * LF_ACCURACY_PRIOR_STRENGTH
    b0 = (1.0 - LF_ACCURACY_PRIOR_MEAN) * LF_ACCURACY_PRIOR_STRENGTH

    sum_final = sum(probs)
    prior_pseudo_final = prior_anchor * CLASS_PRIOR_PSEUDO_STRENGTH
    prior = (sum_final + prior_pseudo_final) / (len(probs) + CLASS_PRIOR_PSEUDO_STRENGTH)
    prior = min(max(_clamp_prob(prior), prior_floor), prior_ceiling)

    num_alpha = [a0] * n_lfs
    den_alpha = [a0 + b0] * n_lfs
    num_beta = [a0] * n_lfs
    den_beta = [a0 + b0] * n_lfs
    for i, row in enumerate(votes):
        p_final = probs[i]
        p_inter = 1.0 - p_final
        for j, vote in enumerate(row):
            if vote == VOTE_FINAL:
                num_alpha[j] += p_final
                den_alpha[j] += p_final
                den_beta[j] += p_inter
            elif vote == VOTE_INTERMEDIATE:
                num_beta[j] += p_inter
                den_beta[j] += p_inter
                den_alpha[j] += p_final
    alphas = [num_alpha[j] / den_alpha[j] for j in range(n_lfs)]
    betas = [num_beta[j] / den_beta[j] for j in range(n_lfs)]
    return prior, alphas, betas


def _max_delta(
    prior: float,
    prior_new: float,
    alphas: Sequence[float],
    alphas_new: Sequence[float],
    betas: Sequence[float],
    betas_new: Sequence[float],
) -> float:
    delta = abs(prior_new - prior)
    for old, new in zip(alphas, alphas_new, strict=True):
        delta = max(delta, abs(new - old))
    for old, new in zip(betas, betas_new, strict=True):
        delta = max(delta, abs(new - old))
    return delta


def _flip_solution(
    prior: float, alphas: Sequence[float], betas: Sequence[float], probs: Sequence[float]
) -> tuple[float, list[float], list[float], list[float]]:
    return (
        1.0 - prior,
        [1.0 - b for b in betas],
        [1.0 - a for a in alphas],
        [1.0 - p for p in probs],
    )


def _needs_flip(mv_labels: Sequence[int | None], probs: Sequence[float]) -> bool:
    sum_f = cnt_f = sum_i = cnt_i = 0.0
    for label, prob in zip(mv_labels, probs, strict=True):
        if label == 1:
            sum_f += prob
            cnt_f += 1
        elif label == 0:
            sum_i += prob
            cnt_i += 1
    if cnt_f == 0 or cnt_i == 0:
        return False
    return (sum_f / cnt_f) < (sum_i / cnt_i)


def _count_conflicts_and_abstains(votes: Sequence[Sequence[int]]) -> tuple[int, int]:
    conflicts = 0
    abstains = 0
    for row in votes:
        has_final = any(v == VOTE_FINAL for v in row)
        has_inter = any(v == VOTE_INTERMEDIATE for v in row)
        if has_final and has_inter:
            conflicts += 1
        if not has_final and not has_inter:
            abstains += 1
    return conflicts, abstains


def _initial_prior(mv_labels: Sequence[int | None], prior_floor: float, prior_ceiling: float) -> float:
    definite = [label for label in mv_labels if label is not None]
    prior = sum(definite) / len(definite) if definite else 0.5
    prior = min(max(prior, INIT_PRIOR_CLAMP), 1.0 - INIT_PRIOR_CLAMP)
    return min(max(prior, prior_floor), prior_ceiling)


def fit_label_model(
    votes: Sequence[Sequence[int]],
    lf_names: Sequence[str] | None = None,
    *,
    max_iterations: int = MAX_EM_ITERATIONS,
    epsilon: float = EM_CONVERGENCE_EPSILON,
    prior_anchor: float = DEFAULT_PRIOR_ANCHOR,
    prior_floor: float = PROB_FLOOR,
    prior_ceiling: float = 1.0 - PROB_FLOOR,
) -> LabelModel:
    if not 0.0 < prior_floor <= prior_ceiling < 1.0:
        raise ValueError(f"require 0 < prior_floor ({prior_floor}) <= prior_ceiling ({prior_ceiling}) < 1")
    n_lfs = _validate_matrix(votes)
    n_rows = len(votes)
    names = tuple(lf_names) if lf_names is not None else tuple(f"lf{j}" for j in range(n_lfs))
    if len(names) != n_lfs:
        raise ValueError(f"lf_names has {len(names)} entries but the matrix has {n_lfs} LFs")

    mv_labels = majority_vote_labels(votes)
    prior = _initial_prior(mv_labels, prior_floor, prior_ceiling)
    alphas = [LF_ACCURACY_PRIOR_MEAN] * n_lfs
    betas = [LF_ACCURACY_PRIOR_MEAN] * n_lfs

    iterations = 0
    converged = False
    delta = math.inf
    if n_rows > 0:
        for _ in range(max_iterations):
            iterations += 1
            probs = _estep(votes, alphas, betas, prior)
            prior_new, alphas_new, betas_new = _mstep(
                votes, probs, n_lfs, prior_anchor=prior_anchor, prior_floor=prior_floor, prior_ceiling=prior_ceiling
            )
            delta = _max_delta(prior, prior_new, alphas, alphas_new, betas, betas_new)
            prior, alphas, betas = prior_new, alphas_new, betas_new
            if delta < epsilon:
                converged = True
                break

    probs = _estep(votes, alphas, betas, prior) if n_rows > 0 else []

    flipped = _needs_flip(mv_labels, probs)
    if flipped:
        prior, alphas, betas, probs = _flip_solution(prior, alphas, betas, probs)

    conflicts, abstain_all = _count_conflicts_and_abstains(votes)
    accuracies: dict[str, LFAccuracy] = {}
    for j, name in enumerate(names):
        n_final = sum(1 for row in votes if row[j] == VOTE_FINAL)
        n_inter = sum(1 for row in votes if row[j] == VOTE_INTERMEDIATE)
        accuracies[name] = LFAccuracy(
            final_accuracy=alphas[j],
            intermediate_accuracy=betas[j],
            n_votes=n_final + n_inter,
            n_final_votes=n_final,
            n_intermediate_votes=n_inter,
        )

    diagnostics = EMDiagnostics(
        n_rows=n_rows,
        n_lfs=n_lfs,
        n_conflicts=conflicts,
        n_abstain_all=abstain_all,
        iterations=iterations,
        converged=converged,
        flipped=flipped,
        max_delta=delta if n_rows > 0 else 0.0,
    )
    return LabelModel(
        probabilities=tuple(probs),
        class_prior=prior,
        lf_accuracies=accuracies,
        diagnostics=diagnostics,
    )


FINAL_CLASS = "final"
INTERMEDIATE_CLASS = "intermediate"


@dataclass(frozen=True)
class ConformalCalibration:

    quantile: float
    alpha: float
    n: int


def calibrate(scores: Sequence[float], labels: Sequence[bool], alpha: float) -> ConformalCalibration:
    if len(scores) != len(labels):
        raise ValueError(f"scores ({len(scores)}) and labels ({len(labels)}) differ in length")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ValueError("calibration requires at least one (score, label) pair")

    nonconformity = sorted(
        (1.0 - score) if label else (1.0 - (1.0 - score)) for score, label in zip(scores, labels, strict=True)
    )
    rank = math.ceil((n + 1) * (1.0 - alpha))
    quantile = 1.0 if rank > n else nonconformity[rank - 1]
    return ConformalCalibration(quantile=quantile, alpha=alpha, n=n)


def predict_set(calibration: ConformalCalibration, p_final: float) -> frozenset[str]:
    admitted: set[str] = set()
    if (1.0 - p_final) <= calibration.quantile:
        admitted.add(FINAL_CLASS)
    if p_final <= calibration.quantile:
        admitted.add(INTERMEDIATE_CLASS)
    return frozenset(admitted)


def flip_rate(runs: Sequence[Mapping[str, bool]]) -> float:
    refs: dict[str, bool | None] = {}
    flipped: set[str] = set()
    if len(runs) < 2:
        return 0.0
    for run in runs:
        for ref, label in run.items():
            if ref not in refs:
                refs[ref] = label
            elif refs[ref] != label:
                flipped.add(ref)
    if not refs:
        return 0.0
    return len(flipped) / len(refs)
