from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from formuloom.labelmodel import (
    ABSTAIN,
    LABELING_FUNCTIONS,
    LF_ACCURACY_PRIOR_MEAN,
    VOTE_FINAL,
    VOTE_INTERMEDIATE,
    ConformalCalibration,
    LabelableRow,
    _flip_solution,
    apply_labeling_functions,
    calibrate,
    fit_label_model,
    flip_rate,
    lf_aggregation,
    lf_bold_bordered,
    lf_graph_sink,
    lf_input_colored,
    lf_lexicon,
    lf_no_formula_no_lexicon,
    lf_pass_through,
    majority_vote_labels,
    predict_set,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
TASKS = ["synthetic-statement", "synthetic-budget", "synthetic-rollup", "synthetic-projection", "synthetic-summary"]
LABELMODEL_ALL_TASKS_BUDGET_SECONDS = 30.0


def _lcg(seed: int) -> Iterator[float]:
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


@dataclass(frozen=True)
class _FakeRow:

    n_dependents_outside_row: int = 1
    aggregates_range: bool = False
    lexicon_hits: tuple[str, ...] = ()
    input_colored_fraction: float = 0.0
    bold_or_bordered: bool = False
    formula_pattern: str | None = None
    pattern_kind: str = "none"


def _mean_accuracy(model_acc: object) -> float:

    return (model_acc.final_accuracy + model_acc.intermediate_accuracy) / 2  # type: ignore[attr-defined]


def _roc_auc(scores: list[float], labels: list[bool]) -> float:
    pos = sum(1 for label in labels if label)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    sum_pos_ranks = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (sum_pos_ranks - pos * (pos + 1) / 2) / (pos * neg)


def test_lf_graph_sink_fires_only_for_formula_sinks() -> None:
    assert lf_graph_sink(_FakeRow(pattern_kind="r1c1", n_dependents_outside_row=0)) == VOTE_FINAL
    assert lf_graph_sink(_FakeRow(pattern_kind="none", n_dependents_outside_row=0)) == ABSTAIN
    assert lf_graph_sink(_FakeRow(pattern_kind="r1c1", n_dependents_outside_row=3)) == ABSTAIN


def test_lf_aggregation() -> None:
    assert lf_aggregation(_FakeRow(aggregates_range=True)) == VOTE_FINAL
    assert lf_aggregation(_FakeRow(aggregates_range=False)) == ABSTAIN


def test_lf_lexicon() -> None:
    assert lf_lexicon(_FakeRow(lexicon_hits=("total",))) == VOTE_FINAL
    assert lf_lexicon(_FakeRow(lexicon_hits=())) == ABSTAIN


def test_lf_input_colored() -> None:
    assert lf_input_colored(_FakeRow(input_colored_fraction=0.8)) == VOTE_INTERMEDIATE
    assert lf_input_colored(_FakeRow(input_colored_fraction=0.5)) == ABSTAIN
    assert lf_input_colored(_FakeRow(input_colored_fraction=0.2)) == ABSTAIN


def test_lf_no_formula_no_lexicon() -> None:
    assert lf_no_formula_no_lexicon(_FakeRow(pattern_kind="none")) == VOTE_INTERMEDIATE
    assert lf_no_formula_no_lexicon(_FakeRow(pattern_kind="none", bold_or_bordered=True)) == ABSTAIN
    assert lf_no_formula_no_lexicon(_FakeRow(pattern_kind="none", lexicon_hits=("net",))) == ABSTAIN
    assert lf_no_formula_no_lexicon(_FakeRow(pattern_kind="sketch")) == ABSTAIN


def test_lf_bold_bordered() -> None:
    assert lf_bold_bordered(_FakeRow(bold_or_bordered=True, lexicon_hits=("ebitda",))) == VOTE_FINAL
    assert lf_bold_bordered(_FakeRow(bold_or_bordered=True, aggregates_range=True)) == VOTE_FINAL
    assert lf_bold_bordered(_FakeRow(bold_or_bordered=True)) == ABSTAIN
    assert lf_bold_bordered(_FakeRow(bold_or_bordered=False, lexicon_hits=("ebitda",))) == ABSTAIN


def test_lf_pass_through() -> None:
    assert lf_pass_through(_FakeRow(pattern_kind="sketch", formula_pattern="REF")) == VOTE_INTERMEDIATE
    assert lf_pass_through(_FakeRow(pattern_kind="r1c1", formula_pattern="=RC[-1]")) == VOTE_INTERMEDIATE
    assert lf_pass_through(_FakeRow(pattern_kind="r1c1", formula_pattern="=R[-1]C")) == VOTE_INTERMEDIATE
    assert lf_pass_through(_FakeRow(pattern_kind="r1c1", formula_pattern="='Rev'!R26C5")) == VOTE_INTERMEDIATE

    assert lf_pass_through(_FakeRow(pattern_kind="sketch", formula_pattern="REF*REF")) == ABSTAIN
    assert lf_pass_through(_FakeRow(pattern_kind="r1c1", formula_pattern="=RC[-1]+RC[-2]")) == ABSTAIN
    assert lf_pass_through(_FakeRow(pattern_kind="none", formula_pattern=None)) == ABSTAIN


def test_registry_complete_and_typed() -> None:
    assert set(LABELING_FUNCTIONS) == {
        "lf_graph_sink",
        "lf_aggregation",
        "lf_lexicon",
        "lf_input_colored",
        "lf_no_formula_no_lexicon",
        "lf_bold_bordered",
        "lf_pass_through",
    }
    row: LabelableRow = _FakeRow(pattern_kind="r1c1", aggregates_range=True, lexicon_hits=("total",))
    for name, fn in LABELING_FUNCTIONS.items():
        vote = fn(row)
        assert vote in (VOTE_FINAL, VOTE_INTERMEDIATE, ABSTAIN), name


def test_apply_labeling_functions_shape() -> None:
    rows = [_FakeRow(pattern_kind="r1c1", n_dependents_outside_row=0), _FakeRow(input_colored_fraction=0.9)]
    vm = apply_labeling_functions(rows)
    assert vm.lf_names == tuple(LABELING_FUNCTIONS)
    assert len(vm.votes) == 2
    assert all(len(row) == len(LABELING_FUNCTIONS) for row in vm.votes)


def _expert_and_crowd(n: int) -> tuple[list[int], list[list[int]]]:
    gen = _lcg(20260707)
    truth: list[int] = []
    votes: list[list[int]] = []
    for _ in range(n):
        label = 1 if next(gen) < 0.5 else 0
        truth.append(label)
        correct = VOTE_FINAL if label == 1 else VOTE_INTERMEDIATE
        wrong = VOTE_INTERMEDIATE if label == 1 else VOTE_FINAL
        expert = correct if next(gen) < 0.9 else wrong
        weak0 = correct if next(gen) < 0.6 else wrong
        weak1 = correct if next(gen) < 0.6 else wrong
        votes.append([expert, weak0, weak1])
    return truth, votes


def test_em_recovers_ordering_and_beats_majority_vote() -> None:
    truth, votes = _expert_and_crowd(1000)
    names = ["expert", "weak0", "weak1"]
    model = fit_label_model(votes, names)

    expert_acc = _mean_accuracy(model.lf_accuracies["expert"])
    weak_acc = max(_mean_accuracy(model.lf_accuracies["weak0"]), _mean_accuracy(model.lf_accuracies["weak1"]))
    assert expert_acc > weak_acc + 0.05
    assert expert_acc > 0.7
    assert not model.diagnostics.flipped

    em_pred = [1 if p > 0.5 else 0 for p in model.probabilities]
    em_acc = sum(1 for p, t in zip(em_pred, truth, strict=True) if p == t) / len(truth)

    mv = majority_vote_labels(votes)
    mv_pred = [label if label is not None else 0 for label in mv]
    mv_acc = sum(1 for p, t in zip(mv_pred, truth, strict=True) if p == t) / len(truth)

    assert em_acc > mv_acc
    assert em_acc > 0.8


def test_em_all_abstain_row_gets_class_prior() -> None:
    votes = [
        [VOTE_FINAL, VOTE_FINAL],
        [VOTE_FINAL, VOTE_INTERMEDIATE],
        [VOTE_INTERMEDIATE, VOTE_INTERMEDIATE],
        [ABSTAIN, ABSTAIN],
    ]
    model = fit_label_model(votes)
    assert model.diagnostics.n_abstain_all == 1
    assert model.diagnostics.n_conflicts == 1
    assert model.probabilities[3] == pytest.approx(model.class_prior, abs=1e-9)


def test_em_never_firing_lf_keeps_prior_accuracy() -> None:
    votes = [
        [VOTE_FINAL, VOTE_FINAL, ABSTAIN],
        [VOTE_FINAL, VOTE_FINAL, ABSTAIN],
        [VOTE_INTERMEDIATE, VOTE_INTERMEDIATE, ABSTAIN],
        [VOTE_INTERMEDIATE, VOTE_INTERMEDIATE, ABSTAIN],
        [VOTE_FINAL, VOTE_INTERMEDIATE, ABSTAIN],
    ]
    model = fit_label_model(votes, ["a", "b", "never"])
    never = model.lf_accuracies["never"]
    assert never.n_votes == 0
    assert never.final_accuracy == pytest.approx(LF_ACCURACY_PRIOR_MEAN)
    assert never.intermediate_accuracy == pytest.approx(LF_ACCURACY_PRIOR_MEAN)
    assert not model.diagnostics.flipped


def test_em_log_space_stable_on_2000_rows() -> None:
    gen = _lcg(4242)
    votes: list[list[int]] = []
    for _ in range(2000):
        row: list[int] = []
        for _ in range(7):
            u = next(gen)
            row.append(VOTE_FINAL if u < 0.33 else VOTE_INTERMEDIATE if u < 0.66 else ABSTAIN)
        votes.append(row)
    model = fit_label_model(votes)
    assert model.diagnostics.n_rows == 2000
    assert len(model.probabilities) == 2000
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in model.probabilities)


def test_flip_solution_is_involutive() -> None:
    prior, alphas, betas, probs = 0.3, [0.9, 0.6], [0.8, 0.55], [0.7, 0.2, 0.9]
    p1, a1, b1, pr1 = _flip_solution(prior, alphas, betas, probs)
    p2, a2, b2, pr2 = _flip_solution(p1, a1, b1, pr1)
    assert p2 == pytest.approx(prior)
    assert a2 == pytest.approx(alphas)
    assert b2 == pytest.approx(betas)
    assert pr2 == pytest.approx(probs)


def test_fit_empty_matrix_is_safe() -> None:
    model = fit_label_model([])
    assert model.probabilities == ()
    assert model.diagnostics.n_rows == 0
    assert not model.diagnostics.converged


def test_ragged_matrix_raises() -> None:
    with pytest.raises(ValueError, match="ragged"):
        fit_label_model([[VOTE_FINAL, ABSTAIN], [VOTE_FINAL]])


def test_conformal_exact_threshold_n9_alpha_0_2() -> None:

    scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15]
    labels = [True] * 9
    cal = calibrate(scores, labels, alpha=0.2)
    assert cal.n == 9

    assert cal.quantile == pytest.approx(0.75)


def test_conformal_rank_exceeds_n_gives_full_sets() -> None:
    cal = calibrate([0.9, 0.8, 0.7], [True, True, True], alpha=0.01)
    assert cal.quantile == 1.0
    assert predict_set(cal, 0.99) == frozenset({"final", "intermediate"})


def test_conformal_predict_set_cases() -> None:
    cal = ConformalCalibration(quantile=0.75, alpha=0.2, n=9)
    assert predict_set(cal, 0.9) == frozenset({"final"})
    assert predict_set(cal, 0.5) == frozenset({"final", "intermediate"})
    assert predict_set(cal, 0.1) == frozenset({"intermediate"})
    tight = ConformalCalibration(quantile=0.2, alpha=0.8, n=5)
    assert predict_set(tight, 0.5) == frozenset()


def test_conformal_marginal_coverage_holds() -> None:
    gen = _lcg(31337)

    def draw() -> tuple[float, bool]:
        is_final = next(gen) < 0.4
        base = 0.7 if is_final else 0.3
        noise = (next(gen) - 0.5) * 0.4
        return min(max(base + noise, 0.0), 1.0), is_final

    cal_scores: list[float] = []
    cal_labels: list[bool] = []
    for _ in range(300):
        score, label = draw()
        cal_scores.append(score)
        cal_labels.append(label)
    alpha = 0.1
    cal = calibrate(cal_scores, cal_labels, alpha=alpha)

    covered = 0
    total = 0
    for _ in range(300):
        score, label = draw()
        true_class = "final" if label else "intermediate"
        total += 1
        if true_class in predict_set(cal, score):
            covered += 1
    coverage = covered / total
    assert coverage >= (1.0 - alpha) - 0.1


def test_calibrate_input_validation() -> None:
    with pytest.raises(ValueError, match="length"):
        calibrate([0.5, 0.6], [True], alpha=0.1)
    with pytest.raises(ValueError, match="alpha"):
        calibrate([0.5], [True], alpha=1.5)
    with pytest.raises(ValueError, match="at least one"):
        calibrate([], [], alpha=0.1)


def test_flip_rate_trivial_cases() -> None:
    assert flip_rate([]) == 0.0
    assert flip_rate([{"a": True, "b": False}]) == 0.0
    assert flip_rate([{"a": True, "b": False}, {"a": True, "b": False}]) == 0.0
    assert flip_rate([{"a": True, "b": True}, {"a": False, "b": True}]) == 0.5
    assert flip_rate([{"a": True}, {"a": True}, {"a": False}]) == 1.0
    assert flip_rate([{"a": True}, {"a": True, "b": False}]) == 0.0
