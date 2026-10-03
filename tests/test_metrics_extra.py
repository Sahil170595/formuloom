from __future__ import annotations

import math

import pytest

from formuloom.metrics_extra import (
    ClusterCounts,
    balanced_accuracy,
    bootstrap_delta_f1_ci,
    cluster_bootstrap_ci,
    cost_of_pass,
    extended_metrics,
    f2,
    fbeta,
    fleiss_kappa_binary,
    flip_rate,
    mcc,
    mcnemar,
    paired_permutation_pvalue,
    per_dollar,
    wilson_interval,
    youdens_j,
)


def test_mcc_worked_2x2() -> None:

    assert mcc(tp=3, fp=1, fn=0, tn=4) == pytest.approx(math.sqrt(0.6))


def test_mcc_perfect_classifier() -> None:
    assert mcc(tp=10, fp=0, fn=0, tn=10) == pytest.approx(1.0)


def test_mcc_worst_classifier() -> None:
    assert mcc(tp=0, fp=10, fn=10, tn=0) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "tp,fp,fn,tn",
    [
        (0, 0, 5, 10),
        (5, 10, 0, 0),
        (0, 5, 0, 10),
        (5, 0, 10, 0),
    ],
)
def test_mcc_zero_margin_guard(tp: int, fp: int, fn: int, tn: int) -> None:
    assert mcc(tp, fp, fn, tn) == 0.0


def test_fbeta_worked_example() -> None:

    assert fbeta(precision=0.5, recall=0.8, beta=2.0) == pytest.approx(5.0 / 7.0)


def test_f2_matches_fbeta_beta_2() -> None:
    assert f2(0.5, 0.8) == pytest.approx(fbeta(0.5, 0.8, 2.0))


def test_fbeta_recall_weighted_more_than_f1() -> None:

    p, r = 0.2, 0.9
    f1_equiv = fbeta(p, r, 1.0)
    assert f2(p, r) > f1_equiv


def test_fbeta_zero_guard() -> None:
    assert fbeta(0.0, 0.0, 2.0) == 0.0


def test_balanced_accuracy_worked_example() -> None:

    assert balanced_accuracy(tp=8, fp=2, fn=2, tn=8) == pytest.approx(0.8)
    assert youdens_j(tp=8, fp=2, fn=2, tn=8) == pytest.approx(0.6)


def test_balanced_accuracy_relation_to_youdens_j() -> None:
    tp, fp, fn, tn = 7, 3, 1, 9
    ba = balanced_accuracy(tp, fp, fn, tn)
    j = youdens_j(tp, fp, fn, tn)
    assert ba == pytest.approx((j + 1.0) / 2.0)


def test_balanced_accuracy_zero_actual_positives_guard() -> None:

    assert balanced_accuracy(tp=0, fp=3, fn=0, tn=7) == pytest.approx(0.35)
    assert youdens_j(tp=0, fp=3, fn=0, tn=7) == pytest.approx(-0.3)


def test_balanced_accuracy_zero_actual_negatives_guard() -> None:

    assert balanced_accuracy(tp=4, fp=0, fn=1, tn=0) == pytest.approx(0.4)


def test_wilson_all_successes_z1() -> None:

    lo, hi = wilson_interval(1, 1, z=1.0)
    assert (lo, hi) == pytest.approx((0.5, 1.0))


def test_wilson_all_failures_z1() -> None:

    lo, hi = wilson_interval(0, 1, z=1.0)
    assert (lo, hi) == pytest.approx((0.0, 0.5))


def test_wilson_quarter_z1() -> None:

    lo, hi = wilson_interval(1, 4, z=1.0)
    assert (lo, hi) == pytest.approx((0.1, 0.5))


def test_wilson_contains_point_estimate() -> None:
    lo, hi = wilson_interval(8, 10, z=1.96)
    assert lo <= 0.8 <= hi


def test_wilson_bounds_within_unit_interval() -> None:
    lo, hi = wilson_interval(9, 10, z=1.96)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_zero_total_guard() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_narrows_with_more_data_same_proportion() -> None:
    lo_small, hi_small = wilson_interval(1, 2)
    lo_big, hi_big = wilson_interval(50, 100)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def _homogeneous_clusters(n: int = 10) -> list[ClusterCounts]:
    return [ClusterCounts(sheet=f"S{i}", tp=8, fp=2, fn=2) for i in range(n)]


def _heterogeneous_clusters() -> list[ClusterCounts]:
    perfect = [ClusterCounts(sheet=f"P{i}", tp=10, fp=0, fn=0) for i in range(5)]
    terrible = [ClusterCounts(sheet=f"T{i}", tp=0, fp=10, fn=10) for i in range(5)]
    return perfect + terrible


def _granular_clusters() -> list[ClusterCounts]:

    return [ClusterCounts(sheet=f"S{i}", tp=i + 1, fp=max(1, 10 - i), fn=(i % 4) + 1) for i in range(10)]


def test_cluster_bootstrap_deterministic_same_seed() -> None:
    clusters = _heterogeneous_clusters()
    result_a = cluster_bootstrap_ci(clusters, n_resamples=200, seed=13)
    result_b = cluster_bootstrap_ci(clusters, n_resamples=200, seed=13)
    assert result_a == result_b


def test_cluster_bootstrap_different_seed_can_differ() -> None:
    clusters = _granular_clusters()
    result_a = cluster_bootstrap_ci(clusters, n_resamples=200, seed=13)
    result_b = cluster_bootstrap_ci(clusters, n_resamples=200, seed=99)
    assert result_a["f1"] != result_b["f1"]


def test_cluster_bootstrap_ci_contains_point_estimate() -> None:
    clusters = _heterogeneous_clusters()
    result = cluster_bootstrap_ci(clusters, n_resamples=1000, seed=13)
    for key in ("precision", "recall", "f1"):
        block = result[key]
        assert isinstance(block, dict)
        assert block["ci_lo"] <= block["point"] <= block["ci_hi"]


def test_cluster_bootstrap_wider_for_heterogeneous_clusters() -> None:
    homogeneous = cluster_bootstrap_ci(_homogeneous_clusters(), n_resamples=1000, seed=13)
    heterogeneous = cluster_bootstrap_ci(_heterogeneous_clusters(), n_resamples=1000, seed=13)
    homo_f1 = homogeneous["f1"]
    hetero_f1 = heterogeneous["f1"]
    assert isinstance(homo_f1, dict) and isinstance(hetero_f1, dict)
    homo_width = homo_f1["ci_hi"] - homo_f1["ci_lo"]
    hetero_width = hetero_f1["ci_hi"] - hetero_f1["ci_lo"]

    assert homo_width == pytest.approx(0.0, abs=1e-9)
    assert hetero_width > homo_width


def test_cluster_bootstrap_empty_clusters_guard() -> None:
    result = cluster_bootstrap_ci([], n_resamples=100, seed=13)
    assert result["n_clusters"] == 0
    f1_block = result["f1"]
    assert isinstance(f1_block, dict)
    assert f1_block["point"] == f1_block["ci_lo"] == f1_block["ci_hi"] == 1.0


def test_flip_rate_no_flips() -> None:
    runs = [{"A1": True, "A2": False}, {"A1": True, "A2": False}, {"A1": True, "A2": False}]
    assert flip_rate(runs) == 0.0


def test_flip_rate_all_flip() -> None:
    runs = [{"A1": True}, {"A1": False}]
    assert flip_rate(runs) == 1.0


def test_flip_rate_partial() -> None:
    runs = [
        {"A1": True, "A2": True},
        {"A1": True, "A2": False},
    ]

    assert flip_rate(runs) == pytest.approx(0.5)


def test_flip_rate_missing_ref_in_one_run_counts_as_flip() -> None:

    runs = [{"A1": True, "A2": True}, {"A1": True}]
    assert flip_rate(runs) == pytest.approx(0.5)


def test_flip_rate_empty_runs_guard() -> None:
    assert flip_rate([]) == 0.0
    assert flip_rate([{}, {}]) == 0.0


def test_fleiss_kappa_hand_worked_example() -> None:

    runs = [
        {"r1": True, "r2": False, "r3": True, "r4": False},
        {"r1": True, "r2": False, "r3": True, "r4": True},
        {"r1": True, "r2": False, "r3": False, "r4": False},
    ]
    assert fleiss_kappa_binary(runs) == pytest.approx(1.0 / 3.0)


def test_fleiss_kappa_perfect_agreement() -> None:
    runs = [{"r1": True, "r2": False}, {"r1": True, "r2": False}, {"r1": True, "r2": False}]
    assert fleiss_kappa_binary(runs) == pytest.approx(1.0)


def test_fleiss_kappa_only_common_refs() -> None:

    runs = [{"r1": True, "r2": False, "r3": True}, {"r1": True, "r2": False}]
    assert fleiss_kappa_binary(runs) == pytest.approx(1.0)


def test_fleiss_kappa_no_common_refs_guard() -> None:
    runs = [{"r1": True}, {"r2": True}]
    assert fleiss_kappa_binary(runs) == 1.0


def test_fleiss_kappa_requires_at_least_two_runs() -> None:
    with pytest.raises(ValueError):
        fleiss_kappa_binary([{"r1": True}])


def test_mcnemar_exact_known_binomial() -> None:

    assert mcnemar(1, 9) == pytest.approx(11.0 / 512.0)


def test_mcnemar_symmetric_in_b_c() -> None:
    assert mcnemar(1, 9) == pytest.approx(mcnemar(9, 1))


def test_mcnemar_no_discordant_pairs_guard() -> None:
    assert mcnemar(0, 0) == 1.0


def test_mcnemar_equal_discordant_pairs_not_significant() -> None:
    assert mcnemar(5, 5) == pytest.approx(1.0)


def test_mcnemar_chi_square_branch_matches_erfc_closed_form() -> None:

    b, c = 10, 40
    n = b + c
    chi2 = (abs(b - c) - 1) ** 2 / n
    expected = math.erfc(math.sqrt(chi2 / 2.0))
    assert mcnemar(b, c) == pytest.approx(expected)
    assert mcnemar(b, c) < 0.001


def test_paired_permutation_a_dominates_is_significant() -> None:
    a_correct = [True] * 20
    b_correct = [False] * 20
    p = paired_permutation_pvalue(a_correct, b_correct, n_resamples=2000, seed=13)
    assert p < 0.05


def test_paired_permutation_identical_is_not_significant() -> None:
    a_correct = [True, False, True, False, True]
    b_correct = [True, False, True, False, True]
    p = paired_permutation_pvalue(a_correct, b_correct, n_resamples=2000, seed=13)
    assert p == pytest.approx(1.0)


def test_paired_permutation_deterministic_same_seed() -> None:
    a_correct = [True, True, False, True, False, False, True]
    b_correct = [False, True, True, True, False, True, False]
    p1 = paired_permutation_pvalue(a_correct, b_correct, n_resamples=500, seed=13)
    p2 = paired_permutation_pvalue(a_correct, b_correct, n_resamples=500, seed=13)
    assert p1 == p2


def test_paired_permutation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_permutation_pvalue([True, False], [True], n_resamples=10, seed=1)


def test_paired_permutation_empty_guard() -> None:
    assert paired_permutation_pvalue([], [], n_resamples=10, seed=1) == 1.0


def test_bootstrap_delta_f1_ci_identical_clusters_zero_delta() -> None:
    clusters = _heterogeneous_clusters()
    result = bootstrap_delta_f1_ci(clusters, clusters, n_resamples=200, seed=13)
    delta_block = result["delta_f1"]
    assert isinstance(delta_block, dict)
    assert delta_block["point"] == pytest.approx(0.0)
    assert delta_block["ci_lo"] <= 0.0 <= delta_block["ci_hi"]


def test_bootstrap_delta_f1_ci_a_strictly_better() -> None:
    clusters_a = [ClusterCounts(sheet=f"S{i}", tp=10, fp=0, fn=0) for i in range(10)]
    clusters_b = [ClusterCounts(sheet=f"S{i}", tp=0, fp=10, fn=10) for i in range(10)]
    result = bootstrap_delta_f1_ci(clusters_a, clusters_b, n_resamples=200, seed=13)
    delta_block = result["delta_f1"]
    assert isinstance(delta_block, dict)
    assert delta_block["point"] == pytest.approx(1.0)
    assert delta_block["ci_lo"] > 0.0


def test_bootstrap_delta_f1_ci_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_delta_f1_ci(_homogeneous_clusters(3), _homogeneous_clusters(4))


def test_bootstrap_delta_f1_ci_deterministic_same_seed() -> None:
    clusters_a = _heterogeneous_clusters()
    clusters_b = _homogeneous_clusters()
    r1 = bootstrap_delta_f1_ci(clusters_a, clusters_b, n_resamples=200, seed=13)
    r2 = bootstrap_delta_f1_ci(clusters_a, clusters_b, n_resamples=200, seed=13)
    assert r1 == r2


def test_cost_of_pass_worked_example() -> None:
    assert cost_of_pass(cost_usd=10.0, metric_value=0.5) == pytest.approx(20.0)


def test_cost_of_pass_zero_metric_guard() -> None:
    assert cost_of_pass(cost_usd=5.0, metric_value=0.0) == math.inf


def test_cost_of_pass_negative_cost_raises() -> None:
    with pytest.raises(ValueError):
        cost_of_pass(cost_usd=-1.0, metric_value=0.5)


def test_per_dollar_worked_example() -> None:
    assert per_dollar(metric_value=0.5, cost_usd=10.0) == pytest.approx(0.05)


def test_per_dollar_zero_cost_positive_metric_guard() -> None:
    assert per_dollar(metric_value=0.5, cost_usd=0.0) == math.inf


def test_per_dollar_zero_cost_zero_metric_guard() -> None:
    assert per_dollar(metric_value=0.0, cost_usd=0.0) == 0.0


def test_per_dollar_negative_cost_raises() -> None:
    with pytest.raises(ValueError):
        per_dollar(metric_value=0.5, cost_usd=-2.0)


def test_extended_metrics_minimal_shape() -> None:
    clusters = _heterogeneous_clusters()
    result = extended_metrics(clusters, seed=13, n_resamples=50)
    assert set(result.keys()) >= {
        "pooled",
        "f2",
        "balanced_accuracy",
        "youdens_j",
        "mcc",
        "mcc_note",
        "wilson_recall",
        "wilson_precision",
        "cluster_bootstrap",
        "cost_of_pass_citation",
    }
    assert result["mcc"] is None
    assert result["balanced_accuracy"] is None
    assert result["youdens_j"] is None
    assert "flip_rate" not in result
    assert "cost_of_pass" not in result


def test_extended_metrics_with_tn_computes_mcc_family() -> None:
    clusters = [ClusterCounts(sheet="S1", tp=8, fp=2, fn=2)]
    result = extended_metrics(clusters, tn=8, seed=13, n_resamples=50)
    assert result["mcc"] is not None
    assert result["balanced_accuracy"] == pytest.approx(0.8)
    assert result["youdens_j"] == pytest.approx(0.6)
    assert "mcc_note" not in result


def test_extended_metrics_with_runs_adds_consistency_keys() -> None:
    clusters = [ClusterCounts(sheet="S1", tp=1, fp=0, fn=0)]
    runs = [{"A1": True, "A2": False}, {"A1": True, "A2": False}]
    result = extended_metrics(clusters, runs=runs, seed=13, n_resamples=50)
    assert result["flip_rate"] == 0.0
    assert result["fleiss_kappa"] == pytest.approx(1.0)


def test_extended_metrics_with_cost_adds_efficiency_keys() -> None:
    clusters = [ClusterCounts(sheet="S1", tp=8, fp=2, fn=2)]
    result = extended_metrics(clusters, cost_usd=4.0, seed=13, n_resamples=50)
    cost_block = result["cost_of_pass"]
    per_dollar_block = result["per_dollar"]
    assert isinstance(cost_block, dict) and isinstance(per_dollar_block, dict)
    assert cost_block["f1"] > 0.0
    assert per_dollar_block["f1"] > 0.0


def test_extended_metrics_deterministic_same_seed() -> None:
    clusters = _heterogeneous_clusters()
    runs = [{"A1": True, "A2": False}, {"A1": True, "A2": True}]
    r1 = extended_metrics(clusters, tn=20, runs=runs, cost_usd=3.0, seed=13, n_resamples=100)
    r2 = extended_metrics(clusters, tn=20, runs=runs, cost_usd=3.0, seed=13, n_resamples=100)
    assert r1 == r2
