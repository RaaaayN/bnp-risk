import numpy as np

from riskops.evaluate import (
    alerts_per_10k,
    bootstrap_pr_auc_comparison,
    business_case_sweep,
    false_positive_reduction_at_equal_recall,
    metrics_at_threshold,
    pr_auc,
    precision_at_k,
    recall_at_budget,
    threshold_for_budget,
)

Y_TRUE = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1])
Y_SCORE = np.array([0.1, 0.2, 0.05, 0.3, 0.9, 0.15, 0.8, 0.4, 0.25, 0.6])


def test_precision_at_k_perfect_ranking():
    # top-3 scores sont 0.9, 0.8, 0.6 -> tous positifs
    assert precision_at_k(Y_TRUE, Y_SCORE, k=3) == 1.0


def test_recall_at_budget_full_budget_gives_full_recall():
    assert recall_at_budget(Y_TRUE, Y_SCORE, budget=len(Y_TRUE)) == 1.0


def test_recall_at_budget_zero_positives_returns_zero():
    y_true = np.zeros(5)
    assert recall_at_budget(y_true, np.random.rand(5), budget=3) == 0.0


def test_pr_auc_bounds():
    score = pr_auc(Y_TRUE, Y_SCORE)
    assert 0.0 <= score <= 1.0


def test_alerts_per_10k_monotonic_with_threshold():
    low = alerts_per_10k(Y_SCORE, threshold=0.1)
    high = alerts_per_10k(Y_SCORE, threshold=0.8)
    assert low >= high


def test_threshold_for_budget_matches_kth_score():
    thr = threshold_for_budget(Y_SCORE, daily_budget=3, n_days=1)
    assert thr == sorted(Y_SCORE, reverse=True)[2]


def test_business_case_sweep_output_shape():
    evaluation_score = Y_SCORE - 0.1
    df = business_case_sweep(
        Y_SCORE, 1, Y_TRUE, evaluation_score, 1, capacities=(2, 5)
    )
    assert list(df["daily_capacity"]) == [2, 5]
    assert (df["test_recall"] <= 1.0).all() and (df["test_recall"] >= 0.0).all()
    assert not (df["test_alerts_per_day"] == df["daily_capacity"]).all()


def test_metrics_at_threshold_does_not_recalibrate_on_evaluation_scores():
    metrics = metrics_at_threshold(Y_TRUE, Y_SCORE, threshold=0.8, n_days=2)
    assert metrics["alerts"] == 2
    assert metrics["alerts_per_day"] == 1.0
    assert metrics["recall"] == 2 / 3


def test_false_positive_comparison_uses_the_same_positive_target():
    score_b = np.array([0.01, 0.02, 0.03, 0.04, 0.7, 0.05, 0.9, 0.06, 0.08, 0.8])
    comparison = false_positive_reduction_at_equal_recall(
        Y_TRUE, Y_SCORE, score_b, target_recall=0.5
    )
    assert comparison["target_positives"] == 2
    assert comparison["achieved_recall"] == 2 / 3
    assert "model_a_recall" not in comparison


def test_cluster_bootstrap_returns_paired_confidence_interval():
    groups = np.repeat(np.arange(5), 2)
    comparison = bootstrap_pr_auc_comparison(
        Y_TRUE, Y_SCORE - 0.05, Y_SCORE, groups, n_bootstrap=50, seed=1
    )
    assert comparison["valid_resamples"] > 0
    assert len(comparison["difference_b_minus_a_95pct_ci"]) == 2
