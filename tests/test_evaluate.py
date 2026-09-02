import numpy as np

from riskops.evaluate import (
    alerts_per_10k, business_case_sweep, precision_at_k, pr_auc, recall_at_budget,
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
    df = business_case_sweep(Y_TRUE, Y_SCORE, n_days=1, capacities=(2, 5))
    assert list(df["daily_capacity"]) == [2, 5]
    assert (df["recall"] <= 1.0).all() and (df["recall"] >= 0.0).all()
