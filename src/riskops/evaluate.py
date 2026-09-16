"""Metriques metier pour la triage AML (pas d'accuracy: la classe positive est
tres rare, l'accuracy serait proche de 100% meme pour un modele inutile)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def bootstrap_pr_auc_comparison(
    y_true, score_a, score_b, groups, n_bootstrap: int = 400, seed: int = 42
) -> dict:
    """IC bootstrap par compte pour deux PR-AUC et leur différence appariée.

    Les lignes d'un épisode ne sont pas indépendantes. Ré-échantillonner les
    comptes, plutôt que les transactions, conserve cette dépendance dans
    l'estimation de l'incertitude.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    auc_a, auc_b, differences = [], [], []

    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_rows = np.concatenate([rows_by_group[group] for group in sampled_groups])
        sampled_y = y_true[sampled_rows]
        if sampled_y.min() == sampled_y.max():
            continue
        value_a = average_precision_score(sampled_y, score_a[sampled_rows])
        value_b = average_precision_score(sampled_y, score_b[sampled_rows])
        auc_a.append(value_a)
        auc_b.append(value_b)
        differences.append(value_b - value_a)

    if not differences:
        raise ValueError("Impossible de calculer un bootstrap contenant les deux classes")

    def interval(values):
        low, high = np.quantile(values, [0.025, 0.975])
        return [round(float(low), 4), round(float(high), 4)]

    return {
        "model_a_95pct_ci": interval(auc_a),
        "model_b_95pct_ci": interval(auc_b),
        "difference_b_minus_a_95pct_ci": interval(differences),
        "valid_resamples": len(differences),
    }


def precision_at_k(y_true, y_score, k: int) -> float:
    k = min(k, len(y_score))
    order = np.argsort(-np.asarray(y_score))[:k]
    return float(np.asarray(y_true)[order].sum() / k) if k > 0 else 0.0


def recall_at_budget(y_true, y_score, budget: int) -> float:
    """Rappel obtenu si on ne peut investiguer que `budget` alertes (les scores
    les plus eleves)."""
    y_true = np.asarray(y_true)
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    k = min(budget, len(y_score))
    order = np.argsort(-np.asarray(y_score))[:k]
    return float(y_true[order].sum() / total_positives)


def alerts_per_10k(y_score, threshold: float) -> float:
    y_score = np.asarray(y_score)
    n_alerts = (y_score >= threshold).sum()
    return float(n_alerts / len(y_score) * 10_000)


def threshold_for_budget(y_score, daily_budget: int, n_days: int) -> float:
    """Seuil qui produit en moyenne `daily_budget` alertes/jour sur la periode
    de calibration (approx: budget total / volume total)."""
    y_score = np.asarray(y_score)
    total_budget = daily_budget * n_days
    total_budget = min(total_budget, len(y_score))
    if total_budget <= 0:
        return float(y_score.max()) + 1e-9
    sorted_scores = np.sort(y_score)[::-1]
    return float(sorted_scores[total_budget - 1])


def metrics_at_threshold(y_true, y_score, threshold: float, n_days: int) -> dict:
    """Mesure un seuil déjà fixé, sans réordonner ni recalibrer l'évaluation."""
    y_true = np.asarray(y_true)
    selected = np.asarray(y_score) >= threshold
    alerts = int(selected.sum())
    true_positives = int(y_true[selected].sum())
    positives = int(y_true.sum())
    return {
        "alerts": alerts,
        "alerts_per_day": alerts / n_days,
        "precision": true_positives / alerts if alerts else 0.0,
        "recall": true_positives / positives if positives else 0.0,
        "positives_caught": true_positives,
        "total_positives": positives,
    }


def business_case_sweep(
    calibration_scores,
    calibration_n_days: int,
    evaluation_y,
    evaluation_scores,
    evaluation_n_days: int,
    capacities=(5, 10, 20, 50, 100),
) -> pd.DataFrame:
    """Calibre chaque seuil sur validation, puis l'évalue tel quel sur test."""
    rows = []
    for cap in capacities:
        threshold = threshold_for_budget(calibration_scores, cap, calibration_n_days)
        observed = metrics_at_threshold(
            evaluation_y, evaluation_scores, threshold, evaluation_n_days
        )
        rows.append({
            "daily_capacity": cap,
            "threshold": round(threshold, 6),
            "test_alerts": observed["alerts"],
            "test_alerts_per_day": round(observed["alerts_per_day"], 2),
            "test_recall": round(observed["recall"], 4),
            "test_precision": round(observed["precision"], 4),
            "test_positives_caught": observed["positives_caught"],
            "test_total_positives": observed["total_positives"],
        })
    return pd.DataFrame(rows)


def false_positive_reduction_at_equal_recall(y_true, score_a, score_b, target_recall: float) -> dict:
    """Compare le nombre d'alertes requis pour capturer exactement N positifs.

    N est commun aux deux modèles et dérivé du rappel cible. On évite ainsi de
    comparer deux points de courbe dont les rappels réalisés sont différents.
    """
    y_true = np.asarray(y_true)
    total_positives = int(y_true.sum())
    if total_positives == 0:
        raise ValueError("La comparaison requiert au moins un cas positif")
    target_positives = min(max(int(np.ceil(target_recall * total_positives)), 1), total_positives)

    def alerts_to_capture_n_positives(score):
        order = np.argsort(-np.asarray(score), kind="stable")
        cumulative_positives = np.cumsum(y_true[order])
        alerts = int(np.searchsorted(cumulative_positives, target_positives) + 1)
        false_positives = alerts - target_positives
        return alerts, false_positives

    alerts_a, fp_a = alerts_to_capture_n_positives(score_a)
    alerts_b, fp_b = alerts_to_capture_n_positives(score_b)
    reduction = 0.0 if fp_a == 0 else (fp_a - fp_b) / fp_a
    return {
        "target_positives": target_positives,
        "achieved_recall": target_positives / total_positives,
        "model_a_alerts": alerts_a,
        "model_a_fp": fp_a,
        "model_b_alerts": alerts_b,
        "model_b_fp": fp_b,
        "fp_reduction_b_vs_a": round(reduction, 4),
    }
