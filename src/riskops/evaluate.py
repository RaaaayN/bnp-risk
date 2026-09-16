"""Metriques metier pour la triage AML (pas d'accuracy: la classe positive est
tres rare, l'accuracy serait proche de 100% meme pour un modele inutile)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


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
    de test (approx: budget total / volume total)."""
    y_score = np.asarray(y_score)
    total_budget = daily_budget * n_days
    total_budget = min(total_budget, len(y_score))
    if total_budget <= 0:
        return float(y_score.max()) + 1e-9
    sorted_scores = np.sort(y_score)[::-1]
    return float(sorted_scores[total_budget - 1])


def business_case_sweep(y_true, y_score, n_days: int, capacities=(5, 10, 20, 50, 100)) -> pd.DataFrame:
    """Pour chaque capacite d'investigation quotidienne, calcule le seuil
    correspondant, le rappel obtenu et le nombre d'alertes/jour."""
    rows = []
    y_true = np.asarray(y_true)
    total_positives = int(y_true.sum())
    for cap in capacities:
        thr = threshold_for_budget(y_score, cap, n_days)
        recall = recall_at_budget(y_true, y_score, cap * n_days)
        precision = precision_at_k(y_true, y_score, cap * n_days)
        rows.append({
            "daily_capacity": cap,
            "threshold": round(thr, 4),
            "alerts_per_day": cap,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "positives_caught": round(recall * total_positives),
            "total_positives": total_positives,
        })
    return pd.DataFrame(rows)


def false_positive_reduction_at_equal_recall(y_true, score_a, score_b, target_recall: float) -> dict:
    """Compare deux modeles (ex: LogReg vs XGBoost): a rappel egal, combien
    d'alertes (faux positifs) chacun genere-t-il ?"""
    y_true = np.asarray(y_true)

    def alerts_for_recall(score):
        precision, recall, thresh = precision_recall_curve(y_true, score)
        idx = np.argmin(np.abs(recall - target_recall))
        thr = thresh[max(idx - 1, 0)] if idx > 0 else 0.0
        n_alerts = int((np.asarray(score) >= thr).sum())
        n_fp = n_alerts - int(y_true.sum() * recall[idx])
        return n_alerts, max(n_fp, 0), float(recall[idx])

    alerts_a, fp_a, recall_a = alerts_for_recall(score_a)
    alerts_b, fp_b, recall_b = alerts_for_recall(score_b)
    reduction = 0.0 if fp_a == 0 else (fp_a - fp_b) / fp_a
    return {
        "model_a_alerts": alerts_a, "model_a_fp": fp_a, "model_a_recall": recall_a,
        "model_b_alerts": alerts_b, "model_b_fp": fp_b, "model_b_recall": recall_b,
        "fp_reduction_b_vs_a": round(reduction, 4),
    }
