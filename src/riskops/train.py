"""Entrainement: split chronologique, baseline LogisticRegression, XGBoost,
explicabilite SHAP, metriques metier. Sauvegarde des artefacts dans models/."""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from riskops.evaluate import (
    alerts_per_10k,
    bootstrap_pr_auc_comparison,
    business_case_sweep,
    false_positive_reduction_at_equal_recall,
    metrics_at_threshold,
    pr_auc,
    threshold_for_budget,
)
from riskops.features import FEATURE_COLUMNS, TRANSACTION_FEATURE_COLUMNS

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DAILY_INVESTIGATION_CAPACITY = 20


def _new_xgboost(pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        early_stopping_rounds=30, random_state=42, n_jobs=1,
    )


def _period_days(df: pd.DataFrame) -> int:
    duration = df["Timestamp"].max() - df["Timestamp"].min()
    return max(int(np.ceil(duration.total_seconds() / 86_400)), 1)


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    df = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    feats_path = ROOT / "data" / "features.parquet"
    df = pd.read_parquet(feats_path)

    train_df, val_df, test_df = chronological_split(df)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Is Laundering"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["Is Laundering"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Is Laundering"]

    n_days_val = _period_days(val_df)
    n_days_test = _period_days(test_df)

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(X_train), y_train)
    lr_val_scores = lr.predict_proba(scaler.transform(X_val))[:, 1]
    lr_test_scores = lr.predict_proba(scaler.transform(X_test))[:, 1]

    # --- Modele boosted trees ---
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    xgb = _new_xgboost(pos_weight)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_scores = xgb.predict_proba(X_val)[:, 1]
    xgb_test_scores = xgb.predict_proba(X_test)[:, 1]

    # Ablation : même algorithme sans aucune feature d'historique. Ce test
    # vérifie que le générateur donne réellement quelque chose à apprendre aux
    # fenêtres de compte, au-delà du montant et du format courants.
    xgb_transaction_only = _new_xgboost(pos_weight)
    xgb_transaction_only.fit(
        train_df[TRANSACTION_FEATURE_COLUMNS], y_train,
        eval_set=[(val_df[TRANSACTION_FEATURE_COLUMNS], y_val)], verbose=False,
    )
    transaction_only_scores = xgb_transaction_only.predict_proba(
        test_df[TRANSACTION_FEATURE_COLUMNS]
    )[:, 1]

    logreg_pr_auc = pr_auc(y_test, lr_test_scores)
    xgboost_pr_auc = pr_auc(y_test, xgb_test_scores)
    logreg_val_pr_auc = pr_auc(y_val, lr_val_scores)
    xgboost_val_pr_auc = pr_auc(y_val, xgb_val_scores)
    transaction_only_pr_auc = pr_auc(y_test, transaction_only_scores)
    uncertainty = bootstrap_pr_auc_comparison(
        y_test, lr_test_scores, xgb_test_scores, test_df["Account"]
    )
    selected_model = "xgboost" if xgboost_val_pr_auc >= logreg_val_pr_auc else "logreg"
    lr_threshold = threshold_for_budget(
        lr_val_scores, DAILY_INVESTIGATION_CAPACITY, n_days_val
    )
    xgb_threshold = threshold_for_budget(
        xgb_val_scores, DAILY_INVESTIGATION_CAPACITY, n_days_val
    )
    lr_operating = metrics_at_threshold(y_test, lr_test_scores, lr_threshold, n_days_test)
    xgb_operating = metrics_at_threshold(y_test, xgb_test_scores, xgb_threshold, n_days_test)

    metrics = {
        "seed": 42,
        "n_days_validation": n_days_val,
        "n_days_test": n_days_test,
        "threshold_calibration": {
            "dataset": "validation",
            "daily_capacity": DAILY_INVESTIGATION_CAPACITY,
        },
        "logreg": {
            "validation_pr_auc": logreg_val_pr_auc,
            "test_pr_auc": logreg_pr_auc,
            "test_pr_auc_95pct_ci": uncertainty["model_a_95pct_ci"],
            "validation_calibrated_threshold": lr_threshold,
            "test_alerts": lr_operating["alerts"],
            "test_alerts_per_day": lr_operating["alerts_per_day"],
            "test_precision_at_fixed_threshold": lr_operating["precision"],
            "test_recall_at_fixed_threshold": lr_operating["recall"],
            "alerts_per_10k_at_p99": alerts_per_10k(lr_test_scores, np.quantile(lr_test_scores, 0.99)),
        },
        "xgboost": {
            "validation_pr_auc": xgboost_val_pr_auc,
            "best_iteration": xgb.best_iteration,
            "test_pr_auc": xgboost_pr_auc,
            "test_pr_auc_95pct_ci": uncertainty["model_b_95pct_ci"],
            "validation_calibrated_threshold": xgb_threshold,
            "test_alerts": xgb_operating["alerts"],
            "test_alerts_per_day": xgb_operating["alerts_per_day"],
            "test_precision_at_fixed_threshold": xgb_operating["precision"],
            "test_recall_at_fixed_threshold": xgb_operating["recall"],
            "alerts_per_10k_at_p99": alerts_per_10k(xgb_test_scores, np.quantile(xgb_test_scores, 0.99)),
        },
        "xgboost_transaction_only_ablation": {
            "pr_auc": transaction_only_pr_auc,
            "history_feature_pr_auc_lift": xgboost_pr_auc - transaction_only_pr_auc,
        },
        "pr_auc_difference_xgb_minus_logreg_95pct_ci": uncertainty[
            "difference_b_minus_a_95pct_ci"
        ],
        "bootstrap_valid_resamples": uncertainty["valid_resamples"],
        "selection_metric": "validation_pr_auc",
        "selected_model": selected_model,
        "fp_reduction_xgb_vs_logreg_at_equal_recall": false_positive_reduction_at_equal_recall(
            y_test, lr_test_scores, xgb_test_scores, target_recall=0.5
        ),
    }

    champion_val_scores = xgb_val_scores if selected_model == "xgboost" else lr_val_scores
    champion_test_scores = xgb_test_scores if selected_model == "xgboost" else lr_test_scores
    champion_threshold = xgb_threshold if selected_model == "xgboost" else lr_threshold
    business_case = business_case_sweep(
        champion_val_scores, n_days_val, y_test, champion_test_scores, n_days_test
    )
    metrics["business_case"] = business_case.to_dict(orient="records")

    # --- SHAP sur le champion sélectionné en validation ---
    xgb_explainer = shap.TreeExplainer(xgb)
    if selected_model == "xgboost":
        champion_bundle = {
            "model": xgb, "scaler": None, "features": FEATURE_COLUMNS,
            "model_name": "xgboost", "threshold": champion_threshold,
        }
        champion_explainer = xgb_explainer
        shap_values = champion_explainer.shap_values(X_test)
    else:
        champion_bundle = {
            "model": lr, "scaler": scaler, "features": FEATURE_COLUMNS,
            "model_name": "logreg", "threshold": champion_threshold,
        }
        champion_explainer = shap.LinearExplainer(lr, scaler.transform(X_train))
        shap_values = champion_explainer.shap_values(scaler.transform(X_test))
    global_importance = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
    )

    joblib.dump({"model": xgb, "features": FEATURE_COLUMNS}, MODELS_DIR / "xgb_model.joblib")
    joblib.dump({"model": lr, "scaler": scaler, "features": FEATURE_COLUMNS}, MODELS_DIR / "logreg_model.joblib")
    joblib.dump(xgb_explainer, MODELS_DIR / "shap_explainer.joblib")
    # L'API charge le champion déterminé par la métrique primaire. Les anciens
    # artefacts nommés restent écrits pour inspection et compatibilité.
    joblib.dump(champion_bundle, MODELS_DIR / "champion_model.joblib")
    joblib.dump(champion_explainer, MODELS_DIR / "champion_explainer.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    business_case.to_csv(MODELS_DIR / "business_case.csv", index=False)
    global_importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nBusiness case (capacite d'investigation quotidienne):")
    print(business_case.to_string(index=False))


if __name__ == "__main__":
    main()
