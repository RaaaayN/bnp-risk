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
    alerts_per_10k, business_case_sweep, false_positive_reduction_at_equal_recall,
    pr_auc, precision_at_k, recall_at_budget,
)
from riskops.features import FEATURE_COLUMNS

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DAILY_INVESTIGATION_CAPACITY = 100


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

    n_days_test = max((test_df["Timestamp"].max() - test_df["Timestamp"].min()).days, 1)

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(X_train), y_train)
    lr_test_scores = lr.predict_proba(scaler.transform(X_test))[:, 1]

    # --- Modele boosted trees ---
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_test_scores = xgb.predict_proba(X_test)[:, 1]

    budget = DAILY_INVESTIGATION_CAPACITY * n_days_test
    metrics = {
        "n_days_test": n_days_test,
        "logreg": {
            "pr_auc": pr_auc(y_test, lr_test_scores),
            "precision_at_budget": precision_at_k(y_test, lr_test_scores, budget),
            "recall_at_budget": recall_at_budget(y_test, lr_test_scores, budget),
            "alerts_per_10k_at_p99": alerts_per_10k(lr_test_scores, np.quantile(lr_test_scores, 0.99)),
        },
        "xgboost": {
            "pr_auc": pr_auc(y_test, xgb_test_scores),
            "precision_at_budget": precision_at_k(y_test, xgb_test_scores, budget),
            "recall_at_budget": recall_at_budget(y_test, xgb_test_scores, budget),
            "alerts_per_10k_at_p99": alerts_per_10k(xgb_test_scores, np.quantile(xgb_test_scores, 0.99)),
        },
        "fp_reduction_xgb_vs_logreg_at_equal_recall": false_positive_reduction_at_equal_recall(
            y_test, lr_test_scores, xgb_test_scores, target_recall=0.5
        ),
    }

    business_case = business_case_sweep(y_test, xgb_test_scores, n_days_test)

    # --- SHAP (sur le modele retenu: XGBoost) ---
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    global_importance = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
    )

    joblib.dump({"model": xgb, "features": FEATURE_COLUMNS}, MODELS_DIR / "xgb_model.joblib")
    joblib.dump({"model": lr, "scaler": scaler, "features": FEATURE_COLUMNS}, MODELS_DIR / "logreg_model.joblib")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    business_case.to_csv(MODELS_DIR / "business_case.csv", index=False)
    global_importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nBusiness case (capacite d'investigation quotidienne):")
    print(business_case.to_string(index=False))


if __name__ == "__main__":
    main()
