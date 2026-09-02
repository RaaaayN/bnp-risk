"""API FastAPI: file d'alertes priorisees, detail explicable (SHAP + historique
+ synthese LLM), et decision analyste avec justification obligatoire, journalisee
dans l'audit log SQLite."""
import pathlib
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from riskops.audit import get_connection, list_decisions, record_decision
from riskops.evaluate import threshold_for_budget
from riskops.features import FEATURE_COLUMNS
from riskops.llm_summary import generate_synthesis
from riskops.schemas import (
    AccountHistoryItem, AlertDetail, AlertSummary, DecisionRequest, DecisionResponse, RiskFactor,
)
from riskops.train import DAILY_INVESTIGATION_CAPACITY, chronological_split

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
AUDIT_DB_PATH = ROOT / "audit.db"
MODEL_VERSION = "xgb-v1"

_state: dict = {}


def _risk_band(score: float, threshold: float) -> str:
    if score >= max(threshold * 1.5, 0.85):
        return "high"
    if score >= threshold:
        return "medium"
    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="RiskOps Copilot API", version="0.1.0", lifespan=lifespan)


def load_artifacts():
    feats_path = ROOT / "data" / "features.parquet"
    df = pd.read_parquet(feats_path)
    _, _, test_df = chronological_split(df)
    test_df = test_df.reset_index(drop=True)

    bundle = joblib.load(MODELS_DIR / "xgb_model.joblib")
    model = bundle["model"]
    explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")

    X_test = test_df[FEATURE_COLUMNS]
    scores = model.predict_proba(X_test)[:, 1]
    shap_values = explainer.shap_values(X_test)

    n_days = max((test_df["Timestamp"].max() - test_df["Timestamp"].min()).days, 1)
    threshold = threshold_for_budget(scores, DAILY_INVESTIGATION_CAPACITY, n_days)

    test_df["score"] = scores
    _state.update(
        model=model, explainer=explainer, df=test_df, shap_values=shap_values,
        threshold=threshold, conn=get_connection(AUDIT_DB_PATH),
    )


def _get_row(transaction_id: str):
    df = _state["df"]
    matches = df.index[df["Transaction Id"] == transaction_id]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail="Transaction introuvable")
    return matches[0]


@app.get("/alerts", response_model=list[AlertSummary])
def get_alerts(limit: int = 50, offset: int = 0):
    df = _state["df"]
    threshold = _state["threshold"]
    queue = df[df["score"] >= threshold].sort_values("score", ascending=False)
    page = queue.iloc[offset: offset + limit]
    return [
        AlertSummary(
            transaction_id=row["Transaction Id"],
            timestamp=str(row["Timestamp"]),
            amount=float(row["amount"]),
            score=float(row["score"]),
            risk_band=_risk_band(row["score"], threshold),
        )
        for _, row in page.iterrows()
    ]


@app.get("/alerts/{transaction_id}", response_model=AlertDetail)
def get_alert_detail(transaction_id: str, with_llm: bool = True):
    idx = _get_row(transaction_id)
    df = _state["df"]
    row = df.loc[idx]
    threshold = _state["threshold"]

    shap_row = _state["shap_values"][idx]
    top_idx = np.argsort(-np.abs(shap_row))[:5]
    top_factors = [
        RiskFactor(feature=FEATURE_COLUMNS[i], value=float(row[FEATURE_COLUMNS[i]]),
                   shap_contribution=float(shap_row[i]))
        for i in top_idx
    ]

    account = row["Account"]
    history_df = df[(df["Account"] == account) & (df["Timestamp"] <= row["Timestamp"])].sort_values(
        "Timestamp", ascending=False
    ).head(10)
    history = [
        AccountHistoryItem(
            transaction_id=h["Transaction Id"], timestamp=str(h["Timestamp"]),
            amount=float(h["amount"]), counterparty_account=int(h["Account.1"]),
            is_flagged=bool(h["score"] >= threshold),
        )
        for _, h in history_df.iterrows()
    ]

    suspicious_cp = (
        df[(df["Account"] == account) & (df["score"] >= threshold)]["Account.1"].unique().tolist()
    )

    detail = AlertDetail(
        transaction_id=row["Transaction Id"], timestamp=str(row["Timestamp"]),
        amount=float(row["amount"]), score=float(row["score"]),
        risk_band=_risk_band(row["score"], threshold),
        model_name="xgboost", model_version=MODEL_VERSION, threshold=float(threshold),
        top_factors=top_factors, account_history=history,
        suspicious_counterparties=[int(c) for c in suspicious_cp],
    )

    if with_llm:
        context = {
            "transaction_id": detail.transaction_id, "score": detail.score,
            "top_factors": [f.model_dump() for f in top_factors],
            "account_history_count": len(history),
        }
        detail.llm_synthesis = generate_synthesis(context)

    return detail


@app.post("/alerts/{transaction_id}/decision", response_model=DecisionResponse)
def post_decision(transaction_id: str, decision_req: DecisionRequest):
    idx = _get_row(transaction_id)
    row = _state["df"].loc[idx]
    shap_row = _state["shap_values"][idx]
    top_idx = np.argsort(-np.abs(shap_row))[:5]
    top_factors = [
        {"feature": FEATURE_COLUMNS[i], "shap_contribution": float(shap_row[i])} for i in top_idx
    ]

    decision_id = record_decision(
        _state["conn"], transaction_id=transaction_id, model_name="xgboost",
        model_version=MODEL_VERSION, threshold=float(_state["threshold"]), score=float(row["score"]),
        features={c: float(row[c]) for c in FEATURE_COLUMNS}, shap_top_factors=top_factors,
        decision=decision_req.decision, justification=decision_req.justification,
        decision_by=decision_req.decision_by,
    )
    saved = _state["conn"].execute("SELECT * FROM audit_log WHERE id = ?", (decision_id,)).fetchone()
    return DecisionResponse(
        id=saved["id"], transaction_id=saved["transaction_id"],
        decision=saved["decision"], justification=saved["justification"], created_at=saved["created_at"],
    )


@app.get("/audit")
def get_audit(limit: int = 100):
    return list_decisions(_state["conn"], limit=limit)
