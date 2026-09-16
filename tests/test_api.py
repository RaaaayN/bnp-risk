import joblib
import numpy as np
import pandas as pd
import pytest
import shap
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

import riskops.api as api_module
from riskops.features import FEATURE_COLUMNS


@pytest.fixture
def client(tmp_path, monkeypatch):
    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in FEATURE_COLUMNS})
    df["Timestamp"] = pd.date_range("2024-01-01", periods=n, freq="h")
    df["Transaction Id"] = [f"TXN{i}" for i in range(n)]
    df["Account"] = rng.integers(1, 5, n)
    df["Account.1"] = rng.integers(1, 5, n)
    df["amount"] = rng.random(n) * 1000
    df["Is Laundering"] = (rng.random(n) < 0.2).astype(int)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "features.parquet", index=False)

    model = XGBClassifier(n_estimators=10, max_depth=2)
    model.fit(df[FEATURE_COLUMNS], df["Is Laundering"])
    explainer = shap.TreeExplainer(model)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "threshold": 0.5},
        models_dir / "xgb_model.joblib",
    )
    joblib.dump(explainer, models_dir / "shap_explainer.joblib")

    monkeypatch.setattr(api_module, "ROOT", tmp_path)
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "AUDIT_DB_PATH", tmp_path / "audit.db")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with TestClient(api_module.app) as c:
        yield c


def test_get_alerts_returns_sorted_queue(client):
    resp = client.get("/alerts", params={"limit": 100})
    assert resp.status_code == 200
    alerts = resp.json()
    scores = [a["score"] for a in alerts]
    assert scores == sorted(scores, reverse=True)


def test_get_alert_detail_has_shap_and_history(client):
    alerts = client.get("/alerts", params={"limit": 1}).json()
    assert alerts, "au moins une alerte doit depasser le seuil dans le jeu de test"
    txn_id = alerts[0]["transaction_id"]
    detail = client.get(f"/alerts/{txn_id}").json()
    assert detail["top_factors"]
    assert "llm_synthesis" in detail


def test_decision_requires_justification(client):
    alerts = client.get("/alerts", params={"limit": 1}).json()
    txn_id = alerts[0]["transaction_id"]
    resp = client.post(f"/alerts/{txn_id}/decision", json={"decision": "Clear", "justification": "trop court"[:5]})
    assert resp.status_code == 422


def test_decision_is_persisted_in_audit_log(client):
    alerts = client.get("/alerts", params={"limit": 1}).json()
    txn_id = alerts[0]["transaction_id"]
    client.get(f"/alerts/{txn_id}")
    resp = client.post(
        f"/alerts/{txn_id}/decision",
        json={"decision": "Escalate", "justification": "Montant eleve et contrepartie inhabituelle"},
    )
    assert resp.status_code == 200
    audit = client.get("/audit").json()
    assert any(a["transaction_id"] == txn_id and a["decision"] == "Escalate" for a in audit)
    saved = next(a for a in audit if a["transaction_id"] == txn_id)
    assert saved["synthesis_source"] == "fallback"


def test_unknown_transaction_returns_404(client):
    resp = client.get("/alerts/DOES_NOT_EXIST")
    assert resp.status_code == 404
