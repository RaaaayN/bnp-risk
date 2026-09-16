import sqlite3

import pytest

from riskops.audit import SCHEMA, VALID_DECISIONS, list_decisions, record_decision


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    yield c
    c.close()


def test_record_decision_requires_justification(conn):
    with pytest.raises(ValueError):
        record_decision(
            conn, transaction_id="TXN1", model_name="xgboost", model_version="v1",
            threshold=0.5, score=0.9, features={}, shap_top_factors=[],
            decision="Escalate", justification="   ", decision_by="analyst",
        )


def test_record_decision_rejects_invalid_decision(conn):
    with pytest.raises(ValueError):
        record_decision(
            conn, transaction_id="TXN1", model_name="xgboost", model_version="v1",
            threshold=0.5, score=0.9, features={}, shap_top_factors=[],
            decision="Ignore", justification="Justification suffisante", decision_by="analyst",
        )


def test_record_and_list_decision(conn):
    decision_id = record_decision(
        conn, transaction_id="TXN1", model_name="xgboost", model_version="v1",
        threshold=0.5, score=0.9, features={"amount_log": 5.2}, shap_top_factors=[{"feature": "amount_log"}],
        decision="Escalate", justification="Montant inhabituel et contrepartie a risque",
        decision_by="analyst",
    )
    assert decision_id == 1
    rows = list_decisions(conn)
    assert len(rows) == 1
    assert rows[0]["decision"] == "Escalate"
    assert rows[0]["transaction_id"] == "TXN1"


def test_synthesis_source_is_persisted(conn):
    record_decision(
        conn, transaction_id="TXN2", model_name="xgboost", model_version="v1",
        threshold=0.5, score=0.8, features={}, shap_top_factors=[],
        decision="Investigate", justification="Synthese a verifier par analyste",
        decision_by="analyst", llm_summary={"summary": "test", "synthesis_source": "fallback"},
    )
    row = list_decisions(conn)[0]
    assert row["synthesis_source"] == "fallback"


def test_all_decisions_are_valid_enum_values():
    assert VALID_DECISIONS == {"Clear", "Investigate", "Escalate"}
