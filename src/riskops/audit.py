"""Journal d'audit SQLite: trace pour chaque decision le modele, le seuil, les
donnees vues par l'analyste et la decision humaine (exigence de controle/
tracabilite typique d'un contexte reglementaire bancaire)."""
import datetime
import json
import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "audit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    threshold REAL NOT NULL,
    score REAL NOT NULL,
    features_json TEXT NOT NULL,
    shap_top_factors_json TEXT NOT NULL,
    llm_summary_json TEXT,
    synthesis_source TEXT CHECK(synthesis_source IN ('llm', 'fallback') OR synthesis_source IS NULL),
    decision TEXT NOT NULL,
    justification TEXT NOT NULL,
    decision_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

VALID_DECISIONS = {"Clear", "Investigate", "Escalate"}


def get_connection(db_path: pathlib.Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)")}
    if "synthesis_source" not in columns:
        conn.execute("ALTER TABLE audit_log ADD COLUMN synthesis_source TEXT")
    return conn


def record_decision(
    conn: sqlite3.Connection,
    transaction_id: str,
    model_name: str,
    model_version: str,
    threshold: float,
    score: float,
    features: dict,
    shap_top_factors: list,
    decision: str,
    justification: str,
    decision_by: str,
    llm_summary: dict | None = None,
) -> int:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Decision invalide: {decision}. Attendu: {VALID_DECISIONS}")
    if not justification or not justification.strip():
        raise ValueError("La justification est obligatoire pour toute decision.")
    synthesis_source = llm_summary.get("synthesis_source") if llm_summary else None

    cur = conn.execute(
        """INSERT INTO audit_log
        (transaction_id, model_name, model_version, threshold, score, features_json,
         shap_top_factors_json, llm_summary_json, synthesis_source, decision, justification,
         decision_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            transaction_id, model_name, model_version, threshold, score,
            json.dumps(features), json.dumps(shap_top_factors),
            json.dumps(llm_summary) if llm_summary else None,
            synthesis_source,
            decision, justification.strip(), decision_by,
            datetime.datetime.now(datetime.UTC).isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def list_decisions(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_decisions_for_transaction(conn: sqlite3.Connection, transaction_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM audit_log WHERE transaction_id = ? ORDER BY created_at DESC",
        (transaction_id,),
    ).fetchall()
    return [dict(r) for r in rows]
