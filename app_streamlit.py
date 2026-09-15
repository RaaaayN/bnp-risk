"""Interface analyste: file d'alertes priorisees, detail explicable, decision
avec justification obligatoire. Consomme l'API FastAPI (src/riskops/api.py)."""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Triage d'alertes AML", layout="wide")
st.title("Triage d'alertes AML")

with st.sidebar:
    st.subheader("File d'alertes prioritaires")
    limit = st.slider("Nombre d'alertes affichees", 10, 100, 30)
    try:
        alerts = requests.get(f"{API_URL}/alerts", params={"limit": limit}, timeout=10).json()
    except requests.RequestException as e:
        st.error(f"API injoignable ({API_URL}): {e}")
        alerts = []

    options = {
        f"{a['transaction_id']} — score {a['score']:.2f} ({a['risk_band']})": a["transaction_id"]
        for a in alerts
    }
    selected_label = st.radio("Selectionner une alerte", list(options.keys())) if options else None

if not selected_label:
    st.info("Aucune alerte disponible (verifiez que l'API tourne et que le modele est entraine).")
    st.stop()

txn_id = options[selected_label]
detail = requests.get(f"{API_URL}/alerts/{txn_id}").json()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Transaction {detail['transaction_id']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Score de risque ML", f"{detail['score']:.3f}")
    m2.metric("Seuil courant", f"{detail['threshold']:.3f}")
    m3.metric("Montant", f"{detail['amount']:.2f}")

    st.markdown("**Facteurs explicatifs (SHAP)**")
    for f in detail["top_factors"]:
        sign = "+" if f["shap_contribution"] >= 0 else ""
        st.write(f"- `{f['feature']}` = {f['value']:.3f} → contribution {sign}{f['shap_contribution']:.3f}")

    st.markdown("**Historique recent du compte**")
    st.dataframe(detail["account_history"], use_container_width=True)

    st.markdown("**Contreparties suspectes liees**")
    st.write(detail["suspicious_counterparties"] or "Aucune")

    if detail.get("llm_synthesis"):
        st.markdown("**Synthese d'investigation (LLM)**")
        s = detail["llm_synthesis"]
        st.info(s["summary"])
        st.write("Signaux d'alerte:", ", ".join(s["key_red_flags"]))
        st.write(s["risk_narrative"])
        st.caption(f"Action recommandee: {s['recommended_action']} (confiance: {s['confidence']})")

with col2:
    st.subheader("Decision analyste")
    decision = st.radio("Action", ["Clear", "Investigate", "Escalate"], horizontal=False)
    justification = st.text_area("Justification (obligatoire, min. 10 caracteres)", height=150)
    if st.button("Valider la decision", type="primary"):
        if len(justification.strip()) < 10:
            st.error("Justification trop courte.")
        else:
            resp = requests.post(
                f"{API_URL}/alerts/{txn_id}/decision",
                json={"decision": decision, "justification": justification, "decision_by": "analyst"},
            )
            if resp.ok:
                st.success(f"Decision enregistree (audit id={resp.json()['id']}).")
            else:
                st.error(f"Erreur: {resp.text}")

    st.divider()
    st.subheader("Journal d'audit (dernieres decisions)")
    audit = requests.get(f"{API_URL}/audit", params={"limit": 20}).json()
    st.dataframe(audit, use_container_width=True)
