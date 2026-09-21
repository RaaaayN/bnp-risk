
from riskops.llm_summary import generate_synthesis
from riskops.schemas import NarrativeSynthesis


def test_fallback_synthesis_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    context = {
        "transaction_id": "TXN1",
        "score": 0.92,
        "top_factors": [{"feature": "amount_log", "value": 9.2, "shap_contribution": 0.4}],
        "account_history_count": 3,
    }
    result = generate_synthesis(context)
    assert isinstance(result, NarrativeSynthesis)
    assert "amount_log" in result.key_red_flags
    assert result.synthesis_source == "fallback"


def test_fallback_synthesis_is_schema_valid_for_low_score(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    context = {"transaction_id": "TXN2", "score": 0.1, "top_factors": [], "account_history_count": 0}
    result = generate_synthesis(context)
    assert result.synthesis_source == "fallback"
    assert not hasattr(result, "recommended_action")
