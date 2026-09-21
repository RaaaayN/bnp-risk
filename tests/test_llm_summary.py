
from riskops import llm_summary
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


CONTEXT = {"transaction_id": "TXN3", "score": 0.9, "top_factors": [], "account_history_count": 1}
GOOD = NarrativeSynthesis(summary="s", key_red_flags=["a"], risk_narrative="n", synthesis_source="llm")


def test_gemini_is_used_when_only_gemini_key_is_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr(llm_summary, "_gemini_synthesis", lambda ctx, key: GOOD)
    assert generate_synthesis(CONTEXT).synthesis_source == "llm"


def test_claude_takes_precedence_over_gemini(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    called = []
    monkeypatch.setattr(llm_summary, "_claude_synthesis", lambda ctx, key: called.append("claude") or GOOD)
    monkeypatch.setattr(llm_summary, "_gemini_synthesis", lambda ctx, key: called.append("gemini") or GOOD)
    generate_synthesis(CONTEXT)
    assert called == ["claude"]


def test_provider_failure_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")

    def boom(ctx, key):
        raise RuntimeError("quota")

    monkeypatch.setattr(llm_summary, "_gemini_synthesis", boom)
    assert generate_synthesis(CONTEXT).synthesis_source == "fallback"
