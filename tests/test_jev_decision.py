import json

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from riskops import jev_decision
from riskops.jev_decision import (
    AML_DECISION_QUESTIONS,
    PRIORITY_LEVELS,
    deterministic_decision,
    get_decision_support,
)


@pytest.mark.parametrize(
    "score, expected",
    [(0.92, "Escalate"), (0.7, "Investigate"), (0.1, "Clear")],
)
def test_deterministic_decision_thresholds(score, expected):
    result = deterministic_decision({"score": score, "threshold": 0.6})
    assert result.recommended_action == expected
    assert result.decision_source == "fallback"
    assert result.action_probabilities is None


JEV_BODY = {
    "model": "jev-test",
    "answers": {
        "recommended_disposition": {
            "type": "choice", "choice": "Escalate", "confidence": 0.9,
            "probabilities": {"Clear": 0.01, "Investigate": 0.17, "Escalate": 0.82},
        },
        "investigation_priority": {
            "type": "score", "score": 3.4, "confidence": 0.9,
            "legend": {str(i): lvl for i, lvl in enumerate(PRIORITY_LEVELS)},
            "probabilities": {"0": 0, "1": 0, "2": 0.1, "3": 0.5, "4": 0.4},
        },
        "requires_human_review": {"type": "noul", "noul": 0.97},
        "pattern_consistency": {"type": "noul", "noul": 0.88},
    },
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


@pytest.fixture
def jev_http(monkeypatch):
    """Fait tourner le vrai SDK typesafe-sdk contre un transport HTTP simule."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    calls, state = [], {"handler": lambda req: httpx2.Response(200, json=JEV_BODY)}

    def handler(request):
        calls.append(request)
        return state["handler"](request)

    def make_client(api_key):
        return TypeSafeClient(
            api_key=api_key, transport=httpx2.MockTransport(handler),
            retry=RetryPolicy(max_retries=0),
        )

    monkeypatch.setattr(jev_decision, "_make_client", make_client)
    return calls, state


def test_jev_response_is_mapped(jev_http):
    calls, _ = jev_http
    result = get_decision_support({"score": 0.81, "threshold": 0.59})
    assert result.decision_source == "jev"
    assert result.recommended_action == "Escalate"
    assert result.action_probabilities["Escalate"] == 0.82
    assert result.priority_score == 8.5
    assert result.review_probability == 0.97
    assert result.pattern_consistency_probability == 0.88
    assert result.latency_ms is not None

    (request,) = calls
    body = json.loads(request.content)
    assert body["state"]["score"] == 0.81
    assert set(body["questions"]) == set(AML_DECISION_QUESTIONS)
    assert request.headers["authorization"] == "Bearer test-key"


def test_no_api_key_uses_fallback(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert get_decision_support({"score": 0.9, "threshold": 0.5}).decision_source == "fallback"


@pytest.mark.parametrize(
    "handler",
    [
        lambda req: httpx2.Response(500, json={"error": "boom"}),
        lambda req: httpx2.Response(401, json={"error": "bad key"}),
        lambda req: httpx2.Response(200, json={"model": "x", "answers": {}, "usage": {}}),
        lambda req: httpx2.Response(200, text="not json"),
    ],
    ids=["500", "401", "answers-vides", "non-json"],
)
def test_jev_failure_or_bad_output_falls_back(jev_http, handler):
    _, state = jev_http
    state["handler"] = handler
    result = get_decision_support({"score": 0.9, "threshold": 0.5})
    assert result.decision_source == "fallback"
    assert result.recommended_action == "Escalate"


def test_jev_choice_outside_enum_falls_back(jev_http):
    _, state = jev_http
    body = json.loads(json.dumps(JEV_BODY))
    body["answers"]["recommended_disposition"]["choice"] = "Block"
    state["handler"] = lambda req: httpx2.Response(200, json=body)
    assert get_decision_support({"score": 0.3, "threshold": 0.5}).decision_source == "fallback"


def test_jev_connection_error_falls_back(jev_http):
    _, state = jev_http

    def boom(request):
        raise httpx2.ConnectError("down")

    state["handler"] = boom
    assert get_decision_support({"score": 0.3, "threshold": 0.5}).decision_source == "fallback"
