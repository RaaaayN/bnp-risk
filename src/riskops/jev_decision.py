"""Couche d'aide a la decision (Clear / Investigate / Escalate) via Jev
(SDK officiel typesafe-sdk, API System One de TypeSafe).

Separee de la narration LLM: ici on produit une orientation typee, avec
priorite et besoin de revue humaine. Ce n'est jamais une decision: l'analyste
tranche, et l'ecart avec cette orientation est trace dans l'audit.

Sans TYPESAFE_API_KEY, ou si l'appel Jev echoue, on retombe sur des regles
deterministes (mêmes endpoints, même schéma).
"""
import logging
import os
import time

from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

from riskops.schemas import DecisionSupport

LOGGER = logging.getLogger(__name__)

ESCALATE_FLOOR = 0.85

JEV_TIMEOUT_S = 10
JEV_RETRY = RetryPolicy(max_retries=1)  # l'UI attend cet appel: peu de retries
PRIORITY_LEVELS = ["Negligeable", "Faible", "Moyenne", "Elevee", "Critique"]

# Questions posees a Jev en un seul appel (evaluees independamment).
AML_DECISION_QUESTIONS = {
    "recommended_disposition": Choice(
        instructions=(
            "Au vu du score du modele (`score`) par rapport au seuil (`threshold`) et des "
            "facteurs SHAP (`top_factors`), quelle suite un analyste AML devrait-il donner ?"
        ),
        criteria={
            "Clear": "Score et facteurs ne justifient pas d'investigation; classer sans suite",
            "Investigate": "Signaux suffisants pour une investigation standard par un analyste",
            "Escalate": "Signaux forts et convergents; escalade immediate a la conformite",
        },
    ),
    "investigation_priority": Score(
        instructions="Quelle urgence ce dossier a-t-il pour l'equipe d'investigation ?",
        criteria=PRIORITY_LEVELS,
    ),
    "requires_human_review": Noul(
        instructions=(
            "Ce dossier est-il ambigu ou atypique au point qu'une revue humaine attentive "
            "est indispensable avant toute decision ?"
        ),
        criteria={"true": "Ambigu, atypique ou signaux contradictoires", "false": "Cas net"},
    ),
    "pattern_consistency": Noul(
        instructions=(
            "Les facteurs SHAP dominants (`top_factors`) forment-ils un schema coherent "
            "de comportement AML plutot que des signaux isoles ?"
        ),
        criteria={"true": "Schema coherent", "false": "Signaux isoles ou incoherents"},
    ),
}


def deterministic_decision(context: dict) -> DecisionSupport:
    score = float(context.get("score", 0.0))
    threshold = float(context.get("threshold", 0.5))
    if score >= max(threshold, ESCALATE_FLOOR):
        action = "Escalate"
    elif score >= threshold:
        action = "Investigate"
    else:
        action = "Clear"
    return DecisionSupport(
        recommended_action=action,
        priority_score=round(min(max(score, 0.0), 1.0) * 10, 1),
        decision_source="fallback",
    )


def _make_client(api_key: str) -> TypeSafeClient:
    return TypeSafeClient(api_key=api_key, timeout=JEV_TIMEOUT_S, retry=JEV_RETRY)


def _call_jev(context: dict, api_key: str) -> DecisionSupport:
    with _make_client(api_key) as client:
        response = client.system_one(state=context, questions=AML_DECISION_QUESTIONS)

    disposition = response.choices["recommended_disposition"]
    priority = response.scores["investigation_priority"]
    levels = len(PRIORITY_LEVELS) - 1
    return DecisionSupport(
        recommended_action=disposition.choice,
        action_probabilities=disposition.probabilities,
        priority_score=round(priority.score / levels * 10, 1),
        review_probability=response.nouls["requires_human_review"].noul,
        pattern_consistency_probability=response.nouls["pattern_consistency"].noul,
        decision_source="jev",
    )


def get_decision_support(context: dict) -> DecisionSupport:
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        return deterministic_decision(context)
    start = time.perf_counter()
    try:
        result = _call_jev(context, api_key)
    except Exception as exc:  # frontière réseau/SDK: toute panne => fallback
        LOGGER.warning("Jev indisponible, utilisation du fallback: %s", exc)
        return deterministic_decision(context)
    return result.model_copy(update={"latency_ms": (time.perf_counter() - start) * 1000})
