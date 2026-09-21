"""Modeles Pydantic partages par l'API et la synthese LLM."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RiskFactor(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class AccountHistoryItem(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float
    counterparty_account: int
    is_flagged: bool


class AlertSummary(BaseModel):
    transaction_id: str
    timestamp: str
    amount: float
    score: float
    risk_band: Literal["low", "medium", "high"]


class AlertDetail(AlertSummary):
    model_name: str
    model_version: str
    threshold: float
    top_factors: list[RiskFactor]
    account_history: list[AccountHistoryItem]
    suspicious_counterparties: list[int]
    decision_support: Optional["DecisionSupport"] = None
    narrative: Optional["NarrativeSynthesis"] = None


class NarrativeSynthesis(BaseModel):
    """Sortie structuree du LLM: narration uniquement. L'orientation et la
    confiance relevent de DecisionSupport, jamais du LLM."""
    summary: str = Field(description="Resume en 2-3 phrases de la situation")
    key_red_flags: list[str] = Field(description="Liste des signaux d'alerte identifies")
    risk_narrative: str = Field(description="Explication du raisonnement de risque")
    synthesis_source: Literal["llm", "fallback"]


class DecisionSupport(BaseModel):
    """Aide a la decision typee (couche Jev, ou fallback deterministe).
    Les probabilites, la revue humaine et la latence sont None en fallback: on
    n'affiche pas de probabilite qu'aucun modele n'a produite."""
    recommended_action: Literal["Clear", "Investigate", "Escalate"]
    action_probabilities: Optional[dict[str, float]] = None
    priority_score: float = Field(ge=0, le=10)
    review_probability: Optional[float] = Field(default=None, ge=0, le=1)
    pattern_consistency_probability: Optional[float] = Field(default=None, ge=0, le=1)
    decision_source: Literal["jev", "fallback"]
    latency_ms: Optional[float] = None


class DecisionRequest(BaseModel):
    decision: Literal["Clear", "Investigate", "Escalate"]
    justification: str = Field(min_length=10, description="Justification obligatoire (>=10 caracteres)")
    decision_by: str = Field(default="analyst")


class DecisionResponse(BaseModel):
    id: int
    transaction_id: str
    decision: str
    justification: str
    created_at: str
