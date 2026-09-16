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
    llm_synthesis: Optional["LLMSynthesis"] = None


class LLMSynthesis(BaseModel):
    """Sortie structuree attendue du LLM: force un format exploitable par
    l'analyste plutot qu'un texte libre non verifiable."""
    summary: str = Field(description="Resume en 2-3 phrases de la situation")
    key_red_flags: list[str] = Field(description="Liste des signaux d'alerte identifies")
    risk_narrative: str = Field(description="Explication du raisonnement de risque")
    recommended_action: Literal["Clear", "Investigate", "Escalate"]
    confidence: Literal["low", "medium", "high"]
    synthesis_source: Literal["llm", "fallback"]


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
