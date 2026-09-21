"""Narration d'investigation generee par LLM (explication uniquement), validee par un schema Pydantic.

Fournisseur choisi selon la cle presente: ANTHROPIC_API_KEY (Claude) en priorite,
sinon GEMINI_API_KEY (Gemini). Sans cle, ou si l'appel echoue, on retombe sur une
synthese deterministe basee sur les facteurs SHAP, pour que la demo/les tests
fonctionnent hors-ligne. Dans tous les cas on force un JSON conforme a
NarrativeSynthesis (on rejette toute sortie qui ne valide pas le schema).
"""
import json
import logging
import os

from pydantic import BaseModel

from riskops.schemas import NarrativeSynthesis

LOGGER = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-5"
GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """Tu es un assistant d'investigation AML (anti-blanchiment) pour des
analystes risque bancaires. On te donne le score de risque d'une transaction, les
facteurs SHAP qui l'expliquent et un extrait de l'historique du compte. Reponds
UNIQUEMENT avec un objet JSON valide respectant exactement ce schema:
{"summary": str, "key_red_flags": [str], "risk_narrative": str}
Ne recommande aucune action et n'exprime aucune confiance: l'orientation est
produite ailleurs. N'invente aucune donnee absente du contexte fourni. Sois factuel et concis."""


def _fallback_synthesis(context: dict) -> NarrativeSynthesis:
    factors = context.get("top_factors", [])
    top_names = [f["feature"] for f in factors[:3]]
    score = context.get("score", 0.0)
    return NarrativeSynthesis(
        summary=(
            f"Transaction {context.get('transaction_id')} scoree {score:.2f} par le modele, "
            f"principalement expliquee par: {', '.join(top_names) if top_names else 'aucun facteur dominant'}."
        ),
        key_red_flags=top_names or ["Aucun facteur dominant identifie"],
        risk_narrative=(
            "Synthese generee automatiquement (mode hors-ligne, sans appel LLM) a partir "
            "des contributions SHAP les plus fortes du modele."
        ),
        synthesis_source="fallback",
    )


class _NarrativeOutput(BaseModel):
    """Ce que le LLM produit; la provenance est ajoutee par le code, jamais par le modele."""
    summary: str
    key_red_flags: list[str]
    risk_narrative: str


def _claude_synthesis(context: dict, api_key: str) -> NarrativeSynthesis:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(context, default=str)}],
    )
    data = json.loads(response.content[0].text)
    return NarrativeSynthesis.model_validate({**data, "synthesis_source": "llm"})


def _gemini_synthesis(context: dict, api_key: str) -> NarrativeSynthesis:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=json.dumps(context, default=str),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_NarrativeOutput,
        ),
    )
    output = _NarrativeOutput.model_validate_json(response.text)
    return NarrativeSynthesis(**output.model_dump(), synthesis_source="llm")


def generate_synthesis(context: dict) -> NarrativeSynthesis:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if anthropic_key:
        provider, call, key = "Claude", _claude_synthesis, anthropic_key
    elif gemini_key:
        provider, call, key = "Gemini", _gemini_synthesis, gemini_key
    else:
        return _fallback_synthesis(context)

    try:
        return call(context, key)
    except Exception as exc:  # frontière réseau/SDK/parsing: toute panne => fallback
        LOGGER.warning("Synthèse %s indisponible, utilisation du fallback: %s", provider, exc)
        return _fallback_synthesis(context)
