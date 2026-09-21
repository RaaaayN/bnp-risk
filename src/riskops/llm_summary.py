"""Narration d'investigation generee par LLM (explication uniquement), validee par un schema Pydantic.

Sans cle API (ANTHROPIC_API_KEY absente), on retombe sur une synthese
deterministe basee sur les facteurs SHAP, pour que la demo/les tests
fonctionnent hors-ligne. Avec la cle, on appelle Claude et on force un JSON
conforme a NarrativeSynthesis (on rejette toute sortie qui ne valide pas le schema).
"""
import json
import logging
import os

from pydantic import ValidationError

from riskops.schemas import NarrativeSynthesis

LOGGER = logging.getLogger(__name__)

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


def generate_synthesis(context: dict) -> NarrativeSynthesis:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_synthesis(context)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(context, default=str)}],
        )
        raw_text = response.content[0].text
        data = json.loads(raw_text)
        data["synthesis_source"] = "llm"
        return NarrativeSynthesis.model_validate(data)
    except (
        anthropic.APIError,
        ValidationError,
        json.JSONDecodeError,
        IndexError,
        AttributeError,
        TypeError,
    ) as exc:
        LOGGER.warning("Synthèse LLM indisponible, utilisation du fallback: %s", exc)
        return _fallback_synthesis(context)
