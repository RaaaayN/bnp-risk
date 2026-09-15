# Note de conception

Le README dit comment utiliser le projet. Ce document dit pourquoi il est
construit comme ça : les contraintes métier et techniques qui ont pesé sur
chaque choix, ce qui a été écarté et pourquoi, et les limites qu'on assume.

## 1. Le problème que le système doit réellement résoudre

Un modèle de détection AML n'est pas évalué sur sa capacité à « bien
prédire » dans l'absolu, mais sur sa capacité à **améliorer une décision sous
contrainte** : une équipe d'analystes a une capacité de traitement finie
(ici, 100 dossiers/jour), et le coût d'une erreur n'est pas symétrique.

| Erreur | Coût |
|---|---|
| Faux négatif (blanchiment non détecté) | Risque réglementaire et réputationnel — potentiellement des sanctions, très supérieur au coût d'une investigation |
| Faux positif (alerte sur transaction légitime) | Temps analyste perdu — coût réel mais borné, et il est le facteur limitant du système (capacité fixe) |

Cette asymétrie, combinée à une classe positive extrêmement rare (≈0.3% des
transactions dans les données synthétiques, du même ordre que les datasets
AML réels), détermine presque tous les choix de conception ci-dessous : on ne
cherche pas un modèle « précis » au sens classique, mais un **système qui
classe correctement le haut du classement** (les N transactions les plus
suspectes, N = capacité analyste), et qui rend chaque décision **contestable
et traçable** plutôt qu'opaque.

De là découlent trois contraintes non négociables, posées par le contexte
bancaire/réglementaire plus que par la donnée elle-même. D'abord
l'explicabilité : un analyste, et un régulateur a posteriori, doit pouvoir
comprendre pourquoi une transaction a été remontée. Ensuite le contrôle
humain : le modèle priorise, il ne décide jamais seul, toute action
(Clear/Investigate/Escalate) reste un acte humain justifié. Et enfin la
traçabilité : chaque décision est journalisée avec le contexte qui l'a
produite (version du modèle, seuil, score, données vues), pour pouvoir
répondre après coup à "pourquoi a-t-on laissé passer / bloqué cette
transaction ?".

## 2. Enjeux techniques et comment ils ont façonné le pipeline

### 2.1 Déséquilibre de classe extrême → pas d'accuracy, pas de split aveugle

Avec 0.3% de positifs, un modèle qui répond toujours « non suspect » atteint
99.7% d'accuracy tout en étant inutile. Deux conséquences :

- **Métriques** ([`evaluate.py`](../src/riskops/evaluate.py)) : le ROC-AUC
  mesure la capacité à distinguer les deux classes sur *l'ensemble* des
  transactions, y compris les 99.7% de négatifs faciles à écarter — un
  classement médiocre des positifs entre eux y pèse peu, noyé dans la masse
  des négatifs. Le PR-AUC, lui, ne regarde que precision et recall sur la
  classe positive, donc reflète directement la qualité du classement des cas
  suspects — c'est cette métrique qui est utilisée. Et surtout, on utilise des
  métriques *à budget fixé* (precision@K, recall à budget constant) qui
  répondent directement à la question opérationnelle : « avec N dossiers/jour,
  combien de cas détecte-t-on réellement ? ». C'est la seule vraie mesure de
  valeur pour le
  métier.
- **Entraînement** : `class_weight="balanced"` pour la régression logistique
  et `scale_pos_weight` pour XGBoost, pour éviter que le modèle apprenne
  simplement à tout classer négatif.

### 2.2 Fuite temporelle (data leakage) → split chronologique + features causales

C'est le piège le plus courant — et le plus dangereux — en scoring de
transactions : si les features « historique du compte » sont calculées sur
l'ensemble du dataset (passé *et* futur) ou si le split train/test est
aléatoire, le modèle voit indirectement des informations qui n'existeraient
pas au moment réel du scoring. Le modèle a alors une performance excellente
en test... et s'effondre en production.

Deux garde-fous dans [`features.py`](../src/riskops/features.py) :

- Chaque feature d'historique de compte (`acct_txn_count_prior`,
  `acct_avg_amount_prior`, `acct_distinct_counterparties_7d`,
  `acct_txn_count_24h`) est calculée par un `groupby` trié par compte puis
  par timestamp, avec un décalage explicite (`cumsum() - amount`, comptage
  strictement antérieur) — la transaction courante ne voit jamais sa propre
  contribution ni le futur du compte. `tests/test_features.py` teste
  explicitement ce point (`test_no_leakage_first_transaction_has_no_prior_history`,
  `test_prior_avg_only_uses_past_transactions`).
- Le split train/val/test (`chronological_split` dans
  [`train.py`](../src/riskops/train.py)) est fait par coupure temporelle
  (70/15/15), pas par tirage aléatoire de lignes — le modèle est validé
  exactement comme il serait déployé : entraîné sur le passé, évalué sur un
  futur qu'il n'a jamais vu.

### 2.3 Explicabilité → SHAP plutôt qu'un modèle intrinsèquement interprétable

Alternative écartée : se limiter à un modèle interprétable par construction
(arbre peu profond, règles). Rejeté parce que ça sacrifie du pouvoir
prédictif pour un gain d'interprétabilité qu'on peut obtenir autrement.
**SHAP TreeExplainer** sur XGBoost donne une explication *par transaction*
(quelles features ont poussé le score vers le haut ou le bas, et de combien)
sans contraindre l'architecture du modèle. C'est ce qui alimente à la fois
l'UI analyste (`top_factors` dans l'API) et la synthèse LLM — le LLM ne
reçoit jamais la transaction brute, seulement les facteurs SHAP déjà
calculés, ce qui limite son rôle à *reformuler* une explication déjà produite
par une méthode auditable, plutôt qu'à en inventer une.

### 2.4 Deux modèles comparés, pas un seul

La régression logistique n'est pas là par complétude académique : c'est la
**baseline de référence** qui permet de justifier XGBoost par un chiffre
plutôt que par affirmation. Sur le jeu de test (13 jours, ~120k transactions
au total dont 60 positives), le résultat obtenu (`models/metrics.json`) : à
recall égal (50%, soit 30 des 60 cas de blanchiment retrouvés), XGBoost
génère 854 alertes contre 1115 pour la régression logistique, soit **-24% de
faux positifs** à qualité de détection identique. C'est ce chiffre — pas un
PR-AUC abstrait — qui justifie le choix du modèle final en production, parce
qu'il se traduit directement en charge analyste économisée.

### 2.5 Le LLM ne doit jamais être une source de vérité

Deux risques spécifiques aux LLM dans un contexte de décision réglementée :
l'hallucination (inventer un fait absent du contexte) et la sortie non
structurée (impossible à valider ou à afficher de façon fiable dans l'UI).
Ça se traduit concrètement dans
[`llm_summary.py`](../src/riskops/llm_summary.py) et
[`schemas.py`](../src/riskops/schemas.py) par quatre garde-fous.

Le schéma Pydantic est forcé (`LLMSynthesis`) : le prompt système exige un
JSON conforme, et toute sortie qui ne valide pas est rejetée
(`ValidationError` → fallback) plutôt qu'affichée telle quelle. On ne fait
jamais confiance à du texte libre. Le contexte donné au LLM se limite aux
facteurs SHAP déjà calculés, jamais la transaction brute ni une invitation à
"analyser" — il reformule une explication déjà produite statistiquement, il
n'en invente pas une nouvelle. Sans clé API, un fallback déterministe prend
le relais : mêmes endpoints, même schéma de sortie, une synthèse construite
directement à partir des facteurs SHAP. Ce n'est pas qu'une commodité de
démo, ça évite qu'un système de contrôle des risques tombe en panne parce
qu'un fournisseur LLM externe est indisponible. Et la synthèse n'apparaît
jamais seule : elle reste à côté des facteurs SHAP bruts dans l'UI, la
décision et sa justification restent un champ texte rempli par l'analyste,
jamais pré-rempli par le LLM — impossible de valider une décision sans au
moins 10 caractères de justification humaine (`DecisionRequest` dans
`schemas.py`).

### 2.6 Journal d'audit : SQLite, pas de service externe

Le besoin est la traçabilité (modèle, seuil, score, données vues, décision
humaine, justification), pas le volume ni la concurrence à grande échelle.
SQLite est un fichier unique, versionnable, inspectable sans infrastructure —
suffisant pour un projet de cette taille et cohérent avec la contrainte
« pas de microservices, pas d'infra superflue ». Le schéma
([`audit.py`](../src/riskops/audit.py)) refuse en base toute décision sans
justification non vide ou avec une valeur de décision hors de l'énumération
`{Clear, Investigate, Escalate}` — la contrainte de contrôle est appliquée au
niveau code *et* au niveau des types (Pydantic `Literal` dans
`DecisionRequest`), pas seulement dans l'UI.

### 2.7 Seuil piloté par la capacité opérationnelle, pas par un optimum statistique

C'est le point de jonction entre le modèle et la valeur métier. Un seuil
choisi pour maximiser le F1-score ou un point arbitraire de la courbe
precision/recall ignore la vraie contrainte : le nombre de dossiers qu'une
équipe peut traiter par jour. `threshold_for_budget` et
`business_case_sweep` ([`evaluate.py`](../src/riskops/evaluate.py)) inversent
le problème : on part de la capacité (100/jour), on en déduit le seuil, puis
on mesure le recall obtenu. C'est ce calcul qui répond à la question posée en
introduction du projet — voir la section *Business case* du README pour les
chiffres et leur lecture.

## 3. Ce qui a été délibérément exclu, et pourquoi

Le README le mentionne déjà brièvement ; voici le raisonnement complet.

- **Pas de graphe de transactions / GNN** : les patterns de blanchiment
  structurés en couches (smurfing, layering multi-comptes) bénéficieraient
  d'une représentation en graphe, mais ça multiplie la complexité
  d'infrastructure (stockage de graphe, traversée, entraînement GNN) pour un
  gain qui n'est pas démontrable sur 4 jours ni nécessaire pour illustrer les
  compétences visées (triage explicable, pas détection de réseaux).
- **Pas de LangGraph / orchestration d'agents** : il n'y a ici qu'un seul
  appel LLM avec un rôle strictement borné (reformuler des facteurs SHAP déjà
  calculés en résumé lisible). Une orchestration multi-agents résoudrait un
  problème qui n'existe pas dans ce système et ajouterait une source de
  panne et de non-déterminisme supplémentaire dans un contexte où la
  fiabilité prime.
- **Pas de microservices/Kubernetes** : un seul modèle, un seul flux de
  décision, une volumétrie de démonstration. Docker Compose à deux services
  (API + UI) donne l'isolation nécessaire sans la charge opérationnelle d'un
  orchestrateur dimensionné pour un tout autre ordre de grandeur.
- **Pas de ré-entraînement automatique / détection de drift** : ce serait la
  suite logique en production (le comportement de blanchiment évolue pour
  contourner un modèle statique), mais c'est un système de monitoring à part
  entière, hors du périmètre d'un projet de triage explicable.
- **Dataset synthétique plutôt que le CSV IBM réel** : le dataset Kaggle
  pèse plusieurs Go et nécessite des credentials — un obstacle d'accès
  disproportionné par rapport à l'objectif (démontrer une méthodologie). Le
  générateur ([`data_gen.py`](../src/riskops/data_gen.py)) reproduit
  exactement le schéma de colonnes IBM et un déséquilibre de classe réaliste,
  ce qui rend le pipeline directement réutilisable sur le vrai fichier sans
  modification de code — voir README, section *Données*.

## 4. Limites assumées

- Les patterns de blanchiment synthétiques sont volontairement bruités mais
  restent plus simples que des schémas réels multi-sauts — les métriques
  obtenues ne sont pas transposables telles quelles à un dataset réel, elles
  valident la méthodologie, pas une performance absolue.
- Le seuil de production est recalculé sur la période de test uniquement ;
  en usage réel il faudrait le réviser périodiquement à mesure que le volume
  de transactions et le taux de blanchiment évoluent.
- La synthèse LLM, même contrainte par schéma, reste un résumé et non une
  preuve : elle est explicitement positionnée comme aide à la lecture, jamais
  comme justification suffisante d'une décision (voir §2.5).

## 5. Si le périmètre devait grandir

Dans l'ordre de priorité probable pour une mise en production réelle :
monitoring de drift du modèle et du volume d'alertes, ré-entraînement
périodique versionné, authentification/permissions sur l'API (actuellement
absente — hors périmètre démo), et seulement ensuite des features de graphe
pour capter les schémas de blanchiment en réseau.
