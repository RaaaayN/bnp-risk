# Triage explicable d'alertes AML

Petit projet pour tester une idée simple : un modèle qui score les
transactions suspectes, c'est bien, mais un analyste ne va pas faire
confiance à un chiffre qu'il ne peut pas expliquer. Ici chaque score vient
avec ses facteurs SHAP, un historique de compte, une synthèse LLM qui
reformule (jamais qui décide), et la décision finale reste humaine —
justification obligatoire, tout est journalisé.

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, Streamlit,
SQLite, Claude pour la synthèse, Docker, pytest, GitHub Actions.

Ce README couvre l'usage (quickstart, tests, résultats). Le détail du
raisonnement — pourquoi tel choix plutôt qu'un autre, ce qui a été écarté et
pourquoi — est dans [docs/CONCEPTION.md](docs/CONCEPTION.md), je n'ai pas
voulu tout mettre ici pour garder ce fichier lisible.

## À quoi ça ressemble

Un analyste ouvre l'UI, voit la file d'alertes triée par score, clique sur
une transaction : score, seuil courant, facteurs SHAP qui expliquent le
score, historique du compte, et à droite la décision à prendre.

![Vue d'ensemble de l'UI analyste : file d'alertes, score, facteurs SHAP et décision](docs/screenshots/ui_overview.png)

Plus bas, la synthèse LLM (résumé en langage clair des facteurs SHAP, pas
une analyse indépendante) et le journal des décisions déjà prises :

![Synthèse LLM d'investigation et journal d'audit](docs/screenshots/ui_llm_synthesis.png)

Et côté API, la doc Swagger générée par FastAPI si vous préférez taper
directement dans les endpoints :

![Documentation Swagger de l'API FastAPI](docs/screenshots/api_docs.png)

## Données

Le [dataset IBM AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(transactions synthétiques : virements, achats carte, chèques, avec label de
blanchiment) pèse plusieurs Go et nécessite des credentials Kaggle. Pour ce
projet de quelques jours, [`src/riskops/data_gen.py`](src/riskops/data_gen.py)
génère un jeu synthétique avec **exactement le même schéma de colonnes** et un
déséquilibre de classe réaliste (~0.3% de blanchiment). Les labels ne sont pas
tirés ligne par ligne : ils correspondent à des épisodes multi-transactions
portés par un compte (structuration, fan-out, transit rapide). Des rafales
légitimes proches servent de contrôles négatifs difficiles. Le pipeline
(features, entraînement, API) accepte aussi le vrai CSV IBM placé dans
`data/raw_transactions.csv`.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate  # ou source .venv/bin/activate
pip install -r requirements.txt -e .

python src/riskops/data_gen.py        # génère data/raw_transactions.csv
python src/riskops/features.py        # génère data/features.parquet
python src/riskops/train.py           # entraîne LogReg + XGBoost, SHAP, métriques

uvicorn riskops.api:app --reload &     # API sur :8000
streamlit run app_streamlit.py         # UI analyste sur :8501
```

Ou directement avec Docker :

```bash
docker compose up --build
```

Pour activer la synthèse LLM via Claude plutôt que le mode hors-ligne (fallback
déterministe basé sur SHAP), définir `ANTHROPIC_API_KEY` dans l'environnement
avant de lancer l'API.

## Tests

```bash
pytest -v
```

Couverture : signal comportemental du générateur, absence de fuite temporelle
dans les features, métriques métier, journal d'audit (justification/décision
obligatoires), endpoints API
(file d'alertes, détail, décision, 404), synthèse LLM en mode hors-ligne.

## Méthodologie

Quelques décisions qui comptent plus que les autres :

- Split chronologique (70/15/15) train/val/test, jamais un mélange aléatoire
  des lignes. L'idée c'est de simuler un vrai déploiement : le modèle
  n'entraîne que sur du passé et se fait évaluer sur un futur qu'il n'a
  jamais vu.
- Les features "historique de compte" (nombre de transactions antérieures,
  moyenne glissante des montants, contreparties distinctes sur 7 jours,
  nombre et montant cumulé sur 24h, délai depuis le dernier flux) ne regardent
  que le passé de chaque compte. C'est le genre
  de détail qu'on peut louper facilement et qui fausse tout après coup.
- Deux modèles comparés, pas un seul : régression logistique en baseline
  (`class_weight="balanced"`) et XGBoost (`scale_pos_weight` ajusté au
  déséquilibre), pour pouvoir chiffrer l'apport du second plutôt que
  l'affirmer.
- Le champion est choisi sur le PR-AUC de validation, jamais sur le test.
  L'API charge ensuite automatiquement ce champion. SHAP explique chaque
  transaction dans l'API et l'UI.
- Pas d'accuracy comme métrique — avec moins de 1% de positifs, un modèle qui
  répond toujours "non suspect" atteindrait ~99.7% d'accuracy en étant
  complètement inutile. On regarde plutôt le PR-AUC, la precision/recall à
  budget d'investigation constant, et la réduction de faux positifs à recall
  comparable (détail dans [`src/riskops/evaluate.py`](src/riskops/evaluate.py)).

Résultats obtenus sur le jeu de test synthétique (13 jours, 51 transactions
positives ; seed 42, voir `models/metrics.json`) :

| Modèle | PR-AUC test (IC 95% par compte) | Precision @ 20/jour | Recall @ 20/jour |
|---|---|---|---|
| Logistic Regression (baseline) | 0.301 [0.103 ; 0.583] | 17.7% | 90.2% |
| XGBoost | 0.556 [0.237 ; 0.796] | 17.7% | 90.2% |

XGBoost est retenu parce que son PR-AUC de **validation** est supérieur
(0.319 contre 0.265), le test restant hors du critère de sélection. Son avantage
test est important en valeur ponctuelle, mais l'intervalle bootstrap apparié
de la différence contient zéro ([-0.106 ; 0.543]) : avec seulement 51
positifs, on ne prétend donc pas avoir établi statistiquement sa supériorité.
À recall voisin de 50%, il produit 58 alertes contre 71 pour la régression
logistique, soit 29% de faux positifs en moins — là encore un résultat de ce
jeu synthétique, pas une promesse de production.

L'ablation répond directement à la question du signal comportemental : le
même XGBoost limité aux caractéristiques de la transaction courante obtient
0.196 de PR-AUC, contre 0.556 avec l'historique (+0.360). Les fenêtres de
compte apportent donc bien un signal mesurable dans ce générateur.

## Business case : quel seuil pour 20 dossiers/jour ?

Un analyste ne peut traiter qu'un nombre fini de dossiers par jour. Le vrai
levier métier n'est pas "améliorer le modèle" dans l'absolu, mais choisir le
seuil qui maximise les transactions suspectes détectées sous la contrainte de
capacité d'investigation. `src/riskops/evaluate.business_case_sweep` calcule,
pour plusieurs capacités quotidiennes, le seuil correspondant, le recall
obtenu et la précision :

```bash
python src/riskops/train.py   # écrit models/business_case.csv
```

| Capacité (dossiers/jour) | Seuil | Recall | Precision | Positifs détectés / 51 |
|---|---|---|---|---|
| 5 | 0.6884 | 58.8% | 46.2% | 30 |
| 10 | 0.1263 | 72.6% | 28.5% | 37 |
| **20** | **0.0065** | **90.2%** | **17.7%** | **46** |
| 50 | 0.0009 | 100.0% | 7.9% | 51 |
| 100 | 0.0003 | 100.0% | 3.9% | 51 |

Lecture métier : passer de 10 à 20 dossiers/jour retrouve neuf cas
supplémentaires sur la période. Passer de 20 à 50 n'en retrouve que cinq pour
30 investigations quotidiennes supplémentaires. Le seuil associé à 20/jour
est celui utilisé par défaut par l'API. Ces chiffres servent à illustrer le
choix sous contrainte ; ils ne doivent pas être extrapolés hors de ce jeu
synthétique.

Coût/gain : chaque faux positif en moins à recall constant (ici -29% de FP
pour XGBoost vs la baseline, voir plus haut) libère du temps analyste
réinvestissable sur des dossiers à plus fort risque ; chaque vrai positif
détecté en plus évite un risque réglementaire/réputationnel dont le coût
dépasse largement celui d'une investigation individuelle (quelques dizaines de
minutes d'analyste).

## Architecture

```
data_gen.py → features.py → train.py (LogReg + XGBoost + SHAP)
                                   │
                                   ▼
                         models/*.joblib, metrics.json
                                   │
                                   ▼
                    api.py (FastAPI) ── audit.py (SQLite)
                       │                    ▲
                       │                    │ décision + justification
                       ▼                    │
                app_streamlit.py (UI analyste)
                       │
                       ▼
              llm_summary.py (synthèse Claude, schéma Pydantic)
```

## Limites (assumées pour un projet de quelques jours)

- Données synthétiques : les épisodes injectés rendent les tests
  méthodologiques non triviaux, mais ne prouvent aucune performance sur des
  opérations réelles (pas de vérité terrain ni de graphe multi-sauts).
- Pas de ré-entraînement automatique / monitoring de drift.
- La synthèse LLM n'est pas garantie factuellement correcte : elle est
  affichée comme aide à la décision, jamais comme source de vérité — la
  décision et sa justification restent humaines et journalisées.
