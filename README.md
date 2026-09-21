# Triage explicable d'alertes AML

## Pourquoi ce projet

Je voulais aller un peu plus loin que le notebook de classification habituel.
Dans un cas d'usage AML, sortir un score ne suffit pas : il faut décider quelles
alertes un analyste aura réellement le temps d'ouvrir, lui montrer ce qui a
fait monter le score, puis garder une trace de sa décision.

Ce dépôt couvre cette chaîne de bout en bout. Il génère des transactions,
construit des variables causales, compare une régression logistique à XGBoost,
calibre un seuil selon une capacité d'investigation, expose les alertes dans
une API et une petite interface, puis journalise la décision humaine. Deux
couches d'IA restent des aides, jamais des décideurs : Jev propose une
orientation avec des probabilités, un LLM rédige une narration, et aucune des
deux ne valide ni ne bloque une transaction.

Je préfère être clair dès le départ : les données sont synthétiques. Ce projet
montre une méthode et une architecture de démonstration, pas la performance
d'un système AML prêt à être déployé dans une banque.

Le code repose surtout sur pandas, scikit-learn, XGBoost, SHAP, FastAPI,
Streamlit et SQLite. Les choix plus détaillés, y compris ceux que j'ai écartés,
sont expliqués dans [la note de conception](docs/CONCEPTION.md).

## Parcours analyste

L'écran principal est volontairement simple : une file d'alertes triée par
score, puis le détail de la transaction sélectionnée avec son historique de
compte et ses principaux facteurs SHAP.

![Vue d'ensemble de l'UI analyste : file d'alertes, score, facteurs SHAP et décision](docs/screenshots/ui_overview.png)

Plus bas, deux blocs distincts. L'**aide à la décision** (Jev) affiche une
orientation non contraignante, sa priorité et, quand Jev répond, les
probabilités par action. La **narration** (LLM) reformule les facteurs en
langage courant. La provenance de chacun (`jev`/`llm` ou `fallback`) est
affichée et conservée dans l'audit. La décision reste celle de l'analyste, avec
une justification obligatoire.

> Les captures ci-dessous datent d'avant la séparation aide à la décision /
> narration : elles montrent l'ancienne synthèse unique.

![Synthèse LLM d'investigation et journal d'audit](docs/screenshots/ui_llm_synthesis.png)

La même file est disponible par API ; FastAPI fournit la documentation Swagger.

![Documentation Swagger de l'API FastAPI](docs/screenshots/api_docs.png)

## Un mot sur les données

Le point de départ est le schéma du
[dataset IBM AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
mais le dépôt reste exécutable sans télécharger plusieurs gigaoctets ni fournir
d'identifiants Kaggle. Le générateur local produit 120 000 transactions sur 90
jours, dont environ 0,3 % de positives.

Les labels ne sont pas tirés indépendamment à chaque ligne. Ils appartiennent
à des épisodes portés par un compte : structuration, dispersion vers plusieurs
contreparties ou transit rapide. J'ajoute aussi des rafales légitimes qui
ressemblent à ces scénarios, faute de quoi le modèle apprendrait seulement
« gros montant + virement ». Les tests vérifient que les historiques de compte
contiennent effectivement du signal.

Le loader accepte également un CSV au schéma IBM placé dans
`data/raw_transactions.csv`. Cela signifie que le pipeline peut le lire, pas
que les performances obtenues ici se transféreraient telles quelles aux
données IBM ou à des transactions réelles.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate              # Windows : .venv\Scripts\activate
pip install -r requirements.txt -e .

python src/riskops/data_gen.py        # génère data/raw_transactions.csv
python src/riskops/features.py        # génère data/features.parquet
python src/riskops/train.py           # entraîne LogReg + XGBoost, SHAP, métriques

uvicorn riskops.api:app --reload &     # API sur :8000
streamlit run app_streamlit.py         # UI analyste sur :8501
```

La même chose avec Docker :

```bash
docker compose up --build
```

Sans clé LLM, l'application utilise un résumé déterministe construit
à partir des facteurs SHAP. Avec `ANTHROPIC_API_KEY`, elle appelle Claude ; à défaut, avec
`GEMINI_API_KEY`, elle appelle Gemini (Claude a la priorité si les deux sont définies). Dans les deux
cas, le schéma de sortie est le même et la provenance est enregistrée.

L'orientation Clear / Investigate / Escalate, la priorité et la probabilité de revue
humaine viennent de Jev (SDK officiel `typesafe-sdk`) quand
`TYPESAFE_API_KEY` est défini, sinon de règles
déterministes sur le score. Dans tous les cas l'analyste décide.

## Configuration

Aucune clé n'est obligatoire : sans elles, tout fonctionne en mode dégradé
déterministe. Les variables se placent dans l'environnement du processus ; un
fichier `.env` à la racine (ignoré par git) est lu par `docker compose`, mais
**pas** par un lancement local d'`uvicorn`, où il faut les exporter.

| Variable | Rôle | Sans elle |
|---|---|---|
| `TYPESAFE_API_KEY` | Aide à la décision via Jev ([console.typesafe.ai](https://console.typesafe.ai/)) | Règles déterministes sur le score |
| `ANTHROPIC_API_KEY` | Narration via Claude (prioritaire) | Essaie Gemini, sinon narration déterministe |
| `GEMINI_API_KEY` | Narration via Gemini si Claude n'est pas configuré | Narration déterministe |

## Tests

```bash
ruff check .
pytest -v
```

La suite couvre notamment le signal du générateur, la causalité des fenêtres
temporelles, les métriques à seuil figé, l'audit SQLite, les endpoints API et
les fournisseurs Jev et LLM (le vrai SDK Jev est exercé contre un transport HTTP
simulé : succès, 401, 500, réponse invalide, coupure réseau) et leurs fallbacks.
Aucun test n'appelle un service réel, même si des clés sont exportées. La CI relance aussi tout l'entraînement et vérifie que les
métriques publiées restent reproductibles.

## Les choix qui comptent vraiment

Quelques détails ont beaucoup plus d'impact que le choix entre deux modèles :

- Le découpage train/validation/test est chronologique (70/15/15). Le futur ne
  revient jamais dans l'entraînement.
- Les fenêtres de compte ne voient que les transactions antérieures : volume
  et nombre de flux sur 24 h, contreparties sur sept jours, montant habituel et
  délai depuis le dernier mouvement.
- La validation sert trois fois : choisir le modèle, arrêter XGBoost et fixer
  le seuil opérationnel. Le test ne sert qu'au reporting final.
- La régression logistique est une vraie baseline, pas une ligne ajoutée pour
  la forme. Si XGBoost ne fait pas mieux, le README le dit.
- L'accuracy n'est pas publiée. Avec 0,3 % de positifs, elle serait excellente
  même pour un modèle qui ne remonte aucune alerte. Je regarde plutôt le
  PR-AUC, le rappel, la précision et le volume d'alertes à seuil figé.

Les calculs sont dans [`evaluate.py`](src/riskops/evaluate.py). Les intervalles
de confiance sont obtenus par bootstrap des comptes plutôt que des lignes, car
plusieurs transactions d'un même épisode ne sont pas indépendantes.

## Résultats, sans les embellir

<!-- GENERATED_METRICS_START -->
Le test couvre 14 jours et contient 51 transactions positives.
Voici le résultat reproductible avec la graine aléatoire 42 ; les valeurs brutes sont dans
[`models/metrics.json`](models/metrics.json).

| Modèle | PR-AUC test (IC 95% par compte) | Seuil calibré sur validation | Précision test | Rappel test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.301 [0.103 ; 0.583] | 0.742299 | 22.0% | 78.4% |
| XGBoost | 0.357 [0.109 ; 0.638] | 0.593909 | 20.9% | 72.5% |

J'ai retenu XGBoost avant de regarder le test, car son PR-AUC de validation était
meilleur (0.334 contre 0.265). Une fois le test ouvert, l'avantage est
beaucoup moins net : l'intervalle de confiance de la différence de PR-AUC va
de -0.251 à 0.315 et traverse zéro. Je n'en conclus donc pas que
XGBoost est réellement supérieur.

Au seuil figé, la baseline fait même un peu mieux : 78.4% de rappel et
22.0% de précision pour LogReg, contre
72.5% et 20.9% pour XGBoost. Je garde le choix fait sur la
validation ; remplacer le modèle après lecture du test serait précisément le
biais que ce découpage cherche à éviter.

Le résultat le plus utile est ailleurs : sans les variables d'historique, le
PR-AUC de XGBoost tombe à 0.178, contre 0.357 avec l'historique
(+0.179). Le modèle apprend donc bien une partie du comportement du
compte, et pas seulement le montant courant.

Enfin, le seuil XGBoost 0.593909, prévu pour 20 dossiers/jour sur la
validation, n'en produit que 12.64 sur le test. Ce n'est pas corrigé après coup :
cet écart est une information sur le changement de distribution.
<!-- GENERATED_METRICS_END -->

## Du score à une file de travail

Un seuil abstrait n'aide pas beaucoup une équipe d'investigation. Je pars donc
d'une question plus concrète : combien de dossiers peut-elle ouvrir chaque
jour ? J'ai pris 20 comme hypothèse de travail, puis j'ai calibré le seuil sur
la validation. Ce seuil reste ensuite inchangé sur le test, même si le volume
d'alertes s'éloigne de 20. C'est précisément l'écart que l'on voudrait voir en
cas de changement de distribution.

```bash
python src/riskops/train.py   # écrit models/business_case.csv
```

<!-- GENERATED_BUSINESS_CASE_START -->
Chaque ligne pose la même question : si cette capacité avait servi à
fixer le seuil sur la validation, qu'aurait-on observé ensuite sur le test ?
Les seuils ne sont jamais recalculés sur le test.

| Capacité cible sur validation | Seuil | Alertes/jour test | Rappel test | Précision test | Positifs détectés |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.949983 | 3.07 | 37.2% | 44.2% | 19/51 |
| 10 | 0.886270 | 7.14 | 56.9% | 29.0% | 29/51 |
| 20 | 0.593909 | 12.64 | 72.5% | 20.9% | 37/51 |
| 50 | 0.132952 | 32.21 | 88.2% | 10.0% | 45/51 |
| 100 | 0.053822 | 78.86 | 100.0% | 4.6% | 51/51 |
<!-- GENERATED_BUSINESS_CASE_END -->

Ce tableau sert à montrer le compromis. Il ne permet pas de conclure qu'une
équipe réelle devrait traiter 20 dossiers par jour, ni que les mêmes seuils
fonctionneraient sur une autre population.

## Aide à la décision avec Jev : ce que ça apporte, et ce que ça n'apporte pas

Jusqu'ici le LLM faisait deux métiers : rédiger une explication et choisir
Clear / Investigate / Escalate avec un niveau de confiance. Le second est une
petite décision typée, pas de la génération de texte. Les rôles sont maintenant
séparés :

```
XGBoost prédit → SHAP explique → Jev oriente → le LLM raconte → l'analyste décide
```

[`jev_decision.py`](src/riskops/jev_decision.py) pose quatre questions à Jev
([TypeSafe](https://docs.typesafe.ai/), SDK officiel `typesafe-sdk`) en un seul
appel : l'action recommandée (probabilités par option), une priorité sur cinq
niveaux ramenée à 0-10, la probabilité qu'une revue humaine soit indispensable
et la cohérence du schéma de signaux. Le LLM n'émet plus ni action ni confiance.
Si Jev est absent ou en échec (timeout de 10 s, un retry), des règles
déterministes prennent le relais et la source `fallback` est tracée ; dans ce
mode les probabilités restent vides plutôt que fabriquées.

**Mesuré, sans embellir.** Sur les 177 alertes de la file de test locale (40 vrais
positifs), Jev n'améliore pas la détection :

| | Score XGBoost | Jev P(Escalate) | Jev priorité |
|---|---|---|---|
| ROC-AUC dans la file | 0.703 | 0.687 | 0.677 |
| PR-AUC dans la file | 0.545 | 0.503 | 0.412 |

Escalades : 105 avec la règle `score ≥ 0.85` (précision 0.29, 30/40 vrais
positifs captés) contre 113 avec Jev (précision 0.27, 30/40). Les deux sont
d'accord dans 93 % des cas, et la probabilité de revue humaine ne discrimine pas
(0.52 sur les vrais positifs, 0.57 sur les faux). Jev ne reçoit aujourd'hui que le
score, le seuil, les facteurs SHAP et la taille de l'historique : il reproduit
donc surtout le score. Il ajoute environ 0,6 s par alerte ; côté narration, Gemini
prend 7 à 11 s.

L'intérêt est **architectural** : rôles séparés, probabilités par action
affichées, et un audit qui enregistre la source de chaque aide et si l'analyste
s'en est écarté (`human_overrode_recommendation`), ce qui permettra de mesurer
les désaccords. Ce n'est pas un gain de performance ni de vitesse. Ces chiffres
viennent d'un environnement local dont les versions (numpy, xgboost, shap)
diffèrent de `requirements.txt` ; ils sont indicatifs, pas des métriques de
référence. Piste pour un vrai gain : fournir à Jev les signaux bruts (montant
relatif à l'habituel, contreparties distinctes, délai) plutôt que le seul score.

## Comment les pièces s'enchaînent

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
          ┌────────────┴─────────────┐
          ▼                          ▼
 jev_decision.py              llm_summary.py
 (orientation typée,          (narration Claude/Gemini,
  fallback règles)             fallback SHAP)
```

## Ce que ce projet ne prouve pas

- Les scénarios suspects ont été écrits à la main. Ils permettent de tester le
  pipeline, mais ils ne remplacent ni une vérité terrain ni une validation par
  des spécialistes AML.
- Le test ne contient que 51 positifs. Les intervalles de confiance sont larges
  et la supériorité de XGBoost n'est pas établie.
- Le générateur crée des épisodes par compte, pas un véritable réseau de
  layering multi-sauts. Il n'y a ni feature de graphe ni détection de cycles.
- Le seuil est un exemple de calibration, pas un seuil de production. Il
  faudrait le suivre dans le temps avec le volume d'alertes, le drift et le
  retour des analystes.
- L'API de démonstration n'a pas d'authentification ni de gestion des rôles.
- Une narration LLM peut être maladroite ou fausse. Sa source est visible et
  auditée ; la décision et sa justification restent humaines.
- Jev n'a pas été montré meilleur que la règle de seuil (voir plus haut) : ses
  probabilités affichent une incertitude, elles ne prouvent pas une meilleure
  détection. Une orientation Jev est une aide, jamais une décision.
- Le champ `llm_synthesis` de `/alerts/{id}` est devenu `narrative`, et
  `decision_support` a été ajouté : changement d'API pour d'éventuels clients.
