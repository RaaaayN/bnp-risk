"""Régénère les résultats publiés dans le README depuis models/metrics.json."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "metrics.json"
README_PATH = ROOT / "README.md"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _replace_generated_block(text: str, name: str, content: str) -> str:
    start = f"<!-- GENERATED_{name}_START -->"
    end = f"<!-- GENERATED_{name}_END -->"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    replacement = f"{start}\n{content.rstrip()}\n{end}"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"Bloc généré {name} absent ou dupliqué dans README.md")
    return updated


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    logreg = metrics["logreg"]
    xgboost = metrics["xgboost"]
    ablation = metrics["xgboost_transaction_only_ablation"]
    difference_ci = metrics["pr_auc_difference_xgb_minus_logreg_95pct_ci"]
    capacity = metrics["threshold_calibration"]["daily_capacity"]

    results = f"""Le test couvre {metrics['n_days_test']} jours et contient {metrics['business_case'][0]['test_total_positives']} transactions positives.
Voici le résultat reproductible avec la graine aléatoire {metrics['seed']} ; les valeurs brutes sont dans
[`models/metrics.json`](models/metrics.json).

| Modèle | PR-AUC test (IC 95% par compte) | Seuil calibré sur validation | Précision test | Rappel test |
|---|---:|---:|---:|---:|
| Logistic Regression | {logreg['test_pr_auc']:.3f} [{logreg['test_pr_auc_95pct_ci'][0]:.3f} ; {logreg['test_pr_auc_95pct_ci'][1]:.3f}] | {logreg['validation_calibrated_threshold']:.6f} | {_pct(logreg['test_precision_at_fixed_threshold'])} | {_pct(logreg['test_recall_at_fixed_threshold'])} |
| XGBoost | {xgboost['test_pr_auc']:.3f} [{xgboost['test_pr_auc_95pct_ci'][0]:.3f} ; {xgboost['test_pr_auc_95pct_ci'][1]:.3f}] | {xgboost['validation_calibrated_threshold']:.6f} | {_pct(xgboost['test_precision_at_fixed_threshold'])} | {_pct(xgboost['test_recall_at_fixed_threshold'])} |

J'ai retenu XGBoost avant de regarder le test, car son PR-AUC de validation était
meilleur ({xgboost['validation_pr_auc']:.3f} contre {logreg['validation_pr_auc']:.3f}). Une fois le test ouvert, l'avantage est
beaucoup moins net : l'intervalle de confiance de la différence de PR-AUC va
de {difference_ci[0]:.3f} à {difference_ci[1]:.3f} et traverse zéro. Je n'en conclus donc pas que
XGBoost est réellement supérieur.

Au seuil figé, la baseline fait même un peu mieux : {_pct(logreg['test_recall_at_fixed_threshold'])} de rappel et
{_pct(logreg['test_precision_at_fixed_threshold'])} de précision pour LogReg, contre
{_pct(xgboost['test_recall_at_fixed_threshold'])} et {_pct(xgboost['test_precision_at_fixed_threshold'])} pour XGBoost. Je garde le choix fait sur la
validation ; remplacer le modèle après lecture du test serait précisément le
biais que ce découpage cherche à éviter.

Le résultat le plus utile est ailleurs : sans les variables d'historique, le
PR-AUC de XGBoost tombe à {ablation['pr_auc']:.3f}, contre {xgboost['test_pr_auc']:.3f} avec l'historique
(+{ablation['history_feature_pr_auc_lift']:.3f}). Le modèle apprend donc bien une partie du comportement du
compte, et pas seulement le montant courant.

Enfin, le seuil XGBoost {xgboost['validation_calibrated_threshold']:.6f}, prévu pour {capacity} dossiers/jour sur la
validation, n'en produit que {xgboost['test_alerts_per_day']:.2f} sur le test. Ce n'est pas corrigé après coup :
cet écart est une information sur le changement de distribution."""

    rows = []
    for row in metrics["business_case"]:
        rows.append(
            f"| {row['daily_capacity']} | {row['threshold']:.6f} | "
            f"{row['test_alerts_per_day']:.2f} | {_pct(row['test_recall'])} | "
            f"{_pct(row['test_precision'])} | "
            f"{row['test_positives_caught']}/{row['test_total_positives']} |"
        )
    business = """Chaque ligne pose la même question : si cette capacité avait servi à
fixer le seuil sur la validation, qu'aurait-on observé ensuite sur le test ?
Les seuils ne sont jamais recalculés sur le test.

| Capacité cible sur validation | Seuil | Alertes/jour test | Rappel test | Précision test | Positifs détectés |
|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows)

    readme = README_PATH.read_text()
    readme = _replace_generated_block(readme, "METRICS", results)
    readme = _replace_generated_block(readme, "BUSINESS_CASE", business)
    README_PATH.write_text(readme)


if __name__ == "__main__":
    main()
