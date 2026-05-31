"""
copd_classifier.py
==================
Paso 3 — Asignación de severidad COPD0–4 desde el perfil acústico.

Dos enfoques comparados sobre el perfil por paciente (pct_<evento>):
  1) ESTADÍSTICO: ANOVA (qué eventos difieren entre niveles COPD) + NearestCentroid
     (asignar al prototipo de clase más cercano). Sin entrenamiento complejo, interpretable.
  2) ML: Regresión logística multinomial y árbol de decisión.

Validación: LeaveOneOut por paciente (cada paciente es una muestra; ~40 pacientes).
Métricas: balanced accuracy, Macro F1, MAE ordinal, matriz de confusión.
Se loguea en MLflow (experimento 'copd_clasificador').

Uso:
  uv run python -m pulmoia.models.copd_classifier
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pulmoia import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROFILES = config.OUTPUTS_DIR / "copd" / "profiles_patient.csv"
OUT_DIR = config.OUTPUTS_DIR / "copd"
EXPERIMENT_COPD = "copd_clasificador"
# Probabilidades medias continuas (más informativas que el % binario). Se descarta
# stridor porque el detector nunca lo dispara en TR (meanp ≈ 0, feature degenerada).
PROFILE_FEATS = ["meanp_has_wheeze", "meanp_has_crackle", "meanp_has_rhonchus"]


def anova_report(df: pd.DataFrame) -> pd.DataFrame:
    """ANOVA de cada feature de perfil entre los niveles COPD."""
    rows = []
    groups_by_class = [g for _, g in df.groupby("diagnosis")]
    for feat in PROFILE_FEATS:
        samples = [g[feat].to_numpy() for g in groups_by_class]
        F, p = f_oneway(*samples)
        rows.append(
            {
                "feature": feat,
                "F": round(F, 3),
                "p_value": round(p, 4),
                "significativo_0.05": p < 0.05,
            }
        )
    return pd.DataFrame(rows)


def evaluate(name, estimator, X, y, groups_labels):
    """LeaveOneOut + métricas + matriz de confusión. Devuelve (metrics, y_pred)."""
    cv = LeaveOneOut()
    y_pred = cross_val_predict(estimator, X, y, cv=cv)
    bacc = balanced_accuracy_score(y, y_pred)
    mf1 = f1_score(y, y_pred, average="macro", zero_division=0)
    mae = float(np.mean(np.abs(y - y_pred)))
    log.info("  %-22s bAcc=%.3f | MacroF1=%.3f | MAE=%.3f", name, bacc, mf1, mae)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred,
        display_labels=groups_labels,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"COPD — {name} (LeaveOneOut)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"copd_confusion_{name}.png", dpi=150)
    plt.close(fig)
    return {"model": name, "balanced_acc": bacc, "macro_f1": mf1, "mae_ordinal": mae}, y_pred


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROFILES)
    log.info(
        "Perfiles: %d pacientes | clases: %s",
        len(df),
        dict(df["diagnosis"].value_counts().sort_index()),
    )

    X = df[PROFILE_FEATS].to_numpy()
    y = df["diagnosis"].map(config.COPD_ORDER).to_numpy()
    labels = list(config.COPD_ORDER.keys())

    # ── ANOVA ──
    av = anova_report(df)
    av.to_csv(OUT_DIR / "copd_anova.csv", index=False)
    log.info("ANOVA (perfil acústico vs nivel COPD):\n%s", av.to_string(index=False))

    # ── Modelos ──
    models = {
        "estadistico_centroide": make_pipeline(StandardScaler(), NearestCentroid()),
        "ml_logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
            ),
        ),
        "ml_arbol": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
    }

    config.setup_mlflow(EXPERIMENT_COPD)
    rows = []
    log.info("Evaluación (LeaveOneOut por paciente):")
    for name, est in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"task": "copd_severity", "method": name})
            mlflow.log_params({"n_patients": len(df), "features": ",".join(PROFILE_FEATS)})
            metrics, _ = evaluate(name, est, X, y, labels)
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
            mlflow.log_artifact(str(OUT_DIR / f"copd_confusion_{name}.png"))
            rows.append(metrics)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "copd_resultados.csv", index=False)
    log.info("=" * 60)
    log.info("Comparativa final:\n%s", res.round(3).to_string(index=False))

    # Reporte markdown
    best = res.sort_values("balanced_acc", ascending=False).iloc[0]
    md = [
        "# Paso 3 — Asignación COPD0–4 (perfil acústico)\n",
        f"Pacientes: {len(df)}. Validación: LeaveOneOut. Features: {', '.join(PROFILE_FEATS)}.\n",
        "## ANOVA (perfil vs nivel COPD)\n",
        av.to_markdown(index=False),
        "",
        "## Comparativa de métodos\n",
        res.round(3).to_markdown(index=False),
        "",
        f"**Mejor por balanced accuracy:** {best['model']} "
        f"(bAcc={best['balanced_acc']:.3f}, MacroF1={best['macro_f1']:.3f}, "
        f"MAE={best['mae_ordinal']:.3f}).",
        "\n> Nota: con ~40 pacientes los resultados tienen alta varianza; interpretar con cautela.",
    ]
    (OUT_DIR / "copd_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Reporte: %s", OUT_DIR / "copd_report.md")


if __name__ == "__main__":
    main()
