"""
baseline.py
===========
Fase 1.3 — Establecimiento del baseline de rendimiento.

Dos baselines de referencia (modelos triviales + regresión logística), uno por cada
"cabeza" del producto. NO son los modelos finales (eso es Fase 2); fijan el piso contra
el que se comparará todo modelo posterior.

1) Detector multilabel (HF + ICBHI)
   - Datos: outputs/cleaning/HF_ICBHI/{train,test}_scaled.csv (split por archivo → sin leakage)
   - Targets: has_wheeze, has_crackle, has_stridor, has_rhonchus
   - Métrica primaria: Macro ROC-AUC. Secundarias: Macro F1, PR-AUC por label.

2) Clasificador COPD0–4 (RespiratoryDatabase@TR)
   - Datos: outputs/features_TR.csv (una fila por archivo-canal)
   - Validación: StratifiedGroupKFold por patient_id (sin leakage entre canales del paciente)
   - Métricas: balanced accuracy, Macro F1, MAE ordinal, matriz de confusión.

Uso:
  uv run python -m pulmoia.models.baseline
  uv run python -m pulmoia.models.baseline --only detector
  uv run python -m pulmoia.models.baseline --only copd
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────
DETECTOR_DIR = Path("outputs/cleaning/HF_ICBHI")
TR_CSV = Path("outputs/features_TR.csv")
OUT_DIR = Path("outputs/baseline")

TARGETS = ["has_wheeze", "has_crackle", "has_stridor", "has_rhonchus"]
META_COLS = [
    "filename", "source", "patient_id", "location", "channel", "mode",
    "equipment", "t_start_s", "t_end_s", "sr_original", "resampled",
    "diagnosis", "duration_s",
]
COPD_ORDER = {"COPD0": 0, "COPD1": 1, "COPD2": 2, "COPD3": 3, "COPD4": 4}
RANDOM_STATE = 42
N_SPLITS_COPD = 5


def sep(title=""):
    log.info("=" * 70)
    if title:
        log.info("  %s", title)
        log.info("=" * 70)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas numéricas de features (excluye meta y targets)."""
    cols = [c for c in df.columns if c not in META_COLS and c not in TARGETS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


# ─── BASELINE 1: DETECTOR MULTILABEL ──────────────────────────────────────────
def run_detector_baseline() -> dict:
    sep("BASELINE 1 — DETECTOR MULTILABEL (HF + ICBHI)")
    train = pd.read_csv(DETECTOR_DIR / "train_scaled.csv")
    test = pd.read_csv(DETECTOR_DIR / "test_scaled.csv")
    log.info("Train: %d filas | Test: %d filas", len(train), len(test))

    feats = feature_columns(train)
    log.info("Features: %d", len(feats))
    X_tr = train[feats].fillna(train[feats].median())
    X_te = test[feats].fillna(train[feats].median())
    y_tr = train[TARGETS].astype(int)
    y_te = test[TARGETS].astype(int)

    log.info("Prevalencia en test:")
    for t in TARGETS:
        log.info("  %-14s %5.2f%%", t, 100 * y_te[t].mean())

    models = {
        "dummy_prior": MultiOutputClassifier(
            DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
        ),
        "logreg": MultiOutputClassifier(
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
    }

    rows = []
    for name, model in models.items():
        log.info("Entrenando %s ...", name)
        model.fit(X_tr, y_tr)
        # probas: lista de arrays (n_muestras, 2) por target
        probas = model.predict_proba(X_te)
        preds = model.predict(X_te)
        per_label = {}
        for i, t in enumerate(TARGETS):
            p_pos = probas[i][:, 1]
            try:
                auc = roc_auc_score(y_te[t], p_pos)
            except ValueError:
                auc = np.nan
            try:
                ap = average_precision_score(y_te[t], p_pos)
            except ValueError:
                ap = np.nan
            f1 = f1_score(y_te[t], preds[:, i], zero_division=0)
            per_label[t] = {"roc_auc": auc, "pr_auc": ap, "f1": f1}
            rows.append({"model": name, "label": t, "roc_auc": auc, "pr_auc": ap, "f1": f1})

        macro_auc = np.nanmean([per_label[t]["roc_auc"] for t in TARGETS])
        macro_f1 = np.mean([per_label[t]["f1"] for t in TARGETS])
        rows.append(
            {"model": name, "label": "MACRO", "roc_auc": macro_auc, "pr_auc": np.nan, "f1": macro_f1}
        )
        log.info("  %-12s Macro ROC-AUC=%.4f | Macro F1=%.4f", name, macro_auc, macro_f1)

    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "baseline_detector.csv", index=False)
    log.info("Guardado: %s", OUT_DIR / "baseline_detector.csv")

    macro = res[(res.model == "logreg") & (res.label == "MACRO")].iloc[0]
    return {
        "detector_logreg_macro_roc_auc": float(macro.roc_auc),
        "detector_logreg_macro_f1": float(macro.f1),
        "table": res,
    }


# ─── BASELINE 2: CLASIFICADOR COPD ────────────────────────────────────────────
def run_copd_baseline() -> dict:
    sep("BASELINE 2 — CLASIFICADOR COPD0–4 (RespiratoryDatabase@TR)")
    df = pd.read_csv(TR_CSV)
    log.info("Filas: %d | Pacientes: %d", len(df), df["patient_id"].nunique())
    log.info("Distribución diagnosis:\n%s", df["diagnosis"].value_counts().to_string())

    feats = feature_columns(df)
    log.info("Features: %d", len(feats))
    X = df[feats].fillna(df[feats].median()).to_numpy()
    y = df["diagnosis"].map(COPD_ORDER).to_numpy()
    groups = df["patient_id"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=N_SPLITS_COPD, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "dummy_most_frequent": lambda: DummyClassifier(strategy="most_frequent"),
        "logreg": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    rows = []
    oof_true, oof_pred = {n: [] for n in models}, {n: [] for n in models}
    for name, make in models.items():
        log.info("Modelo: %s", name)
        for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
            scaler = StandardScaler().fit(X[tr_idx])  # fit solo en train del fold
            clf = make().fit(scaler.transform(X[tr_idx]), y[tr_idx])
            pred = clf.predict(scaler.transform(X[te_idx]))
            bacc = balanced_accuracy_score(y[te_idx], pred)
            mf1 = f1_score(y[te_idx], pred, average="macro", zero_division=0)
            mae = np.mean(np.abs(y[te_idx] - pred))
            rows.append(
                {"model": name, "fold": fold, "balanced_acc": bacc, "macro_f1": mf1, "mae_ordinal": mae}
            )
            oof_true[name].extend(y[te_idx])
            oof_pred[name].extend(pred)
        sub = pd.DataFrame([r for r in rows if r["model"] == name])
        log.info(
            "  %-20s bAcc=%.3f±%.3f | MacroF1=%.3f | MAE=%.3f",
            name, sub.balanced_acc.mean(), sub.balanced_acc.std(),
            sub.macro_f1.mean(), sub.mae_ordinal.mean(),
        )

    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "baseline_copd.csv", index=False)
    log.info("Guardado: %s", OUT_DIR / "baseline_copd.csv")

    # Matriz de confusión out-of-fold del logreg
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        oof_true["logreg"], oof_pred["logreg"],
        display_labels=list(COPD_ORDER.keys()), ax=ax, colorbar=False,
    )
    ax.set_title("Baseline COPD (LogReg) — Confusión out-of-fold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "copd_confusion_matrix.png", dpi=150)
    plt.close(fig)
    log.info("Guardado: %s", OUT_DIR / "copd_confusion_matrix.png")

    lr = res[res.model == "logreg"]
    return {
        "copd_logreg_balanced_acc": float(lr.balanced_acc.mean()),
        "copd_logreg_macro_f1": float(lr.macro_f1.mean()),
        "copd_logreg_mae_ordinal": float(lr.mae_ordinal.mean()),
        "table": res,
    }


def write_report(det: dict | None, copd: dict | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Fase 1.3 — Reporte de Baseline\n"]
    summary = {}

    if det is not None:
        lines += [
            "## Detector multilabel (HF + ICBHI)\n",
            "Métrica primaria: **Macro ROC-AUC**. Evaluado en test split por archivo.\n",
            det["table"].to_markdown(index=False),
            "",
            f"- **Baseline LogReg → Macro ROC-AUC = {det['detector_logreg_macro_roc_auc']:.4f}**",
            "- Objetivo del proyecto: ≥ 0.88 (MVP ≥ 0.80)\n",
        ]
        summary.update({k: v for k, v in det.items() if k != "table"})

    if copd is not None:
        lines += [
            "## Clasificador COPD0–4 (TR)\n",
            "Validación: StratifiedGroupKFold por patient_id.\n",
            copd["table"].drop(columns=["fold"]).groupby("model").mean(numeric_only=True).round(4).to_markdown(),
            "",
            f"- **Baseline LogReg → Balanced Acc = {copd['copd_logreg_balanced_acc']:.3f}, "
            f"Macro F1 = {copd['copd_logreg_macro_f1']:.3f}, "
            f"MAE ordinal = {copd['copd_logreg_mae_ordinal']:.3f}**",
            "- Azar (5 clases) ≈ 0.20 balanced acc.\n",
        ]
        summary.update({k: v for k, v in copd.items() if k != "table"})

    (OUT_DIR / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Guardado: %s", OUT_DIR / "baseline_report.md")
    log.info("Guardado: %s", OUT_DIR / "baseline_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Baseline de rendimiento (Fase 1.3)")
    parser.add_argument(
        "--only", choices=["detector", "copd"], default=None,
        help="Ejecutar solo un baseline (por defecto: ambos)",
    )
    args = parser.parse_args()

    det = run_detector_baseline() if args.only in (None, "detector") else None
    copd = run_copd_baseline() if args.only in (None, "copd") else None
    write_report(det, copd)
    sep("BASELINE COMPLETADO")


if __name__ == "__main__":
    main()
