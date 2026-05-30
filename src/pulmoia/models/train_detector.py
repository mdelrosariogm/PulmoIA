"""
train_detector.py
=================
Fase 2.2 — Experimentos del detector multilabel con MLflow.

Por cada algoritmo (logreg, random_forest, xgboost) y cada etiqueta:
  1. RandomizedSearchCV con GroupKFold por `filename` (anti-leakage), scoring=ROC-AUC.
  2. Selección de umbral por F1 sobre predicciones out-of-fold (sin tocar test).
  3. Refit de los mejores hiperparámetros sobre TODO el train.
  4. Evaluación en test (split por archivo) y logging en MLflow.

Estructura MLflow:
  Experimento "detector_adventicios"
    └── run padre (por algoritmo) → métricas macro + modelo ensamblado (artefacto)
          └── runs anidados (por etiqueta) → hiperparámetros + métricas por clase

Uso:
  uv run python -m pulmoia.models.train_detector                 # RF + XGB + LogReg
  uv run python -m pulmoia.models.train_detector --algos xgboost
  uv run python -m pulmoia.models.train_detector --smoke         # validación rápida
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from xgboost import XGBClassifier

from pulmoia import config
from pulmoia.models.detector import MultiLabelDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Definición de algoritmos y espacios de búsqueda ──────────────────────────
def make_estimator_and_space(algo: str, scale_pos_weight: float):
    """Devuelve (estimador_base, espacio_de_búsqueda) para un algoritmo y una etiqueta."""
    rs = config.RANDOM_STATE
    if algo == "logreg":
        est = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs)
        space = {"C": loguniform(1e-2, 1e2)}
    elif algo == "random_forest":
        est = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=rs)
        space = {
            "n_estimators": randint(150, 400),
            "max_depth": randint(6, 24),
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_leaf": randint(1, 12),
        }
    elif algo == "xgboost":
        est = XGBClassifier(
            tree_method="hist", eval_metric="logloss", n_jobs=-1,
            random_state=rs, scale_pos_weight=scale_pos_weight,
        )
        space = {
            "n_estimators": randint(200, 600),
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(1e-2, 3e-1),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 8),
        }
    else:
        raise ValueError(f"Algoritmo desconocido: {algo}")
    return est, space


def best_threshold(y_true, proba) -> float:
    """Umbral que maximiza F1 sobre (y_true, proba) — usado con predicciones OOF."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # precision_recall_curve devuelve len(thr) = len(prec) - 1
    if len(thr) == 0:
        return 0.5
    return float(thr[np.argmax(f1[:-1])])


def train_one_label(algo, label, X_tr, y_tr, groups, X_te, y_te, X_search, y_search, g_search,
                    n_iter, n_splits):
    """Tunea, selecciona umbral, refit en train completo y evalúa en test. Loguea en MLflow."""
    pos = int(y_search.sum())
    neg = int(len(y_search) - pos)
    spw = neg / max(pos, 1)
    est, space = make_estimator_and_space(algo, scale_pos_weight=spw)

    cv = GroupKFold(n_splits=n_splits)
    search = RandomizedSearchCV(
        est, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
        n_jobs=-1, random_state=config.RANDOM_STATE, refit=True, error_score="raise",
    )
    search.fit(X_search, y_search, groups=g_search)
    cv_auc = float(search.best_score_)

    # Umbral por F1 sobre predicciones out-of-fold (sin leakage de test)
    oof = cross_val_predict(
        search.best_estimator_, X_search, y_search, cv=cv, groups=g_search,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    thr = best_threshold(y_search, oof)

    # Refit de los mejores hiperparámetros sobre TODO el train
    final_est, _ = make_estimator_and_space(algo, scale_pos_weight=spw)
    final_est.set_params(**search.best_params_)
    final_est.fit(X_tr, y_tr)

    # Evaluación en test
    p_te = final_est.predict_proba(X_te)[:, 1]
    test_auc = roc_auc_score(y_te, p_te) if y_te.nunique() > 1 else float("nan")
    test_ap = average_precision_score(y_te, p_te) if y_te.nunique() > 1 else float("nan")
    test_f1 = f1_score(y_te, (p_te >= thr).astype(int), zero_division=0)

    with mlflow.start_run(run_name=f"{algo}__{label}", nested=True):
        mlflow.set_tags({"algo": algo, "label": label, "stage": "experiment"})
        mlflow.log_params({f"hp_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_param("scale_pos_weight", round(spw, 3))
        mlflow.log_metrics({
            "cv_roc_auc": cv_auc,
            "test_roc_auc": test_auc,
            "test_pr_auc": test_ap,
            "test_f1": test_f1,
            "threshold": thr,
            "pos_rate_test": float(y_te.mean()),
        })
    log.info(
        "  [%s] %-13s cv_auc=%.4f test_auc=%.4f test_f1=%.3f thr=%.2f",
        algo, label, cv_auc, test_auc, test_f1, thr,
    )
    return final_est, thr, {"cv_roc_auc": cv_auc, "test_roc_auc": test_auc,
                            "test_pr_auc": test_ap, "test_f1": test_f1}


def train_algo(algo, data, n_iter, n_splits):
    """Entrena el detector OvR completo para un algoritmo. Devuelve (macro_auc, detector)."""
    X_tr, y_tr_df, groups, X_te, y_te_df, feats = data
    X_s, y_s_df, g_s = data_search(data)

    estimators, thresholds, per_label = {}, {}, {}
    with mlflow.start_run(run_name=algo) as parent:
        mlflow.set_tags({"algo": algo, "strategy": "OvR", "stage": "experiment"})
        mlflow.log_params({"n_iter": n_iter, "cv_splits": n_splits, "n_features": len(feats),
                           "n_train": len(X_tr), "n_search": len(X_s)})
        for label in config.TARGETS:
            est, thr, m = train_one_label(
                algo, label, X_tr, y_tr_df[label], groups, X_te, y_te_df[label],
                X_s, y_s_df[label], g_s, n_iter, n_splits,
            )
            estimators[label], thresholds[label], per_label[label] = est, thr, m

        macro_auc = float(np.nanmean([per_label[t]["test_roc_auc"] for t in config.TARGETS]))
        macro_f1 = float(np.mean([per_label[t]["test_f1"] for t in config.TARGETS]))
        mlflow.log_metrics({"macro_test_roc_auc": macro_auc, "macro_test_f1": macro_f1})

        detector = MultiLabelDetector(
            estimators=estimators, feature_names=feats,
            labels=config.TARGETS, thresholds=thresholds,
        )
        example = pd.DataFrame(X_te[:5], columns=feats)
        signature = infer_signature(example, detector.predict(example))
        mlflow.sklearn.log_model(
            detector, artifact_path="model", signature=signature, input_example=example,
        )
        mlflow.set_tag("macro_test_roc_auc", f"{macro_auc:.4f}")
        log.info(">>> %s  Macro ROC-AUC=%.4f | Macro F1=%.4f (run %s)",
                 algo, macro_auc, macro_f1, parent.info.run_id)
    return macro_auc, detector


def data_search(data):
    """Submuestra el train para la búsqueda de hiperparámetros (acelera el CV)."""
    X_tr, y_tr_df, groups, *_ = data
    n = len(X_tr)
    sample = getattr(data_search, "sample_n", None)
    if sample is None or sample >= n:
        return X_tr, y_tr_df, groups
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(n, size=sample, replace=False)
    return X_tr[idx], y_tr_df.iloc[idx].reset_index(drop=True), groups[idx]


def load_data():
    train = pd.read_csv(config.DETECTOR_DATA_DIR / "train_scaled.csv", low_memory=False)
    test = pd.read_csv(config.DETECTOR_DATA_DIR / "test_scaled.csv", low_memory=False)
    feats = config.feature_columns(train)
    X_tr = train[feats].fillna(train[feats].median()).to_numpy()
    X_te = test[feats].fillna(train[feats].median()).to_numpy()
    y_tr = train[config.TARGETS].astype(int).reset_index(drop=True)
    y_te = test[config.TARGETS].astype(int).reset_index(drop=True)
    groups = train[config.GROUP_COL].to_numpy()
    return X_tr, y_tr, groups, X_te, y_te, feats


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del detector multilabel (Fase 2.2)")
    parser.add_argument("--algos", nargs="+",
                        default=["logreg", "random_forest", "xgboost"],
                        choices=["logreg", "random_forest", "xgboost"])
    parser.add_argument("--n-iter", type=int, default=15, help="iteraciones de RandomizedSearch")
    parser.add_argument("--cv", type=int, default=3, help="folds de GroupKFold")
    parser.add_argument("--sample", type=int, default=15000,
                        help="submuestra para la búsqueda (0 = usar todo el train)")
    parser.add_argument("--smoke", action="store_true",
                        help="validación rápida (sample chico, n_iter=2, solo xgboost)")
    args = parser.parse_args()

    if args.smoke:
        args.algos, args.n_iter, args.cv, args.sample = ["xgboost"], 2, 2, 3000

    config.setup_mlflow(config.EXPERIMENT_DETECTOR)
    data = load_data()
    data_search.sample_n = None if args.sample == 0 else args.sample
    log.info("Train=%d Test=%d Features=%d | algos=%s n_iter=%d cv=%d sample=%s",
             len(data[0]), len(data[3]), len(data[5]), args.algos, args.n_iter, args.cv,
             data_search.sample_n)

    results = {}
    for algo in args.algos:
        macro_auc, _ = train_algo(algo, data, args.n_iter, args.cv)
        results[algo] = macro_auc

    log.info("=" * 60)
    best = max(results, key=results.get)
    for a, s in sorted(results.items(), key=lambda kv: -kv[1]):
        marca = "  <== MEJOR" if a == best else ""
        log.info("  %-15s Macro ROC-AUC=%.4f%s", a, s, marca)
    log.info("Tracking: %s", config.MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    main()
