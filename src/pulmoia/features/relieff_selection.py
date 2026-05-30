"""
relieff_selection.py
====================
Selección de características por **ReliefF** para el detector multi-etiqueta.

ReliefF pondera cada feature según qué tan bien distingue vecinos cercanos de distinta
clase (capta relevancia e interacciones). Se corre **por etiqueta** (cada target binario)
sobre las 127 features ya limpiadas por correlación y estandarizadas; luego se agregan los
pesos normalizados y se selecciona el subconjunto que cubre el `coverage` del peso total.

Submuestreo balanceado por etiqueta: imprescindible para clases ultra-raras (stridor ≈ 0.3%),
de modo que ReliefF "vea" suficientes positivos.

Uso:
  uv run python -m pulmoia.features.relieff_selection                 # ranking + reentrena
  uv run python -m pulmoia.features.relieff_selection --no-retrain    # solo ranking
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skrebate import ReliefF

from pulmoia import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = config.OUTPUTS_DIR / "feature_selection" / "relieff"
SELECTED_PATH = OUT_DIR / "selected_features.txt"


def balanced_subsample(df, feats, target, max_total=5000, max_pos=1500, seed=42):
    """Submuestra equilibrando el target (todos los positivos hasta un tope + negativos)."""
    rng = np.random.RandomState(seed)
    pos = df.index[df[target] == 1].to_numpy()
    neg = df.index[df[target] == 0].to_numpy()
    if len(pos) > max_pos:
        pos = rng.choice(pos, max_pos, replace=False)
    n_neg = min(len(neg), max(max_total - len(pos), len(pos)))
    neg = rng.choice(neg, n_neg, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    X = df.loc[idx, feats].fillna(df[feats].median()).to_numpy()
    y = df.loc[idx, target].astype(int).to_numpy()
    return X, y, len(pos), len(neg)


def compute_relieff_weights(train, feats, n_neighbors=100, seed=42) -> pd.DataFrame:
    """Pesos ReliefF por etiqueta. Devuelve DataFrame (feature × [labels..., aggregate])."""
    weights = {}
    for t in config.TARGETS:
        X, y, n_pos, n_neg = balanced_subsample(train, feats, t, seed=seed)
        log.info("ReliefF '%s': subsample pos=%d neg=%d", t, n_pos, n_neg)
        k = min(n_neighbors, n_pos - 1 if n_pos > 1 else 1)
        rf = ReliefF(n_neighbors=max(k, 1), n_jobs=-1)
        rf.fit(X, y)
        weights[t] = rf.feature_importances_

    W = pd.DataFrame(weights, index=feats)
    # Normaliza cada etiqueta: clip a [0,∞) y divide por el máximo positivo
    norm = W.clip(lower=0)
    norm = norm / norm.max().replace(0, np.nan)
    norm = norm.fillna(0.0)
    W["aggregate"] = norm.mean(axis=1)
    return W.sort_values("aggregate", ascending=False)


def select_by_coverage(W: pd.DataFrame, coverage=0.90, min_features=20) -> list[str]:
    """Selecciona las features cuyo peso agregado acumulado cubre `coverage` del total."""
    agg = W["aggregate"].to_numpy()
    total = agg.sum()
    if total <= 0:
        return list(W.index[:min_features])
    cum = np.cumsum(agg) / total
    k = int(np.searchsorted(cum, coverage) + 1)
    k = max(k, min_features)
    return list(W.index[:k])


def save_artifacts(W: pd.DataFrame, selected: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    W.to_csv(OUT_DIR / "relieff_weights.csv")
    SELECTED_PATH.write_text(
        f"# Features seleccionados por ReliefF (cobertura de peso agregado)\n"
        f"# Total: {len(selected)} de {len(W)}\n\n" + "\n".join(selected),
        encoding="utf-8",
    )
    # Gráfico: top-30 por peso agregado
    top = W.head(30)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(top.index[::-1], top["aggregate"][::-1], color="teal")
    ax.set_title("ReliefF — Top 30 features (peso agregado multi-etiqueta)")
    ax.set_xlabel("Peso ReliefF normalizado (media sobre etiquetas)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "relieff_top30.png", dpi=150)
    plt.close(fig)
    log.info("Guardado: %s | %s | relieff_top30.png", "relieff_weights.csv", SELECTED_PATH.name)


def run_selection(n_neighbors=100, coverage=0.90) -> list[str]:
    train = pd.read_csv(config.DETECTOR_DATA_DIR / "train_scaled.csv", low_memory=False)
    feats = config.feature_columns(train)
    log.info("Calculando ReliefF sobre %d features, %d etiquetas...", len(feats), len(config.TARGETS))
    W = compute_relieff_weights(train, feats, n_neighbors=n_neighbors)
    selected = select_by_coverage(W, coverage=coverage)
    save_artifacts(W, selected)
    log.info("Seleccionadas %d/%d features (cobertura %.0f%%). Top-5: %s",
             len(selected), len(feats), coverage * 100, selected[:5])
    return selected


def main():
    parser = argparse.ArgumentParser(description="Selección de features por ReliefF (detector)")
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--coverage", type=float, default=0.90,
                        help="fracción del peso agregado a cubrir (0-1)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="solo generar el ranking, sin reentrenar")
    parser.add_argument("--from-saved", action="store_true",
                        help="reutilizar selected_features.txt (no recalcular ReliefF)")
    args = parser.parse_args()

    if args.from_saved and SELECTED_PATH.exists():
        selected = [ln.strip() for ln in SELECTED_PATH.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")]
        log.info("Reutilizando %d features de %s", len(selected), SELECTED_PATH.name)
    else:
        selected = run_selection(n_neighbors=args.neighbors, coverage=args.coverage)

    if not args.no_retrain:
        from pulmoia.models.train_detector import run_training
        log.info("=" * 60)
        log.info("Reentrenando detector con %d features ReliefF...", len(selected))
        results = run_training(
            ["xgboost", "logreg"], n_iter=15, cv=3, sample=15000,
            feature_subset=selected, feature_set="relieff",
        )
        log.info("=" * 60)
        log.info("Comparativa (Macro ROC-AUC):")
        log.info("  XGBoost  ReliefF(%d feats) = %.4f   | all(127) = 0.8240",
                 len(selected), results.get("xgboost", float("nan")))
        log.info("  LogReg   ReliefF(%d feats) = %.4f   | all(127) = 0.8093",
                 len(selected), results.get("logreg", float("nan")))


if __name__ == "__main__":
    main()
