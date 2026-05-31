"""
drift.py
========
Fase 5 — Monitoreo de data drift (demo funcional).

Compara la distribución de las features entre un conjunto de REFERENCIA (el train del
detector, HF+ICBHI) y uno ACTUAL (p. ej. datos nuevos de TR), por feature:
  - PSI (Population Stability Index): <0.1 sin drift · 0.1–0.25 moderado · >0.25 significativo
  - KS test (Kolmogorov–Smirnov): p<0.05 ⇒ distribuciones distintas

Genera un reporte (CSV + markdown + gráfico). Sin dependencias externas más allá de scipy.

Uso:
  uv run python -m pulmoia.monitoring.drift                 # referencia=train vs actual=TR
  uv run python -m pulmoia.monitoring.drift --current test  # control: train vs test (poco drift)
"""

from __future__ import annotations

import argparse
import logging
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pulmoia import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RELIEFF_FEATS = config.OUTPUTS_DIR / "feature_selection" / "relieff" / "selected_features.txt"
SELECTED_127 = config.CLEANING_DIR / "HF_ICBHI" / "selected_features.txt"
SCALER = config.MODELS_DIR / "scaler_HF_ICBHI.pkl"
TR_WINDOWS = config.OUTPUTS_DIR / "features_TR_windows.csv"
OUT_DIR = config.OUTPUTS_DIR / "monitoring"


def _read_list(path):
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre dos muestras de una feature."""
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e_perc = np.histogram(expected, bins=quantiles)[0] / len(expected)
    a_perc = np.histogram(actual, bins=quantiles)[0] / len(actual)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def classify_psi(v: float) -> str:
    return "sin_drift" if v < 0.1 else ("moderado" if v < 0.25 else "significativo")


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = []
    for f in feats:
        e, a = ref[f].dropna().to_numpy(), cur[f].dropna().to_numpy()
        p = psi(e, a)
        ks_stat, ks_p = ks_2samp(e, a)
        rows.append({"feature": f, "psi": round(p, 4), "drift": classify_psi(p),
                     "ks_stat": round(float(ks_stat), 4), "ks_p": round(float(ks_p), 4)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _load_reference_and_current(current: str):
    """Devuelve (ref_df, cur_df) con las 79 features ReliefF escaladas."""
    relieff = _read_list(RELIEFF_FEATS)
    feats127 = _read_list(SELECTED_127)
    train = pd.read_csv(config.DETECTOR_DATA_DIR / "train_scaled.csv", low_memory=False)
    ref = train[relieff]  # ya escaladas

    if current == "test":
        cur_raw = pd.read_csv(config.DETECTOR_DATA_DIR / "test_scaled.csv", low_memory=False)
        cur = cur_raw[relieff]
    else:  # 'tr' → preprocesar igual que el detector (127 → scaler → 79)
        tr = pd.read_csv(TR_WINDOWS, low_memory=False)
        scaler = pickle.load(open(SCALER, "rb"))
        scaled = pd.DataFrame(scaler.transform(tr[feats127].fillna(tr[feats127].median())),
                              columns=feats127)
        cur = scaled[relieff]
    return ref, cur, relieff


def main():
    parser = argparse.ArgumentParser(description="Monitoreo de data drift (Fase 5)")
    parser.add_argument("--current", choices=["tr", "test"], default="tr",
                        help="conjunto actual a comparar contra el train de referencia")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref, cur, feats = _load_reference_and_current(args.current)
    log.info("Referencia (train)=%d filas | Actual (%s)=%d filas | features=%d",
             len(ref), args.current, len(cur), len(feats))

    res = compute_drift(ref, cur, feats)
    res.to_csv(OUT_DIR / f"drift_{args.current}.csv", index=False)

    n_sig = int((res.drift == "significativo").sum())
    n_mod = int((res.drift == "moderado").sum())
    log.info("Drift: %d significativo, %d moderado, %d sin drift (de %d features)",
             n_sig, n_mod, len(res) - n_sig - n_mod, len(res))
    log.info("Top-10 features con más drift:\n%s",
             res.head(10).to_string(index=False))

    # Gráfico top-20 PSI
    top = res.head(20)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"significativo": "#dc2626", "moderado": "#f59e0b", "sin_drift": "#22c55e"}
    ax.barh(top.feature[::-1], top.psi[::-1],
            color=[colors[d] for d in top.drift[::-1]])
    ax.axvline(0.1, ls="--", c="#f59e0b", lw=1)
    ax.axvline(0.25, ls="--", c="#dc2626", lw=1)
    ax.set_title(f"Data drift PSI — train vs {args.current} (top 20)")
    ax.set_xlabel("PSI")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"drift_{args.current}.png", dpi=150)
    plt.close(fig)

    verdict = ("ALERTA: drift significativo de datos" if n_sig >= max(3, len(feats) // 5)
               else "OK: drift dentro de lo tolerable")
    log.info("Veredicto: %s", verdict)
    log.info("Reporte: %s", OUT_DIR / f"drift_{args.current}.csv")


if __name__ == "__main__":
    main()
