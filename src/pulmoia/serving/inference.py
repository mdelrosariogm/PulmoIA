"""
inference.py
============
Fase 4 — Servicio de inferencia: wav → COPD0–4 + perfil acústico.

Carga el bundle autónomo (sin MLflow) y ejecuta la cadena completa:
  wav → preprocesar+ventanear → features → 127 escaladas → detector (79 ReliefF) →
  perfil acústico (prob. media por evento) → modelo COPD → COPD0–4.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from pulmoia import config
from pulmoia.features.extract_tr import extract_window_rows, load_and_preprocess

_BUNDLE = None


def load_bundle() -> dict:
    """Carga (y cachea) el bundle de serving."""
    global _BUNDLE
    if _BUNDLE is None:
        if not config.SERVING_BUNDLE.exists():
            raise FileNotFoundError(
                f"No existe {config.SERVING_BUNDLE}. Genera con "
                "`uv run python -m pulmoia.serving.bundle`."
            )
        with open(config.SERVING_BUNDLE, "rb") as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE


def features_from_wav(path) -> pd.DataFrame:
    """Extrae las features por ventana de un wav (mismo pipeline que la extracción batch)."""
    signal, fs, _sr, _res = load_and_preprocess(str(path))
    return pd.DataFrame(extract_window_rows(signal, fs))


def predict_from_wavs(paths) -> dict:
    """Cadena completa sobre uno o varios wav (se agregan todas sus ventanas)."""
    bundle = load_bundle()
    frames = [features_from_wav(p) for p in paths]
    feats = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if feats.empty:
        raise ValueError("No se extrajeron ventanas de los audios (¿archivos válidos?).")

    feats127 = bundle["feats127"]
    missing = [c for c in feats127 if c not in feats.columns]
    if missing:
        raise ValueError(f"Faltan {len(missing)} features esperadas: {missing[:3]}...")

    X = feats[feats127].fillna(0.0)
    Xs = pd.DataFrame(bundle["scaler"].transform(X), columns=feats127)

    detector = bundle["detector"]
    proba = detector.predict_proba(Xs)   # DataFrame (n, labels)
    pred = detector.predict(Xs)
    labels = bundle["detector_labels"]

    perfil = {
        lab: {"pct_ventanas": round(float(pred[lab].mean()), 4),
              "prob_media": round(float(proba[lab].mean()), 4)}
        for lab in labels
    }
    meanp = {f"meanp_{lab}": float(proba[lab].mean()) for lab in labels}
    x_copd = np.array([[meanp[f] for f in bundle["profile_feats"]]])
    y_copd = int(bundle["copd_model"].predict(x_copd)[0])
    copd = bundle["copd_classes"][y_copd]

    return {
        "copd": copd,
        "n_ventanas": int(len(feats)),
        "n_audios": len(paths),
        "perfil_acustico": perfil,
        "version": bundle.get("version", {}),
    }
