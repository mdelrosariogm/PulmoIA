"""
inference.py
============
Fase 4 — Servicio de inferencia: wav → COPD0–4 + perfil acústico.

Carga el bundle autónomo (sin MLflow) y ejecuta la cadena completa:
  wav → preprocesar+ventanear → features → 127 escaladas → detector (79 ReliefF) →
  perfil acústico (prob. media por evento) → modelo COPD → COPD0–4.
"""

from __future__ import annotations

import io
import pickle

import librosa
import numpy as np
import pandas as pd
from scipy.signal import sosfilt

from pulmoia import config
from pulmoia.features.extract_tr import (
    _HP_SOS,
    TARGET_SR,
    extract_window_rows,
    load_and_preprocess,
)

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


def features_from_bytes(audio_bytes: bytes) -> pd.DataFrame:
    """Igual que features_from_wav pero desde bytes en memoria (uploads de la web)."""
    signal, sr_orig = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    if sr_orig != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr_orig, target_sr=TARGET_SR)
    signal = sosfilt(_HP_SOS, signal)  # mismo pasa-altos que el pipeline batch
    return pd.DataFrame(extract_window_rows(signal, TARGET_SR))


def _predict_from_features(feats: pd.DataFrame, n_audios: int) -> dict:
    """Cadena: features por ventana → detector → perfil → COPD0–4 + confianza."""
    bundle = load_bundle()
    if feats.empty:
        raise ValueError("No se extrajeron ventanas del audio (¿archivo válido?).")

    feats127 = bundle["feats127"]
    missing = [c for c in feats127 if c not in feats.columns]
    if missing:
        raise ValueError(f"Faltan {len(missing)} features esperadas: {missing[:3]}...")

    Xs = pd.DataFrame(bundle["scaler"].transform(feats[feats127].fillna(0.0)), columns=feats127)
    detector = bundle["detector"]
    proba = detector.predict_proba(Xs)
    pred = detector.predict(Xs)
    labels = bundle["detector_labels"]

    perfil = {
        lab: {
            "pct_ventanas": round(float(pred[lab].mean()), 4),
            "prob_media": round(float(proba[lab].mean()), 4),
        }
        for lab in labels
    }
    meanp = {f"meanp_{lab}": float(proba[lab].mean()) for lab in labels}
    x_copd = np.array([[meanp[f] for f in bundle["profile_feats"]]])

    copd_model = bundle["copd_model"]
    y_copd = int(copd_model.predict(x_copd)[0])
    if hasattr(copd_model, "predict_proba"):
        probs = copd_model.predict_proba(x_copd)[0]
        confidence = float(probs[list(copd_model.classes_).index(y_copd)])
    else:
        confidence = float("nan")

    return {
        "copd": bundle["copd_classes"][y_copd],
        "copd_level": y_copd,
        "confidence": round(confidence, 4),
        "n_ventanas": int(len(feats)),
        "n_audios": n_audios,
        "perfil_acustico": perfil,
        "version": bundle.get("version", {}),
    }


def predict_from_wavs(paths) -> dict:
    """Cadena completa sobre uno o varios wav (se agregan todas sus ventanas)."""
    frames = [features_from_wav(p) for p in paths]
    feats = (
        pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    return _predict_from_features(feats, len(paths))


def predict_from_audio_bytes(audio_bytes: bytes) -> dict:
    """Cadena completa sobre el contenido de un audio en memoria (upload web)."""
    return _predict_from_features(features_from_bytes(audio_bytes), 1)
