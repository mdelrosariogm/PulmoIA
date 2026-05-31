"""
copd_profile.py
===============
Paso 2 — Inferencia del detector sobre RespiratoryDatabase@TR → perfil acústico.

Aplica el detector champion (Model Registry) a cada ventana de 1s de los audios TR y
agrega los resultados a un PERFIL ACÚSTICO por audio y por paciente:
  pct_<evento>   = fracción de ventanas con el evento (predicción binaria, umbral por etiqueta)
  meanp_<evento> = probabilidad media del evento sobre las ventanas

Preprocesamiento (idéntico al de entrenamiento del detector):
  features crudas TR → seleccionar las 127 (correlación) → StandardScaler(HF_ICBHI) → 127 escaladas.
El detector internamente selecciona sus 79 features ReliefF por nombre.

Uso:
  uv run python -m pulmoia.models.copd_profile
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import pandas as pd

from pulmoia import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

TR_WINDOWS = config.OUTPUTS_DIR / "features_TR_windows.csv"
SELECTED_127 = config.CLEANING_DIR / "HF_ICBHI" / "selected_features.txt"
SCALER = config.MODELS_DIR / "scaler_HF_ICBHI.pkl"
OUT_DIR = config.OUTPUTS_DIR / "copd"
CHAMPION_URI = f"models:/{config.REGISTERED_MODEL_DETECTOR}@champion"


def load_selected_127() -> list[str]:
    return [
        ln.strip()
        for ln in SELECTED_127.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def preprocess(df: pd.DataFrame, feats127: list[str]):
    """Selecciona las 127 features, imputa y escala con el scaler de HF_ICBHI."""
    missing = [c for c in feats127 if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan {len(missing)} features en TR windows: {missing[:5]}...")
    X = df[feats127].fillna(df[feats127].median())
    scaler = pickle.load(open(SCALER, "rb"))
    Xs = pd.DataFrame(scaler.transform(X), columns=feats127, index=df.index)
    return Xs


def build_profiles():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Cargando ventanas TR: %s", TR_WINDOWS)
    df = pd.read_csv(TR_WINDOWS, low_memory=False)
    log.info(
        "Ventanas=%d | audios=%d | pacientes=%d",
        len(df),
        df["filename"].nunique(),
        df["patient_id"].nunique(),
    )

    feats127 = load_selected_127()
    Xs = preprocess(df, feats127)

    config.setup_mlflow(config.EXPERIMENT_DETECTOR)
    log.info("Cargando detector champion: %s", CHAMPION_URI)
    detector = mlflow.sklearn.load_model(CHAMPION_URI)

    proba = detector.predict_proba(Xs)  # DataFrame (n, 4 labels)
    pred = detector.predict(Xs)  # binario por umbral de etiqueta
    labels = list(proba.columns)

    base = df[["filename", "patient_id", "channel", "diagnosis"]].copy()
    for lab in labels:
        base[f"p_{lab}"] = proba[lab].to_numpy()
        base[f"b_{lab}"] = pred[lab].to_numpy()

    # ── Perfil por audio (filename) ──
    agg = {}
    for lab in labels:
        agg[f"pct_{lab}"] = (f"b_{lab}", "mean")
        agg[f"meanp_{lab}"] = (f"p_{lab}", "mean")
    audio = base.groupby(["filename", "patient_id", "diagnosis"]).agg(**agg).reset_index()
    audio["n_ventanas"] = base.groupby(["filename", "patient_id", "diagnosis"]).size().to_numpy()
    audio.to_csv(OUT_DIR / "profiles_audio.csv", index=False)

    # ── Perfil por paciente (agrega TODAS sus ventanas) ──
    pagg = {}
    for lab in labels:
        pagg[f"pct_{lab}"] = (f"b_{lab}", "mean")
        pagg[f"meanp_{lab}"] = (f"p_{lab}", "mean")
    patient = base.groupby(["patient_id", "diagnosis"]).agg(**pagg).reset_index()
    patient["n_ventanas"] = base.groupby(["patient_id", "diagnosis"]).size().to_numpy()
    patient.to_csv(OUT_DIR / "profiles_patient.csv", index=False)

    log.info("Perfiles guardados en %s (audio=%d, paciente=%d)", OUT_DIR, len(audio), len(patient))
    log.info("Perfil por paciente (medias por clase COPD):")
    cols = [f"pct_{lab}" for lab in labels]
    summary = patient.groupby("diagnosis")[cols].mean().round(3)
    log.info("\n%s", summary.to_string())
    return patient, audio


if __name__ == "__main__":
    build_profiles()
