"""
bundle.py
=========
Fase 4 — Empaqueta el modelo servible en un único archivo autónomo (sin MLflow en runtime).

El bundle contiene todo lo necesario para la cadena wav → COPD0–4 + perfil:
  - detector champion (MultiLabelDetector, 79 features ReliefF)
  - scaler de HF_ICBHI + lista de 127 features (preprocesamiento del detector)
  - modelo COPD (NearestCentroid) entrenado sobre TODOS los perfiles de pacientes
  - metadatos (etiquetas de eventos, clases COPD, features de perfil)

Uso:
  uv run python -m pulmoia.serving.bundle      # genera models/serving_bundle.pkl
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pulmoia import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SELECTED_127 = config.CLEANING_DIR / "HF_ICBHI" / "selected_features.txt"
SCALER = config.MODELS_DIR / "scaler_HF_ICBHI.pkl"
PROFILES = config.OUTPUTS_DIR / "copd" / "profiles_patient.csv"
CHAMPION_URI = f"models:/{config.REGISTERED_MODEL_DETECTOR}@champion"


def build_bundle() -> dict:
    feats127 = [
        ln.strip()
        for ln in SELECTED_127.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    scaler = pickle.load(open(SCALER, "rb"))

    config.setup_mlflow(config.EXPERIMENT_DETECTOR)
    log.info("Cargando detector champion: %s", CHAMPION_URI)
    detector = mlflow.sklearn.load_model(CHAMPION_URI)

    # Modelo COPD entrenado sobre TODOS los pacientes (producción)
    df = pd.read_csv(PROFILES)
    X = df[config.COPD_PROFILE_FEATS].to_numpy()
    y = df["diagnosis"].map(config.COPD_ORDER).to_numpy()
    # LogReg multinomial: da predict_proba → confianza real para la web.
    copd_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
    )
    copd_model.fit(X, y)
    log.info("Modelo COPD (LogReg multinomial) entrenado con %d pacientes", len(df))

    bundle = {
        "detector": detector,
        "scaler": scaler,
        "feats127": feats127,
        "detector_labels": config.TARGETS,
        "copd_model": copd_model,
        "profile_feats": config.COPD_PROFILE_FEATS,
        "copd_classes": list(config.COPD_ORDER.keys()),
        "version": {"detector": "pulmoia_detector@champion", "copd": "logreg_all"},
    }
    return bundle


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle()
    with open(config.SERVING_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Bundle guardado: %s", config.SERVING_BUNDLE)


if __name__ == "__main__":
    main()
