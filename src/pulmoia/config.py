"""
config.py
=========
Configuración central del proyecto: rutas, constantes y setup de MLflow.

Todas las rutas se resuelven respecto a la raíz del proyecto (no al CWD), de modo que
los módulos funcionen sin importar desde dónde se invoquen.
"""

from __future__ import annotations

from pathlib import Path

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
# config.py vive en src/pulmoia/, la raíz está 2 niveles arriba.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
CLEANING_DIR = OUTPUTS_DIR / "cleaning"
DETECTOR_DATA_DIR = CLEANING_DIR / "HF_ICBHI"
TR_CSV = OUTPUTS_DIR / "features_TR.csv"

# ─── MLflow ───────────────────────────────────────────────────────────────────
# SQLite habilita el Model Registry (Fase 2.3). Artefactos en ./mlartifacts.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"

EXPERIMENT_DETECTOR = "detector_adventicios"
REGISTERED_MODEL_DETECTOR = "pulmoia_detector"

# ─── Esquema de datos ─────────────────────────────────────────────────────────
TARGETS = ["has_wheeze", "has_crackle", "has_stridor", "has_rhonchus"]
META_COLS = [
    "filename", "source", "patient_id", "location", "channel", "mode",
    "equipment", "t_start_s", "t_end_s", "sr_original", "resampled",
    "diagnosis", "duration_s",
]
GROUP_COL = "filename"  # agrupación anti-leakage (ventanas del mismo audio)

COPD_ORDER = {"COPD0": 0, "COPD1": 1, "COPD2": 2, "COPD3": 3, "COPD4": 4}
RANDOM_STATE = 42


def setup_mlflow(experiment: str = EXPERIMENT_DETECTOR) -> str:
    """Configura el tracking de MLflow (SQLite local) y selecciona el experimento.

    Devuelve el experiment_id.
    """
    import mlflow

    MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = mlflow.create_experiment(
            experiment,
            artifact_location=(MLFLOW_ARTIFACT_DIR / experiment).as_uri(),
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id


def feature_columns(df) -> list[str]:
    """Columnas numéricas de features (excluye meta y targets)."""
    import pandas as pd

    cols = [c for c in df.columns if c not in META_COLS and c not in TARGETS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
