"""
preprocessing.py
================
Fase 3.2 — Flow de datos (ETL) y validación con Prefect.

Parte de los CSV de features ya extraídos. Por defecto **reutiliza** los datos preparados
(outputs/cleaning/HF_ICBHI/{train,test}_scaled.csv) y los **valida**; opcionalmente puede
**regenerarlos** invocando `pulmoia.data.preparation` como subproceso (transformación
reproducible: limpieza por correlación → split por archivo → estandarización).
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
from prefect import flow, get_run_logger, task

from pulmoia import config
from pulmoia.data.validation import validate_prepared_detector_data


@task(name="regenerar-datos-preparados")
def regenerate_prepared(db: str = "combined") -> str:
    """Regenera los datos preparados ejecutando el preprocesamiento (subproceso)."""
    logger = get_run_logger()
    cmd = [sys.executable, "-m", "pulmoia.data.preparation", "--db", db]
    logger.info("Ejecutando: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=config.PROJECT_ROOT)
    if res.returncode != 0:
        logger.error(res.stderr[-2000:])
        raise RuntimeError(f"Preprocesamiento falló (rc={res.returncode})")
    return "ok"


@task(name="cargar-datos-preparados")
def load_prepared(data_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or config.DETECTOR_DATA_DIR
    train = pd.read_csv(data_dir / "train_scaled.csv", low_memory=False)
    test = pd.read_csv(data_dir / "test_scaled.csv", low_memory=False)
    return train, test


@task(name="validar-datos")
def validate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    logger = get_run_logger()
    report = validate_prepared_detector_data(train, test)
    logger.info("Validación OK | train=%d test=%d filas",
                report["train"]["n_rows"], report["test"]["n_rows"])
    return report


@flow(name="preprocesamiento-detector")
def preprocessing_flow(regenerate: bool = False, db: str = "combined") -> dict:
    """ETL + validación de los datos del detector.

    Parameters
    ----------
    regenerate : si True, regenera los datos preparados antes de validar.
    db : base a regenerar (por defecto 'combined' = HF+ICBHI).
    """
    logger = get_run_logger()
    if regenerate:
        regenerate_prepared(db)
    train, test = load_prepared()
    report = validate(train, test)
    logger.info("Preprocesamiento completado.")
    return report


if __name__ == "__main__":
    preprocessing_flow()
