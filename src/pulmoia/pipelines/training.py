"""
training.py
===========
Fase 3.1 / 3.3 — Flow de entrenamiento del detector con Prefect.

Orquesta: validación de datos → entrenamiento (XGBoost + LogReg) → registro/promoción.
Random Forest queda fuera del pipeline (su búsqueda de ~2 h no es apta para retraining).
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from pulmoia.models.register_model import register_best
from pulmoia.models.train_detector import run_training
from pulmoia.pipelines.preprocessing import load_prepared, validate


@task(name="entrenar-detector")
def train_task(algos: list[str], n_iter: int, cv: int, sample: int) -> dict:
    logger = get_run_logger()
    results = run_training(algos, n_iter=n_iter, cv=cv, sample=sample)
    for algo, auc in sorted(results.items(), key=lambda kv: -kv[1]):
        logger.info("  %-15s Macro ROC-AUC=%.4f", algo, auc)
    return results


@task(name="registrar-modelo")
def register_task(production: str, staging: str | None) -> dict:
    logger = get_run_logger()
    out = register_best(production=production, staging=staging)
    logger.info("Registrado: %s", out)
    return out


@flow(name="entrenamiento-detector")
def training_flow(
    algos: list[str] | None = None,
    n_iter: int = 15,
    cv: int = 3,
    sample: int = 15000,
    do_register: bool = True,
    production: str = "xgboost",
    staging: str | None = None,
) -> dict:
    """Pipeline de entrenamiento del detector multilabel.

    Por defecto entrena XGBoost + LogReg y promueve XGBoost a Production.
    """
    logger = get_run_logger()
    algos = algos or ["xgboost", "logreg"]

    train, test = load_prepared()
    validate(train, test)  # falla rápido si los datos no cumplen el contrato

    results = train_task(algos, n_iter, cv, sample)

    registry = None
    if do_register:
        registry = register_task(production, staging)
    logger.info(
        "Entrenamiento completado. Mejor=%s", max(results, key=results.get) if results else "—"
    )
    return {"results": results, "registry": registry}


if __name__ == "__main__":
    training_flow()
