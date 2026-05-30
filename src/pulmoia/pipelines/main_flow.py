"""
main_flow.py
============
Fase 3 — Flow principal: encadena preprocesamiento (ETL + validación) y entrenamiento.

Disparo manual:
  uv run python -m pulmoia.pipelines.main_flow
  uv run python -m pulmoia.pipelines.main_flow --smoke      # validación rápida

Scheduling (demo): ver pulmoia/pipelines/deployment.py
"""

from __future__ import annotations

import argparse

from prefect import flow, get_run_logger

from pulmoia.pipelines.preprocessing import preprocessing_flow
from pulmoia.pipelines.training import training_flow


@flow(name="pulmoia-pipeline")
def full_pipeline(
    regenerate: bool = False,
    algos: list[str] | None = None,
    n_iter: int = 15,
    cv: int = 3,
    sample: int = 15000,
    do_register: bool = True,
) -> dict:
    """Pipeline completo: datos validados → entrenamiento → registro."""
    logger = get_run_logger()
    report = preprocessing_flow(regenerate=regenerate)
    result = training_flow(
        algos=algos, n_iter=n_iter, cv=cv, sample=sample, do_register=do_register,
    )
    logger.info("Pipeline pulmoia finalizado.")
    return {"validation": report, "training": result}


def main():
    parser = argparse.ArgumentParser(description="Flow principal pulmoia (Fase 3)")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerar datos preparados antes de entrenar")
    parser.add_argument("--smoke", action="store_true",
                        help="validación rápida (xgboost, búsqueda mínima, sin registro)")
    args = parser.parse_args()

    if args.smoke:
        full_pipeline(algos=["xgboost"], n_iter=2, cv=2, sample=3000, do_register=False)
    else:
        full_pipeline(regenerate=args.regenerate)


if __name__ == "__main__":
    main()
