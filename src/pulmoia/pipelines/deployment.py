"""
deployment.py
=============
Fase 3.1 — Scheduling (demo) del pipeline de retraining con Prefect.

Crea un deployment del flow principal con un cron semanal (lunes 03:00) y levanta un
proceso que lo sirve. Mientras este proceso esté activo, Prefect dispara el pipeline
según el cron; además permite disparos manuales desde la UI/CLI.

Uso:
  # Servir el deployment con schedule (bloquea; Ctrl+C para detener)
  uv run python -m pulmoia.pipelines.deployment

  # Ver la UI de Prefect en otra terminal
  uv run prefect server start

Disparo manual de una corrida (sin esperar al cron), en otra terminal:
  uv run prefect deployment run 'pulmoia-pipeline/retraining-semanal'
"""

from __future__ import annotations

from pulmoia.pipelines.main_flow import full_pipeline

CRON_SEMANAL = "0 3 * * 1"  # lunes 03:00


def main():
    # Parámetros conservadores para el retraining programado.
    full_pipeline.serve(
        name="retraining-semanal",
        cron=CRON_SEMANAL,
        parameters={
            "algos": ["xgboost", "logreg"],
            "n_iter": 15,
            "cv": 3,
            "sample": 15000,
            "do_register": True,
        },
        tags=["pulmoia", "detector", "retraining"],
        description="Retraining semanal del detector multilabel (XGBoost + LogReg).",
    )


if __name__ == "__main__":
    main()
