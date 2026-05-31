"""
register_model.py
=================
Fase 2.3 — Model Registry.

Registra varias versiones del detector en el Model Registry y las promueve:
  - XGBoost      → Production (alias 'champion')   [mejor opción práctica]
  - RandomForest → Staging    (alias 'challenger') [mejor Macro ROC-AUC, más lento]

Cada versión se busca como el mejor run padre (estrategia OvR) de su algoritmo, por
Macro ROC-AUC en test. Se versiona con tags y descripción.

Uso:
  uv run python -m pulmoia.models.register_model
  uv run python -m pulmoia.models.register_model --production xgboost --staging random_forest
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient

from pulmoia import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def best_run_for_algo(client: MlflowClient, exp_id: str, algo: str, feature_set: str | None = None):
    """Mejor run padre OvR *completo* (run_type=full) de un algoritmo por Macro ROC-AUC.

    Excluye runs de prueba (run_type='smoke'). Si `feature_set` se indica, filtra por ese
    conjunto de features (p. ej. 'relieff' o 'all').
    """
    fs = f" and tags.feature_set = '{feature_set}'" if feature_set else ""
    runs = client.search_runs(
        [exp_id],
        filter_string=(
            f"tags.strategy = 'OvR' and tags.algo = '{algo}' and tags.run_type = 'full'{fs}"
        ),
        order_by=["metrics.macro_test_roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(
            f"No hay runs 'full' OvR para algo='{algo}' feature_set='{feature_set}'. "
            "Entrena (no smoke) primero."
        )
    return runs[0]


def register_and_tag(client: MlflowClient, run, alias: str, stage: str, archive_existing=False):
    """Registra el modelo de un run, lo etiqueta y lo promueve (stage + alias)."""
    algo = run.data.tags.get("algo", "?")
    feature_set = run.data.tags.get("feature_set", "all")
    macro_auc = run.data.metrics.get("macro_test_roc_auc", float("nan"))
    macro_f1 = run.data.metrics.get("macro_test_f1", float("nan"))
    n_features = run.data.params.get("n_features", "?")

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", config.REGISTERED_MODEL_DETECTOR)
    name, ver = config.REGISTERED_MODEL_DETECTOR, mv.version

    for k, v in {
        "algo": algo,
        "feature_set": feature_set,
        "n_features": str(n_features),
        "macro_test_roc_auc": f"{macro_auc:.4f}",
        "macro_test_f1": f"{macro_f1:.4f}",
        "dataset": "HF_ICBHI",
    }.items():
        client.set_model_version_tag(name, ver, k, v)

    client.update_model_version(
        name,
        ver,
        description=(
            f"Detector multilabel OvR ({algo}, feature_set={feature_set}, {n_features} feats). "
            f"Macro ROC-AUC test={macro_auc:.4f}, Macro F1={macro_f1:.4f}. "
            "HF+ICBHI, split por archivo, umbral por etiqueta."
        ),
    )
    client.set_registered_model_alias(name, alias, ver)
    try:
        client.transition_model_version_stage(
            name, ver, stage=stage, archive_existing_versions=archive_existing
        )
        log.info(
            "  %s v%s [%s/%s] → stage=%s, alias='%s' (Macro AUC=%.4f)",
            name,
            ver,
            algo,
            feature_set,
            stage,
            alias,
            macro_auc,
        )
    except Exception as e:
        log.warning(
            "  Stage no aplicado (%s); alias '%s' sí. (Macro AUC=%.4f)", e, alias, macro_auc
        )
    return mv


def register_best(
    production: str = "xgboost",
    staging: str | None = "random_forest",
    production_feature_set: str | None = None,
    staging_feature_set: str | None = None,
) -> dict:
    """Registra y promueve el detector. Reutilizable por el pipeline de Prefect (Fase 3).

    `*_feature_set` permite distinguir variantes (p. ej. 'relieff' vs 'all') del mismo algoritmo.
    Devuelve {'production': (algo, version), 'staging': (algo, version) | None}.
    """
    config.setup_mlflow(config.EXPERIMENT_DETECTOR)
    client = MlflowClient()
    exp = client.get_experiment_by_name(config.EXPERIMENT_DETECTOR)
    if exp is None:
        raise SystemExit("No existe el experimento. Ejecuta train_detector primero.")

    try:
        client.create_registered_model(
            config.REGISTERED_MODEL_DETECTOR,
            description="Detector multilabel de sonidos adventicios (wheeze/crackle/stridor/rhonchus).",
        )
    except Exception:
        pass  # ya existe

    log.info("Registrando versiones en '%s':", config.REGISTERED_MODEL_DETECTOR)
    out = {}
    prod_run = best_run_for_algo(client, exp.experiment_id, production, production_feature_set)
    mv_p = register_and_tag(
        client, prod_run, alias="champion", stage="Production", archive_existing=True
    )
    out["production"] = (production, mv_p.version)

    if staging and (staging, staging_feature_set) != (production, production_feature_set):
        stg_run = best_run_for_algo(client, exp.experiment_id, staging, staging_feature_set)
        mv_s = register_and_tag(client, stg_run, alias="challenger", stage="Staging")
        out["staging"] = (staging, mv_s.version)

    log.info(
        "Registry actualizado. UI: uv run mlflow ui --backend-store-uri %s",
        config.MLFLOW_TRACKING_URI,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Registrar y promover detectores")
    parser.add_argument(
        "--production", default="xgboost", choices=["xgboost", "random_forest", "logreg"]
    )
    parser.add_argument(
        "--staging", default="random_forest", choices=["xgboost", "random_forest", "logreg"]
    )
    parser.add_argument(
        "--production-feature-set",
        default=None,
        help="filtra el run de producción por feature_set (p. ej. 'relieff')",
    )
    parser.add_argument("--staging-feature-set", default=None)
    args = parser.parse_args()
    register_best(
        production=args.production,
        staging=args.staging,
        production_feature_set=args.production_feature_set,
        staging_feature_set=args.staging_feature_set,
    )


if __name__ == "__main__":
    main()
