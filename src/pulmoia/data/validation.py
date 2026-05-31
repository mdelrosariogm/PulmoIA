"""
validation.py
=============
Validación de datos ligera (Fase 3.2). Aserciones explícitas, sin dependencias externas.

Comprueba contratos de los datos antes de entrenar:
  - esquema esperado (columnas de features, targets, columna de grupo)
  - ausencia de columnas completamente nulas
  - balance/prevalencia de cada etiqueta (avisa si una clase queda sin positivos)
  - **sin fuga de grupos** entre train y test (ningún `filename` en ambos)
"""

from __future__ import annotations

import logging

import pandas as pd

from pulmoia import config

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Error de validación de datos (rompe el pipeline)."""


def _require(condition: bool, msg: str, errors: list[str]):
    if not condition:
        errors.append(msg)


def validate_features_frame(df: pd.DataFrame, name: str, expect_targets: bool = True) -> dict:
    """Valida un DataFrame de features. Devuelve un reporte; lanza si hay errores duros."""
    errors: list[str] = []
    warnings: list[str] = []

    # Columna de grupo (anti-leakage)
    _require(
        config.GROUP_COL in df.columns,
        f"[{name}] falta la columna de grupo '{config.GROUP_COL}'",
        errors,
    )

    # Features numéricos
    feats = config.feature_columns(df)
    _require(len(feats) > 0, f"[{name}] no se hallaron columnas de features numéricas", errors)

    # Columnas totalmente nulas
    all_null = [c for c in feats if df[c].isna().all()]
    _require(not all_null, f"[{name}] columnas 100% nulas: {all_null}", errors)

    # Nulos parciales → warning
    n_null = int(df[feats].isna().any(axis=1).sum())
    if n_null:
        warnings.append(f"[{name}] {n_null} filas con algún nulo (se imputarán por mediana)")

    # Targets y prevalencia
    prevalence = {}
    if expect_targets:
        for t in config.TARGETS:
            _require(t in df.columns, f"[{name}] falta el target '{t}'", errors)
            if t in df.columns:
                p = float(df[t].astype(float).mean())
                prevalence[t] = p
                if df[t].nunique() < 2:
                    warnings.append(f"[{name}] target '{t}' sin variación (una sola clase)")
                elif p < 0.005:
                    warnings.append(f"[{name}] target '{t}' muy raro ({p:.2%})")

    if errors:
        for e in errors:
            log.error(e)
        raise DataValidationError(f"Validación fallida en '{name}': {errors}")
    for w in warnings:
        log.warning(w)

    report = {
        "name": name,
        "n_rows": len(df),
        "n_features": len(feats),
        "n_groups": int(df[config.GROUP_COL].nunique()) if config.GROUP_COL in df else 0,
        "prevalence": prevalence,
        "warnings": warnings,
    }
    log.info(
        "[%s] OK: %d filas, %d features, %d grupos",
        name,
        report["n_rows"],
        report["n_features"],
        report["n_groups"],
    )
    return report


def assert_no_group_leakage(train: pd.DataFrame, test: pd.DataFrame) -> int:
    """Verifica que ningún grupo (`filename`) aparezca en train y test a la vez."""
    g = config.GROUP_COL
    if g not in train.columns or g not in test.columns:
        raise DataValidationError(f"No se puede verificar leakage: falta '{g}'")
    overlap = set(train[g]) & set(test[g])
    if overlap:
        raise DataValidationError(
            f"¡Fuga de datos! {len(overlap)} grupos en train y test: {list(overlap)[:5]}..."
        )
    log.info(
        "Sin fuga de grupos: train=%d / test=%d grupos disjuntos",
        train[g].nunique(),
        test[g].nunique(),
    )
    return len(overlap)


def validate_prepared_detector_data(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Suite completa para los datos preparados del detector (train/test scaled)."""
    rep_tr = validate_features_frame(train, "train")
    rep_te = validate_features_frame(test, "test")
    assert_no_group_leakage(train, test)
    return {"train": rep_tr, "test": rep_te}
