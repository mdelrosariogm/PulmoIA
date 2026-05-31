"""
detector.py
===========
Detector multilabel de sonidos adventicios (estrategia One-vs-Rest).

`MultiLabelDetector` encapsula un clasificador binario por etiqueta + su umbral de
decisión + la lista de features esperadas. Es picklable e importable, por lo que puede
loguearse en MLflow (Model Registry) y servirse en la API sin dependencias del script
de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MultiLabelDetector:
    """Detector OvR: un estimador binario por etiqueta.

    Parameters
    ----------
    estimators : dict[label -> estimador sklearn ya entrenado con predict_proba]
    thresholds : dict[label -> umbral de decisión en [0,1]]
    feature_names : orden exacto de columnas de entrada esperado
    labels : orden de las etiquetas de salida
    """

    estimators: dict
    feature_names: list[str]
    labels: list[str]
    thresholds: dict = field(default_factory=dict)

    def __post_init__(self):
        for lab in self.labels:
            self.thresholds.setdefault(lab, 0.5)

    def _as_matrix(self, X) -> np.ndarray:
        """Acepta DataFrame (reordena por feature_names) o array (asume orden correcto)."""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].to_numpy()
        return np.asarray(X)

    def predict_proba(self, X) -> pd.DataFrame:
        """Probabilidad de la clase positiva por etiqueta. Devuelve DataFrame (n, n_labels)."""
        M = self._as_matrix(X)
        out = {lab: self.estimators[lab].predict_proba(M)[:, 1] for lab in self.labels}
        return pd.DataFrame(out, columns=self.labels)

    def predict(self, X) -> pd.DataFrame:
        """Predicción binaria por etiqueta aplicando el umbral de cada una."""
        proba = self.predict_proba(X)
        return pd.DataFrame(
            {
                lab: (proba[lab].to_numpy() >= self.thresholds[lab]).astype(int)
                for lab in self.labels
            },
            columns=self.labels,
        )
