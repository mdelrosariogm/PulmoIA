"""Tests de la función de drift (PSI) — puros, sin datos en disco (Fase 6)."""

import numpy as np

from pulmoia.monitoring.drift import classify_psi, psi


def test_psi_cero_para_misma_distribucion():
    rng = np.random.RandomState(0)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    val = psi(x, y)
    assert val < 0.1  # misma distribución → sin drift


def test_psi_alto_para_distribuciones_distintas():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, size=5000)
    y = rng.normal(5, 1, size=5000)  # media desplazada → drift fuerte
    assert psi(x, y) > 0.25


def test_classify_psi_umbrales():
    assert classify_psi(0.05) == "sin_drift"
    assert classify_psi(0.15) == "moderado"
    assert classify_psi(0.40) == "significativo"
