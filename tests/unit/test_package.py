"""Smoke tests del paquete pulmoia y su estructura.

Valida que el paquete y sus subpaquetes son importables tras el setup de la Fase 1.2.
"""

import importlib

import pulmoia


def test_version():
    assert pulmoia.__version__ == "0.1.0"


def test_subpackages_importables():
    for sub in ["data", "features", "models", "api", "monitoring"]:
        mod = importlib.import_module(f"pulmoia.{sub}")
        assert mod is not None
