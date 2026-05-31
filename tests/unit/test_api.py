"""Tests de la API (Fase 4). Se omiten si falta httpx o el bundle servible."""

import pytest

from pulmoia import config

pytest.importorskip("httpx", reason="TestClient requiere httpx")
pytestmark = pytest.mark.skipif(
    not config.SERVING_BUNDLE.exists(),
    reason="bundle no generado (uv run python -m pulmoia.serving.bundle)",
)


def _client():
    from fastapi.testclient import TestClient

    from pulmoia.api.main import app

    return TestClient(app)


def test_health_ok():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_rechaza_no_wav():
    with _client() as client:
        r = client.post("/predict", files={"files": ("x.txt", b"hola", "text/plain")})
        assert r.status_code == 400


def test_predict_rechaza_vacio():
    with _client() as client:
        r = client.post("/predict", files={"files": ("x.wav", b"", "audio/wav")})
        assert r.status_code == 400
