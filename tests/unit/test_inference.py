"""Test de la cadena de inferencia servible (Fase 6). Se omite si falta el bundle o datos."""

from pathlib import Path

import pytest

from pulmoia import config

_SAMPLE_WAV = config.PROJECT_ROOT / "data" / "RDB" / "H016_L1.wav"

pytestmark = pytest.mark.skipif(
    not (config.SERVING_BUNDLE.exists() and _SAMPLE_WAV.exists()),
    reason="requiere serving_bundle.pkl y un wav de muestra (no disponibles en CI)",
)


def test_predict_from_audio_bytes_estructura():
    from pulmoia.serving.inference import predict_from_audio_bytes

    result = predict_from_audio_bytes(Path(_SAMPLE_WAV).read_bytes())

    assert result["copd"] in {"COPD0", "COPD1", "COPD2", "COPD3", "COPD4"}
    assert 0 <= result["copd_level"] <= 4
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["n_ventanas"] > 0
    for lab in config.TARGETS:
        assert 0.0 <= result["perfil_acustico"][lab]["prob_media"] <= 1.0
