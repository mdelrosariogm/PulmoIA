"""
Adaptador del modelo real de pulmoia para la API web.

Envuelve la cadena `pulmoia.serving.inference` (wav → detector → perfil acústico →
COPD0–4 + confianza) y la mapea al esquema AnalysisResult que consume el frontend.
"""

from schemas.analysis import AnalysisResult, COPDLevel

from pulmoia.serving.inference import load_bundle, predict_from_audio_bytes

# Metadatos clínicos por nivel COPD (texto mostrado en la UI).
COPD_META = {
    0: {
        "name": "Bajo Riesgo",
        "fev1_fvc": "Normal (≥ 70%)",
        "fev1_pct": "Normal (≥ 80%)",
        "pattern": "PFT normal",
        "recommendation": "Seguimiento normal — controles anuales recomendados",
        "emergency": False,
    },
    1: {
        "name": "Nivel Leve",
        "fev1_fvc": "< 70%",
        "fev1_pct": "≥ 80%",
        "pattern": "Obstrucción leve",
        "recommendation": "Observación recomendada — consulta médica en 3 meses",
        "emergency": False,
    },
    2: {
        "name": "Nivel Moderado",
        "fev1_fvc": "< 70%",
        "fev1_pct": "50–80%",
        "pattern": "Obstrucción moderada",
        "recommendation": "Acudir al médico esta semana",
        "emergency": False,
    },
    3: {
        "name": "Nivel Grave",
        "fev1_fvc": "< 70%",
        "fev1_pct": "30–50%",
        "pattern": "Obstrucción grave",
        "recommendation": "Consulta médica urgente — 24 a 48 horas",
        "emergency": False,
    },
    4: {
        "name": "Nivel Muy Grave",
        "fev1_fvc": "< 70%",
        "fev1_pct": "< 30% (o < 50% con I.R.)",
        "pattern": "Insuficiencia respiratoria crónica",
        "recommendation": "Atención de emergencia inmediata — llame al 123",
        "emergency": True,
    },
}


class MLPredictor:
    """Carga el bundle servible una vez y predice desde bytes de audio."""

    def __init__(self):
        self.bundle = load_bundle()
        self.model_version = self.bundle.get("version", {}).get("copd", "1.0.0")

    def predict_from_bytes(self, audio_bytes: bytes) -> AnalysisResult:
        r = predict_from_audio_bytes(audio_bytes)
        level = int(r["copd_level"])
        meta = COPD_META[level]
        return AnalysisResult(
            copd_level=COPDLevel(level),
            level_name=meta["name"],
            confidence=float(r["confidence"]),
            fev1_fvc=meta["fev1_fvc"],
            fev1_pct=meta["fev1_pct"],
            pattern=meta["pattern"],
            recommendation=meta["recommendation"],
            emergency=meta["emergency"],
            features_used=len(self.bundle["detector"].feature_names),
            model_version=str(self.model_version),
        )
