"""
Ruta principal de análisis de auscultación pulmonar.
Endpoint: POST /api/v1/analyze  →  usa el modelo real de pulmoia.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from schemas.analysis import AnalysisResult, ErrorResponse
from services.ml_predictor import MLPredictor

router = APIRouter(tags=["Análisis Pulmonar"])

# Carga el modelo una vez por proceso (warm start).
ml_predictor = MLPredictor()

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/aac", "audio/flac", "audio/ogg", "audio/m4a",
    "application/octet-stream",  # fallback genérico de algunos navegadores
}
MAX_BYTES = 50 * 1024 * 1024


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               500: {"model": ErrorResponse}},
    summary="Analizar audio de auscultación",
    description="Recibe un audio de auscultación pulmonar y retorna el triage COPD (0–4) "
                "con confianza, perfil acústico y recomendación clínica.",
)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Archivo de audio de auscultación"),
    role: str = Form("medico"),
    patient_name: str | None = Form(None),
    patient_age: int | None = Form(None),
    symptoms: str | None = Form(""),
):
    # Validación de formato (acepta también por extensión .wav si el navegador no envía content-type)
    ext_ok = (audio.filename or "").lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"))
    if audio.content_type not in ALLOWED_AUDIO_TYPES and not ext_ok:
        raise HTTPException(status_code=400,
                            detail=f"Formato no soportado: {audio.content_type or audio.filename}.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Archivo de audio vacío.")
    if len(audio_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 50 MB).")

    analysis_id = str(uuid.uuid4())
    try:
        result = ml_predictor.predict_from_bytes(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error durante el análisis: {exc}") from exc

    background_tasks.add_task(
        _log_analysis, analysis_id=analysis_id, role=role,
        patient_age=patient_age, copd_level=int(result.copd_level), confidence=result.confidence,
    )
    return result


async def _log_analysis(**kwargs):
    """Log de auditoría (placeholder; conectar a BD en producción)."""
    print(f"[LOG] Análisis {kwargs.get('analysis_id')} · COPD{kwargs.get('copd_level')} "
          f"· conf={kwargs.get('confidence', 0):.2%}")
