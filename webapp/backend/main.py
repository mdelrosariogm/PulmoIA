"""
PulmoIA — Backend web (FastAPI)
Sirve el frontend (mockup intacto) y expone la API que usa el modelo real de pulmoia.

Arranque (desde webapp/backend):
  uv run uvicorn main:app --reload --port 8080
Luego abrir http://localhost:8080
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.analysis import router as analysis_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="PulmoIA API",
    description="Análisis de auscultación pulmonar con IA · clasificación de severidad EPOC (COPD0–4).",
    version="1.0.0",
)

# CORS (permite consumir la API desde otros orígenes en desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API REST (se registra antes del montaje estático para tener prioridad)
app.include_router(analysis_router, prefix="/api/v1")


@app.middleware("http")
async def no_cache(request, call_next):
    """Evita que el navegador cachee el frontend (siempre sirve la versión actual)."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.on_event("startup")
def _warmup():
    """Precalienta numba/librosa con un audio sintético para que el PRIMER análisis
    real del usuario sea rápido (evita el JIT de ~25s en la primera petición)."""
    try:
        import io
        import wave

        import numpy as np

        from pulmoia.serving.inference import predict_from_audio_bytes

        sr = 8000
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        sig = (np.sin(2 * np.pi * 300 * t) * 0.3 * 32767).astype("<i2")
        b = io.BytesIO()
        w = wave.open(b, "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sig.tobytes())
        w.close()
        predict_from_audio_bytes(b.getvalue())
    except Exception:
        pass  # el warmup es best-effort; no debe impedir el arranque


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "PulmoIA API", "version": "1.0.0"}


# Frontend estático en la raíz (html=True sirve index.html y resuelve rutas relativas
# css/, js/, assets/ exactamente como el mockup original).
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
