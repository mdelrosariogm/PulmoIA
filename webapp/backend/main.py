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


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "PulmoIA API", "version": "1.0.0"}


# Frontend estático en la raíz (html=True sirve index.html y resuelve rutas relativas
# css/, js/, assets/ exactamente como el mockup original).
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
