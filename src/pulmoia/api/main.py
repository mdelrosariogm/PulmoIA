"""
main.py
=======
Fase 4 — API REST (FastAPI) para servir la cadena pulmoia: wav → COPD0–4 + perfil acústico.

Arranque:
  uv run uvicorn pulmoia.api.main:app --reload
Docs interactivas: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from pulmoia.serving.inference import load_bundle, predict_from_wavs

app = FastAPI(
    title="pulmoia API",
    version="0.1.0",
    description="Clasificación de severidad EPOC (COPD0–4) desde auscultación pulmonar.",
)

MAX_FILES = 24
MAX_BYTES = 50 * 1024 * 1024  # 50 MB por archivo


class EventoPerfil(BaseModel):
    pct_ventanas: float
    prob_media: float


class Prediccion(BaseModel):
    copd: str
    n_ventanas: int
    n_audios: int
    perfil_acustico: dict[str, EventoPerfil]
    version: dict


@app.get("/health")
def health():
    """Liveness/readiness: confirma que el bundle está cargado."""
    bundle = load_bundle()
    return {"status": "ok", "version": bundle.get("version", {})}


@app.on_event("startup")
def _warmup():
    # Carga el bundle al iniciar (falla rápido si no existe).
    load_bundle()


@app.post("/predict", response_model=Prediccion)
async def predict(files: list[UploadFile] = File(..., description="Uno o varios .wav de TR")):
    """Recibe audios .wav y devuelve la severidad COPD0–4 + perfil acústico."""
    if not files:
        raise HTTPException(status_code=400, detail="Sube al menos un archivo .wav.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Máximo {MAX_FILES} archivos por petición.")

    tmpdir = tempfile.mkdtemp(prefix="pulmoia_")
    paths = []
    try:
        for f in files:
            if not (f.filename or "").lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail=f"Solo .wav: '{f.filename}'.")
            dest = os.path.join(tmpdir, os.path.basename(f.filename))
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            if os.path.getsize(dest) == 0:
                raise HTTPException(status_code=400, detail=f"Archivo vacío: '{f.filename}'.")
            if os.path.getsize(dest) > MAX_BYTES:
                raise HTTPException(
                    status_code=413, detail=f"Archivo demasiado grande: '{f.filename}'."
                )
            paths.append(dest)

        try:
            return predict_from_wavs(paths)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
