# Web App — Integración del mockup con el modelo real

> Aplicación web funcional (local) que usa el modelo pulmoia. La UI del mockup se conserva
> **exacta**; solo se cableó el modelo real por detrás.

---

## Estructura

```
webapp/
├── frontend/            ← mockup INTACTO (HTML/CSS/JS, imágenes, video)
│   ├── index.html       ← 9 pantallas (home, upload, processing, results, ...)
│   ├── css/styles.css
│   ├── js/app.js        ← ÚNICO cambio: llamada real a la API (sin tocar visuales)
│   └── assets/          ← imágenes (video .mp4 no versionado por tamaño)
└── backend/             ← FastAPI cableado a pulmoia
    ├── main.py          ← sirve el frontend en / + API en /api/v1
    ├── routes/analysis.py
    ├── services/ml_predictor.py  ← adaptador a pulmoia.serving.inference
    └── schemas/analysis.py       ← esquema AnalysisResult (intacto)
```

## Qué se cambió (mínimo)

- **Frontend `app.js`**: la simulación (`startProcessing` siempre mostraba COPD2) ahora hace
  `fetch('/api/v1/analyze')` con el audio subido y muestra el **COPD real + confianza real**.
  Cero cambios de estilos, pantallas o textos.
- **Backend**: `ml_predictor` y la ruta `/analyze` ahora usan la cadena real
  `pulmoia.serving.inference.predict_from_audio_bytes` (detector → perfil → COPD0–4 LogReg).
  Se eliminó `audio_processor.py` (stub muerto).
- El modelo COPD del bundle se cambió a **LogReg** para exponer una **confianza** real (%).

## Cómo ejecutarla

Requisitos: entorno del repo (`uv sync --extra api`) y el bundle generado.

```bash
# 1. Generar el bundle servible (si no existe)
uv run python -m pulmoia.serving.bundle

# 2. Levantar la web (sirve frontend + API)
uv run uvicorn main:app --app-dir webapp/backend --port 8080

# 3. Abrir en el navegador
#    http://localhost:8080
```

Flujo: **Comenzar análisis → elegir rol → subir un .wav de auscultación → Analizar →**
la barra de procesamiento corre y muestra el **nivel COPD0–4 real** con su confianza y
recomendación clínica.

## Contrato de la API

`POST /api/v1/analyze` (multipart): `audio` (.wav/.mp3/.flac/.ogg), `role`, `patient_name?`,
`patient_age?`, `symptoms?` → `AnalysisResult`:

```json
{
  "copd_level": 0, "level_name": "Bajo Riesgo", "confidence": 0.65,
  "fev1_fvc": "Normal (≥ 70%)", "fev1_pct": "Normal (≥ 80%)",
  "pattern": "PFT normal", "recommendation": "Seguimiento normal — controles anuales",
  "emergency": false, "features_used": 79, "model_version": "logreg_all"
}
```

> Recordatorio: el clasificador COPD es **orientativo, no diagnóstico** (~40 pacientes de
> entrenamiento). Para mejor señal, subir varios canales del mismo paciente.

## Notas
- El video intro (`assets/video/*.mp4`, 53 MB) **no se versiona** (.gitignore). La UI funciona
  sin él (el overlay simplemente no reproduce). Cópialo manualmente si lo quieres.
- `.claude/launch.json` define el server `webapp` para previsualización.
