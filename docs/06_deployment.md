# Fase 4 — Deployment (API FastAPI + Docker)

> Sirve la cadena completa **wav → COPD0–4 + perfil acústico**. Modelo empaquetado
> (`models/serving_bundle.pkl`), sin dependencia de MLflow en runtime.

---

## Arquitectura de serving

```
wav (.wav) ──> load+highpass+ventaneo ──> features por ventana
           ──> seleccionar 127 + scaler(HF_ICBHI) ──> detector (79 ReliefF)
           ──> perfil acústico (prob. media por evento)
           ──> modelo COPD (NearestCentroid) ──> COPD0–4
```

| Componente | Módulo |
|---|---|
| Empaquetado del modelo | `pulmoia.serving.bundle` → `models/serving_bundle.pkl` |
| Inferencia (cadena) | `pulmoia.serving.inference` |
| API REST | `pulmoia.api.main` |

El bundle contiene: detector champion, scaler + 127 features, modelo COPD (entrenado sobre los
42 pacientes) y metadatos. Se deserializa **sin MLflow** (verificado).

---

## Generar el bundle y correr la API (local)

```bash
# 1. Empaquetar el modelo (requiere MLflow/registry — solo en build time)
uv run python -m pulmoia.serving.bundle

# 2. Levantar la API
uv run uvicorn pulmoia.api.main:app --host 127.0.0.1 --port 8000
# Docs interactivas: http://localhost:8000/docs
```

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Liveness; confirma bundle cargado + versión |
| POST | `/predict` | Sube uno o varios `.wav` → COPD0–4 + perfil acústico |

**Ejemplo:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "files=@data/RDB/H016_L1.wav" -F "files=@data/RDB/H016_L2.wav"
```
**Respuesta:**
```json
{
  "copd": "COPD0",
  "n_ventanas": 82,
  "n_audios": 2,
  "perfil_acustico": {
    "has_wheeze":   {"pct_ventanas": 0.10, "prob_media": 0.26},
    "has_crackle":  {"pct_ventanas": 0.83, "prob_media": 0.49},
    "has_stridor":  {"pct_ventanas": 0.0,  "prob_media": 0.0},
    "has_rhonchus": {"pct_ventanas": 0.0,  "prob_media": 0.005}
  },
  "version": {"detector": "pulmoia_detector@champion", "copd": "nearest_centroid_all"}
}
```

**Validación de inputs:** solo `.wav`, archivos no vacíos, ≤50 MB c/u, ≤24 por petición.

---

## Docker (Fase 4.1)

```bash
# 1. Generar el bundle (si no existe)
uv run python -m pulmoia.serving.bundle

# 2. Construir la imagen
docker build -t pulmoia-api:latest .

# 3. Ejecutar
docker run --rm -p 8000:8000 pulmoia-api:latest
# Probar: curl http://localhost:8000/health
```

- La imagen usa `python:3.11-slim` + `uv sync --extra api --no-dev` (sin MLflow/Prefect/dev).
- `libsndfile1` y `ffmpeg` para audio. `HEALTHCHECK` contra `/health`.
- `.dockerignore` excluye datos pesados; solo se copia `models/serving_bundle.pkl`.

> **Optimización (nice-to-have):** mover dependencias no usadas en serving (shap, sweetviz,
> skrebate, matplotlib) a un extra aparte para aligerar la imagen; build multi-stage.

---

## Notas de producción
- El clasificador COPD es débil (~40 pacientes); la salida es **orientativa, no diagnóstica**.
- Para una predicción por paciente, enviar **todos los canales** (L1–L6, R1–R6) en una petición.
