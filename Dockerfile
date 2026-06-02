# pulmoia — Web App (UI + modelo real)
# Sirve el frontend + la API (/api/v1/analyze): wav -> COPD0-4 + perfil acústico.
# No requiere MLflow en runtime: el modelo va empaquetado en models/serving_bundle.pkl.

FROM python:3.11-slim

# Dependencias de sistema para audio (librosa / soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Gestor uv (binario oficial)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Instala dependencias (núcleo + extra api, sin grupo dev) desde el lockfile
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra api

# Web app (frontend + backend) y artefacto servible
# El bundle se genera antes con: python -m pulmoia.serving.bundle
COPY webapp ./webapp
COPY models/serving_bundle.pkl ./models/serving_bundle.pkl

EXPOSE 8080
# Healthcheck contra /health (start-period alto: el warmup de numba tarda al iniciar)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

# Sirve la web app (UI en / y API en /api/v1)
CMD ["uvicorn", "main:app", "--app-dir", "webapp/backend", "--host", "0.0.0.0", "--port", "8080"]
