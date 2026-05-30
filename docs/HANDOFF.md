# 🤝 HANDOFF — Estado del proyecto pulmoia y cómo continuar

> Documento de traspaso. Lee esto **completo** antes de continuar el trabajo.
> Última actualización: 2026-05-30.

---

## ⚠️ LO PRIMERO: los datos NO están en GitHub

Por `.gitignore`, **no se versionan**: audio crudo (`data/`), CSVs (`outputs/`),
modelos/scalers (`models/`), la base de MLflow (`mlflow.db`) ni `mlartifacts/`.

**Necesitas que te compartan por otro canal (Drive, etc.):**

| Carpeta/archivo | Para qué | ¿Imprescindible? |
|---|---|---|
| `data/RDB`, `data/LUNGS`, `data/ICBHI` | audio crudo | Solo si re-extraes features |
| `outputs/features_*.csv` | features extraídos | Sí, para preparar datos |
| `outputs/cleaning/HF_ICBHI/*_scaled.csv` | train/test del detector | **Sí, para entrenar** |
| `outputs/feature_selection/relieff/` | selección ReliefF (79 feats) | Sí, para reproducir |
| `models/scaler_*.pkl` | scalers | Sí, para servir |
| `mlflow.db` + `mlartifacts/` | experimentos + Model Registry | **Sí, para el registry** |

Sin esto, el código está pero no hay datos ni modelos entrenados. Pídelos antes de empezar.

---

## ✅ Lo que YA está hecho (Fases 1–3 + selección de features)

### Fase 1 — Planificación y Setup
- `docs/01_planificacion.md`: problema ML, problema de negocio, métricas, alcance MVP,
  timeline, decisiones.
- Entorno con **uv** (`pyproject.toml`, `uv.lock`, Python 3.11). Estructura `src/pulmoia/`.
- **Baseline** (`pulmoia.models.baseline`): detector LogReg Macro ROC-AUC 0.809; COPD directo
  apenas supera azar (0.247 vs 0.20) → justifica la arquitectura de 3 pasos.

### Fase 2 — Experiment Tracking (MLflow)
- MLflow con backend **SQLite** (`mlflow.db`), `docs/02_experimentos.md`.
- Detector multilabel OvR (`pulmoia.models.detector.MultiLabelDetector`).
- `pulmoia.models.train_detector`: RF/XGBoost/LogReg, RandomizedSearchCV + GroupKFold por
  `filename`, umbral por etiqueta (F1 out-of-fold). **Resultados test Macro ROC-AUC:**
  RF 0.829 · XGBoost 0.824 · LogReg 0.809.
- **Model Registry** `pulmoia_detector`.

### Selección de features — ReliefF
- `pulmoia.features.relieff_selection` (skrebate): 127 → **79 features**. Reentreno: XGBoost
  0.815 / LogReg 0.808 (casi igual con 38% menos features). `docs/04_seleccion_features.md`.

### Fase 3 — Pipeline (Prefect)
- `pulmoia.pipelines`: `preprocessing`, `training`, `main_flow`, `deployment`.
- Validación de datos ligera (`pulmoia.data.validation` + tests). `docs/03_pipelines.md`.
- **Prefect aislado al proyecto** (`PREFECT_HOME=<root>/.prefect`, en `pulmoia/__init__.py`).

### Estado del Model Registry (ahora mismo)
| Versión | Modelo | Features | Stage | Macro AUC |
|---|---|---|---|---|
| **v3** | XGBoost / relieff | 79 | **Production** (champion) | 0.815 |
| v4 | XGBoost / all | 127 | Staging (challenger) | 0.824 |
| v1, v2 | — | — | Archived | — |

> El modelo en producción es el **detector de 79 features ReliefF** (elegido por eficiencia).

---

## 🔜 Lo que FALTA (en orden sugerido)

### A. Pasos 2–3 del producto COPD (¡bloquea la API!)
El modelo servido (según la planificación) es la **cadena completa → COPD0–4**, pero:
- **Paso 2** (NO implementado): aplicar el detector a cada audio de TR → perfil acústico
  (% ventanas con wheeze/crackle/stridor/rhonchus por audio).
- **Paso 3** (NO implementado): algoritmo estadístico (ANOVA + umbrales) que asigna COPD0–4
  a partir del perfil acústico. Datos: `outputs/features_TR.csv` (¡es **1 fila por archivo-canal**,
  504 filas, 42 pacientes — NO por ventana, ojo con el README §Tamaños que dice por-ventana).
- Validar COPD con **GroupKFold por `patient_id`** (sin leakage; solo ~40 pacientes).

### B. Fase 4 — Deployment
- **FastAPI** (`src/pulmoia/api/`): endpoint que reciba audio TR (o features) y devuelva COPD0–4
  + perfil acústico. Cargar el modelo del Registry (`models:/pulmoia_detector@champion`).
- **Dockerfile**.

### C. Fase 5 — Monitoreo (solo diseño)
- Propuesta de monitoreo: drift de datos/feature, caída de performance, alertas, retraining.
  Esqueleto en `src/pulmoia/monitoring/`.

### D. Fase 6 — Testing y calidad
- Más unit tests. **Limpiar ruff en scripts heredados** (`features/extract_*`, `data/eda.py`,
  `data/preparation.py` tienen ~20 issues). Configurar pre-commit (nice-to-have).
- Corregir el README §"Tamaños de Dataset" (TR es por archivo, no por ventana).
- CI/CD GitHub Actions (nice-to-have), cloud (nice-to-have).

---

## 🧭 Cómo levantar el entorno

```bash
git clone https://github.com/mdelrosariogm/PulmoIA.git && cd pulmoia
uv sync --extra mlops --extra api          # crea .venv con Python 3.11 + todo
# (coloca los datos/artefactos compartidos en data/, outputs/, models/, y mlflow.db en la raíz)
uv run pytest                              # debe pasar
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # ver experimentos/registry
```

Comandos clave (todos con `uv run`):
```bash
uv run python -m pulmoia.models.baseline                      # baseline
uv run python -m pulmoia.models.train_detector               # entrenar (RF+XGB+LogReg)
uv run python -m pulmoia.features.relieff_selection --from-saved   # reentrenar con ReliefF
uv run python -m pulmoia.models.register_model               # registrar/promover
uv run python -m pulmoia.pipelines.main_flow                 # pipeline Prefect completo
```

---

## 🚩 Gotchas (errores a evitar)

1. **Datos fuera de git** (ver arriba). Sin ellos no hay nada que entrenar/servir.
2. **Sin leakage**: split y CV SIEMPRE por grupo (`filename` en detector, `patient_id` en COPD).
   Nunca por ventana.
3. **Runs MLflow**: los de prueba se etiquetan `run_type=smoke` y `register_best` los ignora.
   Usa `run_type=full`. No promuevas smokes.
4. **RandomForest tarda ~2 h** → está EXCLUIDO del pipeline de retraining (solo XGBoost+LogReg).
5. **Prefect** está aislado en `.prefect/` del proyecto (no usa la instalación global).
6. **Python 3.11** fijo (no 3.12+, incompatibilidades).
7. **`features_TR.csv` es por archivo-canal**, no por ventana.

---

## 📂 Mapa de módulos

```
src/pulmoia/
├── config.py            # rutas + setup_mlflow + feature_columns
├── data/
│   ├── preparation.py   # limpieza correlación + split + scaler + SHAP (heredado)
│   ├── eda.py           # EDA (heredado)
│   └── validation.py    # validación ligera de datos
├── features/
│   ├── extract_*.py     # extracción de audio→features (heredado, ya ejecutado)
│   └── relieff_selection.py  # selección ReliefF
├── models/
│   ├── detector.py      # MultiLabelDetector (OvR)
│   ├── baseline.py      # baselines de referencia
│   ├── train_detector.py # entrenamiento + MLflow
│   └── register_model.py # Model Registry
├── pipelines/           # Prefect (preprocessing, training, main_flow, deployment)
├── api/                 # (VACÍO — Fase 4)
└── monitoring/          # (VACÍO — Fase 5)
```
