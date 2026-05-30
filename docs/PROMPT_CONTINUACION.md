# 📋 Prompt para continuar el proyecto con Claude (copiar y pegar)

> Pega TODO el bloque de abajo como primer mensaje a tu agente de Claude, dentro de la carpeta
> del proyecto `pulmoia`. Antes, asegúrate de tener los **datos** (ver `docs/HANDOFF.md`).

---

```
Actúa como ingeniero de datos, analista de datos, arquitecto de datos, ingeniero de
machine learning y de deep learning. Vamos a CONTINUAR un proyecto MLOps end-to-end que
ya está en construcción (no empieces de cero). Es un proyecto de grado evaluado por una
rúbrica de 6 fases.

REGLAS DE TRABAJO (obligatorias):
1. Trabajamos por FASES con gate de aprobación: NO avances a la siguiente fase sin mi
   aprobación explícita. Al terminar cada entregable, muéstrame un resumen y pídeme OK.
2. Hazme las preguntas necesarias ANTES de iniciar cada fase.
3. NO alucines: si no sabes el estado de algo, léelo del repo. Antes de afirmar que algo
   existe/funciona, verifícalo con las herramientas. Documenta cada decisión.
4. No sobre-ingenierices. Mantén simplicidad. Commits frecuentes y descriptivos.
5. Verifica con datos reales (corre el código), no asumas resultados.

PRIMER PASO OBLIGATORIO: lee estos archivos del repo antes de proponer nada:
  - docs/HANDOFF.md            (estado, qué falta, cómo levantar, gotchas)
  - docs/01_planificacion.md   (problema, métricas, alcance MVP, decisiones)
  - docs/02_experimentos.md    (MLflow, resultados del detector)
  - docs/03_pipelines.md       (Prefect)
  - docs/04_seleccion_features.md (ReliefF)
  - README.md, pyproject.toml
Luego dame un diagnóstico del estado actual y las preguntas para arrancar la fase pendiente.

CONTEXTO DEL PROYECTO (pulmoia):
Clasificación de severidad EPOC (COPD0–COPD4) desde sonidos de auscultación pulmonar, con
arquitectura de 3 pasos:
  - Paso 1: detector multilabel de sonidos adventicios (wheeze/crackle/stridor/rhonchus)
    por ventana de 1s. Es el ÚNICO modelo ML supervisado. Datos HF_Lung_V1 + ICBHI (~168k
    ventanas). YA ENTRENADO.
  - Paso 2: aplicar el detector a los audios de TR → perfil acústico por audio. NO IMPLEMENTADO.
  - Paso 3: algoritmo estadístico (ANOVA + umbrales) que asigna COPD0–4 desde el perfil. NO
    IMPLEMENTADO.
El modelo SERVIDO (decisión tomada) es la cadena completa Paso 1→3 → COPD0–4.

LO QUE YA ESTÁ HECHO (verifícalo, no lo repitas):
- Fase 1 COMPLETA: planificación, entorno uv (Python 3.11, pyproject.toml), estructura
  src/pulmoia/, baseline (detector LogReg Macro ROC-AUC 0.809; COPD directo ~0.247 apenas
  supera azar → justifica los 3 pasos).
- Fase 2 COMPLETA: MLflow (backend sqlite:///mlflow.db). Detector OvR (un XGBoost por
  etiqueta) con RandomizedSearchCV + GroupKFold por filename + umbral por etiqueta.
  Resultados test Macro ROC-AUC: RandomForest 0.829, XGBoost 0.824, LogReg 0.809.
  Model Registry 'pulmoia_detector'.
- Selección de features con ReliefF (skrebate): 127→79 features; XGBoost 0.815 con 38% menos
  features. El modelo de 79 features ReliefF está PROMOVIDO a Production (champion); el de 127
  queda en Staging (challenger).
- Fase 3 COMPLETA: pipelines de Prefect (preprocessing, training, main_flow, deployment con
  cron semanal demo). Validación de datos ligera + tests. Prefect aislado en .prefect/ local.

LO QUE FALTA (orden sugerido), y la RÚBRICA que debes cubrir:

>>> PENDIENTE A — Pasos 2 y 3 del producto COPD (bloquea la API):
  - Paso 2: cargar el detector champion del Registry (models:/pulmoia_detector@champion),
    aplicarlo a las ventanas de cada audio TR (outputs/features_TR.csv; OJO: es 1 fila POR
    ARCHIVO-CANAL, 504 filas, 42 pacientes, NO por ventana), agregar a perfil acústico por audio.
  - Paso 3: análisis estadístico (ANOVA + umbrales por percentiles) para asignar COPD0–4.
    Validación por GroupKFold por patient_id (solo ~40 pacientes; reporta con intervalos).

>>> FASE 4 — Deployment:
  4.1 Containerización: Dockerfile (optimizar imagen = nice-to-have).
  4.2 API REST con FastAPI (src/pulmoia/api/): endpoints de predicción (audio TR → COPD0–4 +
      perfil acústico), validación de inputs (pydantic). Cargar modelo desde el Registry.
  4.3 Cloud + CI/CD = nice-to-have.

>>> FASE 5 — Monitoreo (solo DISEÑO):
  Propuesta de monitoreo: drift de datos/features, caída de performance, métricas a vigilar,
  alertas y disparo de retraining (conectar con el deployment de Prefect). Esqueleto en
  src/pulmoia/monitoring/.

>>> FASE 6 — Testing y Best Practices:
  6.1 Unit tests (ampliar cobertura).
  6.2 Code Quality: linter (flake8/black/ruff) — HAY ~20 issues de ruff en scripts heredados
      (features/extract_*, data/eda.py, data/preparation.py) por limpiar. Pre-commit = nice-to-have.
  6.3 Documentación: README detallado (corregir §"Tamaños de Dataset": TR es por archivo, no por
      ventana), documentar endpoints de la API, guías de deployment.

MÉTRICAS DE ÉXITO (ya definidas):
- Detector (Paso 1): primaria = Macro ROC-AUC (objetivo ≥0.88, MVP ≥0.80). Secundarias:
  Macro F1, PR-AUC por etiqueta, recall, calibración.
- COPD0–4: balanced accuracy, Macro F1 (5 clases), MAE ordinal, matriz de confusión.

RESTRICCIONES TÉCNICAS (para no romper nada / no alucinar):
- Python 3.11 fijo. Todo se corre con `uv run`. Entorno: `uv sync --extra mlops --extra api`.
- SIN data leakage: split/CV por grupo (filename en detector, patient_id en COPD). NUNCA por ventana.
- Runs MLflow de prueba van con tag run_type=smoke y register_best los ignora; los reales son
  run_type=full. No promuevas modelos smoke.
- RandomForest tarda ~2h → excluido del pipeline de retraining (solo XGBoost+LogReg).
- Los DATOS no están en git (data/, outputs/, models/, mlflow.db son .gitignore). Confírmame que
  los tienes localmente antes de entrenar/servir.
- No hagas force-push. Rama main compartida con otros; commitea y haz pull antes de push.

Empieza leyendo los docs y dame: (1) diagnóstico del estado, (2) preguntas para arrancar el
PENDIENTE A (Pasos 2–3 COPD), que es lo siguiente.
```

---

## Notas para quien entrega (no forman parte del prompt)
- Asegúrate de compartir los datos/artefactos listados en `docs/HANDOFF.md`.
- El agente debe **leer** los docs antes de actuar; el prompt se lo exige explícitamente.
- Si el compañero quiere otro orden (p. ej. Fase 6 calidad antes que Pasos 2–3), puede
  reordenar; el prompt sugiere un orden pero respeta el gate de aprobación.
