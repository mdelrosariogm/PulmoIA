# Fase 3 — Pipeline de Entrenamiento (Prefect)

> Orquestación reproducible del flujo de datos y entrenamiento del detector con Prefect.
> Modo **local** (API efímera); UI opcional con `prefect server start`.

---

## Aislamiento de Prefect

El paquete fija `PREFECT_HOME=<proyecto>/.prefect` (en `pulmoia/__init__.py`) para usar una base
de datos local propia y **no chocar** con instalaciones globales de Prefect. Respeta un
`PREFECT_HOME` definido por el usuario.

---

## 3.1 Flows

```
pulmoia-pipeline (main_flow.full_pipeline)
├── preprocesamiento-detector (preprocessing_flow)
│     ├── [opcional] regenerar-datos-preparados   (subproceso → pulmoia.data.preparation)
│     ├── cargar-datos-preparados
│     └── validar-datos                            (esquema + sin fuga de grupos)
└── entrenamiento-detector (training_flow)
      ├── validar-datos
      ├── entrenar-detector                        (XGBoost + LogReg, MLflow)
      └── registrar-modelo                         (champion/challenger)
```

| Flow | Módulo | Responsabilidad |
|---|---|---|
| `preprocessing_flow` | `pulmoia.pipelines.preprocessing` | ETL + validación de datos (3.2) |
| `training_flow` | `pulmoia.pipelines.training` | Entrenamiento + registro (3.1/3.3) |
| `full_pipeline` | `pulmoia.pipelines.main_flow` | Encadena todo (3.1) |

**Ejecución manual:**
```bash
uv run python -m pulmoia.pipelines.main_flow            # pipeline completo
uv run python -m pulmoia.pipelines.main_flow --smoke    # validación rápida
uv run python -m pulmoia.pipelines.preprocessing        # solo datos + validación
uv run python -m pulmoia.pipelines.training             # solo entrenamiento
```

---

## 3.2 Data pipeline y validación

- **ETL:** parte de `outputs/features_*.csv`. Por defecto reutiliza los datos preparados
  (`outputs/cleaning/HF_ICBHI/{train,test}_scaled.csv`); con `--regenerate` reconstruye la
  transformación (limpieza por correlación → split por archivo → estandarización) de forma
  reproducible vía `pulmoia.data.preparation`.
- **Validación** (`pulmoia.data.validation`, checks ligeros propios):
  - esquema esperado (features numéricos, targets, columna de grupo `filename`),
  - columnas no totalmente nulas; aviso de nulos parciales,
  - prevalencia por etiqueta (aviso si una clase es ultra-rara o sin variación),
  - **sin fuga de grupos** entre train y test (assert de `filename` disjuntos).
- **Logging/monitoreo:** trazas por task vía `get_run_logger`; visibles en la UI de Prefect.

---

## 3.3 Model pipeline y retraining

- **Automatización:** `training_flow` encadena validación → entrenamiento → registro.
- **Feature engineering / evaluación:** reutiliza el pipeline OvR de la Fase 2 (umbral por
  etiqueta, métricas en MLflow). **Random Forest se excluye** del pipeline (su búsqueda de ~2 h
  no es apta para retraining; queda como experimento puntual).
- **Retraining programado (demo, 3.3 nice-to-have):** `pulmoia.pipelines.deployment` crea un
  deployment con cron **semanal (lunes 03:00)** y lo sirve:
  ```bash
  uv run python -m pulmoia.pipelines.deployment          # sirve el schedule (bloquea)
  uv run prefect server start                            # UI en otra terminal
  uv run prefect deployment run 'pulmoia-pipeline/retraining-semanal'   # disparo manual
  ```

---

## Decisiones de configuración (Fase 3)

| Decisión | Elección |
|---|---|
| Modo Prefect | Local + UI opcional |
| Alcance ETL | Desde CSV de features (audio crudo fuera del flow) |
| Validación de datos | Checks ligeros propios (sin dependencias extra) |
| Scheduling | Deployment con cron semanal (demo) + disparo manual |
| Algoritmos en el pipeline | XGBoost + LogReg (RF excluido) |
