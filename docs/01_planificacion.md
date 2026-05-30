# Fase 1.1 — Planificación del Proyecto

> **Proyecto:** pulmoia — Detección de sonidos adventicios pulmonares y clasificación de severidad EPOC
> **Tipo:** Proyecto MLOps end-to-end (Especialización en IA)
> **Fecha de planificación:** 2026-05-30
> **Estado:** Propuesta para aprobación

---

## 1. Selección del Proyecto

### Problema de ML
Clasificación **multilabel** de **sonidos adventicios** (eventos acústicos anormales) en grabaciones de
auscultación pulmonar. Dada una ventana de audio de 1 segundo, el modelo predice la presencia simultánea
de cuatro patologías acústicas:

| Etiqueta | Significado clínico |
|---|---|
| `has_wheeze` | Sibilancia (obstrucción bronquial) |
| `has_crackle` | Crepitancia (apertura alveolar anormal) |
| `has_stridor` | Estridor (obstrucción de vía aérea alta) |
| `has_rhonchus` | Roncus (secreciones en vía aérea baja) |

El detector multilabel es el **núcleo de ML** del producto, pero el **modelo servido (MVP)** es la
**cadena completa de 3 pasos** que entrega la **severidad EPOC (COPD0–COPD4)** a partir de un audio TR:

```
Audio TR (.wav) ─► [Paso 1] Detector multilabel por ventana
                ─► [Paso 2] Agregación → perfil acústico del audio (%wheeze, %crackle, ...)
                ─► [Paso 3] Asignación estadística → COPD0 | COPD1 | COPD2 | COPD3 | COPD4
```

### ¿Por qué esta arquitectura como MVP?
- El **detector (Paso 1)** es el único componente de ML supervisado real y tiene **volumen suficiente**
  (~168k ventanas HF+ICBHI) para entrenar, versionar y monitorear con rigor.
- El **producto servido (Pasos 1→3)** entrega la salida clínicamente relevante: **severidad EPOC**.
- Permite demostrar **todas** las competencias del curso (tracking, orquestación, deployment, monitoreo)
  sobre un pipeline end-to-end real.

---

## 2. Problema de Negocio (hipotético)

**Contexto:** En atención primaria y telemedicina, la auscultación pulmonar depende de la pericia del clínico.
Los sonidos adventicios sutiles (estridor temprano, sibilancias leves) se pasan por alto, retrasando la
derivación a espirometría y el diagnóstico de EPOC/asma.

**Solución hipotética:** Un servicio de **cribado asistido** que, a partir de grabaciones de estetoscopio
electrónico (p. ej. Littmann 3200), devuelve un **perfil acústico** con la probabilidad de cada evento adventicio
y una **estimación de severidad EPOC (COPD0–4)** para orientar la prioridad de derivación.

**Valor:**
- **Triaje automatizado**: prioriza pacientes con perfil acústico anormal para evaluación especializada.
- **Telemedicina rural**: apoyo diagnóstico donde no hay neumólogo.
- **Documentación objetiva**: cuantifica hallazgos auscultatorios (hoy subjetivos) en la historia clínica.

> ⚠️ **Disclaimer:** Herramienta de apoyo/cribado, **no** diagnóstica. No reemplaza criterio médico ni espirometría.

---

## 3. Métricas de Éxito

El producto tiene dos componentes evaluables: el **detector (Paso 1)** y la **salida COPD (Pasos 1→3)**.

### 3.1 Detector multilabel (Paso 1)

**Métrica primaria: Macro-averaged ROC-AUC** sobre las 4 etiquetas — independiente del umbral, robusta al
fuerte desbalance (stridor ≈ 0.4%, crackle ≈ 17.6%) y comparable con la literatura (Hsu et al. 2021).

**Métricas secundarias:**
| Métrica | Por qué |
|---|---|
| **Macro F1** (umbral por label optimizado) | Balance precisión/recall operativo en cribado |
| **PR-AUC por etiqueta** | Crítico para clases raras (stridor ≈ 0.4%, rhonchus ≈ 7%) donde ROC-AUC es optimista |
| **Recall por etiqueta** | Falsos negativos > falsos positivos en cribado |
| **Brier score / calibración** | La salida es probabilidad → debe estar bien calibrada para el Paso 2 |

**Objetivos cuantitativos:**
| Hito | Macro ROC-AUC (primaria) | Macro F1 (ref.) | Nota |
|---|---|---|---|
| **Baseline** (Dummy / LogReg) | referencia | referencia | Comparación obligatoria |
| **MVP aceptable** | ≥ 0.80 | — | Mínimo útil |
| **Objetivo** | ≥ 0.88 | — | Meta del proyecto |
| **Stretch** | ≥ 0.92 | — | Coherente con benchmarks HF_Lung_V1 |

> **Nota sobre clases raras:** ROC-AUC tolera bien el desbalance, por lo que los objetivos son alcanzables
> incluso con `stridor`. El **PR-AUC por etiqueta** (secundaria) vigila específicamente el desempeño en las
> clases raras, donde ROC-AUC puede ser optimista.

### 3.2 Salida COPD0–4 (Pasos 1→3, producto servido)

| Métrica | Por qué |
|---|---|
| **Balanced accuracy** | Robusta al desbalance entre niveles COPD (COPD4=17 vs COPD1=5 pacientes) |
| **Macro F1 (5 clases)** | Desempeño por nivel de severidad |
| **MAE ordinal** | COPD0<1<2<3<4 es ordinal: confundir COPD3↔COPD4 < confundir COPD0↔COPD4 |
| **Matriz de confusión** | Interpretabilidad clínica de los errores |

> **Validación COPD:** Leave-One-Patient-Out o GroupKFold **por `patient_id`** (solo ~40 pacientes).
> El objetivo cuantitativo COPD se fija tras el baseline (muestra pequeña → se reporta con intervalos de confianza).

### 3.3 Manejo de clases raras
1. Medir baseline y revisar prevalencia efectiva tras split por audio.
2. Vigilar `stridor` y `rhonchus` con **PR-AUC y recall por etiqueta** (no solo el macro-promedio de ROC-AUC).
3. Técnicas a evaluar en Fase 2: `class_weight`/`scale_pos_weight`, umbral por label, focal loss (si DL).

> **Validación sin leakage:** split y CV **por archivo de audio** (GroupKFold por `filename`) en Paso 1,
> y **por `patient_id`** en COPD. Nunca por ventana.

---

## 4. Alcance: MVP vs. Funcionalidad Completa

### 🎯 MVP (entregable mínimo aprobado)
1. Detector multilabel entrenado (HF+ICBHI) con baseline establecido.
2. **Pasos 2–3 implementados**: agregación a perfil acústico + asignación estadística COPD0–4 (ANOVA + umbrales).
3. Experiment tracking en **MLflow** (params, métricas, artefactos, model registry).
4. Pipeline de entrenamiento reproducible orquestado en **Prefect**.
5. **API FastAPI** que recibe audio TR y devuelve **severidad COPD0–4** + perfil acústico, con validación de inputs.
6. **Dockerfile** funcional.
7. **Unit tests** + linter (black/flake8) + README + docs de deployment.
8. Diseño de **monitoreo** (propuesta).

### 🚀 Funcionalidad completa (nice-to-have)
- Model registry con staging/production y retraining automático.
- CI/CD (GitHub Actions) + despliegue en cloud.
- Optimización de imagen Docker (multi-stage) + pre-commit hooks.
- Modelo de deep learning (CNN sobre espectrograma) como experimento comparativo.
- Endpoint adicional que exponga solo el detector (perfil acústico) para casos sin diagnóstico COPD.

---

## 5. Timeline y Responsables

> Proyecto individual: un mismo responsable asume distintos **roles**. La columna "Rol" indica el sombrero
> técnico que gobierna cada fase.

| Fase | Entregable | Rol responsable | Estimación | Estado |
|---|---|---|---|---|
| **1.1** Planificación | Este documento | Arquitecto de datos | 0.5 día | 🟡 En revisión |
| **1.2** Setup entorno | Migración a `uv`, `pyproject.toml`, estructura `src/` | Ingeniero de datos | 0.5 día | ⬜ Pendiente |
| **1.3** Análisis + baseline | EDA (✓) + baseline de rendimiento | Analista / ML Engineer | 1 día | 🟢 EDA hecho, falta baseline |
| **2.1–2.2** MLflow + experimentos | Tracking, RF/XGBoost, CV, hiperparámetros | ML Engineer | 2 días | ⬜ Pendiente |
| **2.3** Model Registry | Registro, tags, staging/production | ML Engineer | 0.5 día | ⬜ Pendiente |
| **3.1–3.3** Prefect pipelines | Flows ETL + entrenamiento + scheduling | Ingeniero de datos | 2 días | ⬜ Pendiente |
| **3.x** Pipeline COPD (Pasos 2–3) | Agregación perfil acústico + asignación estadística COPD0–4 | ML / Analista | 1.5 días | ⬜ Pendiente |
| **4.1–4.2** Deployment | Dockerfile + API FastAPI (audio TR → COPD0–4) | ML / Backend Engineer | 1.5 días | ⬜ Pendiente |
| **5** Monitoreo | Diseño de monitoreo (drift, performance) | Arquitecto de datos | 0.5 día | ⬜ Pendiente |
| **6** Testing + calidad + docs | Tests, linter, pre-commit, README, guías | ML Engineer | 1.5 días | ⬜ Pendiente |
| *(opcional)* DL comparativo | CNN sobre espectrograma | Deep Learning Engineer | 2 días | ⬜ Nice-to-have |

**Duración estimada núcleo MVP:** ~10 días efectivos.

---

## 6. Decisiones técnicas registradas (Fase 1)

| Decisión | Elección | Justificación |
|---|---|---|
| Gestor de entorno | **uv** + `pyproject.toml` | Estándar moderno, lock reproducible, cumple rúbrica |
| Estructura de código | Refactor completo a `src/` | Modularidad, testing, packaging limpio |
| Modelo servido | Cadena completa Pasos 1→3 → COPD0–4 | Entrega la salida clínica relevante (severidad EPOC) |
| Métrica primaria detector | Macro ROC-AUC | Robusta al desbalance, independiente del umbral, comparable con literatura |
| Formato de trabajo | Scripts `.py` + notebooks ligeros | Reproducibilidad + narrativa de EDA/experimentos |
| Tracking | MLflow | Requisito del curso |
| Orquestación | Prefect | Requisito del curso |
| API | FastAPI | Requisito del curso |
| Validación | GroupKFold por `filename` (Paso 1) / `patient_id` (COPD) | Evita data leakage |

---

## 7. Criterios de aprobación de la Fase 1

- [x] Problema de ML definido y acotado
- [x] Problema de negocio hipotético articulado
- [x] Métricas de éxito con objetivos cuantitativos
- [x] Alcance MVP vs. completo delimitado
- [x] Timeline con roles/responsables
- [ ] **Baseline de rendimiento establecido** (Fase 1.3 — pendiente de ejecutar)
- [ ] Entorno `uv` + estructura `src/` montados (Fase 1.2 — pendiente)
