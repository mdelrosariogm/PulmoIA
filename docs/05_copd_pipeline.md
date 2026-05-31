# Pasos 2–3 — Cadena COPD0–4 (producto servido)

> Cierra la arquitectura de 3 pasos: detector → perfil acústico → severidad EPOC.

---

## Paso 2 — Perfil acústico (`pulmoia.models.copd_profile`)

Aplica el **detector champion** (Model Registry, `models:/pulmoia_detector@champion`) a cada
ventana de 1s de los audios TR y agrega a un perfil por audio y por paciente.

| Aspecto | Detalle |
|---|---|
| Entrada | `outputs/features_TR_windows.csv` (21,158 ventanas, 504 audios, 42 pacientes) |
| Preprocesamiento | seleccionar 127 features (correlación) → `scaler_HF_ICBHI.pkl` → detector (79 ReliefF) |
| Salida | `outputs/copd/profiles_patient.csv`, `profiles_audio.csv` |
| Features de perfil | `pct_<evento>` (% ventanas) y `meanp_<evento>` (prob. media) |

**Hallazgo:** el detector **nunca dispara stridor/rhonchus en binario** sobre TR (umbral alto +
población distinta) → `pct_stridor`/`pct_rhonchus` ≈ 0. Las **probabilidades medias continuas**
(`meanp_*`) sí tienen gradiente con la severidad, salvo `meanp_stridor` (≈0, degenerada).

```
Re-extracción por ventana:
  uv run python -m pulmoia.features.extract_tr --folder data/RDB --output outputs/features_TR_windows
Perfil:
  uv run python -m pulmoia.models.copd_profile
```

---

## Paso 3 — Asignación COPD0–4 (`pulmoia.models.copd_classifier`)

Sobre el perfil por paciente (features: `meanp_wheeze`, `meanp_crackle`, `meanp_rhonchus`;
se descarta stridor por degenerado). Validación **LeaveOneOut** (42 pacientes). Logueado en
MLflow (experimento `copd_clasificador`).

**ANOVA (perfil vs nivel COPD):**

| Feature | F | p-value | Significativo (0.05) |
|---|---|---|---|
| meanp_rhonchus | 2.86 | 0.037 | ✅ |
| meanp_wheeze | 2.19 | 0.089 | marginal |
| meanp_crackle | 0.61 | 0.659 | no |

**Comparativa de métodos:**

| Método | Balanced Acc | Macro F1 | MAE ordinal |
|---|---|---|---|
| Estadístico (NearestCentroid) | 0.299 | 0.246 | 1.62 |
| ML LogReg multinomial | 0.295 | 0.252 | 1.62 |
| ML Árbol de decisión | 0.181 | 0.212 | 1.74 |

```
uv run python -m pulmoia.models.copd_classifier
```

---

## Conclusiones

- **Azar (5 clases) = 0.20** balanced accuracy. El estadístico (0.30) y LogReg (0.295) lo superan
  modestamente; el árbol hace overfit.
- El pipeline de 3 pasos (**0.30**) supera al baseline directo de features espectrales (**0.247**):
  la representación del detector **aporta señal**, coherente con la hipótesis de la arquitectura.
- En términos absolutos el desempeño es **débil** (esperable: ~40 pacientes; los eventos acústicos
  predicen la severidad solo parcialmente). `rhonchus` es el único evento significativo.
- **Estadístico ≈ ML**: con tan pocos datos el método interpretable es tan bueno como el ML, lo que
  respalda la decisión de planificación de usar el enfoque estadístico para COPD.

> Limitación clave: TR tiene ~40 pacientes. Cualquier conclusión clínica requiere más datos.
> El valor del MVP es la **cadena end-to-end funcionando**, no un clasificador COPD de alta precisión.

---

## Artefactos
`outputs/copd/`: `profiles_patient.csv`, `profiles_audio.csv`, `copd_anova.csv`,
`copd_resultados.csv`, `copd_report.md`, `copd_confusion_*.png`.
