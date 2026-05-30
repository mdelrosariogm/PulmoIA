# Fase 2 — Experiment Tracking (MLflow)

> Detector multilabel de sonidos adventicios. Tracking en MLflow con backend SQLite.

---

## 2.1 Configuración de MLflow

| Item | Valor |
|---|---|
| Tracking URI | `sqlite:///mlflow.db` (habilita Model Registry) |
| Artefactos | `./mlartifacts/` |
| Experimento | `detector_adventicios` |
| Modelo registrado | `pulmoia_detector` |
| Setup centralizado | `pulmoia.config.setup_mlflow()` |

**Logueo por run:** hiperparámetros (`hp_*`), `scale_pos_weight`, métricas (`cv_roc_auc`,
`test_roc_auc`, `test_pr_auc`, `test_f1`, `threshold`), y el modelo ensamblado como artefacto
con **signature + input_example** (listo para servir en Fase 4).

**Estructura de runs (jerárquica):**
```
Experimento "detector_adventicios"
└── run padre (por algoritmo)        → métricas macro + modelo (artefacto)
      └── run anidado (por etiqueta) → hiperparámetros + métricas por clase
```

Abrir la UI:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 2.2 Experimentos

**Metodología (por algoritmo y etiqueta, estrategia One-vs-Rest):**
1. `RandomizedSearchCV` con **GroupKFold por `filename`** (anti-leakage), scoring = ROC-AUC.
2. Selección de **umbral por F1 sobre predicciones out-of-fold** (sin tocar test).
3. Refit de los mejores hiperparámetros sobre **todo el train**.
4. Evaluación en **test split por archivo**.

Reproducir:
```bash
uv run python -m pulmoia.models.train_detector            # RF + XGBoost + LogReg
uv run python -m pulmoia.models.train_detector --smoke    # validación rápida
```

### Resultados (test split por archivo)

| Algoritmo | Macro ROC-AUC | Macro F1 | Comentario |
|---|---|---|---|
| **Random Forest** | **0.829** | 0.344 | Mejor métrica, pero ~2 h de entrenamiento |
| **XGBoost** | 0.824 | 0.340 | Casi igual, rápido, mejor en stridor → **producción** |
| LogReg (baseline) | 0.809 | 0.323 | Referencia lineal |

**Por etiqueta (ROC-AUC / F1):**

| Etiqueta | Prevalencia | Random Forest | XGBoost |
|---|---|---|---|
| has_wheeze | 13.2% | 0.819 / 0.475 | 0.826 / 0.431 |
| has_crackle | 22.0% | 0.780 / 0.516 | 0.781 / 0.504 |
| has_rhonchus | 5.6% | 0.861 / 0.373 | 0.874 / 0.323 |
| has_stridor | 0.4% | 0.856 / 0.013 | 0.816 / **0.104** |

**Aprendizajes:**
- Los 3 algoritmos superan el umbral MVP (Macro ROC-AUC ≥ 0.80); el objetivo ≥0.88 queda para
  mejoras de feature engineering / arquitectura.
- `stridor` (0.4% de prevalencia) es la clase más difícil: ROC-AUC alto pero F1 bajo. XGBoost con
  `scale_pos_weight` la maneja ~8× mejor en F1 (0.104 vs 0.013).
- RF gana por 0.005 en macro AUC pero cuesta ~2 h de entrenamiento → no es práctico para retraining.

---

## 2.3 Model Registry

Reproducir:
```bash
uv run python -m pulmoia.models.register_model
```

| Versión | Algoritmo | Stage | Alias | Macro ROC-AUC |
|---|---|---|---|---|
| **v1** | XGBoost | **Production** | champion | 0.824 |
| v2 | Random Forest | Staging | challenger | 0.829 |

**Criterio de promoción:** XGBoost a Production por su equilibrio rendimiento/coste (entrena en
minutos, sirve rápido, mejor recall en clases raras). RF queda en Staging como referencia de
máxima métrica. Cada versión lleva tags (`algo`, `macro_test_roc_auc`, `dataset`) y descripción.

---

## Decisiones para Fase 3 (Prefect)

- El pipeline de retraining usará **solo XGBoost (+ LogReg como baseline)**. Random Forest queda
  documentado como experimento puntual (su búsqueda de ~2 h no es apta para un flujo reproducible).
