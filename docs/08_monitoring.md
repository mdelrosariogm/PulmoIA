# Fase 5 — Monitoreo (diseño + demo de drift)

> Propuesta de monitoreo del modelo en producción, con un **demo funcional** de data drift.

---

## Qué monitorear

| Categoría | Señal | Cómo | Acción si se dispara |
|---|---|---|---|
| **Data drift** | Cambio en la distribución de las features de entrada | **PSI** + **KS test** por feature (implementado) | Revisar fuente de datos; reentrenar si persiste |
| **Prediction drift** | Cambio en la distribución de salidas (perfil acústico, mix COPD0–4) | PSI sobre las probabilidades/proporciones predichas | Investigar; comparar con histórico |
| **Performance decay** | Caída de Macro ROC-AUC (detector) / balanced acc (COPD) | Evaluar en lotes etiquetados periódicos | Reentrenar (flow de Prefect) |
| **Calibración** | Probabilidades mal calibradas | Brier score / reliability curve | Recalibrar umbrales |
| **Operacional** | Latencia, tasa de error, nº de ventanas por audio | Logs de la API + métricas | Escalar / alertar |

---

## Demo funcional de data drift (`pulmoia.monitoring.drift`)

Compara la distribución de las **79 features ReliefF** entre referencia y actual:
- **PSI**: <0.1 sin drift · 0.1–0.25 moderado · >0.25 significativo
- **KS test**: p<0.05 ⇒ distribuciones distintas

```bash
uv run python -m pulmoia.monitoring.drift --current test   # control (mismo dominio)
uv run python -m pulmoia.monitoring.drift --current tr      # nuevo dominio (TR)
```

### Resultados

| Comparación | Features con drift | Veredicto |
|---|---|---|
| train vs **test** (mismo dominio) | 0 / 79 | ✅ OK (sin drift) |
| train vs **TR** (otra población) | **44 significativo + 5 moderado** / 79 | 🚨 ALERTA |

**Lectura:** el monitor no reporta drift in-domain (test) y detecta drift **severo** en TR
(MFCC y descriptores espectrales con PSI 5–12, KS p≈0). Esto **explica empíricamente** por qué
el clasificador COPD rinde poco: los audios de TR están distribucionalmente muy lejos del
dominio de entrenamiento del detector (otro hospital/dispositivo/población).

Artefactos: `outputs/monitoring/drift_{tr,test}.csv` + `.png`.

---

## Lazo de reentrenamiento (integración con Fase 3)

```
[API en producción] → logs de entradas
        ↓ (batch periódico)
[monitoring.drift] → ¿drift significativo?
        ↓ sí
[Prefect: pulmoia-pipeline] → valida datos → reentrena (XGBoost+LogReg) → registra
        ↓
[Model Registry] → nueva versión champion → la API la sirve
```

- **Programación:** un flow de Prefect ejecuta el chequeo de drift (p. ej. semanal) y, si el
  veredicto es ALERTA, dispara el `full_pipeline` de reentrenamiento (Fase 3).
- **Alertas:** el veredicto y el nº de features drifteadas se loguean; en producción se
  enviarían a Slack/email/Prometheus-Alertmanager.

---

## Próximos pasos (no implementados — diseño)
- Persistir las entradas de la API para construir los lotes de monitoreo.
- Prediction drift sobre el mix COPD0–4 servido vs histórico.
- Dashboard (Grafana/Evidently) con PSI por feature en el tiempo.
- Detección de outliers/audio fuera de especificación (sample rate, duración, silencio).
