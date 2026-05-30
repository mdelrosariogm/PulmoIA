# Selección de Características — ReliefF (Detector multi-etiqueta)

> Refuerza la selección por correlación con un método basado en relevancia (ReliefF),
> aplicado al detector de sonidos adventicios (HF+ICBHI).

---

## Motivación

La poda inicial fue **por correlación** (umbral 0.85): de ~159 → **127 features**, eliminando
redundancia. Faltaba un criterio de **relevancia** respecto a las etiquetas. Se eligió **ReliefF**
porque, a diferencia de correlación o información mutua univariada, pondera cada feature según
qué tan bien **distingue instancias vecinas de distinta clase**, capturando interacciones.

## Metodología

| Aspecto | Detalle |
|---|---|
| Entrada | 127 features correlación-podadas y estandarizadas (`train_scaled.csv`) |
| Estrategia | ReliefF **por etiqueta** (4 targets binarios), luego agregación |
| Submuestreo | **Balanceado por etiqueta** (≤1500 positivos + negativos hasta ~5000) |
| Vecinos | `n_neighbors=100` |
| Agregación | Pesos por etiqueta clip≥0, normalizados por máximo, **promediados** |
| Selección | Features que cubren el **90% del peso agregado** acumulado |
| Librería | `skrebate==0.62` |

> El submuestreo balanceado es imprescindible para `has_stridor` (≈0.3% de prevalencia): sin él,
> ReliefF no vería suficientes positivos. Para stridor se usaron los 336 positivos disponibles.

## Resultado

**Seleccionadas 79 de 127 features** (cobertura 90%).

**Top-10 por peso agregado:**

| # | Feature | Peso |
|---|---|---|
| 1 | SpectralEntropy_mean | 0.837 |
| 2 | SpectralEntropy_std | 0.780 |
| 3 | SpectralCentroid_mean | 0.727 |
| 4 | MFCC_mean_3 | 0.600 |
| 5 | SpectralSpread_mean | 0.566 |
| 6 | SpectralRolloffPoint_std | 0.473 |
| 7 | SpectralRolloffPoint_coefv | 0.473 |
| 8 | SpectralFlatness_std | 0.458 |
| 9 | MFCC_mean_5 | 0.425 |
| 10 | SpectralCentroid_std | 0.423 |

Interpretación: dominan **entropía espectral** (regularidad del sonido), **centroide/spread**
(brillo y dispersión espectral), **rolloff/flatness** y varios **MFCC** — descriptores que
caracterizan los sonidos adventicios frente a la respiración normal.

Artefactos: `outputs/feature_selection/relieff/` (`relieff_weights.csv`, `selected_features.txt`,
`relieff_top30.png`).

## Reentrenamiento con features ReliefF

Comando: `uv run python -m pulmoia.features.relieff_selection --from-saved`
(reentrena XGBoost + LogReg con las 79 features; runs en MLflow con tag `feature_set=relieff`).

| Modelo | Features | Macro ROC-AUC | Δ vs todas |
|---|---|---|---|
| XGBoost | 127 (todas) | **0.824** | — |
| XGBoost | 79 (ReliefF) | 0.815 | −0.009 |
| LogReg | 127 (todas) | 0.809 | — |
| LogReg | 79 (ReliefF) | 0.808 | −0.001 |

**Por etiqueta (XGBoost):** wheeze 0.823 (vs 0.826) · crackle 0.780 (vs 0.781) ·
rhonchus 0.875 (vs 0.874) · **stridor 0.783 (vs 0.816)** ⬇️.

**Conclusión:** con 38% menos features, 3 de 4 etiquetas quedan prácticamente idénticas; la única
caída relevante es `stridor` (clase ultra-rara, −0.03). ReliefF demuestra que ~79 features
concentran casi toda la señal útil — valioso para **eficiencia**, aunque no supera la métrica
primaria del modelo de 127 features. Los runs quedan en MLflow con tag `feature_set=relieff`
(no promovidos a producción salvo decisión explícita por eficiencia).
