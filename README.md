# 🫁 pulmoia — Pulmonary Disease Classification from Respiratory Auscultation Sounds

> **Proyecto de Grado** | Especialización en Inteligencia Artificial  
> Pipeline de extracción de features espectrales para clasificación automática de EPOC (COPD) a partir de sonidos de auscultación pulmonar.

---

## 📌 Resumen del Proyecto

Este repositorio implementa un pipeline de dos etapas para la clasificación automática de la **Enfermedad Pulmonar Obstructiva Crónica (EPOC/COPD)** a partir de grabaciones de auscultación pulmonar:

1. **Etapa 1 — Detector de sonidos adventicios** *(Camino B — Transfer Learning)*: Se entrena un modelo detector de eventos acústicos anormales (wheeze, crackle, stridor, rhonchus) usando bases de datos con anotaciones acústicas (HF_Lung_V1, ICBHI 2017). Este modelo aprende a reconocer los patrones sonoros que los clínicos asocian con patología pulmonar.

2. **Etapa 2 — Clasificador de severidad COPD**: Las representaciones aprendidas por el detector se combinan con features espectrales clásicos para clasificar la severidad de la enfermedad (COPD0 → COPD4) usando la base de datos RespiratoryDatabase@TR, que sí posee diagnósticos clínicos validados.

Esta arquitectura en dos etapas permite que el pipeline sea **escalable**: nuevas bases de datos con anotaciones acústicas pueden incorporarse al Paso 1 sin necesidad de reentrenar el clasificador de COPD.

---

## 🗄️ Bases de Datos

### 1. RespiratoryDatabase@TR (TR)
> Altan G. et al. (2017). *Multimedia Respiratory Database (RespiratoryDatabase@TR): Auscultation Sounds and Chest X-rays*. Natural and Engineering Sciences, 2(3), 59–72.

| Característica | Detalle |
|---|---|
| Institución | Antakya State Hospital, Hatay, Turkey |
| Dispositivo | Littmann 3200 Electronic Stethoscope |
| Frecuencia de muestreo | 4,000 Hz, 16 bits |
| Formato | WAV (.wav) |
| Sujetos totales | 77 (30 sanos + 47 con patología) |
| Canales de pulmón | 12 canales (L1–L6, R1–R6) |
| Canales de corazón | 4 canales (R7, L7, R8, L8) |
| Duración por grabación | Variable (hasta ~20s) |
| **Etiqueta clínica** | **Diagnóstico de COPD por PFT (FEV1/FVC)** |

**Distribución de diagnósticos (pacientes usados en este proyecto):**

| Diagnóstico | Descripción clínica | Criterio PFT | N pacientes |
|---|---|---|---|
| COPD0 | Bajo riesgo | PFT normal + síntomas crónicos | 6 |
| COPD1 | Leve | FEV1/FVC < 70%, FEV1 ≥ 80% | 5 |
| COPD2 | Moderado | FEV1/FVC < 70%, 50% < FEV1 < 80% | 6 |
| COPD3 | Severo | FEV1/FVC < 70%, 30% < FEV1 < 50% | 6 |
| COPD4 | Muy severo | FEV1/FVC < 70%, FEV1 < 30% | 17 |

**Estructura de archivos:**
```
data/RDB/
├── H002_L1.wav     ← paciente H002, canal L1
├── H002_L2.wav
├── H002_R1.wav
└── ...
```

**Nombre de archivo:** `{PatientID}_{Canal}.wav`  
donde `Canal` ∈ {L1–L6, R1–R6, L7–L8, R7–R8}

---

### 2. HF_Lung_V1 (HF)
> Hsu F-S. et al. (2021). *Benchmarking of eight recurrent neural network variants for breath phase and adventitious sound detection on a self-developed open-access lung sound database — HF_Lung_V1*. PLoS ONE 16(7): e0254134.

| Característica | Detalle |
|---|---|
| Institución | Far Eastern Memorial Hospital + TSECC, Taiwan |
| Dispositivos | Littmann 3200 (steth_) / HF-Type-1 multicanal (trunc_) |
| Frecuencia de muestreo | 4,000 Hz, 16 bits |
| Formato | WAV (.wav) + etiquetas TXT (.txt) |
| Sujetos totales | 279 (261 TSECC + 18 RCW/RCC) |
| Grabaciones totales | 9,765 archivos de 15 segundos |
| **Etiqueta clínica** | **Eventos acústicos con timestamps** |

**Archivos utilizados en este proyecto:** Solo archivos `steth_*` (grabados con Littmann 3200, mismo dispositivo que TR).

| Subconjunto | N archivos | N sujetos |
|---|---|---|
| steth_ (Littmann 3200) | ~4,504 | 261 |
| trunc_ (HF-Type-1) — **no usado** | ~5,261 | 18 |

**Etiquetas acústicas disponibles (por timestamps):**

| Etiqueta | Tipo | Descripción clínica |
|---|---|---|
| I | Fase respiratoria | Inhalación — **ignorada** |
| E | Fase respiratoria | Exhalación — **ignorada** |
| Wheeze | CAS | Sonido continuo adventicio, >250ms |
| Stridor | CAS | Sonido continuo adventicio de vía aérea alta |
| Rhonchus | CAS | Sonido continuo adventicio de vía aérea baja |
| D | DAS | Sonido discontinuo adventicio (crackle) |

> **Nota metodológica:** Las etiquetas de fase respiratoria (I/E) se ignoran deliberadamente para que el modelo no aprenda patrones dependientes de la segmentación por ciclo respiratorio, asegurando generalización a bases de datos sin esta información.

**Estructura de archivos:**
```
data/LUNGS/
├── steth_20190801_09_46_05.wav
├── steth_20190801_09_46_05_label.txt   ← timestamps de eventos
└── ...
```

**Formato del archivo de etiquetas:**
```
Wheeze  00:00:03.144  00:00:03.682
D       00:00:04.005  00:00:04.543
I       00:00:06.307  00:00:07.527    ← ignorado
E       00:00:07.548  00:00:08.308    ← ignorado
```

---

### 3. ICBHI 2017 Respiratory Sound Database (ICBHI)
> Rocha BM. et al. (2019). *An open access database for the evaluation of respiratory sound classification algorithms*. Physiological Measurement, 40, 035001.

| Característica | Detalle |
|---|---|
| Instituciones | Univ. Aveiro (Portugal) + Univ. Thessaloniki / Coimbra (Greece) |
| Dispositivos | AKG C417L, Littmann Classic II SE, Littmann 3200, Meditron |
| Frecuencia de muestreo | Variable (resamplada a 4,000 Hz en este proyecto) |
| Formato | WAV (.wav) + anotaciones TXT (.txt) |
| Sujetos totales | 126 |
| Grabaciones totales | 920 archivos (10s a 90s de duración) |
| Ciclos respiratorios | 6,898 ciclos anotados |
| **Etiqueta** | **Crackle y Wheeze por ciclo respiratorio** |

**Distribución de etiquetas acústicas:**

| Contenido | N ciclos |
|---|---|
| Normal (sin adventicios) | 3,642 |
| Solo crackles | 1,864 |
| Solo wheezes | 886 |
| Crackles + wheezes | 506 |

**Estructura de archivos:**
```
data/ICBHI/
├── 101_1b1_Al_sc_Meditron.wav
├── 101_1b1_Al_sc_Meditron.txt
└── ...
```

**Nombre de archivo:** `{PatientID}_{RecordingIndex}_{Location}_{Mode}_{Equipment}`

| Campo | Valores posibles |
|---|---|
| Location | Tc, Al, Ar, Pl, Pr, Ll, Lr |
| Mode | sc (single channel), mc (multichannel) |
| Equipment | AKGC417L, LittC2SE, Litt3200, Meditron |

**Formato del archivo de anotaciones (ciclos respiratorios):**
```
t_inicio    t_fin    crackle    wheeze
0.036       1.207    0          0
1.207       3.550    0          0
3.550       5.750    1          0      ← crackle presente
5.750       7.879    1          0
```

> **Diagnósticos disponibles:** COPD, LRTI (Lower Respiratory Tract Infection), URTI (Upper Respiratory Tract Infection), Healthy, Asthma, Bronchiectasis, entre otros. En este proyecto ICBHI se usa exclusivamente para el Paso 1 (detección de eventos acústicos), no para clasificación de COPD.

---

## ⚙️ Metodología de Extracción de Features

### Preprocesamiento Estándar (todas las bases)

Todos los audios pasan por el mismo pipeline de preprocesamiento antes de la extracción, garantizando homogeneidad entre bases de datos:

```
Audio crudo (.wav)
       ↓
1. Conversión a mono (canal único)
       ↓
2. Verificación de frecuencia de muestreo
   → Si ≠ 4,000 Hz: resampleo a 4,000 Hz (librosa)
       ↓
3. Filtro pasa-altos Butterworth
   → Orden: 10 | Frecuencia de corte: 80 Hz
   → Elimina interferencia eléctrica (60 Hz) y ruido cardíaco
   → Implementado en formato SOS (numéricamente estable)
   → Igual que Hsu et al. (HF_Lung_V1, PLoS ONE 2021)
       ↓
Señal lista para extracción
```

### Ventaneo Deslizante

En lugar de extraer un único vector por archivo, se aplica ventaneo deslizante para obtener múltiples observaciones por grabación:

| Parámetro | Valor | Justificación |
|---|---|---|
| Duración de ventana | 1.0 s | Captura al menos un ciclo respiratorio parcial |
| Solapamiento | 50% (hop = 0.5s) | Aumentación de datos natural |
| Ventanas por archivo 15s | ~29 | Depende de duración real |
| Muestras por ventana | 4,000 | A 4,000 Hz |

**Una fila en el CSV = una ventana de 1 segundo.**

### Etiquetado por Ventana

El etiquetado difiere según la base de datos:

**HF_Lung_V1 y ICBHI** (etiquetas acústicas por timestamp):
```
Ventana [t_start, t_end] tiene has_wheeze=1 si:
    ∃ evento Wheeze tal que: wheeze_start < t_end AND wheeze_end > t_start
```
Criterio de solapamiento temporal. Una ventana puede tener múltiples etiquetas activas simultáneamente (multilabel).

**RespiratoryDatabase@TR** (etiqueta de diagnóstico):
```
Todas las ventanas de un mismo archivo heredan el diagnóstico
clínico del paciente: COPD0, COPD1, COPD2, COPD3, o COPD4
```

### Features Espectrales Extraídos

Por cada ventana de 1 segundo se extraen los siguientes descriptores, cada uno resumido en tres estadísticos (mean, std, coef. de variación):

| Feature | Descripción | N columnas |
|---|---|---|
| Spectral Centroid | Centro de masa del espectro de frecuencias | 3 |
| Spectral Spread | Dispersión alrededor del centroide (bandwidth) | 3 |
| Spectral Rolloff | Frecuencia por debajo de la cual se concentra el 85% de la energía | 3 |
| Spectral Flatness | Cociente entre media geométrica y aritmética del espectro | 3 |
| Spectral Flux | Tasa de cambio espectral entre frames consecutivos | 3 |
| Spectral Skewness | Asimetría de la distribución espectral de potencia | 3 |
| Spectral Kurtosis | Curtosis de la distribución espectral de potencia | 3 |
| Spectral Entropy | Entropía de la distribución espectral (regularidad) | 3 |
| Spectral Crest | Relación entre pico máximo y media espectral | 3 |
| Spectral Slope | Pendiente lineal del espectro de magnitud | 3 |
| Spectral Decrease | Tasa de decrecimiento espectral ponderada | 3 |
| MFCC (×14 coefs) | Coeficientes Mel-Frequency Cepstral | 42 |
| MFCC Δ (×14 coefs) | Deltas de los MFCCs (velocidad temporal) | 42 |
| MFCC Δ² (×14 coefs) | Delta-deltas de los MFCCs (aceleración temporal) | 42 |
| **Total** | | **~157 columnas de features** |

**Parámetros del espectrograma:**

| Parámetro | Valor |
|---|---|
| FFT size (N_FFT) | 512 puntos |
| Hop length interno | 128 muestras |
| Resolución frecuencial | ≈ 7.8 Hz a 4,000 Hz |
| Bandas Mel (MFCCs) | 14 coeficientes |

---

## 📊 Tamaños de Dataset Resultantes

### Antes de la extracción (archivos de audio)

| Base de datos | Archivos .wav | Sujetos | Duración total |
|---|---|---|---|
| HF_Lung_V1 (steth_) | ~4,504 | 261 | ~1,126 min |
| RespiratoryDatabase@TR | ~500 | 42 | Variable |
| ICBHI 2017 | 920 | 126 | ~330 min |
| **Total** | **~5,924** | **~429** | **~1,800 min** |

### Después de la extracción (ventanas de 1s)

| Base de datos | Archivos .wav | Ventanas reales | Columnas | Uso en pipeline |
|---|---|---|---|---|
| HF_Lung_V1 (steth_) | 4,504 | **130,616** | 169 | Paso 1: detector acústico |
| ICBHI 2017 | 920 | **~37,800** | 169 | Paso 1: refuerzo detector |
| RespiratoryDatabase@TR | ~500 | **~15,000** | 169 | Paso 2: clasificador COPD |
| **Total** | **~5,924** | **~183,000** | | |

> Ventana: 1s duración, 50% solapamiento (hop=0.5s). Valores de ICBHI y TR se actualizarán al finalizar extracción.

**Distribución de etiquetas acústicas — HF_Lung_V1 (130,616 ventanas):**

| Etiqueta | Ventanas positivas | % del total |
|---|---|---|
| has_crackle | 23,048 | 17.6% |
| has_wheeze | 13,720 | 10.5% |
| has_rhonchus | 9,173 | 7.0% |
| has_stridor | 520 | 0.4% |
| Normal (sin adventicios) | ~84,155 | ~64.4% |

**Distribución de etiquetas acústicas — ICBHI 2017 (ciclos respiratorios):**

| Contenido | Ciclos | % |
|---|---|---|
| Normal (sin adventicios) | 3,642 | 52.8% |
| Solo crackle | 1,864 | 27.0% |
| Solo wheeze | 886 | 12.8% |
| Crackle + wheeze | 506 | 7.3% |
| **Total** | **6,898** | |

**Distribución de diagnósticos COPD — RespiratoryDatabase@TR (por paciente):**

| Diagnóstico | Pacientes | Criterio FEV1/FVC |
|---|---|---|
| COPD4 (muy severo) | 17 | FEV1 < 30% |
| COPD3 (severo) | 6 | 30% < FEV1 < 50% |
| COPD2 (moderado) | 6 | 50% < FEV1 < 80% |
| COPD0 (bajo riesgo) | 6 | PFT normal |
| COPD1 (leve) | 5 | FEV1 ≥ 80% |
| **Total** | **40** | |

---

## 🚀 Instalación y Uso

### Requisitos

- Python 3.10 o superior
- Git

### Configuración del entorno

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/pulmoia.git
cd pulmoia

# 2. Crear ambiente virtual
python -m venv vpulmoia

# 3. Activar el ambiente
# macOS/Linux:
source vpulmoia/bin/activate
# Windows:
vpulmoia\Scripts\activate

# 4. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### Estructura del repositorio

```
pulmoia/
├── data/
│   ├── RDB/          ← audios TR (.wav) — no versionado
│   ├── LUNGS/        ← audios HF (.wav + _label.txt) — no versionado
│   ├── ICBHI/        ← audios ICBHI (.wav + .txt) — no versionado
│   └── .gitkeep
├── outputs/          ← CSVs y Excel generados — no versionado
├── models/           ← modelos entrenados — no versionado
├── src/              ← módulos reutilizables (futuro)
├── scripts/          ← scripts auxiliares
├── tests/            ← pruebas unitarias
├── extract_features_TR.py    ← extracción RespiratoryDatabase@TR
├── extract_features_HF.py    ← extracción HF_Lung_V1
├── verify_labels_HF.py       ← verificación de etiquetado HF
├── requirements.txt
└── README.md
```

### Extracción de features

```bash
# RespiratoryDatabase@TR (una fila por ventana, etiqueta: diagnosis)
python extract_features_TR.py --folder data/RDB --output outputs/features_TR

# HF_Lung_V1 (una fila por ventana, etiquetas: has_wheeze, has_crackle...)
python extract_features_HF.py --folder data/LUNGS --output outputs/features_HF

# Verificar etiquetado de HF antes de la extracción completa
python verify_labels_HF.py --folder data/LUNGS
python verify_labels_HF.py --folder data/LUNGS --file steth_20190801_09_46_05
```

### Estructura de los archivos de salida

**features_HF.csv** (HF_Lung_V1):
```
filename | source | t_start_s | t_end_s | has_wheeze | has_crackle | has_stridor | has_rhonchus | SpectralCentroid_mean | ...
```

**features_TR.csv** (RespiratoryDatabase@TR):
```
filename | source | patient_id | channel | t_start_s | t_end_s | diagnosis | sr_original | resampled | SpectralCentroid_mean | ...
```

---

## 🗺️ Roadmap del Proyecto

- [x] Extracción de features espectrales — RespiratoryDatabase@TR (por ventana)
- [x] Extracción de features espectrales — HF_Lung_V1 steth_ (por ventana, multilabel)
- [x] Verificación de etiquetado por solapamiento temporal
- [ ] Extracción de features espectrales — ICBHI 2017
- [ ] Paso 1: Entrenamiento del detector de sonidos adventicios (HF + ICBHI)
- [ ] Paso 2: Aplicación del detector a TR → embeddings acústicos
- [ ] Paso 3: Clasificador de severidad COPD (features espectrales + embeddings)
- [ ] Evaluación con validación cruzada por paciente
- [ ] Análisis de importancia de features

---

## 📚 Referencias

1. Altan G. et al. (2017). Multimedia Respiratory Database (RespiratoryDatabase@TR): Auscultation Sounds and Chest X-rays. *Natural and Engineering Sciences*, 2(3), 59–72.

2. Hsu F-S. et al. (2021). Benchmarking of eight recurrent neural network variants for breath phase and adventitious sound detection on a self-developed open-access lung sound database — HF_Lung_V1. *PLoS ONE* 16(7): e0254134. https://doi.org/10.1371/journal.pone.0254134

3. Rocha BM. et al. (2019). An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 40, 035001. https://doi.org/10.1088/1361-6579/ab03ea

4. Decramer M. et al. (2012). Chronic obstructive pulmonary disease. *Lancet*, 379(9823), 1341–51.

---

## 📄 Licencia

- HF_Lung_V1: Creative Commons Attribution 4.0 (CC BY 4.0)
- ICBHI 2017: Libre para investigación (citar Rocha et al. 2019)
- RespiratoryDatabase@TR: Uso académico
- Código: MIT License
