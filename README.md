# 🫁 Respiratory Feature Extraction — RespiratoryDatabase@TR

Extracción automática de features espectrales de audio para la base de datos **RespiratoryDatabase@TR** (Altan et al., 2017), orientada al entrenamiento de modelos de Machine Learning para clasificación de enfermedades pulmonares (COPD0–COPD4).

---

## 📁 Estructura del repositorio

```
.
├── extract_features_TR.py   # Script principal de extracción
├── requirements.txt         # Dependencias Python
└── README.md                # Este archivo
```

---

## ⚙️ Instalación del entorno

### Requisitos previos
- Python **3.10 o superior** (probado con 3.12)
- `pip` actualizado

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd TU_REPO

# 2. Crear un ambiente virtual
python -m venv venv

# 3. Activar el ambiente virtual
#    En macOS / Linux:
source venv/bin/activate
#    En Windows:
venv\Scripts\activate

# 4. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Uso

```bash
python extract_features_TR.py --folder /ruta/a/tus/wavs --output features_TR.xlsx
```

| Argumento  | Descripción                                          | Por defecto        |
|------------|------------------------------------------------------|--------------------|
| `--folder` | Carpeta que contiene los archivos `.wav` de TR        | *(requerido)*      |
| `--output` | Nombre del archivo de salida (sin extensión necesaria)| `features_TR.xlsx` |

El script genera **dos archivos** en simultáneo:
- `features_TR.csv` — para procesamiento programático
- `features_TR.xlsx` — para exploración rápida en Excel

### Ejemplo con estructura típica

```
audios/
├── H002_L1.wav
├── H002_L2.wav
├── H002_R1.wav
├── H003_L1.wav
└── ...
```

```bash
python extract_features_TR.py --folder audios/ --output resultados/features_TR
```

---

## 📊 Features extraídos

Por cada archivo `.wav` se extrae **una fila** con los siguientes descriptores. El ventaneo es deslizante (1 s, 50 % solapamiento) y cada feature se agrega como `mean`, `std` y `coef. de variación`.

| Grupo              | Features                                                                 | Columnas |
|--------------------|--------------------------------------------------------------------------|----------|
| Metadatos          | `filename`, `patient_id`, `channel`, `diagnosis`                        | 4        |
| Spectral Centroid  | mean / std / coefv                                                       | 3        |
| Spectral Spread    | mean / std / coefv                                                       | 3        |
| Spectral Rolloff   | mean / std / coefv                                                       | 3        |
| Spectral Flatness  | mean / std / coefv                                                       | 3        |
| Spectral Flux      | mean / std / coefv                                                       | 3        |
| Spectral Skewness  | mean / std / coefv                                                       | 3        |
| Spectral Kurtosis  | mean / std / coefv                                                       | 3        |
| Spectral Entropy   | mean / std / coefv                                                       | 3        |
| Spectral Crest     | mean / std / coefv                                                       | 3        |
| Spectral Slope     | mean / std / coefv                                                       | 3        |
| Spectral Decrease  | mean / std / coefv                                                       | 3        |
| MFCC (×14)         | mean / std / coefv por coeficiente                                       | 42       |
| MFCC delta (×14)   | mean / std / coefv por coeficiente                                       | 42       |
| MFCC Δ² (×14)      | mean / std / coefv por coeficiente                                       | 42       |
| **Total**          |                                                                          | **~157** |

---

## 🏷️ Etiquetas de diagnóstico

| Etiqueta | Descripción                        |
|----------|------------------------------------|
| COPD0    | Bajo riesgo (PFT normal)           |
| COPD1    | Nivel leve                         |
| COPD2    | Nivel moderado                     |
| COPD3    | Nivel severo                       |
| COPD4    | Nivel muy severo                   |

---

## 🔧 Parámetros ajustables

En la sección de constantes al inicio de `extract_features_TR.py`:

```python
WINDOW_SEC = 1.0   # Duración de ventana en segundos
HOP_SEC    = 0.5   # Paso entre ventanas (50 % solapamiento)
N_MFCC     = 14    # Número de coeficientes MFCC
N_FFT      = 512   # Tamaño de la FFT
HOP_FFT    = 128   # Hop interno del espectrograma
```

---

## 📚 Referencia

```
Altan, G., Kutlu, Y., Garbi, Y., Pekmezci, A. Ö., & Nural, S. (2017).
Multimedia Respiratory Database (RespiratoryDatabase@TR): Auscultation Sounds and Chest X-rays.
Natural and Engineering Sciences, 2(3), 59–72.
```

---

## 📝 Notas

- Los archivos deben seguir el patrón de nombre `HXXX_YN.wav` (ej: `H002_L1.wav`).
- Si un audio es más corto que `WINDOW_SEC`, se procesa completo como una única ventana.
- Archivos cuyo ID de paciente no esté en el diccionario de diagnósticos son omitidos con advertencia.
