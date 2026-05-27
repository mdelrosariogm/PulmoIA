"""
extract_features_TR.py
======================
Extracción de features espectrales — RespiratoryDatabase@TR

Pipeline por archivo:
  1. Carga el audio (cualquier sr)
  2. Convierte a mono
  3. Resamplea a TARGET_SR (4 kHz) si es necesario
  4. Aplica filtro pasa-altos Butterworth orden 10 a 80 Hz
     (elimina interferencia eléctrica 60 Hz y ruido cardíaco,
      igual que Hsu et al., HF_Lung_V1, PLoS ONE 2021)
  5. Ventaneo deslizante → features espectrales por ventana
  6. Agrega (mean, std, coef. de variación) → 1 fila por archivo
  7. Guarda CSV + Excel

Uso:
  python extract_features_TR.py --folder /ruta/RDB --output outputs/features_TR

Dependencias:
  pip install numpy scipy librosa pandas openpyxl tqdm
"""

import os
import re
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, sosfilt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Parámetros globales ───────────────────────────────────────────────────────
TARGET_SR    = 4_000   # Hz — frecuencia objetivo (igual que TR y HF_Lung_V1)
HP_CUTOFF    = 80      # Hz — corte del filtro pasa-altos
HP_ORDER     = 10      # orden del filtro Butterworth (igual que HF_Lung_V1)
WINDOW_SEC   = 1.0     # duración de ventana en segundos
HOP_SEC      = 0.5     # solapamiento 50 %
N_MFCC       = 14      # coeficientes MFCC (igual que código MATLAB original)
N_FFT        = 512     # tamaño FFT (resolución ≈ 7.8 Hz a 4 kHz)
HOP_FFT      = 128     # hop interno espectrograma

# ─── Diccionario de diagnósticos ──────────────────────────────────────────────
DIAGNOSIS = {
    "H002": "COPD4", "H003": "COPD4", "H004": "COPD4", "H005": "COPD4",
    "H006": "COPD4", "H007": "COPD3", "H008": "COPD3", "H009": "COPD4",
    "H010": "COPD3", "H011": "COPD4", "H012": "COPD4", "H013": "COPD4",
    "H014": "COPD4", "H015": "COPD4", "H016": "COPD0", "H017": "COPD1",
    "H018": "COPD2", "H021": "COPD0", "H022": "COPD4", "H023": "COPD4",
    "H024": "COPD4", "H025": "COPD4", "H026": "COPD3", "H028": "COPD2",
    "H029": "COPD1", "H030": "COPD2", "H031": "COPD2", "H032": "COPD4",
    "H033": "COPD3", "H034": "COPD3", "H035": "COPD4", "H036": "COPD3",
    "H037": "COPD0", "H038": "COPD2", "H039": "COPD1", "H040": "COPD0",
    "H041": "COPD0", "H042": "COPD2", "H043": "COPD1", "H044": "COPD2",
    "H045": "COPD1", "H050": "COPD0",
}

# ─── Filtro pasa-altos ────────────────────────────────────────────────────────

def build_highpass(cutoff: float = HP_CUTOFF,
                   order: int = HP_ORDER,
                   fs: int = TARGET_SR):
    """Construye un filtro Butterworth pasa-altos en formato SOS (estable)."""
    return butter(order, cutoff, btype="high", fs=fs, output="sos")

# Precalculado una sola vez para evitar recomputar en cada archivo
_HP_SOS = build_highpass()


def apply_highpass(signal: np.ndarray) -> np.ndarray:
    """Aplica el filtro pasa-altos a la señal."""
    return sosfilt(_HP_SOS, signal)


# ─── Carga y preprocesamiento de audio ───────────────────────────────────────

def load_and_preprocess(filepath: str) -> tuple[np.ndarray, int, dict]:
    """
    Carga el audio, convierte a mono, resamplea a TARGET_SR si es necesario
    y aplica el filtro pasa-altos.

    Returns
    -------
    signal : np.ndarray  señal procesada
    fs     : int         frecuencia de muestreo final (= TARGET_SR)
    info   : dict        metadatos de preprocesamiento (sr_original, resampled, duration_s)
    """
    # Carga raw para verificar sr original
    signal_raw, sr_orig = librosa.load(filepath, sr=None, mono=True)
    resampled = False

    if sr_orig != TARGET_SR:
        log.debug("  Resampleando %s Hz → %s Hz", sr_orig, TARGET_SR)
        signal_raw = librosa.resample(signal_raw, orig_sr=sr_orig, target_sr=TARGET_SR)
        resampled = True

    # Filtro pasa-altos 80 Hz
    signal_filtered = apply_highpass(signal_raw)

    info = {
        "sr_original":  sr_orig,
        "resampled":    resampled,
        "duration_s":   round(len(signal_filtered) / TARGET_SR, 3),
    }
    return signal_filtered, TARGET_SR, info


# ─── Features espectrales por ventana ────────────────────────────────────────

def aggregate(values: np.ndarray) -> tuple[float, float, float]:
    """Devuelve (mean, std, coef_variacion) de un vector 1-D."""
    m  = float(np.mean(values))
    s  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = s / m if m != 0 else np.nan
    return m, s, cv


def spectral_features_window(frame: np.ndarray, fs: int) -> dict:
    """Extrae todos los features espectrales de un frame."""
    feats = {}

    # Espectrograma de magnitud
    S_mag   = np.abs(librosa.stft(frame, n_fft=N_FFT, hop_length=HOP_FFT))
    freqs   = librosa.fft_frequencies(sr=fs, n_fft=N_FFT)
    power   = S_mag ** 2
    p_norm  = power / (power.sum(axis=0, keepdims=True) + 1e-12)

    # Spectral Centroid y Spread (librosa)
    feats["SpectralCentroid"] = librosa.feature.spectral_centroid(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT)[0]
    feats["SpectralSpread"]   = librosa.feature.spectral_bandwidth(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT)[0]

    # Rolloff
    feats["SpectralRolloffPoint"] = librosa.feature.spectral_rolloff(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT, roll_percent=0.85)[0]

    # Flatness
    feats["SpectralFlatness"] = librosa.feature.spectral_flatness(
        y=frame, n_fft=N_FFT, hop_length=HOP_FFT)[0]

    # Flux
    flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
    feats["SpectralFlux"] = flux if len(flux) > 0 else np.array([0.0])

    # Momentos espectrales (Skewness, Kurtosis, Entropy)
    mu1   = (freqs[:, None] * p_norm).sum(axis=0)
    mu2   = (((freqs[:, None] - mu1) ** 2) * p_norm).sum(axis=0)
    mu3   = (((freqs[:, None] - mu1) ** 3) * p_norm).sum(axis=0)
    sigma = np.sqrt(mu2) + 1e-12
    feats["SpectralSkewness"] = mu3 / sigma ** 3

    mu4 = (((freqs[:, None] - mu1) ** 4) * p_norm).sum(axis=0)
    feats["SpectralKurtosis"] = mu4 / sigma ** 4

    p_safe = np.where(p_norm > 1e-12, p_norm, 1e-12)
    feats["SpectralEntropy"] = -np.sum(p_safe * np.log2(p_safe), axis=0)

    # Crest
    feats["SpectralCrest"] = S_mag.max(axis=0) / (S_mag.mean(axis=0) + 1e-12)

    # Slope
    K      = len(freqs)
    f_mean = freqs.mean()
    num_sl = ((freqs - f_mean)[:, None] * S_mag).sum(axis=0)
    den_sl = ((freqs - f_mean) ** 2).sum() + 1e-12
    feats["SpectralSlope"] = num_sl / den_sl

    # Decrease
    k_idx  = np.arange(1, K)
    num_dc = ((S_mag[1:] - S_mag[[0]]) / (k_idx[:, None] + 1e-12)).sum(axis=0)
    den_dc = S_mag[1:].sum(axis=0) + 1e-12
    feats["SpectralDecrease"] = num_dc / den_dc

    # MFCCs + delta + delta-delta (N_MFCC coefs, sin energía logarítmica)
    mfcc   = librosa.feature.mfcc(y=frame, sr=fs, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_FFT)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats["MFCC"]             = mfcc.T
    feats["MFCC_delta"]       = delta1.T
    feats["MFCC_delta_delta"] = delta2.T

    return feats


# ─── Extracción sobre el archivo completo ────────────────────────────────────

SCALAR_KEYS = [
    "SpectralCentroid", "SpectralSpread", "SpectralRolloffPoint",
    "SpectralFlatness", "SpectralFlux", "SpectralSkewness",
    "SpectralKurtosis", "SpectralEntropy", "SpectralCrest",
    "SpectralSlope", "SpectralDecrease",
]
VECTOR_KEYS = ["MFCC", "MFCC_delta", "MFCC_delta_delta"]


def extract_file_features(signal: np.ndarray, fs: int) -> dict | None:
    """
    Ventaneo deslizante sobre la señal → features agregados (mean, std, coefv).
    Retorna None si la señal está vacía.
    """
    win_samples = int(WINDOW_SEC * fs)
    hop_samples = int(HOP_SEC   * fs)

    if len(signal) == 0:
        return None

    # Si la señal es más corta que una ventana → una sola ventana
    if len(signal) < win_samples:
        frames = [signal]
    else:
        starts = range(0, len(signal) - win_samples + 1, hop_samples)
        frames = [signal[s:s + win_samples] for s in starts]

    scalar_acc = {k: [] for k in SCALAR_KEYS}
    vector_acc = {k: [] for k in VECTOR_KEYS}

    for frame in frames:
        fw = spectral_features_window(frame, fs)
        for k in SCALAR_KEYS:
            scalar_acc[k].extend(fw[k].tolist())
        for k in VECTOR_KEYS:
            vector_acc[k].append(fw[k])

    row = {}

    # Agrega escalares
    for k in SCALAR_KEYS:
        m, s, cv = aggregate(np.array(scalar_acc[k]))
        row[f"{k}_mean"]  = m
        row[f"{k}_std"]   = s
        row[f"{k}_coefv"] = cv

    # Agrega MFCC (14 coefs cada uno)
    for k in VECTOR_KEYS:
        mat = np.vstack(vector_acc[k])   # (total_frames, N_MFCC)
        for i in range(N_MFCC):
            m, s, cv = aggregate(mat[:, i])
            row[f"{k}_mean_{i+1}"]  = m
            row[f"{k}_std_{i+1}"]   = s
            row[f"{k}_coefv_{i+1}"] = cv

    return row


# ─── Pipeline principal ───────────────────────────────────────────────────────

def parse_patient_id(filename: str) -> str | None:
    m = re.match(r"(H\d+)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def run_extraction(folder: str, output: str):
    folder_path = Path(folder)
    wav_files   = sorted(folder_path.glob("*.wav"))

    if not wav_files:
        log.error("No se encontraron archivos .wav en: %s", folder)
        return

    log.info("Encontrados %d archivos .wav en %s", len(wav_files), folder)
    log.info("Parámetros: target_sr=%d Hz | highpass=%d Hz orden %d | "
             "ventana=%.1fs solapamiento=%.0f%%",
             TARGET_SR, HP_CUTOFF, HP_ORDER, WINDOW_SEC, HOP_SEC * 100)

    rows, skipped = [], []
    resampled_count = 0

    for fpath in tqdm(wav_files, desc="Extrayendo features", unit="file"):
        fname      = fpath.name
        patient_id = parse_patient_id(fname)

        if patient_id is None:
            skipped.append((fname, "No se pudo extraer ID de paciente"))
            continue

        diagnosis = DIAGNOSIS.get(patient_id)
        if diagnosis is None:
            skipped.append((fname, f"ID '{patient_id}' no en diccionario"))
            continue

        try:
            signal, fs, info = load_and_preprocess(str(fpath))
        except Exception as e:
            skipped.append((fname, f"Error al cargar: {e}"))
            continue

        if info["resampled"]:
            resampled_count += 1
            log.debug("  %s resampleado %d→%d Hz", fname, info["sr_original"], TARGET_SR)

        feats = extract_file_features(signal, fs)
        if feats is None:
            skipped.append((fname, "Señal vacía tras preprocesamiento"))
            continue

        channel = re.search(r"_(L\d+|R\d+)", fname, re.IGNORECASE)
        row = {
            "filename":    fpath.stem,
            "patient_id":  patient_id,
            "channel":     channel.group(1).upper() if channel else "?",
            "diagnosis":   diagnosis,
            "sr_original": info["sr_original"],
            "resampled":   info["resampled"],
            "duration_s":  info["duration_s"],
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        log.error("No se extrajeron features de ningún archivo.")
        return

    df = pd.DataFrame(rows)

    # Guardar
    out_base  = Path(output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path  = out_base.with_suffix(".csv")
    xlsx_path = out_base.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    # Reporte final
    log.info("─" * 55)
    log.info("✅  Features extraídos: %d filas × %d columnas", len(df), len(df.columns))
    log.info("   CSV  → %s", csv_path)
    log.info("   XLSX → %s", xlsx_path)

    if resampled_count:
        log.info("   Archivos resampleados a %d Hz: %d", TARGET_SR, resampled_count)

    # Distribución de clases
    log.info("\nDistribución de clases:")
    for label, count in df["diagnosis"].value_counts().items():
        log.info("   %-8s: %d archivos", label, count)

    if skipped:
        log.warning("\n⚠️  Archivos omitidos (%d):", len(skipped))
        for fname, reason in skipped:
            log.warning("   %s — %s", fname, reason)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracción de features espectrales — RespiratoryDatabase@TR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder",  required=True,
                        help="Carpeta con los archivos .wav de TR")
    parser.add_argument("--output",  default="outputs/features_TR",
                        help="Prefijo de salida (se generan .csv y .xlsx)")
    args = parser.parse_args()
    run_extraction(args.folder, args.output)
