"""
extract_features_TR.py
======================
Extracción de features espectrales de la base de datos RespiratoryDatabase@TR.

Esquema:
  - Una fila por archivo de audio (H002_L1, H002_R1, etc.)
  - Ventaneo deslizante sobre el audio crudo
  - Features por ventana → agregación (mean, std, coef. de variación)
  - MFCCs (14 coeficientes) + sus deltas y delta-deltas
  - Salida: CSV y Excel listos para ML

Uso:
  python extract_features_TR.py --folder /ruta/a/audios --output features_TR.xlsx

Dependencias:
  pip install numpy scipy librosa pandas openpyxl tqdm
"""

import os
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DICCIONARIO DE DIAGNÓSTICOS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# PARÁMETROS DE VENTANEO
# ─────────────────────────────────────────────
WINDOW_SEC   = 1.0   # duración de cada ventana en segundos
HOP_SEC      = 0.5   # paso entre ventanas (50 % solapamiento)
N_MFCC       = 14    # número de coeficientes MFCC (igual que tu código MATLAB)
N_FFT        = 512   # tamaño FFT (adecuado para 4 kHz)
HOP_FFT      = 128   # hop interno para features espectrales de librosa


# ─────────────────────────────────────────────
# FUNCIONES DE FEATURES (equivalentes a MATLAB)
# ─────────────────────────────────────────────

def aggregate(values: np.ndarray) -> tuple:
    """Devuelve (mean, std, coef_variacion) de un vector 1-D."""
    m  = float(np.mean(values))
    s  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = s / m if m != 0 else np.nan
    return m, s, cv


def spectral_features_window(frame: np.ndarray, fs: int) -> dict:
    """
    Extrae todos los features espectrales de una ventana (frame).
    Equivale al bucle interno de tu código MATLAB.
    """
    feats = {}

    # ── Espectro de magnitud (base para features manuales) ──────────────────
    S_mag = np.abs(librosa.stft(frame, n_fft=N_FFT, hop_length=HOP_FFT))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=N_FFT)

    # Power spectrum normalizado por frame para cálculos manuales
    power = S_mag ** 2
    power_sum = power.sum(axis=0, keepdims=True) + 1e-12
    p_norm = power / power_sum  # distribución de probabilidad por frame

    # ── Spectral Centroid ───────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT)[0]
    feats["SpectralCentroid"] = centroid

    # ── Spectral Bandwidth (≈ Spread) ───────────────────────────────────────
    spread = librosa.feature.spectral_bandwidth(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT)[0]
    feats["SpectralSpread"] = spread

    # ── Spectral Rolloff ────────────────────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT, roll_percent=0.85)[0]
    feats["SpectralRolloffPoint"] = rolloff

    # ── Spectral Flatness ───────────────────────────────────────────────────
    flatness = librosa.feature.spectral_flatness(
        y=frame, n_fft=N_FFT, hop_length=HOP_FFT)[0]
    feats["SpectralFlatness"] = flatness

    # ── Spectral Flux ───────────────────────────────────────────────────────
    # Diferencia L2 entre frames consecutivos del espectrograma
    flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
    if len(flux) == 0:
        flux = np.array([0.0])
    feats["SpectralFlux"] = flux

    # ── Spectral Skewness ───────────────────────────────────────────────────
    mu1 = (freqs[:, None] * p_norm).sum(axis=0)
    mu2 = (((freqs[:, None] - mu1) ** 2) * p_norm).sum(axis=0)
    mu3 = (((freqs[:, None] - mu1) ** 3) * p_norm).sum(axis=0)
    sigma = np.sqrt(mu2) + 1e-12
    skewness = mu3 / (sigma ** 3)
    feats["SpectralSkewness"] = skewness

    # ── Spectral Kurtosis ───────────────────────────────────────────────────
    mu4 = (((freqs[:, None] - mu1) ** 4) * p_norm).sum(axis=0)
    kurtosis = mu4 / (sigma ** 4)
    feats["SpectralKurtosis"] = kurtosis

    # ── Spectral Entropy ────────────────────────────────────────────────────
    p_safe = np.where(p_norm > 1e-12, p_norm, 1e-12)
    entropy = -np.sum(p_safe * np.log2(p_safe), axis=0)
    feats["SpectralEntropy"] = entropy

    # ── Spectral Crest ──────────────────────────────────────────────────────
    crest = S_mag.max(axis=0) / (S_mag.mean(axis=0) + 1e-12)
    feats["SpectralCrest"] = crest

    # ── Spectral Slope ──────────────────────────────────────────────────────
    K = len(freqs)
    f_mean = freqs.mean()
    num = ((freqs - f_mean)[:, None] * S_mag).sum(axis=0)
    den = ((freqs - f_mean) ** 2).sum() + 1e-12
    slope = num / den
    feats["SpectralSlope"] = slope

    # ── Spectral Decrease ───────────────────────────────────────────────────
    k_idx = np.arange(1, K)
    num_dec = ((S_mag[1:] - S_mag[[0]]) / (k_idx[:, None] + 1e-12)).sum(axis=0)
    den_dec = S_mag[1:].sum(axis=0) + 1e-12
    decrease = num_dec / den_dec
    feats["SpectralDecrease"] = decrease

    # ── MFCCs + delta + delta-delta ─────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=frame, sr=fs, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_FFT)          # shape: (N_MFCC, T)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Transponemos para que cada fila sea un frame: (T, N_MFCC)
    feats["MFCC"]            = mfcc.T
    feats["MFCC_delta"]      = delta1.T
    feats["MFCC_delta_delta"] = delta2.T

    return feats


def extract_file_features(signal: np.ndarray, fs: int) -> dict | None:
    """
    Aplica ventaneo deslizante sobre la señal completa.
    Devuelve un dict con los features agregados (mean, std, coefv)
    o None si la señal es demasiado corta.
    """
    win_samples = int(WINDOW_SEC * fs)
    hop_samples = int(HOP_SEC   * fs)

    if len(signal) < win_samples:
        # Señal más corta que una ventana: se usa completa como única ventana
        frames = [signal]
    else:
        starts = range(0, len(signal) - win_samples + 1, hop_samples)
        frames = [signal[s:s + win_samples] for s in starts]

    if not frames:
        return None

    # Acumuladores por feature escalar
    scalar_keys = [
        "SpectralCentroid", "SpectralSpread", "SpectralRolloffPoint",
        "SpectralFlatness", "SpectralFlux", "SpectralSkewness",
        "SpectralKurtosis", "SpectralEntropy", "SpectralCrest",
        "SpectralSlope", "SpectralDecrease",
    ]
    # Acumuladores por feature vectorial (MFCC × 14)
    vector_keys = ["MFCC", "MFCC_delta", "MFCC_delta_delta"]

    scalar_acc = {k: [] for k in scalar_keys}
    vector_acc = {k: [] for k in vector_keys}

    for frame in frames:
        fw = spectral_features_window(frame, fs)
        for k in scalar_keys:
            scalar_acc[k].extend(fw[k].tolist())
        for k in vector_keys:
            vector_acc[k].append(fw[k])  # (T_frame, 14)

    row = {}

    # ── Agregación de features escalares ────────────────────────────────────
    for k in scalar_keys:
        vals = np.array(scalar_acc[k])
        m, s, cv = aggregate(vals)
        row[f"{k}_mean"] = m
        row[f"{k}_std"]  = s
        row[f"{k}_coefv"] = cv

    # ── Agregación de MFCCs (14 coefs cada uno) ──────────────────────────────
    for k in vector_keys:
        mat = np.vstack(vector_acc[k])   # (total_frames, 14)
        for i in range(N_MFCC):
            vals = mat[:, i]
            m, s, cv = aggregate(vals)
            row[f"{k}_mean_{i+1}"]  = m
            row[f"{k}_std_{i+1}"]   = s
            row[f"{k}_coefv_{i+1}"] = cv

    return row


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def parse_patient_id(filename: str) -> str | None:
    """Extrae el ID de paciente del nombre de archivo (ej: H002_L1 → H002)."""
    m = re.match(r"(H\d+)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def run_extraction(folder: str, output: str):
    wav_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    if not wav_files:
        print(f"[!] No se encontraron archivos .wav en: {folder}")
        return

    rows = []
    skipped = []

    for fname in tqdm(wav_files, desc="Extrayendo features"):
        patient_id = parse_patient_id(fname)
        if patient_id is None:
            skipped.append((fname, "No se pudo extraer ID de paciente"))
            continue

        diagnosis = DIAGNOSIS.get(patient_id)
        if diagnosis is None:
            skipped.append((fname, f"ID '{patient_id}' no encontrado en diccionario"))
            continue

        filepath = os.path.join(folder, fname)
        try:
            signal, fs = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            skipped.append((fname, f"Error al leer audio: {e}"))
            continue

        feats = extract_file_features(signal, fs)
        if feats is None:
            skipped.append((fname, "Señal demasiado corta"))
            continue

        # Metadatos
        channel = re.search(r"_(L\d+|R\d+)", fname, re.IGNORECASE)
        row = {
            "filename":   os.path.splitext(fname)[0],
            "patient_id": patient_id,
            "channel":    channel.group(1).upper() if channel else "?",
            "diagnosis":  diagnosis,
        }
        row.update(feats)
        rows.append(row)

    if not rows:
        print("[!] No se extrajeron features de ningún archivo.")
        return

    df = pd.DataFrame(rows)

    # ── Guardar ──────────────────────────────────────────────────────────────
    base = os.path.splitext(output)[0]
    csv_path   = base + ".csv"
    xlsx_path  = base + ".xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    print(f"\n✅  Features extraídos: {len(df)} filas × {len(df.columns)} columnas")
    print(f"   CSV  → {csv_path}")
    print(f"   XLSX → {xlsx_path}")

    if skipped:
        print(f"\n⚠️  Archivos omitidos ({len(skipped)}):")
        for fname, reason in skipped:
            print(f"   {fname}: {reason}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracción de features espectrales — RespiratoryDatabase@TR"
    )
    parser.add_argument(
        "--folder", required=True,
        help="Carpeta con los archivos .wav de TR"
    )
    parser.add_argument(
        "--output", default="features_TR.xlsx",
        help="Archivo de salida (.csv y .xlsx se generan automáticamente)"
    )
    args = parser.parse_args()
    run_extraction(args.folder, args.output)
