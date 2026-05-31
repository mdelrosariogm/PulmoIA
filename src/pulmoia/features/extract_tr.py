"""
extract_features_TR.py
======================
Extracción de features espectrales — RespiratoryDatabase@TR

Pipeline por ventana (una fila por ventana, igual que HF):
  1. Carga el audio (cualquier sr)
  2. Convierte a mono
  3. Resamplea a TARGET_SR (4 kHz) si es necesario
  4. Aplica filtro pasa-altos Butterworth orden 10 a 80 Hz
  5. Ventaneo deslizante → features espectrales por ventana
  6. Una fila por ventana con etiqueta diagnosis (COPD0-COPD4)
  7. Guarda CSV + Excel

Estructura de salida (igual que features_HF.csv):
  filename | source | patient_id | channel | t_start_s | t_end_s |
  diagnosis | sr_original | resampled | features...

Uso:
  python extract_features_TR.py --folder data/RDB --output outputs/features_TR

Dependencias:
  pip install numpy scipy librosa pandas openpyxl tqdm
"""

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
TARGET_SR  = 4_000
HP_CUTOFF  = 80
HP_ORDER   = 10
WINDOW_SEC = 1.0     # igual que HF
HOP_SEC    = 0.5     # igual que HF
N_MFCC     = 14
N_FFT      = 512
HOP_FFT    = 128

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
def build_highpass(cutoff=HP_CUTOFF, order=HP_ORDER, fs=TARGET_SR):
    return butter(order, cutoff, btype="high", fs=fs, output="sos")

_HP_SOS = build_highpass()

def apply_highpass(signal: np.ndarray) -> np.ndarray:
    return sosfilt(_HP_SOS, signal)

# ─── Carga y preprocesamiento ─────────────────────────────────────────────────
def load_and_preprocess(filepath: str) -> tuple:
    signal, sr_orig = librosa.load(filepath, sr=None, mono=True)
    resampled = False
    if sr_orig != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr_orig, target_sr=TARGET_SR)
        resampled = True
    signal = apply_highpass(signal)
    return signal, TARGET_SR, sr_orig, resampled

# ─── Features espectrales ─────────────────────────────────────────────────────
def aggregate(values: np.ndarray) -> tuple:
    m  = float(np.mean(values))
    s  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = s / m if m != 0 else np.nan
    return m, s, cv


def spectral_features_window(frame: np.ndarray, fs: int) -> dict:
    feats = {}
    S_mag  = np.abs(librosa.stft(frame, n_fft=N_FFT, hop_length=HOP_FFT))
    freqs  = librosa.fft_frequencies(sr=fs, n_fft=N_FFT)
    power  = S_mag ** 2
    p_norm = power / (power.sum(axis=0, keepdims=True) + 1e-12)

    feats["SpectralCentroid"] = librosa.feature.spectral_centroid(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT)[0]
    feats["SpectralSpread"] = librosa.feature.spectral_bandwidth(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT)[0]
    feats["SpectralRolloffPoint"] = librosa.feature.spectral_rolloff(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT, roll_percent=0.85)[0]
    feats["SpectralFlatness"] = librosa.feature.spectral_flatness(
        y=frame, n_fft=N_FFT, hop_length=HOP_FFT)[0]

    flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
    feats["SpectralFlux"] = flux if len(flux) > 0 else np.array([0.0])

    mu1   = (freqs[:, None] * p_norm).sum(axis=0)
    mu2   = (((freqs[:, None] - mu1) ** 2) * p_norm).sum(axis=0)
    mu3   = (((freqs[:, None] - mu1) ** 3) * p_norm).sum(axis=0)
    sigma = np.sqrt(mu2) + 1e-12
    feats["SpectralSkewness"] = mu3 / sigma ** 3

    mu4 = (((freqs[:, None] - mu1) ** 4) * p_norm).sum(axis=0)
    feats["SpectralKurtosis"] = mu4 / sigma ** 4

    p_safe = np.where(p_norm > 1e-12, p_norm, 1e-12)
    feats["SpectralEntropy"] = -np.sum(p_safe * np.log2(p_safe), axis=0)

    feats["SpectralCrest"] = S_mag.max(axis=0) / (S_mag.mean(axis=0) + 1e-12)

    f_mean = freqs.mean()
    num_sl = ((freqs - f_mean)[:, None] * S_mag).sum(axis=0)
    den_sl = ((freqs - f_mean) ** 2).sum() + 1e-12
    feats["SpectralSlope"] = num_sl / den_sl

    K      = len(freqs)
    k_idx  = np.arange(1, K)
    num_dc = ((S_mag[1:] - S_mag[[0]]) / (k_idx[:, None] + 1e-12)).sum(axis=0)
    den_dc = S_mag[1:].sum(axis=0) + 1e-12
    feats["SpectralDecrease"] = num_dc / den_dc

    mfcc   = librosa.feature.mfcc(y=frame, sr=fs, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_FFT)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats["MFCC"]             = mfcc.T
    feats["MFCC_delta"]       = delta1.T
    feats["MFCC_delta_delta"] = delta2.T
    return feats


SCALAR_KEYS = [
    "SpectralCentroid", "SpectralSpread", "SpectralRolloffPoint",
    "SpectralFlatness", "SpectralFlux", "SpectralSkewness",
    "SpectralKurtosis", "SpectralEntropy", "SpectralCrest",
    "SpectralSlope", "SpectralDecrease",
]
VECTOR_KEYS = ["MFCC", "MFCC_delta", "MFCC_delta_delta"]


def extract_window_features(frame: np.ndarray, fs: int) -> dict:
    """Extrae y agrega features de una sola ventana."""
    fw  = spectral_features_window(frame, fs)
    row = {}
    for k in SCALAR_KEYS:
        m, s, cv = aggregate(fw[k])
        row[f"{k}_mean"]  = m
        row[f"{k}_std"]   = s
        row[f"{k}_coefv"] = cv
    for k in VECTOR_KEYS:
        mat = fw[k]
        for i in range(N_MFCC):
            m, s, cv = aggregate(mat[:, i])
            row[f"{k}_mean_{i+1}"]  = m
            row[f"{k}_std_{i+1}"]   = s
            row[f"{k}_coefv_{i+1}"] = cv
    return row


def extract_window_rows(signal: np.ndarray, fs: int) -> list[dict]:
    """Ventanea una señal ya preprocesada y devuelve una fila de features por ventana.

    Reutilizable por el servicio de inferencia (mismo ventaneo que la extracción batch).
    """
    win_samples = int(WINDOW_SEC * fs)
    hop_samples = int(HOP_SEC * fs)
    if len(signal) < win_samples:
        starts = [0]
    else:
        starts = range(0, len(signal) - win_samples + 1, hop_samples)
    rows = []
    for s_idx in starts:
        frame = signal[s_idx:s_idx + win_samples]
        if len(frame) < win_samples:
            frame = np.pad(frame, (0, win_samples - len(frame)))
        try:
            rows.append(extract_window_features(frame, fs))
        except Exception:
            continue
    return rows


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

    win_samples = int(WINDOW_SEC * TARGET_SR)
    hop_samples = int(HOP_SEC   * TARGET_SR)

    rows    = []
    skipped = []
    resampled_count = 0

    for fpath in tqdm(wav_files, desc="Extrayendo features", unit="file"):
        fname      = fpath.stem
        patient_id = parse_patient_id(fname)

        if patient_id is None:
            skipped.append((fname, "No se pudo extraer ID de paciente"))
            continue

        diagnosis = DIAGNOSIS.get(patient_id)
        if diagnosis is None:
            skipped.append((fname, f"ID '{patient_id}' no en diccionario"))
            continue

        try:
            signal, fs, sr_orig, resampled = load_and_preprocess(str(fpath))
        except Exception as e:
            skipped.append((fname, f"Error al cargar: {e}"))
            continue

        if resampled:
            resampled_count += 1

        if len(signal) < win_samples:
            # Señal más corta que una ventana → una sola ventana
            starts = [0]
        else:
            starts = range(0, len(signal) - win_samples + 1, hop_samples)

        channel = re.search(r"_(L\d+|R\d+)", fname, re.IGNORECASE)
        ch      = channel.group(1).upper() if channel else "?"

        for s_idx in starts:
            t_start = s_idx / fs
            t_end   = min((s_idx + win_samples) / fs, len(signal) / fs)
            frame   = signal[s_idx:s_idx + win_samples]
            if len(frame) < win_samples:
                frame = np.pad(frame, (0, win_samples - len(frame)))

            try:
                feat_row = extract_window_features(frame, fs)
            except Exception:
                continue

            row = {
                "filename":    fname,
                "source":      "TR",
                "patient_id":  patient_id,
                "channel":     ch,
                "t_start_s":   round(t_start, 3),
                "t_end_s":     round(t_end,   3),
                "diagnosis":   diagnosis,
                "sr_original": sr_orig,
                "resampled":   resampled,
            }
            row.update(feat_row)
            rows.append(row)

    if not rows:
        log.error("No se extrajeron features de ningún archivo.")
        return

    df = pd.DataFrame(rows)

    out_base  = Path(output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path  = out_base.with_suffix(".csv")
    xlsx_path = out_base.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    log.info("─" * 55)
    log.info("✅  Ventanas extraídas: %d filas × %d columnas",
             len(df), len(df.columns))
    log.info("   CSV  → %s", csv_path)
    log.info("   XLSX → %s", xlsx_path)
    if resampled_count:
        log.info("   Archivos resampleados: %d", resampled_count)

    log.info("\nDistribución de clases (ventanas):")
    for label, count in df["diagnosis"].value_counts().items():
        pct = count / len(df) * 100
        log.info("   %-8s: %6d ventanas (%.1f%%)", label, count, pct)

    if skipped:
        log.warning("\n⚠️  Archivos omitidos (%d):", len(skipped))
        for fname, reason in skipped:
            log.warning("   %s — %s", fname, reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracción de features espectrales — RespiratoryDatabase@TR (por ventana)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder",  required=True,
                        help="Carpeta con los archivos .wav de TR")
    parser.add_argument("--output",  default="outputs/features_TR",
                        help="Prefijo de salida (.csv y .xlsx)")
    args = parser.parse_args()
    run_extraction(args.folder, args.output)
