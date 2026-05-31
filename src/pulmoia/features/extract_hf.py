"""
extract_features_HF.py
======================
Extracción de features espectrales — HF_Lung_V1 (solo archivos steth_)

Pipeline por ventana:
  1. Carga el audio steth_ (15s, 4kHz, 16bit)
  2. Convierte a mono
  3. Resamplea a TARGET_SR (4 kHz) si es necesario
  4. Aplica filtro pasa-altos Butterworth orden 10 a 80 Hz
  5. Parsea el _label.txt → timestamps de Wheeze, Crackle, Stridor, Rhonchus
     (ignora I/E para no condicionar al modelo por fases respiratorias)
  6. Ventaneo deslizante → features espectrales por ventana
  7. Etiqueta cada ventana con solapamiento temporal (multilabel binario)
  8. Una fila por ventana → CSV y Excel

Etiquetas por ventana (multilabel):
  has_wheeze | has_crackle | has_stridor | has_rhonchus
  Una ventana puede tener múltiples etiquetas en 1.

Uso:
  python extract_features_HF.py --folder data/LUNGS --output outputs/features_HF

Dependencias:
  pip install numpy scipy librosa pandas openpyxl tqdm
"""

import argparse
import logging
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
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
TARGET_SR = 4_000  # Hz
HP_CUTOFF = 80  # Hz
HP_ORDER = 10
WINDOW_SEC = 1.0  # duración de ventana en segundos
HOP_SEC = 0.5  # solapamiento 50%
N_MFCC = 14
N_FFT = 512
HOP_FFT = 128

# Eventos acústicos que nos interesan (ignoramos I y E)
ADVENTITIOUS_EVENTS = {"wheeze", "crackle", "stridor", "rhonchus"}

# Mapeo de variantes de nombres encontradas en los archivos de HF_Lung_V1
# hacia los nombres canónicos usados en este proyecto
EVENT_NAME_MAP = {
    # Wheeze
    "wheeze": "wheeze",
    "wheezes": "wheeze",
    "w": "wheeze",
    # Crackle / DAS
    "d": "crackle",
    "crackle": "crackle",
    "crackles": "crackle",
    # Stridor
    "stridor": "stridor",
    "s": "stridor",
    # Rhonchus — múltiples variantes encontradas en HF
    "rhonchus": "rhonchus",
    "rhonchi": "rhonchus",  # ← variante plural latina
    "r": "rhonchus",
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


# ─── Parser de etiquetas ──────────────────────────────────────────────────────
def parse_timestamp(ts: str) -> float:
    """
    Convierte 'HH:MM:SS.mmm' o 'MM:SS.mmm' a segundos float.
    Ejemplos: '00:00:01.120' → 1.12, '00:01:23.456' → 83.456
    """
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])
    except Exception:
        return -1.0


def parse_label_file(label_path: str) -> list[dict]:
    """
    Parsea el archivo _label.txt y devuelve lista de eventos acústicos
    adventiciosos (ignora I y E).

    Formato de línea: EventType HH:MM:SS.mmm HH:MM:SS.mmm
    Ejemplo: Wheeze 00:00:03.144 00:00:03.682

    Returns:
        list of dicts: [{"event": "wheeze", "start": 3.144, "end": 3.682}, ...]
    """
    events = []
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                event_type = parts[0].lower()
                # Ignorar fases respiratorias
                if event_type in {"i", "e"}:
                    continue
                # Normalizar nombre del evento (maneja variantes como Rhonchi)
                event_type = EVENT_NAME_MAP.get(event_type)
                if event_type is None:
                    continue
                t_start = parse_timestamp(parts[1])
                t_end = parse_timestamp(parts[2])
                if t_start < 0 or t_end < 0:
                    continue
                events.append(
                    {
                        "event": event_type,
                        "start": t_start,
                        "end": t_end,
                    }
                )
    except Exception as e:
        log.warning("Error parseando %s: %s", label_path, e)
    return events


def label_window(t_start: float, t_end: float, events: list[dict]) -> dict:
    """
    Determina si una ventana [t_start, t_end] contiene cada evento acústico.
    Criterio: solapamiento temporal (overlap > 0).
    """
    labels = {f"has_{ev}": 0 for ev in ADVENTITIOUS_EVENTS}
    for ev in events:
        # Solapamiento: el evento empieza antes de que acabe la ventana
        # y termina después de que empiece la ventana
        if ev["start"] < t_end and ev["end"] > t_start:
            labels[f"has_{ev['event']}"] = 1
    return labels


# ─── Features espectrales por ventana ────────────────────────────────────────
def aggregate(values: np.ndarray) -> tuple:
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = s / m if m != 0 else np.nan
    return m, s, cv


def spectral_features_window(frame: np.ndarray, fs: int) -> dict:
    feats = {}

    S_mag = np.abs(librosa.stft(frame, n_fft=N_FFT, hop_length=HOP_FFT))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=N_FFT)
    power = S_mag**2
    p_norm = power / (power.sum(axis=0, keepdims=True) + 1e-12)

    feats["SpectralCentroid"] = librosa.feature.spectral_centroid(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT
    )[0]
    feats["SpectralSpread"] = librosa.feature.spectral_bandwidth(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT
    )[0]
    feats["SpectralRolloffPoint"] = librosa.feature.spectral_rolloff(
        y=frame, sr=fs, n_fft=N_FFT, hop_length=HOP_FFT, roll_percent=0.85
    )[0]
    feats["SpectralFlatness"] = librosa.feature.spectral_flatness(
        y=frame, n_fft=N_FFT, hop_length=HOP_FFT
    )[0]

    flux = np.sqrt(np.sum(np.diff(S_mag, axis=1) ** 2, axis=0))
    feats["SpectralFlux"] = flux if len(flux) > 0 else np.array([0.0])

    mu1 = (freqs[:, None] * p_norm).sum(axis=0)
    mu2 = (((freqs[:, None] - mu1) ** 2) * p_norm).sum(axis=0)
    mu3 = (((freqs[:, None] - mu1) ** 3) * p_norm).sum(axis=0)
    sigma = np.sqrt(mu2) + 1e-12
    feats["SpectralSkewness"] = mu3 / sigma**3

    mu4 = (((freqs[:, None] - mu1) ** 4) * p_norm).sum(axis=0)
    feats["SpectralKurtosis"] = mu4 / sigma**4

    p_safe = np.where(p_norm > 1e-12, p_norm, 1e-12)
    feats["SpectralEntropy"] = -np.sum(p_safe * np.log2(p_safe), axis=0)

    feats["SpectralCrest"] = S_mag.max(axis=0) / (S_mag.mean(axis=0) + 1e-12)

    K = len(freqs)
    f_mean = freqs.mean()
    num_sl = ((freqs - f_mean)[:, None] * S_mag).sum(axis=0)
    den_sl = ((freqs - f_mean) ** 2).sum() + 1e-12
    feats["SpectralSlope"] = num_sl / den_sl

    k_idx = np.arange(1, K)
    num_dc = ((S_mag[1:] - S_mag[[0]]) / (k_idx[:, None] + 1e-12)).sum(axis=0)
    den_dc = S_mag[1:].sum(axis=0) + 1e-12
    feats["SpectralDecrease"] = num_dc / den_dc

    mfcc = librosa.feature.mfcc(y=frame, sr=fs, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_FFT)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats["MFCC"] = mfcc.T
    feats["MFCC_delta"] = delta1.T
    feats["MFCC_delta_delta"] = delta2.T

    return feats


SCALAR_KEYS = [
    "SpectralCentroid",
    "SpectralSpread",
    "SpectralRolloffPoint",
    "SpectralFlatness",
    "SpectralFlux",
    "SpectralSkewness",
    "SpectralKurtosis",
    "SpectralEntropy",
    "SpectralCrest",
    "SpectralSlope",
    "SpectralDecrease",
]
VECTOR_KEYS = ["MFCC", "MFCC_delta", "MFCC_delta_delta"]


def extract_window_features(frame: np.ndarray, fs: int) -> dict:
    """Extrae y agrega features de una sola ventana."""
    fw = spectral_features_window(frame, fs)
    row = {}

    for k in SCALAR_KEYS:
        m, s, cv = aggregate(fw[k])
        row[f"{k}_mean"] = m
        row[f"{k}_std"] = s
        row[f"{k}_coefv"] = cv

    for k in VECTOR_KEYS:
        mat = fw[k]  # (T_frame, N_MFCC)
        for i in range(N_MFCC):
            m, s, cv = aggregate(mat[:, i])
            row[f"{k}_mean_{i+1}"] = m
            row[f"{k}_std_{i+1}"] = s
            row[f"{k}_coefv_{i+1}"] = cv

    return row


# ─── Pipeline principal ───────────────────────────────────────────────────────
def run_extraction(folder: str, output: str):
    folder_path = Path(folder)

    # Solo archivos steth_ .wav
    wav_files = sorted([f for f in folder_path.glob("steth_*.wav")])

    if not wav_files:
        log.error("No se encontraron archivos steth_*.wav en: %s", folder)
        return

    log.info("Encontrados %d archivos steth_*.wav", len(wav_files))
    log.info(
        "Parámetros: target_sr=%d Hz | highpass=%d Hz orden %d | "
        "ventana=%.1fs solapamiento=%.0f%%",
        TARGET_SR,
        HP_CUTOFF,
        HP_ORDER,
        WINDOW_SEC,
        HOP_SEC * 100,
    )

    win_samples = int(WINDOW_SEC * TARGET_SR)
    hop_samples = int(HOP_SEC * TARGET_SR)

    rows = []
    skipped = []
    total_windows = 0

    for fpath in tqdm(wav_files, desc="Extrayendo features", unit="file"):
        fname = fpath.stem
        label_path = fpath.parent / f"{fname}_label.txt"

        if not label_path.exists():
            skipped.append((fname, "Sin archivo _label.txt"))
            continue

        # Parsear etiquetas (solo adventicios)
        events = parse_label_file(str(label_path))

        # Cargar y preprocesar audio
        try:
            signal, fs, sr_orig, resampled = load_and_preprocess(str(fpath))
        except Exception as e:
            skipped.append((fname, f"Error al cargar: {e}"))
            continue

        if len(signal) < win_samples:
            skipped.append((fname, "Señal más corta que una ventana"))
            continue

        # Ventaneo deslizante
        starts = range(0, len(signal) - win_samples + 1, hop_samples)

        for s_idx in starts:
            t_start = s_idx / fs
            t_end = (s_idx + win_samples) / fs
            frame = signal[s_idx : s_idx + win_samples]

            # Features espectrales de la ventana
            try:
                feat_row = extract_window_features(frame, fs)
            except Exception:
                continue

            # Etiquetas de la ventana
            win_labels = label_window(t_start, t_end, events)

            row = {
                "filename": fname,
                "source": "steth",
                "t_start_s": round(t_start, 3),
                "t_end_s": round(t_end, 3),
                "sr_original": sr_orig,
                "resampled": resampled,
            }
            row.update(win_labels)
            row.update(feat_row)
            rows.append(row)
            total_windows += 1

    if not rows:
        log.error("No se extrajeron features de ningún archivo.")
        return

    df = pd.DataFrame(rows)

    # Reordenar: metadatos primero, luego etiquetas, luego features
    meta_cols = ["filename", "source", "t_start_s", "t_end_s", "sr_original", "resampled"]
    label_cols = [f"has_{ev}" for ev in ADVENTITIOUS_EVENTS]
    feat_cols = [c for c in df.columns if c not in meta_cols + label_cols]
    df = df[meta_cols + label_cols + feat_cols]

    # Guardar
    out_base = Path(output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    xlsx_path = out_base.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    # Reporte
    log.info("─" * 55)
    log.info("✅  Ventanas extraídas: %d filas × %d columnas", len(df), len(df.columns))
    log.info("   CSV  → %s", csv_path)
    log.info("   XLSX → %s", xlsx_path)
    log.info(
        "   Archivos procesados: %d | Omitidos: %d", len(wav_files) - len(skipped), len(skipped)
    )

    # Distribución de etiquetas
    log.info("\nDistribución de etiquetas (ventanas positivas):")
    for col in label_cols:
        n_pos = int(df[col].sum())
        pct = n_pos / len(df) * 100
        log.info("   %-15s: %6d / %d ventanas (%.1f%%)", col, n_pos, len(df), pct)

    if skipped:
        log.warning("\n⚠️  Archivos omitidos (%d):", len(skipped))
        for fname, reason in skipped:
            log.warning("   %s — %s", fname, reason)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracción de features espectrales — HF_Lung_V1 (steth_)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--folder", required=True, help="Carpeta con archivos steth_*.wav y _label.txt"
    )
    parser.add_argument(
        "--output", default="outputs/features_HF", help="Prefijo de salida (.csv y .xlsx)"
    )
    args = parser.parse_args()
    run_extraction(args.folder, args.output)
