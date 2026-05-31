"""
extract_features_ICBHI.py
=========================
Extracción de features espectrales — ICBHI 2017 Respiratory Sound Database

Pipeline por ventana (una fila por ventana, igual que HF y TR):
  1. Carga el audio (frecuencia variable → resamplea a 4 kHz)
  2. Convierte a mono
  3. Resamplea a TARGET_SR (4 kHz) si es necesario
  4. Aplica filtro pasa-altos Butterworth orden 10 a 80 Hz
  5. Parsea el .txt de anotaciones → ciclos respiratorios con flags
     de crackle y wheeze (columnas 3 y 4)
  6. Ventaneo deslizante → features espectrales por ventana
  7. Etiqueta cada ventana por solapamiento con ciclos anotados
  8. Una fila por ventana → CSV y Excel

Formato del archivo de anotaciones ICBHI:
  t_inicio  t_fin   crackle  wheeze
  0.036     1.207   0        0
  3.550     5.750   1        0      ← crackle presente en este ciclo
  5.750     7.879   1        0

Nota: ICBHI no tiene stridor ni rhonchus en sus anotaciones.
      Solo crackle y wheeze. has_stridor y has_rhonchus = 0 siempre.

Uso:
  python extract_features_ICBHI.py --folder data/ICBHI --output outputs/features_ICBHI

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
TARGET_SR = 4_000  # Hz — igual que HF y TR
HP_CUTOFF = 80  # Hz
HP_ORDER = 10
WINDOW_SEC = 1.0  # igual que HF y TR
HOP_SEC = 0.5  # igual que HF y TR
N_MFCC = 14
N_FFT = 512
HOP_FFT = 128

# Equipos de grabación válidos en ICBHI
VALID_EQUIPMENT = {"AKGC417L", "LittC2SE", "Litt3200", "Meditron"}


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


# ─── Parser de anotaciones ICBHI ─────────────────────────────────────────────
def parse_annotation_file(txt_path: str) -> list[dict]:
    """
    Parsea el archivo .txt de ICBHI.

    Formato: t_inicio  t_fin  crackle  wheeze
    (separado por tabs o espacios)

    Returns:
        list of dicts: [{"start": float, "end": float,
                         "crackle": int, "wheeze": int}, ...]
    """
    cycles = []
    try:
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    t_start = float(parts[0])
                    t_end = float(parts[1])
                    crackle = int(parts[2])
                    wheeze = int(parts[3])
                    cycles.append(
                        {
                            "start": t_start,
                            "end": t_end,
                            "crackle": crackle,
                            "wheeze": wheeze,
                        }
                    )
                except ValueError:
                    continue
    except Exception as e:
        log.warning("Error parseando %s: %s", txt_path, e)
    return cycles


def label_window_icbhi(t_start: float, t_end: float, cycles: list[dict]) -> dict:
    """
    Etiqueta una ventana [t_start, t_end] según solapamiento con ciclos.

    Criterio: si un ciclo que contiene crackle/wheeze se solapa
    con la ventana, la ventana hereda esa etiqueta.

    ICBHI no tiene stridor ni rhonchus → siempre 0.
    """
    has_crackle = 0
    has_wheeze = 0

    for cycle in cycles:
        # Solapamiento temporal
        if cycle["start"] < t_end and cycle["end"] > t_start:
            if cycle["crackle"] == 1:
                has_crackle = 1
            if cycle["wheeze"] == 1:
                has_wheeze = 1
        # Optimización: si ya tenemos ambos no hace falta seguir
        if has_crackle and has_wheeze:
            break

    return {
        "has_wheeze": has_wheeze,
        "has_crackle": has_crackle,
        "has_stridor": 0,  # no disponible en ICBHI
        "has_rhonchus": 0,  # no disponible en ICBHI
    }


# ─── Parser de nombre de archivo ICBHI ───────────────────────────────────────
def parse_icbhi_filename(stem: str) -> dict | None:
    """
    Parsea el nombre de archivo ICBHI:
    {PatientID}_{RecordingIndex}_{Location}_{Mode}_{Equipment}

    Ejemplo: 101_1b1_Al_sc_Meditron
    """
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    try:
        patient_id = parts[0]
        rec_index = parts[1]
        location = parts[2]
        mode = parts[3]
        equipment = "_".join(parts[4:])  # por si el equipo tiene _
        return {
            "patient_id": patient_id,
            "rec_index": rec_index,
            "location": location,
            "mode": mode,
            "equipment": equipment,
        }
    except Exception:
        return None


# ─── Features espectrales ─────────────────────────────────────────────────────
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

    f_mean = freqs.mean()
    num_sl = ((freqs - f_mean)[:, None] * S_mag).sum(axis=0)
    den_sl = ((freqs - f_mean) ** 2).sum() + 1e-12
    feats["SpectralSlope"] = num_sl / den_sl

    K = len(freqs)
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
    fw = spectral_features_window(frame, fs)
    row = {}
    for k in SCALAR_KEYS:
        m, s, cv = aggregate(fw[k])
        row[f"{k}_mean"] = m
        row[f"{k}_std"] = s
        row[f"{k}_coefv"] = cv
    for k in VECTOR_KEYS:
        mat = fw[k]
        for i in range(N_MFCC):
            m, s, cv = aggregate(mat[:, i])
            row[f"{k}_mean_{i+1}"] = m
            row[f"{k}_std_{i+1}"] = s
            row[f"{k}_coefv_{i+1}"] = cv
    return row


# ─── Pipeline principal ───────────────────────────────────────────────────────
def run_extraction(folder: str, output: str):
    folder_path = Path(folder)

    # Buscar todos los .wav con su .txt correspondiente
    # Excluir archivos que no son anotaciones de ciclos respiratorios
    EXCLUDE_TXT = {"filename_differences.txt", "filename_format.txt"}
    wav_files = sorted([f for f in folder_path.glob("*.wav")])

    if not wav_files:
        log.error("No se encontraron archivos .wav en: %s", folder)
        return

    log.info("Encontrados %d archivos .wav en %s", len(wav_files), folder)
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
    resampled_count = 0

    for fpath in tqdm(wav_files, desc="Extrayendo features", unit="file"):
        fname = fpath.stem

        # Verificar que existe el .txt de anotaciones (no es un archivo excluido)
        txt_path = fpath.parent / f"{fname}.txt"
        if not txt_path.exists() or txt_path.name in EXCLUDE_TXT:
            skipped.append((fname, "Sin archivo .txt de anotaciones"))
            continue

        # Parsear nombre de archivo
        meta = parse_icbhi_filename(fname)
        if meta is None:
            skipped.append((fname, "Nombre de archivo no reconocido"))
            continue

        # Parsear anotaciones
        cycles = parse_annotation_file(str(txt_path))

        # Cargar y preprocesar audio
        try:
            signal, fs, sr_orig, resampled = load_and_preprocess(str(fpath))
        except Exception as e:
            skipped.append((fname, f"Error al cargar: {e}"))
            continue

        if resampled:
            resampled_count += 1

        if len(signal) < win_samples:
            skipped.append((fname, f"Señal muy corta: {len(signal)/fs:.1f}s"))
            continue

        # Ventaneo deslizante
        starts = range(0, len(signal) - win_samples + 1, hop_samples)

        for s_idx in starts:
            t_start = s_idx / fs
            t_end = (s_idx + win_samples) / fs
            frame = signal[s_idx : s_idx + win_samples]

            try:
                feat_row = extract_window_features(frame, fs)
            except Exception:
                continue

            # Etiquetas por solapamiento con ciclos anotados
            win_labels = label_window_icbhi(t_start, t_end, cycles)

            row = {
                "filename": fname,
                "source": "ICBHI",
                "patient_id": meta["patient_id"],
                "location": meta["location"],
                "mode": meta["mode"],
                "equipment": meta["equipment"],
                "t_start_s": round(t_start, 3),
                "t_end_s": round(t_end, 3),
                "sr_original": sr_orig,
                "resampled": resampled,
            }
            row.update(win_labels)
            row.update(feat_row)
            rows.append(row)

    if not rows:
        log.error("No se extrajeron features de ningún archivo.")
        return

    df = pd.DataFrame(rows)

    # Reordenar columnas: metadatos → etiquetas → features
    meta_cols = [
        "filename",
        "source",
        "patient_id",
        "location",
        "mode",
        "equipment",
        "t_start_s",
        "t_end_s",
        "sr_original",
        "resampled",
    ]
    label_cols = ["has_wheeze", "has_crackle", "has_stridor", "has_rhonchus"]
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
    if resampled_count:
        log.info("   Archivos resampleados a %d Hz: %d", TARGET_SR, resampled_count)

    # Distribución de etiquetas
    log.info("\nDistribución de etiquetas (ventanas positivas):")
    for col in label_cols:
        n_pos = int(df[col].sum())
        pct = n_pos / len(df) * 100
        log.info("   %-15s: %6d / %d ventanas (%.1f%%)", col, n_pos, len(df), pct)

    # Distribución por equipo
    log.info("\nDistribución por equipo de grabación:")
    for equip, count in df["equipment"].value_counts().items():
        pct = count / len(df) * 100
        log.info("   %-15s: %6d ventanas (%.1f%%)", equip, count, pct)

    if skipped:
        log.warning("\n⚠️  Archivos omitidos (%d):", len(skipped))
        for fname, reason in skipped:
            log.warning("   %s — %s", fname, reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracción de features espectrales — ICBHI 2017 (por ventana)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder", required=True, help="Carpeta con archivos .wav y .txt de ICBHI")
    parser.add_argument(
        "--output", default="outputs/features_ICBHI", help="Prefijo de salida (.csv y .xlsx)"
    )
    args = parser.parse_args()
    run_extraction(args.folder, args.output)
