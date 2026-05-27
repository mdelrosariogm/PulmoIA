"""
run_pipeline.py
===============
Script maestro que corre todo el pipeline de extraccion en secuencia:

  1. Verificacion de etiquetado HF (muestra reporte, no bloquea)
  2. Extraccion de features HF    → outputs/features_HF.csv
  3. Escaneo de labels ICBHI      → verifica formato antes de extraer
  4. Extraccion de features ICBHI → outputs/features_ICBHI.csv
  5. Extraccion de features TR    → outputs/features_TR.csv
  6. Reporte final consolidado

Si alguna extraccion ya existe (el CSV ya fue generado), la omite
automaticamente a menos que uses --force para reprocessar todo.

Uso:
  # Correr todo lo que falta
  python run_pipeline.py

  # Forzar re-extraccion de todo aunque ya existan los CSV
  python run_pipeline.py --force

  # Omitir HF (si ya lo tienes) y solo correr ICBHI y TR
  python run_pipeline.py --skip-hf

  # Solo verificar sin extraer
  python run_pipeline.py --verify-only

Dependencias:
  pip install numpy scipy librosa pandas openpyxl tqdm
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuracion de rutas ───────────────────────────────────────────────────
FOLDERS = {
    "hf":    "data/LUNGS",
    "icbhi": "data/ICBHI",
    "tr":    "data/RDB",
}
OUTPUTS = {
    "hf":    "outputs/features_HF",
    "icbhi": "outputs/features_ICBHI",
    "tr":    "outputs/features_TR",
}
SCRIPTS = {
    "verify_hf":   "verify_labels_HF.py",
    "scan_hf":     "scan_labels_HF.py",
    "extract_hf":  "extract_features_HF.py",
    "scan_icbhi":  "scan_labels_ICBHI.py",
    "extract_icbhi": "extract_features_ICBHI.py",
    "extract_tr":  "extract_features_TR.py",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def separator(title: str = ""):
    line = "=" * 60
    if title:
        log.info(line)
        log.info("  %s", title)
    log.info(line)


def csv_exists(key: str) -> bool:
    return Path(f"{OUTPUTS[key]}.csv").exists()


def folder_exists(key: str) -> bool:
    return Path(FOLDERS[key]).exists()


def run_script(script: str, args: list[str] = []) -> tuple[bool, float]:
    """
    Corre un script Python y devuelve (exito, duracion_segundos).
    Muestra output en tiempo real.
    """
    script_path = Path(script)
    if not script_path.exists():
        log.error("Script no encontrado: %s", script)
        return False, 0.0

    cmd     = [sys.executable, script] + args
    t_start = time.time()

    log.info("Corriendo: %s %s", script, " ".join(args))

    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t_start
        success = result.returncode == 0
        if not success:
            log.error("Script termino con error (codigo %d)", result.returncode)
        return success, elapsed
    except Exception as e:
        elapsed = time.time() - t_start
        log.error("Error al correr %s: %s", script, e)
        return False, elapsed


def format_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"

# ─── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(skip_hf: bool, skip_icbhi: bool, skip_tr: bool,
                 force: bool, verify_only: bool):

    results  = {}   # {etapa: (exito, duracion)}
    t_global = time.time()

    log.info("")
    separator("PIPELINE DE EXTRACCION DE FEATURES — pulmoia")
    log.info("  Inicio : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  Modo   : %s", "VERIFICACION SOLO" if verify_only else "EXTRACCION COMPLETA")
    log.info("  Force  : %s", "SI" if force else "NO (omite si CSV ya existe)")

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 1 — HF_Lung_V1
    # ─────────────────────────────────────────────────────────────────────────
    if not skip_hf:
        separator("PASO 1 — HF_Lung_V1 (steth_)")

        if not folder_exists("hf"):
            log.warning("Carpeta no encontrada: %s — omitiendo HF", FOLDERS["hf"])
            results["hf_extract"] = (False, 0)
        else:
            # 1a. Escaneo de labels
            log.info("[1a] Escaneando labels HF...")
            ok, dur = run_script(SCRIPTS["scan_hf"],
                                 ["--folder", FOLDERS["hf"]])
            results["hf_scan"] = (ok, dur)

            # 1b. Verificacion de etiquetado (archivo automatico)
            log.info("[1b] Verificando etiquetado HF (archivo automatico)...")
            ok, dur = run_script(SCRIPTS["verify_hf"],
                                 ["--folder", FOLDERS["hf"]])
            results["hf_verify"] = (ok, dur)

            if verify_only:
                log.info("Modo verify-only: omitiendo extraccion HF.")
            elif csv_exists("hf") and not force:
                log.info("[OK] CSV ya existe: %s.csv — omitiendo extraccion.",
                         OUTPUTS["hf"])
                log.info("     Usa --force para re-extraer.")
                results["hf_extract"] = (True, 0)
            else:
                # 1c. Extraccion
                log.info("[1c] Extrayendo features HF...")
                ok, dur = run_script(SCRIPTS["extract_hf"],
                                     ["--folder", FOLDERS["hf"],
                                      "--output", OUTPUTS["hf"]])
                results["hf_extract"] = (ok, dur)
                if ok:
                    log.info("HF completado en %s", format_duration(dur))
    else:
        log.info("HF omitido por --skip-hf")
        results["hf_extract"] = (True, 0)  # no es fallo, fue intencional

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 2 — ICBHI 2017
    # ─────────────────────────────────────────────────────────────────────────
    if not skip_icbhi and not verify_only:
        separator("PASO 2 — ICBHI 2017")

        if not folder_exists("icbhi"):
            log.warning("Carpeta no encontrada: %s — omitiendo ICBHI",
                        FOLDERS["icbhi"])
            results["icbhi_extract"] = (False, 0)
        else:
            # 2a. Escaneo
            log.info("[2a] Escaneando labels ICBHI...")
            ok, dur = run_script(SCRIPTS["scan_icbhi"],
                                 ["--folder", FOLDERS["icbhi"]])
            results["icbhi_scan"] = (ok, dur)

            if csv_exists("icbhi") and not force:
                log.info("[OK] CSV ya existe: %s.csv — omitiendo extraccion.",
                         OUTPUTS["icbhi"])
                results["icbhi_extract"] = (True, 0)
            else:
                # 2b. Extraccion
                log.info("[2b] Extrayendo features ICBHI...")
                ok, dur = run_script(SCRIPTS["extract_icbhi"],
                                     ["--folder", FOLDERS["icbhi"],
                                      "--output", OUTPUTS["icbhi"]])
                results["icbhi_extract"] = (ok, dur)
                if ok:
                    log.info("ICBHI completado en %s", format_duration(dur))
    else:
        if verify_only:
            log.info("ICBHI omitido (modo verify-only)")
        else:
            log.info("ICBHI omitido por --skip-icbhi")
            results["icbhi_extract"] = (True, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 3 — RespiratoryDatabase@TR
    # ─────────────────────────────────────────────────────────────────────────
    if not skip_tr and not verify_only:
        separator("PASO 3 — RespiratoryDatabase@TR")

        if not folder_exists("tr"):
            log.warning("Carpeta no encontrada: %s — omitiendo TR", FOLDERS["tr"])
            results["tr_extract"] = (False, 0)
        else:
            if csv_exists("tr") and not force:
                log.info("[OK] CSV ya existe: %s.csv — omitiendo extraccion.",
                         OUTPUTS["tr"])
                results["tr_extract"] = (True, 0)
            else:
                log.info("[3] Extrayendo features TR...")
                ok, dur = run_script(SCRIPTS["extract_tr"],
                                     ["--folder", FOLDERS["tr"],
                                      "--output", OUTPUTS["tr"]])
                results["tr_extract"] = (ok, dur)
                if ok:
                    log.info("TR completado en %s", format_duration(dur))
    else:
        if verify_only:
            log.info("TR omitido (modo verify-only)")
        else:
            log.info("TR omitido por --skip-tr")
            results["tr_extract"] = (True, 0)

    # ─────────────────────────────────────────────────────────────────────────
    # REPORTE FINAL
    # ─────────────────────────────────────────────────────────────────────────
    t_total = time.time() - t_global
    separator("REPORTE FINAL")
    log.info("  Duracion total: %s", format_duration(t_total))
    log.info("")

    # Estado de cada CSV generado
    log.info("  Archivos generados:")
    for key in ["hf", "icbhi", "tr"]:
        csv_path = Path(f"{OUTPUTS[key]}.csv")
        if csv_path.exists():
            size_mb = csv_path.stat().st_size / 1_048_576
            log.info("  [OK] %s  (%.1f MB)", csv_path, size_mb)
        else:
            log.info("  [--] %s  (no generado)", csv_path)

    # Resumen de etapas
    log.info("")
    log.info("  Resumen de etapas:")
    for etapa, (ok, dur) in results.items():
        status = "[OK]" if ok else "[FALLO]"
        durstr = format_duration(dur) if dur > 0 else "omitido"
        log.info("  %s  %-20s %s", status, etapa, durstr)

    # Exit code
    failed = [k for k, (ok, _) in results.items() if not ok]
    if failed:
        log.warning("")
        log.warning("  Etapas con error: %s", ", ".join(failed))
        log.warning("  Revisa los logs arriba para mas detalle.")
        sys.exit(1)
    else:
        log.info("")
        log.info("  Pipeline completado exitosamente.")
        separator()
        sys.exit(0)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline maestro de extraccion de features — pulmoia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--force",       action="store_true",
                        help="Re-extraer aunque el CSV ya exista")
    parser.add_argument("--skip-hf",     action="store_true",
                        help="Omitir extraccion de HF_Lung_V1")
    parser.add_argument("--skip-icbhi",  action="store_true",
                        help="Omitir extraccion de ICBHI")
    parser.add_argument("--skip-tr",     action="store_true",
                        help="Omitir extraccion de TR")
    parser.add_argument("--verify-only", action="store_true",
                        help="Solo verificar labels, no extraer features")
    args = parser.parse_args()

    run_pipeline(
        skip_hf    = args.skip_hf,
        skip_icbhi = args.skip_icbhi,
        skip_tr    = args.skip_tr,
        force      = args.force,
        verify_only= args.verify_only,
    )
