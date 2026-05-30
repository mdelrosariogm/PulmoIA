"""
git_commit.py
=============
Hace commit y push de todos los scripts y el README actualizado.
NO sube los datos (CSV, XLSX, WAV) porque estan en .gitignore.

Corre esto despues de que el pipeline termine:
  python git_commit.py

O con un mensaje personalizado:
  python git_commit.py --message "feat: extraccion ICBHI completada"
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(cmd: list[str], check: bool = True) -> tuple[bool, str]:
    """Corre un comando y devuelve (exito, output)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=check
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.CalledProcessError as e:
        return False, (e.stdout + e.stderr).strip()
    except Exception as e:
        return False, str(e)


def git_commit(message: str):

    log.info("=" * 55)
    log.info("  GIT COMMIT & PUSH — pulmoia")
    log.info("=" * 55)

    # Verificar que estamos en un repo git
    ok, out = run(["git", "rev-parse", "--git-dir"], check=False)
    if not ok:
        log.error("No se encontro repositorio git en el directorio actual.")
        sys.exit(1)

    # Mostrar estado actual
    log.info("Estado actual del repo:")
    ok, out = run(["git", "status", "--short"], check=False)
    if out:
        for line in out.splitlines():
            log.info("  %s", line)
    else:
        log.info("  (sin cambios)")

    # Archivos que se van a commitear (solo scripts y docs, no datos)
    files_to_add = [
        "extract_features_TR.py",
        "extract_features_HF.py",
        "extract_features_ICBHI.py",
        "verify_labels_HF.py",
        "scan_labels_HF.py",
        "scan_labels_ICBHI.py",
        "run_pipeline.py",
        "git_commit.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "data/.gitkeep",
        "outputs/.gitkeep",
        "models/.gitkeep",
        "src/.gitkeep",
        "scripts/.gitkeep",
        "tests/.gitkeep",
    ]

    # Agregar solo los que existen
    added = []
    for f in files_to_add:
        if Path(f).exists():
            ok, out = run(["git", "add", f], check=False)
            if ok:
                added.append(f)
            else:
                log.warning("No se pudo agregar %s: %s", f, out)

    log.info("")
    log.info("Archivos agregados al commit (%d):", len(added))
    for f in added:
        log.info("  + %s", f)

    if not added:
        log.warning("No hay archivos para commitear.")
        sys.exit(0)

    # Verificar si hay algo staged
    ok, out = run(["git", "diff", "--cached", "--name-only"], check=False)
    if not out.strip():
        log.info("Nada nuevo para commitear — todo ya estaba actualizado.")
        sys.exit(0)

    # Commit
    log.info("")
    log.info("Haciendo commit...")
    ok, out = run(["git", "commit", "-m", message], check=False)
    if not ok:
        log.error("Error en commit: %s", out)
        sys.exit(1)
    log.info("Commit exitoso: %s", out.splitlines()[0] if out else "")

    # Push
    log.info("")
    log.info("Haciendo push a origin main...")
    ok, out = run(["git", "push", "origin", "main"], check=False)
    if not ok:
        log.error("Error en push: %s", out)
        log.error("Intenta manualmente: git push origin main")
        sys.exit(1)

    log.info("Push exitoso.")
    log.info("")
    log.info("=" * 55)
    log.info("  Repositorio actualizado en GitHub.")
    log.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Git commit y push de scripts y README",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parser.add_argument(
        "--message", "-m",
        default=f"feat: pipeline extraccion features completado ({timestamp})",
        help="Mensaje del commit"
    )
    args = parser.parse_args()
    git_commit(args.message)
