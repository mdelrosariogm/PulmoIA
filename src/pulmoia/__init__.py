"""pulmoia — Detección de sonidos adventicios pulmonares y clasificación de severidad EPOC."""

import os as _os
from pathlib import Path as _Path

__version__ = "0.1.0"

# Aísla Prefect al proyecto (DB local fresca), evitando choques con una instalación
# global previa. Debe fijarse ANTES de importar `prefect`. Respeta un PREFECT_HOME ya
# definido por el usuario.
_os.environ.setdefault("PREFECT_HOME", str(_Path(__file__).resolve().parents[2] / ".prefect"))
