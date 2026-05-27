"""
scan_labels_HF.py
=================
Lee TODOS los archivos _label.txt de steth_ y reporta:
  1. Todas las clases/eventos únicos encontrados
  2. Frecuencia de cada uno
  3. Ejemplos de líneas para cada clase nueva/inesperada

Uso:
  python scan_labels_HF.py --folder data/LUNGS
"""

import argparse
from pathlib import Path
from collections import Counter

def scan(folder: str):
    folder_path = Path(folder)
    label_files = sorted(folder_path.glob("steth_*_label.txt"))

    if not label_files:
        print(f"[ERROR] No se encontraron archivos steth_*_label.txt en: {folder}")
        return

    print(f"\n{'═'*55}")
    print(f"  Escaneando {len(label_files)} archivos _label.txt...")
    print(f"{'═'*55}\n")

    event_counter  = Counter()   # conteo por nombre de evento (lowercase)
    event_examples = {}          # ejemplo de línea por evento
    files_with_event = Counter() # cuántos archivos tienen cada evento
    parse_errors   = []

    for fpath in label_files:
        events_in_file = set()
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    event = parts[0].lower()
                    event_counter[event] += 1
                    events_in_file.add(event)
                    # Guardar ejemplo si es primera vez
                    if event not in event_examples:
                        event_examples[event] = {
                            "raw":  parts[0],       # nombre original
                            "line": line,
                            "file": fpath.name,
                        }
        except Exception as e:
            parse_errors.append((fpath.name, str(e)))

        for ev in events_in_file:
            files_with_event[ev] += 1

    # ── Reporte ───────────────────────────────────────────────────────────────
    print(f"{'Evento (raw)':<18} {'Evento (lower)':<18} "
          f"{'N líneas':>10} {'N archivos':>12}")
    print("─" * 62)

    # Ordenar por frecuencia descendente
    for event_lower, count in event_counter.most_common():
        raw   = event_examples[event_lower]["raw"]
        n_files = files_with_file = files_with_event[event_lower]
        print(f"  {raw:<16} {event_lower:<18} {count:>10,} {n_files:>12,}")

    print(f"\n{'─'*62}")
    print(f"  Total clases únicas encontradas: {len(event_counter)}")
    print(f"  Total archivos escaneados      : {len(label_files)}")

    # ── Ejemplos de línea por clase ───────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  EJEMPLOS DE LÍNEA POR CLASE:")
    print(f"{'═'*55}")
    for event_lower, _ in event_counter.most_common():
        ex = event_examples[event_lower]
        print(f"\n  [{ex['raw']}]  (en {ex['file']})")
        print(f"    → {ex['line']}")

    # ── Clases NO reconocidas por el parser actual ────────────────────────────
    known = {"i", "e", "wheeze", "wheezes", "w",
             "d", "crackle", "crackles",
             "stridor", "s",
             "rhonchus", "rhonchi", "r"}
    unknown = {ev for ev in event_counter if ev not in known}

    if unknown:
        print(f"\n{'═'*55}")
        print(f"  ⚠️  CLASES NO RECONOCIDAS POR EL PARSER ACTUAL ({len(unknown)}):")
        print(f"{'═'*55}")
        for ev in sorted(unknown):
            ex = event_examples[ev]
            print(f"\n  [{ev}]  N={event_counter[ev]:,}  (en {ex['file']})")
            print(f"    → {ex['line']}")
        print(f"\n  → Agrega estas clases al EVENT_NAME_MAP antes de extraer.")
    else:
        print(f"\n  ✅ Todas las clases son reconocidas por el parser actual.")

    if parse_errors:
        print(f"\n  ⚠️  Errores de lectura ({len(parse_errors)}):")
        for fname, err in parse_errors:
            print(f"    {fname}: {err}")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Escanea todos los _label.txt de steth_ y reporta clases únicas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder", required=True,
                        help="Carpeta con archivos steth_*_label.txt")
    args = parser.parse_args()
    scan(args.folder)
