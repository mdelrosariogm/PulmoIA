"""
scan_labels_ICBHI.py
====================
Lee TODOS los archivos .txt de ICBHI y reporta:
  1. Formato de las anotaciones (columnas)
  2. Distribucion de crackle y wheeze
  3. Rango de duraciones de audio
  4. Equipos de grabacion encontrados
  5. Alertas de formato inesperado

Formato esperado de anotacion ICBHI:
  t_inicio  t_fin  crackle  wheeze
  0.036     1.207  0        0
  3.550     5.750  1        0

Uso:
  python scan_labels_ICBHI.py --folder data/ICBHI
"""

import argparse
from pathlib import Path
from collections import Counter

def parse_icbhi_filename(stem: str) -> dict | None:
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    return {
        "patient_id": parts[0],
        "rec_index":  parts[1],
        "location":   parts[2],
        "mode":       parts[3],
        "equipment":  "_".join(parts[4:]),
    }

def scan(folder: str):
    folder_path = Path(folder)
    # Excluir archivos que no son anotaciones de ciclos respiratorios
    EXCLUDE_TXT = {"filename_differences.txt", "filename_format.txt"}
    txt_files   = sorted([f for f in folder_path.glob("*.txt")
                          if f.name not in EXCLUDE_TXT])
    wav_files   = sorted(folder_path.glob("*.wav"))

    if not txt_files:
        print(f"[ERROR] No se encontraron archivos .txt en: {folder}")
        return

    print(f"\n{'='*60}")
    print(f"  Escaneando {len(txt_files)} archivos .txt de ICBHI...")
    print(f"{'='*60}\n")

    # Contadores globales
    total_cycles    = 0
    crackle_cycles  = 0
    wheeze_cycles   = 0
    both_cycles     = 0
    normal_cycles   = 0

    equipment_counter = Counter()
    location_counter  = Counter()
    mode_counter      = Counter()
    patient_ids       = set()

    duration_list     = []
    format_errors     = []
    files_processed   = 0

    for fpath in txt_files:
        stem = fpath.stem
        # Verificar que tiene .wav correspondiente
        wav_path = fpath.parent / f"{stem}.wav"

        # Parsear nombre
        meta = parse_icbhi_filename(stem)
        if meta:
            equipment_counter[meta["equipment"]] += 1
            location_counter[meta["location"]]   += 1
            mode_counter[meta["mode"]]            += 1
            patient_ids.add(meta["patient_id"])

        # Parsear anotaciones
        file_ok     = True
        file_cycles = 0
        t_max       = 0.0

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()

                    # Verificar formato: debe tener exactamente 4 columnas numéricas
                    if len(parts) != 4:
                        format_errors.append(
                            f"{fpath.name} línea {lineno}: "
                            f"esperadas 4 columnas, encontradas {len(parts)} → '{line}'"
                        )
                        file_ok = False
                        continue

                    try:
                        t_start  = float(parts[0])
                        t_end    = float(parts[1])
                        crackle  = int(parts[2])
                        wheeze   = int(parts[3])
                    except ValueError:
                        format_errors.append(
                            f"{fpath.name} línea {lineno}: "
                            f"valores no numericos → '{line}'"
                        )
                        file_ok = False
                        continue

                    # Verificar valores binarios
                    if crackle not in {0, 1} or wheeze not in {0, 1}:
                        format_errors.append(
                            f"{fpath.name} línea {lineno}: "
                            f"crackle={crackle} wheeze={wheeze} no son binarios"
                        )

                    total_cycles   += 1
                    file_cycles    += 1
                    t_max           = max(t_max, t_end)

                    if crackle == 1 and wheeze == 1:
                        both_cycles    += 1
                    elif crackle == 1:
                        crackle_cycles += 1
                    elif wheeze == 1:
                        wheeze_cycles  += 1
                    else:
                        normal_cycles  += 1

            if file_ok and file_cycles > 0:
                duration_list.append(t_max)
                files_processed += 1

        except Exception as e:
            format_errors.append(f"{fpath.name}: error de lectura — {e}")

    # ── Reporte general ───────────────────────────────────────────────────────
    print(f"  Archivos .txt encontrados : {len(txt_files)}")
    print(f"  Archivos .wav encontrados : {len(wav_files)}")
    print(f"  Pacientes unicos          : {len(patient_ids)}")
    print(f"  Archivos procesados OK    : {files_processed}")

    print(f"\n{'='*60}")
    print("  DISTRIBUCION DE CICLOS RESPIRATORIOS:")
    print(f"{'='*60}")
    print(f"  Total ciclos anotados  : {total_cycles:>8,}")
    print(f"  Normal (sin adventicios): {normal_cycles:>8,}  "
          f"({normal_cycles/total_cycles*100:.1f}%)")
    print(f"  Solo crackle           : {crackle_cycles:>8,}  "
          f"({crackle_cycles/total_cycles*100:.1f}%)")
    print(f"  Solo wheeze            : {wheeze_cycles:>8,}  "
          f"({wheeze_cycles/total_cycles*100:.1f}%)")
    print(f"  Crackle + wheeze       : {both_cycles:>8,}  "
          f"({both_cycles/total_cycles*100:.1f}%)")

    # Ventanas estimadas (asumiendo ventana 1s, hop 0.5s)
    if duration_list:
        import statistics
        dur_mean = statistics.mean(duration_list)
        dur_min  = min(duration_list)
        dur_max  = max(duration_list)
        dur_med  = statistics.median(duration_list)
        # Estimacion de ventanas por archivo
        est_wins_per_file = (dur_mean - 1.0) / 0.5 + 1
        est_total_wins    = est_wins_per_file * files_processed

        print(f"\n{'='*60}")
        print("  DURACION DE AUDIOS (segundos):")
        print(f"{'='*60}")
        print(f"  Minima  : {dur_min:>8.1f}s")
        print(f"  Maxima  : {dur_max:>8.1f}s")
        print(f"  Media   : {dur_mean:>8.1f}s")
        print(f"  Mediana : {dur_med:>8.1f}s")
        print(f"\n  Ventanas estimadas (1s, 50% overlap):")
        print(f"  Por archivo (media): ~{est_wins_per_file:.0f}")
        print(f"  Total estimado     : ~{est_total_wins:,.0f}")

    print(f"\n{'='*60}")
    print("  EQUIPOS DE GRABACION:")
    print(f"{'='*60}")
    for equip, count in equipment_counter.most_common():
        pct = count / len(txt_files) * 100
        print(f"  {equip:<20}: {count:>4} archivos ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("  UBICACIONES DE AUSCULTACION:")
    print(f"{'='*60}")
    for loc, count in location_counter.most_common():
        pct = count / len(txt_files) * 100
        print(f"  {loc:<10}: {count:>4} archivos ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("  MODO DE ADQUISICION:")
    print(f"{'='*60}")
    for mode, count in mode_counter.most_common():
        print(f"  {mode:<10}: {count:>4} archivos")

    # ── Errores de formato ────────────────────────────────────────────────────
    if format_errors:
        print(f"\n{'='*60}")
        print(f"  [!] ERRORES DE FORMATO ({len(format_errors)}):")
        print(f"{'='*60}")
        for err in format_errors[:20]:  # mostrar max 20
            print(f"  {err}")
        if len(format_errors) > 20:
            print(f"  ... y {len(format_errors)-20} mas")
    else:
        print(f"\n  [OK] Formato correcto en todos los archivos.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Escanea anotaciones ICBHI y reporta distribucion de clases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder", required=True,
                        help="Carpeta con archivos .wav y .txt de ICBHI")
    args = parser.parse_args()
    scan(args.folder)
