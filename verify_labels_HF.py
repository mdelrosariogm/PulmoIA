"""
verify_labels_HF.py
===================
Verifica que el etiquetado por ventana es correcto comparando
visualmente los timestamps del _label.txt contra las ventanas generadas.

Toma UN archivo steth_ al azar (o el que indiques) y muestra:
  1. Los eventos adventicios del txt
  2. Las ventanas generadas con sus etiquetas
  3. Un diagrama ASCII del solapamiento

Uso:
  python verify_labels_HF.py --folder data/LUNGS
  python verify_labels_HF.py --folder data/LUNGS --file steth_20190801_09_46_05
"""

import argparse
import random
from pathlib import Path

WINDOW_SEC = 1.0
HOP_SEC    = 0.5
TARGET_SR  = 4_000
ADVENTITIOUS_EVENTS = {"wheeze", "crackle", "stridor", "rhonchus"}

EVENT_NAME_MAP = {
    "wheeze":   "wheeze",   "wheezes":  "wheeze",  "w": "wheeze",
    "d":        "crackle",  "crackle":  "crackle", "crackles": "crackle",
    "stridor":  "stridor",  "s":        "stridor",
    "rhonchus": "rhonchus", "rhonchi":  "rhonchus", "r": "rhonchus",
}


def parse_timestamp(ts: str) -> float:
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


def parse_label_file(label_path: str) -> tuple[list, list]:
    """Devuelve (todos_los_eventos, solo_adventicios)."""
    all_events  = []
    adv_events  = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            event_type = parts[0].lower()
            t_start    = parse_timestamp(parts[1])
            t_end      = parse_timestamp(parts[2])
            if t_start < 0 or t_end < 0:
                continue
            all_events.append({
                "event":  parts[0],
                "start":  t_start,
                "end":    t_end,
            })
            if event_type in {"i", "e"}:
                continue
            event_type = EVENT_NAME_MAP.get(event_type)
            if event_type is None:
                continue
            adv_events.append({
                "event": event_type,
                "start": t_start,
                "end":   t_end,
            })
    return all_events, adv_events


def label_window(t_start, t_end, events):
    labels = {ev: 0 for ev in ADVENTITIOUS_EVENTS}
    for ev in events:
        if ev["start"] < t_end and ev["end"] > t_start:
            labels[ev["event"]] = 1
    return labels


def ascii_timeline(t_start_win, t_end_win, events, total_dur=15.0, width=60):
    """Dibuja una línea de tiempo ASCII para la ventana actual."""
    scale = width / total_dur

    def to_pos(t):
        return int(t * scale)

    lines = []
    # Ventana actual
    bar = [" "] * width
    ws = to_pos(t_start_win)
    we = min(to_pos(t_end_win), width - 1)
    for i in range(ws, we + 1):
        bar[i] = "█"
    lines.append(f"  Ventana  |{''.join(bar)}|")

    # Cada evento adventicio
    for ev in events:
        bar = [" "] * width
        es = to_pos(ev["start"])
        ee = min(to_pos(ev["end"]), width - 1)
        for i in range(es, ee + 1):
            bar[i] = "─"
        overlap = ev["start"] < t_end_win and ev["end"] > t_start_win
        marker  = "✅" if overlap else "  "
        lines.append(f"  {ev['event'][:8]:<8} |{''.join(bar)}| {marker}")

    # Regla de tiempo
    ruler = ""
    for i in range(0, width, int(width / 5)):
        t = i / scale
        ruler += f"{t:.0f}s".ljust(int(width / 5))
    lines.append(f"  {'time':<8} |{ruler[:width]}|")
    return "\n".join(lines)


def verify(folder: str, file_stem: str = None):
    folder_path = Path(folder)

    # Seleccionar archivo
    if file_stem:
        wav_path = folder_path / f"{file_stem}.wav"
        if not wav_path.exists():
            print(f"[ERROR] No se encontró: {wav_path}")
            return
    else:
        candidates = list(folder_path.glob("steth_*.wav"))
        if not candidates:
            print(f"[ERROR] No hay archivos steth_*.wav en {folder}")
            return
        # Preferir uno con eventos adventicios
        random.shuffle(candidates)
        wav_path = candidates[0]
        # Intentar encontrar uno con adventicios
        for c in candidates[:20]:
            lp = c.parent / f"{c.stem}_label.txt"
            if lp.exists():
                _, adv = parse_label_file(str(lp))
                if adv:
                    wav_path = c
                    break

    label_path = wav_path.parent / f"{wav_path.stem}_label.txt"
    if not label_path.exists():
        print(f"[ERROR] No se encontró: {label_path}")
        return

    all_events, adv_events = parse_label_file(str(label_path))

    print("\n" + "═" * 65)
    print(f"  Archivo : {wav_path.name}")
    print(f"  Labels  : {label_path.name}")
    print("═" * 65)

    # Mostrar todos los eventos del txt
    print("\n📋  TODOS LOS EVENTOS EN EL TXT:")
    print(f"  {'Evento':<12} {'Inicio':>8} {'Fin':>8}  {'Incluido?'}")
    print("  " + "-" * 45)
    for ev in all_events:
        et = ev["event"].lower()
        canonical = EVENT_NAME_MAP.get(et)
        if et in {"i", "e"}:
            incluido = "ignorado (I/E)"
        elif canonical in ADVENTITIOUS_EVENTS:
            incluido = "[OK] incluido como " + canonical
        else:
            incluido = "[?] desconocido"
        print(f"  {ev['event']:<12} {ev['start']:>8.3f}s {ev['end']:>8.3f}s  {incluido}")

    print(f"\n🎯  EVENTOS ADVENTICIOS DETECTADOS: {len(adv_events)}")
    if not adv_events:
        print("  (ninguno — audio normal/sin adventicios)")
    else:
        for ev in adv_events:
            dur = ev["end"] - ev["start"]
            print(f"  {ev['event']:<10} [{ev['start']:.3f}s → {ev['end']:.3f}s]  duración={dur:.3f}s")

    # Generar ventanas y mostrar etiquetado
    print("\n🪟  VENTANAS GENERADAS Y SUS ETIQUETAS:")
    print(f"  {'#':<4} {'t_start':>8} {'t_end':>8}  "
          f"{'wheeze':>8} {'crackle':>8} {'stridor':>8} {'rhonchus':>9}")
    print("  " + "-" * 65)

    win_samples = int(WINDOW_SEC * TARGET_SR)
    hop_samples = int(HOP_SEC   * TARGET_SR)
    audio_dur   = 15.0  # steth_ siempre 15s
    n_samples   = int(audio_dur * TARGET_SR)

    starts = range(0, n_samples - win_samples + 1, hop_samples)
    windows_with_events = []

    for idx, s_idx in enumerate(starts):
        t_start = s_idx / TARGET_SR
        t_end   = (s_idx + win_samples) / TARGET_SR
        labels  = label_window(t_start, t_end, adv_events)

        has_any = any(v == 1 for v in labels.values())
        marker  = " ←" if has_any else ""

        print(f"  {idx+1:<4} {t_start:>7.2f}s {t_end:>7.2f}s  "
              f"  {labels['wheeze']:>6}   {labels['crackle']:>7}"
              f"   {labels['stridor']:>7}  {labels['rhonchus']:>8}{marker}")

        if has_any:
            windows_with_events.append((idx + 1, t_start, t_end, labels))

    # Diagrama ASCII para ventanas con eventos
    if windows_with_events and adv_events:
        print(f"\n📊  DIAGRAMA DE SOLAPAMIENTO (primeras 3 ventanas con eventos):")
        print(f"  Escala: 0s {'─'*50} 15s\n")
        for win_num, t_start, t_end, labels in windows_with_events[:3]:
            active = [ev for ev in adv_events
                      if ev["start"] < t_end and ev["end"] > t_start]
            print(f"  Ventana #{win_num} [{t_start:.2f}s → {t_end:.2f}s] "
                  f"labels={labels}")
            print(ascii_timeline(t_start, t_end, active))
            print()

    # Resumen
    total_wins = len(list(starts))
    print(f"\n📈  RESUMEN:")
    print(f"  Total ventanas generadas : {total_wins}")
    print(f"  Ventanas con ≥1 evento   : {len(windows_with_events)}")
    print(f"  Ventanas sin eventos     : {total_wins - len(windows_with_events)}")
    if total_wins > 0:
        pct = len(windows_with_events) / total_wins * 100
        print(f"  % ventanas positivas     : {pct:.1f}%")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifica el etiquetado por ventana de HF_Lung_V1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--folder", required=True,
                        help="Carpeta con archivos steth_*.wav y _label.txt")
    parser.add_argument("--file",   default=None,
                        help="Nombre del archivo sin extensión (opcional). "
                             "Si no se indica, elige uno con adventicios automáticamente.")
    args = parser.parse_args()
    verify(args.folder, args.file)
