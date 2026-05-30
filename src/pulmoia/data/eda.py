"""
eda_report.py
=============
Estadística descriptiva y reporte EDA para las 3 bases de datos
en el contexto del Paso 1 — Detector multilabel de sonidos adventicios.

Genera:
  1. Reporte Sweetviz por base de datos (HF, ICBHI, TR)
  2. Reporte Sweetviz del dataset combinado HF + ICBHI
  3. Estadísticas descriptivas en consola
  4. Gráficos de distribución de etiquetas multilabel
  5. Comparación entre bases de datos

Los reportes HTML se guardan en outputs/eda/

Uso:
  python eda_report.py                 # todos los reportes
  python eda_report.py --db hf         # solo HF
  python eda_report.py --db icbhi      # solo ICBHI
  python eda_report.py --db tr         # solo TR
  python eda_report.py --db combined   # HF + ICBHI combinados

Dependencias:
  pip install sweetviz pandas numpy matplotlib seaborn
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────
PATHS = {
    "hf":    "outputs/features_HF.csv",
    "icbhi": "outputs/features_ICBHI.csv",
    "tr":    "outputs/features_TR.csv",
}
OUT_DIR = Path("outputs/eda")

TARGETS   = ["has_wheeze", "has_crackle", "has_stridor", "has_rhonchus"]
META_COLS = ["filename", "source", "patient_id", "location", "channel",
             "mode", "equipment", "t_start_s", "t_end_s",
             "sr_original", "resampled", "diagnosis"]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def sep(title=""):
    log.info("=" * 65)
    if title:
        log.info("  %s", title)
        log.info("=" * 65)


def save_fig(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figura: %s", p)


def get_feat_cols(df):
    return [c for c in df.columns
            if c not in META_COLS and c not in TARGETS]


# ─── CARGA ────────────────────────────────────────────────────────────────────
def load_db(key: str) -> pd.DataFrame | None:
    path = PATHS.get(key)
    if not path or not Path(path).exists():
        log.warning("  CSV no encontrado: %s", path)
        return None
    df = pd.read_csv(path)
    log.info("  %s cargado: %d filas x %d columnas",
             key.upper(), len(df), len(df.columns))
    return df


# ─── ESTADÍSTICA DESCRIPTIVA EN CONSOLA ───────────────────────────────────────
def console_stats(df: pd.DataFrame, name: str):
    sep(f"ESTADISTICA DESCRIPTIVA — {name}")
    feat_cols = get_feat_cols(df)

    log.info("  Dimensiones     : %d filas x %d columnas", *df.shape)
    log.info("  Features        : %d", len(feat_cols))

    # Valores nulos
    nan_total = df[feat_cols].isna().sum().sum()
    nan_pct   = nan_total / (df.shape[0] * len(feat_cols)) * 100
    log.info("  NaN en features : %d (%.2f%%)", nan_total, nan_pct)

    # Metadatos
    if "source" in df.columns:
        log.info("\n  Distribucion por fuente:")
        for src, cnt in df["source"].value_counts().items():
            log.info("  %-10s: %6d ventanas (%.1f%%)",
                     src, cnt, cnt/len(df)*100)

    if "patient_id" in df.columns:
        log.info("  Pacientes unicos: %d", df["patient_id"].nunique())

    if "equipment" in df.columns:
        log.info("\n  Equipos de grabacion:")
        for eq, cnt in df["equipment"].value_counts().items():
            log.info("  %-20s: %d", eq, cnt)

    if "diagnosis" in df.columns:
        log.info("\n  Diagnosticos (TR):")
        for dx, cnt in df["diagnosis"].value_counts().items():
            log.info("  %-8s: %6d ventanas (%.1f%%)",
                     dx, cnt, cnt/len(df)*100)

    # Etiquetas multilabel
    available = [t for t in TARGETS if t in df.columns]
    if available:
        log.info("\n  Distribucion de etiquetas multilabel:")
        for t in available:
            n   = int(df[t].sum())
            pct = n / len(df) * 100
            bar = "#" * int(pct / 2)
            log.info("  %-15s: %6d / %d (%.1f%%) |%s",
                     t, n, len(df), pct, bar)

        # Co-ocurrencia
        log.info("\n  Co-ocurrencia de etiquetas (top 10):")
        combos = df[available].astype(str).agg("-".join, axis=1)
        for combo, cnt in combos.value_counts().head(10).items():
            parts  = combo.split("-")
            active = [t.replace("has_", "")
                      for t, p in zip(available, parts) if p == "1"]
            label  = "+".join(active) if active else "normal"
            log.info("  %-30s: %6d (%.1f%%)",
                     label, cnt, cnt/len(df)*100)

        # Numero de etiquetas activas
        df_tmp = df.copy()
        df_tmp["n_labels"] = df[available].sum(axis=1)
        log.info("\n  Ventanas por numero de etiquetas activas:")
        for n, cnt in df_tmp["n_labels"].value_counts().sort_index().items():
            label = {0: "normal", 1: "1 evento",
                     2: "2 eventos", 3: "3 eventos",
                     4: "4 eventos"}.get(int(n), str(n))
            log.info("  %-12s: %6d (%.1f%%)",
                     label, cnt, cnt/len(df)*100)

    # Stats de features (muestra)
    log.info("\n  Estadisticas de features (primeros 6):")
    sample = feat_cols[:6]
    stats  = df[sample].describe().T[["mean", "std", "min", "max"]]
    for feat, row in stats.iterrows():
        log.info("  %-40s mean=%8.3f std=%8.3f min=%8.3f max=%8.3f",
                 feat[:40], row["mean"], row["std"],
                 row["min"], row["max"])


# ─── GRÁFICOS ─────────────────────────────────────────────────────────────────
def plot_label_distribution(df: pd.DataFrame, name: str):
    available = [t for t in TARGETS if t in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Distribucion de etiquetas — {name}", fontsize=13)

    n_total = len(df)

    # 1. Barras por etiqueta
    ax = axes[0]
    counts = {t: int(df[t].sum()) for t in available}
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars   = ax.bar(
        [t.replace("has_", "") for t in available],
        [counts[t] / n_total * 100 for t in available],
        color=colors[:len(available)], alpha=0.85, edgecolor="white"
    )
    ax.set_ylabel("% ventanas positivas")
    ax.set_title("Prevalencia por evento")
    for bar, t in zip(bars, available):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{counts[t]:,}\n({counts[t]/n_total*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    # 2. Pie de combinaciones
    ax = axes[1]
    combos     = df[available].astype(str).agg("-".join, axis=1).value_counts()
    top_combos = combos.head(6)
    other      = combos[6:].sum()
    if other > 0:
        top_combos["otros"] = other
    labels_clean = []
    for c in top_combos.index:
        parts  = c.split("-")
        active = [t.replace("has_", "")
                  for t, p in zip(available, parts) if p == "1"]
        labels_clean.append("+".join(active) if active else "normal")
    ax.pie(top_combos.values, labels=labels_clean,
           autopct="%1.1f%%", startangle=90,
           colors=plt.cm.Set3.colors[:len(top_combos)])
    ax.set_title("Combinaciones de etiquetas")

    # 3. N etiquetas activas
    ax      = axes[2]
    df_tmp  = df.copy()
    df_tmp["n_labels"] = df[available].sum(axis=1)
    n_counts = df_tmp["n_labels"].value_counts().sort_index()
    xlabels  = [f"{int(n)} evento{'s' if n>1 else ''}"
                if n > 0 else "normal" for n in n_counts.index]
    ax.bar(xlabels, n_counts.values / n_total * 100,
           color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.set_ylabel("% ventanas")
    ax.set_title("Etiquetas activas por ventana")
    for i, (_, v) in enumerate(n_counts.items()):
        ax.text(i, v/n_total*100 + 0.3, f"{v:,}",
                ha="center", fontsize=8)

    plt.tight_layout()
    save_fig(fig, f"labels_{name.lower().replace(' ', '_').replace('+','_')}.png")


def plot_feature_distributions(df: pd.DataFrame, name: str):
    feat_cols = get_feat_cols(df)
    available = [t for t in TARGETS if t in df.columns]

    # Features representativos de cada grupo
    key = ["SpectralCentroid_mean", "SpectralSpread_mean",
           "SpectralEntropy_mean", "SpectralFlatness_mean",
           "MFCC_mean_1", "MFCC_mean_2", "MFCC_mean_3", "MFCC_mean_4",
           "MFCC_delta_mean_1", "MFCC_delta_delta_mean_1",
           "SpectralFlux_mean", "SpectralCrest_mean"]
    sample = [f for f in key if f in feat_cols][:12]
    if not sample:
        sample = feat_cols[:12]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f"Distribucion de features — {name}", fontsize=13)

    colors = {"has_wheeze": "#4C72B0", "has_crackle": "#DD8452",
              "has_stridor": "#55A868", "has_rhonchus": "#C44E52"}

    for ax, feat in zip(axes.flat, sample):
        ax.hist(df[feat].dropna(), bins=50, alpha=0.4,
                color="gray", label="todos", density=True)
        for t in available[:2]:
            pos = df[df[t] == 1][feat].dropna()
            if len(pos) > 10:
                ax.hist(pos, bins=50, alpha=0.5,
                        color=colors[t],
                        label=t.replace("has_", ""),
                        density=True)
        ax.set_title(feat[:35], fontsize=8)
        ax.tick_params(labelsize=7)
        if feat == sample[0]:
            ax.legend(fontsize=7)

    for ax in axes.flat[len(sample):]:
        ax.set_visible(False)

    plt.tight_layout()
    save_fig(fig, f"feat_dists_{name.lower().replace(' ', '_').replace('+','_')}.png")


def plot_correlation_matrix(df: pd.DataFrame, name: str):
    feat_cols = get_feat_cols(df)
    key = ["SpectralCentroid_mean", "SpectralSpread_mean",
           "SpectralEntropy_mean", "SpectralFlatness_mean",
           "SpectralFlux_mean", "SpectralCrest_mean",
           "MFCC_mean_1", "MFCC_mean_2", "MFCC_mean_3",
           "MFCC_mean_4", "MFCC_mean_5",
           "MFCC_delta_mean_1", "MFCC_delta_delta_mean_1"]
    sample = [f for f in key if f in feat_cols][:15]
    if len(sample) < 5:
        sample = feat_cols[:15]

    corr = df[sample].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm",
                center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=[f[:20] for f in sample],
                yticklabels=[f[:20] for f in sample])
    ax.set_title(f"Correlacion entre features — {name}", fontsize=12)
    plt.tight_layout()
    save_fig(fig, f"corr_{name.lower().replace(' ', '_').replace('+','_')}.png")


def plot_comparison(dfs: dict):
    sep("COMPARACION ENTRE BASES DE DATOS")
    available = {k: v for k, v in dfs.items() if v is not None}
    if len(available) < 2:
        return

    key_features = ["SpectralCentroid_mean", "SpectralEntropy_mean",
                    "MFCC_mean_1", "SpectralFlux_mean"]
    colors_db = {"HF": "#4C72B0", "ICBHI": "#DD8452", "TR": "#55A868"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparacion de distribuciones entre bases de datos",
                 fontsize=13)

    for ax, feat in zip(axes.flat, key_features):
        for name, df in available.items():
            if feat in df.columns:
                vals = df[feat].dropna()
                if len(vals) > 10000:
                    vals = vals.sample(10000, random_state=42)
                ax.hist(vals, bins=60, alpha=0.5, label=name,
                        density=True,
                        color=colors_db.get(name.upper(), "gray"))
        ax.set_title(feat[:35], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")

    plt.tight_layout()
    save_fig(fig, "comparison_distributions.png")

    # Barplot prevalencia por DB
    data_plot = []
    for name, df in available.items():
        for t in TARGETS:
            if t in df.columns:
                data_plot.append({
                    "DB":           name,
                    "Etiqueta":     t.replace("has_", ""),
                    "Prevalencia":  df[t].mean() * 100,
                })
    if data_plot:
        df_plot = pd.DataFrame(data_plot)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_plot, x="Etiqueta", y="Prevalencia",
                    hue="DB", ax=ax,
                    palette=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title("Prevalencia de eventos acusticos por base de datos",
                     fontsize=12)
        ax.set_ylabel("% ventanas positivas")
        ax.legend(title="Base de datos")
        plt.tight_layout()
        save_fig(fig, "comparison_prevalence.png")


# ─── SWEETVIZ REPORT ──────────────────────────────────────────────────────────
def generate_sweetviz(df: pd.DataFrame, name: str,
                      target: str = None, compare_df: pd.DataFrame = None,
                      compare_name: str = None):
    sep(f"SWEETVIZ REPORT — {name}")

    feat_cols = get_feat_cols(df)
    available = [t for t in TARGETS if t in df.columns]

    # Seleccionar columnas para el reporte
    # (subset representativo — sweetviz es lento con 157 features)
    key_features = []
    for prefix in ["SpectralCentroid", "SpectralSpread", "SpectralEntropy",
                   "SpectralFlatness", "SpectralFlux", "SpectralCrest",
                   "SpectralSkewness", "SpectralKurtosis",
                   "SpectralSlope", "SpectralDecrease", "SpectralRolloffPoint"]:
        for suffix in ["_mean", "_std", "_coefv"]:
            col = f"{prefix}{suffix}"
            if col in feat_cols:
                key_features.append(col)

    for i in range(1, 8):
        for prefix in ["MFCC", "MFCC_delta", "MFCC_delta_delta"]:
            col = f"{prefix}_mean_{i}"
            if col in feat_cols:
                key_features.append(col)

    meta_keep = [c for c in ["source", "equipment", "location",
                              "diagnosis", "t_start_s"] if c in df.columns]
    cols_report = available + key_features + meta_keep
    df_report   = df[[c for c in cols_report if c in df.columns]].copy()

    # Submuestreo para rapidez
    if len(df_report) > 30_000:
        log.info("  Submuestreando a 30,000 filas...")
        df_report = df_report.sample(30_000, random_state=42)

    log.info("  Dimensiones reporte: %d filas x %d cols", *df_report.shape)

    try:
        # Target principal para el reporte
        sv_target = target if target and target in df_report.columns else None

        if compare_df is not None:
            # Reporte comparativo entre dos datasets
            df_compare = compare_df[[c for c in cols_report
                                     if c in compare_df.columns]].copy()
            if len(df_compare) > 30_000:
                df_compare = df_compare.sample(30_000, random_state=42)
            report = sv.compare(
                [df_report, name],
                [df_compare, compare_name or "Comparacion"],
                target_feat=sv_target,
            )
        else:
            report = sv.analyze(
                df_report,
                target_feat=sv_target,
                pairwise_analysis="off",  # mas rapido
            )

        out_path = OUT_DIR / f"sweetviz_{name.lower().replace(' ', '_').replace('+','_')}.html"
        report.show_html(str(out_path), open_browser=False)
        log.info("  Reporte guardado: %s", out_path)

    except Exception as e:
        log.error("  Error en sweetviz: %s", e)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main(db_filter: str | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sep("EDA — ESTADISTICA DESCRIPTIVA | pulmoia")

    # Cargar bases
    dfs = {}
    if db_filter is None or db_filter == "hf":
        dfs["HF"] = load_db("hf")
    if db_filter is None or db_filter == "icbhi":
        dfs["ICBHI"] = load_db("icbhi")
    if db_filter is None or db_filter == "tr":
        dfs["TR"] = load_db("tr")

    # Dataset combinado HF + ICBHI
    if db_filter is None or db_filter == "combined":
        parts = [v for k, v in dfs.items()
                 if k in ("HF", "ICBHI") and v is not None]
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            dfs["HF + ICBHI"] = combined
            log.info("  Combinado HF+ICBHI: %d filas", len(combined))

    # Análisis individual por base
    for name, df in dfs.items():
        if df is None:
            continue
        log.info("\n")
        console_stats(df, name)
        plot_label_distribution(df, name)
        plot_feature_distributions(df, name)
        plot_correlation_matrix(df, name)

        # Sweetviz individual con has_crackle como target principal
        target = "has_crackle" if "has_crackle" in df.columns else None
        generate_sweetviz(df, name, target=target)

    # Reporte comparativo HF vs ICBHI
    if dfs.get("HF") is not None and dfs.get("ICBHI") is not None:
        log.info("\n")
        sep("SWEETVIZ COMPARATIVO — HF vs ICBHI")
        generate_sweetviz(
            dfs["HF"], "HF",
            target="has_crackle",
            compare_df=dfs["ICBHI"],
            compare_name="ICBHI"
        )

    # Comparación gráfica entre DBs
    plot_comparison({k: v for k, v in dfs.items()
                     if k in ("HF", "ICBHI", "TR") and v is not None})

    # Resumen final
    sep("RESUMEN FINAL")
    log.info("  Archivos generados en %s:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*")):
        log.info("  %-55s %.1f KB", f.name, f.stat().st_size/1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EDA y estadistica descriptiva — 3 bases de datos | pulmoia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", default=None,
        choices=["hf", "icbhi", "tr", "combined"],
        help="Base de datos a analizar (default: todas)"
    )
    args = parser.parse_args()
    main(args.db)
