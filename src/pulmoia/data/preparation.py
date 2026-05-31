"""
data_preparation.py
====================
Preparación completa de datos para el Paso 1 — Detector multilabel.

Pasos:
  1. Carga HF, ICBHI y HF+ICBHI combinados
  2. Limpieza por correlación (umbral 0.8) — sobre todos los datos
  3. Split 70/30 por archivo de audio (sin leakage)
  4. Estandarización (StandardScaler fit sobre train únicamente)
  5. Guarda scaler → models/scaler_{DB}.pkl
  6. SHAP sobre train estandarizado (XGBoost por target)
  7. Guarda todo en carpetas organizadas por DB

Estructura de salida:
  outputs/
  ├── cleaning/
  │   ├── HF/
  │   │   ├── 01_corr_matrix_before.png
  │   │   ├── 02_corr_matrix_after.png
  │   │   ├── 03_dist_all_features_before.png
  │   │   ├── 04_dist_all_features_after.png
  │   │   ├── removed_features.txt
  │   │   ├── selected_features.txt
  │   │   ├── train.csv
  │   │   └── test.csv
  │   ├── ICBHI/  (igual)
  │   └── HF_ICBHI/  (igual)
  ├── shap/
  │   ├── HF/
  │   │   ├── shap_summary_has_wheeze.png
  │   │   ├── shap_bar_has_wheeze.png
  │   │   └── ... (por cada target)
  │   ├── ICBHI/
  │   └── HF_ICBHI/
  models/
  ├── scaler_HF.pkl
  ├── scaler_ICBHI.pkl
  └── scaler_HF_ICBHI.pkl

Uso:
  python data_preparation.py
  python data_preparation.py --db hf        # solo HF
  python data_preparation.py --db icbhi     # solo ICBHI
  python data_preparation.py --db combined  # solo HF+ICBHI
  python data_preparation.py --corr 0.85   # umbral de correlacion

Dependencias:
  pip install scikit-learn xgboost shap matplotlib seaborn pandas numpy
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

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
    "HF": "outputs/features_HF.csv",
    "ICBHI": "outputs/features_ICBHI.csv",
}
OUT_CLEAN = Path("outputs/cleaning")
OUT_SHAP = Path("outputs/shap")
OUT_MODELS = Path("models")

TARGETS = ["has_wheeze", "has_crackle", "has_stridor", "has_rhonchus"]
META_COLS = [
    "filename",
    "source",
    "patient_id",
    "location",
    "channel",
    "mode",
    "equipment",
    "t_start_s",
    "t_end_s",
    "sr_original",
    "resampled",
    "diagnosis",
]

CORR_THRESHOLD = 0.80  # umbral de correlación
TEST_SIZE = 0.30  # 70/30 split
RANDOM_STATE = 42
SAMPLE_SHAP = 2000  # ventanas para SHAP (rapidez)
N_ESTIMATORS = 200  # XGBoost para SHAP


# ─── Helpers ──────────────────────────────────────────────────────────────────
def sep(title=""):
    log.info("=" * 65)
    if title:
        log.info("  %s", title)
        log.info("=" * 65)


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Figura: %s", path)


def get_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c not in TARGETS]


def get_group_col(df: pd.DataFrame) -> str:
    """Columna para agrupar el split por archivo de audio."""
    if "filename" in df.columns:
        return "filename"
    return None


# ─── CARGA ────────────────────────────────────────────────────────────────────
def load_databases(db_filter: str | None) -> dict[str, pd.DataFrame]:
    sep("CARGA DE DATOS")
    dbs = {}

    for name, path in PATHS.items():
        if db_filter and db_filter != name.lower():
            continue
        if not Path(path).exists():
            log.warning("  No encontrado: %s", path)
            continue
        df = pd.read_csv(path)
        # Rellenar NaN con mediana
        feat_cols = get_feat_cols(df)
        nan_count = df[feat_cols].isna().sum().sum()
        if nan_count > 0:
            log.info("  %s: %d NaN → rellenando con mediana", name, nan_count)
            df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
        dbs[name] = df
        log.info(
            "  %s: %d filas x %d cols | features: %d",
            name,
            len(df),
            len(df.columns),
            len(feat_cols),
        )

    # Combinado HF + ICBHI
    if (db_filter is None or db_filter == "combined") and "HF" in dbs and "ICBHI" in dbs:
        combined = pd.concat([dbs["HF"], dbs["ICBHI"]], ignore_index=True)
        feat_cols = get_feat_cols(combined)
        combined[feat_cols] = combined[feat_cols].fillna(combined[feat_cols].median())
        dbs["HF_ICBHI"] = combined
        log.info("  HF_ICBHI: %d filas x %d cols", len(combined), len(combined.columns))

    return dbs


# ─── PASO 1: LIMPIEZA POR CORRELACIÓN ─────────────────────────────────────────
def plot_correlation(
    df: pd.DataFrame, feat_cols: list[str], out_dir: Path, suffix: str, max_vars: int = 60
):
    """
    Genera mapa de correlación.
    Si hay más de max_vars features, muestra los primeros max_vars.
    """
    sample = feat_cols[:max_vars]
    corr = df[sample].corr()

    # Tamaño dinámico según número de variables
    size = max(12, len(sample) * 0.18)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Correlacion de Pearson", "shrink": 0.8},
    )
    n_shown = len(sample)
    n_total = len(feat_cols)
    ax.set_title(
        f"Matriz de correlacion {suffix}\n" f"(mostrando {n_shown} de {n_total} features)",
        fontsize=11,
    )
    plt.tight_layout()
    save_fig(fig, out_dir / f"corr_matrix_{suffix}.png")


def plot_distributions_all(
    df: pd.DataFrame, feat_cols: list[str], out_dir: Path, suffix: str, available_targets: list[str]
):
    """
    Genera gráficos de distribución para TODAS las variables.
    Las divide en páginas de 30 features cada una.
    """
    colors = {
        "has_wheeze": "#4C72B0",
        "has_crackle": "#DD8452",
        "has_stridor": "#55A868",
        "has_rhonchus": "#C44E52",
    }

    page_size = 30
    n_pages = (len(feat_cols) + page_size - 1) // page_size

    log.info("  Generando distribuciones para %d features (%d paginas)...", len(feat_cols), n_pages)

    for page in range(n_pages):
        start = page * page_size
        end = min(start + page_size, len(feat_cols))
        chunk = feat_cols[start:end]
        n_cols = 6
        n_rows = (len(chunk) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8))
        fig.suptitle(f"Distribuciones {suffix} — features {start+1} a {end}", fontsize=12, y=1.01)
        axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

        for ax, feat in zip(axes_flat, chunk, strict=False):
            # Distribución general
            vals = df[feat].dropna()
            if len(vals) > 20_000:
                vals = vals.sample(20_000, random_state=42)
            ax.hist(vals, bins=40, alpha=0.35, color="gray", density=True, label="todos")
            # Por target (solo wheeze y crackle para no saturar)
            for t in available_targets[:2]:
                pos = df[df[t] == 1][feat].dropna()
                if len(pos) > 10:
                    if len(pos) > 10_000:
                        pos = pos.sample(10_000, random_state=42)
                    ax.hist(
                        pos,
                        bins=40,
                        alpha=0.5,
                        color=colors.get(t, "blue"),
                        density=True,
                        label=t.replace("has_", ""),
                    )
            ax.set_title(feat[:30], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("")
            if feat == chunk[0]:
                ax.legend(fontsize=6)

        # Ocultar ejes vacíos
        for ax in list(axes_flat)[len(chunk) :]:
            ax.set_visible(False)

        plt.tight_layout()
        fname = f"dist_{suffix}_page{page+1:02d}_of_{n_pages:02d}.png"
        save_fig(fig, out_dir / fname)

    log.info("  Distribuciones guardadas: %d paginas", n_pages)


def clean_by_correlation(
    df: pd.DataFrame, feat_cols: list[str], threshold: float, out_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    """
    Elimina features con correlación > threshold.
    Estrategia: de cada par correlacionado, elimina el segundo.
    Devuelve (df_clean, feat_cols_clean).
    """
    sep(f"PASO 1 — LIMPIEZA POR CORRELACION (umbral={threshold})")

    out_dir.mkdir(parents=True, exist_ok=True)
    available = [t for t in TARGETS if t in df.columns]

    # Plot ANTES
    log.info("  Generando matriz de correlacion ANTES...")
    plot_correlation(df, feat_cols, out_dir, "01_antes")

    # Plot distribuciones ANTES
    plot_distributions_all(df, feat_cols, out_dir, "before", available)

    # Calcular correlación
    log.info("  Calculando correlacion entre %d features...", len(feat_cols))
    X = df[feat_cols]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Identificar features a eliminar
    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        if high_corr:
            to_drop.add(col)

    removed = sorted(to_drop)
    remaining = [c for c in feat_cols if c not in to_drop]

    log.info("  Features originales  : %d", len(feat_cols))
    log.info("  Eliminados (corr>%.2f): %d", threshold, len(removed))
    log.info("  Features restantes   : %d", len(remaining))

    # Guardar listas
    with open(out_dir / "removed_features.txt", "w") as f:
        f.write(f"# Features eliminados por correlacion > {threshold}\n")
        f.write(f"# Total eliminados: {len(removed)}\n\n")
        for feat in removed:
            f.write(f"{feat}\n")

    with open(out_dir / "selected_features.txt", "w") as f:
        f.write("# Features seleccionados tras limpieza por correlacion\n")
        f.write(f"# Total: {len(remaining)}\n\n")
        for feat in remaining:
            f.write(f"{feat}\n")

    # Plot DESPUÉS
    log.info("  Generando matriz de correlacion DESPUES...")
    plot_correlation(df, remaining, out_dir, "02_despues")

    # Plot distribuciones DESPUÉS
    plot_distributions_all(df, remaining, out_dir, "after", available)

    # Dataset limpio
    df_clean = df[remaining + TARGETS + [c for c in META_COLS if c in df.columns]].copy()

    return df_clean, remaining


# ─── PASO 2: SPLIT 70/30 POR ARCHIVO ─────────────────────────────────────────
def split_by_file(
    df: pd.DataFrame, feat_cols: list[str], out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep("PASO 2 — SPLIT 70/30 POR ARCHIVO DE AUDIO")

    group_col = get_group_col(df)

    if group_col is None:
        log.warning("  No hay columna de agrupacion — split aleatorio")
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    else:
        # Split por archivo — todas las ventanas del mismo audio van juntas
        groups = df[group_col].values
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

    log.info(
        "  Train: %d ventanas (%d archivos)",
        len(train),
        train[group_col].nunique() if group_col else "?",
    )
    log.info(
        "  Test : %d ventanas (%d archivos)",
        len(test),
        test[group_col].nunique() if group_col else "?",
    )

    # Distribución de etiquetas en train y test
    available = [t for t in TARGETS if t in df.columns]
    log.info("\n  Distribucion de etiquetas:")
    log.info("  %-15s  %8s  %8s", "Target", "Train%", "Test%")
    log.info("  " + "-" * 35)
    for t in available:
        tr_pct = train[t].mean() * 100
        te_pct = test[t].mean() * 100
        log.info("  %-15s  %7.1f%%  %7.1f%%", t, tr_pct, te_pct)

    return train, test


# ─── PASO 3: ESTANDARIZACIÓN ──────────────────────────────────────────────────
def standardize(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], db_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    sep("PASO 3 — ESTANDARIZACION (fit sobre train)")

    scaler = StandardScaler()

    # Fit SOLO sobre train
    scaler.fit(train[feat_cols])

    # Transform train y test
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[feat_cols] = scaler.transform(train[feat_cols])
    test_scaled[feat_cols] = scaler.transform(test[feat_cols])

    log.info("  Scaler ajustado sobre %d ventanas de train", len(train))
    log.info("  Media  (primeros 3 features): %s", scaler.mean_[:3].round(4))
    log.info("  Std    (primeros 3 features): %s", scaler.scale_[:3].round(4))

    # Verificar que quedó media~0 y std~1
    means = train_scaled[feat_cols].mean()
    stds = train_scaled[feat_cols].std()
    log.info("  Verificacion post-scaling:")
    log.info("  Media promedio  : %.6f (esperado ~0)", means.mean())
    log.info("  Std promedio    : %.6f (esperado ~1)", stds.mean())

    # Guardar scaler
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    scaler_path = OUT_MODELS / f"scaler_{db_name}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("  Scaler guardado: %s", scaler_path)

    return train_scaled, test_scaled, scaler


# ─── PASO 4: SHAP ─────────────────────────────────────────────────────────────
def run_shap(train: pd.DataFrame, feat_cols: list[str], db_name: str, out_dir: Path):
    sep(f"PASO 4 — SHAP SOBRE TRAIN ESTANDARIZADO — {db_name}")

    available = [t for t in TARGETS if t in train.columns]
    X = train[feat_cols].values

    for target in available:
        y = train[target].values
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        log.info(
            "\n  Target: %s | pos=%d neg=%d ratio=1:%.1f",
            target,
            n_pos,
            n_neg,
            n_neg / max(n_pos, 1),
        )

        if n_pos < 20:
            log.warning("  Muy pocos positivos — omitiendo SHAP para %s", target)
            continue

        # Entrenar XGBoost
        scale_pos = n_neg / max(n_pos, 1)
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )
        # Submuestra si hay muchos datos
        if len(X) > 50_000:
            idx = np.random.RandomState(RANDOM_STATE).choice(len(X), 50_000, replace=False)
            model.fit(X[idx], y[idx])
        else:
            model.fit(X, y)

        # SHAP — muestra para rapidez
        n_shap = min(SAMPLE_SHAP, len(X))
        idx_sh = np.random.RandomState(RANDOM_STATE).choice(len(X), n_shap, replace=False)
        X_shap = X[idx_sh]

        log.info("  Calculando SHAP values (%d muestras)...", n_shap)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        target_dir = out_dir / db_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Summary plot (beeswarm — dirección del efecto)
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feat_cols,
            show=False,
            max_display=30,
            plot_size=(12, 10),
        )
        plt.title(f"SHAP Summary — {target} | {db_name}", fontsize=11)
        plt.tight_layout()
        save_fig(plt.gcf(), target_dir / f"shap_summary_{target}.png")

        # Bar plot (importancia media absoluta)
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feat_cols,
            plot_type="bar",
            show=False,
            max_display=30,
            plot_size=(10, 10),
        )
        plt.title(f"SHAP Importancia — {target} | {db_name}", fontsize=11)
        plt.tight_layout()
        save_fig(plt.gcf(), target_dir / f"shap_bar_{target}.png")

        # Guardar top features por target
        shap_mean = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(shap_mean)[::-1][:30]
        top_feats = [(feat_cols[i], float(shap_mean[i])) for i in top_idx]

        with open(target_dir / f"shap_top30_{target}.txt", "w") as f:
            f.write(f"# Top 30 features por SHAP — {target} | {db_name}\n\n")
            for i, (feat, score) in enumerate(top_feats, 1):
                f.write(f"{i:2d}. {feat:<45} {score:.6f}\n")

        log.info("  SHAP %s completado.", target)


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────
def run(db_filter: str | None, corr_threshold: float):
    global CORR_THRESHOLD
    CORR_THRESHOLD = corr_threshold

    # Cargar datos
    dbs = load_databases(db_filter)
    if not dbs:
        log.error("No se cargaron datos. Verifica los CSVs en outputs/")
        return

    for db_name, df in dbs.items():
        sep(f"PROCESANDO: {db_name}")

        clean_dir = OUT_CLEAN / db_name
        shap_dir = OUT_SHAP
        clean_dir.mkdir(parents=True, exist_ok=True)

        feat_cols = get_feat_cols(df)
        available = [t for t in TARGETS if t in df.columns]

        log.info("  Features iniciales : %d", len(feat_cols))
        log.info("  Targets disponibles: %s", available)
        log.info("  Ventanas totales   : %d", len(df))

        # ── Paso 1: Limpieza por correlación ──────────────────────────────────
        df_clean, clean_cols = clean_by_correlation(df, feat_cols, CORR_THRESHOLD, clean_dir)

        # ── Paso 2: Split 70/30 por archivo ───────────────────────────────────
        train, test = split_by_file(df_clean, clean_cols, clean_dir)

        # Guardar train y test
        train.to_csv(clean_dir / "train.csv", index=False)
        test.to_csv(clean_dir / "test.csv", index=False)
        log.info("  Train guardado: %s/train.csv", clean_dir)
        log.info("  Test guardado : %s/test.csv", clean_dir)

        # ── Paso 3: Estandarización ────────────────────────────────────────────
        train_sc, test_sc, scaler = standardize(train, test, clean_cols, db_name)

        # Guardar train y test estandarizados
        train_sc.to_csv(clean_dir / "train_scaled.csv", index=False)
        test_sc.to_csv(clean_dir / "test_scaled.csv", index=False)
        log.info("  Train scaled: %s/train_scaled.csv", clean_dir)
        log.info("  Test scaled : %s/test_scaled.csv", clean_dir)

        # ── Paso 4: SHAP ───────────────────────────────────────────────────────
        run_shap(train_sc, clean_cols, db_name, shap_dir)

        # ── Resumen por DB ─────────────────────────────────────────────────────
        sep(f"RESUMEN — {db_name}")
        log.info("  Features originales  : %d", len(feat_cols))
        log.info("  Features tras limpieza: %d", len(clean_cols))
        log.info("  Train ventanas       : %d", len(train_sc))
        log.info("  Test ventanas        : %d", len(test_sc))
        log.info("  Archivos generados:")
        for f in sorted(clean_dir.glob("*")):
            log.info("  - %-50s %.1f KB", f.name, f.stat().st_size / 1024)
        shap_sub = shap_dir / db_name
        if shap_sub.exists():
            for f in sorted(shap_sub.glob("*")):
                log.info("  - shap/%-45s %.1f KB", f.name, f.stat().st_size / 1024)

    sep("PIPELINE COMPLETADO")
    log.info("  Resultados en:")
    log.info("  - outputs/cleaning/  (datos limpios, train/test, distribuciones)")
    log.info("  - outputs/shap/      (importancia de features por target)")
    log.info("  - models/            (scalers guardados)")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preparacion de datos — limpieza, split, estandarizacion y SHAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default=None,
        choices=["hf", "icbhi", "combined"],
        help="Base de datos a procesar (default: todas)",
    )
    parser.add_argument(
        "--corr", type=float, default=0.80, help="Umbral de correlacion para eliminar features"
    )
    args = parser.parse_args()

    db_map = {"hf": "HF", "icbhi": "ICBHI", "combined": "HF_ICBHI"}
    db_filter = db_map.get(args.db) if args.db else None

    run(db_filter, args.corr)
