"""
model_pipeline.py
Training, evaluation, and persistence for all 7 ML models × 4 poverty thresholds.

Approach A: Predict poverty headcounts up to 2050 (full feature coverage).
28 trained models total (7 models × 4 thresholds).

Input  : data/final/X_train.csv, X_test.csv, y_train.csv, y_test.csv, cv_folds.json
         (features are pre-scaled by StandardScaler from feature_engineering.py)
Output : models/{name}_{threshold}_approach_a.pkl        (28 files)
         outputs/model_comparison_approach_a.csv
         outputs/model_comparison_by_threshold.csv
         outputs/model_comparison_approach_a.png
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from pygam import LinearGAM, s

from config import RANDOM_STATE, MODELS_DIR, DATA_FINAL_DIR, OUTPUTS_DIR


# ── Threshold mappings ─────────────────────────────────────────────────────────

THRESHOLD_COL_MAP: dict[str, str] = {
    "$3":    "poverty_3",
    "$4.20": "poverty_4_20",
    "$8.30": "poverty_8_30",
    "$10":   "poverty_10",
}

THRESHOLD_SLUG: dict[str, str] = {
    "$3":    "3",
    "$4.20": "4_20",
    "$8.30": "8_30",
    "$10":   "10",
}

ALL_THRESHOLDS = list(THRESHOLD_COL_MAP.keys())

# Model display order (for tables / plots)
MODEL_NAMES = [
    "xgboost_cpu", "xgboost_gpu", "lightgbm",
    "random_forest", "gam", "ridge", "mlp",
]


# ── GPU detection ──────────────────────────────────────────────────────────────

def _check_gpu() -> bool:
    """Return True if XGBoost CUDA is usable on this machine."""
    try:
        probe = xgb.XGBRegressor(
            tree_method="hist", device="cuda",
            n_estimators=1, verbosity=0,
        )
        probe.fit(np.zeros((4, 2)), np.zeros(4))
        return True
    except Exception:
        return False


GPU_AVAILABLE: bool = _check_gpu()


# ── Metrics ────────────────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.1) -> float:
    """
    MAPE computed only where y_true > eps (avoids division by ~0).
    eps = 0.1 percentage points.  Returns NaN if no valid observations.
    """
    mask = y_true > eps
    if mask.sum() == 0:
        return np.nan
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R², MAPE — skips NaN rows in y_true."""
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae":  float(mean_absolute_error(yt, yp)),
        "r2":   float(r2_score(yt, yp)),
        "mape": _mape(yt, yp),
    }


# ── Model construction ────────────────────────────────────────────────────────

def _build_gam(n_features: int, n_splines: int = 20, lam: float = 0.6) -> LinearGAM:
    """Build a LinearGAM with one spline term per feature."""
    terms = sum(s(i, n_splines=n_splines, lam=lam) for i in range(n_features))
    return LinearGAM(terms=terms)


def _build_model(
    name: str,
    params: dict,
    n_features: int,
    gpu: bool = GPU_AVAILABLE,
) -> Any:
    """Instantiate a fresh (unfitted) model with the given hyperparameters."""
    if name == "xgboost_cpu":
        return xgb.XGBRegressor(
            tree_method="hist", device="cpu",
            random_state=RANDOM_STATE, verbosity=0, **params,
        )
    if name == "xgboost_gpu":
        device = "cuda" if gpu else "cpu"
        return xgb.XGBRegressor(
            tree_method="hist", device=device,
            random_state=RANDOM_STATE, verbosity=0, **params,
        )
    if name == "lightgbm":
        return lgb.LGBMRegressor(
            random_state=RANDOM_STATE, verbose=-1, **params,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, **params,
        )
    if name == "gam":
        return _build_gam(
            n_features,
            n_splines=params.get("n_splines", 20),
            lam=params.get("lam", 0.6),
        )
    if name == "ridge":
        # Input is pre-scaled; no internal scaler needed.
        return Ridge(**params)
    if name == "mlp":
        # Input is pre-scaled; no internal scaler needed.
        return MLPRegressor(
            activation="relu", max_iter=1000, early_stopping=True,
            random_state=RANDOM_STATE, **params,
        )
    raise ValueError(f"Unknown model name: '{name}'")


# ── Hyperparameter grids ───────────────────────────────────────────────────────
# Each value is a list of candidate kwarg dicts.
# Grids are deliberately compact to stay practical (28 models × tuning).

PARAM_GRIDS: dict[str, list[dict]] = {
    "xgboost_cpu": [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10,
         "subsample": 0.8,  "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8,  "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
         "subsample": 0.9,  "colsample_bytree": 0.9},
    ],
    "xgboost_gpu": [   # identical candidates; device set by GPU_AVAILABLE
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10,
         "subsample": 0.8,  "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8,  "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
         "subsample": 0.9,  "colsample_bytree": 0.9},
    ],
    "lightgbm": [
        {"n_estimators": 300, "max_depth": 4,  "learning_rate": 0.10},
        {"n_estimators": 500, "max_depth": 6,  "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": -1, "learning_rate": 0.05,
         "num_leaves": 63},
    ],
    "random_forest": [
        {"n_estimators": 200, "max_depth": 10,   "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2},
    ],
    "gam": [
        {"n_splines": 10, "lam": 0.6},
        {"n_splines": 20, "lam": 0.6},
        {"n_splines": 20, "lam": 10.0},
    ],
    "ridge": [
        {"alpha": 0.01},
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
        {"alpha": 100.0},
    ],
    "mlp": [
        {"hidden_layer_sizes": (64, 32),      "alpha": 0.001,
         "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64),     "alpha": 0.001,
         "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (128, 64, 32), "alpha": 0.01,
         "learning_rate_init": 0.0005},
    ],
}


# ── Cross-validation helpers ───────────────────────────────────────────────────

def _cv_rmse(
    name: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    gpu: bool,
) -> float:
    """
    Mean RMSE across pre-defined folds for a single hyperparameter candidate.
    NaN targets in the validation set are excluded from scoring.
    """
    n_features = X.shape[1]
    scores: list[float] = []

    for tr_idx, va_idx in folds:
        va_valid = va_idx[~np.isnan(y[va_idx])]
        if len(va_valid) == 0:
            continue
        try:
            m = _build_model(name, params, n_features, gpu)
            # Fit on non-NaN train targets
            tr_valid = tr_idx[~np.isnan(y[tr_idx])]
            m.fit(X[tr_valid], y[tr_valid])
            pred = m.predict(X[va_valid])
            scores.append(float(np.sqrt(mean_squared_error(y[va_valid], pred))))
        except Exception as exc:
            warnings.warn(f"  CV fold failed for {name} params={params}: {exc}")

    return float(np.mean(scores)) if scores else np.inf


def tune_hyperparams(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    gpu: bool = GPU_AVAILABLE,
) -> tuple[dict, float]:
    """
    Grid search over PARAM_GRIDS[name] using pre-defined CV folds.
    Returns (best_params, best_cv_rmse).
    """
    candidates = PARAM_GRIDS.get(name, [{}])
    best_params: dict = candidates[0]
    best_rmse: float  = np.inf

    for params in candidates:
        score = _cv_rmse(name, params, X_train, y_train, folds, gpu)
        if score < best_rmse:
            best_rmse, best_params = score, params

    return best_params, best_rmse


# ── Single-model train + evaluate + save ───────────────────────────────────────

def train_single_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    threshold: str,
    gpu: bool = GPU_AVAILABLE,
    models_dir: Path = MODELS_DIR,
    skip_tuning: bool = False,
) -> dict:
    """
    Full lifecycle for one model × one poverty threshold:
      1. Tune hyperparameters via CV
      2. Train on full training set with best params
      3. Evaluate on test set (RMSE, MAE, R², MAPE)
      4. Save to models/{name}_{slug}_approach_a.pkl

    Returns a metrics record dict (JSON-safe + '_model_obj' / '_test_pred' internal keys).
    """
    t0 = time.time()
    n_features = X_train.shape[1]

    # ── 1. Hyperparameter tuning ──
    candidates = PARAM_GRIDS.get(name, [{}])
    if skip_tuning or len(candidates) == 1:
        best_params = candidates[0]
        best_cv_rmse = np.nan
    else:
        n_cands = len(candidates)
        n_folds = len(folds)
        print(f"    Tuning ({n_cands} candidates × {n_folds} folds)…", end=" ", flush=True)
        best_params, best_cv_rmse = tune_hyperparams(name, X_train, y_train, folds, gpu)
        print(f"CV RMSE={best_cv_rmse:.4f}")
    print(f"    Best params: {best_params}")

    # ── 2. Fit on full training set (skip NaN targets) ──
    valid_tr = ~np.isnan(y_train)
    model = _build_model(name, best_params, n_features, gpu)
    model.fit(X_train[valid_tr], y_train[valid_tr])

    # ── 3. Evaluate ──
    train_pred = np.clip(model.predict(X_train), 0, 100)
    test_pred  = np.clip(model.predict(X_test),  0, 100)

    train_m = compute_metrics(y_train, train_pred)
    test_m  = compute_metrics(y_test,  test_pred)

    elapsed = time.time() - t0
    gpu_used = (name == "xgboost_gpu" and gpu)

    # ── 4. Save model bundle ──
    slug = THRESHOLD_SLUG[threshold]
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / f"{name}_{slug}_approach_a.pkl"
    with open(save_path, "wb") as fh:
        pickle.dump({
            "model":       model,
            "best_params": best_params,
            "threshold":   threshold,
            "gpu_used":    gpu_used,
            "feature_count": n_features,
        }, fh)

    return {
        # identifiers
        "model_name":    name,
        "threshold":     threshold,
        "gpu_used":      gpu_used,
        "best_params":   str(best_params),
        # train metrics (for overfit check)
        "train_rmse":    train_m["rmse"],
        "train_mae":     train_m["mae"],
        "train_r2":      train_m["r2"],
        # test metrics (primary)
        "rmse":          test_m["rmse"],
        "mae":           test_m["mae"],
        "r2":            test_m["r2"],
        "mape":          test_m["mape"],
        "cv_rmse":       best_cv_rmse,
        "elapsed_s":     round(elapsed, 1),
        "model_path":    str(save_path),
        # internal — stripped before CSV export
        "_model_obj":    model,
        "_test_pred":    test_pred,
    }


# ── Full 28-model pipeline ─────────────────────────────────────────────────────

def run_full_pipeline(
    final_dir:   Path = DATA_FINAL_DIR,
    models_dir:  Path = MODELS_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    thresholds:  list[str] | None = None,
    skip_tuning: bool = False,
) -> pd.DataFrame:
    """
    Train all 7 models × 4 poverty thresholds (28 models total).

    Parameters
    ----------
    skip_tuning : bool
        If True, use the first candidate in each PARAM_GRIDS list without CV search.
        Useful for a quick smoke-test run.

    Returns
    -------
    pd.DataFrame with one row per (model, threshold) and all metrics.
    """
    if thresholds is None:
        thresholds = ALL_THRESHOLDS

    # ── Load artefacts ──
    print("Loading training artefacts from data/final/…")
    X_train_df = pd.read_csv(final_dir / "X_train.csv")
    X_test_df  = pd.read_csv(final_dir / "X_test.csv")
    y_train_df = pd.read_csv(final_dir / "y_train.csv")
    y_test_df  = pd.read_csv(final_dir / "y_test.csv")

    X_train = X_train_df.values.astype(float)
    X_test  = X_test_df.values.astype(float)

    with open(final_dir / "cv_folds.json") as f:
        folds_raw = json.load(f)
    # Folds are stratified by country income quartile (from feature_engineering.py)
    folds: list[tuple[np.ndarray, np.ndarray]] = [
        (np.array(v["train_idx"]), np.array(v["val_idx"]))
        for v in folds_raw.values()
    ]

    # Banner
    n_candidates = {n: len(PARAM_GRIDS.get(n, [{}])) for n in MODEL_NAMES}
    print(f"\nGPU available : {GPU_AVAILABLE}")
    if not GPU_AVAILABLE:
        print("  NOTE: xgboost_gpu will run on CPU (device='cpu') — "
              "results will be identical to xgboost_cpu.")
    print(f"Thresholds    : {thresholds}")
    print(f"Models        : {MODEL_NAMES}")
    print(f"Candidates    : { {n: c for n, c in n_candidates.items()} }")
    print(f"CV folds      : {len(folds)}")
    print(f"Total fits    : {sum(n_candidates.values()) * len(folds) * len(thresholds)} "
          f"(tuning) + {len(MODEL_NAMES) * len(thresholds)} (final)\n")

    all_records: list[dict] = []

    for threshold in thresholds:
        col = THRESHOLD_COL_MAP[threshold]
        y_tr = y_train_df[col].values.astype(float)
        y_te = y_test_df[col].values.astype(float)

        n_tr_valid = int((~np.isnan(y_tr)).sum())
        n_te_valid = int((~np.isnan(y_te)).sum())

        print(f"\n{'━'*65}")
        print(f"  Threshold: {threshold}  (column: {col})")
        print(f"  Train labels: {n_tr_valid:,} non-NaN / {len(y_tr):,} rows")
        print(f"  Test  labels: {n_te_valid:,} non-NaN / {len(y_te):,} rows")
        print(f"{'━'*65}")

        for name in MODEL_NAMES:
            print(f"\n  ▶ {name}")
            result = train_single_model(
                name=name,
                X_train=X_train, y_train=y_tr,
                X_test=X_test,   y_test=y_te,
                folds=folds,
                threshold=threshold,
                gpu=GPU_AVAILABLE,
                models_dir=models_dir,
                skip_tuning=skip_tuning,
            )
            print(
                f"    Test  → RMSE={result['rmse']:.3f}  MAE={result['mae']:.3f}  "
                f"R²={result['r2']:.3f}  MAPE={result['mape']:.1f}%"
            )
            print(
                f"    Train → RMSE={result['train_rmse']:.3f}  "
                f"(overfit gap: {result['train_rmse'] - result['rmse']:.3f})  "
                f"[{result['elapsed_s']}s]"
            )

            # Strip internal objects before accumulating
            rec = {k: v for k, v in result.items() if not k.startswith("_")}
            all_records.append(rec)

    results_df = pd.DataFrame(all_records)

    # ── Save outputs ──
    outputs_dir.mkdir(parents=True, exist_ok=True)

    _export_csvs(results_df, outputs_dir)
    _print_summary(results_df)
    _plot_comparison(results_df, outputs_dir)

    return results_df


# ── CSV exports ────────────────────────────────────────────────────────────────

_INTERNAL_COLS = [
    "best_params", "gpu_used", "model_path",
    "train_rmse", "train_mae", "train_r2", "elapsed_s", "cv_rmse",
]


def _export_csvs(results_df: pd.DataFrame, outputs_dir: Path) -> None:
    """Write the two comparison CSV files."""
    # Primary: $3/day, test metrics only, sorted by RMSE
    primary = (
        results_df[results_df["threshold"] == "$3"]
        .drop(columns=_INTERNAL_COLS, errors="ignore")
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    primary.to_csv(outputs_dir / "model_comparison_approach_a.csv", index=False)
    print(f"\nSaved: outputs/model_comparison_approach_a.csv")

    # All thresholds (full detail)
    results_df.to_csv(outputs_dir / "model_comparison_by_threshold.csv", index=False)
    print(f"Saved: outputs/model_comparison_by_threshold.csv")


# ── Console summary ────────────────────────────────────────────────────────────

def _print_summary(results_df: pd.DataFrame) -> None:
    """Print a formatted comparison table for each threshold."""
    print("\n\n" + "═" * 72)
    print("  MODEL COMPARISON SUMMARY — Approach A (test set 2016–2022)")
    print("═" * 72)

    for threshold in ALL_THRESHOLDS:
        sub = (
            results_df[results_df["threshold"] == threshold]
            .sort_values("rmse")
            [["model_name", "rmse", "mae", "r2", "mape", "elapsed_s"]]
        )
        print(f"\n  ── {threshold}/day ──")
        header = f"  {'Model':<20}  {'RMSE':>7}  {'MAE':>7}  {'R²':>7}  {'MAPE%':>7}  {'Time(s)':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in sub.iterrows():
            print(
                f"  {row['model_name']:<20}  {row['rmse']:>7.3f}  "
                f"{row['mae']:>7.3f}  {row['r2']:>7.3f}  "
                f"{row['mape']:>7.1f}  {row['elapsed_s']:>8.1f}"
            )

    print("\n" + "═" * 72)


# ── Comparison plot ────────────────────────────────────────────────────────────

def _plot_comparison(results_df: pd.DataFrame, outputs_dir: Path) -> None:
    """
    Two-panel figure:
      Left:  bar chart of RMSE/MAE/R²/MAPE for $3/day (primary comparison)
      Right: heatmap of RMSE per model × threshold
    """
    prim = (
        results_df[results_df["threshold"] == "$3"]
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    names   = prim["model_name"].tolist()
    x       = np.arange(len(names))
    labels  = [n.replace("_", "\n") for n in names]

    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── 4 metric bar charts ──
    metric_specs = [
        ("rmse", "RMSE  (↓ better)", "#d62728", gs[0, 0]),
        ("mae",  "MAE   (↓ better)", "#ff7f0e", gs[0, 1]),
        ("r2",   "R²    (↑ better)", "#2ca02c", gs[1, 0]),
        ("mape", "MAPE% (↓ better)", "#9467bd", gs[1, 1]),
    ]
    for metric, title, color, cell in metric_specs:
        ax = fig.add_subplot(cell)
        vals = prim[metric].values
        bars = ax.bar(x, vals, color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.35, axis="y")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3f}" if metric != "mape" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

    # ── RMSE heatmap across thresholds ──
    ax_heat = fig.add_subplot(gs[:, 2])
    pivot = results_df.pivot_table(
        index="model_name", columns="threshold", values="rmse"
    ).reindex(index=names, columns=ALL_THRESHOLDS)

    im = ax_heat.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax_heat.set_xticks(range(len(ALL_THRESHOLDS)))
    ax_heat.set_xticklabels(ALL_THRESHOLDS, fontsize=8)
    ax_heat.set_yticks(range(len(names)))
    ax_heat.set_yticklabels(names, fontsize=8)
    ax_heat.set_title("RMSE heatmap\n(all thresholds)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="RMSE")

    # Annotate heatmap cells
    for r in range(len(names)):
        for c in range(len(ALL_THRESHOLDS)):
            v = pivot.values[r, c]
            if not np.isnan(v):
                ax_heat.text(c, r, f"{v:.2f}", ha="center", va="center",
                             fontsize=6.5, color="black")

    fig.suptitle(
        "Approach A — Model Comparison  ($3/day bars, all-threshold heatmap)\n"
        f"Test set: 2016–2022  |  GPU: {GPU_AVAILABLE}",
        fontsize=11, y=1.01,
    )

    save_path = outputs_dir / "model_comparison_approach_a.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: outputs/model_comparison_approach_a.png")


# ── Inference helpers ──────────────────────────────────────────────────────────

def load_model_bundle(
    model_name: str,
    threshold: str,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """
    Load a saved model bundle.
    Returns dict with keys: model, best_params, threshold, gpu_used, feature_count.
    """
    slug = THRESHOLD_SLUG[threshold]
    path = models_dir / f"{model_name}_{slug}_approach_a.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_ssp(
    model: Any,
    forecast_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run inference on an SSP forecast panel (pre-scaled features).
    Adds 'predicted_poverty' column clipped to [0, 100].
    """
    X = forecast_df[feature_cols].values
    out = forecast_df.copy()
    out["predicted_poverty"] = np.clip(model.predict(X), 0, 100)
    return out
