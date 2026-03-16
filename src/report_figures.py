"""
report_figures.py
Generate all tables and figures required for the written report.

Each function is independent: load from outputs/ or data/final/, produce a
plot or CSV, save it, and return any underlying DataFrame for further use in
the notebook.

Functions
---------
fig1_model_performance_table   — Model comparison table (RMSE/MAE/R²/MAPE/time)
fig2_feature_importance_consensus — Grouped bar chart across all 7 models
fig3_prediction_trajectories   — Country-level SSP scenario trajectories
fig4_regional_trends           — Region-level poverty trends under each SSP
fig5_approach_divergence       — Approach A vs B split, per model family
fig6_learning_curve            — Train vs val RMSE as training size grows
generate_all_figures           — Run everything in order
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import (
    DATA_FINAL_DIR, DATA_PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR,
    RANDOM_STATE,
)
from preprocessing import ISO3_TO_REGION
from model_pipeline import (
    MODEL_NAMES, ALL_THRESHOLDS, THRESHOLD_SLUG,
    PARAM_GRIDS, _build_model, GPU_AVAILABLE,
)
from utils import SCENARIO_COLORS
from explainability import FEATURE_LABELS

# ── Styling constants ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.35,
    "figure.dpi":       150,
})

REGION_LABELS = {
    "EAP":  "East Asia & Pacific",
    "ECA":  "Europe & Central Asia",
    "LAC":  "Latin America & Caribbean",
    "MENA": "Middle East & N. Africa",
    "NAC":  "North America",
    "SAS":  "South Asia",
    "SSA":  "Sub-Saharan Africa",
}

THRESHOLD_DISPLAY = {
    "$3":    "$3/day",
    "$4.20": "$4.20/day",
    "$8.30": "$8.30/day",
    "$10":   "$10/day",
}

MODEL_DISPLAY = {
    "xgboost_cpu":  "XGBoost CPU",
    "xgboost_gpu":  "XGBoost GPU",
    "lightgbm":     "LightGBM",
    "random_forest": "Random Forest",
    "gam":          "GAM",
    "ridge":        "Ridge",
    "mlp":          "MLP",
}

REPORT_DIR = OUTPUTS_DIR / "report"


def _save(fig: plt.Figure, name: str, dpi: int = 180) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


def _require(path: Path, label: str) -> bool:
    if not path.exists():
        print(f"  SKIP {label}: {path.name} not found (run pipeline first).")
        return False
    return True


# ── Figure 1: Model performance table ─────────────────────────────────────────

def fig1_model_performance_table(
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Build two formatted tables:
      primary   — $3/day only: RMSE, MAE, R², MAPE, training time
      all_thresholds — RMSE for all 4 thresholds in one wide table

    Saves:
      report/model_performance_primary.csv
      report/model_performance_all_thresholds.csv
      report/fig1_model_performance.png
    """
    csv_path = outputs_dir / "model_comparison_by_threshold.csv"
    if not _require(csv_path, "fig1"):
        return {}

    df = pd.read_csv(csv_path)

    # ── Primary table: $3/day ──
    primary = (
        df[df["threshold"] == "$3"]
        .assign(model_label=lambda d: d["model_name"].map(MODEL_DISPLAY))
        .sort_values("rmse")
        [["model_label", "rmse", "mae", "r2", "mape", "elapsed_s",
          "train_rmse"]]
        .rename(columns={
            "model_label": "Model",
            "rmse":        "RMSE",
            "mae":         "MAE",
            "r2":          "R²",
            "mape":        "MAPE (%)",
            "elapsed_s":   "Time (s)",
            "train_rmse":  "Train RMSE",
        })
        .round(4)
        .reset_index(drop=True)
    )

    # ── All-thresholds table: RMSE only, wide format ──
    wide = (
        df[["model_name", "threshold", "rmse"]]
        .pivot(index="model_name", columns="threshold", values="rmse")
        .reindex(index=MODEL_NAMES, columns=ALL_THRESHOLDS)
        .rename(index=MODEL_DISPLAY)
        .round(4)
    )
    wide.columns.name = "RMSE"
    wide["Mean RMSE"] = wide.mean(axis=1).round(4)
    wide = wide.sort_values("Mean RMSE")

    # ── Save CSVs ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    primary.to_csv(REPORT_DIR / "model_performance_primary.csv", index=False)
    wide.to_csv(REPORT_DIR / "model_performance_all_thresholds.csv")
    print("  Saved: model_performance_primary.csv")
    print("  Saved: model_performance_all_thresholds.csv")

    # ── Figure: side-by-side bar charts (RMSE + R²) ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("RMSE",     "RMSE (pp)",  "#d62728", True),
        ("MAE",      "MAE (pp)",   "#ff7f0e", True),
        ("R²",       "R²",         "#2ca02c", False),
    ]
    models_sorted = primary["Model"].tolist()
    x = np.arange(len(models_sorted))

    for ax, (col, ylabel, color, lower_better) in zip(axes, metrics):
        vals = primary.set_index("Model")[col].reindex(models_sorted).values
        bars = ax.bar(x, vals, color=color, alpha=0.80,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace(" ", "\n") for m in models_sorted], fontsize=7.5
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"{col}  ({'↓ better' if lower_better else '↑ better'})",
            fontsize=9, fontweight="bold",
        )
        best_val = vals.min() if lower_better else vals.max()
        for bar, v in zip(bars, vals):
            color_ann = "white" if v == best_val else "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.5,
                f"{v:.3f}",
                ha="center", va="center",
                fontsize=7, color=color_ann, fontweight="bold",
            )
        if lower_better:
            ax.set_ylim(0, max(vals) * 1.15)
        else:
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Model Performance Comparison — $3/day Poverty Threshold  (Test set 2016–2022)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    _save(fig, "fig1_model_performance.png")

    # ── Supplementary: RMSE heatmap across thresholds ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    vals_heat = wide.drop(columns=["Mean RMSE"]).values
    im = ax2.imshow(vals_heat, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(ALL_THRESHOLDS)))
    ax2.set_xticklabels(ALL_THRESHOLDS, fontsize=9)
    ax2.set_yticks(range(len(wide)))
    ax2.set_yticklabels(wide.index.tolist(), fontsize=9)
    ax2.set_title("RMSE across all 4 poverty thresholds", fontsize=10)
    plt.colorbar(im, ax=ax2, shrink=0.7, label="RMSE (pp)")
    for r in range(len(wide)):
        for c in range(len(ALL_THRESHOLDS)):
            v = vals_heat[r, c]
            if not np.isnan(v):
                ax2.text(c, r, f"{v:.3f}", ha="center", va="center",
                         fontsize=7.5,
                         color="white" if v > vals_heat.max() * 0.6 else "black")
    plt.tight_layout()
    _save(fig2, "fig1b_rmse_heatmap_all_thresholds.png")

    return {"primary": primary, "wide": wide}


# ── Figure 2: Feature importance consensus ─────────────────────────────────────

def fig2_feature_importance_consensus(
    outputs_dir: Path = OUTPUTS_DIR,
) -> pd.DataFrame | None:
    """
    Grouped horizontal bar chart: mean |SHAP| per feature, one bar per model.
    Features sorted by mean-across-models importance.

    Saves: report/fig2_feature_importance_consensus.png
    """
    csv_path = outputs_dir / "shap_feature_importance_comparison.csv"
    if not _require(csv_path, "fig2"):
        return None

    imp = pd.read_csv(csv_path, index_col=0)
    model_cols  = [m for m in MODEL_NAMES if m in imp.columns]
    feat_labels = imp["feature_label"].values
    data        = imp[model_cols].values      # shape: n_features × n_models
    sorted_idx  = np.argsort(imp["mean_across_models"].values)  # ascending
    sorted_idx  = sorted_idx[::-1]            # top feature first

    n_feat   = len(sorted_idx)
    n_models = len(model_cols)
    bar_h    = 0.8 / n_models
    y_base   = np.arange(n_feat)

    palette  = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    fig, ax = plt.subplots(figsize=(12, max(6, n_feat * 0.7)))
    for mi, model_name in enumerate(model_cols):
        vals  = data[sorted_idx, mi]
        ypos  = y_base - (n_models / 2 - mi - 0.5) * bar_h
        label = MODEL_DISPLAY.get(model_name, model_name)
        ax.barh(ypos, vals, height=bar_h * 0.9,
                color=palette[mi], label=label, alpha=0.85)

    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [feat_labels[i] for i in sorted_idx], fontsize=8.5
    )
    ax.set_xlabel("Mean |SHAP value|  (average impact on output)", fontsize=9)
    ax.set_title(
        "Feature Importance Consensus — All 7 Models\n"
        "$3/day poverty threshold  |  sorted by cross-model mean",
        fontsize=10,
    )
    ax.legend(
        title="Model", fontsize=8, title_fontsize=8,
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    plt.tight_layout()
    _save(fig, "fig2_feature_importance_consensus.png", dpi=180)
    return imp


# ── Figure 3: Prediction trajectories per country ─────────────────────────────

def fig3_prediction_trajectories(
    final_dir: Path = DATA_FINAL_DIR,
    countries: list[str] | None = None,
    thresholds: list[str] | None = None,
) -> None:
    """
    For each country: one figure with 4 subplots (one per threshold).
    Each subplot has three lines (SSP1/4/5).
    Vertical dotted line at 2050; light red shading for Approach B.

    Saves: report/fig3_trajectory_{country}.png
    """
    csv_path = final_dir / "poverty_predictions_ssp.csv"
    if not _require(csv_path, "fig3"):
        return

    preds = pd.read_csv(csv_path)

    if countries is None:
        countries = ["Nigeria", "India", "Brazil", "Germany", "United States"]
    if thresholds is None:
        thresholds = ALL_THRESHOLDS

    available = set(preds["country_name"].unique())
    countries  = [c for c in countries if c in available]

    for country in countries:
        sub = preds[preds["country_name"] == country]

        fig, axes = plt.subplots(
            1, len(thresholds),
            figsize=(5 * len(thresholds), 4.5),
            sharey=False,
        )
        if len(thresholds) == 1:
            axes = [axes]

        for ax, threshold in zip(axes, thresholds):
            sub_t = sub[sub["poverty_threshold"] == threshold]
            for ssp in ["SSP1", "SSP4", "SSP5"]:
                g = sub_t[sub_t["scenario"] == ssp].sort_values("year")
                if g.empty:
                    continue
                c = SCENARIO_COLORS.get(ssp, "grey")
                within = g[g["approach"] == "A"]
                beyond = g[g["approach"] == "B"]
                ax.plot(within["year"], within["predicted_poverty"],
                        color=c, linewidth=2, label=ssp, zorder=3)
                ax.plot(beyond["year"], beyond["predicted_poverty"],
                        color=c, linewidth=2, alpha=0.4,
                        linestyle="--", zorder=2)

            ax.axvline(2050, color="black", linestyle=":", linewidth=1, zorder=4)
            ax.axvspan(2051, 2100, alpha=0.06, color="tomato", zorder=1)
            ax.set_title(THRESHOLD_DISPLAY[threshold], fontsize=9,
                         fontweight="bold")
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel("Poverty headcount (%)", fontsize=8)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
            )
            ax.tick_params(labelsize=7.5)

        # Shared legend on first subplot only
        handles = [
            plt.Line2D([0], [0], color=SCENARIO_COLORS[s], linewidth=2, label=s)
            for s in ["SSP1", "SSP4", "SSP5"]
        ]
        handles += [
            plt.Line2D([0], [0], color="black", linewidth=1,
                       linestyle=":", label="Approach A limit"),
            plt.Line2D([0], [0], color="tomato", linewidth=6,
                       alpha=0.4, label="Extrapolated (B)"),
        ]
        axes[0].legend(handles=handles, fontsize=7.5, loc="upper right")

        fig.suptitle(
            f"{country} — Predicted Poverty Headcount under SSP Scenarios",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        safe = country.replace(" ", "_").replace(".", "")
        _save(fig, f"fig3_trajectory_{safe}.png")


# ── Figure 4: Regional aggregation ────────────────────────────────────────────

def fig4_regional_trends(
    final_dir: Path = DATA_FINAL_DIR,
    threshold: str = "$3",
) -> pd.DataFrame | None:
    """
    For each World Bank region: mean predicted poverty over time,
    one line per SSP scenario, 7-panel figure.

    Saves: report/fig4_regional_trends.png
    """
    csv_path = final_dir / "poverty_predictions_ssp.csv"
    if not _require(csv_path, "fig4"):
        return None

    preds = pd.read_csv(csv_path)
    preds["region"] = preds["country_code"].map(ISO3_TO_REGION).fillna("Other")

    sub = preds[preds["poverty_threshold"] == threshold]
    agg = (
        sub.groupby(["region", "scenario", "year", "approach"])
        ["predicted_poverty"]
        .mean()
        .reset_index()
    )

    regions  = sorted([r for r in agg["region"].unique() if r != "Other"])
    n_cols   = 4
    n_rows   = int(np.ceil(len(regions) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4 * n_rows),
        sharey=False,
    )
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax     = axes[i]
        sub_r  = agg[agg["region"] == region]
        y_max  = sub_r["predicted_poverty"].max()

        for ssp in ["SSP1", "SSP4", "SSP5"]:
            g      = sub_r[sub_r["scenario"] == ssp].sort_values("year")
            c      = SCENARIO_COLORS.get(ssp, "grey")
            within = g[g["approach"] == "A"]
            beyond = g[g["approach"] == "B"]
            ax.plot(within["year"], within["predicted_poverty"],
                    color=c, linewidth=2, label=ssp)
            ax.plot(beyond["year"], beyond["predicted_poverty"],
                    color=c, linewidth=2, alpha=0.4, linestyle="--")

        ax.axvline(2050, color="black", linestyle=":", linewidth=1)
        ax.set_title(REGION_LABELS.get(region, region),
                     fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Year", fontsize=7.5)
        ax.set_ylabel("Mean poverty (%)", fontsize=7.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS[s], linewidth=2, label=s)
        for s in ["SSP1", "SSP4", "SSP5"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

    for j in range(len(regions), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Regional Poverty Trends — {THRESHOLD_DISPLAY[threshold]}\n"
        "Mean across countries in each World Bank region",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig4_regional_trends.png")
    return agg


# ── Figure 5: Approach A vs B divergence ──────────────────────────────────────

def fig5_approach_divergence(
    outputs_dir: Path = OUTPUTS_DIR,
    threshold: str = "$3",
    country: str = "Nigeria",
) -> None:
    """
    Two-panel figure showing how model predictions diverge after 2050:
      Panel A: All 7 models for a single country under SSP1 — solid (≤2050),
               dashed (>2050).  Tree models should flatten; linear/MLP continue.
      Panel B: Mean cross-model spread (max−min) per year across all countries,
               by scenario — a measure of forecast uncertainty.

    Saves: report/fig5_approach_divergence.png
    """
    csv_path = outputs_dir / "approach_comparison.csv"
    if not _require(csv_path, "fig5"):
        return

    preds = pd.read_csv(csv_path)
    preds = preds[preds["threshold"] == threshold]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: single country, all models, SSP1 ──
    ax = axes[0]
    sub_c = preds[
        (preds["country_name"] == country) &
        (preds["scenario"] == "SSP1")
    ]
    model_colors = {m: f"C{i}" for i, m in enumerate(MODEL_NAMES)}
    tree_models  = {"xgboost_cpu", "xgboost_gpu", "lightgbm", "random_forest"}

    for model_name in MODEL_NAMES:
        g = sub_c[sub_c["model_name"] == model_name].sort_values("year")
        if g.empty:
            continue
        c  = model_colors[model_name]
        ls = "-" if model_name in tree_models else "--"
        within = g[g["year"] <= 2050]
        beyond = g[g["year"] >= 2050]
        ax.plot(within["year"], within["predicted_poverty"],
                color=c, linewidth=2, linestyle=ls,
                label=MODEL_DISPLAY.get(model_name, model_name))
        ax.plot(beyond["year"], beyond["predicted_poverty"],
                color=c, linewidth=2, linestyle=ls, alpha=0.4)

    ax.axvline(2050, color="black", linestyle=":", linewidth=1.2,
               label="Approach A limit (2050)")
    ax.axvspan(2051, 2100, alpha=0.06, color="tomato")
    ax.set_title(
        f"{country} — SSP1, {THRESHOLD_DISPLAY[threshold]}\n"
        "Solid/dashed = tree/linear family  |  Faded = Approach B",
        fontsize=9,
    )
    ax.set_xlabel("Year"); ax.set_ylabel("Predicted poverty (%)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8)

    # ── Panel B: cross-model spread across all countries ──
    ax = axes[1]
    spread = (
        preds.groupby(["country_name", "scenario", "year"])["predicted_poverty"]
        .agg(lambda x: x.max() - x.min())
        .reset_index()
        .rename(columns={"predicted_poverty": "spread"})
    )
    mean_spread = spread.groupby(["scenario", "year"])["spread"].mean().reset_index()

    for ssp in ["SSP1", "SSP4", "SSP5"]:
        g = mean_spread[mean_spread["scenario"] == ssp].sort_values("year")
        ax.plot(g["year"], g["spread"],
                color=SCENARIO_COLORS.get(ssp, "grey"),
                linewidth=2, label=ssp)

    ax.axvline(2050, color="black", linestyle=":", linewidth=1.2)
    ax.axvspan(2051, 2100, alpha=0.06, color="tomato")
    ax.set_title(
        "Mean cross-model spread (max − min) per country\n"
        "Increasing spread = growing model uncertainty post-2050",
        fontsize=9,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean spread across 7 models (pp)")
    ax.legend(title="Scenario", fontsize=8)
    ax.tick_params(labelsize=8)

    fig.suptitle(
        "Approach A vs B: Model Behaviour Under Feature Extrapolation",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig5_approach_divergence.png")


# ── Figure 6: Learning curve ───────────────────────────────────────────────────

def fig6_learning_curve(
    final_dir: Path   = DATA_FINAL_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    threshold: str    = "$3",
    n_sizes:   int    = 8,
    cv_folds:  int    = 5,
) -> None:
    """
    Train-vs-validation RMSE as a function of training set size for the best
    model, using sklearn's learning_curve utility.

    Uses the best model hyperparameters from model_comparison_by_threshold.csv.

    Saves: report/fig6_learning_curve.png
    """
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import mean_squared_error, make_scorer

    X_train_path = final_dir / "X_train.csv"
    y_train_path = final_dir / "y_train.csv"
    if not _require(X_train_path, "fig6"):
        return

    X_train = pd.read_csv(X_train_path).values.astype(float)
    y_train = pd.read_csv(y_train_path)["poverty_3"].values.astype(float)

    # Drop NaN targets
    valid   = ~np.isnan(y_train)
    X_clean = X_train[valid]
    y_clean = y_train[valid]

    # Load best model hyperparameters
    comparison_path = outputs_dir / "model_comparison_by_threshold.csv"
    best_name, best_params_str = "xgboost_cpu", "{}"
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        sub  = comp[comp["threshold"] == threshold].sort_values("rmse")
        if not sub.empty:
            best_name       = sub.iloc[0]["model_name"]
            best_params_str = sub.iloc[0].get("best_params", "{}")

    # Parse params (stored as repr string)
    try:
        import ast
        best_params = ast.literal_eval(str(best_params_str))
    except Exception:
        best_params = PARAM_GRIDS.get(best_name, [{}])[0]

    print(f"  Learning curve for: {best_name}  params={best_params}")

    model = _build_model(best_name, best_params,
                         n_features=X_clean.shape[1],
                         gpu=False)  # always CPU for reproducibility

    rmse_scorer = make_scorer(
        lambda y_t, y_p: -np.sqrt(mean_squared_error(y_t, y_p)),
        greater_is_better=False,
    )
    # Use neg_root_mean_squared_error if available
    try:
        scorer = "neg_root_mean_squared_error"
        train_sizes_abs = np.linspace(0.1, 1.0, n_sizes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_clean, y_clean,
                train_sizes=train_sizes_abs,
                cv=cv_folds,
                scoring=scorer,
                n_jobs=-1,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
        train_rmse = -train_scores
        val_rmse   = -val_scores
    except Exception as exc:
        print(f"  WARN: learning_curve failed: {exc}")
        return

    train_mean = train_rmse.mean(axis=1)
    train_std  = train_rmse.std(axis=1)
    val_mean   = val_rmse.mean(axis=1)
    val_std    = val_rmse.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, color="#2ca02c", linewidth=2,
            marker="o", markersize=5, label="Train RMSE")
    ax.fill_between(train_sizes,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2ca02c")
    ax.plot(train_sizes, val_mean, color="#d62728", linewidth=2,
            marker="s", markersize=5, label=f"CV Val RMSE ({cv_folds}-fold)")
    ax.fill_between(train_sizes,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#d62728")

    ax.set_xlabel("Training set size (fraction)", fontsize=9)
    ax.set_ylabel("RMSE (percentage points)", fontsize=9)
    ax.set_title(
        f"Learning Curve — {MODEL_DISPLAY.get(best_name, best_name)}\n"
        f"$3/day threshold  |  {cv_folds}-fold CV  |  ±1 SD shaded",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _save(fig, "fig6_learning_curve.png")


# ── Bonus: Residual analysis ───────────────────────────────────────────────────

def fig7_residual_analysis(
    final_dir: Path   = DATA_FINAL_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    threshold: str    = "$3",
) -> None:
    """
    Residual diagnostics for the best model on the test set:
      - Residuals vs predicted values
      - Q-Q plot of residuals
      - Residuals by region (systematic bias check)

    Saves: report/fig7_residual_analysis.png
    """
    X_test_path = final_dir / "X_test.csv"
    y_test_path = final_dir / "y_test.csv"
    if not _require(X_test_path, "fig7"):
        return

    X_test = pd.read_csv(X_test_path).values.astype(float)
    y_test = pd.read_csv(y_test_path)["poverty_3"].values.astype(float)

    # Load best model
    comparison_path = outputs_dir / "model_comparison_by_threshold.csv"
    best_name = "xgboost_cpu"
    if comparison_path.exists():
        comp      = pd.read_csv(comparison_path)
        sub       = comp[comp["threshold"] == threshold].sort_values("rmse")
        best_name = sub.iloc[0]["model_name"] if not sub.empty else best_name

    slug     = THRESHOLD_SLUG[threshold]
    pkl_path = MODELS_DIR / f"{best_name}_{slug}_approach_a.pkl"
    if not _require(pkl_path, "fig7 model"):
        return

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    valid   = ~np.isnan(y_test)
    y_true  = y_test[valid]
    y_pred  = np.clip(model.predict(X_test[valid]), 0, 100)
    resid   = y_true - y_pred

    # Reconstruct region labels from feature matrix
    with open(final_dir / "feature_names.json") as f:
        feat_names = json.load(f)
    X_test_df = pd.read_csv(X_test_path)

    region_map = {
        "region_EAP": "EAP", "region_ECA": "ECA", "region_LAC": "LAC",
        "region_MENA": "MENA", "region_NAC": "NAC", "region_SAS": "SAS",
    }
    region_col = pd.Series("SSA", index=range(len(X_test)))
    for col, label in region_map.items():
        if col in X_test_df.columns:
            region_col[X_test_df[col].astype(bool)] = label
    region_col = region_col[valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: residuals vs predicted
    ax = axes[0]
    ax.scatter(y_pred, resid, alpha=0.3, s=10, color="#1f77b4")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    # LOESS-style trend (rolling mean)
    order  = np.argsort(y_pred)
    window = max(1, len(y_pred) // 20)
    smooth = pd.Series(resid[order]).rolling(window, center=True).mean()
    ax.plot(y_pred[order], smooth, color="#d62728", linewidth=2, label="Trend")
    ax.set_xlabel("Predicted poverty (%)", fontsize=9)
    ax.set_ylabel("Residual (actual − predicted, pp)", fontsize=9)
    ax.set_title("Residuals vs Predicted", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel B: Q-Q plot
    ax = axes[1]
    from scipy import stats as scipy_stats
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, alpha=0.4, s=10, color="#1f77b4")
    line_x = np.array([min(osm), max(osm)])
    ax.plot(line_x, slope * line_x + intercept, color="#d62728",
            linewidth=2, label=f"r={r:.3f}")
    ax.set_xlabel("Theoretical quantiles", fontsize=9)
    ax.set_ylabel("Sample quantiles", fontsize=9)
    ax.set_title("Q-Q Plot of Residuals", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel C: residual by region (box plot)
    ax = axes[2]
    region_resid = pd.DataFrame({"region": region_col.values, "residual": resid})
    region_order = sorted(region_resid["region"].unique())
    data_by_region = [
        region_resid[region_resid["region"] == r]["residual"].values
        for r in region_order
    ]
    bp = ax.boxplot(data_by_region, labels=region_order,
                    patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    colors_box = plt.cm.Set2(np.linspace(0, 1, len(region_order)))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticklabels(region_order, fontsize=8)
    ax.set_ylabel("Residual (pp)", fontsize=9)
    ax.set_title("Residuals by Region\n(bias check)", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Residual Analysis — {MODEL_DISPLAY.get(best_name, best_name)}"
        f"  |  {THRESHOLD_DISPLAY[threshold]}  |  Test set",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "fig7_residual_analysis.png")


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_all_figures(
    final_dir:   Path = DATA_FINAL_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    countries:   list[str] | None = None,
) -> dict:
    """Run all 7 report figures in sequence. Skips any with missing inputs."""
    print("=" * 65)
    print("  REPORT FIGURE GENERATION")
    print("=" * 65)

    results = {}

    print("\n[1/7] Model performance table…")
    results["fig1"] = fig1_model_performance_table(outputs_dir)

    print("\n[2/7] Feature importance consensus…")
    results["fig2"] = fig2_feature_importance_consensus(outputs_dir)

    print("\n[3/7] Prediction trajectories (per country)…")
    fig3_prediction_trajectories(final_dir, countries=countries)

    print("\n[4/7] Regional trends…")
    results["fig4"] = fig4_regional_trends(final_dir)

    print("\n[5/7] Approach A vs B divergence…")
    fig5_approach_divergence(outputs_dir)

    print("\n[6/7] Learning curve…")
    fig6_learning_curve(final_dir, outputs_dir)

    print("\n[7/7] Residual analysis…")
    fig7_residual_analysis(final_dir, outputs_dir)

    report_files = sorted(REPORT_DIR.rglob("*")) if REPORT_DIR.exists() else []
    print(f"\n{'=' * 65}")
    print(f"  Done. {len(report_files)} files in outputs/report/")
    for p in report_files:
        print(f"    {p.name:<55}  {p.stat().st_size:>8,} bytes")
    print("=" * 65)
    return results
