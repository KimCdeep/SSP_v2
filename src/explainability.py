"""
explainability.py
Comprehensive SHAP analysis for all 7 trained models × $3/day poverty threshold.

Explainer strategy
------------------
Model family         | Explainer              | Notes
---------------------|------------------------|-----------------------------------
xgboost_cpu/gpu      | shap.TreeExplainer     | Exact, fast
lightgbm             | shap.TreeExplainer     | Exact, fast
random_forest        | shap.TreeExplainer     | Exact, fast
ridge                | shap.LinearExplainer   | Exact, fast (linear coefficients)
gam                  | shap.KernelExplainer   | Approximate; also use GAM's native
                     |                        | .partial_dependence() which is exact
mlp                  | shap.KernelExplainer   | Approximate; slow — subsample to 200

Outputs (all under outputs/shap/)
------------------
shap_summary_{model}.png          — beeswarm: importance + direction
shap_importance_{model}.png       — bar: mean |SHAP|
shap_dependence_{model}_{feat}.png — top-3 features, scatter vs feature value
shap_waterfall_{country}_{model}.png  — best model only: per-country waterfall
shap_force_{country}_{model}.html     — best model only: interactive force plot
shap_interaction_{feat1}x{feat2}.png  — best tree model: pairwise interaction
shap_scenario_comparison_{country}.png — SHAP by scenario (SSP1 vs 4 vs 5)
gam_partial_dependence/pd_{feature}.png — GAM smooth functions (exact)
../shap_feature_importance_comparison.csv — cross-model importance ranking
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap

from config import (
    DATA_FINAL_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    RANDOM_STATE,
)
from feature_engineering import (
    FINAL_FEATURE_COLS, POVERTY_TARGET_COLS,
    temporal_split, build_Xy,
    add_gdp_per_capita, add_log_transforms,
    add_gdp_growth_5y, add_region_onehot,
    load_historical_population_from_excel,
)
from model_pipeline import (
    MODEL_NAMES, THRESHOLD_SLUG, ALL_THRESHOLDS,
)
from approach_b import prepare_forecast_features, apply_scaler
from utils import SCENARIO_COLORS


# ── Constants ──────────────────────────────────────────────────────────────────

PRIMARY_THRESHOLD  = "$3"
PRIMARY_COL        = "poverty_3"
SHAP_DIR           = OUTPUTS_DIR / "shap"
GAM_PD_DIR         = OUTPUTS_DIR / "gam_partial_dependence"

TREE_MODELS    = {"xgboost_cpu", "xgboost_gpu", "lightgbm", "random_forest"}
LINEAR_MODELS  = {"ridge"}
KERNEL_MODELS  = {"gam", "mlp"}

# Human-readable feature labels for plots
FEATURE_LABELS: dict[str, str] = {
    "log_gdp_pc":              "log(GDP per capita)",
    "log_population":          "log(Population)",
    "hdi":                     "Human Development Index",
    "control_of_corruption":   "Control of Corruption",
    "employment_agriculture":  "Employment in Agriculture (%)",
    "gini_coefficient":        "Gini Coefficient",
    "gdp_growth_5y":           "5-yr GDP Growth (log return)",
    "region_EAP":              "Region: East Asia & Pacific",
    "region_ECA":              "Region: Europe & Central Asia",
    "region_LAC":              "Region: Latin America & Caribbean",
    "region_MENA":             "Region: Middle East & N. Africa",
    "region_NAC":              "Region: North America",
    "region_SAS":              "Region: South Asia",
}

INTERESTING_COUNTRIES = {
    "high_income":  "Germany",
    "middle_income": "Brazil",
    "low_income":   "Nigeria",
}


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_model_bundle(model_name: str, threshold: str = PRIMARY_THRESHOLD) -> dict:
    slug = THRESHOLD_SLUG[threshold]
    path = MODELS_DIR / f"{model_name}_{slug}_approach_a.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_training_artefacts(final_dir: Path = DATA_FINAL_DIR) -> dict:
    """Load X_train, X_test, y_train, y_test, scaler, feature names."""
    X_train = pd.read_csv(final_dir / "X_train.csv")
    X_test  = pd.read_csv(final_dir / "X_test.csv")
    y_train = pd.read_csv(final_dir / "y_train.csv")
    y_test  = pd.read_csv(final_dir / "y_test.csv")
    with open(final_dir / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(final_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler":  scaler,  "feature_names": feature_names,
    }


def load_test_metadata(
    panel_path:  Path = DATA_PROCESSED_DIR / "historical_panel.csv",
    excel_path:  Path = None,
    final_dir:   Path = DATA_FINAL_DIR,
) -> pd.DataFrame:
    """
    Reconstruct country_name / year / poverty_3 labels for each row in X_test.csv
    by replaying the feature_engineering temporal split + dropna.

    Returns DataFrame with columns: [country_name, year, poverty_3, country_code]
    in the same row order as X_test.csv.
    """
    from config import DATA_RAW_DIR
    if excel_path is None:
        excel_path = DATA_RAW_DIR / "GDP(Forecast)_POP_SSP_1950_2100.xlsx"

    panel = pd.read_csv(panel_path)
    pop_df = load_historical_population_from_excel(excel_path)
    panel  = add_gdp_per_capita(panel, pop_df)
    panel  = add_log_transforms(panel)
    panel  = add_gdp_growth_5y(panel)
    panel  = add_region_onehot(panel)

    _, test_df = temporal_split(panel)

    # Replicate the dropna used in build_Xy
    feat_col  = FINAL_FEATURE_COLS
    test_clean = test_df.dropna(subset=feat_col).reset_index(drop=True)

    meta_cols = ["country_name", "country_code", "year", "poverty_3"]
    available = [c for c in meta_cols if c in test_clean.columns]
    return test_clean[available].reset_index(drop=True)


# ── SHAP explainer factory ─────────────────────────────────────────────────────

def get_explainer(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    n_background: int = 100,
    random_state: int = RANDOM_STATE,
) -> Any:
    """
    Return the appropriate SHAP explainer for a given model.

    Parameters
    ----------
    n_background : int
        Number of background samples for KernelExplainer (MLP, GAM).
        Larger = more accurate but slower.  100 is a good balance.
    """
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(model)

    if model_name in LINEAR_MODELS:
        # LinearExplainer needs background data to compute expected values
        background = shap.sample(X_train, n_background, random_state=random_state)
        return shap.LinearExplainer(model, background)

    # KernelExplainer: sample background via kmeans clustering
    print(f"    Building KernelExplainer background ({n_background} rows)…")
    background = shap.kmeans(X_train, k=min(n_background, len(X_train)))
    return shap.KernelExplainer(model.predict, background)


def compute_shap_values(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    X_train: np.ndarray,
    n_background: int = 100,
    kernel_subsample: int = 200,
) -> shap.Explanation:
    """
    Compute SHAP values for X_test.

    For KernelExplainer (MLP, GAM), X_test is subsampled to `kernel_subsample`
    rows to keep runtime manageable.

    Returns a shap.Explanation object with .values (n_samples × n_features).
    """
    explainer = get_explainer(model_name, model, X_train, n_background)

    if model_name in KERNEL_MODELS:
        print(f"    KernelExplainer: subsampling test set to {kernel_subsample} rows…")
        rng  = np.random.default_rng(RANDOM_STATE)
        idx  = rng.choice(len(X_test), size=min(kernel_subsample, len(X_test)),
                          replace=False)
        X_sub = X_test[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = explainer.shap_values(X_sub)
        # Wrap in Explanation for uniform downstream handling
        ev = explainer.expected_value
        explanation = shap.Explanation(
            values=sv,
            base_values=np.full(len(X_sub), float(ev)),
            data=X_sub,
        )
        return explanation, idx   # also return indices so caller can align metadata

    # Tree / Linear: run on full X_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = explainer(X_test)
    return explanation, np.arange(len(X_test))


# ── 1. Summary plot (beeswarm) ─────────────────────────────────────────────────

def plot_summary(
    explanation: shap.Explanation,
    feature_names: list[str],
    model_name: str,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
) -> None:
    """Beeswarm plot: feature importance + direction."""
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=labels,
        show=False,
        plot_size=None,
    )
    plt.title(
        f"SHAP Summary — {model_name}\nPoverty {threshold}/day",
        fontsize=10, pad=12,
    )
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"shap_summary_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── 2. Bar plot (mean |SHAP|) ──────────────────────────────────────────────────

def plot_importance_bar(
    explanation: shap.Explanation,
    feature_names: list[str],
    model_name: str,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
) -> pd.DataFrame:
    """Bar chart of mean absolute SHAP values; returns importance DataFrame."""
    mean_abs = np.abs(explanation.values).mean(axis=0)
    labels   = [FEATURE_LABELS.get(f, f) for f in feature_names]

    imp_df = pd.DataFrame({"feature": feature_names, "label": labels,
                           "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(imp_df)))
    bars    = ax.barh(imp_df["label"], imp_df["mean_abs_shap"],
                      color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=9)
    ax.set_title(
        f"Global Feature Importance — {model_name}\nPoverty {threshold}/day",
        fontsize=10,
    )
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    for bar, v in zip(bars, imp_df["mean_abs_shap"]):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"shap_importance_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return imp_df.sort_values("mean_abs_shap", ascending=False)


# ── 3. Dependence plots (top-3 features) ──────────────────────────────────────

def plot_dependence(
    explanation: shap.Explanation,
    feature_names: list[str],
    model_name: str,
    top_n: int = 3,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
) -> None:
    """
    Scatter of SHAP value vs feature value for the top-N most important features.
    Colour encodes a potential interaction feature (highest absolute correlation
    with the displayed SHAP values).
    """
    mean_abs   = np.abs(explanation.values).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    labels     = [FEATURE_LABELS.get(f, f) for f in feature_names]

    save_dir.mkdir(parents=True, exist_ok=True)
    for rank, fidx in enumerate(top_idx):
        feat    = feature_names[fidx]
        label   = labels[fidx]
        x_vals  = explanation.data[:, fidx]
        s_vals  = explanation.values[:, fidx]

        # Find best interaction feature (highest correlation of OTHER feature values
        # with these SHAP values)
        corrs = []
        for j in range(explanation.data.shape[1]):
            if j == fidx:
                corrs.append(0.0)
                continue
            try:
                c = np.corrcoef(explanation.data[:, j], s_vals)[0, 1]
                corrs.append(abs(c) if not np.isnan(c) else 0.0)
            except Exception:
                corrs.append(0.0)
        int_idx   = int(np.argmax(corrs))
        int_feat  = feature_names[int_idx]
        int_vals  = explanation.data[:, int_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(x_vals, s_vals, c=int_vals, cmap="coolwarm",
                        alpha=0.5, s=15, linewidths=0)
        plt.colorbar(sc, ax=ax,
                     label=FEATURE_LABELS.get(int_feat, int_feat))
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(f"SHAP value for {label}", fontsize=9)
        ax.set_title(
            f"SHAP Dependence: {label}\n"
            f"{model_name} | {threshold}/day  "
            f"(colour = {FEATURE_LABELS.get(int_feat, int_feat)})",
            fontsize=9,
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()

        safe_feat = feat.replace("/", "_")
        out = save_dir / f"shap_dependence_{model_name}_{safe_feat}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── 4. Waterfall plots (best model, specific countries) ───────────────────────

def plot_waterfall(
    explanation: shap.Explanation,
    feature_names: list[str],
    sample_indices: dict[str, int],   # {label: row_index_in_explanation}
    model_name: str,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
) -> None:
    """
    One waterfall chart per country showing feature contributions.
    sample_indices maps a descriptive label to a row index in explanation.
    """
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    save_dir.mkdir(parents=True, exist_ok=True)

    for country_label, row_idx in sample_indices.items():
        if row_idx >= len(explanation.values):
            print(f"  SKIP waterfall for {country_label}: row {row_idx} out of range")
            continue

        sv     = explanation.values[row_idx]
        base   = float(explanation.base_values[row_idx]) \
                 if explanation.base_values is not None else 0.0
        x_data = explanation.data[row_idx]

        # Sort by absolute SHAP for cleaner chart
        order  = np.argsort(np.abs(sv))[::-1]
        sv_ord = sv[order]
        lb_ord = [labels[i] for i in order]
        xv_ord = x_data[order]

        # Build waterfall manually for full control
        cumulative = [base]
        for s in sv_ord:
            cumulative.append(cumulative[-1] + s)
        pred = cumulative[-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ["#d62728" if s > 0 else "#2ca02c" for s in sv_ord]

        for i, (s, lb, xv, col) in enumerate(
            zip(sv_ord, lb_ord, xv_ord, bar_colors)
        ):
            bottom = cumulative[i]
            ax.barh(i, s, left=bottom, color=col, alpha=0.82,
                    edgecolor="white", linewidth=0.5, height=0.6)
            ax.text(
                bottom + s + (0.05 if s >= 0 else -0.05),
                i, f"{s:+.3f}  [{xv:.3f}]",
                va="center", ha="left" if s >= 0 else "right",
                fontsize=7,
            )

        ax.axvline(base, color="grey", linestyle="--", linewidth=1,
                   label=f"Base value = {base:.3f}")
        ax.axvline(pred, color="black", linestyle="-", linewidth=1.5,
                   label=f"Prediction = {pred:.3f}%")
        ax.set_yticks(range(len(lb_ord)))
        ax.set_yticklabels(lb_ord, fontsize=8)
        ax.set_xlabel("Poverty headcount contribution (percentage points)", fontsize=9)
        ax.set_title(
            f"SHAP Waterfall — {country_label}\n"
            f"{model_name} | {threshold}/day\n"
            f"Prediction: {pred:.2f}%  (base: {base:.2f}%)",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35, axis="x")
        plt.tight_layout()

        safe_name = country_label.replace(" ", "_").replace(",", "")
        out = save_dir / f"shap_waterfall_{safe_name}_{model_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── 5. Force plots (HTML) ──────────────────────────────────────────────────────

def save_force_plots(
    explanation: shap.Explanation,
    feature_names: list[str],
    sample_indices: dict[str, int],
    model_name: str,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
) -> None:
    """Save interactive HTML force plots for specific countries."""
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    save_dir.mkdir(parents=True, exist_ok=True)
    shap.initjs()

    for country_label, row_idx in sample_indices.items():
        if row_idx >= len(explanation.values):
            continue
        base = float(explanation.base_values[row_idx]) \
               if explanation.base_values is not None else 0.0
        try:
            fp = shap.force_plot(
                base_value=base,
                shap_values=explanation.values[row_idx],
                features=explanation.data[row_idx],
                feature_names=labels,
                show=False,
            )
            safe_name = country_label.replace(" ", "_").replace(",", "")
            out = save_dir / f"shap_force_{safe_name}_{model_name}.html"
            shap.save_html(str(out), fp)
            print(f"  Saved: {out.name}")
        except Exception as exc:
            print(f"  WARN: force plot failed for {country_label}: {exc}")


# ── 6. Interaction effects (tree models only) ─────────────────────────────────

def plot_interactions(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    feature_names: list[str],
    interaction_pairs: list[tuple[str, str]] | None = None,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
    max_rows: int = 500,
) -> None:
    """
    Pairwise SHAP interaction plots via shap.dependence_plot with interaction_index.
    Only supported for tree models (shap_interaction_values).

    interaction_pairs: list of (feature_a, feature_b) to plot.
    Default: (log_gdp_pc, gini_coefficient) and (log_gdp_pc, control_of_corruption).
    """
    if model_name not in TREE_MODELS:
        print(f"  Interaction plots only available for tree models. Skipping {model_name}.")
        return

    if interaction_pairs is None:
        interaction_pairs = [
            ("log_gdp_pc",            "gini_coefficient"),
            ("log_gdp_pc",            "control_of_corruption"),
            ("gini_coefficient",      "control_of_corruption"),
            ("employment_agriculture","log_gdp_pc"),
        ]

    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for speed
    rng   = np.random.default_rng(RANDOM_STATE)
    n     = min(max_rows, len(X_test))
    idx   = rng.choice(len(X_test), size=n, replace=False)
    X_sub = X_test[idx]

    print(f"  Computing interaction values ({n} rows)…")
    try:
        explainer    = shap.TreeExplainer(model)
        sv_interact  = explainer.shap_interaction_values(X_sub)
    except Exception as exc:
        print(f"  WARN: interaction values failed: {exc}")
        return

    for feat_a, feat_b in interaction_pairs:
        if feat_a not in feature_names or feat_b not in feature_names:
            continue
        fa_idx = feature_names.index(feat_a)
        fb_idx = feature_names.index(feat_b)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            fa_idx,
            sv_interact[:, :, fa_idx],   # SHAP values for feat_a
            X_sub,
            feature_names=labels,
            interaction_index=fb_idx,
            ax=ax,
            show=False,
        )
        ax.set_title(
            f"SHAP Interaction: {FEATURE_LABELS.get(feat_a, feat_a)}\n"
            f"coloured by {FEATURE_LABELS.get(feat_b, feat_b)}  |  {model_name}",
            fontsize=9,
        )
        plt.tight_layout()

        out = save_dir / f"shap_interaction_{model_name}_{feat_a}_x_{feat_b}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")


# ── 7. Scenario comparison via SHAP ───────────────────────────────────────────

def plot_scenario_shap_comparison(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    feature_names: list[str],
    forecast_panel_path: Path = DATA_PROCESSED_DIR / "ssp_forecast_panel.csv",
    scaler_path: Path = DATA_FINAL_DIR / "feature_scaler.pkl",
    country: str = "Nigeria",
    year_subset: list[int] | None = None,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = SHAP_DIR,
) -> None:
    """
    For a selected country, compute SHAP values for each SSP scenario at
    2025, 2035, 2050, 2075, 2100 and show stacked bar charts of feature
    contributions by scenario.

    This is the "money shot" for the report: it shows WHY the model predicts
    different poverty levels under SSP1 vs SSP4 vs SSP5.
    """
    if year_subset is None:
        year_subset = [2025, 2035, 2050, 2075, 2100]

    print(f"  Computing scenario SHAP for {country}…")

    # Load and prepare forecast features
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    panel = pd.read_csv(forecast_panel_path)
    df_feat = prepare_forecast_features(panel)
    df_feat = df_feat[
        (df_feat["country_name"] == country) &
        (df_feat["year"].isin(year_subset))
    ].copy()

    if df_feat.empty:
        print(f"  WARN: No forecast data for {country}. Skipping.")
        return

    # Drop rows with NaN features
    nan_mask = df_feat[feature_names].isna().any(axis=1)
    df_feat  = df_feat[~nan_mask].reset_index(drop=True)
    X_scaled = apply_scaler(df_feat, scaler, feature_names)

    # Compute SHAP values
    explainer = get_explainer(model_name, model, X_train, n_background=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name in KERNEL_MODELS:
            sv = explainer.shap_values(X_scaled)
            base_val = float(explainer.expected_value)
        else:
            expl = explainer(X_scaled)
            sv   = expl.values
            base_val = float(np.mean(expl.base_values))

    scenarios  = ["SSP1", "SSP4", "SSP5"]
    labels     = [FEATURE_LABELS.get(f, f) for f in feature_names]
    n_feats    = len(feature_names)
    n_years    = len(year_subset)
    n_scen     = len(scenarios)

    # Gather SHAP contributions: shape (n_scenarios, n_years, n_features)
    shap_grid = np.full((n_scen, n_years, n_feats), np.nan)
    pred_grid = np.full((n_scen, n_years), np.nan)

    for si, ssp in enumerate(scenarios):
        for yi, yr in enumerate(year_subset):
            mask = (df_feat["scenario"] == ssp) & (df_feat["year"] == yr)
            if mask.sum() == 0:
                continue
            row_idx = np.where(mask.values)[0][0]
            shap_grid[si, yi, :] = sv[row_idx]
            pred_grid[si, yi]    = base_val + sv[row_idx].sum()

    # Plot: one column per year, one row per scenario
    fig, axes = plt.subplots(
        n_scen, n_years,
        figsize=(4.5 * n_years, 4.5 * n_scen),
        sharey=True, sharex=False,
    )
    if n_scen == 1:
        axes = axes[np.newaxis, :]
    if n_years == 1:
        axes = axes[:, np.newaxis]

    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
               "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a"]

    feat_colors = {f: palette[i % len(palette)] for i, f in enumerate(feature_names)}

    for si, ssp in enumerate(scenarios):
        for yi, yr in enumerate(year_subset):
            ax = axes[si][yi]
            sv_row = shap_grid[si, yi, :]
            if np.isnan(sv_row).all():
                ax.set_visible(False)
                continue

            pred_val = pred_grid[si, yi]
            order    = np.argsort(np.abs(sv_row))[::-1][:8]  # top-8 features

            colors_bar = [feat_colors[feature_names[i]] for i in order]
            ax.barh(
                range(len(order)),
                sv_row[order],
                color=colors_bar, alpha=0.82, edgecolor="white", linewidth=0.4,
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(
                [FEATURE_LABELS.get(feature_names[i], feature_names[i])[:20]
                 for i in order],
                fontsize=6.5,
            )
            ax.set_title(
                f"{ssp} — {yr}\npred={pred_val:.1f}%",
                fontsize=8, fontweight="bold",
            )
            ax.grid(True, linestyle="--", alpha=0.3, axis="x")
            ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        f"SHAP Feature Contributions by Scenario — {country}\n"
        f"{model_name} | {threshold}/day  "
        f"(base value ≈ {base_val:.2f}%)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_country = country.replace(" ", "_")
    out = save_dir / f"shap_scenario_comparison_{safe_country}_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── 8. GAM partial dependence plots ───────────────────────────────────────────

def plot_gam_partial_dependence(
    model: Any,
    feature_names: list[str],
    X_train: np.ndarray,
    threshold: str = PRIMARY_THRESHOLD,
    save_dir: Path = GAM_PD_DIR,
) -> None:
    """
    Use pyGAM's built-in .partial_dependence() to plot the exact smooth function
    for each feature.  This is a unique advantage of GAMs: exact interpretability
    without the approximations of SHAP.

    Each plot shows:
      - The smooth function f_j(x_j) with 95% confidence interval
      - Rug plot (data density)
    """
    try:
        from pygam import LinearGAM
    except ImportError:
        print("  pyGAM not installed. Skipping GAM partial dependence plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    n_terms = len(feature_names)

    # One figure per feature
    for term_idx, feat in enumerate(feature_names):
        label = FEATURE_LABELS.get(feat, feat)
        try:
            XX = model.generate_X_grid(term=term_idx)
            pdep, confi = model.partial_dependence(
                term=term_idx, X=XX, width=0.95,
            )
        except Exception as exc:
            print(f"  WARN: partial_dependence failed for {feat}: {exc}")
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(XX[:, term_idx], pdep, color="#1f77b4", linewidth=2,
                label="Smooth f(x)")
        ax.fill_between(
            XX[:, term_idx], confi[:, 0], confi[:, 1],
            alpha=0.25, color="#1f77b4", label="95% CI",
        )
        # Rug plot (data density)
        ax.plot(X_train[:, term_idx],
                np.full(len(X_train), pdep.min() - 0.05 * np.ptp(pdep)),
                "|", color="black", alpha=0.2, markersize=4)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(f"Partial effect on poverty {threshold}/day (pp)", fontsize=9)
        ax.set_title(
            f"GAM Smooth Function: {label}\n"
            f"{threshold}/day  (exact, not SHAP approximation)",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()

        safe_feat = feat.replace("/", "_")
        out = save_dir / f"pd_{safe_feat}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # Also produce an overview grid of all smooth functions
    n_cols = 4
    n_rows = int(np.ceil(n_terms / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for term_idx, feat in enumerate(feature_names):
        ax = axes[term_idx]
        label = FEATURE_LABELS.get(feat, feat)
        try:
            XX   = model.generate_X_grid(term=term_idx)
            pdep, confi = model.partial_dependence(
                term=term_idx, X=XX, width=0.95,
            )
            ax.plot(XX[:, term_idx], pdep, color="#1f77b4", linewidth=2)
            ax.fill_between(XX[:, term_idx], confi[:, 0], confi[:, 1],
                            alpha=0.25, color="#1f77b4")
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        except Exception:
            ax.set_visible(False)
        ax.set_title(label[:25], fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_terms, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"GAM Partial Dependence (all features) — {threshold}/day",
        fontsize=11,
    )
    plt.tight_layout()
    out_grid = save_dir / "pd_all_features_grid.png"
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_grid.name}")


# ── 9. Cross-model importance comparison ──────────────────────────────────────

def build_importance_comparison(
    importance_records: dict[str, pd.DataFrame],
    feature_names: list[str],
    save_path: Path = OUTPUTS_DIR / "shap_feature_importance_comparison.csv",
) -> pd.DataFrame:
    """
    Aggregate mean-|SHAP| importance rankings across all 7 models.
    Returns a wide DataFrame: rows = features, columns = models.
    Also saves a CSV and a comparison heatmap.
    """
    all_dfs = []
    for model_name, imp_df in importance_records.items():
        col = imp_df.set_index("feature")["mean_abs_shap"].rename(model_name)
        all_dfs.append(col)

    wide = pd.concat(all_dfs, axis=1).reindex(feature_names)
    wide["mean_across_models"] = wide.mean(axis=1)
    wide["rank_mean"] = wide["mean_across_models"].rank(ascending=False).astype(int)
    wide = wide.sort_values("mean_across_models", ascending=False)

    # Add human labels
    wide.insert(0, "feature_label",
                [FEATURE_LABELS.get(f, f) for f in wide.index])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(save_path)
    print(f"Saved: {save_path.name}")

    # ── Heatmap ──
    numeric_cols = [m for m in MODEL_NAMES if m in wide.columns]
    heatmap_data = wide[numeric_cols].copy()
    # Normalise per model so all columns are [0, 1] for visual comparison
    heatmap_norm = heatmap_data.div(heatmap_data.max(axis=0), axis=1)

    labels = [FEATURE_LABELS.get(f, f) for f in heatmap_norm.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        import seaborn as sns
        sns.heatmap(
            heatmap_norm, annot=heatmap_data.round(4), fmt=".4f",
            cmap="YlOrRd", linewidths=0.5, annot_kws={"size": 7},
            yticklabels=labels, ax=ax,
        )
    except ImportError:
        im = ax.imshow(heatmap_norm.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax)

    ax.set_title(
        "Feature Importance Comparison (mean |SHAP|, normalised per model)\n"
        "$3/day poverty threshold",
        fontsize=10,
    )
    plt.tight_layout()
    heat_path = OUTPUTS_DIR / "shap" / "shap_importance_comparison_heatmap.png"
    heat_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heat_path.name}")

    return wide


# ── 10. Rank agreement summary ─────────────────────────────────────────────────

def print_consensus_summary(importance_df: pd.DataFrame) -> None:
    """Print a readable cross-model feature importance consensus."""
    print("\n" + "═" * 70)
    print("  FEATURE IMPORTANCE CONSENSUS  ($3/day, mean |SHAP|)")
    print("═" * 70)
    model_cols = [m for m in MODEL_NAMES if m in importance_df.columns]

    print(f"\n  {'Feature':<40}  {'Mean':>7}  {'Rank':>5}  {'Min':>7}  {'Max':>7}")
    print("  " + "-" * 68)
    for feat, row in importance_df.iterrows():
        vals  = row[model_cols].dropna()
        mean_ = row["mean_across_models"]
        rank_ = int(row["rank_mean"])
        min_  = vals.min()
        max_  = vals.max()
        label = FEATURE_LABELS.get(feat, feat)[:38]
        print(f"  {label:<40}  {mean_:>7.4f}  {rank_:>5}  {min_:>7.4f}  {max_:>7.4f}")

    print("\n  Top-3 features across all models:")
    top3 = importance_df.head(3)
    for feat, row in top3.iterrows():
        label = FEATURE_LABELS.get(feat, feat)
        models_agree = (
            importance_df[model_cols].rank(ascending=False)
            .loc[feat].le(3).sum()
        )
        print(f"    {label}: ranked top-3 by {models_agree}/{len(model_cols)} models")
    print("═" * 70)


# ── 11. Full pipeline ──────────────────────────────────────────────────────────

def run_full_shap_analysis(
    final_dir:           Path = DATA_FINAL_DIR,
    models_dir:          Path = MODELS_DIR,
    outputs_dir:         Path = OUTPUTS_DIR,
    threshold:           str  = PRIMARY_THRESHOLD,
    best_model_override: str | None = None,
    kernel_subsample:    int  = 200,
    scenario_countries:  list[str] | None = None,
    waterfall_countries: dict[str, str] | None = None,
) -> dict:
    """
    Full SHAP analysis pipeline for all 7 models.

    Steps
    -----
    1.  Load artefacts (X_train, X_test, scaler, feature names)
    2.  Reconstruct test metadata (country labels)
    3.  For each of 7 models:
        a.  Compute SHAP values
        b.  Summary beeswarm
        c.  Bar importance
        d.  Dependence plots (top-3)
    4.  For best model only:
        e.  Waterfall + force plots (3 countries)
        f.  Interaction effects
        g.  Scenario comparison (SSP1/4/5) for selected countries
    5.  GAM: partial dependence plots
    6.  Cross-model importance comparison table + heatmap
    7.  Print consensus summary

    Parameters
    ----------
    best_model_override : str, optional
        Force a specific model as "best" (e.g. "xgboost_cpu").
        If None, determined by lowest RMSE in model_comparison_approach_a.csv.
    """
    shap_dir = outputs_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    GAM_PD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load training artefacts ──
    print("Loading artefacts…")
    arts = load_training_artefacts(final_dir)
    X_train_np = arts["X_train"].values.astype(float)
    X_test_np  = arts["X_test"].values.astype(float)
    y_test     = arts["y_test"][PRIMARY_COL].values.astype(float)
    feat_names = arts["feature_names"]

    print(f"  X_train: {X_train_np.shape}  |  X_test: {X_test_np.shape}")
    print(f"  Features: {feat_names}")

    # ── Determine best model ──
    comparison_path = outputs_dir / "model_comparison_approach_a.csv"
    if best_model_override:
        best_model_name = best_model_override
    elif comparison_path.exists():
        comp_df = pd.read_csv(comparison_path)
        best_model_name = comp_df.sort_values("rmse").iloc[0]["model_name"]
    else:
        best_model_name = "xgboost_cpu"
        print(f"  NOTE: model_comparison_approach_a.csv not found. Defaulting to {best_model_name}.")
    print(f"  Best model: {best_model_name}")

    # ── Reconstruct test metadata for waterfall ──
    print("\nReconstructing test metadata…")
    try:
        test_meta = load_test_metadata()
        print(f"  Test metadata: {len(test_meta)} rows, "
              f"{test_meta['country_name'].nunique()} countries")
    except Exception as exc:
        print(f"  WARN: could not load test metadata: {exc}")
        test_meta = None

    # ── Per-model SHAP analysis ──
    importance_records: dict[str, pd.DataFrame] = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*60}")
        print(f"  [{model_name}]")

        try:
            bundle = load_model_bundle(model_name, threshold)
            model  = bundle["model"]
        except FileNotFoundError:
            print(f"  SKIP: model file not found.")
            continue

        # Compute SHAP values
        explanation, used_idx = compute_shap_values(
            model_name, model, X_test_np, X_train_np,
            n_background=100, kernel_subsample=kernel_subsample,
        )

        # Attach feature names to explanation
        explanation = shap.Explanation(
            values=explanation.values,
            base_values=explanation.base_values,
            data=explanation.data,
            feature_names=feat_names,
        )

        # 1. Summary beeswarm
        plot_summary(explanation, feat_names, model_name, threshold, shap_dir)

        # 2. Bar importance
        imp_df = plot_importance_bar(
            explanation, feat_names, model_name, threshold, shap_dir
        )
        importance_records[model_name] = imp_df

        # 3. Dependence plots (top-3)
        plot_dependence(explanation, feat_names, model_name, top_n=3,
                        threshold=threshold, save_dir=shap_dir)

        # 4–7: Best model only
        if model_name == best_model_name:
            print(f"\n  ── Best model extras ({model_name}) ──")

            # Build sample_indices for waterfall/force (country → row index)
            if waterfall_countries is None:
                wf_countries = INTERESTING_COUNTRIES
            else:
                wf_countries = waterfall_countries

            sample_idx = {}
            if test_meta is not None:
                for label, country in wf_countries.items():
                    mask = (test_meta["country_name"] == country) & \
                           test_meta.index.isin(used_idx)
                    hits = np.where(test_meta["country_name"] == country)[0]
                    hits = [h for h in hits if h in set(used_idx.tolist())]
                    if hits:
                        # Map from original test row index to explanation row index
                        orig_to_expl = {orig: expl for expl, orig in enumerate(used_idx)}
                        sample_idx[f"{country} ({label})"] = orig_to_expl[hits[0]]
            if not sample_idx and len(explanation.values) >= 3:
                sample_idx = {f"Row {i}": i for i in range(3)}

            # 4. Waterfall
            plot_waterfall(explanation, feat_names, sample_idx,
                           model_name, threshold, shap_dir)

            # 5. Force plots (HTML)
            save_force_plots(explanation, feat_names, sample_idx,
                             model_name, threshold, shap_dir)

            # 6. Interactions
            plot_interactions(model, model_name, X_test_np, feat_names,
                              threshold=threshold, save_dir=shap_dir)

            # 7. Scenario comparison
            if scenario_countries is None:
                sc_countries = ["Nigeria", "India", "Germany"]
            else:
                sc_countries = scenario_countries

            for country in sc_countries:
                try:
                    plot_scenario_shap_comparison(
                        model_name, model, X_train_np, feat_names,
                        country=country, threshold=threshold, save_dir=shap_dir,
                    )
                except Exception as exc:
                    print(f"  WARN: scenario SHAP failed for {country}: {exc}")

        # GAM partial dependence (GAM only)
        if model_name == "gam":
            print(f"\n  ── GAM partial dependence plots ──")
            plot_gam_partial_dependence(
                model, feat_names, X_train_np,
                threshold=threshold, save_dir=GAM_PD_DIR,
            )

    # ── Cross-model importance comparison ──
    print(f"\n{'─'*60}")
    print("  Cross-model importance comparison…")
    if importance_records:
        imp_wide = build_importance_comparison(
            importance_records, feat_names,
            save_path=outputs_dir / "shap_feature_importance_comparison.csv",
        )
        print_consensus_summary(imp_wide)
    else:
        imp_wide = pd.DataFrame()

    # ── Final file listing ──
    all_shap_files = sorted(shap_dir.rglob("*.png")) + \
                     sorted(shap_dir.rglob("*.html"))
    gam_files      = sorted(GAM_PD_DIR.rglob("*.png"))
    print(f"\nTotal SHAP plots:        {len(all_shap_files)}")
    print(f"GAM partial dep plots:   {len(gam_files)}")
    print(f"Importance comparison:   outputs/shap_feature_importance_comparison.csv")

    return {
        "importance_records": importance_records,
        "importance_comparison": imp_wide,
        "best_model": best_model_name,
    }
