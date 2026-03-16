"""
predict_ssp.py
Generate poverty predictions under SSP1, SSP4, SSP5 for all 4 poverty thresholds
using the best-performing model from Step 6 training.

Approach A  — years 2025–2050: all features have real SSP projections
Approach B  — years 2025–2100: some features are linearly extrapolated beyond
              their original source range (flagged via extrapolation_flag)

Usage
-----
    from predict_ssp import run_predictions
    run_predictions()

Outputs
-------
  data/final/poverty_predictions_ssp.csv          — main dashboard feed
  outputs/top10_countries_by_scenario.csv         — league tables
  outputs/prediction_summary_stats.csv            — aggregated statistics
  outputs/prediction_trajectories_{threshold}.png — overview charts per threshold
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import (
    DATA_FINAL_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    SSP_SCENARIOS,
)
from approach_b import prepare_forecast_features, apply_scaler
from model_pipeline import (
    MODEL_NAMES, ALL_THRESHOLDS, THRESHOLD_COL_MAP, THRESHOLD_SLUG,
)
from utils import SCENARIO_COLORS


# ── Constants ──────────────────────────────────────────────────────────────────

APPROACH_A_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
APPROACH_B_YEARS = [
    2025, 2030, 2035, 2040, 2045, 2050,
    2055, 2060, 2065, 2070, 2075, 2080,
    2085, 2090, 2095, 2100,
]

# Source-level extrapolation onset per feature (from forecast_loader.py)
_EXTRAP_ONSET: dict[str, int] = {
    "employment_agriculture": 2050,
    "hdi":                    2075,
    "control_of_corruption":  2099,
}

THRESHOLD_DISPLAY: dict[str, str] = {
    "$3":    "$3/day",
    "$4.20": "$4.20/day",
    "$8.30": "$8.30/day",
    "$10":   "$10/day",
}


# ── Best-model selection ───────────────────────────────────────────────────────

def select_best_models(
    outputs_dir: Path = OUTPUTS_DIR,
    fallback_model: str = "xgboost_cpu",
) -> dict[str, str]:
    """
    For each poverty threshold, select the model with the lowest test RMSE.

    Reads outputs/model_comparison_by_threshold.csv (written by run_full_pipeline).
    Falls back to fallback_model if the CSV is not yet available.

    Returns
    -------
    dict mapping threshold label → model name
    e.g. {"$3": "xgboost_cpu", "$4.20": "lightgbm", ...}
    """
    csv_path = outputs_dir / "model_comparison_by_threshold.csv"
    if not csv_path.exists():
        print(f"  NOTE: {csv_path.name} not found — using {fallback_model} for all thresholds.")
        return {t: fallback_model for t in ALL_THRESHOLDS}

    df = pd.read_csv(csv_path)
    best: dict[str, str] = {}
    for threshold in ALL_THRESHOLDS:
        sub = df[df["threshold"] == threshold]
        if sub.empty:
            best[threshold] = fallback_model
            continue
        winner = sub.sort_values("rmse").iloc[0]["model_name"]
        best[threshold] = winner

    return best


def load_model(model_name: str, threshold: str, models_dir: Path = MODELS_DIR) -> object:
    """Load a trained model bundle from disk."""
    slug = THRESHOLD_SLUG[threshold]
    path = models_dir / f"{model_name}_{slug}_approach_a.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"]


# ── Feature preparation ────────────────────────────────────────────────────────

def _build_extrapolation_flag(panel: pd.DataFrame) -> pd.Series:
    """
    True for any row where at least one feature is drawn from a linearly
    extrapolated source (beyond the original data coverage).

    Combines the three per-feature flags already stored in the forecast panel.
    """
    flag_cols = [
        "employment_agriculture_extrap",
        "hdi_extrap",
        "coc_extrap",
    ]
    present = [c for c in flag_cols if c in panel.columns]
    if not present:
        return pd.Series(False, index=panel.index)
    # Cast to bool in case they were loaded as 0/1 ints
    flags = panel[present].astype(bool)
    return flags.any(axis=1)


def prepare_all_features(
    panel: pd.DataFrame,
    scaler,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply the full feature engineering pipeline to the forecast panel and
    return (clean_rows_df, X_scaled).

    Rows with any NaN feature are dropped and reported.
    """
    df_feat = prepare_forecast_features(panel)

    nan_mask = df_feat[feature_names].isna().any(axis=1)
    n_dropped = nan_mask.sum()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} rows with NaN features "
              f"({n_dropped / len(df_feat) * 100:.1f}%)")

    df_clean = df_feat[~nan_mask].reset_index(drop=True)
    X_scaled = apply_scaler(df_clean, scaler, feature_names)
    return df_clean, X_scaled


# ── Prediction engine ─────────────────────────────────────────────────────────

def predict_for_threshold(
    model,
    df_clean: pd.DataFrame,
    X_scaled: np.ndarray,
    threshold: str,
    approach_a_years: list[int] = APPROACH_A_YEARS,
) -> pd.DataFrame:
    """
    Run inference for one threshold and annotate each row with:
      - predicted_poverty  (clipped to [0, 100])
      - approach           ("A" for year ≤ 2050, "B" for year > 2050)
      - extrapolation_flag (from source-level extrap flags)
    """
    raw_preds = model.predict(X_scaled).astype(float)
    clipped   = np.clip(raw_preds, 0.0, 100.0)

    extrap_flag = _build_extrapolation_flag(df_clean)

    result = df_clean[["country_name", "country_code", "scenario", "year"]].copy()
    result["poverty_threshold"]  = threshold
    result["predicted_poverty"]  = clipped
    result["predicted_poverty_raw"] = raw_preds          # kept for diagnostics
    result["approach"]           = np.where(
        result["year"].isin(approach_a_years), "A", "B"
    )
    result["extrapolation_flag"] = extrap_flag.values
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_predictions(
    forecast_panel_path: Path = DATA_PROCESSED_DIR / "ssp_forecast_panel.csv",
    scaler_path:         Path = DATA_FINAL_DIR     / "feature_scaler.pkl",
    feature_names_path:  Path = DATA_FINAL_DIR     / "feature_names.json",
    models_dir:          Path = MODELS_DIR,
    outputs_dir:         Path = OUTPUTS_DIR,
    final_dir:           Path = DATA_FINAL_DIR,
    thresholds:          list[str] | None = None,
    best_model_override: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Full prediction pipeline: load models → prepare features → infer → save.

    Parameters
    ----------
    best_model_override : dict, optional
        Force specific models per threshold, e.g.:
        {"$3": "xgboost_cpu", "$4.20": "lightgbm"}.
        If None, determined automatically from model_comparison_by_threshold.csv.

    Returns
    -------
    pd.DataFrame — full predictions table (all countries × scenarios × years × thresholds)
    """
    if thresholds is None:
        thresholds = ALL_THRESHOLDS

    # ── Load shared artefacts ──
    print("Loading artefacts…")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(feature_names_path) as f:
        feature_names: list[str] = json.load(f)

    panel = pd.read_csv(forecast_panel_path)
    panel = panel[panel["scenario"].isin(SSP_SCENARIOS)].copy()
    # Coerce extrap flags to bool (may be loaded as 0/1)
    for col in ["employment_agriculture_extrap", "hdi_extrap", "coc_extrap"]:
        if col in panel.columns:
            panel[col] = panel[col].astype(bool)

    print(f"  Forecast panel: {len(panel):,} rows  |  "
          f"{panel['country_name'].nunique()} countries  |  "
          f"Years: {panel['year'].min()}–{panel['year'].max()}")

    # ── Feature preparation (shared across thresholds) ──
    print("\nPreparing features…")
    df_clean, X_scaled = prepare_all_features(panel, scaler, feature_names)
    print(f"  Clean rows: {len(df_clean):,}  |  Features: {X_scaled.shape[1]}")

    # ── Select best models ──
    if best_model_override:
        best_models = {**select_best_models(outputs_dir), **best_model_override}
    else:
        best_models = select_best_models(outputs_dir)

    print("\nBest models per threshold:")
    for t, m in best_models.items():
        print(f"  {t:<8} → {m}")

    # ── Inference ──
    all_chunks: list[pd.DataFrame] = []

    for threshold in thresholds:
        model_name = best_models.get(threshold, "xgboost_cpu")
        print(f"\n  [{threshold}] → {model_name}")

        try:
            model = load_model(model_name, threshold, models_dir)
        except FileNotFoundError as exc:
            print(f"    ERROR: model file not found: {exc}")
            print(f"    Run 03_model_training.ipynb first.")
            continue

        chunk = predict_for_threshold(
            model, df_clean, X_scaled, threshold, APPROACH_A_YEARS,
        )
        chunk["model_name"] = model_name

        n_neg   = (chunk["predicted_poverty_raw"] < 0).sum()
        n_over  = (chunk["predicted_poverty_raw"] > 100).sum()
        n_extrap = chunk["extrapolation_flag"].sum()
        n_b     = (chunk["approach"] == "B").sum()

        print(f"    Rows: {len(chunk):,}  |  "
              f"Approach A: {(chunk['approach']=='A').sum():,}  |  "
              f"Approach B: {n_b:,}")
        print(f"    Clipped: {n_neg} negative, {n_over} over-100  |  "
              f"Extrapolation-flagged: {n_extrap:,}")

        all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError("No predictions generated. Check that model files exist.")

    predictions = pd.concat(all_chunks, ignore_index=True)

    # Drop raw preds and model_name from the final dashboard file
    dashboard_cols = [
        "country_name", "country_code", "scenario", "year",
        "poverty_threshold", "predicted_poverty",
        "approach", "extrapolation_flag",
    ]
    dashboard_df = predictions[dashboard_cols].copy()

    # ── Save main output ──
    final_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_dir / "poverty_predictions_ssp.csv"
    dashboard_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(dashboard_df):,} rows)")

    # ── Derived outputs ──
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _save_top10(dashboard_df, outputs_dir)
    _save_summary_stats(dashboard_df, outputs_dir)
    _plot_trajectories(dashboard_df, outputs_dir)
    _plot_global_heatmap(dashboard_df, outputs_dir)
    _print_prediction_report(predictions, best_models)

    return dashboard_df


# ── Derived outputs ────────────────────────────────────────────────────────────

def _save_top10(
    df: pd.DataFrame,
    outputs_dir: Path,
    highlight_years: list[int] | None = None,
) -> None:
    """
    For each SSP × threshold × highlight year, the 10 countries with the
    highest predicted poverty headcount.
    """
    if highlight_years is None:
        highlight_years = [2030, 2050, 2075, 2100]

    rows: list[dict] = []
    for scenario in SSP_SCENARIOS:
        for threshold in ALL_THRESHOLDS:
            for year in highlight_years:
                sub = df[
                    (df["scenario"] == scenario) &
                    (df["poverty_threshold"] == threshold) &
                    (df["year"] == year)
                ]
                if sub.empty:
                    continue
                top10 = (
                    sub.sort_values("predicted_poverty", ascending=False)
                    .head(10)[["country_name", "country_code",
                               "predicted_poverty", "extrapolation_flag"]]
                )
                for rank, (_, r) in enumerate(top10.iterrows(), start=1):
                    rows.append({
                        "scenario":           scenario,
                        "poverty_threshold":  threshold,
                        "year":               year,
                        "rank":               rank,
                        "country_name":       r["country_name"],
                        "country_code":       r["country_code"],
                        "predicted_poverty":  round(r["predicted_poverty"], 3),
                        "extrapolation_flag": r["extrapolation_flag"],
                    })

    top10_df = pd.DataFrame(rows)
    path = outputs_dir / "top10_countries_by_scenario.csv"
    top10_df.to_csv(path, index=False)
    print(f"Saved: {path.name}  ({len(top10_df)} rows)")


def _save_summary_stats(df: pd.DataFrame, outputs_dir: Path) -> None:
    """
    Mean / median / std / p10 / p90 of predicted poverty
    grouped by scenario × threshold × year.
    """
    def p10(x):  return np.nanpercentile(x, 10)
    def p90(x):  return np.nanpercentile(x, 90)

    summary = (
        df.groupby(["scenario", "poverty_threshold", "year", "approach"])
        ["predicted_poverty"]
        .agg(
            mean   = "mean",
            median = "median",
            std    = "std",
            p10    = p10,
            p90    = p90,
            n      = "count",
        )
        .round(4)
        .reset_index()
    )
    path = outputs_dir / "prediction_summary_stats.csv"
    summary.to_csv(path, index=False)
    print(f"Saved: {path.name}  ({len(summary)} rows)")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_trajectories(
    df: pd.DataFrame,
    outputs_dir: Path,
    sample_countries: list[str] | None = None,
) -> None:
    """
    One figure per threshold: line charts for a selection of countries,
    colour = SSP scenario, shading past the 2050 approach-A boundary.
    """
    if sample_countries is None:
        sample_countries = ["Nigeria", "India", "Germany", "China", "Brazil",
                            "Ethiopia", "Bangladesh", "Indonesia"]
        # Keep only countries present in data
        available = df["country_name"].unique()
        sample_countries = [c for c in sample_countries if c in available][:6]

    for threshold in ALL_THRESHOLDS:
        sub = df[
            (df["poverty_threshold"] == threshold) &
            (df["country_name"].isin(sample_countries))
        ]
        if sub.empty:
            continue

        n_countries = len(sample_countries)
        n_cols      = 3
        n_rows      = int(np.ceil(n_countries / n_cols))
        fig, axes   = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 4 * n_rows),
            sharex=True,
        )
        axes = axes.flatten() if n_countries > 1 else [axes]

        for i, country in enumerate(sample_countries):
            ax  = axes[i]
            sub_c = sub[sub["country_name"] == country]
            for ssp in SSP_SCENARIOS:
                g = sub_c[sub_c["scenario"] == ssp].sort_values("year")
                if g.empty:
                    continue
                within = g[g["approach"] == "A"]
                beyond = g[g["approach"] == "B"]
                c = SCENARIO_COLORS.get(ssp, "grey")
                ax.plot(within["year"], within["predicted_poverty"],
                        color=c, linewidth=2, label=ssp)
                ax.plot(beyond["year"], beyond["predicted_poverty"],
                        color=c, linewidth=2, alpha=0.45, linestyle="--")

            ax.axvline(2050, color="black", linestyle=":", linewidth=1)
            ax.axvspan(2051, 2100, alpha=0.05, color="red")
            ax.set_title(country, fontsize=9, fontweight="bold")
            ax.set_xlabel("Year", fontsize=8)
            ax.set_ylabel("Poverty (%)", fontsize=8)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.35)

        # Hide unused axes
        for j in range(n_countries, len(axes)):
            axes[j].set_visible(False)

        # Shared legend
        handles = [
            plt.Line2D([0], [0], color=SCENARIO_COLORS[s], linewidth=2, label=s)
            for s in SSP_SCENARIOS
        ]
        handles += [
            plt.Line2D([0], [0], color="black", linewidth=1,
                       linestyle=":", label="Approach A limit (2050)"),
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))

        slug = threshold.replace("$", "").replace(".", "_")
        fig.suptitle(
            f"Predicted Poverty {THRESHOLD_DISPLAY[threshold]} — SSP Scenarios\n"
            "Solid = Approach A (≤2050), Dashed = Approach B (2051–2100)",
            fontsize=10,
        )
        plt.tight_layout()
        out = outputs_dir / f"prediction_trajectories_{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out.name}")


def _plot_global_heatmap(
    df: pd.DataFrame,
    outputs_dir: Path,
    highlight_years: list[int] | None = None,
) -> None:
    """
    For the primary $3 threshold: a 3-row (SSP) × N-year heatmap showing
    mean predicted poverty by region at each time step.
    Useful for the report to show global trends at a glance.
    """
    if highlight_years is None:
        highlight_years = [2030, 2050, 2075, 2100]

    from preprocessing import ISO3_TO_REGION  # region lookup

    sub = df[df["poverty_threshold"] == "$3"].copy()
    sub["region"] = sub["country_code"].map(ISO3_TO_REGION).fillna("Other")

    # Mean poverty per scenario × region × year
    pivot_data = (
        sub[sub["year"].isin(highlight_years)]
        .groupby(["scenario", "region", "year"])["predicted_poverty"]
        .mean()
        .reset_index()
    )

    scenarios  = SSP_SCENARIOS
    regions    = sorted(pivot_data["region"].unique())
    n_scen     = len(scenarios)
    n_years    = len(highlight_years)

    fig, axes = plt.subplots(
        n_scen, 1,
        figsize=(n_years * 2.5, 3.5 * n_scen),
        sharey=True,
    )
    if n_scen == 1:
        axes = [axes]

    for si, ssp in enumerate(scenarios):
        ax = axes[si]
        sub_s = pivot_data[pivot_data["scenario"] == ssp]
        heat  = sub_s.pivot_table(
            index="region", columns="year", values="predicted_poverty"
        ).reindex(regions, fill_value=np.nan)

        im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=50)
        ax.set_xticks(range(n_years))
        ax.set_xticklabels(highlight_years, fontsize=8)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels(regions, fontsize=8)
        ax.set_title(f"{ssp}", fontsize=9, fontweight="bold")

        for r in range(len(regions)):
            for c in range(n_years):
                v = heat.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > 30 else "black")

        plt.colorbar(im, ax=ax, shrink=0.7, label="Mean poverty %")

    fig.suptitle(
        "Mean Predicted Poverty $3/day — by Region × Year × Scenario",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    out = outputs_dir / "prediction_global_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


# ── Console report ─────────────────────────────────────────────────────────────

def _print_prediction_report(
    predictions: pd.DataFrame,
    best_models: dict[str, str],
) -> None:
    """Print a structured summary of the prediction run."""
    print("\n" + "═" * 68)
    print("  PREDICTION SUMMARY")
    print("═" * 68)

    print(f"\n  Best models used:")
    for t, m in best_models.items():
        print(f"    {t:<8} → {m}")

    print(f"\n  Coverage:")
    print(f"    Countries:  {predictions['country_name'].nunique()}")
    print(f"    Scenarios:  {predictions['scenario'].unique().tolist()}")
    print(f"    Thresholds: {predictions['poverty_threshold'].unique().tolist()}")
    print(f"    Years:      {predictions['year'].min()}–{predictions['year'].max()}")
    print(f"    Total rows: {len(predictions):,}")

    print(f"\n  Approach A (≤2050) rows: {(predictions['approach']=='A').sum():,}")
    print(f"  Approach B (>2050) rows: {(predictions['approach']=='B').sum():,}")
    print(f"  Extrapolation-flagged:   {predictions['extrapolation_flag'].sum():,}")
    n_clip = (predictions["predicted_poverty_raw"] < 0).sum() + \
             (predictions["predicted_poverty_raw"] > 100).sum()
    print(f"  Clipped predictions:     {n_clip:,}")

    print(f"\n  Global mean poverty (all countries, $3/day):")
    sub = predictions[predictions["poverty_threshold"] == "$3"]
    for ssp in SSP_SCENARIOS:
        by_yr = (
            sub[sub["scenario"] == ssp]
            .groupby("year")["predicted_poverty"]
            .mean()
        )
        yrs   = [2030, 2050, 2100]
        parts = "  ".join(
            f"{y}: {by_yr.get(y, np.nan):.1f}%"
            for y in yrs if y in by_yr.index
        )
        print(f"    {ssp}: {parts}")

    print("\n" + "═" * 68)
