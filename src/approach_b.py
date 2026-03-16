"""
approach_b.py
Extrapolation analysis: how do Approach A models behave on SSP forecast data
through 2100, when several input features are extrapolated beyond their
original source range?

Background
----------
Approach A trains models on historical data (1996–2015) and evaluates on
2016–2022.  All SSP features are present up to 2050 without source-level
extrapolation (except CoC at 2100).

Approach B applies the SAME trained models to the full 2025–2100 horizon.
Three features are extrapolated in the forecast panel beyond their source limits:

  Feature                    | Source limit | Extrapolated years
  ---------------------------|--------------|-------------------
  employment_agriculture     | 2050         | 2055–2100
  hdi                        | 2075         | 2080–2100
  control_of_corruption      | 2099         | 2100

Additionally, all forecast features may be out-of-distribution (OOD) relative
to the TRAINING range (1996–2015), even where source data is available.

Tree models (XGBoost, LightGBM, Random Forest) cannot extrapolate beyond the
training range — they will produce flat predictions once features cross the
training boundary.  Linear models and MLP can extrapolate but may produce
implausible values.

Workflow
--------
1.  Load ssp_forecast_panel.csv
2.  Compute the same 13 features used in training
3.  Apply the training StandardScaler (fitted on 1996–2015 data only)
4.  Run inference with all 28 saved models
5.  Flag OOD features and source-extrapolated rows
6.  Compare predictions: 2025–2050 (Approach A window) vs 2051–2100
7.  Save outputs

Outputs
-------
  outputs/approach_comparison.csv          — all predictions, both horizons
  outputs/approach_comparison_plots.png    — trajectory plots + divergence
  outputs/extrapolation_report.csv         — per-country OOD feature summary
  outputs/ood_summary.csv                  — fraction of forecast rows OOD per feature
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_FINAL_DIR, DATA_PROCESSED_DIR,
    MODELS_DIR, OUTPUTS_DIR,
    SSP_SCENARIOS,
)
from feature_engineering import FINAL_FEATURE_COLS, POVERTY_TARGET_COLS
from model_pipeline import (
    MODEL_NAMES, ALL_THRESHOLDS, THRESHOLD_COL_MAP, THRESHOLD_SLUG
)
from preprocessing import ISO3_TO_REGION
from utils import SCENARIO_COLORS


# ── Constants ──────────────────────────────────────────────────────────────────

# Source-level extrapolation boundaries (year at which source data ends)
SOURCE_EXTRAP_BOUNDARY: dict[str, int] = {
    "employment_agriculture": 2050,
    "hdi":                    2075,
    "control_of_corruption":  2099,
    # GDP, population, gini all have native 2100 coverage — no source extrapolation
}

# Approach A window: all features within their original source range
APPROACH_A_MAX_YEAR = 2050

REGION_COLS = [
    "region_EAP", "region_ECA", "region_LAC",
    "region_MENA", "region_NAC", "region_SAS",
]


# ── 1. Feature preparation ─────────────────────────────────────────────────────

def prepare_forecast_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive the 13 training features from the SSP forecast panel.

    Forecast gdp_per_capita is in thousand USD_2017 PPP (IIASA).
    We compute log(gdp_per_capita * 1000) to convert to raw USD scale
    before applying the historical StandardScaler.

    NOTE: The training scaler was fitted on log(historical current-USD gdp_pc),
    which differs from PPP-adjusted values.  This creates a systematic level shift
    in log_gdp_pc.  Within-scenario comparisons (A vs B) and relative rankings
    across models are still valid; absolute bias vs historical predictions is expected.

    Returns
    -------
    DataFrame with all FINAL_FEATURE_COLS columns, plus identifiers:
        country_name, country_code, scenario, year,
        employment_agriculture_extrap, hdi_extrap, coc_extrap
    """
    df = panel.copy().sort_values(["country_name", "scenario", "year"])

    # log_gdp_pc — convert k-USD_2017_PPP → USD then log
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdp_usd = df["gdp_per_capita"] * 1000          # thousand → raw USD/person
        df["log_gdp_pc"] = np.log(gdp_usd.clip(lower=1e-9))

    # log_population — population is already in millions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pop_abs = df["population"] * 1e6
        df["log_population"] = np.log(pop_abs.clip(lower=1e-9))

    # gdp_growth_5y — 5-year log return, per country × scenario
    df["gdp_growth_5y"] = (
        df.groupby(["country_name", "scenario"])["log_gdp_pc"]
        .transform(lambda s: s.diff(periods=1))   # 5-year steps → each step IS 5 years
        .fillna(0.0)
    )

    # Region one-hot (SSA = all zeros baseline)
    df["_region"] = df["country_code"].map(ISO3_TO_REGION).fillna("SSA")
    dummies = pd.get_dummies(df["_region"], prefix="region")
    expected = ["region_EAP", "region_ECA", "region_LAC",
                "region_MENA", "region_NAC", "region_SAS", "region_SSA"]
    for col in expected:
        if col not in dummies.columns:
            dummies[col] = False
    dummies = dummies[expected].astype(int).drop(columns=["region_SSA"])
    df = pd.concat([df.drop(columns=["_region"]), dummies], axis=1)

    return df


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str] = FINAL_FEATURE_COLS,
) -> np.ndarray:
    """Apply the training StandardScaler to forecast features. Returns ndarray."""
    X = df[feature_cols].values.astype(float)
    return scaler.transform(X)


# ── 2. OOD detection ───────────────────────────────────────────────────────────

def compute_ood_flags(
    X_forecast: np.ndarray,
    X_train_path: Path = DATA_FINAL_DIR / "X_train.csv",
    feature_cols: list[str] = FINAL_FEATURE_COLS,
) -> pd.DataFrame:
    """
    For each feature, flag forecast rows whose scaled value falls outside
    the training data range (min, max).

    Returns a DataFrame with columns: {feat}_ood for each feature, plus
    'any_ood' (True if at least one feature is OOD).
    """
    X_train = pd.read_csv(X_train_path).values.astype(float)
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)

    result = {}
    for i, feat in enumerate(feature_cols):
        col_vals = X_forecast[:, i]
        result[f"{feat}_ood"] = (col_vals < train_min[i]) | (col_vals > train_max[i])

    result_df = pd.DataFrame(result)
    result_df["any_ood"] = result_df.any(axis=1)
    return result_df, train_min, train_max


# ── 3. Run inference for all models × thresholds ──────────────────────────────

def generate_predictions(
    df_with_features: pd.DataFrame,
    X_scaled: np.ndarray,
    models_dir: Path = MODELS_DIR,
    model_names: list[str] = MODEL_NAMES,
    thresholds: list[str] = ALL_THRESHOLDS,
) -> pd.DataFrame:
    """
    Run all 28 saved Approach A models on the prepared forecast features.

    Returns a long-format DataFrame:
        country_name, country_code, scenario, year,
        employment_agriculture_extrap, hdi_extrap, coc_extrap,
        model_name, threshold, predicted_poverty, predicted_poverty_raw,
        is_negative, is_over100, is_unrealistic
    """
    id_cols = ["country_name", "country_code", "scenario", "year",
               "employment_agriculture_extrap", "hdi_extrap", "coc_extrap"]
    id_df = df_with_features[id_cols].reset_index(drop=True)

    records: list[pd.DataFrame] = []

    for threshold in thresholds:
        for model_name in model_names:
            slug = THRESHOLD_SLUG[threshold]
            pkl_path = models_dir / f"{model_name}_{slug}_approach_a.pkl"

            if not pkl_path.exists():
                print(f"  SKIP (not found): {pkl_path.name}")
                continue

            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
            model = bundle["model"]

            raw_preds = model.predict(X_scaled).astype(float)
            clipped   = np.clip(raw_preds, 0.0, 100.0)

            chunk = id_df.copy()
            chunk["model_name"]            = model_name
            chunk["threshold"]             = threshold
            chunk["predicted_poverty_raw"] = raw_preds
            chunk["predicted_poverty"]     = clipped
            chunk["is_negative"]           = raw_preds < 0
            chunk["is_over100"]            = raw_preds > 100
            chunk["is_unrealistic"]        = (raw_preds < 0) | (raw_preds > 100)
            records.append(chunk)

    return pd.concat(records, ignore_index=True)


# ── 4. Divergence analysis ─────────────────────────────────────────────────────

def compute_divergence(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (country, scenario, model, threshold), compute:
      - mean absolute divergence between prediction at year T and prediction at 2050
        (anchored comparison: how much does the trajectory shift after the anchor?)
      - rate of unrealistic predictions in the extrapolated window (2051–2100)

    Returns a summary DataFrame.
    """
    anchor_year = 2050
    extrap_mask = predictions_df["year"] > anchor_year

    # Anchor predictions at 2050
    anchor = (
        predictions_df[predictions_df["year"] == anchor_year]
        [["country_name", "scenario", "model_name", "threshold", "predicted_poverty"]]
        .rename(columns={"predicted_poverty": "anchor_poverty"})
    )

    extrap = predictions_df[extrap_mask].merge(
        anchor,
        on=["country_name", "scenario", "model_name", "threshold"],
        how="left",
    )
    extrap["divergence_from_2050"] = extrap["predicted_poverty"] - extrap["anchor_poverty"]
    extrap["abs_divergence"]       = extrap["divergence_from_2050"].abs()

    # Summarise per (model, threshold, year)
    summary = (
        extrap.groupby(["model_name", "threshold", "year"])
        .agg(
            mean_pred       = ("predicted_poverty", "mean"),
            mean_abs_div    = ("abs_divergence", "mean"),
            pct_unrealistic = ("is_unrealistic", "mean"),
        )
        .reset_index()
    )
    return summary, extrap


# ── 5. Plots ───────────────────────────────────────────────────────────────────

def plot_approach_comparison(
    predictions_df: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
    sample_countries: list[str] | None = None,
    sample_threshold: str = "$3",
    sample_models: list[str] | None = None,
) -> None:
    """
    Multi-panel figure:
      Panel A: Prediction trajectories for sample countries (best model, $3/day)
               with vertical dashed line at 2050 and shading for extrapolated zones.
      Panel B: Fraction of unrealistic predictions per year × model type.
      Panel C: Mean absolute divergence from 2050 anchor, by model family.
      Panel D: OOD fraction per feature per year.
    """
    if sample_countries is None:
        sample_countries = ["Nigeria", "India", "Germany", "China", "Brazil"]
    if sample_models is None:
        sample_models = ["xgboost_cpu", "ridge", "mlp", "gam"]

    fig = plt.figure(figsize=(20, 16))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    # ── A: Trajectories — best model, $3/day ──
    ax_traj = fig.add_subplot(gs[0, :])
    sub = predictions_df[
        (predictions_df["threshold"] == sample_threshold) &
        (predictions_df["model_name"] == "xgboost_cpu") &
        (predictions_df["country_name"].isin(sample_countries))
    ].copy()

    colors_c = plt.cm.tab10(np.linspace(0, 0.9, len(sample_countries)))
    for i, country in enumerate(sample_countries):
        for j, ssp in enumerate(SSP_SCENARIOS):
            s = sub[(sub["country_name"] == country) &
                    (sub["scenario"] == ssp)].sort_values("year")
            if s.empty:
                continue
            within = s[s["year"] <= 2050]
            beyond = s[s["year"] >= 2050]
            ls = ["-", "--", ":"][j]
            ax_traj.plot(within["year"], within["predicted_poverty"],
                         color=colors_c[i], linewidth=2, linestyle=ls,
                         label=f"{country} ({ssp})" if j == 0 else "_")
            ax_traj.plot(beyond["year"], beyond["predicted_poverty"],
                         color=colors_c[i], linewidth=2, linestyle=ls,
                         alpha=0.5)

    ax_traj.axvline(2050, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_traj.axvspan(2051, 2100, alpha=0.06, color="red", label="Extrapolated features")
    # Source-extrapolation boundaries
    ax_traj.axvline(2055, color="orange", linestyle=":", linewidth=1, alpha=0.8)
    ax_traj.axvline(2080, color="purple", linestyle=":", linewidth=1, alpha=0.8)

    ax_traj.set_title(
        f"Poverty Predictions — XGBoost CPU, {sample_threshold}/day\n"
        "Solid = within Approach A window (≤2050) | Faded = extrapolated (2051–2100)\n"
        "Orange dotted = Agri extrapolation starts (2055) | Purple = HDI (2080)",
        fontsize=9,
    )
    ax_traj.set_xlabel("Year"); ax_traj.set_ylabel("Poverty headcount (%)")
    ax_traj.legend(fontsize=7, loc="upper right", ncol=2)
    ax_traj.grid(True, linestyle="--", alpha=0.35)

    # ── B: Unrealistic prediction rate per model × year ──
    ax_unreal = fig.add_subplot(gs[1, 0])
    extrap_preds = predictions_df[
        (predictions_df["year"] > 2050) &
        (predictions_df["threshold"] == sample_threshold)
    ]
    unreal_by_model = (
        extrap_preds.groupby(["model_name", "year"])["is_unrealistic"]
        .mean()
        .reset_index()
    )
    model_colors = {m: f"C{i}" for i, m in enumerate(MODEL_NAMES)}
    for name, grp in unreal_by_model.groupby("model_name"):
        g = grp.sort_values("year")
        ax_unreal.plot(g["year"], g["is_unrealistic"] * 100,
                       label=name, color=model_colors.get(name),
                       linewidth=1.8, marker="o", markersize=3)
    ax_unreal.set_title(
        f"Unrealistic predictions (>100% or <0%)\n{sample_threshold}/day, years 2051–2100",
        fontsize=9,
    )
    ax_unreal.set_xlabel("Year"); ax_unreal.set_ylabel("% of country-scenario rows")
    ax_unreal.legend(fontsize=7)
    ax_unreal.grid(True, linestyle="--", alpha=0.35)

    # ── C: Mean absolute divergence from 2050 anchor ──
    ax_div = fig.add_subplot(gs[1, 1])
    _, extrap_df = compute_divergence(
        predictions_df[predictions_df["threshold"] == sample_threshold]
    )
    div_by_model = (
        extrap_df.groupby(["model_name", "year"])["abs_divergence"]
        .mean()
        .reset_index()
    )
    for name, grp in div_by_model.groupby("model_name"):
        g = grp.sort_values("year")
        ax_div.plot(g["year"], g["abs_divergence"],
                    label=name, color=model_colors.get(name),
                    linewidth=1.8, marker="o", markersize=3)
    ax_div.set_title(
        f"Mean absolute divergence from 2050 anchor\n{sample_threshold}/day",
        fontsize=9,
    )
    ax_div.set_xlabel("Year"); ax_div.set_ylabel("Mean |ΔPoverty| (pp)")
    ax_div.legend(fontsize=7)
    ax_div.grid(True, linestyle="--", alpha=0.35)

    # ── D: Model comparison at 2100 (bar chart) ──
    ax_bar2100 = fig.add_subplot(gs[2, 0])
    preds_2100 = predictions_df[
        (predictions_df["year"] == 2100) &
        (predictions_df["threshold"] == sample_threshold)
    ]
    mean_2100 = (
        preds_2100.groupby("model_name")["predicted_poverty"]
        .mean()
        .reindex(MODEL_NAMES)
    )
    std_2100 = (
        preds_2100.groupby("model_name")["predicted_poverty"]
        .std()
        .reindex(MODEL_NAMES)
    )
    x = np.arange(len(MODEL_NAMES))
    ax_bar2100.bar(x, mean_2100.values, yerr=std_2100.values,
                   color=[model_colors[m] for m in MODEL_NAMES],
                   alpha=0.75, capsize=4, edgecolor="white")
    ax_bar2100.set_xticks(x)
    ax_bar2100.set_xticklabels(
        [m.replace("_", "\n") for m in MODEL_NAMES], fontsize=7
    )
    ax_bar2100.set_title(
        f"Mean predicted poverty at 2100 (±1 SD)\n"
        f"across all countries × scenarios | {sample_threshold}/day",
        fontsize=9,
    )
    ax_bar2100.set_ylabel("Poverty headcount (%)")
    ax_bar2100.grid(True, linestyle="--", alpha=0.35, axis="y")

    # ── E: Cross-model range (disagreement) at each year ──
    ax_range = fig.add_subplot(gs[2, 1])
    for ssp in SSP_SCENARIOS:
        sub2 = predictions_df[
            (predictions_df["scenario"] == ssp) &
            (predictions_df["threshold"] == sample_threshold)
        ]
        model_spread = (
            sub2.groupby(["year", "country_name"])["predicted_poverty"]
            .agg(lambda x: x.max() - x.min())   # spread across models per country
            .groupby("year").mean()
        )
        ax_range.plot(model_spread.index, model_spread.values,
                      color=SCENARIO_COLORS.get(ssp, "grey"),
                      linewidth=2, label=ssp)
    ax_range.axvline(2050, color="black", linestyle="--", linewidth=1)
    ax_range.axvspan(2051, 2100, alpha=0.06, color="red")
    ax_range.set_title(
        f"Mean cross-model spread (max−min) per country\n{sample_threshold}/day",
        fontsize=9,
    )
    ax_range.set_xlabel("Year"); ax_range.set_ylabel("Poverty spread across models (pp)")
    ax_range.legend(title="Scenario"); ax_range.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        "Approach B: Model Behaviour in Extrapolated Feature Space (2025–2100)\n"
        "All models trained identically (Approach A); "
        "predictions extended to 2100 where features are extrapolated.",
        fontsize=11, y=1.01,
    )
    plt.savefig(outputs_dir / "approach_comparison_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/approach_comparison_plots.png")


def plot_ood_heatmap(
    ood_df: pd.DataFrame,
    df_with_features: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    For each forecast year, show what fraction of country-scenario rows
    have each feature outside the training range.
    """
    ood_feat_cols = [c for c in ood_df.columns if c.endswith("_ood") and c != "any_ood"]
    feat_labels   = [c.replace("_ood", "").replace("_", " ") for c in ood_feat_cols]

    combined = pd.concat([
        df_with_features[["year"]].reset_index(drop=True),
        ood_df[ood_feat_cols].reset_index(drop=True),
    ], axis=1)

    ood_by_year = combined.groupby("year")[ood_feat_cols].mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(ood_by_year.T.values * 100, aspect="auto",
                   cmap="Reds", vmin=0, vmax=100)

    ax.set_xticks(range(len(ood_by_year.index)))
    ax.set_xticklabels(ood_by_year.index.astype(str), rotation=45, fontsize=7)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_title(
        "% of country-scenario rows with feature value outside training range\n"
        "(training range = min/max of scaled X_train, 1996–2015)",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label="% rows OOD", shrink=0.7)

    # Annotate cells
    for r in range(len(feat_labels)):
        for c in range(len(ood_by_year.index)):
            v = ood_by_year.T.values[r, c] * 100
            if v > 5:
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        fontsize=5.5, color="black" if v < 60 else "white")

    plt.tight_layout()
    plt.savefig(outputs_dir / "ood_heatmap_approach_b.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: outputs/ood_heatmap_approach_b.png")


# ── 6. Report text ─────────────────────────────────────────────────────────────

EXTRAPOLATION_DISCUSSION = """
=======================================================================
APPROACH B: EXTRAPOLATION ANALYSIS — DISCUSSION POINTS FOR REPORT
=======================================================================

1. WHAT APPROACH B IS
   ---------------------------------------------------------
   Approach A and B use identical models (same training data, same
   hyperparameters).  The only difference is the prediction horizon:
   - Approach A:  2025–2050   (all features within source range)
   - Approach B:  2025–2100   (some features extrapolated post-source)

   This allows a controlled comparison of model behaviour inside vs
   outside the reliable feature space.

2. SOURCE-LEVEL EXTRAPOLATION BOUNDARIES
   ---------------------------------------------------------
   Feature                     | Original source limit | Extrapolated from
   employment_agriculture      | 2050                  | 2055
   hdi                         | 2075                  | 2080
   control_of_corruption       | 2099                  | 2100
   gdp_per_capita, population  | 2100                  | — (no extrapolation)
   gini_coefficient            | 2100                  | — (no extrapolation)

3. MODEL-LEVEL EXTRAPOLATION CONCERN (TRAINING RANGE)
   ---------------------------------------------------------
   Even where source data exists (e.g. Gini up to 2100), the predicted
   feature VALUES at 2070–2100 may fall outside the range observed in
   training data (1996–2015).  This affects ALL models:

   - Tree models (XGBoost, LightGBM, Random Forest):
     Predict the mean of the training leaf that best matches the input.
     If all features are beyond the training maximum, the model uses the
     last leaf — producing FLAT predictions after a certain year.
     This is visible as trajectories that stop changing despite
     continuing feature trends.

   - Linear models (Ridge):
     Will extrapolate the linear relationship beyond training range.
     Can produce poverty < 0% or implausible rapid declines.
     However, Ridge is regularised, which dampens extreme extrapolation.

   - MLP:
     Non-linear extrapolation depends on activation functions.  ReLU
     activations can produce linear extrapolation in extreme regions,
     but outputs are hard to predict without inspection.

   - GAM:
     Uses splines with basis functions that plateau at the knot boundary.
     Behaviour similar to tree models — flat predictions beyond the
     training data range for each smooth term.

4. WHAT THE OOD HEATMAP SHOWS
   ---------------------------------------------------------
   The OOD (out-of-distribution) heatmap shows, for each forecast year,
   what fraction of country-scenario rows have at least one feature
   outside the min/max range seen in training.  Key expected findings:
   - GDP per capita will be OOD for virtually all developed-country rows
     from the 2040s onward (continued growth exceeds 2015 levels).
   - HDI will be OOD for high-HDI countries by the 2060s.
   - Employment in agriculture will be OOD for countries that have
     already largely completed the structural transformation by 2030.

5. CROSS-MODEL DISAGREEMENT AS A RISK INDICATOR
   ---------------------------------------------------------
   When models disagree significantly (high cross-model spread), it
   indicates low confidence in the prediction.  Disagreement typically
   increases after 2050 as features enter extrapolated territory.
   For the dashboard, the cross-model spread can be used as an
   uncertainty band around the best-model prediction.

6. PRACTICAL GUIDANCE FOR SUPERVISOR PRESENTATION
   ---------------------------------------------------------
   - Frame Approach A (≤2050) as the RELIABLE window: features are
     within source range, models are within training distribution.
   - Frame Approach B (2051–2100) as a SCENARIO ANALYSIS, not a
     forecast: predictions show directional trends but with increasing
     uncertainty, especially for tree-based models.
   - Recommend using the linear/MLP model for 2051–2100 if extrapolation
     is needed, noting the clipping applied (0–100% headcount).
   - The cross-model disagreement band is a natural uncertainty measure.

=======================================================================
"""


# ── 7. Main pipeline ──────────────────────────────────────────────────────────

def run_approach_b(
    forecast_panel_path: Path  = DATA_PROCESSED_DIR / "ssp_forecast_panel.csv",
    scaler_path:         Path  = DATA_FINAL_DIR     / "feature_scaler.pkl",
    feature_names_path:  Path  = DATA_FINAL_DIR     / "feature_names.json",
    final_dir:           Path  = DATA_FINAL_DIR,
    models_dir:          Path  = MODELS_DIR,
    outputs_dir:         Path  = OUTPUTS_DIR,
    sample_countries:    list[str] | None = None,
) -> dict:
    """
    Full Approach B pipeline.

    Returns
    -------
    dict with keys: predictions, ood_flags, divergence_summary
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load forecast panel ──
    print("Loading SSP forecast panel…")
    panel = pd.read_csv(forecast_panel_path)
    panel["employment_agriculture_extrap"] = panel.get(
        "employment_agriculture_extrap", False
    )
    panel["hdi_extrap"]  = panel.get("hdi_extrap", False)
    panel["coc_extrap"]  = panel.get("coc_extrap", False)
    print(f"  Rows: {len(panel):,}  |  Countries: {panel['country_name'].nunique()}"
          f"  |  Years: {panel['year'].min()}–{panel['year'].max()}")

    # ── Load scaler and feature names ──
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    with open(feature_names_path) as f:
        feature_names: list[str] = json.load(f)

    print(f"  Feature names ({len(feature_names)}): {feature_names}")

    # ── Prepare features ──
    print("\nPreparing forecast features…")
    df_feat = prepare_forecast_features(panel)

    # Check all required feature columns are present
    missing = [c for c in feature_names if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing feature columns after preparation: {missing}")

    # Rows with any NaN feature
    nan_mask = df_feat[feature_names].isna().any(axis=1)
    print(f"  Rows with NaN features: {nan_mask.sum():,} "
          f"({nan_mask.sum()/len(df_feat)*100:.1f}%) — will be dropped from inference")
    df_clean  = df_feat[~nan_mask].reset_index(drop=True)
    X_scaled  = apply_scaler(df_clean, scaler, feature_names)

    # ── OOD detection ──
    print("\nComputing out-of-distribution flags…")
    ood_flags, train_min, train_max = compute_ood_flags(X_scaled, final_dir / "X_train.csv", feature_names)

    n_any_ood = ood_flags["any_ood"].sum()
    print(f"  Rows with ≥1 OOD feature: {n_any_ood:,} / {len(ood_flags):,} "
          f"({n_any_ood/len(ood_flags)*100:.1f}%)")
    print("  OOD rate per feature:")
    for feat in feature_names:
        col = f"{feat}_ood"
        r   = ood_flags[col].mean() * 100
        print(f"    {feat:<35} {r:>5.1f}%")

    # ── Inference ──
    print("\nRunning inference (28 models)…")
    predictions = generate_predictions(
        df_with_features=df_clean,
        X_scaled=X_scaled,
        models_dir=models_dir,
        model_names=MODEL_NAMES,
        thresholds=ALL_THRESHOLDS,
    )
    print(f"  Predictions shape: {predictions.shape}")

    # Add OOD flag
    ood_any = ood_flags["any_ood"].values
    # predictions has n_models * n_thresholds rows per original row — repeat flags
    n_rows_orig = len(df_clean)
    n_combos    = len(MODEL_NAMES) * len(ALL_THRESHOLDS)
    predictions["any_ood"] = np.tile(ood_any, n_combos)

    # Unrealistic summary
    n_unreal = predictions["is_unrealistic"].sum()
    print(f"  Unrealistic predictions: {n_unreal:,} "
          f"({n_unreal/len(predictions)*100:.1f}%) — clipped to [0, 100]")

    # ── Save main output ──
    out_csv = outputs_dir / "approach_comparison.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(predictions):,} rows)")

    # ── OOD summary CSV ──
    ood_feat_cols = [c for c in ood_flags.columns if c.endswith("_ood")]
    ood_by_year = pd.concat([
        df_clean[["year"]].reset_index(drop=True),
        ood_flags[ood_feat_cols + ["any_ood"]].reset_index(drop=True),
    ], axis=1)
    ood_summary = ood_by_year.groupby("year").mean().round(4)
    ood_summary.to_csv(outputs_dir / "ood_summary.csv")
    print(f"Saved: {outputs_dir / 'ood_summary.csv'}")

    # ── Extrapolation report: per-country, per-feature OOD onset year ──
    ood_full = pd.concat([
        df_clean[["country_name", "scenario", "year"]].reset_index(drop=True),
        ood_flags.reset_index(drop=True),
    ], axis=1)

    onset_rows = []
    for feat in feature_names:
        col = f"{feat}_ood"
        if col not in ood_full.columns:
            continue
        onset = (
            ood_full[ood_full[col]]
            .groupby(["country_name", "scenario"])["year"]
            .min()
            .reset_index()
            .rename(columns={"year": f"{feat}_ood_onset"})
        )
        onset_rows.append(onset)

    if onset_rows:
        from functools import reduce
        extrap_report = reduce(
            lambda l, r: l.merge(r, on=["country_name", "scenario"], how="outer"),
            onset_rows,
        )
        extrap_report.to_csv(outputs_dir / "extrapolation_report.csv", index=False)
        print(f"Saved: {outputs_dir / 'extrapolation_report.csv'}")

    # ── Divergence summary ──
    print("\nComputing divergence from 2050 anchor…")
    div_summary, _ = compute_divergence(
        predictions[predictions["threshold"] == "$3"]
    )
    div_summary.to_csv(outputs_dir / "divergence_summary.csv", index=False)
    print(f"Saved: {outputs_dir / 'divergence_summary.csv'}")

    # ── Plots ──
    print("\nGenerating comparison plots…")
    if sample_countries is None:
        # Pick countries with data across all models
        available = predictions["country_name"].unique().tolist()
        candidates = ["Nigeria", "India", "Germany", "China", "Brazil",
                      "Ethiopia", "United States", "Bangladesh"]
        sample_countries = [c for c in candidates if c in available][:5]

    plot_approach_comparison(
        predictions_df=predictions,
        outputs_dir=outputs_dir,
        sample_countries=sample_countries,
        sample_threshold="$3",
        sample_models=["xgboost_cpu", "ridge", "mlp", "gam"],
    )
    plot_ood_heatmap(
        ood_df=ood_flags,
        df_with_features=df_clean,
        outputs_dir=outputs_dir,
    )

    # ── Print discussion ──
    print(EXTRAPOLATION_DISCUSSION)

    return {
        "predictions":        predictions,
        "ood_flags":          ood_flags,
        "divergence_summary": div_summary,
        "ood_summary":        ood_summary,
    }
