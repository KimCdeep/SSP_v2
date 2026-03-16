"""
feature_engineering.py
Transforms processed/historical_panel.csv into ML-ready training data.

Pipeline:
  1. Load historical population from IIASA Excel (Historical Reference, 5-yr steps),
     interpolate to yearly 1993–2022.
  2. Compute gdp_per_capita  = GDP_USD / (population_millions × 1e6)
  3. log_gdp_pc              = log(gdp_per_capita)      [poverty is log-linear in income]
  4. log_population          = log(population_millions × 1e6)
  5. gdp_growth_5y           = log(gdp_pc_t) − log(gdp_pc_{t−5})
  6. One-hot encode WB region (7 regions, SSA dropped as baseline)
  7. Temporal split: train 1996–2015 / test 2016–2022
  8. Fit StandardScaler on training features, transform both splits
  9. Save final/ outputs

Feature set (13 total):
  Continuous:  log_gdp_pc, log_population, hdi, control_of_corruption,
               employment_agriculture, gini_coefficient, gdp_growth_5y
  Region OHE:  region_EAP, region_ECA, region_LAC, region_MENA, region_NAC, region_SAS
               (SSA = all zeros, kept as implicit baseline)

Targets (4):  poverty_3, poverty_4_20, poverty_8_30, poverty_10  [each a % in 0–100]

Note on GDP:  WB file is total GDP in current USD (NY.GDP.MKTP.CD), not PPP-adjusted.
              Cross-country variation >> within-country inflation, so log(gdp_pc)
              is still a strong signal.  A future improvement is to use
              NY.GDP.PCAP.PP.KD (PPP constant 2017 USD).
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_FINAL_DIR, RANDOM_STATE
)
from preprocessing import ISO3_TO_REGION
from utils import SSP_NAME_TO_ISO3

# ── Configuration ──────────────────────────────────────────────────────────────

TRAIN_END_YEAR   = 2015
TEST_START_YEAR  = 2016
N_CV_FOLDS       = 5
MIN_TRAIN_YEAR   = 1996   # skip early years sparse on governance/HDI data

POVERTY_TARGET_COLS = ["poverty_3", "poverty_4_20", "poverty_8_30", "poverty_10"]

FINAL_FEATURE_COLS = [
    "log_gdp_pc",
    "log_population",
    "hdi",
    "control_of_corruption",
    "employment_agriculture",
    "gini_coefficient",
    "gdp_growth_5y",
    # Region one-hot (SSA is baseline = all zeros)
    "region_EAP",
    "region_ECA",
    "region_LAC",
    "region_MENA",
    "region_NAC",
    "region_SAS",
]

# ── 1. Historical population extraction ───────────────────────────────────────

def load_historical_population_from_excel(
    excel_path: Path = DATA_RAW_DIR / "GDP(Forecast)_POP_SSP_1950_2100.xlsx",
    cache_path: Path = DATA_PROCESSED_DIR / "historical_population_iiasa.csv",
) -> pd.DataFrame:
    """
    Extract Historical Reference population from IIASA Excel, interpolate to
    yearly 1993–2022, and cache result as CSV.

    Returns DataFrame with columns: [country_name, country_code, year, population_M]
    (population in millions)
    """
    if cache_path.exists():
        print(f"Loading cached population: {cache_path}")
        return pd.read_csv(cache_path)

    print("Extracting population from IIASA Excel (first run — may take ~60 s)…")
    df = pd.read_excel(excel_path, sheet_name="data", engine="openpyxl")
    df.columns = [str(c) for c in df.columns]

    # Filter to Historical Reference + Population + non-aggregates
    _agg = ["(R5)", "(R9)", "(R10)", "World"]
    mask = (
        (df["Scenario"] == "Historical Reference") &
        (df["Variable"] == "Population") &
        (~df["Region"].apply(lambda r: any(m in str(r) for m in _agg)))
    )
    df = df[mask].copy()

    # Year columns available in the 5-yr historical period
    hist_yr_cols = [c for c in df.columns
                    if c.isdigit() and 1990 <= int(c) <= 2020]
    df = df[["Region"] + hist_yr_cols].rename(columns={"Region": "country_name"})

    # Melt to long format
    df = df.melt(id_vars="country_name", var_name="year", value_name="population_M")
    df["year"] = df["year"].astype(int)
    df["population_M"] = pd.to_numeric(df["population_M"], errors="coerce")

    # Interpolate 5-yr marks → yearly (1993–2022)
    yearly_rows = []
    for country, grp in df.groupby("country_name"):
        grp = grp.set_index("year").sort_index()

        # Build full yearly index 1990–2022
        full_idx = pd.RangeIndex(1990, 2023)
        s = grp["population_M"].reindex(full_idx)

        # Linear interpolate between 5-yr marks
        s = s.interpolate(method="linear", limit_direction="both")

        # For 2021–2022 (beyond last 5-yr mark 2020): forward-fill 2020 value
        s = s.ffill()

        for yr in range(1993, 2023):
            yearly_rows.append({
                "country_name": country,
                "year":         yr,
                "population_M": round(float(s.loc[yr]), 6) if yr in s.index else np.nan,
            })

    pop_df = pd.DataFrame(yearly_rows)
    pop_df["country_code"] = pop_df["country_name"].map(SSP_NAME_TO_ISO3)

    # Cache to CSV
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pop_df.to_csv(cache_path, index=False)
    print(f"Cached to: {cache_path}  ({pop_df['country_name'].nunique()} countries)")
    return pop_df


# ── 2. Derived features ────────────────────────────────────────────────────────

def add_gdp_per_capita(
    panel: pd.DataFrame,
    pop_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge population into panel and compute:
      gdp_per_capita = GDP_USD / (population_M × 1e6)   [USD / person]
    """
    pop_sub = pop_df[["country_name", "year", "population_M"]].drop_duplicates(
        subset=["country_name", "year"]
    )
    panel = panel.merge(pop_sub, on=["country_name", "year"], how="left")

    panel["gdp_per_capita"] = np.where(
        panel["population_M"].notna() & panel["population_M"] > 0,
        panel["gdp"] / (panel["population_M"] * 1e6),
        np.nan,
    )
    return panel


def add_log_transforms(panel: pd.DataFrame) -> pd.DataFrame:
    """
    log_gdp_pc    = log(gdp_per_capita)        — undefined / NaN if ≤ 0
    log_population = log(population_M × 1e6)
    """
    panel = panel.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel["log_gdp_pc"]     = np.log(panel["gdp_per_capita"].clip(lower=1e-9))
        panel["log_population"] = np.log((panel["population_M"] * 1e6).clip(lower=1e-9))
    return panel


def add_gdp_growth_5y(panel: pd.DataFrame) -> pd.DataFrame:
    """
    gdp_growth_5y = log(gdp_pc_t) − log(gdp_pc_{t−5})
    i.e. the 5-year log return in real GDP per capita.
    NaN for the first 5 years per country (filled with 0: no observable growth).
    """
    panel = panel.copy().sort_values(["country_name", "year"])
    panel["_log_gdp_lag5"] = (
        panel.groupby("country_name")["log_gdp_pc"]
        .transform(lambda s: s.shift(5))
    )
    panel["gdp_growth_5y"] = (panel["log_gdp_pc"] - panel["_log_gdp_lag5"]).fillna(0.0)
    panel = panel.drop(columns=["_log_gdp_lag5"])
    return panel


def add_region_onehot(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Map country_code → WB region, then one-hot encode.
    Regions: EAP, ECA, LAC, MENA, NAC, SAS, SSA.
    SSA is dropped (baseline = all region dummies = 0).
    Countries with unknown region → all dummies = 0 (treated as SSA baseline).
    """
    panel = panel.copy()
    panel["_region"] = panel["country_code"].map(ISO3_TO_REGION).fillna("SSA")

    dummies = pd.get_dummies(panel["_region"], prefix="region")
    # Ensure all expected region columns are present (even if a region has no countries)
    expected = ["region_EAP", "region_ECA", "region_LAC", "region_MENA",
                "region_NAC", "region_SAS", "region_SSA"]
    for col in expected:
        if col not in dummies.columns:
            dummies[col] = False
    dummies = dummies[expected].astype(int)

    # Drop SSA (baseline)
    dummies = dummies.drop(columns=["region_SSA"])
    panel = pd.concat([panel, dummies], axis=1)
    panel = panel.drop(columns=["_region"])
    return panel


# ── 3. Split ───────────────────────────────────────────────────────────────────

def temporal_split(
    panel: pd.DataFrame,
    train_end: int = TRAIN_END_YEAR,
    test_start: int = TEST_START_YEAR,
    min_train_year: int = MIN_TRAIN_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split on year.
      Train: [min_train_year, train_end]
      Test:  [test_start, max_year]
    """
    train = panel[
        (panel["year"] >= min_train_year) & (panel["year"] <= train_end)
    ].copy()
    test  = panel[panel["year"] >= test_start].copy()
    return train, test


def build_Xy(
    df: pd.DataFrame,
    feature_cols: list[str] = FINAL_FEATURE_COLS,
    target_cols:  list[str] = POVERTY_TARGET_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract (X, y) from a panel, dropping rows where ANY feature is NaN.
    Returns aligned (X, y) DataFrames with the same index.
    """
    combined = df[feature_cols + target_cols].dropna(subset=feature_cols)
    X = combined[feature_cols].reset_index(drop=True)
    y = combined[target_cols].reset_index(drop=True)
    return X, y


# ── 4. Cross-validation setup ─────────────────────────────────────────────────

def _country_income_strata(
    train_df: pd.DataFrame,
    target_col: str = "poverty_3",
    n_strata: int = 4,
) -> pd.Series:
    """
    Assign each row in train_df an income stratum (0–3) based on the country's
    median poverty rate. Ensures each CV fold sees a mix of rich/poor countries.
    """
    country_median = (
        train_df.groupby("country_name")[target_col]
        .median()
        .fillna(train_df[target_col].median())
    )
    labels = pd.qcut(country_median, q=n_strata, labels=False, duplicates="drop")
    strata_map = labels.to_dict()
    return train_df["country_name"].map(strata_map).fillna(0).astype(int)


def get_cv_folds(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return StratifiedKFold split indices (into X_train rows) stratified by
    country income group (derived from median poverty_3).

    Each fold contains a mix of high, middle, and low poverty countries.
    """
    strata = _country_income_strata(train_df.loc[X_train.index])
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X_train, strata))


# ── 5. Save outputs ────────────────────────────────────────────────────────────

def save_training_files(
    X_train:      pd.DataFrame,
    X_test:       pd.DataFrame,
    y_train:      pd.DataFrame,
    y_test:       pd.DataFrame,
    scaler:       StandardScaler,
    cv_folds:     list[tuple],
    train_meta:   dict,
    output_dir:   Path = DATA_FINAL_DIR,
) -> None:
    """
    Save all ML-ready artefacts to output_dir/.

    Files:
      X_train.csv / X_test.csv         — scaled feature matrices
      y_train.csv / y_test.csv         — 4 poverty threshold targets
      feature_scaler.pkl               — fitted StandardScaler
      feature_names.json               — ordered feature name list
      split_metadata.json              — train/test sizes, year ranges
      cv_folds.json                    — {fold_i: {train: [...], val: [...]}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv( output_dir / "X_test.csv",  index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv( output_dir / "y_test.csv",  index=False)

    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(X_train.columns.tolist(), f, indent=2)

    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(train_meta, f, indent=2)

    folds_serialisable = {
        str(i): {
            "train_idx": tr.tolist(),
            "val_idx":   va.tolist(),
        }
        for i, (tr, va) in enumerate(cv_folds)
    }
    with open(output_dir / "cv_folds.json", "w") as f:
        json.dump(folds_serialisable, f)

    print(f"\nSaved training artefacts to: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name:35s}  {p.stat().st_size:>8,} bytes")


# ── 6. Main pipeline ───────────────────────────────────────────────────────────

def build_training_dataset(
    panel_path:  Path = DATA_PROCESSED_DIR / "historical_panel.csv",
    excel_path:  Path = DATA_RAW_DIR       / "GDP(Forecast)_POP_SSP_1950_2100.xlsx",
    output_dir:  Path = DATA_FINAL_DIR,
) -> dict:
    """
    Full feature-engineering pipeline.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test, scaler, cv_folds
    """
    # ── Load panel ──
    print("Loading historical panel…")
    panel = pd.read_csv(panel_path)
    print(f"  Panel shape: {panel.shape}")

    # ── Population → gdp_per_capita ──
    pop_df = load_historical_population_from_excel(excel_path)
    panel  = add_gdp_per_capita(panel, pop_df)

    # ── Derived features ──
    panel = add_log_transforms(panel)
    panel = add_gdp_growth_5y(panel)
    panel = add_region_onehot(panel)

    # ── Temporal split ──
    train_df, test_df = temporal_split(panel)
    print(f"\nSplit:")
    print(f"  Train: years {MIN_TRAIN_YEAR}–{TRAIN_END_YEAR}  "
          f"({train_df['country_name'].nunique()} countries, {len(train_df):,} rows)")
    print(f"  Test:  years {TEST_START_YEAR}–{panel['year'].max()}  "
          f"({test_df['country_name'].nunique()} countries, {len(test_df):,} rows)")

    # ── Build X/y — drop rows missing any feature ──
    X_train_raw, y_train = build_Xy(train_df)
    X_test_raw,  y_test  = build_Xy(test_df)

    # Rows with valid features but NaN targets: keep in X (for inference),
    # but drop from y and mask for metric computation
    print(f"\n  X_train: {X_train_raw.shape}  |  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test_raw.shape}   |  y_test:  {y_test.shape}")

    # NaN in poverty targets (kept in X, NaN in y → models skip during scoring)
    for col in POVERTY_TARGET_COLS:
        n = y_train[col].isna().sum()
        print(f"    {col:<20}: {n:>4} NaN targets in train "
              f"({n / len(y_train) * 100:.1f}%)")

    # ── Scale features (fit on train only) ──
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
    )

    # ── CV folds (stratified by country income) ──
    cv_folds = get_cv_folds(train_df, X_train_raw)
    print(f"\n  {N_CV_FOLDS}-fold CV: {[len(va) for _,va in cv_folds]} val rows per fold")

    # ── Metadata ──
    meta = {
        "train_year_range":        [MIN_TRAIN_YEAR, TRAIN_END_YEAR],
        "test_year_range":         [TEST_START_YEAR, int(panel["year"].max())],
        "n_train_rows":            len(X_train_scaled),
        "n_test_rows":             len(X_test_scaled),
        "n_train_countries":       int(train_df["country_name"].nunique()),
        "n_test_countries":        int(test_df["country_name"].nunique()),
        "n_features":              len(FINAL_FEATURE_COLS),
        "feature_cols":            FINAL_FEATURE_COLS,
        "target_cols":             POVERTY_TARGET_COLS,
        "primary_target":          "poverty_3",
        "n_cv_folds":              N_CV_FOLDS,
        "scaler":                  "StandardScaler",
        "gdp_note":                "Total GDP in current USD (NY.GDP.MKTP.CD) / population",
        "population_source":       "IIASA Excel Historical Reference, interpolated from 5-yr steps",
    }

    save_training_files(
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, cv_folds, meta,
        output_dir,
    )

    return {
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
        "cv_folds": cv_folds,
        "panel":   panel,
        "meta":    meta,
    }
