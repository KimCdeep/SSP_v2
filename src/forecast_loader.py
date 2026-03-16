"""
forecast_loader.py
Load, resample, extrapolate, harmonize, and merge all SSP forecast files.

Sources:
  Excel  — GDP(Forecast)_POP_SSP_1950_2100.xlsx  (IIASA, sheet='data')
           Variables: GDP|PPP [per capita], Population
  CSV 1  — ControlOfCorruption_Forecast_SSPExtensionExplorer_2015-2099.csv
           Yearly 2015–2099  → resample to 5-year steps, extrapolate 2100
  CSV 2  — EmploymentInAgriculture_Forecast_SSPExtensionExplorer_2016-2050.csv
           Mixed steps (yearly 2016–2030, 5-yr 2035–2050) → extrapolate 2055–2100
  CSV 3  — GiniCoefficient_Forecast_SSPExtensionExplorer_2015-2100.csv
           Already 5-year steps 2015–2100
  CSV 4  — HumanDevelopmentIndex_Forecast_SSPExtensionExplorer_2010-2075.csv
           5-year steps 2010–2075  → extrapolate 2080–2100

All sources use consistent IIASA/SSP country names.
Country codes are added via utils.SSP_NAME_TO_ISO3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SSP_SCENARIOS, FORECAST_YEARS_5Y
from utils import SSP_NAME_TO_ISO3

# ── Constants ──────────────────────────────────────────────────────────────────

# Patterns that identify regional aggregates (not countries) in the Excel file
_AGGREGATE_PATTERNS = [
    "(R5)", "(R9)", "(R10)", "World", "Historical Reference",
]

_FORECAST_YEAR_STRS = [str(y) for y in FORECAST_YEARS_5Y]   # ['2025', '2030', ...]


def _is_aggregate(name: str) -> bool:
    return any(p in str(name) for p in _AGGREGATE_PATTERNS)


# ── Extrapolation helper ───────────────────────────────────────────────────────

def _extrapolate_linear(
    series: pd.Series,
    anchor_years: list[int],
    target_years: list[int],
    n_anchor: int = 3,
    clip_range: tuple[float, float] | None = None,
) -> dict[int, float]:
    """
    Fit a linear trend to the last `n_anchor` points in `anchor_years` and
    extrapolate to `target_years`.

    Parameters
    ----------
    series      : pd.Series indexed by year (integers), containing known values
    anchor_years: full list of known years (sorted)
    target_years: years to extrapolate to
    n_anchor    : number of trailing anchor points to use for the trend
    clip_range  : optional (min, max) to clip extrapolated values

    Returns
    -------
    dict of {year: extrapolated_value}
    """
    fit_years = anchor_years[-n_anchor:]
    fit_vals  = series.reindex(fit_years).values.astype(float)

    # Drop NaNs from fitting
    mask = ~np.isnan(fit_vals)
    if mask.sum() < 2:
        # Not enough points — use last known value (flat extrapolation)
        last_val = series.dropna().iloc[-1] if not series.dropna().empty else np.nan
        result = {y: last_val for y in target_years}
    else:
        slope, intercept = np.polyfit(np.array(fit_years)[mask], fit_vals[mask], 1)
        result = {}
        for yr in target_years:
            val = float(slope * yr + intercept)
            if clip_range is not None:
                val = float(np.clip(val, clip_range[0], clip_range[1]))
            result[yr] = round(val, 6)

    return result


# ── 1. Excel: GDP per capita + Population ─────────────────────────────────────

def load_excel_gdp_pop(
    filepath: Path = DATA_RAW_DIR / "GDP(Forecast)_POP_SSP_1950_2100.xlsx",
    scenarios: list[str] = SSP_SCENARIOS,
    years: list[int] = FORECAST_YEARS_5Y,
) -> pd.DataFrame:
    """
    Load GDP|PPP [per capita] and Population from the IIASA Excel file.

    Units:
      GDP|PPP [per capita]  — thousand USD_2017 / person / year
      Population            — million persons

    NOTE: This file is ~59 MB; loading takes ~20–60 s with openpyxl engine.

    Returns
    -------
    DataFrame with columns:
        [country_name, scenario, year, gdp_per_capita, population]
    """
    print("Loading Excel GDP/Pop file (this may take ~30–60 s)…")
    df = pd.read_excel(filepath, sheet_name="data", engine="openpyxl")
    df.columns = [str(c) for c in df.columns]

    vars_needed = {"GDP|PPP [per capita]", "Population"}

    # Filter
    df = df[df["Scenario"].isin(scenarios)]
    df = df[df["Variable"].isin(vars_needed)]
    df = df[~df["Region"].apply(_is_aggregate)]

    year_cols = [str(y) for y in years if str(y) in df.columns]
    keep = ["Scenario", "Region", "Variable"] + year_cols
    df = df[keep].rename(columns={"Scenario": "scenario", "Region": "country_name"})

    df = df.melt(
        id_vars=["scenario", "country_name", "Variable"],
        var_name="year", value_name="value",
    )
    df["year"]  = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    var_map = {
        "GDP|PPP [per capita]": "gdp_per_capita",
        "Population":           "population",
    }
    df["col"] = df["Variable"].map(var_map)

    result = (
        df.pivot_table(
            index=["country_name", "scenario", "year"],
            columns="col",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    result.columns.name = None
    print(f"  GDP/Pop: {result['country_name'].nunique()} countries, "
          f"{result['scenario'].unique().tolist()}")
    return result


# ── 2. SSP Extension Explorer CSV loader (generic) ────────────────────────────

def _load_ssp_csv(filepath: Path, value_col: str) -> pd.DataFrame:
    """
    Load an SSP Extension Explorer CSV.
    Columns: model, scenario, region, unit, variable, <year cols>

    Returns long-format DataFrame:
        [country_name, scenario, year, <value_col>]
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.strip('"')

    year_cols = [c for c in df.columns if c.isdigit()]
    df = df.rename(columns={"scenario": "scenario", "region": "country_name"})
    id_cols = ["scenario", "country_name"]

    df = df[id_cols + year_cols].copy()
    df = df.melt(id_vars=id_cols, var_name="year", value_name=value_col)
    df["year"]      = df["year"].astype(int)
    df[value_col]   = pd.to_numeric(df[value_col], errors="coerce")
    df["scenario"]  = df["scenario"].str.strip('"')
    df["country_name"] = df["country_name"].str.strip('"')
    return df


def _filter_scenarios(df: pd.DataFrame, scenarios: list[str] = SSP_SCENARIOS) -> pd.DataFrame:
    return df[df["scenario"].isin(scenarios)].copy()


# ── 3. Control of Corruption ───────────────────────────────────────────────────

def load_forecast_corruption(
    filepath: Path  = DATA_RAW_DIR / "ControlOfCorruption_Forecast_SSPExtensionExplorer_2015-2099.csv",
    scenarios: list[str] = SSP_SCENARIOS,
    years: list[int] = FORECAST_YEARS_5Y,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Load ControlOfCorruption forecast.
    Source has yearly data 2015–2099.

    Strategy:
      - For years ≤ 2095: take value at the exact 5-year mark.
      - For year 2100 (not in source): extrapolate using linear trend
        of last 3 known years (2093, 2095, 2097 → use 2095-2099 window).

    Returns
    -------
    (df, extrapolated_years)
        df                  : [country_name, scenario, year, control_of_corruption]
        extrapolated_years  : set of years extrapolated (typically {2100})
    """
    df = _load_ssp_csv(filepath, "control_of_corruption")
    df = _filter_scenarios(df, scenarios)

    extrap_years = [y for y in years if y > 2099]   # only 2100
    keep_years   = [y for y in years if y <= 2099]

    # Filter to 5-year marks present in source
    df_5yr = df[df["year"].isin(keep_years)].copy()

    if extrap_years:
        records = []
        anchor_yrs = [y for y in range(2095, 2100)]  # 2095–2099
        for (country, scenario), grp in df[df["year"].isin(anchor_yrs)].groupby(
            ["country_name", "scenario"]
        ):
            s = grp.set_index("year")["control_of_corruption"]
            extrap = _extrapolate_linear(s, anchor_yrs, extrap_years, n_anchor=3)
            for yr, val in extrap.items():
                records.append(
                    {"country_name": country, "scenario": scenario,
                     "year": yr, "control_of_corruption": val}
                )
        df_5yr = pd.concat([df_5yr, pd.DataFrame(records)], ignore_index=True)

    return df_5yr.sort_values(["country_name", "scenario", "year"]).reset_index(drop=True), \
           set(extrap_years)


# ── 4. Employment in Agriculture ──────────────────────────────────────────────

def load_forecast_employment_agri(
    filepath: Path  = DATA_RAW_DIR / "EmploymentInAgriculture_Forecast_SSPExtensionExplorer_2016-2050.csv",
    scenarios: list[str] = SSP_SCENARIOS,
    years: list[int] = FORECAST_YEARS_5Y,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Load Employment in Agriculture forecast.
    Source has mixed steps: yearly 2016–2030, then 5-yr 2035–2050.
    Max year in source: 2050.

    Strategy:
      - Keep 5-year marks ≤ 2050 (2025, 2030, 2035, 2040, 2045, 2050).
      - Extrapolate 2055–2100 using linear trend of last 3 data points
        (2040, 2045, 2050), clipped to [0, 100] (% of workforce).

    Returns
    -------
    (df, extrapolated_years)
    """
    df = _load_ssp_csv(filepath, "employment_agriculture")
    df = _filter_scenarios(df, scenarios)

    source_5yr_marks = [2025, 2030, 2035, 2040, 2045, 2050]
    extrap_years     = [y for y in years if y > 2050]
    keep_years       = [y for y in years if y <= 2050]

    df_5yr = df[df["year"].isin(keep_years)].copy()

    # Extrapolate beyond 2050
    anchor_yrs = [2040, 2045, 2050]
    records = []
    for (country, scenario), grp in df[df["year"].isin(anchor_yrs)].groupby(
        ["country_name", "scenario"]
    ):
        s = grp.set_index("year")["employment_agriculture"]
        extrap = _extrapolate_linear(
            s, anchor_yrs, extrap_years, n_anchor=3, clip_range=(0.0, 100.0)
        )
        for yr, val in extrap.items():
            records.append(
                {"country_name": country, "scenario": scenario,
                 "year": yr, "employment_agriculture": val}
            )

    df_full = pd.concat([df_5yr, pd.DataFrame(records)], ignore_index=True)
    return df_full.sort_values(["country_name", "scenario", "year"]).reset_index(drop=True), \
           set(extrap_years)


# ── 5. Gini Coefficient ────────────────────────────────────────────────────────

def load_forecast_gini(
    filepath: Path  = DATA_RAW_DIR / "GiniCoefficient_Forecast_SSPExtensionExplorer_2015-2100.csv",
    scenarios: list[str] = SSP_SCENARIOS,
    years: list[int] = FORECAST_YEARS_5Y,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Load Gini coefficient forecast.
    Source already has 5-year steps 2015–2100 — no extrapolation needed.
    """
    df = _load_ssp_csv(filepath, "gini_coefficient")
    df = _filter_scenarios(df, scenarios)
    df = df[df["year"].isin(years)].copy()
    return df.sort_values(["country_name", "scenario", "year"]).reset_index(drop=True), set()


# ── 6. Human Development Index ────────────────────────────────────────────────

def load_forecast_hdi(
    filepath: Path  = DATA_RAW_DIR / "HumanDevelopmentIndex_Forecast_SSPExtensionExplorer_2010-2075.csv",
    scenarios: list[str] = SSP_SCENARIOS,
    years: list[int] = FORECAST_YEARS_5Y,
) -> tuple[pd.DataFrame, set[int]]:
    """
    Load HDI forecast.
    Source has 5-year steps 2010–2075. Max year: 2075.

    Strategy:
      - Keep 5-year marks ≤ 2075 (2025–2075).
      - Extrapolate 2080–2100 using linear trend of last 3 data points
        (2065, 2070, 2075), clipped to [0, 1].

    Returns
    -------
    (df, extrapolated_years)
    """
    df = _load_ssp_csv(filepath, "hdi")
    df = _filter_scenarios(df, scenarios)

    extrap_years = [y for y in years if y > 2075]
    keep_years   = [y for y in years if y <= 2075]

    df_5yr = df[df["year"].isin(keep_years)].copy()

    anchor_yrs = [2065, 2070, 2075]
    records = []
    for (country, scenario), grp in df[df["year"].isin(anchor_yrs)].groupby(
        ["country_name", "scenario"]
    ):
        s = grp.set_index("year")["hdi"]
        extrap = _extrapolate_linear(
            s, anchor_yrs, extrap_years, n_anchor=3, clip_range=(0.0, 1.0)
        )
        for yr, val in extrap.items():
            records.append(
                {"country_name": country, "scenario": scenario,
                 "year": yr, "hdi": val}
            )

    df_full = pd.concat([df_5yr, pd.DataFrame(records)], ignore_index=True)
    return df_full.sort_values(["country_name", "scenario", "year"]).reset_index(drop=True), \
           set(extrap_years)


# ── 7. Add ISO3 country codes ─────────────────────────────────────────────────

def add_iso3_codes(df: pd.DataFrame, name_col: str = "country_name") -> pd.DataFrame:
    """Map SSP country names to ISO3 codes via SSP_NAME_TO_ISO3."""
    df = df.copy()
    df["country_code"] = df[name_col].map(SSP_NAME_TO_ISO3)
    unmapped = df[df["country_code"].isna()][name_col].unique()
    if len(unmapped):
        print(f"  WARNING: {len(unmapped)} unmapped country names: {sorted(unmapped)}")
    return df


# ── 8. Merge all forecasts ─────────────────────────────────────────────────────

def build_forecast_panel(
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    scenarios: list[str] = SSP_SCENARIOS,
    years: list[int] = FORECAST_YEARS_5Y,
) -> pd.DataFrame:
    """
    Load, process, harmonize, and merge all SSP forecast sources.

    Returns a unified panel with columns:
        country_name, country_code, scenario, year,
        gdp_per_capita, population,
        hdi, control_of_corruption,
        employment_agriculture, gini_coefficient,
        employment_agriculture_extrap, hdi_extrap, coc_extrap
    """
    # ── Load each source ──
    gdp_pop_df                       = load_excel_gdp_pop(raw_dir / "GDP(Forecast)_POP_SSP_1950_2100.xlsx", scenarios, years)
    corruption_df, coc_extrap_yrs    = load_forecast_corruption(raw_dir / "ControlOfCorruption_Forecast_SSPExtensionExplorer_2015-2099.csv", scenarios, years)
    agri_df,       agri_extrap_yrs   = load_forecast_employment_agri(raw_dir / "EmploymentInAgriculture_Forecast_SSPExtensionExplorer_2016-2050.csv", scenarios, years)
    gini_df,       _                 = load_forecast_gini(raw_dir / "GiniCoefficient_Forecast_SSPExtensionExplorer_2015-2100.csv", scenarios, years)
    hdi_df,        hdi_extrap_yrs    = load_forecast_hdi(raw_dir / "HumanDevelopmentIndex_Forecast_SSPExtensionExplorer_2010-2075.csv", scenarios, years)

    # ── Add ISO3 codes to each ──
    for df in [gdp_pop_df, corruption_df, agri_df, gini_df, hdi_df]:
        df["country_code"] = df["country_name"].map(SSP_NAME_TO_ISO3)

    # ── Merge on country_name × scenario × year ──
    key = ["country_name", "country_code", "scenario", "year"]

    panel = gdp_pop_df.copy()

    for df, col in [
        (corruption_df, "control_of_corruption"),
        (agri_df,       "employment_agriculture"),
        (gini_df,       "gini_coefficient"),
        (hdi_df,        "hdi"),
    ]:
        sub = df[["country_name", "scenario", "year", col]].drop_duplicates(
            subset=["country_name", "scenario", "year"]
        )
        panel = panel.merge(sub, on=["country_name", "scenario", "year"], how="outer")

    # Fill country_code from the outer-join gaps
    panel["country_code"] = (
        panel.groupby("country_name")["country_code"]
        .transform(lambda s: s.ffill().bfill())
    )

    # ── Extrapolation flags ──
    panel["employment_agriculture_extrap"] = panel["year"].isin(agri_extrap_yrs)
    panel["hdi_extrap"]                    = panel["year"].isin(hdi_extrap_yrs)
    panel["coc_extrap"]                    = panel["year"].isin(coc_extrap_yrs)

    # ── Sort and reorder ──
    final_cols = [
        "country_name", "country_code", "scenario", "year",
        "gdp_per_capita", "population",
        "hdi", "control_of_corruption", "employment_agriculture", "gini_coefficient",
        "employment_agriculture_extrap", "hdi_extrap", "coc_extrap",
    ]
    panel = panel[[c for c in final_cols if c in panel.columns]]
    panel = panel.sort_values(["country_name", "scenario", "year"]).reset_index(drop=True)

    # ── Save ──
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "ssp_forecast_panel.csv"
    panel.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"  Shape:            {panel.shape}")
    print(f"  Countries:        {panel['country_name'].nunique()}")
    print(f"  Scenarios:        {panel['scenario'].unique().tolist()}")
    print(f"  Year range:       {panel['year'].min()}–{panel['year'].max()}")
    print(f"  Agri extrapolated: years {sorted(agri_extrap_yrs)}")
    print(f"  HDI  extrapolated: years {sorted(hdi_extrap_yrs)}")
    print(f"  CoC  extrapolated: years {sorted(coc_extrap_yrs)}")

    return panel
