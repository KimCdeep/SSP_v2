"""
preprocessing.py
Historical data cleaning, imputation, and panel merging.

Imputation strategy (supervisor-mandated — do NOT drop countries):
  Step 1: Linear interpolation within each country's time series (fills interior gaps)
  Step 2: Forward-fill then backward-fill, max 3 years (fills edge NaNs)
  Step 3: Regional median fill for any remaining NaNs (groups by WB region via ISO3 code)
  Flag: 'high_imputation_flag' = True if >50% of a country's values were originally NaN
        for at least one variable. Flagged countries are retained but marked for report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── Year ranges ────────────────────────────────────────────────────────────────

YEARS_HIST: list[int] = list(range(1993, 2024))       # 1993–2023, all sources
YEARS_HDI_RAW: list[int] = list(range(1993, 2020))     # HDI raw data only to 2019
YEARS_HDI_EXTRAP: list[int] = list(range(2020, 2024))  # extrapolated 2020–2023

# ── WB aggregate/regional codes to exclude (not actual countries) ──────────────

WB_AGGREGATE_CODES: set[str] = {
    "AFE", "AFW", "ARB", "CEB", "CSS",
    "EAP", "EAR", "EAS", "ECA", "ECS",
    "EMU", "EUU", "FCS", "HIC", "HPC",
    "IBD", "IBT", "IDA", "IDB", "IDX", "INX",
    "LAC", "LCN", "LDC", "LIC", "LMC", "LMY", "LTE",
    "MEA", "MIC", "MNA", "NAC",
    "OED", "OSS", "PRE", "PSS", "PST",
    "SAS", "SSA", "SSF", "SST",
    "TEA", "TEC", "TLA", "TMN", "TSA", "TSS",
    "UMC", "WLD",
}

# ── ISO3 → World Bank region (for regional-median imputation) ─────────────────
# Regions: EAP, ECA, LAC, MENA, NAC, SAS, SSA

ISO3_TO_REGION: dict[str, str] = {
    # East Asia & Pacific
    "AUS": "EAP", "BRN": "EAP", "CHN": "EAP", "FJI": "EAP", "FSM": "EAP",
    "GUM": "EAP", "IDN": "EAP", "JPN": "EAP", "KHM": "EAP", "KIR": "EAP",
    "KOR": "EAP", "LAO": "EAP", "MHL": "EAP", "MMR": "EAP", "MNG": "EAP",
    "MYS": "EAP", "NRU": "EAP", "NZL": "EAP", "PHL": "EAP", "PLW": "EAP",
    "PNG": "EAP", "PRK": "EAP", "PYF": "EAP", "SLB": "EAP", "THA": "EAP",
    "TLS": "EAP", "TON": "EAP", "TUV": "EAP", "TWN": "EAP", "VNM": "EAP",
    "VUT": "EAP", "WSM": "EAP",
    # Europe & Central Asia
    "ALB": "ECA", "AND": "ECA", "ARM": "ECA", "AUT": "ECA", "AZE": "ECA",
    "BEL": "ECA", "BGR": "ECA", "BIH": "ECA", "BLR": "ECA", "CHE": "ECA",
    "CYP": "ECA", "CZE": "ECA", "DEU": "ECA", "DNK": "ECA", "ESP": "ECA",
    "EST": "ECA", "FIN": "ECA", "FRA": "ECA", "GBR": "ECA", "GEO": "ECA",
    "GRC": "ECA", "HRV": "ECA", "HUN": "ECA", "IRL": "ECA", "ISL": "ECA",
    "ITA": "ECA", "KAZ": "ECA", "KGZ": "ECA", "LIE": "ECA", "LTU": "ECA",
    "LUX": "ECA", "LVA": "ECA", "MCO": "ECA", "MDA": "ECA", "MKD": "ECA",
    "MNE": "ECA", "NLD": "ECA", "NOR": "ECA", "POL": "ECA", "PRT": "ECA",
    "ROU": "ECA", "RUS": "ECA", "SMR": "ECA", "SRB": "ECA", "SVK": "ECA",
    "SVN": "ECA", "SWE": "ECA", "TJK": "ECA", "TKM": "ECA", "TUR": "ECA",
    "UKR": "ECA", "UZB": "ECA", "XKX": "ECA",
    # Latin America & Caribbean
    "ARG": "LAC", "ATG": "LAC", "BHS": "LAC", "BLZ": "LAC", "BOL": "LAC",
    "BRA": "LAC", "BRB": "LAC", "CHL": "LAC", "COL": "LAC", "CRI": "LAC",
    "CUB": "LAC", "DMA": "LAC", "DOM": "LAC", "ECU": "LAC", "GRD": "LAC",
    "GTM": "LAC", "GUY": "LAC", "HND": "LAC", "HTI": "LAC", "JAM": "LAC",
    "KNA": "LAC", "LCA": "LAC", "MEX": "LAC", "NIC": "LAC", "PAN": "LAC",
    "PER": "LAC", "PRY": "LAC", "SLV": "LAC", "SUR": "LAC", "TTO": "LAC",
    "URY": "LAC", "VCT": "LAC", "VEN": "LAC",
    # Middle East & North Africa
    "ARE": "MENA", "BHR": "MENA", "DJI": "MENA", "DZA": "MENA", "EGY": "MENA",
    "IRN": "MENA", "IRQ": "MENA", "ISR": "MENA", "JOR": "MENA", "KWT": "MENA",
    "LBN": "MENA", "LBY": "MENA", "MAR": "MENA", "MLT": "MENA", "OMN": "MENA",
    "QAT": "MENA", "SAU": "MENA", "SYR": "MENA", "TUN": "MENA", "YEM": "MENA",
    # North America
    "BMU": "NAC", "CAN": "NAC", "USA": "NAC",
    # South Asia
    "AFG": "SAS", "BGD": "SAS", "BTN": "SAS", "IND": "SAS", "LKA": "SAS",
    "MDV": "SAS", "NPL": "SAS", "PAK": "SAS",
    # Sub-Saharan Africa
    "AGO": "SSA", "BDI": "SSA", "BEN": "SSA", "BFA": "SSA", "BWA": "SSA",
    "CAF": "SSA", "CIV": "SSA", "CMR": "SSA", "COD": "SSA", "COG": "SSA",
    "COM": "SSA", "CPV": "SSA", "ERI": "SSA", "ETH": "SSA", "GAB": "SSA",
    "GHA": "SSA", "GIN": "SSA", "GMB": "SSA", "GNB": "SSA", "GNQ": "SSA",
    "KEN": "SSA", "LBR": "SSA", "LSO": "SSA", "MDG": "SSA", "MLI": "SSA",
    "MOZ": "SSA", "MRT": "SSA", "MUS": "SSA", "MWI": "SSA", "NAM": "SSA",
    "NER": "SSA", "NGA": "SSA", "RWA": "SSA", "SDN": "SSA", "SEN": "SSA",
    "SLE": "SSA", "SOM": "SSA", "SSD": "SSA", "STP": "SSA", "SWZ": "SSA",
    "SYC": "SSA", "TCD": "SSA", "TGO": "SSA", "TZA": "SSA", "UGA": "SSA",
    "ZAF": "SSA", "ZMB": "SSA", "ZWE": "SSA",
}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_wb_historical(filepath: Path, value_col: str) -> pd.DataFrame:
    """
    Load a World Bank wide CSV, keep country rows only,
    melt to long format, filter to YEARS_HIST.

    Auto-detects the header row by searching for "Country Name" in the
    first 10 lines, so it works regardless of whether blank separator
    lines contain commas or are truly empty.

    Returns columns: [country_name, country_code, year, <value_col>]
    """
    # Auto-detect header row
    header_row = 0
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if "Country Name" in line:
                header_row = i
                break
            if i >= 10:
                break
    df = pd.read_csv(filepath, skiprows=header_row)
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Keep only actual countries (exclude WB regional aggregates)
    df = df[~df["Country Code"].isin(WB_AGGREGATE_CODES)].copy()

    # Select year columns that exist in our target range
    year_cols = [c for c in df.columns if c.isdigit() and int(c) in YEARS_HIST]

    df = df[["Country Name", "Country Code"] + year_cols].rename(
        columns={"Country Name": "country_name", "Country Code": "country_code"}
    )
    df = df.melt(
        id_vars=["country_name", "country_code"],
        var_name="year",
        value_name=value_col,
    )
    df["year"] = df["year"].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.sort_values(["country_name", "year"]).reset_index(drop=True)


def load_poverty_long(filepath: Path, value_col: str) -> pd.DataFrame:
    """
    Load a pre-melted (long-format) poverty CSV with columns like
    'country', 'year', and a headcount ratio column.

    Renames columns to match the standard schema and filters to YEARS_HIST.
    Returns columns: [country_name, country_code, year, <value_col>]
    """
    df = pd.read_csv(filepath)

    # Identify the headcount ratio column (the one that isn't 'country' or 'year')
    ratio_col = [c for c in df.columns if c not in ("country", "year")][0]

    df = df.rename(columns={"country": "country_name", ratio_col: value_col})
    df["year"] = df["year"].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Filter to YEARS_HIST
    df = df[df["year"].isin(YEARS_HIST)].copy()

    # Add country_code via ISO3_TO_REGION lookup (reverse isn't available,
    # so we leave it blank — merge_to_panel will fill it from other sources)
    if "country_code" not in df.columns:
        df["country_code"] = np.nan

    return df.sort_values(["country_name", "year"]).reset_index(drop=True)


def load_hdi(filepath: Path) -> pd.DataFrame:
    """
    Load HDI CSV (no skiprows, different structure).
    Drops 'HDI Rank', renames 'Country' → 'country_name'.
    Filters to YEARS_HDI_RAW, extrapolates to 2023 using geometric
    mean growth over the last 5 available years (2015–2019).

    Returns columns: [country_name, year, hdi]
    Note: HDI has no country_code column — will be joined from another source.
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=["HDI Rank"], errors="ignore")
    df = df.rename(columns={"Country": "country_name"})

    # Replace '..' (WB missing marker) with NaN and convert to numeric
    year_cols = [c for c in df.columns if c.isdigit()]
    df[year_cols] = df[year_cols].replace("..", np.nan).apply(
        pd.to_numeric, errors="coerce"
    )

    # Keep raw years in range
    raw_year_cols = [c for c in year_cols if int(c) in YEARS_HDI_RAW]
    df = df[["country_name"] + raw_year_cols].copy()

    # Extrapolate 2020–2023: geometric mean growth over last 5 raw years (2015–2019)
    # Growth factor = (2019 / 2015)^(1/5)  — matches the original notebook's formula
    df["_growth"] = (df["2019"] / df["2015"]) ** (1 / 5)

    for yr in YEARS_HDI_EXTRAP:
        prev = str(yr - 1)
        df[str(yr)] = df[prev] * df["_growth"]

    df = df.drop(columns=["_growth"])

    # Melt to long
    all_year_cols = raw_year_cols + [str(y) for y in YEARS_HDI_EXTRAP]
    df = df.melt(id_vars=["country_name"], var_name="year", value_name="hdi")
    df["year"] = df["year"].astype(int)
    df["hdi"] = pd.to_numeric(df["hdi"], errors="coerce")

    # Filter to YEARS_HIST
    df = df[df["year"].isin(YEARS_HIST)]
    return df.sort_values(["country_name", "year"]).reset_index(drop=True)


# ── Imputation ─────────────────────────────────────────────────────────────────

def _regional_median_fill(
    df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """
    Fill remaining NaNs using the median of the country's WB region for that year.
    Countries with unknown region codes fall back to global median.
    """
    df = df.copy()
    df["_region"] = df["country_code"].map(ISO3_TO_REGION).fillna("GLOBAL")

    def fill_group(g: pd.DataFrame) -> pd.DataFrame:
        for yr, yr_group in g.groupby("year"):
            mask = g["year"] == yr
            still_nan = mask & g[value_col].isna()
            if still_nan.any():
                region_median = yr_group[value_col].median()
                g.loc[still_nan, value_col] = region_median
        return g

    # Per-region fill
    df = df.groupby("_region", group_keys=False).apply(fill_group)

    # Global fallback for truly isolated NaNs
    global_medians = df.groupby("year")[value_col].transform("median")
    df[value_col] = df[value_col].fillna(global_medians)

    # _region may already be dropped by groupby().apply() in pandas >= 3.0
    df = df.drop(columns=["_region"], errors="ignore")
    return df


def compute_na_pct(df: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Return per-country fraction of NaN values over YEARS_HIST.
    Index = country_name, values = fraction in [0, 1].
    """
    return (
        df[df["year"].isin(YEARS_HIST)]
        .groupby("country_name")[value_col]
        .apply(lambda s: s.isna().mean())
    )


def impute_variable(
    df: pd.DataFrame,
    value_col: str,
    limit_ffill_bfill: int = 3,
) -> pd.DataFrame:
    """
    Apply the full 3-step imputation strategy to a long-format DataFrame.

    Step 1: Linear interpolation per country (fills interior time-series gaps).
    Step 2: Forward-fill then backward-fill, max `limit_ffill_bfill` years.
    Step 3: Regional-median fill for any remaining NaNs.

    NOTE: Countries are NEVER dropped regardless of missingness level.
    """
    df = df.copy().sort_values(["country_name", "year"])

    # Step 1: linear interpolation within each country's time series
    df[value_col] = (
        df.groupby("country_name")[value_col]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )

    # Step 2: forward-fill then backward-fill (handles leading/trailing NaNs)
    df[value_col] = (
        df.groupby("country_name")[value_col]
        .transform(lambda s: s.ffill(limit=limit_ffill_bfill).bfill(limit=limit_ffill_bfill))
    )

    # Step 3: regional median for any remaining NaNs
    if df[value_col].isna().any():
        if "country_code" in df.columns:
            df = _regional_median_fill(df, value_col)
        else:
            # HDI has no country_code — fall back to global year median
            global_medians = df.groupby("year")[value_col].transform("median")
            df[value_col] = df[value_col].fillna(global_medians)

    return df.reset_index(drop=True)


# ── Merging & flagging ─────────────────────────────────────────────────────────

def _add_high_imputation_flag(
    panel: pd.DataFrame,
    na_pcts: dict[str, pd.Series],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Add 'high_imputation_flag' column: True for countries where ANY variable
    had more than `threshold` fraction of NaNs originally.
    """
    flagged_countries: set[str] = set()
    for var, pct_series in na_pcts.items():
        flagged_countries.update(pct_series[pct_series > threshold].index.tolist())

    panel = panel.copy()
    panel["high_imputation_flag"] = panel["country_name"].isin(flagged_countries)
    return panel


def merge_to_panel(
    gdp_df: pd.DataFrame,
    hdi_df: pd.DataFrame,
    corruption_df: pd.DataFrame,
    agri_df: pd.DataFrame,
    gini_df: pd.DataFrame,
    pov3_df: pd.DataFrame,
    pov420_df: pd.DataFrame,
    pov830_df: pd.DataFrame,
    pov10_df: pd.DataFrame,
    na_pcts: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Outer-join all cleaned long-format DataFrames into a single panel.
    Uses country_name + year as the join key.
    country_code is taken from the WB sources (HDI lacks it; filled via lookup).

    Final columns:
        country_name, country_code, year,
        gdp, hdi, control_of_corruption, employment_agriculture,
        gini_coefficient, poverty_3, poverty_4_20, poverty_8_30, poverty_10,
        high_imputation_flag
    """
    # Start from GDP (has country_code)
    panel = gdp_df[["country_name", "country_code", "year", "gdp"]]

    # WB sources that also carry country_code
    for df, col in [
        (corruption_df, "control_of_corruption"),
        (agri_df,       "employment_agriculture"),
        (gini_df,       "gini_coefficient"),
        (pov3_df,       "poverty_3"),
        (pov420_df,     "poverty_4_20"),
        (pov830_df,     "poverty_8_30"),
        (pov10_df,      "poverty_10"),
    ]:
        sub = df[["country_name", "year", col]].drop_duplicates(["country_name", "year"])
        panel = panel.merge(sub, on=["country_name", "year"], how="outer")

    # HDI: no country_code, merge on name only
    hdi_sub = hdi_df[["country_name", "year", "hdi"]].drop_duplicates(["country_name", "year"])
    panel = panel.merge(hdi_sub, on=["country_name", "year"], how="outer")

    # Fill missing country_code where possible from other rows of the same country
    panel["country_code"] = (
        panel.groupby("country_name")["country_code"]
        .transform(lambda s: s.ffill().bfill())
    )

    panel = _add_high_imputation_flag(panel, na_pcts)

    final_cols = [
        "country_name", "country_code", "year",
        "gdp", "hdi", "control_of_corruption", "employment_agriculture",
        "gini_coefficient", "poverty_3", "poverty_4_20", "poverty_8_30", "poverty_10",
        "high_imputation_flag",
    ]
    panel = panel[[c for c in final_cols if c in panel.columns]]
    return panel.sort_values(["country_name", "year"]).reset_index(drop=True)


# ── Quality report ─────────────────────────────────────────────────────────────

def build_quality_report(
    panel_raw: pd.DataFrame,
    panel_imputed: pd.DataFrame,
    na_pcts_before: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Return a DataFrame summarising:
      - n_countries: unique countries in the panel
      - year_min / year_max: actual year coverage
      - pct_missing_before / pct_missing_after: per variable
      - n_high_imputation_countries: countries flagged
    """
    value_cols = [
        "gdp", "hdi", "control_of_corruption", "employment_agriculture",
        "gini_coefficient", "poverty_3", "poverty_4_20", "poverty_8_30", "poverty_10",
    ]
    records = []
    for col in value_cols:
        if col not in panel_raw.columns:
            continue
        pct_before = panel_raw[col].isna().mean() * 100
        pct_after  = panel_imputed[col].isna().mean() * 100
        records.append({
            "variable":            col,
            "pct_missing_before":  round(pct_before, 2),
            "pct_missing_after":   round(pct_after,  2),
            "pct_imputed":         round(pct_before - pct_after, 2),
        })

    summary = pd.DataFrame(records)

    n_countries  = panel_imputed["country_name"].nunique()
    year_min     = panel_imputed["year"].min()
    year_max     = panel_imputed["year"].max()
    n_flagged    = panel_imputed.drop_duplicates("country_name")["high_imputation_flag"].sum()

    print(f"\n{'='*55}")
    print(f"  Panel summary")
    print(f"  Countries:              {n_countries}")
    print(f"  Year range:             {year_min}–{year_max}")
    print(f"  High-imputation flag:   {n_flagged} countries (>50% NaN in ≥1 variable)")
    print(f"{'='*55}")
    print(summary.to_string(index=False))
    print(f"{'='*55}\n")

    return summary
