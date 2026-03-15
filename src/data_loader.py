"""
data_loader.py
Functions to load and parse each raw data source into long-format DataFrames
with columns: [country_name, country_code, year, <value>]
"""

import pandas as pd
from pathlib import Path
from config import DATA_RAW_DIR


# ── Helpers ────────────────────────────────────────────────────────────────────

def _wb_wide_to_long(filepath: Path, value_name: str, skiprows: int = 3) -> pd.DataFrame:
    """
    Load a World Bank wide-format CSV (country rows × year columns) and melt
    into long format. Skips the standard 3-row WB header by default.
    """
    df = pd.read_csv(filepath, skiprows=skiprows)
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df.rename(columns={"Country Name": "country_name", "Country Code": "country_code"})
    df = df[["country_name", "country_code"] + year_cols]
    df = df.melt(id_vars=["country_name", "country_code"],
                 var_name="year", value_name=value_name)
    df["year"] = df["year"].astype(int)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df


def _ssp_explorer_to_long(filepath: Path, value_name: str) -> pd.DataFrame:
    """
    Load an SSP Extension Explorer CSV
    (columns: model, scenario, region, variable, unit, <year cols>) and melt
    into long format with columns: [scenario, country_name, year, <value>].
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    year_cols = [c for c in df.columns if c.isdigit()]
    id_cols = [c for c in ["model", "scenario", "region", "variable", "unit"]
               if c in df.columns]
    df = df[id_cols + year_cols]
    df = df.rename(columns={"region": "country_name"})
    df = df.melt(id_vars=id_cols, var_name="year", value_name=value_name)
    df["year"] = df["year"].astype(int)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df


# ── Historical loaders ─────────────────────────────────────────────────────────

def load_historical_gdp() -> pd.DataFrame:
    return _wb_wide_to_long(
        DATA_RAW_DIR / "GDP_HistoricalData_1960_2024.csv",
        value_name="gdp_per_capita",
    )


def load_historical_hdi() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW_DIR / "HDI_1990_2019.csv")
    df = df.drop(columns=["HDI Rank"], errors="ignore")
    df = df.rename(columns={"Country": "country_name"})
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[["country_name"] + year_cols]
    df = df.melt(id_vars=["country_name"], var_name="year", value_name="hdi")
    df["year"] = df["year"].astype(int)
    df["hdi"] = pd.to_numeric(df["hdi"], errors="coerce")
    return df


def load_historical_corruption() -> pd.DataFrame:
    return _wb_wide_to_long(
        DATA_RAW_DIR / "ControlOfCorruption_Historical_WorldBank_1996-2023.csv",
        value_name="control_of_corruption",
    )


def load_historical_employment_agri() -> pd.DataFrame:
    return _wb_wide_to_long(
        DATA_RAW_DIR / "EmploymentInAgriculture_Historical_WorldBank_1991-2023.csv",
        value_name="employment_agriculture",
    )


def load_historical_gini() -> pd.DataFrame:
    return _wb_wide_to_long(
        DATA_RAW_DIR / "GiniCoefficient_Historical_WorldBank_1963-2024.csv",
        value_name="gini_coefficient",
    )


def load_historical_poverty(threshold: str = "$3") -> pd.DataFrame:
    """
    threshold: one of '$3', '$4.20', '$8.30', '$10'
    """
    filename_map = {
        "$3":    "POV_3$_1963_2024.csv",
        "$4.20": "POV_4.20$_1963_2024.csv",
        "$8.30": "POV_8.30$_1963_2024.csv",
        "$10":   "POV_10$_1963_2024.csv",
    }
    if threshold not in filename_map:
        raise ValueError(f"Unknown threshold '{threshold}'. Choose from {list(filename_map)}")
    return _wb_wide_to_long(
        DATA_RAW_DIR / filename_map[threshold],
        value_name="poverty_headcount_ratio",
    )


# ── Forecast loaders ───────────────────────────────────────────────────────────

def load_forecast_gdp_pop() -> pd.DataFrame:
    """
    Load the GDP/Population SSP Excel file.
    Sheet: 'data', columns: Model | Scenario | Region | Variable | Unit | <year cols>
    Returns long-format with columns: [scenario, country_name, variable, year, value]
    """
    df = pd.read_excel(DATA_RAW_DIR / "GDP(Forecast)_POP_SSP_1950_2100.xlsx", sheet_name="data")
    df.columns = df.columns.str.strip()
    year_cols = [c for c in df.columns if str(c).isdigit()]
    id_cols = [c for c in ["Model", "Scenario", "Region", "Variable", "Unit"]
               if c in df.columns]
    df = df[id_cols + year_cols]
    df = df.rename(columns={
        "Scenario": "scenario",
        "Region":   "country_name",
        "Variable": "variable",
    })
    df = df.melt(id_vars=["scenario", "country_name", "variable"] +
                          [c for c in ["Model", "Unit"] if c in df.columns],
                 var_name="year", value_name="value")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_forecast_corruption() -> pd.DataFrame:
    return _ssp_explorer_to_long(
        DATA_RAW_DIR / "ControlOfCorruption_Forecast_SSPExtensionExplorer_2015-2099.csv",
        value_name="control_of_corruption",
    )


def load_forecast_employment_agri() -> pd.DataFrame:
    return _ssp_explorer_to_long(
        DATA_RAW_DIR / "EmploymentInAgriculture_Forecast_SSPExtensionExplorer_2016-2050.csv",
        value_name="employment_agriculture",
    )


def load_forecast_gini() -> pd.DataFrame:
    return _ssp_explorer_to_long(
        DATA_RAW_DIR / "GiniCoefficient_Forecast_SSPExtensionExplorer_2015-2100.csv",
        value_name="gini_coefficient",
    )


def load_forecast_hdi() -> pd.DataFrame:
    return _ssp_explorer_to_long(
        DATA_RAW_DIR / "HumanDevelopmentIndex_Forecast_SSPExtensionExplorer_2010-2075.csv",
        value_name="hdi",
    )
