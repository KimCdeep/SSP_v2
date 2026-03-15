"""
utils.py
General-purpose helpers: country name harmonization, ISO code lookups,
DataFrame utilities, and plotting helpers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from config import SSP_SCENARIOS, OUTPUTS_DIR


# ── Country utilities ──────────────────────────────────────────────────────────

# Extend as mismatches are discovered during merging
COUNTRY_NAME_MAP: dict[str, str] = {
    "Korea, Rep.":               "South Korea",
    "Korea, Dem. People's Rep.": "North Korea",
    "Kyrgyz Republic":           "Kyrgyzstan",
    "Lao PDR":                   "Laos",
    "Micronesia, Fed. Sts.":     "Micronesia",
    "Slovak Republic":           "Slovakia",
    "Syrian Arab Republic":      "Syria",
    "Turkiye":                   "Turkey",
    "Viet Nam":                  "Vietnam",
    "Yemen, Rep.":               "Yemen",
    "Egypt, Arab Rep.":          "Egypt",
    "Iran, Islamic Rep.":        "Iran",
    "Venezuela, RB":             "Venezuela",
    "Congo, Dem. Rep.":          "Democratic Republic of Congo",
    "Congo, Rep.":               "Republic of Congo",
}


def harmonize_names(series: pd.Series) -> pd.Series:
    return series.replace(COUNTRY_NAME_MAP)


def get_countries_in_all_sources(*dfs: pd.DataFrame, col: str = "country_name") -> set:
    """Return the intersection of country names present in all provided DataFrames."""
    sets = [set(df[col].dropna().unique()) for df in dfs]
    return set.intersection(*sets)


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def pivot_to_wide(
    df: pd.DataFrame,
    index: str = "country_name",
    columns: str = "year",
    values: str = "value",
) -> pd.DataFrame:
    return df.pivot(index=index, columns=columns, values=values)


def describe_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column."""
    total = len(df)
    missing = df.isna().sum()
    return pd.DataFrame({
        "missing_n": missing,
        "missing_pct": (missing / total * 100).round(2),
    }).sort_values("missing_pct", ascending=False)


# ── Scenario colour palette ────────────────────────────────────────────────────

SCENARIO_COLORS = {
    "SSP1": "#2ca02c",   # green  — sustainability
    "SSP4": "#d62728",   # red    — inequality
    "SSP5": "#1f77b4",   # blue   — fossil-fueled
}


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_country_projection(
    df: pd.DataFrame,
    country: str,
    threshold: str = "$3",
    save: bool = False,
) -> None:
    """
    Line chart of predicted poverty headcount over time for a single country,
    with one line per SSP scenario.
    df must have columns: [country_name, year, scenario, predicted_poverty]
    """
    subset = df[df["country_name"] == country]
    if subset.empty:
        print(f"No data for {country}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario in SSP_SCENARIOS:
        s = subset[subset["scenario"] == scenario].sort_values("year")
        if s.empty:
            continue
        ax.plot(s["year"], s["predicted_poverty"],
                label=scenario, color=SCENARIO_COLORS.get(scenario),
                linewidth=2, marker="o", markersize=4)

    ax.set_title(f"{country} — Poverty Headcount {threshold}/day under SSP Scenarios")
    ax.set_xlabel("Year")
    ax.set_ylabel("Poverty Headcount Ratio (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax.legend(title="Scenario")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"projection_{country.replace(' ', '_')}_{threshold.replace('$', '')}.png"
        plt.savefig(OUTPUTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
