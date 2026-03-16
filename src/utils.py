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

# Maps SSP/IIASA country names (used in all forecast files) → ISO 3166-1 alpha-3
# Verified against all four SSP Extension Explorer CSVs and the IIASA Excel file.
SSP_NAME_TO_ISO3: dict[str, str] = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM",
    "Aruba": "ABW", "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
    "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
    "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN",
    "Bhutan": "BTN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Brunei Darussalam": "BRN",
    "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czechia": "CZE", "Côte d'Ivoire": "CIV",
    "Democratic Republic of the Congo": "COD",
    "Denmark": "DNK", "Djibouti": "DJI", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Eswatini": "SWZ",
    "Estonia": "EST", "Ethiopia": "ETH", "Fiji": "FJI", "Finland": "FIN",
    "France": "FRA", "French Guiana": "GUF", "French Polynesia": "PYF",
    "Gabon": "GAB", "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU",
    "Ghana": "GHA", "Greece": "GRC", "Grenada": "GRD", "Guam": "GUM",
    "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB",
    "Guyana": "GUY", "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN",
    "Iceland": "ISL", "India": "IND", "Indonesia": "IDN", "Iran": "IRN",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kiribati": "KIR", "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
    "Laos": "LAO", "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO",
    "Liberia": "LBR", "Libya": "LBY", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Macao": "MAC", "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS",
    "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT", "Mauritania": "MRT",
    "Mauritius": "MUS", "Mayotte": "MYT", "Mexico": "MEX",
    "Micronesia": "FSM", "Moldova": "MDA", "Mongolia": "MNG",
    "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nepal": "NPL", "Netherlands": "NLD",
    "New Caledonia": "NCL", "New Zealand": "NZL", "Nicaragua": "NIC",
    "Niger": "NER", "Nigeria": "NGA", "North Korea": "PRK",
    "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN",
    "Pakistan": "PAK", "Palestine": "PSE", "Panama": "PAN",
    "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT",
    "Romania": "ROU", "Russian Federation": "RUS", "Rwanda": "RWA",
    "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Samoa": "WSM",
    "Sao Tome and Principe": "STP", "Saudi Arabia": "SAU", "Senegal": "SEN",
    "Serbia": "SRB", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Slovakia": "SVK", "Slovenia": "SVN", "Solomon Islands": "SLB",
    "Somalia": "SOM", "South Africa": "ZAF", "South Korea": "KOR",
    "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA",
    "Sudan": "SDN", "Suriname": "SUR", "Sweden": "SWE",
    "Switzerland": "CHE", "Syria": "SYR", "Taiwan": "TWN",
    "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA",
    "Timor-Leste": "TLS", "Togo": "TGO", "Tonga": "TON",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
    "Turkmenistan": "TKM", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "United States": "USA", "United States Virgin Islands": "VIR",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Vanuatu": "VUT",
    "Venezuela": "VEN", "Viet Nam": "VNM", "Western Sahara": "ESH",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

# Maps World Bank historical names → SSP/IIASA names (for historical panel join)
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
