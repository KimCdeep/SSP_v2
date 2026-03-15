from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_FINAL_DIR     = ROOT_DIR / "data" / "final"
MODELS_DIR         = ROOT_DIR / "models"
OUTPUTS_DIR        = ROOT_DIR / "outputs"

# ── Temporal ranges ────────────────────────────────────────────────────────────
HISTORICAL_YEAR_RANGE = range(1993, 2023)   # common overlap of all historical sources

FORECAST_YEARS_5Y = [
    2025, 2030, 2035, 2040, 2045, 2050,
    2055, 2060, 2065, 2070, 2075, 2080,
    2085, 2090, 2095, 2100,
]

# ── Scenarios & thresholds ─────────────────────────────────────────────────────
SSP_SCENARIOS = ["SSP1", "SSP4", "SSP5"]

POVERTY_THRESHOLDS = ["$3", "$4.20", "$8.30", "$10"]  # dashboard lets user choose
PRIMARY_POVERTY_THRESHOLD = "$3"                        # reference for model comparison & SHAP

# ── Column names ───────────────────────────────────────────────────────────────
TARGET_COL = "poverty_headcount_ratio"

FEATURE_COLS = [
    "gdp_per_capita",
    "population",
    "hdi",
    "control_of_corruption",
    "employment_agriculture",
    "gini_coefficient",
    "temperature",
]

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
