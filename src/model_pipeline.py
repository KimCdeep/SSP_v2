"""
model_pipeline.py
Unified training and evaluation pipeline for all 7 ML models.
Each model is trained once per poverty threshold (4×) for a total of 28 fitted models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from pygam import LinearGAM, s
import joblib

from config import FEATURE_COLS, TARGET_COL, RANDOM_STATE, PRIMARY_POVERTY_THRESHOLD, MODELS_DIR


# ── Model definitions ──────────────────────────────────────────────────────────

def get_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Return a dict of {model_name: estimator}."""
    return {
        "xgboost_cpu": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, device="cpu", verbosity=0,
        ),
        "xgboost_gpu": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, device="cuda", verbosity=0,
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=random_state, verbose=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=None,
            random_state=random_state, n_jobs=-1,
        ),
        "gam": LinearGAM(
            # one smooth term per feature; will be re-built at fit time based on n_features
            terms=None,
        ),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp",    MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", max_iter=1000,
                random_state=random_state, early_stopping=True,
            )),
        ]),
    }


# ── Training & evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2":   r2_score(y_test, y_pred),
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: str = PRIMARY_POVERTY_THRESHOLD,
    cv_folds: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Train all 7 models on (X_train, y_train) and evaluate on (X_test, y_test).

    Returns:
        fitted_models: {model_name: fitted_estimator}
        results_df:    DataFrame with MAE / RMSE / R² for each model
    """
    models = get_models()
    feature_names = X_train.columns.tolist()
    fitted_models = {}
    records = []

    for name, model in models.items():
        print(f"  Training {name} …")
        try:
            if name == "gam":
                # Build GAM terms dynamically based on feature count
                n_features = X_train.shape[1]
                model = LinearGAM(terms=sum(s(i) for i in range(n_features)))
            model.fit(X_train.values, y_train.values)
            metrics = evaluate_model(model, X_test.values, y_test.values)
        except Exception as exc:
            print(f"    WARNING: {name} failed — {exc}")
            metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
            model = None

        fitted_models[name] = model
        records.append({"model": name, "threshold": threshold, **metrics})

    results_df = pd.DataFrame(records).sort_values("rmse")
    return fitted_models, results_df


# ── Persistence ────────────────────────────────────────────────────────────────

def save_models(fitted_models: dict[str, Any], threshold: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in fitted_models.items():
        if model is not None:
            path = MODELS_DIR / f"{name}_{threshold.replace('$', '')}.joblib"
            joblib.dump(model, path)
            print(f"  Saved {path.name}")


def load_model(model_name: str, threshold: str) -> Any:
    path = MODELS_DIR / f"{model_name}_{threshold.replace('$', '')}.joblib"
    return joblib.load(path)


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_ssp(
    model,
    forecast_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """
    Run model inference on an SSP forecast DataFrame.
    forecast_df must contain all feature_cols plus [country_name, year, scenario].
    Returns forecast_df with an added 'predicted_poverty' column.
    """
    X = forecast_df[feature_cols].values
    forecast_df = forecast_df.copy()
    forecast_df["predicted_poverty"] = model.predict(X)
    forecast_df["predicted_poverty"] = forecast_df["predicted_poverty"].clip(0, 100)
    return forecast_df
