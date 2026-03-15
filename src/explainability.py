"""
explainability.py
SHAP analysis functions for the best-performing model.
Primary reference: $3/day poverty threshold (PRIMARY_POVERTY_THRESHOLD).
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import FEATURE_COLS, PRIMARY_POVERTY_THRESHOLD, OUTPUTS_DIR


# ── SHAP computation ───────────────────────────────────────────────────────────

def get_shap_explainer(model, X_train: np.ndarray, model_name: str):
    """
    Return the appropriate SHAP explainer for the given model type.
    Tree-based models use TreeExplainer; others fall back to KernelExplainer.
    """
    tree_models = {"xgboost_cpu", "xgboost_gpu", "lightgbm", "random_forest"}
    if model_name in tree_models:
        return shap.TreeExplainer(model)
    else:
        background = shap.kmeans(X_train, k=50)
        return shap.KernelExplainer(model.predict, background)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str,
    X_train: np.ndarray = None,
) -> shap.Explanation:
    explainer = get_shap_explainer(model, X_train if X_train is not None else X.values, model_name)
    shap_values = explainer(X.values)
    # Attach feature names for readable plots
    if hasattr(shap_values, "feature_names"):
        shap_values.feature_names = X.columns.tolist()
    return shap_values


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_summary(
    shap_values,
    X: pd.DataFrame,
    threshold: str = PRIMARY_POVERTY_THRESHOLD,
    model_name: str = "best_model",
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, feature_names=X.columns.tolist(), show=False)
    plt.title(f"SHAP Summary — {model_name} | Poverty {threshold}")
    plt.tight_layout()
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / f"shap_summary_{model_name}_{threshold.replace('$', '')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()


def plot_bar(
    shap_values,
    X: pd.DataFrame,
    threshold: str = PRIMARY_POVERTY_THRESHOLD,
    model_name: str = "best_model",
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_values, max_display=len(FEATURE_COLS), show=False)
    plt.title(f"SHAP Feature Importance — {model_name} | Poverty {threshold}")
    plt.tight_layout()
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / f"shap_bar_{model_name}_{threshold.replace('$', '')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()


def plot_dependence(
    shap_values,
    X: pd.DataFrame,
    feature: str,
    threshold: str = PRIMARY_POVERTY_THRESHOLD,
    model_name: str = "best_model",
    save: bool = True,
) -> None:
    feat_idx = X.columns.tolist().index(feature)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feat_idx, shap_values.values, X.values,
                         feature_names=X.columns.tolist(), ax=ax, show=False)
    ax.set_title(f"SHAP Dependence: {feature} — {model_name} | Poverty {threshold}")
    plt.tight_layout()
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / f"shap_dep_{feature}_{model_name}_{threshold.replace('$', '')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()


# ── Mean absolute SHAP importance table ───────────────────────────────────────

def shap_importance_table(shap_values, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
