"""
utils.py
========
General-purpose utility functions shared across all HDFC loan prediction notebooks.

Functions
---------
- save_model      : Persist a trained model/pipeline to disk using joblib
- load_model      : Load a saved model/pipeline back from disk
- set_plot_style  : Apply consistent matplotlib/seaborn visual settings project-wide
- compare_models  : Concatenate per-model metric DataFrames into one comparison table
- print_summary   : Print a formatted summary of a DataFrame (shape, types, nulls, duplicates)
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model, path):
    """
    Save a fitted sklearn model or pipeline to disk with joblib.

    Creates any intermediate directories in `path` if they don't exist.

    Parameters
    ----------
    model : fitted sklearn estimator or Pipeline
    path  : file path to save to (e.g. '../models/loan_approval_model.pkl')

    Example
    -------
    save_model(best_rf, '../models/loan_approval_model.pkl')
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(path):
    """
    Load a saved sklearn model or pipeline from disk.

    Parameters
    ----------
    path : file path to the saved .pkl file

    Returns
    -------
    Fitted sklearn estimator or Pipeline

    Example
    -------
    model = load_model('../models/loan_approval_model.pkl')
    """
    model = joblib.load(path)
    print(f"Model loaded ← {path}")
    return model


# ---------------------------------------------------------------------------
# Plot Style
# ---------------------------------------------------------------------------

def set_plot_style(style="whitegrid", figsize=(8, 5), font_size=11):
    """
    Apply consistent visual settings across all notebooks.

    Sets seaborn style and updates matplotlib rcParams so every plot
    in the session uses the same baseline appearance.

    Parameters
    ----------
    style     : seaborn style name (default 'whitegrid')
    figsize   : default figure size as (width, height) tuple (default (8, 5))
    font_size : base font size for axis labels and titles (default 11)

    Example
    -------
    set_plot_style()                          # use defaults
    set_plot_style('darkgrid', (10, 6), 12)  # custom settings
    """
    sns.set_style(style)
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = font_size


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models(results, names=None):
    """
    Concatenate a list of per-model metric DataFrames into one comparison table.

    Each DataFrame in `results` should be the output of evaluate_classification()
    or evaluate_regression() — a single-row DataFrame of metrics.

    Parameters
    ----------
    results : list of pandas DataFrames, one per model
    names   : optional list of model name strings to set as the index.
              If None, the existing index values are kept.

    Returns
    -------
    pandas DataFrame with one row per model, reset index

    Example
    -------
    comparison = compare_models(
        [lr_results, rf_results, gb_results],
        names=["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )
    """
    comparison = pd.concat(results, ignore_index=True)
    if names is not None:
        comparison.index = names
    return comparison


# ---------------------------------------------------------------------------
# DataFrame Summary
# ---------------------------------------------------------------------------

def print_summary(df):
    """
    Print a concise formatted summary of a pandas DataFrame.

    Displays:
      - Total rows and columns
      - Number of numerical and categorical columns
      - Duplicate row count
      - Total missing value count

    Parameters
    ----------
    df : pandas DataFrame to summarise

    Example
    -------
    print_summary(df)
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    print("=" * 50)
    print("DataFrame Summary")
    print("=" * 50)
    print(f"  Total Rows          : {df.shape[0]}")
    print(f"  Total Columns       : {df.shape[1]}")
    print(f"  Numerical Columns   : {len(num_cols)}")
    print(f"  Categorical Columns : {len(cat_cols)}")
    print(f"  Duplicate Rows      : {df.duplicated().sum()}")
    print(f"  Missing Values      : {df.isnull().sum().sum()}")
    print("=" * 50)
