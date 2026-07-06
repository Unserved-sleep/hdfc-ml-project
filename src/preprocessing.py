"""
preprocessing.py
================
Utility functions for loading, cleaning, and preparing the HDFC loan dataset
for machine learning pipelines.

Functions
---------
- load_dataset           : Load a CSV file into a pandas DataFrame
- separate_features      : Split DataFrame into feature matrix X and target vector y
- get_feature_types      : Detect numerical and categorical column names
- create_preprocessor    : Build a ColumnTransformer pipeline (impute + scale/encode)
- select_features        : Subset specific columns as features and target
- missing_value_summary  : Report columns with missing values and their percentages
- detect_outliers        : Identify outlier rows in a column using the IQR method
- remove_outliers        : Drop outlier rows from a DataFrame using the IQR method
- basic_cleaning         : Remove duplicates and strip whitespace from column names
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset(file_path):
    """Read a CSV file from `file_path` and return a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df


# ---------------------------------------------------------------------------
# Feature / Target Separation
# ---------------------------------------------------------------------------

def separate_features(df, target_column):
    """
    Split `df` into feature matrix X and target series y.

    Parameters
    ----------
    df            : pandas DataFrame
    target_column : name of the column to use as the target label

    Returns
    -------
    x : DataFrame of all columns except the target
    y : Series of the target column
    """
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y


def get_feature_types(X):
    """
    Identify numerical and categorical columns in X.

    Returns
    -------
    numerical_features   : list of int64/float64 column names
    categorical_features : list of object column names
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()
    return numerical_features, categorical_features


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------

def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Build a sklearn ColumnTransformer that:
      - For numerical columns : imputes with the median, then applies StandardScaler
      - For categorical columns: imputes with the most-frequent value, then OneHotEncodes

    Parameters
    ----------
    numerical_features   : list of numerical column names
    categorical_features : list of categorical column names

    Returns
    -------
    preprocessor : fitted-ready ColumnTransformer
    """
    # Numerical pipeline: median imputation → z-score scaling
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical pipeline: mode imputation → one-hot encoding
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine both sub-pipelines into one ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Feature Selection Helper
# ---------------------------------------------------------------------------

def select_features(df, feature_list, target_column):
    """
    Return a copy of selected feature columns and the target column from `df`.

    Parameters
    ----------
    df            : pandas DataFrame
    feature_list  : list of column names to use as features
    target_column : name of the target column

    Returns
    -------
    x : DataFrame containing only the specified feature columns
    y : Series of the target column
    """
    x = df[feature_list].copy()
    y = df[target_column].copy()
    return x, y


# ---------------------------------------------------------------------------
# Data Quality / Cleaning
# ---------------------------------------------------------------------------

def missing_value_summary(df):
    """
    Summarise columns that contain missing values.

    Returns a DataFrame with columns ['Missing Values', 'Percentage'],
    sorted by percentage descending, excluding columns with no nulls.
    """
    missing = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Percentage": (df.isnull().sum() / len(df)) * 100
    })

    missing = missing[missing["Missing Values"] > 0]
    return missing.sort_values(
        by="Percentage",
        ascending=False
    )


def detect_outliers(df, column):
    """
    Detect rows in `column` that lie outside the IQR fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

    Returns
    -------
    DataFrame subset containing only the outlier rows.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]
    return outliers


def remove_outliers(df, column):
    """
    Remove rows where `column` values fall outside the IQR fence.

    Returns a filtered copy of `df` with outlier rows dropped.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]
    return df


def basic_cleaning(
        df,
        remove_duplicates=True
):
    """
    Perform basic DataFrame housekeeping:
      - Optionally drop duplicate rows (default: True)
      - Strip leading/trailing whitespace from all column names

    Returns the cleaned DataFrame.
    """
    if remove_duplicates:
        df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    return df
