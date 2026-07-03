import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def basic_cleaning(df):
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()
    return df

def separate_features(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def get_feature_types(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numerical_features, categorical_features



def create_preprocessor(
    numerical_features,
    categorical_features
):
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



def select_features(df, feature_list, target_column):

    X = df[feature_list].copy()
    y = df[target_column].copy()

    return X, y


def missing_value_summary(df):
    missing = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Percentage": (df.isnull().sum()/len(df))*100
    })

    missing = missing[missing["Missing Values"] > 0]

    return missing.sort_values(
        by="Percentage",
        ascending=False
    )

def detect_outliers(df, column):
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


###################

def basic_cleaning2(
        df,
        remove_duplicates=True
):

    if remove_duplicates:
        df = df.drop_duplicates()

    df.columns = df.columns.str.strip()

    return df