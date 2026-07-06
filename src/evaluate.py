"""
evaluate.py
===========
Evaluation utilities for both classification and regression models used across
the HDFC loan prediction notebooks.

Functions
---------
Classification
  - evaluate_classification    : Return accuracy, precision, recall, F1 as a DataFrame
  - print_classification_report: Print sklearn's full classification report
  - plot_confusion_matrix       : Plot a colour-coded confusion matrix
  - plot_roc_curve              : Plot the ROC curve and display AUC
  - plot_feature_importance     : Bar chart of the top-N most important features

Regression
  - evaluate_regression  : Return MAE, MSE, RMSE, R² as a DataFrame
  - residual_plot        : Scatter plot of predicted values vs residuals
  - actual_vs_predicted  : Scatter plot comparing actual vs predicted values
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ---------------------------------------------------------------------------
# Classification Evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(y_true, y_pred, model_name="Model"):
    """
    Compute weighted-average classification metrics for a model.

    Parameters
    ----------
    y_true     : array-like of true class labels
    y_pred     : array-like of predicted class labels
    model_name : string label used in the returned DataFrame (default "Model")

    Returns
    -------
    pandas DataFrame with columns [Model, Accuracy, Precision, Recall, F1 Score]
    """
    metrics = pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [accuracy_score(y_true, y_pred)],
        "Precision": [precision_score(
            y_true,
            y_pred,
            average="weighted"
        )],
        "Recall": [recall_score(
            y_true,
            y_pred,
            average="weighted"
        )],
        "F1 Score": [f1_score(
            y_true,
            y_pred,
            average="weighted"
        )]
    })
    return metrics


def print_classification_report(
    y_true,
    y_pred
):
    """Print sklearn's full per-class precision/recall/F1 classification report."""
    print(classification_report(
        y_true,
        y_pred
    ))


def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix"
):
    """
    Display a confusion matrix heatmap using sklearn's ConfusionMatrixDisplay.

    Parameters
    ----------
    y_true : array-like of true labels
    y_pred : array-like of predicted labels
    title  : plot title (default "Confusion Matrix")
    """
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues"
    )
    plt.title(title)
    plt.show()


def plot_roc_curve(
    model,
    x_test,
    y_test,
    title="ROC Curve"
):
    """
    Plot the ROC curve (and AUC) for a trained binary classifier.

    Parameters
    ----------
    model  : fitted sklearn pipeline/estimator with predict_proba support
    x_test : test feature matrix
    y_test : test true labels
    title  : plot title (default "ROC Curve")
    """
    RocCurveDisplay.from_estimator(
        model,
        x_test,
        y_test
    )
    plt.title(title)
    plt.show()


def plot_feature_importance(
    model,
    feature_names,
    top_n=10
):
    """
    Bar chart showing the top-N most important features from a tree-based pipeline.

    Expects the pipeline to contain a step named 'classifier' with a
    `feature_importances_` attribute (e.g. RandomForestClassifier).

    Parameters
    ----------
    model         : fitted sklearn Pipeline containing a 'classifier' step
    feature_names : list of feature names matching the transformed feature space
    top_n         : number of top features to display (default 10)

    Returns
    -------
    pandas DataFrame of all features sorted by importance descending
    """
    # Extract feature importances from the 'classifier' step of the pipeline
    importance = model.named_steps[
        "classifier"
    ].feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(top_n),
        x="Importance",
        y="Feature"
    )
    plt.title("Top Feature Importance")
    plt.show()
    return feature_importance


# ---------------------------------------------------------------------------
# Regression Evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(
    y_true,
    y_pred
):
    """
    Compute regression error metrics.

    Parameters
    ----------
    y_true : array-like of true continuous target values
    y_pred : array-like of predicted values

    Returns
    -------
    pandas DataFrame with columns [MAE, MSE, RMSE, R2 Score]
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse ** 0.5,          # Root Mean Squared Error
        "R2 Score": r2_score(y_true, y_pred)
    }
    return pd.DataFrame(metrics, index=[0])


def residual_plot(
    y_true,
    y_pred
):
    """
    Scatter plot of predicted values vs residuals (y_true - y_pred).
    A horizontal red dashed line at y=0 marks zero error.

    Useful for diagnosing heteroscedasticity or systematic bias in predictions.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def actual_vs_predicted(
    y_true,
    y_pred
):
    """
    Scatter plot comparing actual vs predicted values.
    Points close to the diagonal indicate accurate predictions.
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()
