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


def print_classification_report(
    y_true,
    y_pred
):
    print(classification_report(
        y_true,
        y_pred
    ))


def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix"
):
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

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=feature_importance.head(top_n),
        x="Importance",
        y="Feature"
    )

    plt.title("Top Feature Importance")
    plt.show()
    return feature_importance


def evaluate_regression(
    y_true,
    y_pred
):
    mse = mean_squared_error(
        y_true,
        y_pred
    )
    metrics = {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2 Score": r2_score(
            y_true,
            y_pred
        )
    }

    return pd.DataFrame(
        metrics,
        index=[0]
    )


def residual_plot(
    y_true,
    y_pred
):
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))

    plt.scatter(
        y_pred,
        residuals
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def actual_vs_predicted(
    y_true,
    y_pred
):
    plt.figure(figsize=(7,7))
    plt.scatter(
        y_true,
        y_pred
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

def evaluate_classification(y_true, y_pred, model_name="Model"):
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