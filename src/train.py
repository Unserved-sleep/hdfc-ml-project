"""
train.py
========
Model training utilities for the HDFC loan prediction project.
Provides helpers to build sklearn pipelines, run grid-search hyperparameter
tuning, and generate predictions.

Functions
---------
- predict             : Generate predictions from a fitted model/pipeline
- build_pipeline      : Wrap a preprocessor and a model into a single Pipeline
- perform_grid_search : Run GridSearchCV over a pipeline with given param grid
"""

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, x_test):
    """
    Return predicted labels for `x_test` using a fitted model or pipeline.

    Parameters
    ----------
    model  : fitted sklearn estimator or Pipeline
    x_test : feature matrix to predict on

    Returns
    -------
    numpy array of predicted class labels (classification) or values (regression)
    """
    return model.predict(x_test)


# ---------------------------------------------------------------------------
# Pipeline Construction
# ---------------------------------------------------------------------------

def build_pipeline(preprocessor, model):
    """
    Combine a preprocessor and a model into a two-step sklearn Pipeline.

    The pipeline exposes two named steps:
      - 'preprocessor' : the ColumnTransformer (from preprocessing.py)
      - 'classifier'   : the estimator (classifier or regressor)

    Parameters
    ----------
    preprocessor : sklearn ColumnTransformer (from create_preprocessor)
    model        : sklearn estimator (e.g. RandomForestClassifier)

    Returns
    -------
    sklearn Pipeline ready to be fitted with .fit(X_train, y_train)
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------

def perform_grid_search(
        pipeline,
        parameters,
        x_train,
        y_train,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
):
    """
    Run an exhaustive grid search over `parameters` on a sklearn Pipeline.

    Parameters
    ----------
    pipeline   : sklearn Pipeline (from build_pipeline)
    parameters : dict of param_grid — keys use the 'stepname__param' convention
                 e.g. {'classifier__n_estimators': [100, 200]}
    x_train    : training feature matrix
    y_train    : training target vector
    scoring    : metric to optimise (default "accuracy")
    cv         : number of cross-validation folds (default 5)
    n_jobs     : parallel jobs; -1 uses all available CPU cores (default -1)

    Returns
    -------
    Fitted GridSearchCV object. Access the best model via .best_estimator_
    and the best params via .best_params_.
    """
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs
    )
    grid.fit(x_train, y_train)
    return grid
