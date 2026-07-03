from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(random_state=42)
            )
        ]
    )
    return pipeline


def train_decision_tree(preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(random_state=42)
            )
        ]
    )
    return pipeline

def train_random_forest(preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(random_state=42)
            )
        ]
    )
    return pipeline

def perform_grid_search(
        pipeline,
        parameters,
        X_train,
        y_train,
        scoring="accuracy",
        cv=5
):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


def predict(model, X_test):
    return model.predict(X_test)


#########################################

def build_pipeline(preprocessor, model):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

def perform_grid_search2(
        pipeline,
        parameters,
        X_train,
        y_train,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
):

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs
    )

    grid.fit(X_train, y_train)
    return grid