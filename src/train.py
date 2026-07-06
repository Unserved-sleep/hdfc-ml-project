from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def predict(model, x_test):
    return model.predict(x_test)


def build_pipeline(preprocessor, model):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline

def perform_grid_search(
        pipeline,
        parameters,
        x_train,
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
    grid.fit(x_train, y_train)
    return grid