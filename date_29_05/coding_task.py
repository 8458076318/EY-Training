#Python
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:///quiz.db")
mlflow.set_experiment("quiz-experiment")

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

MAX_DEPTH = 5
N_ESTIMATORS = 100

with mlflow.start_run(run_name="quiz-run"):

    # (a) Log parameters
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("n_estimators", N_ESTIMATORS)

    # Train the model
    model = RandomForestClassifier(
        max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=42
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    # (b) Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_te, preds))
    mlflow.log_metric("f1", f1_score(y_te, preds, average="macro"))

    # (c) Log and register the model
    mlflow.sklearn.log_model(
        model,                          # the fitted model object
        "random-forest-model",
        registered_model_name="iris-rf-classifier"  # registry name
    )

    # (d) Set a tag
    mlflow.set_tag("team", "data-science")