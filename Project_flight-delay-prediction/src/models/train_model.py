import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import json


def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    # Guardar métricas
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm.tolist()
        }, f, indent=4)

    print("✅ Metrics saved to reports/metrics.json")
    return clf


if __name__ == "__main__":
    # Paths
    data_dir = "data/processed"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_col = "is_delay"
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # MLflow experiment
    mlflow.set_experiment("flight_delay_prediction")

    with mlflow.start_run():
        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 50,
            "max_depth": 10,
            "n_jobs": 1,
            "random_state": 42
        })

        model = train_and_evaluate(X_train, y_train, X_test, y_test)

        # MLflow metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("accuracy", acc)
        for cls in ["0", "1"]:
            mlflow.log_metric(f"{cls}_precision", report[cls]["precision"])
            mlflow.log_metric(f"{cls}_recall", report[cls]["recall"])
            mlflow.log_metric(f"{cls}_f1", report[cls]["f1-score"])

        # Log model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(3),
            signature=signature,
            registered_model_name="FlightDelayModel"
        )

        print("✅ Model logged with MLflow")

        # Save locally
        joblib.dump(model, os.path.join(model_dir, "random_forest_model.pkl"))
        print("✅ Model saved to models/random_forest_model.pkl")
