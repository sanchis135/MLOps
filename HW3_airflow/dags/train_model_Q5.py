import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import mlflow.sklearn

def train_model():
    # Leer y preparar datos
    df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X = dv.fit_transform(dicts)

    y = df.duration.values

    lr = LinearRegression()
    lr.fit(X, y)

    print(f"✅ Model intercept: {lr.intercept_:.2f}")

    # Carpeta donde guardar modelo localmente (por si lo quieres)
    model_dir = "/opt/airflow/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f_out:
        pickle.dump((dv, lr), f_out)

    # Configurar tracking local (opcional)
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_experiment("nyc-taxi-experiment")

    # Guardar con MLflow
    with mlflow.start_run():
        mlflow.set_tag("developer", "tu_nombre")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("vectorizer", "DictVectorizer")

        mlflow.log_metric("intercept", lr.intercept_)

        mlflow.sklearn.log_model(lr, artifact_path="model")
        mlflow.log_artifact(model_path, artifact_path="pickle_model")

    return model_path

default_args = {
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    dag_id="yellow_taxi_train_model",
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
    tags=["taxi", "train"],
) as dag:

    train_model_op = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    train_model_op

