from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def load_and_prepare():
    df = read_dataframe(URL)
    print(f"✅ Number of records after preparation: {len(df)}")  # Check filtered rows

default_args = {
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    dag_id="yellow_taxi_prepare_data",
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
    tags=["taxi", "data-prep"],
) as dag:

    prepare_data = PythonOperator(
        task_id="prepare_taxi_data",
        python_callable=load_and_prepare,
    )

    prepare_data