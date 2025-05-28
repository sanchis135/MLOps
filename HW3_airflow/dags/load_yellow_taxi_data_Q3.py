from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd

# Dataset URL
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

def load_and_count_records():
    df = pd.read_parquet(URL)
    print(f"✅ Number of records loaded: {len(df)}")  # This will appear in the logs

default_args = {
    "start_date": datetime(2023, 1, 1),
}

with DAG(
    dag_id="yellow_taxi_load_count",
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
    tags=["taxi", "data-load"],
) as dag:

    count_records = PythonOperator(
        task_id="count_taxi_records",
        python_callable=load_and_count_records,
    )

    count_records
