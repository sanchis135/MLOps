from prefect import flow, task
import pandas as pd
from src.monitoring.evidently_report import create_evidently_report
import os

@task
def load_reference_data() -> pd.DataFrame:
    return pd.read_csv("data/processed/train.csv")

@task
def load_production_data() -> pd.DataFrame:
    return pd.read_csv("data/processed/production_recent.csv")

@task
def save_report():
    os.makedirs("reports", exist_ok=True)

@flow(name="Model Monitoring Flow")
def monitor_model():
    save_report()
    reference_data = load_reference_data()
    production_data = load_production_data()
    create_evidently_report(reference_data, production_data, "reports/evidently_report.html")
    print("Monitoring report generated at reports/evidently_report.html")

if __name__ == "__main__":
    monitor_model()

