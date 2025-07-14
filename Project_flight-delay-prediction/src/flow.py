import sys
import subprocess
from prefect import flow, task

python_executable = sys.executable 

@task
def prepare_data():
    subprocess.run([python_executable, "src/data/prepare_data.py"], check=True)

@task
def make_dataset():
    subprocess.run([python_executable, "src/data/make_dataset.py"], check=True)

@task
def train_model():
    subprocess.run([python_executable, "src/models/train_model.py"], check=True)

@task
def predict_model():
    subprocess.run([python_executable, "src/models/predict_model.py"], check=True)

@flow(name="Flight Delay Prediction Pipeline")
def ml_pipeline():
    prepare_data()
    make_dataset()
    train_model()
    predict_model()

if __name__ == "__main__":
    ml_pipeline()

