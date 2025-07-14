# ✈️ Flight Delay Prediction

An end-to-end Machine Learning project to predict whether a flight will be delayed or not using real-world U.S. domestic flight data.

---

## 📌 Problem Description

Flight delays are a major logistical and economic issue for airlines, airports, and passengers. This project aims to build a predictive model that determines whether a flight will be delayed based on information available before takeoff, such as scheduled departure time, airline, and airport.

Potential use cases:
- Airlines optimizing operational planning
- Airports improving gate and staff management
- Travelers making more informed travel decisions

---

## 📊 Dataset

The dataset is sourced from [Kaggle - US Domestic Flights Delay Prediction (2013 - 2018)](https://www.kaggle.com/datasets/gabrielluizone/us-domestic-flights-delay-prediction-2013-2018?resource=download)

This dataset contains scheduled and actual departure and arrival times reported by certified US air carriers that account for at least 1 percent of domestic scheduled passenger revenues. The data was collected by the U.S. Office of Airline Information, Bureau of Transportation Statistics (BTS). The dataset contains date, time, origin, destination, airline, distance, and delay status of flights for flights between 2013 and 2018.

---

## 🔧 Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Model training | Scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Orchestration | Prefect |
| Deployment | FastAPI, Docker, Render or AWS EC2 |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions |
| Infrastructure as Code | Terraform |
| Testing | Pytest |
| Code quality | Black, Flake8 |
| Pre-commit hooks | Pre-commit |

---

## 🛠️ Project Structure

```bash
flight-delay-prediction/
│
├── data/                  # Raw and processed data
├── notebooks/             # EDA and experimentation
├── src/                   # Source code (data, features, models)
├── docker/                # Docker config
├── deployment/            # FastAPI + Terraform
├── tests/                 # Unit and integration tests
├── mlruns/               
├── models/               
├── reports/               # metrics
├── .github/               # CI/CD workflows
├── Makefile
├── requirements.txt
├── pre-commit-config.yaml
└── README.md
```

## EDA Conclusions and Insights

After conducting exploratory data analysis on the US domestic flights dataset (2013–2018), I uncovered several key findings that help understand flight delay patterns:

### 1. Delay Distribution and Target Variable
- Approximately **X%** of flights experienced delays, as indicated by the binary `is_delay` feature.
- Delays vary widely in magnitude, with some flights delayed over 2 hours.

### 2. Numerical Feature Correlations
- There is a **strong positive correlation** between `ArrDelay` and `ArrDelayMinutes`, confirming these represent the same delay duration.
- **Flight distance** shows a weak correlation with delay likelihood, suggesting delays are not strongly dependent on trip length.
- Scheduled departure hour (`CRSDepHour`) exhibits a moderate correlation with delay probability, hinting that certain times of day are more prone to delays.

### 3. Seasonality Effects
- **Monthly patterns:** Delay rates tend to increase during summer months (June to August) and during winter holiday seasons, likely due to higher air traffic and weather disruptions.
- **Day of week:** Delays are generally more frequent towards the end of the week and weekends, possibly due to increased travel demand.

### 4. Categorical Features Overview
- Airlines, origin and destination airports, and states play significant roles in delay patterns. Some carriers and airports consistently show higher delay rates.
- These categorical variables will be important for feature engineering in predictive modeling.

During the initial exploratory data analysis of the US domestic flights delay dataset (2013-2018), several key insights were observed:

Delay Distribution: About 30-40% of the flights in the dataset experienced delays (marked by is_delay = 1). This indicates that delays are a common occurrence and not a rare event.

Correlation Insights: The correlation heatmap shows that ArrDelay and ArrDelayMinutes are strongly correlated with the delay label (is_delay). Other factors like Cancelled and Diverted also show expected correlations. Interestingly, departure time (CRSDepHour) shows mild correlation with delays, suggesting certain times of day may be more prone to delays.

Seasonality Effects: Delay rates vary throughout the year, with some months showing higher average delays. The data also reveals that delays are more frequent on certain days of the week, potentially related to weekend traffic or operational schedules.

Airline and Airport Impact: Some airlines and airports have noticeably higher delay rates compared to others. The top 10 airlines and origin airports by flight count show significant variability in delay probabilities, which could relate to their operational efficiencies or route difficulties.

These insights guide feature selection and modeling strategies, emphasizing the importance of temporal features, airline/airport categorical data, and cancellation/diversion flags in predicting flight delays.

## How to Run

### 1. Prepare Clean Data (prepare_data.py)
Initial cleaning, type conversion, and saving the cleaned dataset as flights_clean.csv.

```bash
python src/data/prepare_data.py
```

### 2. Create Features and Datasets (make_dataset.py)
Load flights_clean.csv, generate features, split into training and test sets, and save as train.csv and test.csv.

Run:
```bash
python src/data/make_dataset.py
```
### 3. MLFlow
Install MLFlow:
```bash
pip install mlflow
```

### 4. Train Model (train_model.py) with MLflow
Load train.csv and test.csv, train and evaluate the model, and save the trained model.

Run:
```bash
python src/models/train_model.py
```

In another terminal, start the MLflow UI:

```bash
mlflow ui --port 5000
```

Open http://localhost:5000, and explore your experiment(s).


### 5. Predict on test or new data
Predict new data.

Run:
```bash
python src/predict/predict_model.py
```

### 6. Pipeline basic using Prefect
Install Prefect:

```bash
pip install prefect
```

Create flow.py in src/.

Run:
```bash
python src/flow.py
```

Prefect will execute the steps in order, and show you logs and status.

## Deployment (FastAPI + Docker)
Add this section under your existing README.md — ideally after the "Predict on test or new data" section.

### Model Deployment (FastAPI + Docker)
This project includes a FastAPI-based web service that allows you to serve the trained model as an HTTP API.

### Structure
The deployment code is located in:

```bash
deployment/
└── fastapi_app/
    ├── main.py               # FastAPI application
    ├── model_loader.py       # Load and use the trained model
    ├── model.pkl             # Trained model (copied manually)
    ├── Dockerfile            # Container definition
    └── requirements.txt      # API dependencies
```
### Example API Input
Example of the JSON body to send to the /predict endpoint:

```json
{
  "month": 7,
  "day": 14,
  "day_of_week": 1,
  "dep_hour": 16,
  "airline": "DL",
  "origin": "ATL",
  "destination": "LAX",
  "distance": 1946.0
}
```

### Run with Docker (Locally)
Make sure the trained model (model.pkl) is copied into the fastapi_app/ folder.

From the project root:

```bash
cd deployment/fastapi_app
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```
Access the API documentation at: http://localhost:8000/docs

## Cloud Deployment with Terraform (AWS EC2)
This project supports cloud deployment using AWS EC2 and Terraform. The trained model is served via a FastAPI web service inside a Docker container.

### Infrastructure Overview
Terraform is used to provision:

An EC2 instance (Ubuntu)

A security group that opens port 8000

A startup script that installs Docker, clones this repository, builds the Docker image, and starts the FastAPI service

### Folder Structure
Terraform configuration is located in:

```bash
deployment/
└── terraform/
    ├── main.tf               # AWS provider, EC2 instance, security group
    ├── variables.tf          # Instance type, region, SSH key pair
    ├── outputs.tf            # Outputs public IP
    └── user_data.sh          # Startup script to run the API
```
### How to Deploy
1. Prerequisites
AWS CLI configured (aws configure)

A valid SSH key pair in your AWS account

Terraform installed (terraform -version)

Your repository publicly accessible or SSH accessible

2. Deploy with Terraform
From the deployment/terraform/ directory:

```bash
terraform init
terraform plan -var="key_name=your_ssh_key_pair"
terraform apply -var="key_name=your_ssh_key_pair"
```
After a few minutes, Terraform will output the public IP of your EC2 instance.

### Access the API
Open your browser and navigate to:

http://<EC2-public-IP>:8000/docs

You will see the Swagger UI of the FastAPI app with a /predict endpoint.

### Teardown (optional)
To destroy the infrastructure:

```bash
terraform destroy -var="key_name=your_ssh_key_pair"
```
Deployment to the cloud is now fully automated and reproducible.

## Model Monitoring with Evidently and Prefect
To ensure the deployed model maintains high performance over time, this project implements continuous monitoring using Evidently AI and Prefect.

### How Monitoring Works
Evidently AI generates detailed reports tracking data drift, model performance, and feature statistics by comparing recent production data against the training dataset.

Prefect orchestrates the monitoring workflow, running the Evidently report generation automatically on a schedule (e.g., daily or weekly).

Alerts or further actions (like retraining) can be triggered based on detected issues or metric thresholds.

### Monitoring Implementation
src/monitoring/evidently_report.py: Contains the code to create Evidently monitoring reports.

src/monitoring/prefect_flow_monitoring.py: Defines a Prefect flow that loads data, runs the report, and saves it for review.

### How to Run Monitoring Locally
Ensure dependencies are installed:

```bash
pip install evidently prefect pandas
```
Prepare recent production data in data/production_recent.csv.

Run the Prefect monitoring flow:

```bash
python src/monitoring/prefect_flow_monitoring.py
```
Open the generated report at reports/evidently_report.html to inspect data drift and performance metrics.

## Makefile Commands
This project includes a Makefile to simplify and standardize common operations across development, training, and deployment.

Available Commands:

| Command           | Description                                     |
| ----------------- | ----------------------------------------------- |
| `make prepare`    | Clean and prepare raw dataset                   |
| `make features`   | Generate features and split train/test datasets |
| `make train`      | Train the model and track it with MLflow        |
| `make predict`    | Run predictions on test or new data             |
| `make lint`       | Run Flake8 linter on source and test files      |
| `make format`     | Format code using Black                         |
| `make test`       | Run all unit and integration tests              |
| `make start-api`  | Run FastAPI server locally                      |
| `make build-api`  | Build Docker image for the FastAPI service      |
| `make run-api`    | Run the FastAPI service in a Docker container   |
| `make monitoring` | Trigger model monitoring flow with Prefect      |

The Makefile is located at the project root and helps enforce reproducibility and development best practices.

## Testing
This project includes basic unit tests to ensure the reliability of key components in the ML pipeline.

### What is Tested?
| File                                  | Purpose                                                                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `tests/data/test_make_dataset.py`     | Verifies that the feature engineering and data splitting logic runs correctly and outputs valid `train.csv` and `test.csv` files.   |
| `tests/models/test_train_model.py`    | Ensures that the model training script trains and saves a model file successfully, and that the trained model can make predictions. |
| `tests/predict/test_predict_model.py` | Validates that the prediction script loads a model and outputs predictions of the correct format and length.                        |


### How to Run Tests
Install test dependencies (if not already):

```bash
pip install pytest
```
Then run all tests with:

```bash
$env:PYTHONPATH="." ; pytest tests/
```
You should see output similar to:

```bash
================== test session starts ==================
platform win32 -- Python 3.12.0, pytest-8.4.1, pluggy-1.6.0

rootdir: E:\MLOps\Project\flight-delay-prediction

plugins: anyio-4.9.0, Faker-37.4.0

collected 5 items

tests\data\test_make_dataset.py ..    [ 40%]
tests\data\test_prepare_data.py .     [ 60%] 
tests\models\test_train_model.py .    [ 80%]
tests\predict\test_predict_model.py . [100%]   
================== 5 passed in 3.43s ===================
```

## Code Quality & Continuous Integration

### Pre-commit Hooks

To maintain consistent code style and catch common issues early, this project uses [pre-commit](https://pre-commit.com/) with standard Python hooks including:

- Trailing whitespace removal  
- End-of-file fixes  
- YAML validation  
- Large file checks  
- Code formatting with Black  
- Linting with Flake8  

**Setup**

1. Install pre-commit:

```bash
pip install pre-commit
```

2. Install git hooks:

```bash
pre-commit install
```

3. Hooks will run automatically on each commit to enforce code quality.

## Project Checklist
This checklist summarizes how this project meets the required evaluation criteria.
Feel free to use it during peer review!

✅	Criteria	Details
☑️	Problem clearly described	Real-world use case: Predicting flight delays using pre-departure info
☑️	Cloud used	Model deployed on AWS EC2 using Terraform and Docker
☑️	Infrastructure as Code (IaC)	Infrastructure provisioned via Terraform
☑️	Experiment tracking	MLflow used to track experiments and model metrics
☑️	Model registry	Trained models saved and logged in MLflow
☑️	Workflow orchestration	Prefect flows for both training and monitoring
☑️	Model deployed (cloud or local)	FastAPI app containerized and deployed to EC2
☑️	Model monitoring	Evidently AI used with Prefect to monitor data drift and performance
☑️	Reproducibility	Clear instructions, requirements.txt, structured project layout
☑️	Unit tests	Implemented with Pytest (tests/ folder)
☑️	Integration tests	End-to-end API tested in FastAPI deployment
☑️	Linter / Code formatter	Flake8 and Black used for code quality
☑️	Makefile	Included for common development commands
☑️	Pre-commit hooks	Configured with standard Python hooks
☑️	CI/CD pipeline	GitHub Actions used to automate testing and linting
