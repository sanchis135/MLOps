# Makefile for Flight Delay Prediction project

.PHONY: clean prepare features train predict lint format test start-api build-api run-api monitoring

## Create cleaned dataset
prepare:
	python src/data/prepare_data.py

## Generate features and split data
features:
	python src/data/make_dataset.py

## Train the model and log to MLflow
train:
	python src/models/train_model.py

## Run prediction
predict:
	python src/predict/predict_model.py

## Lint code with flake8
lint:
	flake8 src/ tests/

## Format code with black
format:
	black src/ tests/

## Run tests with pytest
test:
	pytest tests/

## Start local FastAPI server (must be inside fastapi_app directory)
start-api:
	cd deployment/fastapi_app && uvicorn main:app --reload

## Build Docker image for API
build-api:
	docker build -t flight-delay-api deployment/fastapi_app

## Run Docker container for API
run-api:
	docker run -p 8000:8000 flight-delay-api

## Run monitoring flow
mo
