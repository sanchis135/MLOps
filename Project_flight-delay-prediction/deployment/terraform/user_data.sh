#!/bin/bash
sudo apt update -y
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# Optional: clone your repo
git clone https://github.com/sanchis135/MLOps/flight-delay-prediction.git
cd flight-delay-prediction/deployment/fastapi_app

# Build and run the container
docker build -t flight-delay-api .
docker run -d -p 8000:8000 flight-delay-api
