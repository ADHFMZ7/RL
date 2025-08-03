#! /bin/bash

source .env

MLFLOW_AUTH_CONFIG_PATH=~/mlruns/auth_config.ini	uv run mlflow server \
  --backend-store-uri sqlite:////home/ahmad/mlruns/mlflow.db \
  --artifacts-destination /home/ahmad/mlruns/artifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --app-name basic-auth
