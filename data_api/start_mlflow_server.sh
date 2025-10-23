#!/bin/bash

export MLFLOW_BACKEND_URI="postgresql+psycopg2://postgres:postgres@localhost:5432/mlflow"
export ARTIFACT_PATH="file:./mlruns"

mlflow server \
  --backend-store-uri $MLFLOW_BACKEND_URI \
  --default-artifact-root $ARTIFACT_PATH \
  --host 0.0.0.0 --port 5000