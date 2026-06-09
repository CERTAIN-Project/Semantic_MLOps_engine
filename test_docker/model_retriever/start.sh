#!/usr/bin/env bash
set -euo pipefail

# Wait for MLflow artifacts folder to exist and contain something
MLFLOW_ARTIFACTS=${MLFLOW_ARTIFACTS:-/app/mlruns}
TIMEOUT=${WAIT_TIMEOUT:-600}
INTERVAL=2

echo "[model_retriever] Starting - watching for MLflow artifacts at: ${MLFLOW_ARTIFACTS}"
start=$(date +%s)
while true; do
  now=$(date +%s)
  elapsed=$((now - start))
  if [ -d "${MLFLOW_ARTIFACTS}" ]; then
    # quick check for any files
    count=$(find "${MLFLOW_ARTIFACTS}" -type f | wc -l)
    if [ "$count" -gt 0 ]; then
      echo "[model_retriever] Found artifacts (count=$count). Launching server."
      break
    fi
  fi
  if [ "$elapsed" -gt "$TIMEOUT" ]; then
    echo "[model_retriever] Timeout waiting for artifacts after ${TIMEOUT}s"
    exit 1
  fi
  sleep $INTERVAL
done

# Start the FastAPI server
exec uvicorn app:app --host 0.0.0.0 --port 8090 --log-level info
