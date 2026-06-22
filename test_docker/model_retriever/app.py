"""
model_retriever/app.py
======================
FastAPI inference server for the energy-load XGBoost model.

Lifecycle
---------
1.  A background thread (watch_for_model) polls the shared MLflow artifacts
    volume (/app/mlruns) every few seconds.
2.  It scans for a directory named "model" that contains "MLmodel" — the
    marker file that mlflow.xgboost.log_model always writes.
3.  When found it loads the model with mlflow.pyfunc.load_model(uri) where
    uri is the local filesystem path (models:/ registry is NOT needed).
4.  The /health endpoint reports model_available=true once loaded.
5.  /predict accepts a JSON list of feature dicts and returns real predictions.
6.  /logs returns the in-memory event log.
7.  /model returns metadata (run_id, artifact path, flavors, signature).
"""

import json
import logging
import os
import threading
import time
from typing import Any, Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from certain_library.tracking.manifest import recover_unfinished_runs
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MLFLOW_ARTIFACTS", "/app/mlruns")

_model: Optional[Any] = None  # loaded mlflow.pyfunc model
_model_uri: Optional[str] = None  # filesystem URI used to load it
_model_meta: dict = {}  # extra info (run_id, features …)
_deploy_logs: list = []
_lock = threading.Lock()

logger = logging.getLogger("model_retriever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Model Retriever – Energy Load XGBoost",
    description=(
        "Serves the XGBoost model trained by test_complete_workflow.py. "
        "The model is loaded automatically from the shared MLflow artifact volume "
        "as soon as training completes."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: list  # list of dicts {feature_name: value, …}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness + model readiness")
def health():
    return {
        "status": "ok",
        "model_available": _model is not None,
        "model_uri": _model_uri,
    }


@app.get("/model", summary="Model metadata")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not yet available")
    return _model_meta


@app.post("/predict", summary="Run inference")
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(
            status_code=503, detail="Model not yet available – training still running"
        )
    try:
        df = pd.DataFrame(req.features)
        preds = _model.predict(df)
        result = [float(p) for p in preds]
        _log(
            {
                "event": "predict",
                "n_rows": len(result),
                "first": result[0] if result else None,
            }
        )
        return {"predictions": result, "n": len(result)}
    except Exception as exc:
        _log({"event": "predict_error", "error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/logs", summary="In-memory deployment event log")
def logs():
    return {"logs": _deploy_logs}


# ---------------------------------------------------------------------------
# Model watcher helpers
# ---------------------------------------------------------------------------


def _log(entry: dict) -> None:
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _deploy_logs.append(entry)
    logger.info(json.dumps(entry))


def _find_model_dirs(root: str):
    """
    Walk the MLflow artifact store and yield every directory that looks like
    an mlflow.xgboost logged model (contains a file called 'MLmodel').

    MLflow stores the model at:
        <root>/<experiment_id>/<run_id>/artifacts/model/MLmodel
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        if "MLmodel" in filenames and os.path.basename(dirpath) == "model":
            yield dirpath


def _load_model(model_dir: str) -> None:
    """Load model from a local filesystem path and update global state."""
    global _model, _model_uri, _model_meta

    uri = f"file://{model_dir}"
    logger.info(f"[model_retriever] Loading model from {uri} …")
    loaded = mlflow.pyfunc.load_model(uri)

    # Parse the MLmodel YAML for metadata (PyYAML is a transitive MLflow dep)
    mlmodel_path = os.path.join(model_dir, "MLmodel")
    try:
        import yaml

        with open(mlmodel_path) as f:
            mlmodel = yaml.safe_load(f)
    except Exception:
        mlmodel = {}

    # Derive run_id from path:  …/<run_id>/artifacts/model
    parts = model_dir.split(os.sep)
    run_id = None
    for i, part in enumerate(parts):
        if i + 2 < len(parts) and parts[i + 1] == "artifacts":
            run_id = part
            break

    with _lock:
        _model = loaded
        _model_uri = uri
        _model_meta = {
            "run_id": run_id,
            "model_dir": model_dir,
            "flavors": list(mlmodel.get("flavors", {}).keys()),
            "signature": str(mlmodel.get("signature", {})),
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    _log(
        {
            "event": "model_loaded",
            "run_id": run_id,
            "uri": uri,
            "flavors": _model_meta["flavors"],
        }
    )
    logger.info(f"[model_retriever] ✅ Model ready  run_id={run_id}")


def watch_for_model(poll_interval: float = 3.0, timeout: float = 3600.0) -> None:
    """
    Background thread: poll the artifact volume until a model appears,
    then load it. Continues watching to reload if a newer model is written
    (e.g. a second training run).
    """
    _log({"event": "watcher_started", "watching": MODEL_DIR})
    start = time.time()
    last_loaded_dir: Optional[str] = None

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            _log({"event": "watcher_timeout", "elapsed_s": int(elapsed)})
            logger.warning("[model_retriever] Timed out waiting for model artifacts.")
            break

        try:
            candidates = list(_find_model_dirs(MODEL_DIR))
        except Exception as exc:
            logger.warning(f"[model_retriever] Scan error: {exc}")
            candidates = []

        if candidates:
            # Pick the most-recently modified model directory (latest training run)
            latest = max(candidates, key=lambda d: os.path.getmtime(d))
            if latest != last_loaded_dir:
                try:
                    _load_model(latest)
                    last_loaded_dir = latest
                except Exception as exc:
                    _log({"event": "load_error", "dir": latest, "error": str(exc)})
                    logger.error(f"[model_retriever] Failed to load model: {exc}")
        else:
            logger.info(
                f"[model_retriever] No model found yet, retrying in {poll_interval}s "
                f"(elapsed {int(elapsed)}s)"
            )

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Startup: launch the watcher as a daemon thread
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup_event():
    recovery = recover_unfinished_runs()
    logger.info("[model_retriever] Mirror recovery result: %s", recovery)
    t = threading.Thread(target=watch_for_model, daemon=True)
    t.start()
    logger.info("[model_retriever] Watcher thread started.")
