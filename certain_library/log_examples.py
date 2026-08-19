import os
import json
from typing import Any, Dict, List

# Utility to write example JSON files under MLflow artifacts structure.
# Usage:
#   from certain_library.log_examples import log_examples_step
#   log_examples_step(run_artifacts_root, step, examples_list)
# where examples_list is a list of dicts: {"input":..., "prediction":..., "ground_truth":...}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _sanitize_step(step: int) -> str:
    return f"{step:02d}"


def log_examples_step(artifacts_root: str, experiment_id: str, run_id: str, step: int, examples: List[Dict[str, Any]]):
    """Write examples to artifacts/<experiment>/<run>/artifacts/certain/examples/example_step_xx.json

    artifacts_root: path to mlflow artifacts root (e.g. /app/mlruns)
    experiment_id: experiment folder name under artifacts_root
    run_id: run folder name under experiment folder
    step: integer step number
    examples: list of example dicts with keys input, prediction, ground_truth, optional stage, timestamp
    """
    base = os.path.join(artifacts_root, str(experiment_id), str(run_id), "artifacts", "certain", "examples")
    _ensure_dir(base)
    filename = f"example_step_{_sanitize_step(step)}.json"
    target = os.path.join(base, filename)

    # Normalize examples into list of dicts
    normalized = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        entry = {
            "input": item.get("input"),
            "prediction": item.get("prediction", ""),
            "ground_truth": item.get("ground_truth", ""),
            "step": int(item.get("step", step)) if item.get("step") is not None else step,
            "stage": item.get("stage", "train") if item.get("stage") is not None else "train",
            "timestamp": int(item.get("timestamp", 0)) if item.get("timestamp") is not None else 0,
        }
        normalized.append(entry)

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(normalized, fh, ensure_ascii=False)

    return target
