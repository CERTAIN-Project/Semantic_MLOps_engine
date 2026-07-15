#!/usr/bin/env python3
"""
Simplified Complete ML workflow example variant that logs per-step training metrics
instead of Optuna trials. This is intended as a companion test to
`test_complete_workflow.py` demonstrating 'steps' during training.
"""

from certain_library.tracking.tracker import tracker
import os
import time
import mlflow
import pandas as pd
import numpy as np

from typing import Dict, Union
from sklearn.metrics import mean_squared_error

from certain_library.train_monitor.log_metrics import log_metrics
from certain_library.train_monitor.log_model import (
    log_model_info,
    log_model_architecture,
    log_model_hyperparameters,
    log_model_signature,
)
from certain_library.data_analysis.log_whylogs import log_whylogs_profile
from Semantic_MLOps_engine.certain_library.log_basic.log_params import (
    log_param,
    log_params,
)
from certain_library.data_analysis.log_dataset import (
    log_dataset,
    log_train_test_dataset,
)
from certain_library.resource_monitor.resource import start_tracker, stop_tracker
from certain_library.metadata.artifact_metadata import (
    save_experiment_tags_as_artifact,
    collect_runtime_environment,
    save_runtime_env_as_artifact,
    save_dataset_manifest,
)
from certain_library.compliance.log_governance import (
    log_risk,
    log_human_oversight,
    log_transparency_measure,
    log_change,
)
from certain_library.compliance.log_experiment_governance import (
    log_ai_actors,
    log_labeling_procedures,
)
from certain_library.compliance.log_documentation import (
    log_declarations_of_conformity,
    log_visual_documentations,
    log_explainable_ai,
)
from certain_library.compliance.log_deployment import (
    log_model_packaging,
    log_build_testing,
    log_standards,
    log_interface,
    log_decommissioning,
)
from certain_library.train_monitor.log_examples import log_examples
from certain_library.train_monitor.log_weights import log_weight_distribution
from certain_library.data_analysis.log_tokenizer import (
    log_tokenizer_config,
    log_tokenization_stats,
)
from certain_library.data_analysis.log_drift_metrics import log_drift_metrics
from certain_library.data_analysis.log_data_techniques import log_data_techniques

# Simple dummy data generator (small)


def _make_dummy_energy_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2018-01-01", periods=n_rows, freq="h", tz="UTC")
    load = (
        50000 + 2000 * np.sin(np.linspace(0, 6.28, n_rows)) + rng.normal(0, 200, n_rows)
    )
    solar = np.clip(8000 * np.sin(np.linspace(0, 3.14, n_rows)), 0, None) + rng.normal(
        0, 50, n_rows
    )
    price = 45 + 0.0003 * (load - load.mean()) + rng.normal(0, 5, n_rows)
    df = pd.DataFrame(
        {
            "utc_timestamp": timestamps,
            "DE_load_actual_entsoe_transparency": load,
            "DE_solar_generation_actual": solar,
            "DE_price_day_ahead": price,
        }
    )
    return df


print("=" * 60)
print("Complete ML Workflow (steps variant)")
print("=" * 60)

experiment_name = "complete_ml_workflow_demo_steps"
tracker.set_experiment(experiment_name)

with tracker.start_run(run_name="rf_steps_training") as run:
    print(f"Run started: {run.info.run_id}")

    # basic metadata
    try:
        save_experiment_tags_as_artifact(experiment_name, {"variant": "steps"})
    except Exception:
        pass

    try:
        env = collect_runtime_environment()
        save_runtime_env_as_artifact(env)
    except Exception:
        pass

    # resource tracking for data processing
    tracker_data, output_location = start_tracker(output_file_name="emissions_data")

    # generate data
    df = _make_dummy_energy_df(n_rows=500)
    # persist lightweight dataset metadata
    try:
        os.makedirs("data", exist_ok=True)
        raw_path = os.path.join("data", "raw_dataset_steps.csv")
        df.to_csv(raw_path, index=False)
        save_dataset_manifest(
            run_id=run.info.run_id, files_or_path=raw_path, write_manifest=False
        )
    except Exception:
        pass

    log_params({"num_rows": df.shape[0]})

    # stop initial resource tracker
    stop_tracker(tracker_data, output_location)

    # prepare train/test split
    from sklearn.model_selection import train_test_split

    y = df["DE_load_actual_entsoe_transparency"]
    X = (
        df.select_dtypes(include=[np.number]).drop(
            columns=["DE_load_actual_entsoe_transparency"]
        )
        if "DE_load_actual_entsoe_transparency" in df.columns
        else df.select_dtypes(include=[np.number])
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    train_combined = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_combined = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )
    log_train_test_dataset(train_combined, test_combined)

    # Log a dummy search space artifact (kept for compatibility)
    try:
        from certain_library.train_monitor.log_metrics import log_search_space

        search_space = {"max_steps": [5, 10, 20], "learning_rate": [0.01, 0.1]}
        log_search_space(search_space)
    except Exception:
        pass

    # final resource tracking for training
    tracker_train, output_location_train = start_tracker(
        output_file_name="emissions_train"
    )
    # Instead of Optuna trials, run a step-based training loop
    max_steps = 1000
    for step in range(max_steps):
        # Simulate training: produce a decreasing MSE and some resource metrics
        fake_pred = np.full(len(y_test), y_test.mean()) + np.random.normal(
            0, 1.0 / (step + 1), len(y_test)
        )
        mse = mean_squared_error(y_test, fake_pred)
        # Log metrics with step
        log_metrics({"mse": float(mse), "step_rmse": float(np.sqrt(mse))}, step=step)

        # Optionally log simple model info at the last step
        if step == max_steps - 1:
            try:
                log_model_info(
                    {"model_type": "dummy_rf_steps", "trained_steps": max_steps}
                )
            except Exception:
                pass

        # periodically log additional artifacts and governance info
        if step % 20 == 0:
            try:
                # example predictions logging
                sample_inputs = X_test.head(3).to_dict(orient="records")
                log_examples(
                    sample_inputs, [0, 1, 2], [0, 1, 2], step=step, stage="train"
                )
            except Exception:
                pass

            try:
                # weight distribution (dummy model passed as None -> unknown_layer)
                log_weight_distribution(model=None, step=step, stage="train")
            except Exception:
                pass

    stop_tracker(tracker_train, output_location_train)

    try:
        # governance - risk and oversight
        log_risk(
            [
                {
                    "risk_description": "example risk",
                    "risk_type": "data_bias",
                    "risk_level": 0.2,
                }
            ]
        )
        log_human_oversight(
            [
                {
                    "oversight_type": "human-in-the-loop",
                    "description": "manual review",
                }
            ]
        )
        log_transparency_measure(
            [
                {
                    "measure_type": ["model_card"],
                    "measure_value": ["s3://model_card.pdf"],
                }
            ]
        )
    except Exception:
        pass

    try:
        log_change(
            [
                {
                    "change_description": "periodic checkpoint",
                    "changed_by": "tester",
                }
            ]
        )
    except Exception:
        pass

    try:
        log_data_techniques({"scaling": {"method": "minmax"}})
    except Exception:
        pass

    # write a tiny model artifact and hyperparameters
    try:
        log_model_hyperparameters({"max_steps": max_steps, "learning_rate": 0.1})
    except Exception:
        pass

    print("Finished steps-based training run")


if __name__ == "__main__":
    # allow running as a script
    pass
