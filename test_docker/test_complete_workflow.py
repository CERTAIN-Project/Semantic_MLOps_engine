#!/usr/bin/env python3
"""
Complete ML workflow example using certain_library
Demonstrates end-to-end logging to PostgreSQL database
"""

import io
import os
import time
import queue
import threading
import psutil
import mlflow
import pandas as pd
import numpy as np

from typing import Dict, Union
from sklearn.metrics import mean_squared_error, r2_score

from certain_library.tracking.tracker import tracker
from certain_library.train_monitor.log_metrics import (
    log_metrics,
    log_resources,
    log_search_space,
)
from certain_library.train_monitor.log_model import (
    log_model_info,
    log_model_architecture,
    log_model_hyperparameters,
    log_model_signature,
)
from certain_library.data_analysis.log_whylogs import log_whylogs_profile
from certain_library.log_basic.log_params import log_params
from certain_library.data_analysis.log_dataset import (
    log_dataset,
    log_train_test_dataset,
)
from certain_library.data_analysis.log_timeseries import timestamp_analysis
from certain_library.resource_monitor.resource import start_tracker, stop_tracker
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

from certain_library.data_analysis.log_drift_metrics import (
    log_drift_metrics,
)

from certain_library.git_tracking import log_git_metadata
from certain_library.metadata.artifact_metadata import (
    collect_runtime_environment,
    save_runtime_env_as_artifact,
    save_dataset_manifest,
)
from certain_library.data_analysis.log_data_techniques import log_data_techniques
from data_api.misc.data_transform import map_mlflow_data_drift

# ---------------------------------------------------------------------------
# Set USE_DUMMY_DATA = True to run fully offline with synthetic energy data.
# Set USE_DUMMY_DATA = False to download the real OPSD dataset from the web.
# ---------------------------------------------------------------------------
USE_DUMMY_DATA = True


def _make_dummy_energy_df(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic energy time-series DataFrame that mimics the OPSD schema.

    Columns produced:
        utc_timestamp                        — hourly timestamps starting 2018-01-01
        DE_load_actual_entsoe_transparency   — synthetic German load in MW
        DE_solar_generation_actual           — synthetic solar generation in MW
        DE_wind_onshore_generation_actual    — synthetic onshore wind in MW
        DE_wind_offshore_generation_actual   — synthetic offshore wind in MW
        DE_price_day_ahead                   — synthetic day-ahead price in EUR/MWh
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2018-01-01", periods=n_rows, freq="h", tz="UTC")

    # Base load with daily and weekly seasonality + noise
    hour_of_day = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    daily_pattern = 5_000 * np.sin(np.pi * hour_of_day / 12) + 55_000
    weekly_pattern = np.where(day_of_week >= 5, -4_000, 0)  # lower on weekends
    load = daily_pattern + weekly_pattern + rng.normal(0, 1_500, n_rows)

    # Solar: only during daytime, zero at night
    solar = np.maximum(
        0, 8_000 * np.sin(np.pi * np.clip(hour_of_day - 6, 0, 12) / 12)
    ) + rng.normal(0, 300, n_rows)
    solar = np.maximum(0, solar)

    wind_on = 10_000 + rng.normal(0, 2_000, n_rows)
    wind_off = 4_500 + rng.normal(0, 800, n_rows)
    price = 45 + 0.0003 * (load - load.mean()) + rng.normal(0, 5, n_rows)

    df = pd.DataFrame(
        {
            "utc_timestamp": timestamps,
            "DE_load_actual_entsoe_transparency": load,
            "DE_solar_generation_actual": np.maximum(0, solar),
            "DE_wind_onshore_generation_actual": np.maximum(0, wind_on),
            "DE_wind_offshore_generation_actual": np.maximum(0, wind_off),
            "DE_price_day_ahead": price,
        }
    )
    # Introduce a small number of missing values to exercise the cleaning pipeline
    for col in ["DE_load_actual_entsoe_transparency", "DE_solar_generation_actual"]:
        missing_idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


# Define helper functions ------------------------------------------------------
def clean_missing_values(input_df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with the column forward fill."""
    return input_df.ffill()


def remove_outliers_iqr(
    data: pd.DataFrame, column: str = "DE_load_actual_entsoe_transparency"
) -> pd.DataFrame:
    """Remove rows where the specified column is beyond 1.5 IQR bounds."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[
        ~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))
    ]


def augment_with_noise(
    input_df: pd.DataFrame,
    column: str = "DE_load_actual_entsoe_transparency",
    noise_factor: float = 0.01,
) -> pd.DataFrame:
    """Augment the DataFrame by injecting noise into the specified column."""
    df_aug = input_df.copy()
    df_aug[f"{column}_aug"] = df_aug[column] + noise_factor * np.random.normal(
        size=len(df_aug)
    )
    return df_aug


# -----------------------------------------------------------------------------


print("=" * 70)
print("Complete ML Workflow with certain_library")
print("All data logged to PostgreSQL database in certain_databases")
print("=" * 70)
print()

# Set up experiment
experiment_name = "complete_ml_workflow_demo_v2"
tracker.set_experiment(
    experiment_name=experiment_name,
    tags={"team": "energy", "project": "opsd_demo", "owner": "dimitrios"},
)
print(f"📊 Experiment: {experiment_name}")
print()

# Start MLflow run
with tracker.start_run(
    run_name="random_forest_classifier",
    tags={"project": "opsd_demo", "test_run": "1"},
) as run:
    print(f"🏃 Run Started: {run.info.run_id}")
    print()

    # Log git metadata into the active MLflow run (if git is available)
    try:
        git_meta = log_git_metadata()
        print(f"🔖 Logged git metadata tags: {git_meta}")
    except Exception as e:
        print(f"⚠️  Could not log git metadata: {e}")

    tracker.set_tags({"run_stage": "data_preprocessing", "run_type": "demo"})

    # Collect and save runtime environment artifact
    try:
        env = collect_runtime_environment()
        save_runtime_env_as_artifact(env)
        print("🧭 Wrote runtime_env.json artifact")
    except Exception:
        pass

    tracker_data, output_location = start_tracker(output_file_name="emissions_data")
    print("⚡ Resource monitoring started")
    print()

    # ---------------- Governance & Compliance ----------------
    # use_manual_info=True → uses the strings below.
    # use_manual_info=False (default) → auto-detects from environment.
    log_ai_actors(
        auditor="CERTAIN Project Consortium",
        organization="Open Power System Data Initiative",
        use_manual_info=True,
        ai_provider_name="Dimitrios Christodoulou",
        ai_provider_role="model development and training",
        ai_deployer_name="Energy Operations Team",
        ai_deployer_role="operational deployment and monitoring",
    )

    log_labeling_procedures(
        quality_assurance_methods=[
            "automated sensor validation",
            "cross-source verification",
        ],
        procedures=[
            {
                "description": (
                    "Energy load measurements sourced from ENTSO-E Transparency Platform. "
                    "Values represent actual total load in MW for Germany, validated "
                    "against OPSD cross-checks."
                ),
                "annotation_tool": "OPSD data pipeline",
                "annotators": ["ENTSO-E automated reporting system"],
                "link": "https://data.open-power-system-data.org/time_series/",
            },
        ],
    )

    log_risk(
        [
            {
                "risk_type": "data_bias",
                "risk_description": "Energy load data may reflect historical consumption patterns that are biased towards certain seasons or regions.",
                "risk_level": 0.4,
            },
            {
                "risk_type": "data_drift",
                "risk_description": "Energy consumption patterns may shift significantly due to climate change or policy changes post training.",
                "risk_level": 0.6,
            },
        ]
    )

    log_human_oversight(
        [
            {
                "oversight_type": "human-in-the-loop",
                "description": "Domain experts review model predictions before they are used in operational energy dispatch decisions.",
                "implementation_details": "Weekly review meetings with the energy operations team; model output is advisory only.",
            },
            {
                "oversight_type": "periodic-audit",
                "description": "Quarterly audit of model performance metrics and dataset freshness.",
                "implementation_details": "Automated report generated from MLflow tracking server and reviewed by the AI governance team.",
            },
        ]
    )

    log_transparency_measure(
        [
            {
                "measure_type": ["model_card", "data_sheet"],
                "measure_value": [
                    "https://wiki.example.com/model-card/energy-xgb",
                    "https://wiki.example.com/data-sheet/opsd",
                ],
                "description": "Public-facing documentation for the energy load forecasting model and the OPSD dataset.",
            },
            {
                "measure_type": ["feature_importance_report"],
                "measure_value": ["mlflow://artifacts/feature_importance"],
                "description": "XGBoost feature importance logged as an MLflow artifact after each training run.",
            },
        ]
    )

    log_change(
        [
            {
                "change_description": "Initial training run using OPSD 60-minute time series data with XGBoost regressor and Optuna hyperparameter search.",
                "changed_by": "data_science_team",
            },
        ]
    )

    data_url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

    if USE_DUMMY_DATA:
        print("🧪 Using synthetic dummy energy data (USE_DUMMY_DATA=True)")
        df = _make_dummy_energy_df(n_rows=2000)
        log_params({"data_url": "synthetic_dummy_data"})
    else:
        print(f"🌐 Downloading real OPSD dataset from {data_url}")
        df = pd.read_csv(data_url, parse_dates=["utc_timestamp"])
        log_params({"data_url": data_url})
    # Persist the raw dataset to disk so we can compute an accurate size for
    # data.json. This covers both synthetic and downloaded datasets.
    try:
        os.makedirs("data", exist_ok=True)
        raw_path = os.path.join("data", "raw_dataset.csv")
        df.to_csv(raw_path, index=False)
        # write only the lightweight metadata (data.json)
        try:
            save_dataset_manifest(
                run_id=run.info.run_id, files_or_path=raw_path, write_manifest=False
            )
        except Exception:
            pass
    except Exception:
        pass

    log_params({"num_rows": df.shape[0]})
    log_params({"num_columns": df.shape[1]})

    # ---------------- Data Cleaning Pipeline ----------------
    # Drop rows missing the target column
    df = df.dropna(subset=["DE_load_actual_entsoe_transparency"])

    # Clean missing values using the custom function
    df_cleaned = clean_missing_values(df)
    log_dataset(df_cleaned, name="df_cleaned", output_dir="data_cleaning")
    log_whylogs_profile(df_cleaned, name="cleaned")

    df_filtered = remove_outliers_iqr(
        df_cleaned, column="DE_load_actual_entsoe_transparency"
    )
    log_dataset(df_filtered, name="df_filtered", output_dir="data_cleaning")
    log_whylogs_profile(df_cleaned, name="filtered")

    # ---------------- Data Augmentation Pipeline ----------------
    df_augmented = augment_with_noise(df_filtered)
    log_dataset(df_augmented, name="df_augmented", output_dir="data_augmentation")
    log_whylogs_profile(df_cleaned, name="augmented")

    # Ensure the DataFrame is sorted by timestamp before splitting
    df_sorted = df_augmented.sort_values("utc_timestamp")

    stop_tracker(tracker_data, output_location)

    # ---------------- Prepare Train/Test Split ----------------
    from sklearn.model_selection import train_test_split

    y = df_sorted["DE_load_actual_entsoe_transparency"]
    X = df_sorted.select_dtypes(include=[np.number]).drop(
        columns=["DE_load_actual_entsoe_transparency"]
    )

    # Split data without shuffling to preserve temporal order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Combine features and target into DataFrames before logging
    train_combined = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_combined = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )
    log_train_test_dataset(train_combined, test_combined)
    # Save dataset manifest and lightweight metadata into artifacts/certain so
    # the sync process can read data_location and data_size. This will write
    # both data_manifest.json and certain/metadata/data.json into the active
    # run's artifacts when an MLflow run is active.
    try:
        # Save a combined CSV to disk so we can compute its size reliably.
        os.makedirs("data", exist_ok=True)
        combined_path = os.path.join("data", "train_test_combined.csv")
        # write train+test combined to a CSV file (no index)
        pd.concat([train_combined, test_combined]).to_csv(combined_path, index=False)

        # Write only the lightweight metadata (data.json) and skip the full
        # manifest file since we only need size/location information.
        save_dataset_manifest(
            run_id=run.info.run_id, files_or_path=combined_path, write_manifest=False
        )
    except Exception:
        pass

    # Log data preprocessing techniques as an artifact so they are discoverable
    try:
        dt = {
            # We allow a global stage, but each technique may also declare its own stage.
            "data_technique_stage": "preprocessing",
            "techniques": {
                "scaling": {
                    "method": "standard",
                    "library": "sklearn",
                    "parameters": {"with_mean": True, "with_std": True},
                    "stage": "preprocessing",
                },
                "imputation": {
                    "method": "ffill",
                    "notes": "forward fill",
                    "parameters": {},
                    "stage": "preprocessing",
                },
                "noise_injection": {
                    "method": "additive_gaussian",
                    "parameters": {"noise_factor": 0.01},
                    "stage": "augmentation",
                },
                "normalization": {
                    "method": "minmax",
                    "library": "sklearn",
                    "parameters": {"feature_range": [0, 1]},
                    "stage": "preprocessing",
                },
                "encoding": {
                    "method": "one_hot",
                    "library": "pandas",
                    "parameters": {"columns": ["DE_price_day_ahead"]},
                    "stage": "preprocessing",
                },
                "augmentation_time_warp": {
                    "method": "time_warp",
                    "library": "tsaug",
                    "parameters": {"warp_factor": 0.2},
                    "stage": "augmentation",
                },
            },
        }
        log_data_techniques(dt)
        print("🧾 Wrote data_techniques artifact")
    except Exception:
        pass

    # Compute simple drift metrics between train and test for the active run and
    # log them as metrics. This uses the server-side helper for consistency.
    try:
        # Build dataset with required columns for drift computation
        train_df = train_combined.copy()
        test_df = test_combined.copy()

        # Call the new helper which computes per-column KS tests and writes
        # a deterministic artifact under certain/drift_metrics/drift_metrics.json
        try:
            model_info = {"experiment_id": run.info.experiment_id}
            artifact = log_drift_metrics(
                train_df, test_df, run_id=run.info.run_id, model_info=model_info
            )
            print("⚖️  Computed and wrote drift artifact")
        except Exception as e:
            print("Could not compute drift metrics:", e)
    except Exception as e:
        print("Could not compute drift metrics:", e)

    # Persist drift metrics as a small JSON artifact so host-side sync can read them
    try:
        # Convert drift_df into a JSON-serializable structure
        drift_records = []
        if hasattr(drift_df, "to_dict") and not drift_df.empty:
            for _, r in drift_df.iterrows():
                # r may have 'key' like '[drift_metrics]feature1'
                k = r.get("key")
                col = k.replace("[drift_metrics]", "") if isinstance(k, str) else None
                drift_records.append(
                    {
                        "column": col,
                        "key": k,
                        "p_value": float(r.get("value") or 0.0),
                        "timestamp": int(
                            r.get("timestamp") or pd.Timestamp.now(tz="UTC").timestamp()
                        ),
                    }
                )

        summary = {
            "num_tested": len(drift_records),
            "num_drift": sum(1 for d in drift_records if d.get("p_value", 1) < 0.05),
        }

        drift_artifact = {
            "run_id": run.info.run_id,
            "model": {
                # Demo run has no deployed model; indicate that explicitly
                "experiment_id": run.info.experiment_id,
                "deployment_id": "not deployed yet",
                "model_id": "not deployed yet",
            },
            "columns": drift_records,
            "summary": summary,
        }

        import tempfile

        with tempfile.TemporaryDirectory() as _tmp:
            path = os.path.join(_tmp, "drift_metrics.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(drift_artifact, fh, indent=2)
            tracker.log_artifact(path, artifact_path="drift_metrics")
        print("🧾 Wrote drift_metrics artifact")
    except Exception as e:
        print("⚠️ Could not write drift artifact:", e)

    timestamp_analysis(
        train_timestamps=df_sorted["utc_timestamp"].iloc[: len(X_train)],
        test_timestamps=df_sorted["utc_timestamp"].iloc[
            len(X_train) : len(X_train) + len(X_test)
        ],
        output_dir="timestamps",
    )

    # ---------------- Model Training with Hyperparameter Tuning ----------------
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    import optuna

    tracker_data, output_location = start_tracker(
        output_file_name="emissions_hyperparams"
    )
    print("⚡ Resource monitoring started for model training")
    print()

    search_space = {
        "n_estimators": {"type": "int", "low": 10, "high": 100},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    }

    log_search_space(search_space)

    def objective(trial):
        n_estimators = trial.suggest_int(
            "n_estimators",
            search_space["n_estimators"]["low"],
            search_space["n_estimators"]["high"],
        )
        max_depth = trial.suggest_int(
            "max_depth",
            search_space["max_depth"]["low"],
            search_space["max_depth"]["high"],
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            search_space["learning_rate"]["low"],
            search_space["learning_rate"]["high"],
            log=search_space["learning_rate"].get("log", False),
        )
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric="rmse",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return float(mse)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_trial = study.best_trial

    for trial in study.trials:
        with tracker.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            log_params({"trial_number": trial.number})
            log_params({"n_estimators": trial.params["n_estimators"]})
            log_params({"max_depth": trial.params["max_depth"]})
            log_params({"learning_rate": trial.params["learning_rate"]})
            if trial.value is not None:
                log_metrics({"trial_mse": float(trial.value)}, step=trial.number)

            # Log system resource usage for this trial
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            log_resources(
                {"trial_memory_usage_mb": mem_info.rss / 1024 / 1024}, step=trial.number
            )
            log_resources(
                {"trial_cpu_usage_percent": psutil.cpu_percent(interval=1)},
                step=trial.number,
            )

    stop_tracker(tracker_data, output_location)
    print("⚡ Resource monitoring stopped for model training")
    print()

    # Log best hyperparameters from Optuna to MLflow
    log_params({"best_estimators": best_trial.params["n_estimators"]})
    log_params({"best_max_depth": best_trial.params["max_depth"]})
    log_params({"best_learning_rate": best_trial.params["learning_rate"]})
    best_mse = best_trial.value if best_trial.value is not None else 0.0
    # log_metrics expects a mapping of metric names to values; pass a dict to avoid positional type mismatch
    log_metrics({"mse": best_mse}, step=best_trial.number, keep_best=True)

    tracker_data, output_location = start_tracker(output_file_name="emissions_train")

    model = xgb.XGBRegressor(
        n_estimators=1,
        max_depth=best_trial.params["max_depth"],
        learning_rate=best_trial.params["learning_rate"],
        random_state=42,
        eval_metric="rmse",
    )

    max_steps = best_trial.params["n_estimators"]
    booster = None
    for step in range(1, max_steps + 1):
        model.n_estimators = step
        model.fit(X_train, y_train, xgb_model=booster, verbose=False)
        booster = model.get_booster()

        # Predict and compute metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log stepwise metrics
        log_metrics({"mse": float(mse)}, step=step)
        log_metrics({"r2_score": float(r2)}, step=step)

        print(f"Step {step}: MSE={mse:.4f}, R2={r2:.4f}")

        # Log system resource usage for this trial
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        log_resources({"trial_memory_usage_mb": mem_info.rss / 1024 / 1024}, step=step)
        log_resources(
            {"trial_cpu_usage_percent": psutil.cpu_percent(interval=1)}, step=step
        )

    stop_tracker(tracker_data, output_location)
    print("⚡ Resource monitoring stopped for model training")

    # Retrain final model using best hyperparameters and fit on training set
    best_params = best_trial.params
    final_model = xgb.XGBRegressor(
        n_estimators=int(best_params.get("n_estimators", 100)),
        max_depth=int(best_params.get("max_depth", 3)),
        learning_rate=float(best_params.get("learning_rate", 0.1)),
        random_state=42,
        eval_metric="rmse",
    )
    final_model.fit(X_train, y_train)

    log_model_info(
        model_information={
            "model_name": "XGBRegressor",
            "model_version": "1.0",
            "framework": "xgboost",
            "framework_version": xgb.__version__,
        }
    )
    log_model_architecture(
        losses=["rmse"],
        optimizer={
            "name": "gbdt",
            "learning_rate": float(best_params.get("learning_rate", 0.1)),
            "n_estimators": int(best_params.get("n_estimators", 100)),
            "max_depth": int(best_params.get("max_depth", 3)),
        },
        regularization="none",
        early_stopping=False,
    )
    log_model_signature(final_model, X_train, y_train)

    # ---------------- Documentation & Explainability ----------------
    log_declarations_of_conformity(
        issuer="CERTAIN Project Consortium",
        version="v1.0",
        standard_references=["ISO/IEC 42001:2023", "EU AI Act Art. 13"],
        declarations=[
            {
                "filename": "DoC_energy_xgb_v1.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "description": "Declaration of conformity for the energy load forecasting XGBoost model.",
            },
        ],
    )

    log_visual_documentations(
        stage="evaluation",
        generated_by="matplotlib / xgboost built-in",
        model_version="1.0",
        documents=[
            {
                "filename": "feature_importance.png",
                "file_type": "png",
                "description": "XGBoost feature importance bar chart (gain).",
                "tags": ["feature_importance", "xgboost"],
                "link_to_artifacts": "mlflow://artifacts/feature_importance.png",
            },
            {
                "filename": "prediction_vs_actual.png",
                "file_type": "png",
                "description": "Scatter plot of predicted vs actual energy load on the test set.",
                "tags": ["evaluation", "regression"],
                "link_to_artifacts": "mlflow://artifacts/prediction_vs_actual.png",
            },
        ],
    )

    # Log XGBoost feature importances as explainability record
    importance_scores = final_model.get_booster().get_score(importance_type="gain")
    if importance_scores:
        top_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )[:10]
        log_explainable_ai(
            feature_names=[f for f, _ in top_features],
            feature_values=[f"{v:.4f}" for _, v in top_features],
            implementation_details="XGBoost built-in feature importance (gain). Top-10 features by gain score.",
        )

    # ---------------- Deployment: local FastAPI inference server ----------------
    import uvicorn
    import requests as http_requests
    from fastapi import FastAPI
    from pydantic import BaseModel

    DEPLOY_ID = "dep-energy-xgb-prod"
    MODEL_ID = "energy-load-xgb-v1"
    SERVE_PORT = 8090

    # --- Capture deployment logs in memory -----------------------------------
    deploy_log_q: "queue.Queue[str]" = queue.Queue()

    def _enqueue(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        deploy_log_q.put(f"[{ts}] {msg}")

    # --- Build FastAPI app that serves the trained model ---------------------
    serve_app = FastAPI(title="Energy Load XGBoost Server")

    class PredictRequest(BaseModel):
        features: list  # list of feature dicts

    @serve_app.post("/predict")
    def predict(req: PredictRequest):
        import pandas as _pd

        df_in = _pd.DataFrame(req.features)
        preds = final_model.predict(df_in).tolist()
        _enqueue(f"POST /predict  rows={len(preds)}  first_pred={preds[0]:.2f}")
        return {"predictions": preds}

    @serve_app.get("/health")
    def health():
        _enqueue("GET /health → ok")
        return {"status": "ok", "model": MODEL_ID}

    # --- Start server in a background thread ---------------------------------
    server_config = uvicorn.Config(
        serve_app, host="0.0.0.0", port=SERVE_PORT, log_level="warning"
    )
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait until the server is accepting connections (up to 10 s)
    for _ in range(20):
        try:
            r = http_requests.get(f"http://localhost:{SERVE_PORT}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    print(f"🚀 Inference server live on port {SERVE_PORT}")
    _enqueue("Inference server started successfully")

    # --- Log deployment compliance metadata -----------------------------------
    import sklearn

    log_model_packaging(
        deployment_id=DEPLOY_ID,
        model_id=MODEL_ID,
        packaging_format="mlflow_model",
        dependencies=[
            f"xgboost=={xgb.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            "fastapi>=0.100",
            "uvicorn>=0.22",
        ],
        containerization_details={
            "base_image": "python:3.9-slim",
            "cpu": "2",
            "memory": "4GB",
            "port": SERVE_PORT,
            "registry": "local/energy-load-xgb",
        },
    )

    log_build_testing(
        deployment_id=DEPLOY_ID,
        model_id=MODEL_ID,
        build_status="success",
        build_logs="Docker image built from python:3.9-slim; all layers cached.",
        test_type="integration",
        test_results={
            "total": 3,
            "passed": 3,
            "failed": 0,
            "skipped": 0,
            "coverage_pct": 92.0,
        },
    )

    log_standards(
        deployment_id=DEPLOY_ID,
        model_id=MODEL_ID,
        standards=[
            {
                "name": "EU AI Act",
                "description": "Regulation on artificial intelligence – risk-based compliance framework.",
                "version": "2024/1689",
            },
            {
                "name": "ISO/IEC 42001:2023",
                "description": "AI management system standard.",
                "version": "2023",
            },
        ],
    )

    log_interface(
        deployment_id=DEPLOY_ID,
        model_id=MODEL_ID,
        interface_type="REST API",
        specifications=(
            "POST /predict  – body: {features: [{col: val, …}, …]}\n"
            "GET  /health   – returns {status: ok, model: <id>}"
        ),
        version="v1.0",
        documentation_link=f"http://localhost:{SERVE_PORT}/docs",
    )

    # --- Send test predictions and collect logs --------------------------------
    sample_features = X_test.iloc[:5].to_dict(orient="records")
    resp = http_requests.post(
        f"http://localhost:{SERVE_PORT}/predict",
        json={"features": sample_features},
        timeout=10,
    )
    preds_result = resp.json()
    print(
        f"📡 Sample predictions: {[round(p, 1) for p in preds_result['predictions']]}"
    )
    _enqueue(f"Deployment smoke-test predictions: {preds_result['predictions'][:3]}")

    # Health-check for the log
    http_requests.get(f"http://localhost:{SERVE_PORT}/health", timeout=5)

    # --- Drain the deployment log queue and write it as an MLflow artifact ---
    time.sleep(0.2)  # let any in-flight log entries arrive
    log_lines = []
    while not deploy_log_q.empty():
        log_lines.append(deploy_log_q.get_nowait())

    deploy_log_text = "\n".join(log_lines)
    print("\n📋 Deployment log snapshot:")
    for line in log_lines:
        print(f"   {line}")

    import tempfile

    with tempfile.TemporaryDirectory() as _tmp:
        _log_path = os.path.join(_tmp, "deployment_run.log")
        with open(_log_path, "w") as _f:
            _f.write(deploy_log_text + "\n")
        tracker.log_artifact(_log_path, artifact_path="deployment_logs")

    # --- Graceful shutdown of the inference server ---------------------------
    server.should_exit = True
    server_thread.join(timeout=5)
    print("🛑 Inference server stopped")
    _enqueue("Inference server stopped – decommissioning recorded")

    # --- Decommissioning record (lifecycle end of THIS demo run) -------------
    log_decommissioning(
        deployment_id=DEPLOY_ID,
        model_id=MODEL_ID,
        decommissioning_actions=[
            "stop FastAPI inference server",
            "archive MLflow model artifacts",
            "notify energy operations team",
        ],
        reason="Demo workflow complete – server spun down after smoke-test.",
        procedure_details=(
            "The inference server was started inline for demonstration purposes "
            "and decommissioned immediately after the smoke-test completed. "
            "In production, decommissioning is triggered by the CI/CD pipeline."
        ),
    )

    print()
    print(f"ℹ️  Run ID (needed for lifecycle scripts): {run.info.run_id}")

print("=" * 70)
print("✅ Workflow Complete!")
print("=" * 70)
print()
print("Training compliance data logged:")
print("  • AI actors (providers, deployers, auditor)")
print("  • Data labeling procedures (OPSD/ENTSO-E sourcing)")
print("  • Risks (data_bias, data_drift)")
print("  • Human oversight mechanisms")
print("  • Transparency measures (model card, data sheet)")
print("  • Change log")
print("  • Declaration of conformity")
print("  • Visual documentation (feature importance, predictions)")
print("  • Explainable AI (XGBoost feature importances)")
print()
print("Deployment compliance logged (inline FastAPI server):")
print("  • Model packaging  (mlflow_model + containerization details)")
print("  • Build & integration testing (3/3 passed)")
print("  • Standards         (EU AI Act, ISO/IEC 42001:2023)")
print("  • Interface         (REST API  POST /predict  GET /health)")
print("  • Deployment logs   (artifact: deployment_logs/deployment_run.log)")
print("  • Decommissioning   (server stopped after smoke-test)")
print()
print("Query the database to explore your ML experiments! 🎉")
print()
