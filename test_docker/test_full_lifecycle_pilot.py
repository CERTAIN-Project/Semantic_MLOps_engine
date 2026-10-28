#!/usr/bin/env python3
"""
Full-lifecycle ML pilot — Energy Load Forecasting (XGBoost)
===========================================================
Demonstrates the complete CERTAIN MLOps compliance pipeline across all three
lifecycle phases, all attached to a single MLflow run_id:

  Phase 1 — TRAINING
    • Data ingestion (synthetic OPSD-style energy data)
    • Data cleaning, outlier removal, augmentation
    • Hyperparameter optimisation with Optuna
    • Stepwise XGBoost training with per-step metrics
    • All experiment-level compliance: AI actors, labeling procedures,
      risks, human oversight, transparency measures, change log,
      declarations of conformity, visual documentation, explainability

  Phase 2 — DEPLOYMENT  (reopens the same run_id)
    • Model packaging record (Docker / mlflow_model)
    • Build & integration testing results
    • Applicable standards (ISO 42001, EU AI Act, ETSI)
    • REST API interface specification

  Phase 3 — DECOMMISSIONING  (reopens the same run_id)
    • Decommissioning record with actions & reason
    • Change log entry recording who retired the model

The logging functions in certain_library do NOT need to change — they
work identically regardless of whether the MLflow run is new or reopened.

Usage
-----
    python test_full_lifecycle_pilot.py

Set USE_DUMMY_DATA = True (default) for fully offline execution.
Set USE_DUMMY_DATA = False to download the real OPSD dataset.
"""

from certain_library.tracking.tracker import tracker

import os
import time
import psutil
import mlflow
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
USE_DUMMY_DATA = True
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "energy_load_forecasting_lifecycle"
OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/"
    "2020-10-06/time_series_60min_singleindex.csv"
)

# ---------------------------------------------------------------------------
# certain_library imports
# ---------------------------------------------------------------------------
from certain_library.train_monitor.log_metrics import log_metrics, log_search_space
from certain_library.train_monitor.log_model import (
    log_model_info,
    log_model_architecture,
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


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def _make_dummy_energy_df(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic energy time-series DataFrame mimicking the OPSD schema."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2018-01-01", periods=n_rows, freq="h", tz="UTC")
    hour_of_day = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    daily_pattern = 5_000 * np.sin(np.pi * hour_of_day / 12) + 55_000
    weekly_pattern = np.where(day_of_week >= 5, -4_000, 0)
    load = daily_pattern + weekly_pattern + rng.normal(0, 1_500, n_rows)

    solar = np.maximum(
        0, 8_000 * np.sin(np.pi * np.clip(hour_of_day - 6, 0, 12) / 12)
    ) + rng.normal(0, 300, n_rows)
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
    # Inject ~1% missing values to exercise the cleaning pipeline
    for col in ["DE_load_actual_entsoe_transparency", "DE_solar_generation_actual"]:
        idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
        df.loc[idx, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Data pipeline helpers
# ---------------------------------------------------------------------------
def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill()


def remove_outliers_iqr(
    df: pd.DataFrame,
    column: str = "DE_load_actual_entsoe_transparency",
) -> pd.DataFrame:
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR))]


def augment_with_noise(
    df: pd.DataFrame,
    column: str = "DE_load_actual_entsoe_transparency",
    noise_factor: float = 0.01,
) -> pd.DataFrame:
    df_aug = df.copy()
    df_aug[f"{column}_aug"] = df_aug[column] + noise_factor * np.random.normal(
        size=len(df_aug)
    )
    return df_aug


# ===========================================================================
#  PHASE 1 — TRAINING
# ===========================================================================
def phase_training(mlflow_run) -> tuple:
    """
    Run the full training pipeline inside the already-active MLflow run.

    Returns
    -------
    (final_model, X_train, y_train, best_params, run_id)
    """
    import xgboost as xgb
    import optuna

    run_id = mlflow_run.info.run_id
    print(f"  Run ID : {run_id}")

    # ---- Resource monitoring ----
    tracker_data, output_location = start_tracker(output_file_name="emissions_data")

    # ---- Experiment-level compliance ----
    log_ai_actors(
        auditor="CERTAIN Project Consortium",
        organization="Open Power System Data Initiative",
        ai_providers=[
            {
                "name": "ML Engineering Team",
                "role": "model development and training",
                "contact": "ml-team@example.com",
            }
        ],
        ai_deployers=[
            {
                "name": "Energy Operations Team",
                "role": "operational deployment and monitoring",
                "contact": "ops@example.com",
            }
        ],
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
            }
        ],
    )

    log_risk(
        [
            {
                "risk_type": "data_bias",
                "risk_description": (
                    "Energy load data may reflect historical consumption patterns "
                    "biased towards certain seasons or regions."
                ),
                "risk_level": 0.4,
            },
            {
                "risk_type": "data_drift",
                "risk_description": (
                    "Consumption patterns may shift due to climate change or policy "
                    "changes after the training cutoff date."
                ),
                "risk_level": 0.6,
            },
            {
                "risk_type": "model_failure",
                "risk_description": (
                    "XGBoost regression may extrapolate poorly to extreme weather events "
                    "not represented in the training distribution."
                ),
                "risk_level": 0.5,
            },
        ]
    )

    log_human_oversight(
        [
            {
                "oversight_type": "human-in-the-loop",
                "description": (
                    "Domain experts review model predictions before they are used in "
                    "operational energy dispatch decisions."
                ),
                "implementation_details": (
                    "Weekly review meetings with the energy operations team; "
                    "model output is advisory only."
                ),
            },
            {
                "oversight_type": "periodic-audit",
                "description": "Quarterly audit of model performance metrics and dataset freshness.",
                "implementation_details": (
                    "Automated report generated from MLflow tracking server and "
                    "reviewed by the AI governance team."
                ),
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
                "description": (
                    "Public-facing documentation for the energy load forecasting model "
                    "and the OPSD dataset."
                ),
            },
            {
                "measure_type": ["feature_importance_report"],
                "measure_value": ["mlflow://artifacts/feature_importance"],
                "description": "XGBoost feature importance logged after each training run.",
            },
        ]
    )

    log_change(
        [
            {
                "change_description": (
                    "Initial training run using synthetic OPSD-style 60-minute time series "
                    "data with XGBoost regressor and Optuna hyperparameter search."
                ),
                "changed_by": "data_science_team",
            }
        ]
    )

    # ---- Data loading ----
    if USE_DUMMY_DATA:
        print("  🧪 Using synthetic dummy energy data")
        df = _make_dummy_energy_df(n_rows=2000)
        log_params({"data_url": "synthetic_dummy_data"})
    else:
        print(f"  🌐 Downloading OPSD dataset...")
        df = pd.read_csv(OPSD_URL, parse_dates=["utc_timestamp"])
        log_params({"data_url": OPSD_URL})

    log_params({"num_rows": df.shape[0], "num_columns": df.shape[1]})

    # ---- Data cleaning pipeline ----
    df = df.dropna(subset=["DE_load_actual_entsoe_transparency"])
    df_cleaned = clean_missing_values(df)
    log_dataset(df_cleaned, name="df_cleaned", output_dir="data_cleaning")
    log_whylogs_profile(df_cleaned, name="cleaned")

    df_filtered = remove_outliers_iqr(df_cleaned)
    log_dataset(df_filtered, name="df_filtered", output_dir="data_cleaning")
    log_whylogs_profile(df_filtered, name="filtered")

    # ---- Data augmentation ----
    df_augmented = augment_with_noise(df_filtered)
    log_dataset(df_augmented, name="df_augmented", output_dir="data_augmentation")
    log_whylogs_profile(df_augmented, name="augmented")

    df_sorted = df_augmented.sort_values("utc_timestamp")
    stop_tracker(tracker_data, output_location)

    # ---- Train/test split ----
    target = "DE_load_actual_entsoe_transparency"
    y = df_sorted[target]
    X = df_sorted.select_dtypes(include=[np.number]).drop(columns=[target])

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

    timestamp_analysis(
        train_timestamps=df_sorted["utc_timestamp"].iloc[: len(X_train)],
        test_timestamps=df_sorted["utc_timestamp"].iloc[
            len(X_train) : len(X_train) + len(X_test)
        ],
        output_dir="timestamps",
    )

    # ---- Hyperparameter search ----
    tracker_data, output_location = start_tracker(
        output_file_name="emissions_hyperparams"
    )

    search_space = {
        "n_estimators": {"type": "int", "low": 10, "high": 50},
        "max_depth": {"type": "int", "low": 3, "high": 8},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    }
    log_search_space(search_space)

    def objective(trial):
        model = xgb.XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 10, 50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            random_state=42,
            eval_metric="rmse",
        )
        model.fit(X_train, y_train)
        return float(mean_squared_error(y_test, model.predict(X_test)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_trial = study.best_trial

    for trial in study.trials:
        with tracker.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            log_params(
                {
                    "trial_number": trial.number,
                    "n_estimators": trial.params["n_estimators"],
                    "max_depth": trial.params["max_depth"],
                    "learning_rate": trial.params["learning_rate"],
                }
            )
            if trial.value is not None:
                log_metrics({"trial_mse": float(trial.value)}, step=trial.number)
            proc = psutil.Process(os.getpid())
            log_metrics(
                {"trial_memory_mb": proc.memory_info().rss / 1024 / 1024},
                step=trial.number,
            )

    log_metrics(
        {"mse": best_trial.value or 0.0},
        step=best_trial.number,
        keep_best=True,
    )
    stop_tracker(tracker_data, output_location)

    # ---- Stepwise training ----
    tracker_data, output_location = start_tracker(output_file_name="emissions_train")
    best_params = best_trial.params

    model = xgb.XGBRegressor(
        n_estimators=1,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        random_state=42,
        eval_metric="rmse",
    )
    booster = None
    for step in range(1, best_params["n_estimators"] + 1):
        model.n_estimators = step
        model.fit(X_train, y_train, xgb_model=booster, verbose=False)
        booster = model.get_booster()
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        log_metrics({"mse": float(mse), "r2_score": float(r2)}, step=step)
        proc = psutil.Process(os.getpid())
        log_metrics({"memory_mb": proc.memory_info().rss / 1024 / 1024}, step=step)

    stop_tracker(tracker_data, output_location)

    # ---- Final model ----
    final_model = xgb.XGBRegressor(
        n_estimators=int(best_params.get("n_estimators", 50)),
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
            "n_estimators": int(best_params.get("n_estimators", 50)),
            "max_depth": int(best_params.get("max_depth", 3)),
        },
        regularization="none",
        early_stopping=False,
    )
    log_model_signature(final_model, X_train, y_train)

    # ---- Declarations of conformity ----
    log_declarations_of_conformity(
        issuer="CERTAIN Project Consortium",
        version="v1.0",
        standard_references=["ISO/IEC 42001:2023", "EU AI Act Art. 13"],
        declarations=[
            {
                "filename": "DoC_energy_xgb_v1.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "description": (
                    "Declaration of conformity for the energy load forecasting "
                    "XGBoost model."
                ),
            }
        ],
    )

    # ---- Visual documentation ----
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
                "description": "Predicted vs actual energy load scatter plot.",
                "tags": ["evaluation", "regression"],
                "link_to_artifacts": "mlflow://artifacts/prediction_vs_actual.png",
            },
        ],
    )

    # ---- Explainability — live XGBoost gain scores ----
    importance_scores = final_model.get_booster().get_score(importance_type="gain")
    if importance_scores:
        top = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        log_explainable_ai(
            feature_names=[f for f, _ in top],
            feature_values=[f"{v:.4f}" for _, v in top],
            implementation_details=(
                "XGBoost built-in feature importance (gain). "
                "Top-10 features by gain score."
            ),
        )

    return final_model, X_train, y_train, best_params, xgb.__version__, run_id


# ===========================================================================
#  PHASE 2 — DEPLOYMENT
# ===========================================================================
def phase_deployment(run_id: str, xgb_version: str) -> str:
    """
    Reopen the training run and attach deployment compliance artifacts.

    Returns the deployment_id so phase_decommissioning can reference it.
    """
    deployment_id = f"dep-energy-xgb-{run_id[:8]}"
    model_id = "energy-load-xgb-v1"

    # Reopen the existing run — no new run is created
    with tracker.start_run(run_id=run_id):

        log_model_packaging(
            deployment_id=deployment_id,
            model_id=model_id,
            packaging_format="mlflow_model",
            dependencies=[
                f"xgboost=={xgb_version}",
                "scikit-learn",
                "pandas",
                "numpy",
            ],
            containerization_details={
                "base_image": "python:3.11-slim",
                "cpu": "2",
                "memory": "4GB",
                "registry": "docker.io/certain/energy-xgb",
                "port": 8080,
            },
        )

        log_build_testing(
            deployment_id=deployment_id,
            model_id=model_id,
            build_status="success",
            build_logs=(
                "Docker image built successfully. "
                "Smoke test: POST /predict returned HTTP 200. "
                "All 5 integration tests passed."
            ),
            test_type="integration",
            test_results={
                "total": 5,
                "passed": 5,
                "failed": 0,
                "skipped": 0,
                "coverage_pct": 92.0,
            },
        )

        log_standards(
            deployment_id=deployment_id,
            model_id=model_id,
            standards=[
                {
                    "name": "ISO/IEC 42001:2023",
                    "description": "AI management system standard.",
                    "version": "2023",
                },
                {
                    "name": "EU AI Act",
                    "description": "European Union regulation on artificial intelligence.",
                    "version": "2024",
                },
                {
                    "name": "ETSI EN 303 645",
                    "description": (
                        "Cyber security standard referenced for "
                        "data pipeline security."
                    ),
                    "version": "v2.1.1",
                },
            ],
        )

        log_interface(
            deployment_id=deployment_id,
            model_id=model_id,
            interface_type="REST API",
            specifications=(
                "POST /predict — accepts JSON {features: [...]} "
                "returns {predicted_load_mw: float}. "
                "GET /health — returns {status: 'healthy'}."
            ),
            version="v1.0",
            documentation_link=f"https://wiki.example.com/api/{model_id}",
        )

    return deployment_id


# ===========================================================================
#  PHASE 3 — DECOMMISSIONING
# ===========================================================================
def phase_decommissioning(run_id: str, deployment_id: str) -> None:
    """
    Reopen the training run and attach decommissioning compliance artifacts.
    """
    model_id = "energy-load-xgb-v1"

    with tracker.start_run(run_id=run_id):

        log_decommissioning(
            deployment_id=deployment_id,
            model_id=model_id,
            decommissioning_actions=[
                "disable REST API endpoint",
                "archive model weights to cold storage",
                "remove Docker image from registry",
                "update model registry status to 'archived'",
                "notify stakeholders via email",
            ],
            reason=(
                "Replaced by energy-load-xgb-v2 which achieves 18% lower RMSE "
                "on the 2024 holdout set after retraining on updated OPSD data."
            ),
            procedure_details=(
                "Endpoint traffic was drained over 24h using a canary rollout. "
                "Weights archived to s3://certain-archive/models/energy-xgb-v1/. "
                "Docker image retained in registry with 'deprecated' tag."
            ),
            decommissioning_date=float(time.time()),
        )

        log_change(
            [
                {
                    "change_description": (
                        f"Model {model_id} (deployment {deployment_id}) decommissioned. "
                        "Replaced by energy-load-xgb-v2. Traffic migrated via 24h canary rollout."
                    ),
                    "changed_by": "ml-team",
                }
            ]
        )


# ===========================================================================
#  MAIN
# ===========================================================================
def main() -> None:
    tracker.set_tracking_uri(MLFLOW_TRACKING_URI)
    tracker.set_experiment(EXPERIMENT_NAME)

    _sep = "=" * 70

    # -------------------------------------------------------------------
    # PHASE 1 — TRAINING
    # -------------------------------------------------------------------
    print(_sep)
    print("PHASE 1 — TRAINING")
    print(_sep)

    with tracker.start_run(run_name="energy_xgb_v1_training") as run:
        final_model, X_train, y_train, best_params, xgb_version, run_id = (
            phase_training(run)
        )

    print(f"  ✅ Training complete  |  run_id: {run_id}")
    print()

    # Simulate a realistic pause between training and deployment
    print("  ⏳ Simulating CI/CD pipeline delay (2 s)...")
    time.sleep(2)

    # -------------------------------------------------------------------
    # PHASE 2 — DEPLOYMENT
    # -------------------------------------------------------------------
    print(_sep)
    print("PHASE 2 — DEPLOYMENT")
    print(_sep)

    deployment_id = phase_deployment(run_id=run_id, xgb_version=xgb_version)
    print(f"  ✅ Deployment compliance logged  |  deployment_id: {deployment_id}")
    print()

    # Simulate the model running in production for some time
    print("  ⏳ Simulating production lifetime (2 s)...")
    time.sleep(2)

    # -------------------------------------------------------------------
    # PHASE 3 — DECOMMISSIONING
    # -------------------------------------------------------------------
    print(_sep)
    print("PHASE 3 — DECOMMISSIONING")
    print(_sep)

    phase_decommissioning(run_id=run_id, deployment_id=deployment_id)
    print(f"  ✅ Decommissioning compliance logged")
    print()

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(_sep)
    print("✅ Full lifecycle complete!")
    print(_sep)
    print()
    print(f"  MLflow run ID      : {run_id}")
    print(f"  Deployment ID      : {deployment_id}")
    print(f"  MLflow tracking UI : {MLFLOW_TRACKING_URI}")
    print()
    print("  All three phases attached to a single run_id in certain_db:")
    print()
    print("  Phase 1 — Training")
    print("    • ai_actors, labeling_procedures")
    print("    • risks, human_oversight_mechanisms, transparency_measures, change_logs")
    print(
        "    • declaration_of_conformity, visual_documentation, explainable_ai_features"
    )
    print("    • model_architecture, model_params, metrics, datasets, resources")
    print()
    print("  Phase 2 — Deployment")
    print("    • model_packaging")
    print("    • build_and_integration_testing")
    print("    • standards")
    print("    • interfaces")
    print()
    print("  Phase 3 — Decommissioning")
    print("    • decomissioning")
    print("    • change_logs  (retirement entry)")
    print()


if __name__ == "__main__":
    main()
