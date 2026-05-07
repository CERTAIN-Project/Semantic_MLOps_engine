#!/usr/bin/env python3
"""
Complete ML workflow example using certain_library
Demonstrates end-to-end logging to PostgreSQL database
"""
import os
import psutil
import mlflow
import pandas as pd
import numpy as np

from typing import Dict, Union
from sklearn.metrics import mean_squared_error, r2_score

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


from certain_library.train_monitor.log_metrics import log_metrics, log_search_space
from certain_library.train_monitor.log_model import (
    log_model_info,
    log_model_architecture,
    log_model_hyperparameters,
    log_model_signature,
)
from certain_library.data_analysis.log_whylogs import log_whylogs_profile
from certain_library.log_basic.log_param import log_param
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
experiment_name = "complete_ml_workflow_demo"
mlflow.set_experiment(experiment_name)
print(f"📊 Experiment: {experiment_name}")
print()

# Start MLflow run
with mlflow.start_run(run_name="random_forest_classifier") as run:
    print(f"🏃 Run Started: {run.info.run_id}")
    print()

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
        log_param("data_url", "synthetic_dummy_data")
    else:
        print(f"🌐 Downloading real OPSD dataset from {data_url}")
        df = pd.read_csv(data_url, parse_dates=["utc_timestamp"])
        log_param("data_url", data_url)

    log_param("num_rows", df.shape[0])
    log_param("num_columns", df.shape[1])

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
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            log_param("trial_number", trial.number)
            log_param("n_estimators", trial.params["n_estimators"])
            log_param("max_depth", trial.params["max_depth"])
            log_param("learning_rate", trial.params["learning_rate"])
            if trial.value is not None:
                log_metrics({"trial_mse": float(trial.value)}, step=trial.number)

            # Log system resource usage for this trial
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            log_metrics(
                {"trial_memory_usage_mb": mem_info.rss / 1024 / 1024}, step=trial.number
            )
            log_metrics(
                {"trial_cpu_usage_percent": psutil.cpu_percent(interval=1)},
                step=trial.number,
            )

    stop_tracker(tracker_data, output_location)
    print("⚡ Resource monitoring stopped for model training")
    print()

    # Log best hyperparameters from Optuna to MLflow
    log_param("best_estimators", best_trial.params["n_estimators"])
    log_param("best_max_depth", best_trial.params["max_depth"])
    log_param("best_learning_rate", best_trial.params["learning_rate"])
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
        log_metrics({"trial_memory_usage_mb": mem_info.rss / 1024 / 1024}, step=step)
        log_metrics(
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

    # ---------------- Deployment & Decommissioning ----------------
    # These are handled by separate lifecycle scripts that reopen this run
    # by run_id — they are NOT called here because deployment happens in
    # CI/CD and decommissioning happens when the model is retired.
    #
    #   After deployment (from CI/CD pipeline):
    #     python lifecycle/log_deployment_compliance.py \
    #         --run-id     <run_id> \
    #         --deploy-id  dep-energy-xgb-prod \
    #         --model-id   energy-load-xgb-v1
    #
    #   When retiring the model:
    #     python lifecycle/log_decommission_compliance.py \
    #         --run-id     <run_id> \
    #         --deploy-id  dep-energy-xgb-prod \
    #         --model-id   energy-load-xgb-v1 \
    #         --reason     "Replaced by v2 model." \
    #         --changed-by ml-team
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
print("Deployment compliance → run lifecycle/log_deployment_compliance.py")
print("Decommissioning      → run lifecycle/log_decommission_compliance.py")
print()
print("Query the database to explore your ML experiments! 🎉")
print()
