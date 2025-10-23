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
from certain_library.train_monitor.log_metrics import log_metrics, log_search_space
from certain_library.train_monitor.log_model import (
    log_model_info,
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

    data_url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
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
    log_model_signature(final_model, X_train, y_train)

print("=" * 70)
print("✅ Workflow Complete!")
print("=" * 70)
print()
print("All data has been logged to PostgreSQL database:")
print("  • Experiment metadata")
print("  • Run information")
print("  • Training & test datasets")
print("  • Model hyperparameters")
print("  • Model information")
print("  • Training metrics per epoch")
print("  • Final evaluation metrics")
print("  • Resource usage metrics")
print("  • Hyperparameter search space")
print()
print("Query the database to explore your ML experiments! 🎉")
print()
