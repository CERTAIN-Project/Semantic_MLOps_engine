import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class OPSDTestVariables:
    """Test variables and data structures for OPSD energy data testing."""

    # Experiment configuration
    EXPERIMENT_NAME = "ML For OPSD new"
    RUN_NAME = "OPSD_Energy_Data_Model_Stepwise__1"
    LOG_SYSTEM_METRICS = True

    # Dataset parameters
    DATASET_PARAMS = {
        "total_samples": "1000",
        "train_split": "0.8",
        "test_split": "0.2",
        "temporal_split": "True",
        "target_column": "DE_load_actual_entsoe_transparency",
        "dataset_name": "Open Power System Data - Time Series",
        "data_url": "https://data.open-power-system-data.org/time_series/",
    }

    # Hyperparameter search space
    SEARCH_SPACE = {
        "n_estimators": {"type": "int", "low": 50, "high": 100},
        "max_depth": {"type": "int", "low": 3, "high": 8},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.7, "high": 1.0},
    }

    # Best hyperparameters from optimization
    BEST_HYPERPARAMS = {
        "n_estimators": 75,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "best_n_estimators": 75,
        "best_max_depth": 6,
        "best_learning_rate": 0.1,
        "best_mse": 0.045,
    }

    # Model parameters
    MODEL_PARAMS = {
        "n_estimators": 75,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
        "eval_metric": "rmse",
        "model_type": "XGBRegressor",
        "max_steps": "75",
    }

    # Sample trial metrics
    SAMPLE_TRIAL_METRICS = [
        {
            "trial_number": 0,
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "trial_mse": 0.15,
            "trial_memory_usage_mb": 1024.0,
            "trial_cpu_usage_percent": 45.2,
        },
        {
            "trial_number": 1,
            "n_estimators": 75,
            "max_depth": 6,
            "learning_rate": 0.1,
            "trial_mse": 0.08,
            "trial_memory_usage_mb": 1280.0,
            "trial_cpu_usage_percent": 52.1,
        },
        {
            "trial_number": 2,
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.05,
            "trial_mse": 0.12,
            "trial_memory_usage_mb": 1536.0,
            "trial_cpu_usage_percent": 48.7,
        },
    ]

    # Stepwise training metrics
    STEPWISE_METRICS = [
        {"step": 1, "mse": 0.15, "r2_score": 0.75},
        {"step": 2, "mse": 0.12, "r2_score": 0.82},
        {"step": 3, "mse": 0.08, "r2_score": 0.88},
        {"step": 4, "mse": 0.06, "r2_score": 0.91},
        {"step": 5, "mse": 0.045, "r2_score": 0.93},
    ]

    # Final evaluation metrics
    FINAL_METRICS = {
        "mse": 0.045,
        "rmse": 0.212,
        "r2_score": 0.93,
        "mae": 0.165,
        "final_mse": 0.043,
        "final_r2_score": 0.93,
        "memory_usage_mb": 1536.0,
    }

    # MLflow tags
    MLFLOW_TAGS = {
        "model_type": "XGBRegressor",
        "dataset": "OPSD",
        "experiment_type": "energy_forecasting",
        "optimization": "optuna",
        "emissions_tracking": "codecarbon",
    }

    # Artifact paths
    ARTIFACT_PATHS = {
        "data_cleaning": "data_cleaning",
        "augmentation": "augmentation",
        "whylogs": "whylogs",
        "code_carbon": "code_carbon",
        "model": "model",
        "timestamps": "timestamps",
        "dataset": "dataset",
    }

    # Emissions files for each phase
    EMISSIONS_FILES = {
        "data_processing": "emissions_data.csv",
        "hyperparameter_optimization": "emissions_hyperparams.csv",
        "model_training": "emissions_train.csv",
    }

    # WhyLogs profiles
    WHYLOGS_PROFILES = {
        "input": "opsd_input_profile",
        "cleaned": "opsd_cleaned_profile",
        "filtered": "opsd_filtered_profile",
        "augmented": "opsd_augmented_profile",
    }

    @classmethod
    def create_sample_opsd_dataframe(cls, n_samples=1000):
        """Create sample OPSD-style energy data DataFrame."""
        np.random.seed(42)

        # Create realistic timestamps
        start_date = datetime(2019, 1, 1)
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq="H")

        # Create realistic energy data patterns
        base_load = 50000  # Base load in MW

        data = {
            "utc_timestamp": timestamps,
            "cet_cest_timestamp": timestamps + timedelta(hours=1),
            "DE_load_actual_entsoe_transparency": (
                base_load
                + np.random.normal(0, 5000, n_samples)  # Random variation
                + 10000 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily pattern
                + 5000
                * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))  # Weekly pattern
            ),
            "DE_solar_generation_actual": np.maximum(
                0,
                5000
                + np.random.normal(0, 2000, n_samples)
                + 3000 * np.sin(2 * np.pi * np.arange(n_samples) / 24),  # Solar pattern
            ),
            "DE_wind_generation_actual": np.maximum(
                0, 8000 + np.random.normal(0, 3000, n_samples)  # Wind variation
            ),
            "DE_wind_offshore_generation_actual": np.maximum(
                0, 3000 + np.random.normal(0, 1500, n_samples)
            ),
            "DE_wind_onshore_generation_actual": np.maximum(
                0, 5000 + np.random.normal(0, 2000, n_samples)
            ),
        }

        df = pd.DataFrame(data)
        df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
        df["cet_cest_timestamp"] = pd.to_datetime(df["cet_cest_timestamp"])

        return df

    @classmethod
    def create_cleaned_data_pipeline_stages(cls, df):
        """Create all data cleaning pipeline stages."""
        stages = {}

        # Stage 1: Drop missing target values
        df_no_missing = df.dropna(subset=["DE_load_actual_entsoe_transparency"])
        stages["dropped_missing_target"] = df_no_missing

        # Stage 2: Clean data (remove outliers)
        Q1 = df_no_missing["DE_load_actual_entsoe_transparency"].quantile(0.25)
        Q3 = df_no_missing["DE_load_actual_entsoe_transparency"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_cleaned = df_no_missing[
            (df_no_missing["DE_load_actual_entsoe_transparency"] >= lower_bound)
            & (df_no_missing["DE_load_actual_entsoe_transparency"] <= upper_bound)
        ]
        stages["df_cleaned"] = df_cleaned

        # Stage 3: Filter data (keep complete records)
        df_filtered = df_cleaned.dropna()
        stages["df_filtered"] = df_filtered

        # Stage 4: Augment data (add features)
        df_augmented = df_filtered.copy()
        df_augmented["hour"] = df_augmented["utc_timestamp"].dt.hour
        df_augmented["day_of_week"] = df_augmented["utc_timestamp"].dt.dayofweek
        df_augmented["month"] = df_augmented["utc_timestamp"].dt.month
        df_augmented["feature1"] = np.random.randn(len(df_augmented))
        df_augmented["feature2"] = np.random.randn(len(df_augmented))
        stages["df_augmented"] = df_augmented

        return stages

    @classmethod
    def get_train_test_split(cls, df):
        """Get train-test split with temporal separation."""
        # Create a copy to avoid modifying original
        df_work = df.copy()

        # Add features if they don't exist
        feature_cols = [
            col
            for col in df_work.columns
            if col
            not in [
                "utc_timestamp",
                "cet_cest_timestamp",
                "DE_load_actual_entsoe_transparency",
            ]
        ]

        # If no feature columns exist, add some basic ones
        if not feature_cols:
            df_work["hour"] = df_work["utc_timestamp"].dt.hour
            df_work["day_of_week"] = df_work["utc_timestamp"].dt.dayofweek
            df_work["month"] = df_work["utc_timestamp"].dt.month
            feature_cols = ["hour", "day_of_week", "month"]

            # Add generation features if they exist
            gen_cols = [col for col in df_work.columns if "generation" in col]
            feature_cols.extend(gen_cols)

        # Sort by timestamp
        df_sorted = df_work.sort_values("utc_timestamp")

        # Temporal split at 80%
        split_idx = int(len(df_sorted) * 0.8)

        train_data = df_sorted.iloc[:split_idx]
        test_data = df_sorted.iloc[split_idx:]

        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        y_train = train_data["DE_load_actual_entsoe_transparency"]
        y_test = test_data["DE_load_actual_entsoe_transparency"]

        train_timestamps = train_data["utc_timestamp"]
        test_timestamps = test_data["utc_timestamp"]

        return X_train, X_test, y_train, y_test, train_timestamps, test_timestamps

    @classmethod
    def get_emissions_file_content(cls, phase):
        """Get emissions file content for a specific phase."""
        base_content = {
            "timestamp": "2024-01-01T10:00:00",
            "project_name": "OPSD_ML_Project",
            "run_id": f"{phase}_run_001",
            "duration": 300.5,
            "emissions": 0.001234,
            "emissions_rate": 0.000004,
            "cpu_power": 42.5,
            "gpu_power": 0.0,
            "ram_power": 3.2,
            "cpu_energy": 0.003542,
            "gpu_energy": 0.0,
            "ram_energy": 0.000267,
            "energy_consumed": 0.003809,
            "country_name": "Germany",
            "country_iso_code": "DEU",
            "region": "Europe",
            "cloud_provider": "",
            "cloud_region": "",
            "os": "macOS-14.7.1-arm64-arm-64bit",
            "python_version": "3.9.20",
            "codecarbon_version": "2.7.0",
            "cpu_count": 8,
            "cpu_model": "Apple M1",
            "gpu_count": 0,
            "gpu_model": "",
            "longitude": 9.0,
            "latitude": 51.0,
            "ram_total_size": 16.0,
            "tracking_mode": "machine",
        }
        return base_content

    @classmethod
    def get_timestamps_file_content(cls, train_ts, test_ts):
        """Get timestamps file content."""
        content = {
            "train_start": str(train_ts.min()),
            "train_end": str(train_ts.max()),
            "test_start": str(test_ts.min()),
            "test_end": str(test_ts.max()),
            "total_train_samples": len(train_ts),
            "total_test_samples": len(test_ts),
        }
        return content

    @classmethod
    def get_dataset_parameters(cls):
        """Get dataset parameters for logging."""
        return cls.DATASET_PARAMS.copy()

    @classmethod
    def get_model_parameters(cls):
        """Get model parameters for logging."""
        return cls.MODEL_PARAMS.copy()

    @classmethod
    def get_best_hyperparameters(cls):
        """Get best hyperparameters for logging."""
        return cls.BEST_HYPERPARAMS.copy()

    @classmethod
    def get_trial_metrics(cls):
        """Get trial metrics for logging."""
        return cls.SAMPLE_TRIAL_METRICS.copy()

    @classmethod
    def get_final_metrics(cls):
        """Get final evaluation metrics for logging."""
        return cls.FINAL_METRICS.copy()

    @classmethod
    def get_whylogs_profile_names(cls):
        """Get WhyLogs profile names for all stages."""
        return list(cls.WHYLOGS_PROFILES.values())

    @classmethod
    def get_emissions_file_names(cls):
        """Get emissions file names for all phases."""
        return list(cls.EMISSIONS_FILES.values())

    @classmethod
    def create_sample_train_test_data(cls, n_samples=1000):
        """Create sample train/test data with proper structure."""
        df = cls.create_sample_opsd_dataframe(n_samples)
        return cls.get_train_test_split(df)

    @classmethod
    def get_opsd_feature_columns(cls):
        """Get list of OPSD feature column names."""
        sample_df = cls.create_sample_opsd_dataframe(10)
        return [
            col
            for col in sample_df.columns
            if col
            not in [
                "utc_timestamp",
                "cet_cest_timestamp",
                "DE_load_actual_entsoe_transparency",
            ]
        ]

    @classmethod
    def get_opsd_target_column(cls):
        """Get OPSD target column name."""
        return "DE_load_actual_entsoe_transparency"

    @classmethod
    def create_mock_whylogs_profile_data(cls, stage_name):
        """Create mock WhyLogs profile data for a specific stage."""
        feature_cols = cls.get_opsd_feature_columns()
        target_col = cls.get_opsd_target_column()

        # Add basic features that might be created during processing
        all_cols = feature_cols + [
            target_col,
            "hour",
            "day_of_week",
            "month",
            "feature1",
            "feature2",
        ]

        profile_data = []
        for col in all_cols:
            if "generation" in col:
                mean_val = np.random.uniform(3000, 8000)
                std_val = np.random.uniform(1000, 3000)
            elif col == target_col:
                mean_val = 50000
                std_val = 5000
            else:
                mean_val = (
                    np.random.uniform(0, 24)
                    if col == "hour"
                    else np.random.uniform(0, 1)
                )
                std_val = np.random.uniform(0.5, 2.0)

            profile_data.append(
                {
                    "column": col,
                    "count": 1000,
                    "mean": mean_val,
                    "stddev": std_val,
                    "min": mean_val - 2 * std_val,
                    "max": mean_val + 2 * std_val,
                }
            )

        return pd.DataFrame(profile_data)

    @classmethod
    def get_mlflow_experiment_config(cls):
        """Get complete MLflow experiment configuration."""
        return {
            "experiment_name": cls.EXPERIMENT_NAME,
            "run_name": cls.RUN_NAME,
            "log_system_metrics": cls.LOG_SYSTEM_METRICS,
            "tags": cls.MLFLOW_TAGS,
            "artifact_paths": cls.ARTIFACT_PATHS,
        }


def test_opsd_variables_initialization():
    """Test that OPSDTestVariables class initializes correctly."""
    assert OPSDTestVariables.EXPERIMENT_NAME == "ML For OPSD new"
    assert OPSDTestVariables.RUN_NAME == "OPSD_Energy_Data_Model_Stepwise__1"
    assert OPSDTestVariables.LOG_SYSTEM_METRICS is True


def test_create_sample_opsd_dataframe():
    """Test sample OPSD DataFrame creation."""
    df = OPSDTestVariables.create_sample_opsd_dataframe(100)

    assert len(df) == 100
    assert "DE_load_actual_entsoe_transparency" in df.columns
    assert "DE_solar_generation_actual" in df.columns
    assert "DE_wind_generation_actual" in df.columns
    assert df["DE_load_actual_entsoe_transparency"].notna().all()


def test_data_cleaning_pipeline_stages():
    """Test data cleaning pipeline stages."""
    df = OPSDTestVariables.create_sample_opsd_dataframe(100)
    stages = OPSDTestVariables.create_cleaned_data_pipeline_stages(df)

    assert "dropped_missing_target" in stages
    assert "df_cleaned" in stages
    assert "df_filtered" in stages
    assert "df_augmented" in stages

    # Check that augmented data has additional features
    assert "hour" in stages["df_augmented"].columns
    assert "day_of_week" in stages["df_augmented"].columns
    assert "feature1" in stages["df_augmented"].columns


def test_train_test_split():
    """Test train-test split functionality."""
    df = OPSDTestVariables.create_sample_opsd_dataframe(100)
    # Add required features for split
    df["feature1"] = np.random.randn(len(df))
    df["feature2"] = np.random.randn(len(df))

    X_train, X_test, y_train, y_test, train_ts, test_ts = (
        OPSDTestVariables.get_train_test_split(df)
    )

    assert len(X_train) == 80  # 80% of 100
    assert len(X_test) == 20  # 20% of 100
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert len(train_ts) == 80
    assert len(test_ts) == 20


def test_emissions_file_content():
    """Test emissions file content generation."""
    content = OPSDTestVariables.get_emissions_file_content("data_processing")

    assert "timestamp" in content
    assert "emissions" in content
    assert "energy_consumed" in content
    assert content["project_name"] == "OPSD_ML_Project"


def test_sample_trial_metrics():
    """Test sample trial metrics structure."""
    metrics = OPSDTestVariables.SAMPLE_TRIAL_METRICS

    assert len(metrics) == 3
    assert all("trial_number" in metric for metric in metrics)
    assert all("trial_mse" in metric for metric in metrics)
    assert metrics[1]["trial_mse"] < metrics[0]["trial_mse"]  # Best trial has lower MSE


def test_stepwise_metrics():
    """Test stepwise training metrics."""
    metrics = OPSDTestVariables.STEPWISE_METRICS

    assert len(metrics) == 5
    assert all("step" in metric for metric in metrics)
    assert all("mse" in metric for metric in metrics)
    assert all("r2_score" in metric for metric in metrics)

    # Check that metrics improve over steps
    mse_values = [m["mse"] for m in metrics]
    assert mse_values == sorted(mse_values, reverse=True)  # MSE should decrease
