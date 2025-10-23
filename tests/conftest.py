import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch
import mlflow
from codecarbon import EmissionsTracker

from tests.test_opsd_variables import OPSDTestVariables

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"


@pytest.fixture
def opsd_variables():
    """Provide access to OPSD test variables."""
    return OPSDTestVariables


@pytest.fixture
def sample_opsd_data():
    """Sample OPSD-style energy data DataFrame matching test_ml_OPSD.py structure."""
    return OPSDTestVariables.create_sample_opsd_dataframe(1000)


@pytest.fixture
def sample_opsd_small_data():
    """Small sample OPSD data for faster tests."""
    return OPSDTestVariables.create_sample_opsd_dataframe(100)


@pytest.fixture
def opsd_cleaned_data_pipeline():
    """Create cleaned data pipeline stages with proper keys."""
    # Create sample data
    sample_data = OPSDTestVariables.create_sample_opsd_dataframe(1000)

    # Create the pipeline stages with the expected keys
    stages = OPSDTestVariables.create_cleaned_data_pipeline_stages(sample_data)

    # Ensure all expected keys exist
    expected_keys = [
        "dropped_missing_target",
        "df_cleaned",
        "df_filtered",
        "df_augmented",
    ]

    # Add missing keys if they don't exist
    if "dropped_missing" not in stages and "dropped_missing_target" in stages:
        stages["dropped_missing"] = stages["dropped_missing_target"]

    if "df_input" not in stages:
        stages["df_input"] = sample_data

    return stages


@pytest.fixture
def opsd_train_test_split(sample_opsd_data):
    """Train-test split matching test_ml_OPSD.py temporal split."""
    return OPSDTestVariables.get_train_test_split(sample_opsd_data)


@pytest.fixture
def opsd_dataset_params():
    """Dataset parameters logged in test_ml_OPSD.py."""
    return OPSDTestVariables.DATASET_PARAMS


@pytest.fixture
def opsd_search_space():
    """Optuna search space from test_ml_OPSD.py."""
    return OPSDTestVariables.SEARCH_SPACE


@pytest.fixture
def opsd_best_hyperparams():
    """Best hyperparameters from Optuna optimization."""
    return OPSDTestVariables.BEST_HYPERPARAMS


@pytest.fixture
def opsd_model_params():
    """Final model parameters from test_ml_OPSD.py."""
    return OPSDTestVariables.MODEL_PARAMS


@pytest.fixture
def opsd_trial_metrics():
    """Sample trial metrics from Optuna optimization."""
    return OPSDTestVariables.SAMPLE_TRIAL_METRICS


@pytest.fixture
def opsd_stepwise_metrics():
    """Stepwise training metrics from incremental XGBoost."""
    return OPSDTestVariables.STEPWISE_METRICS


@pytest.fixture
def opsd_final_metrics():
    """Final evaluation metrics."""
    return OPSDTestVariables.FINAL_METRICS


@pytest.fixture
def opsd_mlflow_tags():
    """MLflow tags used in test_ml_OPSD.py."""
    return OPSDTestVariables.MLFLOW_TAGS


@pytest.fixture
def opsd_artifact_paths():
    """Artifact paths structure from test_ml_OPSD.py."""
    return OPSDTestVariables.ARTIFACT_PATHS


@pytest.fixture
def opsd_emissions_files():
    """Emissions file names for each phase."""
    return OPSDTestVariables.EMISSIONS_FILES


@pytest.fixture
def opsd_whylogs_profiles():
    """WhyLogs profile names for each processing stage."""
    return OPSDTestVariables.WHYLOGS_PROFILES


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def opsd_output_dir(temp_directory):
    """Create output directory structure like test_ml_OPSD.py."""
    output_dir = os.path.join(temp_directory, "whylogs_output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_xgboost_model():
    """Mock XGBoost model matching test_ml_OPSD.py usage."""
    model = Mock()
    model.__class__.__module__ = "xgboost"
    model.__class__.__name__ = "XGBRegressor"

    # Mock model attributes
    model.n_estimators = 75
    model.random_state = 42
    model.eval_metric = "rmse"

    def mock_predict(X):
        # Return realistic predictions for energy load
        return np.random.normal(50000, 5000, len(X))

    def mock_fit(X, y, **kwargs):
        return model

    def mock_get_booster():
        booster_mock = Mock()
        return booster_mock

    model.predict = mock_predict
    model.fit = mock_fit
    model.get_booster = mock_get_booster

    return model


@pytest.fixture
def mock_emissions_tracker():
    """Mock EmissionsTracker following test_ml_OPSD.py patterns."""
    tracker = Mock(spec=EmissionsTracker)
    tracker.start.return_value = None
    tracker.stop.return_value = None
    return tracker


@pytest.fixture
def mock_whylogs_results():
    """Mock WhyLogs results matching test_ml_OPSD.py structure."""
    mock_profile = Mock()

    # Mock profile data that matches OPSD columns
    profile_data = pd.DataFrame(
        {
            "column": [
                "DE_load_actual_entsoe_transparency",
                "DE_solar_generation_actual",
                "DE_wind_generation_actual",
                "feature1",
                "feature2",
            ],
            "count": [1000, 1000, 1000, 1000, 1000],
            "mean": [50000, 5000, 8000, 0.0, 0.0],
            "stddev": [5000, 2000, 3000, 1.0, 1.0],
        }
    )

    mock_profile.view().to_pandas.return_value = profile_data
    mock_profile.to_pandas.return_value = profile_data

    mock_results = Mock()
    mock_results.profile = mock_profile

    return mock_results


@pytest.fixture
def mock_optuna_study():
    """Mock Optuna study with trials matching test_ml_OPSD.py."""
    study = Mock()

    # Create mock trials
    trials = []
    for trial_data in OPSDTestVariables.SAMPLE_TRIAL_METRICS:
        trial = Mock()
        trial.number = trial_data["trial_number"]
        trial.params = {
            "n_estimators": trial_data["n_estimators"],
            "max_depth": trial_data["max_depth"],
            "learning_rate": trial_data["learning_rate"],
        }
        trial.value = trial_data["trial_mse"]
        trials.append(trial)

    study.trials = trials
    study.best_trial = trials[1]  # Best trial is the second one

    return study


@pytest.fixture
def emissions_file_contents():
    """Emissions file contents for all phases."""
    return {
        phase: OPSDTestVariables.get_emissions_file_content(phase)
        for phase in OPSDTestVariables.EMISSIONS_FILES.keys()
    }


@pytest.fixture
def sample_timestamps_content(opsd_train_test_split):
    """Sample timestamps file content."""
    _, _, _, _, train_ts, test_ts = opsd_train_test_split
    return OPSDTestVariables.get_timestamps_file_content(train_ts, test_ts)


@pytest.fixture(autouse=True)
def setup_mlflow():
    """Setup MLflow for testing with temporary tracking URI."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow.set_tracking_uri(f"file://{temp_dir}/mlruns")

        # Set experiment like in test_ml_OPSD.py
        mlflow.set_experiment(OPSDTestVariables.EXPERIMENT_NAME)

        with mlflow.start_run(
            run_name=OPSDTestVariables.RUN_NAME,
            log_system_metrics=OPSDTestVariables.LOG_SYSTEM_METRICS,
        ):
            yield

        # End run if still active
        if mlflow.active_run():
            mlflow.end_run()


@pytest.fixture
def mock_psutil_process():
    """Mock psutil.Process for resource monitoring tests."""
    process = Mock()

    # Mock memory info
    mem_info = Mock()
    mem_info.rss = 1024 * 1024 * 512  # 512 MB in bytes
    process.memory_info.return_value = mem_info

    return process


@pytest.fixture
def mock_mlflow_nested_run():
    """Mock MLflow nested run context manager."""
    mock_run = Mock()
    mock_run.__enter__ = Mock(return_value=mock_run)
    mock_run.__exit__ = Mock(return_value=None)
    return mock_run


@pytest.fixture
def opsd_experiment_config():
    """Complete experiment configuration from test_ml_OPSD.py."""
    return {
        "experiment_name": OPSDTestVariables.EXPERIMENT_NAME,
        "run_name": OPSDTestVariables.RUN_NAME,
        "log_system_metrics": OPSDTestVariables.LOG_SYSTEM_METRICS,
        "tracking_uri": "postgresql+psycopg2://postgres:postgres@localhost:5432/mlflow",
        "output_dir": "whylogs_output",
    }


@pytest.fixture
def opsd_phase_structure():
    """Phase structure from test_ml_OPSD.py workflow."""
    return {
        "data_processing": {
            "emissions_file": "emissions_data.csv",
            "artifacts": ["data_cleaning", "augmentation", "whylogs"],
            "whylogs_profiles": ["input", "cleaned", "filtered", "augmented"],
            "csv_files": [
                "dropped_missing_target.csv",
                "df_cleaned.csv",
                "df_filtered.csv",
                "df_augmented.csv",
            ],
        },
        "hyperparameter_optimization": {
            "emissions_file": "emissions_hyperparams.csv",
            "artifacts": ["code_carbon"],
            "n_trials": 20,
            "nested_runs": True,
        },
        "model_training": {
            "emissions_file": "emissions_train.csv",
            "artifacts": ["model", "dataset", "timestamps"],
            "stepwise_logging": True,
            "csv_files": ["X_train.csv", "X_test.csv"],
            "timestamp_files": ["all_timestamps.txt"],
        },
    }


@pytest.fixture
def sample_model_info():
    """Sample model information dictionary."""
    return {
        "model_name": "test_model",
        "version": "1.0.0",
        "author": "test_user",
        "description": "Test model for unit testing",
    }


@pytest.fixture
def sample_hyperparameters():
    """Sample hyperparameters dictionary."""
    return {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8}


@pytest.fixture
def sample_train_data():
    """Sample training data DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture
def sample_test_data():
    """Sample test data DataFrame."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randn(50),
        }
    )


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary."""
    return {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "mse": 0.05,
    }


@pytest.fixture
def sample_search_space():
    """Sample search space for hyperparameter optimization."""
    return {
        "n_estimators": {"type": "int", "low": 10, "high": 100},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    }


@pytest.fixture
def sample_data_techniques():
    """Sample data techniques dictionary."""
    return {
        "feature_scaling": {
            "method": "StandardScaler",
            "parameters": {"with_mean": "True", "with_std": "True"},
        },
        "feature_selection": {
            "method": "SelectKBest",
            "parameters": {"k": "10", "score_func": "f_classif"},
        },
    }


@pytest.fixture
def sample_timestamps():
    """Sample timestamps for time series testing."""
    return pd.date_range(start="2023-01-01", periods=100, freq="H")


@pytest.fixture
def mock_sklearn_model():
    """Mock sklearn model for testing."""
    model = Mock()
    model.__class__.__module__ = "sklearn"
    model.__class__.__name__ = "RandomForestClassifier"

    def mock_predict(X):
        return np.random.randint(0, 2, len(X))

    model.predict = mock_predict
    return model


@pytest.fixture
def mlflow_stepwise_metrics():
    """Stepwise metrics logged during incremental training."""
    return [
        {"step": 1, "mse": 0.15, "r2_score": 0.75},
        {"step": 2, "mse": 0.12, "r2_score": 0.82},
        {"step": 3, "mse": 0.08, "r2_score": 0.88},
        {"step": 4, "mse": 0.06, "r2_score": 0.91},
        {"step": 5, "mse": 0.045, "r2_score": 0.93},
    ]


@pytest.fixture
def mlflow_input_datasets():
    """MLflow input dataset configurations."""
    return {
        "train_dataset": {
            "source": "X_train split",
            "name": "X_train",
            "context": "training",
        },
        "test_dataset": {
            "source": "X_test split",
            "name": "X_test",
            "context": "testing",
        },
    }


@pytest.fixture
def mlflow_complete_experiment_structure():
    """Complete MLflow experiment structure combining all components."""
    return {
        "run_name": "OPSD_Energy_Data_Model_Stepwise__1",
        "log_system_metrics": True,
        "experiment_name": "ML For OPSD new",
        "phases": [
            {
                "name": "data_processing",
                "artifacts": ["data_cleaning", "augmentation", "whylogs"],
                "emissions_file": "emissions_data.csv",
            },
            {
                "name": "hyperparameter_optimization",
                "artifacts": ["code_carbon"],
                "emissions_file": "emissions_hyperparams.csv",
                "nested_runs": True,
                "n_trials": 20,
            },
            {
                "name": "model_training",
                "artifacts": ["model", "timestamps", "dataset"],
                "emissions_file": "emissions_train.csv",
                "stepwise_logging": True,
            },
        ],
    }
