import json

import pandas as pd
import pytest
from certain_library.train_monitor.log_metrics import log_metrics, log_search_space
from certain_library.train_monitor.log_model import (
    log_model_architecture,
    log_model_hyperparameters,
    log_model_info,
    log_model_signature,
)


class TestLogModelInfo:
    """Test log_model_info function following OPSD patterns."""

    def test_log_opsd_dataset_parameters(self, opsd_dataset_params, mlflow_client, active_run_id):
        """Test logging dataset parameters like in OPSD."""
        log_model_info(opsd_dataset_params)

        # Verify exactly the dataset parameters were logged to the active run
        params = mlflow_client.get_run(active_run_id).data.params
        assert params == opsd_dataset_params
        assert params["dataset_name"] == "Open Power System Data - Time Series"
        assert params["data_url"] == "https://data.open-power-system-data.org/time_series/"

    def test_log_opsd_model_parameters(self, opsd_model_params, mlflow_client, active_run_id):
        """Test logging model parameters from OPSD."""
        # Convert numeric values to strings for log_model_info
        model_params_str = {k: str(v) for k, v in opsd_model_params.items()}
        log_model_info(model_params_str)

        # Verify model parameters logged
        params = mlflow_client.get_run(active_run_id).data.params
        assert len(params) == len(opsd_model_params)
        assert params["model_type"] == "XGBRegressor"
        assert params["max_steps"] == "75"
        assert params["random_state"] == "42"

    def test_log_model_info_empty_dict(self):
        """Test with empty dictionary."""
        with pytest.raises(ValueError, match="Input model_information dictionary cannot be empty"):
            log_model_info({})

    def test_log_model_info_non_string_values(self):
        """Test with non-string values."""
        invalid_info = {"model_type": "XGBRegressor", "max_steps": 75}
        with pytest.raises(ValueError, match="Value for key 'max_steps' must be a string"):
            log_model_info(invalid_info)


class TestLogModelHyperparameters:
    """Test log_model_hyperparameters function following OPSD patterns."""

    def test_log_opsd_best_hyperparameters(
        self, opsd_best_hyperparams, mlflow_client, active_run_id
    ):
        """Test logging best hyperparameters from Optuna optimization."""
        # Extract and convert best hyperparameters
        best_params = {
            "n_estimators": opsd_best_hyperparams["best_n_estimators"],
            "max_depth": opsd_best_hyperparams["best_max_depth"],
            "learning_rate": opsd_best_hyperparams["best_learning_rate"],
        }
        log_model_hyperparameters(best_params, keep_best=True)

        # Verify best hyperparameters logged with prefix (MLflow stores params as strings)
        params = mlflow_client.get_run(active_run_id).data.params
        assert params == {
            "best_n_estimators": "75",
            "best_max_depth": "6",
            "best_learning_rate": "0.1",
        }

    def test_log_trial_hyperparameters(self, opsd_trial_metrics, mlflow_client, active_run_id):
        """Test logging trial hyperparameters from OPSD."""
        trial = opsd_trial_metrics[1]  # Best trial
        trial_params = {
            "n_estimators": trial["n_estimators"],
            "max_depth": trial["max_depth"],
            "learning_rate": trial["learning_rate"],
        }

        log_model_hyperparameters(trial_params)
        params = mlflow_client.get_run(active_run_id).data.params
        assert params["n_estimators"] == "75"
        assert params["max_depth"] == "6"
        assert params["learning_rate"] == "0.1"

    def test_log_hyperparameters_empty_dict(self):
        """Test with empty dictionary."""
        with pytest.raises(ValueError, match="Input dictionary cannot be empty"):
            log_model_hyperparameters({})


class TestLogModelSignature:
    """Test log_model_signature function with OPSD model patterns."""

    # def test_log_xgboost_model_signature(
    #     self, mock_xgboost_model, sample_opsd_small_data
    # ):
    #     """Test logging XGBoost model signature like in OPSD."""
    #     # Prepare features (drop target column)
    #     X = sample_opsd_small_data.select_dtypes(include=[np.number]).drop(
    #         columns=["DE_load_actual_entsoe_transparency"]
    #     )

    #     with patch("mlflow.xgboost.log_model") as mock_log_model:
    #         # Make sure the model has a predict method for the function to work
    #         mock_xgboost_model.predict.return_value = np.array(
    #             [1.0, 2.0, 3.0, 4.0, 5.0]
    #         )

    #         log_model_signature(mock_xgboost_model, X)

    #         # First verify that the function was called
    #         assert (
    #             mock_log_model.call_count > 0
    #         ), "mlflow.xgboost.log_model was never called"

    #         # Then verify the call had the expected arguments
    #         call_args = mock_log_model.call_args
    #         assert call_args is not None, "No call arguments found"
    #         assert call_args[0][0] == mock_xgboost_model  # model argument
    #         assert "signature" in call_args[1]  # keyword arguments
    #         assert "input_example" in call_args[1]

    def test_log_model_signature_empty_data(self, mock_xgboost_model):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Input train_data cannot be empty"):
            log_model_signature(mock_xgboost_model, empty_df)


class TestLogMetrics:
    """Test log_metrics function following OPSD patterns."""

    def test_log_opsd_stepwise_metrics(self, opsd_stepwise_metrics, mlflow_client, active_run_id):
        """Test logging stepwise metrics like incremental XGBoost training."""
        for step_data in opsd_stepwise_metrics:
            metrics = {"mse": step_data["mse"], "r2_score": step_data["r2_score"]}
            log_metrics(metrics)

        # Verify every stepwise value was logged (one history entry per call)
        mse_history = mlflow_client.get_metric_history(active_run_id, "mse")
        r2_history = mlflow_client.get_metric_history(active_run_id, "r2_score")
        expected_calls = len(opsd_stepwise_metrics) * 2  # mse + r2_score
        assert len(mse_history) + len(r2_history) == expected_calls
        assert [m.value for m in mse_history] == [s["mse"] for s in opsd_stepwise_metrics]
        assert [m.value for m in r2_history] == [s["r2_score"] for s in opsd_stepwise_metrics]

    def test_log_opsd_trial_metrics(self, opsd_trial_metrics, mlflow_client, active_run_id):
        """Test logging trial metrics from Optuna optimization."""
        trial = opsd_trial_metrics[0]
        trial_metrics = {
            "trial_mse": trial["trial_mse"],
            "trial_memory_usage_mb": trial["trial_memory_usage_mb"],
            "trial_cpu_usage_percent": trial["trial_cpu_usage_percent"],
            "trial_number": float(trial["trial_number"]),
        }

        log_metrics(trial_metrics)
        metrics = mlflow_client.get_run(active_run_id).data.metrics
        assert metrics["trial_mse"] == trial["trial_mse"]
        assert metrics["trial_memory_usage_mb"] == trial["trial_memory_usage_mb"]

    def test_log_best_mse_metric(self, opsd_best_hyperparams, mlflow_client, active_run_id):
        """Test logging best MSE metric with prefix."""
        log_metrics({"mse": opsd_best_hyperparams["best_mse"]}, keep_best=True)
        assert mlflow_client.get_run(active_run_id).data.metrics == {"best_mse": 0.045}
        assert len(mlflow_client.get_metric_history(active_run_id, "best_mse")) == 1

    def test_log_final_metrics(self, opsd_final_metrics, mlflow_client, active_run_id):
        """Test logging final evaluation metrics."""
        log_metrics(opsd_final_metrics)

        # Verify final metrics logged
        metrics = mlflow_client.get_run(active_run_id).data.metrics
        assert len(metrics) == len(opsd_final_metrics)
        assert metrics["final_mse"] == 0.043
        assert metrics["final_r2_score"] == 0.93


class TestLogSearchSpace:
    """Test log_search_space function following OPSD patterns."""

    def test_log_opsd_search_space(self, opsd_search_space, mlflow_client, active_run_id):
        """Test logging Optuna search space from OPSD."""
        log_search_space(opsd_search_space)

        # Verify search space was logged as JSON with correct parameter name
        assert mlflow_client.get_run(active_run_id).data.params == {
            "optuna_search_space": json.dumps(opsd_search_space)
        }

    def test_log_search_space_empty(self):
        """Test with empty search space."""
        with pytest.raises(ValueError, match="Search space dictionary cannot be empty"):
            log_search_space({})

    def test_log_search_space_missing_type(self):
        """Test with missing type key."""

        # Add type annotation to satisfy the type checker
        invalid_space: dict[str, dict[str, str | float | int | bool | list]] = {
            "param1": {"low": 1, "high": 10}
        }
        with pytest.raises(ValueError, match="Parameter 'param1' is missing required 'type' key"):
            log_search_space(invalid_space)


class TestLogModelArchitecture:
    """Test log_model_architecture function."""

    def test_log_xgboost_architecture(self, mlflow_client, active_run_id):
        """Test logging XGBoost model architecture like OPSD."""
        log_model_architecture(
            losses=["rmse"],
            optimizer="gbdt",
            regularization="l2",
            early_stopping=False,
        )

        # MLflow stores params as strings
        params = mlflow_client.get_run(active_run_id).data.params
        assert len(params) == 4
        assert params["losses"] == str(["rmse"])
        assert params["optimizer"] == "gbdt"
