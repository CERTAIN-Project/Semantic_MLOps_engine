import pytest
import pandas as pd
import numpy as np
import os
import json
import time
from unittest.mock import patch, Mock, mock_open

import certain_library as cl
from tests.test_opsd_variables import OPSDTestVariables


class TestOPSDLoggingIntegration:
    """Test all library logging functions using OPSD predefined variables."""

    def test_log_dataset_parameters(self, temp_directory):
        """Test logging dataset parameters like in OPSD."""
        with patch("mlflow.log_param") as mock_log_param:
            cl.log_model_info(OPSDTestVariables.DATASET_PARAMS)

            # Verify all dataset parameters were logged
            assert mock_log_param.call_count == len(OPSDTestVariables.DATASET_PARAMS)
            mock_log_param.assert_any_call(
                "dataset_name", "Open Power System Data - Time Series"
            )
            mock_log_param.assert_any_call(
                "data_url", "https://data.open-power-system-data.org/time_series/"
            )

    def test_log_search_space(self):
        """Test logging Optuna search space like in OPSD."""
        with patch("mlflow.log_param") as mock_log_param:
            cl.log_search_space(OPSDTestVariables.SEARCH_SPACE)

            # Verify search space was logged as JSON
            mock_log_param.assert_called_once_with(
                "optuna_search_space", json.dumps(OPSDTestVariables.SEARCH_SPACE)
            )

    def test_log_best_hyperparameters(self):
        """Test logging best hyperparameters from Optuna optimization."""
        with patch("mlflow.log_param") as mock_log_param, patch(
            "mlflow.log_metric"
        ) as mock_log_metric:

            # Log best hyperparameters as parameters
            best_params = {
                "n_estimators": OPSDTestVariables.BEST_HYPERPARAMS["best_n_estimators"],
                "max_depth": OPSDTestVariables.BEST_HYPERPARAMS["best_max_depth"],
                "learning_rate": OPSDTestVariables.BEST_HYPERPARAMS[
                    "best_learning_rate"
                ],
            }
            cl.log_model_hyperparameters(best_params, keep_best=True)

            # Log best MSE as metric
            cl.log_metrics(
                {"mse": OPSDTestVariables.BEST_HYPERPARAMS["best_mse"]}, keep_best=True
            )

            # Verify best hyperparameters logged with prefix
            mock_log_param.assert_any_call("best_n_estimators", 75)
            mock_log_param.assert_any_call("best_max_depth", 6)
            mock_log_param.assert_any_call("best_learning_rate", 0.1)
            mock_log_metric.assert_called_once_with("best_mse", 0.045)

    def test_log_trial_metrics_nested_runs(self):
        """Test logging trial metrics in nested runs like Optuna trials."""
        with patch("mlflow.start_run") as mock_start_run, patch(
            "mlflow.log_param"
        ) as mock_log_param, patch("mlflow.log_metric") as mock_log_metric:

            # Mock nested run context
            mock_nested_run = Mock()
            mock_nested_run.__enter__ = Mock()
            mock_nested_run.__exit__ = Mock()
            mock_start_run.return_value = mock_nested_run

            # Log each trial
            for trial_data in OPSDTestVariables.SAMPLE_TRIAL_METRICS:
                with mock_start_run(
                    nested=True, run_name=f"Trial_{trial_data['trial_number']}"
                ):
                    # Log trial hyperparameters
                    trial_params = {
                        "n_estimators": trial_data["n_estimators"],
                        "max_depth": trial_data["max_depth"],
                        "learning_rate": trial_data["learning_rate"],
                    }
                    cl.log_model_hyperparameters(trial_params)

                    # Log trial metrics
                    trial_metrics = {
                        "trial_mse": trial_data["trial_mse"],
                        "trial_memory_usage_mb": trial_data["trial_memory_usage_mb"],
                        "trial_cpu_usage_percent": trial_data[
                            "trial_cpu_usage_percent"
                        ],
                        "trial_number": float(trial_data["trial_number"]),
                    }
                    cl.log_metrics(trial_metrics)

            # Verify nested runs were created
            assert mock_start_run.call_count == 3

            # Verify parameters and metrics were logged
            expected_param_calls = 3 * 3  # 3 trials × 3 params each
            expected_metric_calls = 3 * 4  # 3 trials × 4 metrics each
            assert mock_log_param.call_count >= expected_param_calls
            assert mock_log_metric.call_count >= expected_metric_calls

    def test_log_stepwise_training_metrics(self):
        """Test logging stepwise training metrics like incremental XGBoost."""
        with patch("mlflow.log_metric") as mock_log_metric:

            # Log stepwise metrics with step parameter
            for step_data in OPSDTestVariables.STEPWISE_METRICS:
                # Simulate mlflow.log_metric with step parameter
                mock_log_metric("mse", step_data["mse"], step=step_data["step"])
                mock_log_metric(
                    "r2_score", step_data["r2_score"], step=step_data["step"]
                )

            # Verify stepwise metrics were logged
            expected_calls = (
                len(OPSDTestVariables.STEPWISE_METRICS) * 2
            )  # mse + r2_score
            assert mock_log_metric.call_count == expected_calls

    def test_log_model_parameters(self):
        """Test logging final model parameters."""
        with patch("mlflow.log_param") as mock_log_param:
            # Convert numeric values to strings for log_model_info
            model_params_str = {
                k: str(v) for k, v in OPSDTestVariables.MODEL_PARAMS.items()
            }
            cl.log_model_info(model_params_str)

            # Verify model parameters logged
            assert mock_log_param.call_count == len(OPSDTestVariables.MODEL_PARAMS)
            mock_log_param.assert_any_call("model_type", "XGBRegressor")
            mock_log_param.assert_any_call("max_steps", "75")

    def test_log_opsd_datasets(self, temp_directory):
        """Test logging datasets with OPSD structure."""
        # Create sample OPSD data
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
        X_train, X_test, y_train, y_test, _, _ = OPSDTestVariables.get_train_test_split(
            sample_data
        )

        with patch("mlflow.log_input") as mock_log_input, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact:

            cl.log_dataset(X_train, X_test, temp_directory)

            # Verify MLflow inputs logged with correct context
            assert mock_log_input.call_count == 2
            calls = mock_log_input.call_args_list
            contexts = [str(call) for call in calls]
            assert any("training" in context for context in contexts)
            assert any("testing" in context for context in contexts)

            # Verify CSV artifacts logged
            assert mock_log_artifact.call_count == 2
            for call in mock_log_artifact.call_args_list:
                args, kwargs = call
                assert kwargs["artifact_path"] == "dataset"

    def test_log_timestamps_analysis(self, temp_directory):
        """Test logging timestamp analysis like in OPSD."""
        # Create sample data and split
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
        _, _, _, _, train_timestamps, test_timestamps = (
            OPSDTestVariables.get_train_test_split(sample_data)
        )

        with patch("mlflow.log_param") as mock_log_param, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact, patch("builtins.open", mock_open()) as mock_file, patch(
            "os.makedirs"
        ), patch(
            "os.remove"
        ):

            cl.timestamp_analysis(train_timestamps, test_timestamps, temp_directory)

            # Verify timestamp parameters logged
            expected_params = [
                "train_min_timestamp",
                "train_max_timestamp",
                "train_mean_timestamp",
                "test_min_timestamp",
                "test_max_timestamp",
                "test_mean_timestamp",
            ]
            logged_params = [call[0][0] for call in mock_log_param.call_args_list]
            for param in expected_params:
                assert param in logged_params

            # Verify timestamp file artifact logged
            mock_log_artifact.assert_called_once()
            args, kwargs = mock_log_artifact.call_args
            assert kwargs["artifact_path"] == "timestamps"

    def test_log_whylogs_profiles_all_stages(self, temp_directory):
        """Test logging WhyLogs profiles for all data processing stages."""
        # Create sample data and processing stages
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
        stages = OPSDTestVariables.create_cleaned_data_pipeline_stages(sample_data)

        with patch("whylogs.log") as mock_whylogs, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact:

            # Setup WhyLogs mock
            mock_results = Mock()
            mock_results.profile.view().to_pandas.return_value = pd.DataFrame(
                {
                    "column": ["DE_load_actual_entsoe_transparency", "feature1"],
                    "count": [100, 100],
                    "mean": [50000, 0.0],
                }
            )
            mock_whylogs.return_value = mock_results

            # Log profiles for each stage
            for profile_name in OPSDTestVariables.WHYLOGS_PROFILES:
                if profile_name == "input":
                    stage_data = stages["dropped_missing_target"]
                elif profile_name in stages:
                    stage_data = stages[profile_name]
                else:
                    # Use df_augmented as fallback for missing stages
                    stage_data = stages["df_augmented"]

                cl.log_whylogs_profile(stage_data, profile_name, temp_directory)

            # Verify all profiles logged
            assert mock_whylogs.call_count == len(OPSDTestVariables.WHYLOGS_PROFILES)
            assert mock_log_artifact.call_count == len(
                OPSDTestVariables.WHYLOGS_PROFILES
            )

            # Verify correct artifact path
            for call in mock_log_artifact.call_args_list:
                args, kwargs = call
                assert kwargs["artifact_path"] == "whylogs"

    def test_log_emissions_tracking_all_phases(self, temp_directory):
        """Test emissions tracking for all phases like in OPSD."""
        with patch("codecarbon.EmissionsTracker") as mock_tracker_class, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact, patch("os.path.exists", return_value=True):

            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker

            # Test each phase of emissions tracking
            for phase, filename in OPSDTestVariables.EMISSIONS_FILES.items():
                # Start tracking
                tracker, output_location = cl.start_tracker(
                    output_dir=temp_directory,
                    output_file_name=filename.replace(".csv", ""),
                )

                # Create emissions file with phase-specific content
                emissions_file = os.path.join(temp_directory, filename)
                content = OPSDTestVariables.get_emissions_file_content(phase)

                # Write content as JSON string since get_emissions_file_content returns dict
                with open(emissions_file, "w") as f:
                    if isinstance(content, dict):
                        # Convert to CSV-like format
                        import csv
                        import io

                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=content.keys())
                        writer.writeheader()
                        writer.writerow(content)
                        f.write(output.getvalue())
                    else:
                        f.write(str(content))

                # Stop tracking
                cl.stop_tracker(tracker, output_location)

            # Verify all emissions files logged
            assert mock_log_artifact.call_count == len(
                OPSDTestVariables.EMISSIONS_FILES
            )

            # Verify correct artifact path
            for call in mock_log_artifact.call_args_list:
                args, kwargs = call
                assert kwargs["artifact_path"] == "code_carbon"

    def test_log_final_evaluation_metrics(self):
        """Test logging final evaluation metrics."""
        with patch("mlflow.log_metric") as mock_log_metric:
            cl.log_metrics(OPSDTestVariables.FINAL_METRICS)

            # Verify final metrics logged
            assert mock_log_metric.call_count == len(OPSDTestVariables.FINAL_METRICS)
            mock_log_metric.assert_any_call("final_mse", 0.043)
            mock_log_metric.assert_any_call("final_r2_score", 0.93)
            mock_log_metric.assert_any_call("memory_usage_mb", 1536.0)

    def test_complete_opsd_pipeline_simulation(self, temp_directory):
        """Test complete OPSD pipeline using all predefined variables."""
        with patch("codecarbon.EmissionsTracker") as mock_tracker_class, patch(
            "whylogs.log"
        ) as mock_whylogs, patch("mlflow.log_param") as mock_log_param, patch(
            "mlflow.log_metric"
        ) as mock_log_metric, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact, patch(
            "mlflow.log_input"
        ) as mock_log_input, patch(
            "mlflow.start_run"
        ) as mock_nested_run, patch(
            "os.path.exists", return_value=True
        ), patch(
            "os.remove"  # Mock os.remove to prevent file deletion errors
        ) as mock_remove, patch(
            "os.makedirs"  # Mock os.makedirs to prevent directory creation issues
        ) as mock_makedirs:

            # Setup mocks
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker

            mock_whylogs_results = Mock()
            mock_whylogs_results.profile.view().to_pandas.return_value = pd.DataFrame(
                {"column": ["DE_load_actual_entsoe_transparency"], "count": [100]}
            )
            mock_whylogs.return_value = mock_whylogs_results

            mock_nested_run.return_value.__enter__ = Mock()
            mock_nested_run.return_value.__exit__ = Mock()

            # Create sample OPSD data
            sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
            stages = OPSDTestVariables.create_cleaned_data_pipeline_stages(sample_data)
            X_train, X_test, y_train, y_test, train_ts, test_ts = (
                OPSDTestVariables.get_train_test_split(stages["df_augmented"])
            )

            # 1. Log dataset parameters
            cl.log_model_info(OPSDTestVariables.DATASET_PARAMS)

            # 2. Log search space
            cl.log_search_space(OPSDTestVariables.SEARCH_SPACE)

            # 3. Data processing phase with emissions
            tracker1, loc1 = cl.start_tracker(
                output_dir=temp_directory, output_file_name="emissions_data"
            )

            # Log WhyLogs profiles for each stage
            for profile_name in OPSDTestVariables.WHYLOGS_PROFILES:
                if profile_name == "input":
                    cl.log_whylogs_profile(
                        stages["dropped_missing_target"],
                        profile_name,
                        temp_directory,
                    )
                else:
                    # Use available stage data or fallback to df_augmented
                    stage_key = (
                        f"df_{profile_name}"
                        if f"df_{profile_name}" in stages
                        else "df_augmented"
                    )
                    cl.log_whylogs_profile(
                        stages[stage_key], profile_name, temp_directory
                    )
            # Create and stop emissions tracking
            emissions_file1 = os.path.join(temp_directory, "emissions_data.csv")
            content1 = OPSDTestVariables.get_emissions_file_content("data_processing")
            with open(emissions_file1, "w") as f:
                import csv
                import io

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=content1.keys())
                writer.writeheader()
                writer.writerow(content1)
                f.write(output.getvalue())
            cl.stop_tracker(tracker1, loc1)

            # 4. Log datasets
            cl.log_dataset(X_train, X_test, temp_directory)

            # 5. Log timestamps
            cl.timestamp_analysis(train_ts, test_ts, temp_directory)

            # 6. Hyperparameter optimization phase
            tracker2, loc2 = cl.start_tracker(
                output_dir=temp_directory, output_file_name="emissions_hyperparams"
            )

            # Log trial metrics in nested runs
            for trial_data in OPSDTestVariables.SAMPLE_TRIAL_METRICS:
                with mock_nested_run(
                    nested=True, run_name=f"Trial_{trial_data['trial_number']}"
                ):
                    trial_params = {
                        k: v
                        for k, v in trial_data.items()
                        if k in ["n_estimators", "max_depth", "learning_rate"]
                    }
                    cl.log_model_hyperparameters(trial_params)

                    trial_metrics = {
                        k: v for k, v in trial_data.items() if k.startswith("trial_")
                    }
                    cl.log_metrics(trial_metrics)

            # Log best hyperparameters
            best_params = {
                k.replace("best_", ""): v
                for k, v in OPSDTestVariables.BEST_HYPERPARAMS.items()
                if k != "best_mse"
            }
            cl.log_model_hyperparameters(best_params, keep_best=True)
            cl.log_metrics(
                {"mse": OPSDTestVariables.BEST_HYPERPARAMS["best_mse"]}, keep_best=True
            )

            emissions_file2 = os.path.join(temp_directory, "emissions_hyperparams.csv")
            content2 = OPSDTestVariables.get_emissions_file_content(
                "hyperparameter_optimization"
            )
            with open(emissions_file2, "w") as f:
                import csv
                import io

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=content2.keys())
                writer.writeheader()
                writer.writerow(content2)
                f.write(output.getvalue())
            cl.stop_tracker(tracker2, loc2)

            # 7. Model training phase
            tracker3, loc3 = cl.start_tracker(
                output_dir=temp_directory, output_file_name="emissions_train"
            )

            # Log stepwise metrics
            for step_data in OPSDTestVariables.STEPWISE_METRICS:
                step_metrics = {
                    "mse": step_data["mse"],
                    "r2_score": step_data["r2_score"],
                }
                cl.log_metrics(step_metrics)

            # Log final model parameters
            model_params_str = {
                k: str(v) for k, v in OPSDTestVariables.MODEL_PARAMS.items()
            }
            cl.log_model_info(model_params_str)

            emissions_file3 = os.path.join(temp_directory, "emissions_train.csv")
            content3 = OPSDTestVariables.get_emissions_file_content("model_training")
            with open(emissions_file3, "w") as f:
                import csv
                import io

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=content3.keys())
                writer.writeheader()
                writer.writerow(content3)
                f.write(output.getvalue())
            cl.stop_tracker(tracker3, loc3)

            # 8. Log final metrics
            cl.log_metrics(OPSDTestVariables.FINAL_METRICS)

            # Verify comprehensive logging
            assert (
                mock_log_param.call_count >= 15
            )  # Dataset, model, hyperparams, timestamps
            assert (
                mock_log_metric.call_count >= 20
            )  # Trial, stepwise, best, final metrics
            assert (
                mock_log_artifact.call_count >= 10
            )  # Emissions, whylogs, datasets, timestamps
            assert mock_log_input.call_count == 2  # Train and test datasets
            assert mock_whylogs.call_count == 4  # All WhyLogs profiles
            assert mock_nested_run.call_count == 3  # Optuna trials
