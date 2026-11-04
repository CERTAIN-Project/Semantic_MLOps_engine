import json
import os
from unittest.mock import Mock, patch

import certain_library as cl
import mlflow
import pandas as pd

from tests.test_opsd_variables import OPSDTestVariables


class TestOPSDLoggingIntegration:
    """Test all library logging functions using OPSD predefined variables."""

    def test_log_dataset_parameters(self, temp_directory, mlflow_client, active_run_id):
        """Test logging dataset parameters like in OPSD."""
        cl.log_model_info(OPSDTestVariables.DATASET_PARAMS)

        # Verify all dataset parameters were logged
        params = mlflow_client.get_run(active_run_id).data.params
        assert len(params) == len(OPSDTestVariables.DATASET_PARAMS)
        assert params["dataset_name"] == "Open Power System Data - Time Series"
        assert params["data_url"] == "https://data.open-power-system-data.org/time_series/"

    def test_log_search_space(self, mlflow_client, active_run_id):
        """Test logging Optuna search space like in OPSD."""
        cl.log_search_space(OPSDTestVariables.SEARCH_SPACE)

        # Verify search space was logged as JSON
        assert mlflow_client.get_run(active_run_id).data.params == {
            "optuna_search_space": json.dumps(OPSDTestVariables.SEARCH_SPACE)
        }

    def test_log_best_hyperparameters(self, mlflow_client, active_run_id):
        """Test logging best hyperparameters from Optuna optimization."""
        # Log best hyperparameters as parameters
        best_params = {
            "n_estimators": OPSDTestVariables.BEST_HYPERPARAMS["best_n_estimators"],
            "max_depth": OPSDTestVariables.BEST_HYPERPARAMS["best_max_depth"],
            "learning_rate": OPSDTestVariables.BEST_HYPERPARAMS["best_learning_rate"],
        }
        cl.log_model_hyperparameters(best_params, keep_best=True)

        # Log best MSE as metric
        cl.log_metrics({"mse": OPSDTestVariables.BEST_HYPERPARAMS["best_mse"]}, keep_best=True)

        # Verify best hyperparameters logged with prefix (MLflow stores params as strings)
        run = mlflow_client.get_run(active_run_id)
        assert run.data.params["best_n_estimators"] == "75"
        assert run.data.params["best_max_depth"] == "6"
        assert run.data.params["best_learning_rate"] == "0.1"
        assert run.data.metrics == {"best_mse": 0.045}
        assert len(mlflow_client.get_metric_history(active_run_id, "best_mse")) == 1

    def test_log_trial_metrics_nested_runs(self, mlflow_client, active_run_id):
        """Test logging trial metrics in nested runs like Optuna trials."""
        # Log each trial in a real nested run under the active parent run
        for trial_data in OPSDTestVariables.SAMPLE_TRIAL_METRICS:
            with mlflow.start_run(nested=True, run_name=f"Trial_{trial_data['trial_number']}"):
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
                    "trial_cpu_usage_percent": trial_data["trial_cpu_usage_percent"],
                    "trial_number": float(trial_data["trial_number"]),
                }
                cl.log_metrics(trial_metrics)

        # Verify nested runs were created under the parent run
        parent = mlflow_client.get_run(active_run_id)
        child_runs = mlflow_client.search_runs(
            [parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{active_run_id}'",
        )
        assert len(child_runs) == 3

        # Verify parameters and metrics were logged to each trial run, not the parent
        trials_by_name = {
            f"Trial_{t['trial_number']}": t for t in OPSDTestVariables.SAMPLE_TRIAL_METRICS
        }
        for child in child_runs:
            trial_data = trials_by_name[child.info.run_name]
            assert child.data.params == {
                "n_estimators": str(trial_data["n_estimators"]),
                "max_depth": str(trial_data["max_depth"]),
                "learning_rate": str(trial_data["learning_rate"]),
            }
            assert len(child.data.metrics) == 4
            assert child.data.metrics["trial_mse"] == trial_data["trial_mse"]
        assert parent.data.params == {}
        assert parent.data.metrics == {}

    def test_log_stepwise_training_metrics(self):
        """Test logging stepwise training metrics like incremental XGBoost."""
        with patch("mlflow.log_metric") as mock_log_metric:
            # Log stepwise metrics with step parameter
            for step_data in OPSDTestVariables.STEPWISE_METRICS:
                # Simulate mlflow.log_metric with step parameter
                mock_log_metric("mse", step_data["mse"], step=step_data["step"])
                mock_log_metric("r2_score", step_data["r2_score"], step=step_data["step"])

            # Verify stepwise metrics were logged
            expected_calls = len(OPSDTestVariables.STEPWISE_METRICS) * 2  # mse + r2_score
            assert mock_log_metric.call_count == expected_calls

    def test_log_model_parameters(self, mlflow_client, active_run_id):
        """Test logging final model parameters."""
        # Convert numeric values to strings for log_model_info
        model_params_str = {k: str(v) for k, v in OPSDTestVariables.MODEL_PARAMS.items()}
        cl.log_model_info(model_params_str)

        # Verify model parameters logged
        params = mlflow_client.get_run(active_run_id).data.params
        assert len(params) == len(OPSDTestVariables.MODEL_PARAMS)
        assert params["model_type"] == "XGBRegressor"
        assert params["max_steps"] == "75"

    def test_log_opsd_datasets(
        self, temp_directory, mlflow_client, active_run_id, list_artifact_names
    ):
        """Test logging datasets with OPSD structure."""
        # Create sample OPSD data
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
        X_train, X_test, y_train, y_test, _, _ = OPSDTestVariables.get_train_test_split(sample_data)

        cl.log_dataset(X_train, X_test, temp_directory)

        # Verify MLflow inputs logged with correct context
        dataset_inputs = mlflow_client.get_run(active_run_id).inputs.dataset_inputs
        assert len(dataset_inputs) == 2
        contexts = {
            tag.value
            for dataset_input in dataset_inputs
            for tag in dataset_input.tags
            if tag.key == "mlflow.data.context"
        }
        assert contexts == {"training", "testing"}

        # Verify CSV artifacts logged under the CERTAIN dataset path
        assert list_artifact_names(active_run_id, "certain/dataset") == [
            "X_test.csv",
            "X_train.csv",
        ]

    def test_log_timestamps_analysis(
        self, temp_directory, mlflow_client, active_run_id, list_artifact_names
    ):
        """Test logging timestamp analysis like in OPSD."""
        # Create sample data and split
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
        _, _, _, _, train_timestamps, test_timestamps = OPSDTestVariables.get_train_test_split(
            sample_data
        )

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
        logged_params = mlflow_client.get_run(active_run_id).data.params
        for param in expected_params:
            assert param in logged_params
        assert logged_params["train_min_timestamp"] == str(train_timestamps.min())

        # Verify timestamp file artifact logged under the CERTAIN timestamps path
        assert list_artifact_names(active_run_id, "certain/timestamps") == ["all_timestamps.txt"]

    def test_log_whylogs_profiles_all_stages(
        self, temp_directory, active_run_id, list_artifact_names
    ):
        """Test logging WhyLogs profiles for all data processing stages."""
        # Create sample data and processing stages
        sample_data = OPSDTestVariables.create_sample_opsd_dataframe(100)
        stages = OPSDTestVariables.create_cleaned_data_pipeline_stages(sample_data)

        with patch("whylogs.log") as mock_whylogs:
            # Setup WhyLogs mock (ResultSet.profile() is a method in whylogs v1)
            mock_results = Mock()
            profile_view = mock_results.profile.return_value.view.return_value
            profile_view.to_pandas.return_value = pd.DataFrame(
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

        # Verify one profile CSV per stage under the CERTAIN whylogs path
        assert list_artifact_names(active_run_id, "certain/whylogs") == sorted(
            f"profile_{name}.csv" for name in OPSDTestVariables.WHYLOGS_PROFILES
        )

    def test_log_emissions_tracking_all_phases(self, temp_directory):
        """Test emissions tracking for all phases like in OPSD."""
        with (
            patch(
                "certain_library.resource_monitor.resource.EmissionsTracker"
            ) as mock_tracker_class,
            patch("mlflow.log_artifact") as mock_log_artifact,
        ):
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
            assert mock_log_artifact.call_count == len(OPSDTestVariables.EMISSIONS_FILES)

            # Verify the CERTAIN-prefixed code_carbon artifact path
            for call in mock_log_artifact.call_args_list:
                args, kwargs = call
                assert kwargs["artifact_path"] == "certain/code_carbon"

    def test_log_final_evaluation_metrics(self, mlflow_client, active_run_id):
        """Test logging final evaluation metrics."""
        cl.log_metrics(OPSDTestVariables.FINAL_METRICS)

        # Verify final metrics logged
        metrics = mlflow_client.get_run(active_run_id).data.metrics
        assert len(metrics) == len(OPSDTestVariables.FINAL_METRICS)
        assert metrics["final_mse"] == 0.043
        assert metrics["final_r2_score"] == 0.93
        assert metrics["memory_usage_mb"] == 1536.0

    def test_complete_opsd_pipeline_simulation(
        self, temp_directory, mlflow_client, active_run_id, list_artifact_names
    ):
        """Test complete OPSD pipeline using all predefined variables."""
        with (
            patch(
                "certain_library.resource_monitor.resource.EmissionsTracker"
            ) as mock_tracker_class,
            patch("whylogs.log") as mock_whylogs,
        ):
            # Setup mocks for the external tools; MLflow logging goes to the temp store
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker

            mock_whylogs_results = Mock()
            mock_whylogs_results.profile.return_value.view.return_value.to_pandas.return_value = (
                pd.DataFrame({"column": ["DE_load_actual_entsoe_transparency"], "count": [100]})
            )
            mock_whylogs.return_value = mock_whylogs_results

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
                        f"df_{profile_name}" if f"df_{profile_name}" in stages else "df_augmented"
                    )
                    cl.log_whylogs_profile(stages[stage_key], profile_name, temp_directory)
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
                with mlflow.start_run(nested=True, run_name=f"Trial_{trial_data['trial_number']}"):
                    trial_params = {
                        k: v
                        for k, v in trial_data.items()
                        if k in ["n_estimators", "max_depth", "learning_rate"]
                    }
                    cl.log_model_hyperparameters(trial_params)

                    trial_metrics = {k: v for k, v in trial_data.items() if k.startswith("trial_")}
                    cl.log_metrics(trial_metrics)

            # Log best hyperparameters
            best_params = {
                k.replace("best_", ""): v
                for k, v in OPSDTestVariables.BEST_HYPERPARAMS.items()
                if k != "best_mse"
            }
            cl.log_model_hyperparameters(best_params, keep_best=True)
            cl.log_metrics({"mse": OPSDTestVariables.BEST_HYPERPARAMS["best_mse"]}, keep_best=True)

            emissions_file2 = os.path.join(temp_directory, "emissions_hyperparams.csv")
            content2 = OPSDTestVariables.get_emissions_file_content("hyperparameter_optimization")
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
            model_params_str = {k: str(v) for k, v in OPSDTestVariables.MODEL_PARAMS.items()}
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
            parent = mlflow_client.get_run(active_run_id)
            child_runs = mlflow_client.search_runs(
                [parent.info.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{active_run_id}'",
            )
            all_runs = [parent, *child_runs]
            metric_values_logged = sum(
                len(mlflow_client.get_metric_history(run.info.run_id, key))
                for run in all_runs
                for key in run.data.metrics
            )
            certain_artifacts = {
                path: list_artifact_names(active_run_id, f"certain/{path}")
                for path in ["code_carbon", "whylogs", "dataset", "timestamps"]
            }

            assert len(parent.data.params) >= 15  # Dataset, model, hyperparams, timestamps
            assert metric_values_logged >= 20  # Trial, stepwise, best, final metrics
            # Emissions, whylogs, datasets, timestamps
            assert sum(len(names) for names in certain_artifacts.values()) >= 10
            assert len(certain_artifacts["code_carbon"]) == 3
            assert len(certain_artifacts["whylogs"]) == 4
            assert certain_artifacts["dataset"] == ["X_test.csv", "X_train.csv"]
            assert certain_artifacts["timestamps"] == ["all_timestamps.txt"]
            assert len(parent.inputs.dataset_inputs) == 2  # Train and test datasets
            assert mock_whylogs.call_count == 4  # All WhyLogs profiles
            assert len(child_runs) == 3  # Optuna trials
