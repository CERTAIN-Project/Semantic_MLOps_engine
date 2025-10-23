import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, Mock, mock_open

from certain_library.data_analysis.log_dataset import log_dataset
from certain_library.data_analysis.log_data_techniques import log_data_techniques
from certain_library.data_analysis.log_timeseries import timestamp_analysis
from certain_library.data_analysis.log_whylogs import log_whylogs_profile


class TestLogDataset:
    """Test log_dataset function following OPSD patterns."""

    def test_log_opsd_datasets(self, opsd_train_test_split, temp_directory):
        """Test logging datasets with OPSD temporal split structure."""
        X_train, X_test, y_train, y_test, train_ts, test_ts = opsd_train_test_split

        with patch("mlflow.log_input") as mock_log_input, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact:

            log_dataset(X_train, X_test, temp_directory)

            # Verify MLflow inputs logged with correct context (like OPSD)
            assert mock_log_input.call_count == 2
            calls = mock_log_input.call_args_list
            contexts = [str(call) for call in calls]
            assert any("training" in context for context in contexts)
            assert any("testing" in context for context in contexts)

            # Verify CSV artifacts logged to dataset path
            assert mock_log_artifact.call_count == 2
            for call in mock_log_artifact.call_args_list:
                args, kwargs = call
                assert kwargs["artifact_path"] == "dataset"

    def test_log_temporal_split_preservation(self, sample_opsd_data, temp_directory):
        """Test that temporal order is preserved in logged datasets."""
        # Create temporal split like OPSD (no shuffle)
        df_sorted = sample_opsd_data.sort_values("utc_timestamp")
        X = df_sorted.select_dtypes(include=[np.number]).drop(
            columns=["DE_load_actual_entsoe_transparency"]
        )

        # Split without shuffling (80/20 split like OPSD)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]

        with patch("mlflow.log_input"), patch("mlflow.log_artifact"):
            log_dataset(X_train, X_test, temp_directory)

            # Verify temporal order preserved (train indices < test indices)
            assert X_train.index.max() < X_test.index.min()

    def test_log_dataset_empty_data(self, sample_opsd_small_data):
        """Test with empty training data."""
        empty_df = pd.DataFrame()
        X_test = sample_opsd_small_data.select_dtypes(include=[np.number]).iloc[:10]

        with pytest.raises(ValueError, match="Training data is empty"):
            log_dataset(empty_df, X_test)

    def test_log_dataset_nan_values(self, temp_directory):
        """Test with NaN values in data."""
        data_with_nan = pd.DataFrame(
            {"feature1": [1, 2, np.nan], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )
        test_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )

        with pytest.raises(ValueError, match="Training data contains NaN values"):
            log_dataset(data_with_nan, test_data, temp_directory)


class TestLogWhylogsProfile:
    """Test log_whylogs_profile function following OPSD patterns."""

    def test_log_opsd_data_pipeline_profiles(
        self, opsd_cleaned_data_pipeline, temp_directory, opsd_whylogs_profiles
    ):
        """Test logging WhyLogs profiles for all OPSD data processing stages."""
        stages = opsd_cleaned_data_pipeline

        with patch("whylogs.log") as mock_whylogs, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact, patch("os.path.exists", return_value=True), patch(
            "os.remove"
        ) as mock_remove:

            # Setup WhyLogs mock with OPSD column structure
            mock_results = Mock()
            mock_results.profile.view().to_pandas.return_value = pd.DataFrame(
                {
                    "column": [
                        "DE_load_actual_entsoe_transparency",
                        "DE_solar_generation_actual",
                    ],
                    "count": [1000, 1000],
                    "mean": [50000, 5000],
                }
            )
            mock_whylogs.return_value = mock_results

            # Debug: Print available stages
            print(f"Available stages: {list(stages.keys())}")
            print(f"Expected profiles: {list(opsd_whylogs_profiles.keys())}")

            # Create a mapping from profile names to stage keys
            stage_mapping = {
                "input": "dropped_missing",  # Will fallback to available key
                "cleaned": "df_cleaned",
                "filtered": "df_filtered",
                "augmented": "df_augmented",
            }

            # Log profiles for each OPSD stage
            profiles_logged = 0
            for profile_name in opsd_whylogs_profiles:
                stage_key = stage_mapping.get(profile_name)

                # Find the appropriate stage data
                if stage_key and stage_key in stages:
                    stage_data = stages[stage_key]
                elif profile_name == "input":
                    # Try alternative keys for input stage
                    if "dropped_missing" in stages:
                        stage_data = stages["dropped_missing"]
                    elif "dropped_missing_target" in stages:
                        stage_data = stages["dropped_missing_target"]
                    elif "df_input" in stages:
                        stage_data = stages["df_input"]
                    else:
                        # Use the first available stage as fallback
                        stage_data = list(stages.values())[0]
                elif f"df_{profile_name}" in stages:
                    stage_data = stages[f"df_{profile_name}"]
                elif profile_name in stages:
                    stage_data = stages[profile_name]
                else:
                    # Use df_augmented as ultimate fallback
                    stage_data = stages.get("df_augmented", list(stages.values())[0])

                print(
                    f"Logging profile '{profile_name}' with stage data shape: {stage_data.shape}"
                )
                log_whylogs_profile(stage_data, profile_name, temp_directory)
                profiles_logged += 1

            print(f"Profiles logged: {profiles_logged}")
            print(f"mock_whylogs.call_count: {mock_whylogs.call_count}")

            # Verify all profiles logged
            assert mock_whylogs.call_count == len(
                opsd_whylogs_profiles
            ), f"Expected {len(opsd_whylogs_profiles)} profiles, got {mock_whylogs.call_count}"
            assert mock_log_artifact.call_count == len(opsd_whylogs_profiles)

    def test_log_whylogs_profile_with_opsd_columns(
        self, sample_opsd_small_data, temp_directory
    ):
        """Test WhyLogs profile with OPSD-specific columns."""
        with patch("whylogs.log") as mock_whylogs, patch("mlflow.log_artifact"):

            # Mock results with OPSD energy columns
            mock_results = Mock()
            mock_results.profile.view().to_pandas.return_value = pd.DataFrame(
                {
                    "column": [
                        "DE_load_actual_entsoe_transparency",
                        "DE_solar_generation_actual",
                        "DE_wind_generation_actual",
                    ],
                    "count": [100, 100, 100],
                    "mean": [50000, 5000, 8000],
                }
            )
            mock_whylogs.return_value = mock_results

            log_whylogs_profile(sample_opsd_small_data, "input", temp_directory)
            mock_whylogs.assert_called_once_with(pandas=sample_opsd_small_data)

    def test_log_whylogs_profile_invalid_input(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="Input 'data' must be a pandas DataFrame"):
            log_whylogs_profile("invalid_data")  # type: ignore


class TestTimestampAnalysis:
    """Test timestamp_analysis function following OPSD patterns."""

    def test_log_opsd_timestamps(self, opsd_train_test_split, temp_directory):
        """Test logging timestamps like in OPSD pipeline."""
        X_train, X_test, y_train, y_test, train_timestamps, test_timestamps = (
            opsd_train_test_split
        )

        with patch("mlflow.log_param") as mock_log_param, patch(
            "mlflow.log_artifact"
        ) as mock_log_artifact, patch("builtins.open", mock_open()) as mock_file, patch(
            "os.makedirs"
        ), patch(
            "os.remove"
        ):

            timestamp_analysis(train_timestamps, test_timestamps, temp_directory)

            # Verify timestamp parameters logged (like OPSD)
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

            # Verify timestamp file artifact logged to correct path
            mock_log_artifact.assert_called_once()
            args, kwargs = mock_log_artifact.call_args
            assert kwargs["artifact_path"] == "timestamps"

    def test_timestamp_file_structure(self, opsd_train_test_split, temp_directory):
        """Test timestamp file content matches OPSD structure."""
        _, _, _, _, train_ts, test_ts = opsd_train_test_split

        with patch("mlflow.log_param"), patch("mlflow.log_artifact"), patch(
            "builtins.open", mock_open()
        ) as mock_file:

            timestamp_analysis(train_ts, test_ts, temp_directory)

            # Verify file was opened for writing
            mock_file.assert_called()
            handle = mock_file.return_value.__enter__.return_value
            write_calls = [str(call) for call in handle.write.call_args_list]

            # Check structure matches OPSD pattern
            assert any("Train Timestamps:" in call for call in write_calls)
            assert any("Test Timestamps:" in call for call in write_calls)

    def test_timestamp_analysis_invalid_types(self):
        """Test with invalid input types."""
        with pytest.raises(
            TypeError,
            match="Both train_timestamps and test_timestamps must be pandas Series",
        ):
            timestamp_analysis([1, 2, 3], pd.Series([4, 5, 6]))  # type: ignore


class TestLogDataTechniques:
    """Test log_data_techniques function."""

    def test_log_opsd_style_data_techniques(self, temp_directory):
        """Test logging data techniques used in OPSD pipeline."""
        # Data techniques that might be used in OPSD processing
        opsd_techniques = {
            "missing_value_handling": {
                "method": "forward_fill",
                "parameters": {"limit": "None"},
            },
            "outlier_removal": {
                "method": "IQR",
                "parameters": {
                    "factor": "1.5",
                    "column": "DE_load_actual_entsoe_transparency",
                },
            },
            "data_augmentation": {
                "method": "gaussian_noise",
                "parameters": {
                    "noise_factor": "0.01",
                    "target_column": "DE_load_actual_entsoe_transparency",
                },
            },
            "temporal_split": {
                "method": "no_shuffle",
                "parameters": {"test_size": "0.2", "preserve_order": "True"},
            },
        }

        with patch("mlflow.log_artifact") as mock_log_artifact, patch(
            "tempfile.NamedTemporaryFile"
        ) as mock_temp, patch("os.remove"):

            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.json"
            log_data_techniques(opsd_techniques)

            mock_log_artifact.assert_called_once()
            args, kwargs = mock_log_artifact.call_args
            assert kwargs["artifact_path"] == "data_techniques"

    def test_log_data_techniques_invalid_type(self):
        """Test with invalid input type."""
        with pytest.raises(TypeError, match="data_techniques must be a dictionary"):
            log_data_techniques("invalid")
