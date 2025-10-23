import pytest
import os
from unittest.mock import patch, Mock
from codecarbon import EmissionsTracker

from certain_library.resource_monitor.resource import start_tracker, stop_tracker


class TestResourceMonitoringOPSDPatterns:
    """Test resource monitoring following OPSD three-phase structure."""

    def test_opsd_data_processing_emissions_tracking(
        self, temp_directory, opsd_emissions_files
    ):
        """Test emissions tracking during OPSD data processing phase."""
        with patch(
            "certain_library.resource_monitor.resource.EmissionsTracker"
        ) as mock_tracker_class, patch("mlflow.log_artifact") as mock_log_artifact:

            # Create mock tracker instance
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            mock_tracker.stop.return_value = 0.001  # Mock emissions return value

            # Start tracking for data processing phase (like OPSD) - use REAL function
            tracker, output_location = start_tracker(
                save_to_file=True,
                measure_power_secs=1,
                output_dir=temp_directory,
                output_file_name="emissions_data",
            )

            # Create emissions file matching OPSD structure
            emissions_file = os.path.join(temp_directory, "emissions_data.csv")
            with open(emissions_file, "w") as f:
                f.write("timestamp,duration,emissions_kg,energy_consumed_kwh\n")
                f.write("2023-01-01 00:00:00,1.0,0.001,0.5\n")
                f.write("2023-01-01 00:01:00,1.0,0.0015,0.6\n")

            # Stop tracking - use REAL function
            stop_tracker(tracker, output_location)

            # Verify the tracker was used correctly
            assert tracker == mock_tracker
            mock_tracker.start.assert_called_once()
            mock_tracker.stop.assert_called_once()
            mock_log_artifact.assert_called_once()

    def test_opsd_hyperparameter_optimization_emissions(self, temp_directory):
        """Test emissions tracking during OPSD hyperparameter optimization phase."""
        with patch("codecarbon.EmissionsTracker") as mock_tracker_class, patch(
            "mlflow.log_artifact"
        ):

            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker

            # Track emissions for hyperparameter optimization (Optuna trials)
            tracker, output_location = start_tracker(
                save_to_file=True,
                measure_power_secs=1,
                output_dir=temp_directory,
                output_file_name="emissions_hyperparams",
            )

            # Simulate Optuna optimization work (longer duration, higher emissions)
            emissions_file = os.path.join(temp_directory, "emissions_hyperparams.csv")
            with open(emissions_file, "w") as f:
                f.write("timestamp,duration,emissions_kg,energy_consumed_kwh\n")
                f.write(
                    "2023-01-01 01:00:00,20.0,0.05,15.0\n"
                )  # Longer duration for optimization
                f.write("2023-01-01 01:20:00,20.0,0.048,14.5\n")

            stop_tracker(tracker, output_location)

            assert output_location["output_file_name"] == "emissions_hyperparams.csv"

    def test_opsd_model_training_emissions(self, temp_directory):
        """Test emissions tracking during OPSD incremental model training phase."""
        with patch("codecarbon.EmissionsTracker") as mock_tracker_class, patch(
            "mlflow.log_artifact"
        ):

            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker

            # Track emissions for model training (incremental XGBoost)
            tracker, output_location = start_tracker(
                save_to_file=True,
                measure_power_secs=1,
                output_dir=temp_directory,
                output_file_name="emissions_train",
            )

            # Simulate incremental training work
            emissions_file = os.path.join(temp_directory, "emissions_train.csv")
            with open(emissions_file, "w") as f:
                f.write("timestamp,duration,emissions_kg,energy_consumed_kwh\n")
                f.write("2023-01-01 02:00:00,10.0,0.02,8.0\n")  # Training duration
                f.write("2023-01-01 02:10:00,10.0,0.019,7.8\n")

            stop_tracker(tracker, output_location)

            assert output_location["output_file_name"] == "emissions_train.csv"

    def test_opsd_complete_emissions_workflow(
        self, temp_directory, opsd_emissions_files
    ):
        """Test complete emissions tracking workflow for all OPSD phases."""
        with patch(
            "certain_library.resource_monitor.resource.EmissionsTracker"
        ) as mock_tracker_class, patch("mlflow.log_artifact") as mock_log_artifact:

            # Create ONE mock tracker that will be reused for ALL instances
            mock_tracker = Mock()
            mock_tracker.stop.return_value = 0.001  # Mock emissions return value

            # CRITICAL: Ensure the same mock instance is returned every time
            mock_tracker_class.return_value = mock_tracker

            # Test all three phases from OPSD
            for phase, filename in opsd_emissions_files.items():
                print(f"Processing phase: {phase}")  # Debug output

                # Start tracking for each phase
                tracker, output_location = start_tracker(
                    output_dir=temp_directory,
                    output_file_name=filename.replace(".csv", ""),
                )

                # Verify we got the expected mock tracker
                assert (
                    tracker is mock_tracker
                ), f"Expected same mock tracker, got different instance"

                # Create phase-specific emissions file
                emissions_file = os.path.join(temp_directory, filename)
                with open(emissions_file, "w") as f:
                    if phase == "data_processing":
                        f.write(
                            "timestamp,duration,emissions_kg\n2023-01-01 00:00:00,1.0,0.001\n"
                        )
                    elif phase == "hyperparameter_optimization":
                        f.write(
                            "timestamp,duration,emissions_kg\n2023-01-01 01:00:00,20.0,0.05\n"
                        )
                    elif phase == "model_training":
                        f.write(
                            "timestamp,duration,emissions_kg\n2023-01-01 02:00:00,10.0,0.02\n"
                        )

                # Stop tracking
                stop_tracker(tracker, output_location)

            # Debug output to see what happened
            print(f"EmissionsTracker created {mock_tracker_class.call_count} times")
            print(f"start() called {mock_tracker.start.call_count} times")
            print(f"stop() called {mock_tracker.stop.call_count} times")

            # Verify all phases tracked
            assert (
                mock_tracker_class.call_count == 3
            ), f"Expected 3 tracker creations, got {mock_tracker_class.call_count}"
            assert (
                mock_tracker.start.call_count == 3
            ), f"Expected 3 start calls, got {mock_tracker.start.call_count}"
            assert (
                mock_tracker.stop.call_count == 3
            ), f"Expected 3 stop calls, got {mock_tracker.stop.call_count}"
            assert (
                mock_log_artifact.call_count == 3
            ), f"Expected 3 artifact logs, got {mock_log_artifact.call_count}"

            # Verify all artifacts logged to code_carbon path
            for call in mock_log_artifact.call_args_list:
                args, kwargs = call
                assert kwargs["artifact_path"] == "code_carbon"


# class TestStartTracker:
#     """Test start_tracker function with OPSD-specific configurations."""

#     def test_start_tracker_opsd_config(self, temp_directory):
#         """Test tracker initialization with OPSD-like configuration."""
#         with patch("codecarbon.EmissionsTracker") as mock_tracker_class:

#             mock_tracker = Mock()
#             mock_tracker_class.return_value = mock_tracker

#             # Use OPSD-like configuration - call the REAL start_tracker function
#             tracker, output_location = start_tracker(
#                 save_to_file=True,
#                 measure_power_secs=1,  # Like OPSD
#                 output_dir=temp_directory,
#                 output_file_name="emissions_data",
#             )

#             # Verify tracker was created with correct configuration
#             mock_tracker_class.assert_called_once_with(
#                 save_to_file=True,
#                 measure_power_secs=1,
#                 output_dir=temp_directory,
#                 output_file="emissions_data.csv",
#             )

#             assert tracker == mock_tracker

#             # Fix: Check the actual return format of your implementation
#             if isinstance(output_location, dict):
#                 assert output_location["output_dir"] == temp_directory
#                 assert output_location["output_file_name"] == "emissions_data.csv"
#             else:
#                 # If your implementation returns a string path
#                 assert temp_directory in str(output_location)
#                 assert "emissions_data.csv" in str(output_location)

#             mock_tracker.start.assert_called_once()

#     def test_start_tracker_invalid_filename_chars(self):
#         """Test with invalid characters in filename."""
#         with pytest.raises(
#             ValueError, match="output_file_name contains invalid characters"
#         ):
#             start_tracker(output_file_name="emissions/data")

#     def test_start_tracker_creates_emissions_logs_dir(self, temp_directory):
#         """Test that emissions directory is created like OPSD emissions_logs."""
#         emissions_dir = os.path.join(temp_directory, "emissions_logs")

#         with patch("codecarbon.EmissionsTracker") as mock_tracker_class:
#             mock_tracker = Mock()
#             mock_tracker_class.return_value = mock_tracker

#             start_tracker(output_dir=emissions_dir, output_file_name="test")

#             # Directory should be created
#             assert os.path.exists(emissions_dir)


class TestStopTracker:
    """Test stop_tracker function with OPSD patterns."""

    def test_stop_tracker_opsd_cleanup(self, mock_emissions_tracker, temp_directory):
        """Test tracker stopping with OPSD-style file cleanup."""
        output_location = {
            "output_dir": temp_directory,
            "output_file_name": "emissions_data.csv",
        }

        # Create emissions file like OPSD generates
        emissions_file = os.path.join(temp_directory, "emissions_data.csv")
        with open(emissions_file, "w") as f:
            f.write("timestamp,duration,emissions_kg,energy_consumed_kwh\n")
            f.write("2023-01-01 00:00:00,1.0,0.001,0.5\n")

        with patch("mlflow.log_artifact") as mock_log_artifact:
            stop_tracker(mock_emissions_tracker, output_location)

            # Verify tracker stopped and file logged
            mock_emissions_tracker.stop.assert_called_once()
            mock_log_artifact.assert_called_once()

            # Verify file cleaned up (like OPSD removes files after logging)
            assert not os.path.exists(emissions_file)

    def test_stop_tracker_missing_output_location(self, mock_emissions_tracker):
        """Test with missing output location."""
        with pytest.raises(ValueError, match="Output location is not specified"):
            stop_tracker(mock_emissions_tracker, None)

    def test_stop_tracker_invalid_tracker_type(self):
        """Test with invalid tracker type."""
        with pytest.raises(
            TypeError, match="tracker must be an instance of EmissionsTracker"
        ):
            stop_tracker("invalid_tracker", {})  # type: ignore

    def test_stop_tracker_file_not_exists(self, mock_emissions_tracker, temp_directory):
        """Test when emissions file doesn't exist (should not raise error)."""
        output_location = {
            "output_dir": temp_directory,
            "output_file_name": "nonexistent.csv",
        }

        with patch("mlflow.log_artifact") as mock_log_artifact:
            # Should not raise an error even if file doesn't exist
            stop_tracker(mock_emissions_tracker, output_location)

            mock_emissions_tracker.stop.assert_called_once()
            mock_log_artifact.assert_called_once()
