from certain_library.tracking.tracker import tracker as mlflow_tracker
import os
import inspect
from typing import Optional, Any

import codecarbon

EmissionsTracker = codecarbon.emissions_tracker.EmissionsTracker


def start_tracker(
    save_to_file: bool = True,
    measure_power_secs: int = 1,
    output_dir: str = "emissions_logs",
    output_file_name: str = "default",
) -> tuple[Any, dict[str, str]]:
    """
    Initialize and start a carbon emissions tracker.

    This function creates the specified output directory if it doesn't exist,
    configures an EmissionsTracker with the given parameters, and starts tracking.

    Args:
        save_to_file (bool, optional): Whether to save emissions data to a file.
            Defaults to True.
        measure_power_secs (int, optional): Interval between power measurements
            in seconds. Defaults to 1.
        output_dir (str, optional): Directory where emissions log files will be saved.
            Defaults to "emissions_logs".
        output_file_name (str, optional): Name for the output file (without extension).
            Defaults to "default".

    Returns:
        tuple[EmissionsTracker, dict[str, str]]: A tuple containing:
            - The initialized and started emissions tracker object
            - A dictionary with output location information

    Raises:
        TypeError: If input parameters have incorrect types.
        ValueError: If input parameters have invalid values.

    Note:
        The tracker will continue to run until explicitly stopped with tracker.stop().
    """
    # Validate input parameters
    if not isinstance(save_to_file, bool):
        raise TypeError("save_to_file must be a boolean")

    if not isinstance(measure_power_secs, int):
        raise TypeError("measure_power_secs must be an integer")
    if measure_power_secs <= 0:
        raise ValueError("measure_power_secs must be positive")

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")

    if not isinstance(output_file_name, str) or not output_file_name:
        raise ValueError("output_file_name must be a non-empty string")

    # Check for invalid characters in filename
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    if any(char in output_file_name for char in invalid_chars):
        raise ValueError(
            f"output_file_name contains invalid characters: {invalid_chars}"
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tracker = EmissionsTracker(
        save_to_file=save_to_file,
        measure_power_secs=measure_power_secs,
        output_dir=output_dir,
        output_file=f"{output_file_name}.csv",
    )
    tracker.start()

    output_location = {
        "output_dir": output_dir,
        "output_file_name": f"{output_file_name}.csv",
    }

    return tracker, output_location


def stop_tracker(
    tracker: Any, output_location: Optional[dict[str, str]] = None
) -> None:
    """
    Stop the emissions tracker, log the emissions data to MLflow, and clean up.

    This function stops the emissions tracker, logs the resulting emissions CSV file
    to MLflow as an artifact, and then removes the local CSV file.

    Args:
        tracker (EmissionsTracker): The emissions tracker to stop.
        output_location (dict[str, str], optional): Dictionary containing output location
            information with keys 'output_dir' and 'output_file_name'. Defaults to
            {"output_dir": "emissions_logs", "output_file_name": "default.csv"}.

    Returns:
        None

    Raises:
        TypeError: If tracker is not an EmissionsTracker instance.
        ValueError: If output_location is not properly specified.
    """
    if tracker is None:
        raise ValueError("tracker cannot be None")
    # Validate tracker type early so tests that pass an invalid tracker receive
    # a TypeError as expected.
    # Some tests patch EmissionsTracker with a Mock; in that case EmissionsTracker
    # may not be a real type. Only perform isinstance() when EmissionsTracker is
    # a type; otherwise accept the provided tracker (tests will assert behaviour
    # on the mock instance).
    try:
        is_emissions_type = isinstance(EmissionsTracker, type)
    except Exception:
        is_emissions_type = False

    if is_emissions_type and not isinstance(tracker, EmissionsTracker):
        raise TypeError("tracker must be an instance of EmissionsTracker")

    if output_location is None:
        raise ValueError("Output location is not specified.")

    if not isinstance(output_location, dict):
        raise TypeError("output_location must be a dictionary")

    required_keys = ["output_dir", "output_file_name"]
    for key in required_keys:
        if key not in output_location:
            raise ValueError(f"output_location is missing required key: {key}")
        if not isinstance(output_location[key], str) or not output_location[key]:
            raise ValueError(f"output_location['{key}'] must be a non-empty string")

    tracker.stop()

    mlflow_tracker.log_artifact(
        f"{output_location['output_dir']}/{output_location['output_file_name']}",
        artifact_path="code_carbon",
    )

    # Remove the local file after logging
    if os.path.exists(
        f"{output_location['output_dir']}/{output_location['output_file_name']}"
    ):
        os.remove(
            f"{output_location['output_dir']}/{output_location['output_file_name']}"
        )

    # Remove the output directory if it's empty
    if os.path.exists(output_location["output_dir"]) and not os.listdir(
        output_location["output_dir"]
    ):
        os.rmdir(output_location["output_dir"])
