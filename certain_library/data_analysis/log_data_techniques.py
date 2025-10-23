import os
import json
import tempfile
from typing import Dict
import mlflow


def log_data_techniques(data_techniques: Dict[str, Dict[str, str]]) -> None:
    """
    Log data techniques to MLflow artifacts.

    This function saves the provided data techniques dictionary as a JSON file
    and logs it as an artifact under the 'data_techniques' folder in MLflow.

    Args:
        data_techniques: Dictionary containing data techniques information.
                         Format is {technique_name: {property: value}}.

    Returns:
        None

    Raises:
        TypeError: If data_techniques is not a dictionary or if any value is not a dictionary.
    """
    # Check if data_techniques is a dictionary
    if not isinstance(data_techniques, dict):
        raise TypeError("data_techniques must be a dictionary")

    # Check if all values in data_techniques are dictionaries
    if not all(isinstance(value, dict) for value in data_techniques.values()):
        raise TypeError("All values in data_techniques must be dictionaries")

    # Create a temporary file to store the JSON data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data_techniques, f, indent=2)
        temp_file_name = f.name

    # Log the JSON file as an artifact
    mlflow.log_artifact(temp_file_name, artifact_path="data_techniques")

    # Clean up the temporary file
    os.remove(temp_file_name)
