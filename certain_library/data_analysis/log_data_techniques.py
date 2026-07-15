from certain_library.tracking.tracker import tracker
import os
import json
import tempfile
from typing import Dict


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
    # Accept either the old flat mapping {technique: props} or the new
    # wrapped form {"techniques": { ... }, "data_technique_stage": ...}.
    if not isinstance(data_techniques, dict):
        raise TypeError("data_techniques must be a dictionary")

    # Normalize to canonical form: {"techniques": {...}, "data_technique_stage": ...}
    if "techniques" in data_techniques and isinstance(
        data_techniques.get("techniques"), dict
    ):
        canonical = dict(data_techniques)
    else:
        # Validate that all values are dict-like for the old shape
        if not all(isinstance(v, dict) for v in data_techniques.values()):
            raise TypeError("All values in data_techniques must be dictionaries")
        canonical = {"techniques": dict(data_techniques)}

    # Create a temporary directory and write a deterministically named JSON file
    temp_dir = tempfile.mkdtemp()
    temp_file_name = os.path.join(temp_dir, "data_techniques.json")
    try:
        with open(temp_file_name, "w") as f:
            json.dump(canonical, f, indent=2)

        # Log the JSON file as an artifact. MLflow will preserve the filename
        # when uploading into the `data_techniques` artifact folder.
        tracker.log_artifact(temp_file_name, artifact_path="data_techniques")
    finally:
        # Clean up the temporary file and directory if they exist
        try:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
        except Exception:
            pass
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
