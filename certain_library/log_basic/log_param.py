from typing import Any

from certain_library.tracking.tracker import tracker


# Log parameters to MLflow
def log_param(param_name: str, param_value: Any):
    """
    Log a single parameter to MLflow.

    Args:
        param_name (str): The name of the parameter.
        param_value (Any): The value of the parameter.
    """
    tracker.log_param(param_name, param_value)
