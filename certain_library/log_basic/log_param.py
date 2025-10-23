import mlflow

from typing import Any


# Log parameters to MLflow
def log_param(param_name: str, param_value: Any):
    """
    Log a single parameter to MLflow.

    Args:
        param_name (str): The name of the parameter.
        param_value (Any): The value of the parameter.
    """
    mlflow.log_param(param_name, param_value)
