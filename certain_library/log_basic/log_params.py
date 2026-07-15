from typing import Any

from certain_library.tracking.tracker import tracker


# Log parameters to MLflow
def log_params(params: dict[str, Any]) -> None:
    """
    Log multiple parameters to MLflow.

    Args:
        params (dict[str, Any]): A dictionary of parameter names and values.
    """
    tracker.log_params(params)
