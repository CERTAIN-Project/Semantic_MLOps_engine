import json
from typing import Union, Dict

import mlflow


# Input: A dictionary of metrics to log
# metrics = {"accuracy": 0.95, "loss": 0.05, "precision": 0.9, "recall": 0.85}
def log_metrics(
    dict_of_metrics: Dict[str, float], step=0, keep_best: bool = False
) -> None:
    """
    Validate and log numeric metrics to MLflow.

    Parameters
    ----------
    dict_of_metrics : dict[str, float]
        Mapping of metric names to numeric values (or values convertible to float).
    step : int, optional
        Step or epoch associated with the metrics. When keep_best is False, this value
        is passed to mlflow.log_metric for each metric. Default is 0.
    keep_best : bool, default=False
        If True, metrics are logged with the prefix "best_" and the step is not used.
        If False, metrics are logged with their original names and the provided step.

    Notes
    -----
    - All metric values are converted to float and must be finite (not NaN or +/-inf).
    - This function requires an active MLflow run; it calls mlflow.log_metric.
    - Raises ValueError if any metric cannot be converted to a finite float.

    Returns
    -------
    None
    """
    import math  # Add this import at the top of your file

    # Validate all metrics before logging
    for key, value in dict_of_metrics.items():
        try:
            float_value = float(value)
            if not math.isfinite(float_value):
                raise ValueError(f"Value for metric '{key}' is not finite: {value}")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Value for metric '{key}' cannot be converted to float: {value}"
            ) from exc

    if not keep_best:
        for key, value in dict_of_metrics.items():
            mlflow.log_metric(key, float(value), step=step)
    else:
        for key, value in dict_of_metrics.items():
            mlflow.log_metric(f"best_{key}", float(value))


# search_space = {
#     "n_estimators": {"type": "int", "low": 10, "high": 100},
#     "max_depth": {"type": "int", "low": 3, "high": 10},
#     "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
#     "subsample": {"type": "float", "low": 0.5, "high": 1.0}
# }
def log_search_space(
    search_space: Dict[str, Dict[str, Union[str, float, int, bool, list]]],
) -> None:
    """
    Log the search space for hyperparameters to MLflow.

    This function logs each hyperparameter and its properties (type, low, high, etc.)
    to MLflow as parameters.

    Parameters
    ----------
    search_space : dict[str, dict[str, str | float | int | bool | list]]
        A dictionary where keys are hyperparameter names and values are dictionaries
        containing properties of the hyperparameters.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    ValueError
        If the search space structure is invalid or contains incorrect values.
    """
    if not search_space:
        raise ValueError("Search space dictionary cannot be empty")

    for param_name, param_config in search_space.items():
        if not isinstance(param_config, dict):
            raise ValueError(
                f"Configuration for parameter '{param_name}' must be a dictionary"
            )

        if "type" not in param_config:
            raise ValueError(f"Parameter '{param_name}' is missing required 'type' key")

        param_type = param_config["type"]

        if param_type in ["int", "float"]:
            # Numeric parameter validations
            if "low" not in param_config:
                raise ValueError(
                    f"Numeric parameter '{param_name}' is missing required 'low' key"
                )
            if "high" not in param_config:
                raise ValueError(
                    f"Numeric parameter '{param_name}' is missing required 'high' key"
                )

            low = param_config["low"]
            high = param_config["high"]

            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError(
                    f"Parameter '{param_name}': 'low' and 'high' must be numeric values"
                )

            if low >= high:
                raise ValueError(
                    f"Parameter '{param_name}': 'low' value must be less than 'high' value"
                )

            if "log" in param_config and not isinstance(param_config["log"], bool):
                raise ValueError(
                    f"Parameter '{param_name}': 'log' must be a boolean value"
                )

        elif param_type == "categorical":
            if "choices" not in param_config:
                raise ValueError(
                    f"Categorical parameter '{param_name}' is missing required 'choices' key"
                )

            if (
                not isinstance(param_config["choices"], list)
                or not param_config["choices"]
            ):
                raise ValueError(
                    f"Parameter '{param_name}': 'choices' must be a non-empty list"
                )

        else:
            raise ValueError(
                f"Parameter '{param_name}': Unsupported parameter type '{param_type}'"
            )

    mlflow.log_param("optuna_search_space", json.dumps(search_space))
