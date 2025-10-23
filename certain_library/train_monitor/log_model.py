import pandas as pd
from typing import Dict, Union

import mlflow
from mlflow.models import infer_signature


# E.g Save model name, version, and other metadata
def log_model_info(model_information: Dict[str, str]) -> None:
    """
    Log model parameters to MLflow.

    This function logs the model parameters to MLflow.
    It is useful for tracking the parameters of the model used in training.

    Parameters
    ----------
    model_information : dict[str, str]
        A dictionary where keys are parameter names and values are parameter values.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    ValueError
        If the input dictionary is empty or contains non-string values.
    """
    # Check if dictionary is empty
    if not model_information:
        raise ValueError("Input model_information dictionary cannot be empty")

    # Check if values are strings
    for key, value in model_information.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for key '{key}' must be a string")

    for key, value in model_information.items():
        mlflow.log_param(key, value)


def log_model_architecture(
    losses: list, optimizer: str, regularization: str, early_stopping: bool
) -> None:
    """
    Log model architecture to MLflow.

    This function logs the model parameters to MLflow.
    It is useful for tracking the parameters of the model used in training.

    Parameters
    ----------
    losses : list
        List of loss functions used in the model.
    optimizer : str
        Name of the optimizer used.
    regularization : str
        Type of regularization applied.
    early_stopping : bool
        Whether early stopping was used.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    ValueError
        If any input parameter is of incorrect type or has invalid values.
    """
    # Validate input types
    if not isinstance(losses, list):
        raise ValueError("losses must be a list")
    if not isinstance(optimizer, str):
        raise ValueError("optimizer must be a string")
    if not isinstance(regularization, str):
        raise ValueError("regularization must be a string")
    if not isinstance(early_stopping, bool):
        raise ValueError("early_stopping must be a boolean")

    # Validate non-empty values
    if not losses:
        raise ValueError("losses list cannot be empty")
    if not optimizer:
        raise ValueError("optimizer cannot be empty")
    if not regularization:
        raise ValueError("regularization cannot be empty")

    mlflow.log_param("losses", losses)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("regularization", regularization)
    mlflow.log_param("early_stopping", early_stopping)


# Input: A dictionary of hyperparameters to log
# hyperparameters = {"n_estimator": 10, "learning_rate": 0.05, "max_depth": 30}
def log_model_hyperparameters(
    dict_of_hyperparameters: Dict[str, Union[float, int, str]], keep_best: bool = False
) -> None:
    """
    Log hyperparameters to MLflow.

    This function logs each key-value pair in the provided dictionary to MLflow as parameters.

    Parameters
    ----------
    dict_of_hyperparameters : dict[str, Union[float, int, str]]
        A dictionary where keys are hyperparameter names and values are the hyperparameter values.
    keep_best : bool, default=False
        If True, hyperparameters will be logged with a "best_" prefix.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    ValueError
        If the input dictionary is empty or contains values that are not float, int, or str.
    """
    # Check if dictionary is empty
    if not dict_of_hyperparameters:
        raise ValueError("Input dictionary cannot be empty")

    # Check if values are of expected types
    for key, value in dict_of_hyperparameters.items():
        if not isinstance(value, (float, int, str)):
            raise ValueError(f"Value for key '{key}' is not of type float, int, or str")

    if not keep_best:
        for key, value in dict_of_hyperparameters.items():
            mlflow.log_param(key, value)
    else:
        for key, value in dict_of_hyperparameters.items():
            mlflow.log_param(f"best_{key}", value)


# model should be generic since the type can vary based on the framework used
def log_model_signature(model, train_data: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Log the model architecture to MLflow.

    This function logs the model architecture as a JSON string to MLflow.
    It is useful for tracking the structure of the model used in training.

    Parameters
    ----------
    model : object
        The trained model object.
    train_data : pd.DataFrame
        The training data used for the model.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    ValueError
        If the input data is empty, contains NaN values, or the model cannot make predictions on it.
    """
    import xgboost as xgb

    # Handle DMatrix objects by extracting the underlying data
    if isinstance(train_data, xgb.DMatrix):
        # For DMatrix, we need to reconstruct a DataFrame from the data
        # This is tricky since DMatrix doesn't preserve column names
        raise ValueError(
            "DMatrix objects are not supported. Please pass the original pandas DataFrame."
        )

    if not isinstance(train_data, pd.DataFrame):
        raise ValueError("train_data must be a pandas DataFrame")

    # Check if DataFrame is empty
    if train_data.empty:
        raise ValueError("Input train_data cannot be empty")

    # # Check for NaN values
    # if train_data.isna().any().any():
    #     raise ValueError("Input train_data contains NaN values")

    # Check if the model can make predictions
    try:
        predictions = model.predict(train_data)
    except Exception as e:
        raise ValueError(
            f"Model cannot make predictions on the provided data: {str(e)}"
        ) from e

    input_example = train_data.iloc[:5]  # a few example rows
    signature = infer_signature(train_data, predictions)

    # Log the model with input example and signature
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",  # This creates a separate "model" folder
        input_example=input_example,
        signature=signature,
    )

    # Log the model signature as a parameter for tracking
    mlflow.log_param("model_signature", str(signature))
    mlflow.log_param("input_shape", str(train_data.shape))
    mlflow.log_param("n_features", len(train_data.columns))
