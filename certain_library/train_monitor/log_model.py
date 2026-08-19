from certain_library.tracking.tracker import tracker
import pandas as pd
from typing import Any, Dict, Union

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

    tracker.log_params(model_information)


def log_model_architecture(
    losses: list,
    optimizer: Union[str, Dict[str, Any]],
    regularization: str,
    early_stopping: bool,
) -> None:
    """
    Log model architecture to MLflow.

    This function logs the model parameters to MLflow.
    It is useful for tracking the parameters of the model used in training.

    Parameters
    ----------
    losses : list
        List of loss functions used in the model.
    optimizer : str or dict
        The optimizer used during training. Accepts two formats:

        * **Simple string** — just the optimizer name::

              optimizer="Adam"

        * **Dict** — name plus any optimizer-specific hyperparameters::

              optimizer={"name": "Adam", "lr": 0.001, "weight_decay": 1e-4}
              optimizer={"name": "SGD", "momentum": 0.9, "nesterov": True}
              optimizer={"name": "gbdt"}   # XGBoost / LightGBM style

        The ``"name"`` key is required when a dict is provided.
        All additional keys are logged individually as
        ``optimizer.<key>`` MLflow parameters, making them queryable.
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
        If any input parameter is of incorrect type, has invalid values, or
        the optimizer dict is missing the ``"name"`` key.
    """
    # Validate input types
    if not isinstance(losses, list):
        raise ValueError("losses must be a list")
    if not isinstance(optimizer, (str, dict)):
        raise ValueError("optimizer must be a string or a dict")
    if not isinstance(regularization, str):
        raise ValueError("regularization must be a string")
    if not isinstance(early_stopping, bool):
        raise ValueError("early_stopping must be a boolean")

    # Validate non-empty values
    if not losses:
        raise ValueError("losses list cannot be empty")
    if not regularization:
        raise ValueError("regularization cannot be empty")

    # Normalise optimizer → always work with a dict internally
    if isinstance(optimizer, str):
        if not optimizer:
            raise ValueError("optimizer cannot be empty")
        optimizer_dict: Dict[str, Any] = {"name": optimizer}
    else:
        if "name" not in optimizer:
            raise ValueError("optimizer dict must contain a 'name' key")
        if not optimizer["name"]:
            raise ValueError("optimizer 'name' cannot be empty")
        optimizer_dict = optimizer

    tracker.log_params({"losses": losses})
    # Log the optimizer name as the primary "optimizer" param for backwards compat
    tracker.log_params({"optimizer": optimizer_dict["name"]})
    # Log every additional optimizer hyperparameter under "optimizer.<key>"
    for key, value in optimizer_dict.items():
        if key != "name":
            tracker.log_params({f"optimizer.{key}": value})
    tracker.log_params({"regularization": regularization})
    tracker.log_params({"early_stopping": early_stopping})


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
        tracker.log_params(dict_of_hyperparameters)
    else:
        best_hyperparameters = {
            f"best_{key}": value for key, value in dict_of_hyperparameters.items()
        }
        tracker.log_params(best_hyperparameters)


# model should be generic since the type can vary based on the framework used
from typing import Optional


def log_model_signature(
    model, train_data: pd.DataFrame, y_train: Optional[pd.Series] = None
) -> None:
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
    tracker.log_xgboost_model(
        model,
        artifact_path="model",  # This creates a separate "model" folder
        input_example=input_example,
        signature=signature,
    )

    # Log the model signature as a parameter for tracking
    tracker.log_params({"model_signature": str(signature)})
    tracker.log_params({"input_shape": str(train_data.shape)})
    tracker.log_params({"n_features": len(train_data.columns)})

    # Try to compute parameter counts and number of layers for common model types
    # so this information is available as MLflow params and can be synced into
    # the `model_architecture` table by the data API.
    num_layers = None
    num_total_parameters = None
    num_trainable_parameters = None
    num_non_trainable_parameters = None

    try:
        # XGBoost (scikit-learn wrapper)
        if hasattr(model, "get_booster"):
            try:
                booster = model.get_booster()
                # number of trees
                try:
                    dumps = booster.get_dump()
                    num_trees = len(dumps)
                except Exception:
                    num_trees = None

                # attempt to count leaf nodes via trees_to_dataframe when available
                try:
                    df_trees = booster.trees_to_dataframe()
                    num_leaves = int((df_trees["Feature"] == "Leaf").sum())
                except Exception:
                    try:
                        dump_json = booster.get_dump(dump_format="json")
                        # count occurrences of the JSON leaf key
                        num_leaves = sum(s.count('"leaf"') for s in dump_json)
                    except Exception:
                        num_leaves = None

                if num_leaves is not None:
                    num_total_parameters = int(num_leaves)
                    num_trainable_parameters = int(num_leaves)
                    num_non_trainable_parameters = 0
                if num_trees is not None:
                    num_layers = int(num_trees)
            except Exception:
                pass

        # Keras / TensorFlow models
        if num_total_parameters is None:
            try:
                import tensorflow as _tf

                from tensorflow.keras import Model as KerasModel

                if isinstance(model, KerasModel):
                    try:
                        total = model.count_params()
                        # trainable params via summing trainable weights
                        trainable = 0
                        for w in getattr(model, "trainable_weights", []):
                            try:
                                trainable += int(_tf.keras.backend.count_params(w))
                            except Exception:
                                continue
                        non_trainable = int(total - trainable)
                        num_total_parameters = int(total)
                        num_trainable_parameters = int(trainable)
                        num_non_trainable_parameters = int(non_trainable)
                        num_layers = len(getattr(model, "layers", []))
                    except Exception:
                        pass
            except Exception:
                pass

        # PyTorch models
        if num_total_parameters is None:
            try:
                import torch

                if isinstance(model, torch.nn.Module):
                    total = sum(p.numel() for p in model.parameters())
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    non_trainable = int(total - trainable)
                    num_total_parameters = int(total)
                    num_trainable_parameters = int(trainable)
                    num_non_trainable_parameters = int(non_trainable)
            except Exception:
                pass
    except Exception:
        # Keep tolerant: do not fail model logging if counting fails
        pass

    # Record any discovered counts as MLflow params so they are discoverable
    try:
        if num_layers is not None:
            tracker.log_params({"number_of_layers": int(num_layers)})
        if num_total_parameters is not None:
            tracker.log_params({"number_of_total_parameters": int(num_total_parameters)})
        if num_trainable_parameters is not None:
            tracker.log_params({"number_of_trainable_parameters": int(num_trainable_parameters)})
        if num_non_trainable_parameters is not None:
            tracker.log_params({"number_of_non_trainable_parameters": int(num_non_trainable_parameters)})
    except Exception:
        # ignore any errors while logging these optional params
        pass
