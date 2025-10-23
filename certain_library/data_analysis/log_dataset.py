import os
import mlflow

import pandas as pd

from mlflow.data.pandas_dataset import from_pandas


def log_dataset(
    data: pd.DataFrame,
    name: str = "dataset",
    output_dir: str = "dataset",
    non_nan: bool = False,
) -> None:
    """
    Log a dataset to MLflow.

    This function converts a pandas DataFrame to an MLflow dataset, logs it as an input
    with the appropriate context, and also saves and logs it as a CSV artifact.
    A temporary CSV file is created and then removed after logging.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to log.
    name : str, optional
        Name identifier to use in the dataset. Default is "dataset".
    output_dir : str, optional
        Directory where the temporary CSV file will be saved before logging to MLflow.
        Default is "dataset".
    non_nan : bool, optional
        If True, raises ValueError if the DataFrame contains any NaN values. Default is False.

    Returns
    -------
    None
        This function doesn't return anything.

    Notes
    -----
    The function requires an active MLflow run context.
    The CSV file is temporarily saved to disk and then removed after being logged to MLflow.

    Raises
    ------
    ValueError
        If the input data is empty or contains NaN values (when non_nan=True).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check for NaN values
    if non_nan and data.isna().any().any():
        raise ValueError("Input DataFrame contains NaN values")

    if data.empty:
        raise ValueError("Input DataFrame is empty")

    dataset = from_pandas(data, source="logged dataset", name=name)
    mlflow.log_input(dataset, context="data_analysis")

    csv_path = os.path.join(output_dir, f"{name}.csv")
    data.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path=output_dir)

    # Remove the local file after logging
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # remove the output directory if it's empty
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)


def log_train_test_dataset(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str = "dataset",
    non_nan: bool = False,
) -> None:
    """
    Log training and testing datasets to MLflow.

    This function converts pandas DataFrames to MLflow datasets, logs them as inputs
    with appropriate contexts, and also saves and logs them as CSV artifacts.
    Temporary CSV files are created and then removed after logging.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset to log.
    test_data : pd.DataFrame
        The testing dataset to log.
    output_dir : str, optional
        Directory where temporary CSV files will be saved before logging to MLflow.
        Default is "dataset".
    non_nan : bool, optional
        If True, raises ValueError if train_data or test_data contains any NaN values.
        Default is False.

    Returns
    -------
    None
        This function doesn't return anything.

    Notes
    -----
    The function requires an active MLflow run context.
    CSV files are temporarily saved to disk and then removed after being logged to MLflow.

    Raises
    ------
    ValueError
        If the input data is empty, contains NaN values (when non_nan=True), or has invalid content.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check for NaN values
    if non_nan and train_data.isna().any().any():
        raise ValueError("Training data contains NaN values")
    if non_nan and test_data.isna().any().any():
        raise ValueError("Test data contains NaN values")

    # Ensure columns match between train and test data
    if not train_data.empty and not test_data.empty:
        if not set(train_data.columns) == set(test_data.columns):
            raise ValueError("Training and test data have different column sets")

    # train dataset
    if not train_data.empty:
        dataset = from_pandas(train_data, source="X_train split", name="X_train")
        mlflow.log_input(dataset, context="training")

        train_csv_path = os.path.join(output_dir, "X_train.csv")
        train_data.to_csv(train_csv_path, index=False)
        mlflow.log_artifact(train_csv_path, artifact_path="dataset")

    # test dataset
    if not test_data.empty:
        dataset_test = from_pandas(test_data, source="X_test split", name="X_test")
        mlflow.log_input(dataset_test, context="testing")

        test_csv_path = os.path.join(output_dir, "X_test.csv")
        test_data.to_csv(test_csv_path, index=False)
        mlflow.log_artifact(test_csv_path, artifact_path="dataset")

    # Remove the local files after logging
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    # remove the output directory if it's empty
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)
