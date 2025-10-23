"""
This module provides functionality to interact with MLflow and a target database.
It includes methods to fetch data from MLflow's tracking database, retrieve artifacts,
and process data signatures from a target database.

Environment variables required:
- MLFLOW_DB: Connection string for the MLflow database.
- TARGET_DB: Connection string for the target database.
- MLFLOW_ARTIFACTS: URI for the MLflow artifacts storage.
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import urlparse
from fastapi import HTTPException
import yaml

from mlflow.store.tracking.dbmodels import models as tracking_models

# from mlflow.store.model_registry.dbmodels import models as model_registry_models

from data_api.app import models as certain_db_models


load_dotenv()
MLFLOW_DB = os.getenv("MLFLOW_DB")
if MLFLOW_DB is None:
    raise ValueError("MLFLOW_DB is not set in the environment or .env file")
mlflow_engine = create_engine(MLFLOW_DB)

TARGET_DB = os.getenv("TARGET_DB")
if TARGET_DB is None:
    raise ValueError("TARGET_DB is not set in the environment or .env file")
target_engine = create_engine(TARGET_DB)

MLFLOW_ARTIFACTS = os.getenv("MLFLOW_ARTIFACTS")
if MLFLOW_ARTIFACTS is None:
    raise ValueError("MLFLOW_ARTIFACTS is not set in the environment or .env file")
mlflow_artifacts_uri = MLFLOW_ARTIFACTS


def get_experiments_data():
    """Fetch experiments table from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing MLflow experiments.
    """
    return pd.read_sql(tracking_models.SqlExperiment.__tablename__, mlflow_engine)


def get_experiment_tags_data():
    """Fetch experiment tags from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing experiment tags.
    """
    return pd.read_sql(tracking_models.SqlExperimentTag.__tablename__, mlflow_engine)


def get_datasets_data():
    """Fetch datasets table from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing datasets information.
    """
    return pd.read_sql(tracking_models.SqlDataset.__tablename__, mlflow_engine)


def get_runs_data():
    """Fetch runs from MLflow tracking database and clamp time fields to BIGINT range.

    The function reads the runs table and ensures the `start_time` and `end_time`
    values are within the signed 64-bit integer range to avoid overflow issues.

    Returns:
        pandas.DataFrame: DataFrame containing runs with adjusted time fields.
    """
    runs_df = pd.read_sql(tracking_models.SqlRun.__tablename__, mlflow_engine)
    # Ensure start_time and end_time are within BIGINT range
    runs_df["start_time"] = runs_df["start_time"].apply(
        lambda x: min(max(x, -(2**63)), 2**63 - 1)
    )
    runs_df["end_time"] = runs_df["end_time"].apply(
        lambda x: min(max(x, -(2**63)), 2**63 - 1)
    )
    return runs_df


def get_metrics_data():
    """Fetch metrics from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing metric records.
    """
    return pd.read_sql(tracking_models.SqlMetric.__tablename__, mlflow_engine)


def get_latest_metrics_data():
    """Fetch latest metrics from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing latest metric values per run/step.
    """
    return pd.read_sql(tracking_models.SqlLatestMetric.__tablename__, mlflow_engine)


def get_params_data():
    """Fetch parameters from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing parameter records.
    """
    return pd.read_sql(tracking_models.SqlParam.__tablename__, mlflow_engine)


def get_tags_data():
    """Fetch run tags from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing run tags.
    """
    return pd.read_sql(tracking_models.SqlTag.__tablename__, mlflow_engine)


def get_data_signature():
    """Read data signatures from target database and return a mapping.

    Reads the DataSignatures table from the target database and builds a dict
    mapping each data_id to an index->column_name mapping based on the signature.

    Returns:
        dict: { data_id: { index: column_name, ... }, ... }
    """
    data = pd.read_sql(certain_db_models.DataSignatures.__tablename__, target_engine)

    # create a dictionary with key the data_id and value a dict of values the index of each and value the name
    data_signatures = {}
    for _, row in data.iterrows():
        data_id = row["data_id"]

        data_temp = {}
        signature = row["signature"]
        for index in range(len(signature)):
            column_name = signature[index]["name"]
            data_temp[index] = column_name

        print(f"data signature size: {len(data_temp)}")

        data_signatures[data_id] = data_temp

    return data_signatures


def get_artifacts_data(folder_name: str = "whylogs", file_extension: str = ".csv"):
    """Collect artifacts from MLflow local artifacts store.

    Parameters:
        folder_name (str): Subfolder under run artifacts to search (default "whylogs").
        file_extension (str): File extension filter (e.g. ".csv", ".json", ".pkl", ".txt", "MLmodel").

    Returns:
        pandas.DataFrame or dict: Concatenated DataFrame of found artifacts (for tabular types)
        or a dict for pickled/model artifacts. Returns empty DataFrame if nothing found.

    Raises:
        fastapi.HTTPException: If expected local artifacts directory is missing.
    """
    # list of files and folders we don't want
    unwanted_files = [".trash", ".DS_Store"]
    parsed_uri = urlparse(mlflow_artifacts_uri)

    # Handle both URI format (file://) and plain path format
    if "file" in parsed_uri.scheme or not parsed_uri.scheme:
        # Use the appropriate path based on URI format
        artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

        # Experiment sub folder
        data_list = []
        data_dict = {}
        experiment_files_lists = os.listdir(artifacts_path)
        for experiment_file_name in experiment_files_lists:
            if experiment_file_name in unwanted_files:
                continue
            experiment_id = experiment_file_name
            experiment_path = os.path.join(artifacts_path, experiment_file_name)
            # Run sub folder
            run_files_lists = os.listdir(experiment_path)
            for run_file_name in run_files_lists:
                if run_file_name in unwanted_files:
                    continue
                run_id = run_file_name
                run_path = os.path.join(experiment_path, run_file_name)
                # Artifacts sub folder
                folder_path = os.path.join(run_path, f"artifacts/{folder_name}")
                if not os.path.exists(folder_path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"2: Local artifacts directory not found, file {folder_path} does not exist",
                    )
                if len(file_extension.split(".")[0]) != 0:
                    files_list = [file_extension]
                else:
                    files_list = os.listdir(folder_path)
                for file_name in files_list:
                    if file_name.endswith(file_extension):
                        file_path = os.path.join(folder_path, file_name)
                        data = None
                        if ".csv" in file_extension:
                            data = pd.read_csv(file_path)
                            data["run_id"] = run_id
                            data["experiment_id"] = experiment_id
                            data["stage"] = file_name.split("_")[-1].split(".")[0]
                            data_list.append(data)

                        elif ".json" in file_extension:
                            data = pd.read_json(file_path)
                        elif ".pkl" in file_extension:
                            # Read the pickle file and convert it to JSON format then load as DataFrame
                            data = pd.read_pickle(file_path)
                            # Check if the loaded object is a DataFrame
                            if isinstance(data, pd.DataFrame):
                                json_str = data.to_json(orient="records")
                                # Continue processing with the DataFrame
                            else:
                                # Assume it's a model with a get_params() method (e.g., RandomForestRegressor)
                                if hasattr(data, "get_params"):
                                    model_params = data.get_params()
                                    json_str = json.dumps(model_params)
                                else:
                                    raise TypeError(
                                        "The loaded object cannot be serialized to JSON using this approach."
                                    )

                            data_dict[run_id] = json_str
                        elif ".txt" in file_extension:
                            # Read the text file and convert it to JSON format then load as DataFrame
                            with open(file_path, "r", encoding="utf-8") as file:
                                content = file.read()

                            # Convert content to a list of timestamps
                            content = content.splitlines()
                            content = [line.strip() for line in content if line.strip()]

                            data = pd.DataFrame([{"timestamps": content}])
                            data["run_id"] = run_id
                            data["experiment_id"] = experiment_id
                            data_list.append(data)
                        elif "MLmodel" in file_extension:
                            # read txt file and convert it to JSON format then load as DataFrame
                            # Read MLmodel file from the artifact path
                            with open(file_path, "r", encoding="utf-8") as mlmodel_file:
                                mlmodel_content = mlmodel_file.read()

                            # Convert MLmodel content to a structured dictionary using yaml parser
                            try:
                                # PyYAML can handle the MLmodel format which is YAML-like
                                mlmodel_data = yaml.safe_load(mlmodel_content)

                                # Flatten the nested dictionary structure for DataFrame conversion
                                flat_mlmodel_data = {}

                                def flatten_dict(nested_dict, prefix=""):
                                    for key, value in nested_dict.items():
                                        new_key = f"{prefix}.{key}" if prefix else key
                                        if isinstance(value, dict):
                                            flatten_dict(value, new_key)
                                        else:
                                            flat_mlmodel_data[new_key] = value

                                flatten_dict(mlmodel_data)

                                # Convert the flattened dictionary to dataframe
                                mlmodel_df = pd.DataFrame([flat_mlmodel_data])
                            except Exception as e:
                                # Fallback if YAML parsing fails
                                print(f"Failed to parse MLmodel with YAML: {e}")

                                # Create a simple representation of the entire content
                                mlmodel_df = pd.DataFrame(
                                    [{"mlmodel_content": mlmodel_content}]
                                )
                            mlmodel_df["run_id"] = run_id
                            mlmodel_df["experiment_id"] = experiment_id
                            data_list.append(mlmodel_df)

        if data_list:
            return pd.concat(data_list, ignore_index=True)
        if data_dict:
            return data_dict
    # else:
    #     # REMOTE FILE — Redirect or download
    #     try:
    #         # Try to directly download (if public/readable)
    #         remote_response = requests.get(mlflow_artifacts_uri, timeout=10)
    #         remote_response.raise_for_status()
    #         return data = remote_response.content
    #     except Exception:
    #         # Fallback: just redirect
    #         return {"data": "Redirecting to remote artifacts", "status": 00000000000}

    return pd.DataFrame()
