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
    # Prefer a compact experiments.json export placed in the artifacts root
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    candidates = [
        os.path.join(artifacts_root, "experiments.json"),
        os.path.join(artifacts_root, "metadata", "experiments.json"),
    ]

    for cand in candidates:
        try:
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Expecting a list of experiment dicts
                if isinstance(data, dict):
                    data = [data]
                df = pd.DataFrame(data)
                # Ensure expected columns exist
                for col in (
                    "experiment_id",
                    "name",
                    "lifecycle_stage",
                    "artifact_location",
                    "creation_time",
                    "last_update_time",
                ):
                    if col not in df.columns:
                        df[col] = None
                # Fill DB-required non-nullable defaults
                if "experiment_id" in df.columns:
                    df["experiment_id"] = df["experiment_id"].astype(str)
                df["name"] = df["name"].fillna("default")
                # experiments.lifecycle_stage is non-nullable in the DB; use schema default
                df["lifecycle_stage"] = df["lifecycle_stage"].fillna("data_processing")
                now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
                df["creation_time"] = df["creation_time"].fillna(now_ts)
                df["last_update_time"] = df["last_update_time"].fillna(
                    df["creation_time"].fillna(now_ts)
                )
                return df
        except Exception:
            # If the compact export is malformed or unreadable, fall back to other methods
            pass

    # Prefer reading certain/ai_actors/ai_actors.json when present. This file
    # can contain richer experiment / actor metadata produced by the certain
    # artifact exporter. Use it to populate non-nullable experiment fields.
    ai_actors_path = os.path.join(
        artifacts_root, "certain", "ai_actors", "ai_actors.json"
    )
    try:
        if os.path.exists(ai_actors_path):
            with open(ai_actors_path, "r", encoding="utf-8") as fh:
                actors_blob = json.load(fh)

            # actors_blob is expected to be a dict mapping experiment_id -> info
            experiments = []
            if isinstance(actors_blob, dict):
                for exp_id, info in actors_blob.items():
                    # info may be a dict with fields like 'experiment_name',
                    # 'lifecycle_stage', 'artifact_location', 'creation_time',
                    # 'last_update_time'
                    if not isinstance(info, dict):
                        continue
                    experiments.append(
                        {
                            "experiment_id": str(exp_id),
                            "name": info.get("experiment_name")
                            or info.get("name")
                            or f"exp_{exp_id}",
                            "lifecycle_stage": info.get("lifecycle_stage"),
                            "artifact_location": info.get("artifact_location"),
                            "creation_time": info.get("creation_time"),
                            "last_update_time": info.get("last_update_time"),
                            "description": info.get("description"),
                        }
                    )

            if experiments:
                df = pd.DataFrame(experiments)
                # Ensure schema-required columns exist and have safe defaults
                if "experiment_id" in df.columns:
                    df["experiment_id"] = df["experiment_id"].astype(str)
                df["name"] = df["name"].fillna("default")
                df["lifecycle_stage"] = df["lifecycle_stage"].fillna("active")
                now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
                df["creation_time"] = df["creation_time"].fillna(now_ts)
                df["last_update_time"] = df["last_update_time"].fillna(
                    df["creation_time"].fillna(now_ts)
                )
                return df
    except Exception:
        # Non-fatal: fall back to other experiment discovery methods
        pass

    # Fall back to constructing experiments from artifact directory listing
    experiments = []
    if os.path.isdir(artifacts_root):
        # Each top-level directory under artifacts_root is an experiment id
        for experiment_id in os.listdir(artifacts_root):
            exp_path = os.path.join(artifacts_root, experiment_id)
            if not os.path.isdir(exp_path):
                continue
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "name": None,
                    "lifecycle_stage": "data_processing",
                    "artifact_location": None,
                    "creation_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    "last_update_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
                }
            )

        if experiments:
            return pd.DataFrame(experiments)

    # Return Empty DataFrame if no experiments found
    return pd.DataFrame()


def get_experiment_tags_data():
    """Fetch experiment tags from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing experiment tags.
    """
    # Artifact store does not currently keep experiment-level tags in a
    # standard place; fallback to SQL.
    return pd.read_sql(tracking_models.SqlExperimentTag.__tablename__, mlflow_engine)


def get_datasets_data():
    """Fetch datasets table from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing datasets information.
    """
    return pd.read_sql(tracking_models.SqlDataset.__tablename__, mlflow_engine)


def get_runs_data():
    """Fetch runs from MLflow and populate parent_id from mlflow.parentRunId tags."""

    tags_df = get_tags_data()

    parent_id_map = {}

    if tags_df is not None and not tags_df.empty:
        # MLflow SQL uses run_id; artifact-normalized data may use run_uuid
        if "run_id" in tags_df.columns and "run_uuid" not in tags_df.columns:
            tags_df = tags_df.rename(columns={"run_id": "run_uuid"})

        for col in ("run_uuid", "key", "value"):
            if col not in tags_df.columns:
                tags_df[col] = None

        tags_df["run_uuid"] = tags_df["run_uuid"].astype(str)
        tags_df["key"] = tags_df["key"].astype(str)

        parent_tags = tags_df.loc[
            tags_df["key"] == "mlflow.parentRunId",
            ["run_uuid", "value"],
        ].dropna(subset=["run_uuid", "value"])

        parent_id_map = (
            parent_tags.drop_duplicates(subset=["run_uuid"])
            .set_index("run_uuid")["value"]
            .astype(str)
            .to_dict()
        )

    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    runs = []

    if os.path.isdir(artifacts_root):
        for experiment_id in os.listdir(artifacts_root):
            exp_path = os.path.join(artifacts_root, experiment_id)
            if not os.path.isdir(exp_path):
                continue

            for run_id in os.listdir(exp_path):
                run_path = os.path.join(exp_path, run_id)
                if not os.path.isdir(run_path):
                    continue

                metadata_path = os.path.join(
                    run_path, "artifacts", "metadata", "run_metadata.json"
                )

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        runs.append(data)
                    except Exception:
                        continue

    if runs:
        df = pd.DataFrame(runs)

        if "run_id" in df.columns and "run_uuid" not in df.columns:
            df = df.rename(columns={"run_id": "run_uuid"})

        if "run_name" in df.columns and "name" not in df.columns:
            df = df.rename(columns={"run_name": "name"})

        expected = [
            "run_uuid",
            "name",
            "source_type",
            "source_name",
            "user_id",
            "status",
            "start_time",
            "end_time",
            "source_version",
            "experiment_id",
            "parent_id",
        ]

        for col in expected:
            if col not in df.columns:
                df[col] = None

        df["run_uuid"] = df["run_uuid"].astype(str)

        df["parent_id"] = df["run_uuid"].map(parent_id_map)
        df["parent_id"] = df["parent_id"].where(pd.notna(df["parent_id"]), None)

        for col in ("start_time", "end_time"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].apply(lambda x: min(max(int(x), -(2**63)), 2**63 - 1))

        return df

    # Fall back to SQL-backed reads if no artifact-based runs found
    # runs_df = pd.read_sql(tracking_models.SqlRun.__tablename__, mlflow_engine)

    # # MLflow SQL uses run_id
    # if "run_id" in runs_df.columns and "run_uuid" not in runs_df.columns:
    #     runs_df = runs_df.rename(columns={"run_id": "run_uuid"})

    # if "parent_id" not in runs_df.columns:
    #     runs_df["parent_id"] = None

    # runs_df["run_uuid"] = runs_df["run_uuid"].astype(str)

    # runs_df["parent_id"] = runs_df["run_uuid"].map(parent_id_map)
    # runs_df["parent_id"] = runs_df["parent_id"].where(
    #     pd.notna(runs_df["parent_id"]), None
    # )

    # for col in ("start_time", "end_time"):
    #     if col not in runs_df.columns:
    #         runs_df[col] = 0

    #     runs_df[col] = pd.to_numeric(runs_df[col], errors="coerce").fillna(0)
    #     runs_df[col] = runs_df[col].apply(
    #         lambda x: min(max(int(x), -(2**63)), 2**63 - 1)
    #     )

    return pd.DataFrame()


def get_metrics_data():
    """Fetch metrics from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing metric records.
    """
    # Metrics are best reconstructed from events.jsonl if present
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    metrics = []
    if os.path.isdir(artifacts_root):
        for exp in os.listdir(artifacts_root):
            exp_path = os.path.join(artifacts_root, exp)
            if not os.path.isdir(exp_path):
                continue
            for run in os.listdir(exp_path):
                events_path = os.path.join(
                    exp_path, run, "artifacts", "metadata", "events.jsonl"
                )
                if not os.path.exists(events_path):
                    continue
                try:
                    with open(events_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            if ev.get("event_type") == "metric":
                                metrics.append(ev)
                except Exception:
                    continue

    if metrics:
        df = pd.DataFrame(metrics)

        # Standardize field names
        if "run_id" in df.columns:
            df = df.rename(columns={"run_id": "run_uuid"})

        if "is_NaN" in df.columns and "is_nan" not in df.columns:
            df = df.rename(columns={"is_NaN": "is_nan"})

        for col in ("run_uuid", "key", "value", "step", "timestamp", "is_nan"):
            if col not in df.columns:
                df[col] = None

        if "timestamp" in df.columns:
            try:
                df["timestamp"] = df["timestamp"].fillna(0).astype(int)
            except Exception:
                df["timestamp"] = (
                    pd.to_numeric(df["timestamp"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

        return df

    # return pd.read_sql(tracking_models.SqlMetric.__tablename__, mlflow_engine)
    return pd.DataFrame()


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
    # Reconstruct params from artifact events when available
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    params = []
    if os.path.isdir(artifacts_root):
        for exp in os.listdir(artifacts_root):
            exp_path = os.path.join(artifacts_root, exp)
            if not os.path.isdir(exp_path):
                continue
            for run in os.listdir(exp_path):
                events_path = os.path.join(
                    exp_path, run, "artifacts", "metadata", "events.jsonl"
                )
                if not os.path.exists(events_path):
                    continue
                try:
                    with open(events_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            if ev.get("event_type") == "param":
                                params.append(ev)
                except Exception:
                    continue

    if params:
        df = pd.DataFrame(params)

        if "run_id" in df.columns:
            df = df.rename(columns={"run_id": "run_uuid"})

        for col in ("run_uuid", "key", "value"):
            if col not in df.columns:
                df[col] = None

        return df

    # return pd.read_sql(tracking_models.SqlParam.__tablename__, mlflow_engine)
    return pd.DataFrame()


def get_tags_data():
    """Fetch run tags from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing run tags.
    """
    # Reconstruct tags from artifact events when available
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    tags = []
    if os.path.isdir(artifacts_root):
        for exp in os.listdir(artifacts_root):
            exp_path = os.path.join(artifacts_root, exp)
            if not os.path.isdir(exp_path):
                continue
            for run in os.listdir(exp_path):
                events_path = os.path.join(
                    exp_path, run, "artifacts", "metadata", "events.jsonl"
                )
                if not os.path.exists(events_path):
                    continue
                try:
                    with open(events_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            if ev.get("event_type") == "tag":
                                tags.append(ev)
                except Exception:
                    continue

    if tags:
        df = pd.DataFrame(tags)

        if "run_id" in df.columns:
            df = df.rename(columns={"run_id": "run_uuid"})

        for col in ("run_uuid", "key", "value"):
            if col not in df.columns:
                df[col] = None

        return df

    # return pd.read_sql(tracking_models.SqlTag.__tablename__, mlflow_engine)
    return pd.DataFrame()


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
                    print(
                        f"[WARNING] Skipping run {run_id}: "
                        f"artifact folder '{folder_name}' not found at {folder_path}"
                    )
                    continue
                if len(file_extension.split(".")[0]) != 0:
                    files_list = [file_extension]
                else:
                    files_list = os.listdir(folder_path)
                for file_name in files_list:
                    if file_name.endswith(file_extension):
                        file_path = os.path.join(folder_path, file_name)
                        if not os.path.exists(file_path):
                            print(
                                f"[WARNING] Skipping run {run_id}: "
                                f"artifact file '{file_name}' not found at {file_path}"
                            )
                            continue
                        data = None
                        if ".csv" in file_extension:
                            data = pd.read_csv(file_path)
                            data["run_id"] = run_id
                            data["experiment_id"] = experiment_id
                            data["stage"] = file_name.split("_")[-1].split(".")[0]
                            data_list.append(data)
                        elif ".json" in file_extension:
                            # Read JSON artifact into a DataFrame and attach run/experiment metadata
                            data = pd.read_json(file_path)
                            # Ensure we have a DataFrame (single object -> one-row DataFrame)
                            if isinstance(data, dict):
                                data = pd.DataFrame([data])
                            # Attach run/experiment identifiers so sync functions can map correctly
                            data["run_id"] = run_id
                            data["experiment_id"] = experiment_id
                            # Derive a stage from filename suffix similar to CSV handling
                            try:
                                data["stage"] = file_name.split("_")[-1].split(".")[0]
                            except Exception:
                                data["stage"] = "json"
                            data_list.append(data)
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

    return pd.DataFrame()


def get_json_artifacts_data(folder_name: str) -> list:
    """Collect JSON artifact records from MLflow local artifacts store.

    Walks the artifacts directory tree looking for ``folder_name`` subdirectories
    and reads every ``*.json`` file found there.  Each JSON file is expected to
    contain a single dict (the record written by the corresponding
    ``certain_library`` logging function).

    Parameters:
        folder_name (str): Subfolder under ``<experiment>/<run>/artifacts/`` to search.

    Returns:
        list[tuple[str, str, dict]]: A list of ``(run_id, experiment_id, record)``
        tuples for every JSON file discovered.  Returns an empty list when the
        folder does not exist for any run (no ``HTTPException`` is raised, unlike
        :func:`get_artifacts_data`).
    """
    unwanted = {".trash", ".DS_Store"}
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    results: list = []
    if not os.path.isdir(artifacts_path):
        return results

    for experiment_id in os.listdir(artifacts_path):
        if experiment_id in unwanted:
            continue
        experiment_path = os.path.join(artifacts_path, experiment_id)
        if not os.path.isdir(experiment_path):
            continue
        for run_id in os.listdir(experiment_path):
            if run_id in unwanted:
                continue
            folder_path = os.path.join(
                experiment_path, run_id, "artifacts", folder_name
            )
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if not file_name.endswith(".json"):
                    continue
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        record = json.load(fh)
                    if isinstance(record, dict):
                        results.append((run_id, experiment_id, record))
                    elif isinstance(record, list):
                        for item in record:
                            if isinstance(item, dict):
                                results.append((run_id, experiment_id, item))
                except Exception as exc:
                    print(
                        f"[get_json_artifacts_data] Could not read {file_path}: {exc}"
                    )
    return results


def get_dataset_manifest_for_run(run_id: str):
    """Return parsed data_manifest.json for a given run_id when present in local artifacts.

    Searches the local artifacts root for any experiment folder containing
    the run_id and reads artifacts/certain/dataset/data_manifest.json if present.
    Returns dict or None when not found.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    # If the artifacts path isn't present locally, we'll still attempt an MLflow
    # client download below; only search the local tree when it's available.

    for exp in os.listdir(artifacts_path):
        exp_path = os.path.join(artifacts_path, exp)
        if not os.path.isdir(exp_path):
            continue
        run_path = os.path.join(exp_path, run_id)
        if not os.path.isdir(run_path):
            continue
        manifest_path = os.path.join(
            run_path, "artifacts", "certain", "dataset", "data_manifest.json"
        )
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None

        # Also check the metadata location for a lightweight data.json
        meta_path = os.path.join(
            run_path, "artifacts", "certain", "metadata", "data.json"
        )
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None

    # If not found on local filesystem, try MLflow client to download artifacts
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        # Try to download the artifact to a temp dir
        import tempfile as _tmp

        with _tmp.TemporaryDirectory() as td:
            try:
                # Mlflow client download_artifacts may accept a run_id and a
                # path relative to the run's artifact root.
                rel_path = "certain/dataset/data_manifest.json"
                local = client.download_artifacts(run_id, rel_path, dst_path=td)
                if os.path.exists(local):
                    with open(local, "r", encoding="utf-8") as fh:
                        return json.load(fh)
            except Exception:
                # try metadata path
                try:
                    rel_meta = "certain/metadata/data.json"
                    local2 = client.download_artifacts(run_id, rel_meta, dst_path=td)
                    if os.path.exists(local2):
                        with open(local2, "r", encoding="utf-8") as fh:
                            return json.load(fh)
                except Exception:
                    return None
            except Exception:
                # Can't download via client — give up
                return None
    except Exception:
        return None
