import json
import os
import logging
import pandas as pd
from scipy.stats import ks_2samp
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def map_mlflow_runs(run):
    return {
        "run_id": run.run_uuid,
        "run_name": run.name,
        "parent_id": run.get("parent_id", None),
        "source_type": run.source_type,
        "source_name": run.source_name,
        "user_id": run.user_id,
        "status": run.status,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "source_version": run.source_version,
        "experiment_id": run.experiment_id,
    }


def map_mlflow_experiments(experiments):
    return {
        "experiment_id": experiments.experiment_id,
        "experiment_name": experiments.name,
        "lifecycle_stage": experiments.lifecycle_stage,
        # Some DB schemas expect an explicit 'experiment_stage' (non-nullable).
        # Use any available attribute or fall back to lifecycle_stage or 'active'.
        "experiment_stage": getattr(
            experiments,
            "experiment_stage",
            getattr(experiments, "lifecycle_stage", "active"),
        ),
        # Include description if available (nullable in DB migrations).
        "description": getattr(experiments, "description", None),
        "creation_time": experiments.creation_time,
        "last_update_time": experiments.last_update_time,
    }


def map_mlflow_datasets(datasets, run_id, id_mapping):
    # Normalize incoming dataset record (could be a pandas Series)
    record = {}
    try:
        if isinstance(datasets, pd.Series):
            record = datasets.to_dict()
        elif isinstance(datasets, dict):
            record = dict(datasets)
        else:
            # fallback: treat as empty mapping
            record = {}
    except Exception:
        record = {}

    # Helper: try several likely keys for location and size
    def first_present(keys):
        for k in keys:
            v = record.get(k) if isinstance(record, dict) else None
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
        return None

    location_keys = [
        "location",
        "uri",
        "path",
        "artifact_uri",
        "dataset_uri",
        "source",
        "file_path",
        "absolute_path",
    ]
    size_keys = ["size", "file_size", "bytes", "length"]

    data_location = first_present(location_keys)
    data_size = first_present(size_keys)

    # If a candidate location was found, try to normalize and stat it
    resolved_location = None
    resolved_size = None
    try:
        if isinstance(data_location, str):
            loc = data_location
            # strip file:// prefix if present
            if loc.startswith("file://"):
                loc = urlparse(loc).path
            # If it's an absolute path on disk, and exists, stat it
            if os.path.isabs(loc) and os.path.exists(loc):
                resolved_location = loc
                try:
                    resolved_size = int(os.path.getsize(loc))
                except Exception:
                    resolved_size = None
            else:
                # It might be an artifact-path relative to MLflow artifacts root
                mlflow_artifacts = os.getenv("MLFLOW_ARTIFACTS")
                if mlflow_artifacts:
                    parsed = urlparse(mlflow_artifacts)
                    artifacts_root = parsed.path if parsed.path else mlflow_artifacts
                    # look for files under <artifacts_root>/**/<run_id>/artifacts/**
                    run_base = None
                    # search for a matching run folder
                    if os.path.isdir(artifacts_root):
                        for exp in os.listdir(artifacts_root):
                            run_path = os.path.join(artifacts_root, exp, str(run_id))
                            if os.path.isdir(run_path):
                                run_base = run_path
                                break

                    if run_base:
                        # Look for any file whose name or path contains the provided location fragment
                        for root, _, files in os.walk(run_base):
                            for fname in files:
                                candidate = os.path.join(root, fname)
                                if (
                                    loc in candidate
                                    or fname == loc
                                    or fname.startswith(loc)
                                ):
                                    resolved_location = candidate
                                    try:
                                        resolved_size = int(os.path.getsize(candidate))
                                    except Exception:
                                        resolved_size = None
                                    break
                            if resolved_location:
                                break

    except Exception:
        resolved_location = None
        resolved_size = None

    # If size candidate exists but not yet resolved, coerce numeric-like values
    if resolved_size is None and data_size is not None:
        try:
            resolved_size = int(data_size)
        except Exception:
            resolved_size = None

    # Prefer dataset manifest if present (written by save_dataset_manifest)
    try:
        from data_api.app.mlflow_connector import get_dataset_manifest_for_run

        manifest = get_dataset_manifest_for_run(str(run_id))
        if manifest and isinstance(manifest, dict):
            # Manifest may be the full data_manifest (with 'files' and 'total_size_bytes')
            # or a lightweight metadata/data.json (with 'data_location' and 'data_size').
            if "data_size" in manifest:
                try:
                    resolved_size = int(
                        manifest.get("data_size")
                        or manifest.get("total_size_bytes")
                        or 0
                    )
                except Exception:
                    resolved_size = resolved_size
            else:
                msize = manifest.get("total_size_bytes")
                if msize is not None:
                    try:
                        resolved_size = int(msize)
                    except Exception:
                        pass

            # Prefer explicit data_location if present
            if manifest.get("data_location"):
                resolved_location = manifest.get("data_location")
            else:
                files = manifest.get("files") or []
                if files:
                    first = files[0]
                    if isinstance(first, dict) and first.get("path"):
                        resolved_location = first.get("path")
    except Exception:
        # ignore failure to import or parse manifest
        manifest = None

    # Final fallbacks to keep previous behaviour but avoid using '/home' as a default
    final_location = resolved_location or (
        data_location if data_location is not None else ""
    )
    final_size = (
        resolved_size
        if resolved_size is not None
        else (
            int(data_size)
            if isinstance(data_size, (int, float)) and not pd.isna(data_size)
            else 0
        )
    )

    return {
        "run_id": run_id,
        "data_id": id_mapping[run_id]["data_id"],
        "data_stage": record.get("data_stage", "training"),
        "data_type": record.get("data_type", "int"),
        "data_source": record.get("source", "local"),
        "data_version": record.get("version", "v1"),
        "data_location": final_location,
        "data_size": final_size,
        "data_format": record.get("format", "csv"),
        "creation_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
        "last_update_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
    }


def map_mlflow_data_metrics(metrics, id_mapping):

    if isinstance(metrics, pd.Series):
        metrics = metrics.to_frame().T

    data = []
    for _, row in metrics.iterrows():
        run_id = row["run_id"]
        data_id = id_mapping[run_id]["data_id"]
        stage = row.get("stage", "train_default")
        column_name = row.get("column", "NaN")
        columns_to_skip = [
            "run_id",
            "data_id",
            "stage",
            "column_index",
            "experiment_id",
            "column",
        ]
        for col_name, value in row.items():
            if col_name in columns_to_skip:
                continue
            data.append(
                {
                    "run_id": run_id,
                    "data_id": data_id,
                    "key": f"[{column_name}]{col_name}",
                    "value": 0 if pd.isna(value) else value,
                    "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    "data_stage": stage,
                    "is_NaN": True if pd.isna(value) else False,
                }
            )
    return data


def map_mlflow_model_metrics(metrics, id_mapping):
    stage = "train"

    return {
        "run_id": metrics["run_uuid"],
        "model_id": id_mapping[metrics["run_uuid"]]["model_id"],
        "key": metrics["key"],
        "value": metrics["value"],
        "step": metrics["step"],
        "timestamp": metrics["timestamp"],
        "stage": stage,
        "is_NaN": metrics["is_nan"],
    }


def map_mlflow_model_params(params, id_mapping):
    return {
        "run_id": params["run_uuid"],
        "model_id": id_mapping[params["run_uuid"]]["model_id"],
        "key": params.get("key", "param_key"),
        "value": params.get("value", "param_value"),
    }


def map_mlflow_runs_tags(runs_tags):
    return {
        "run_id": runs_tags.get("run_uuid", 0),
        "key": runs_tags.get("key", "tag_key"),
        "value": runs_tags.get("value", "tag_value"),
    }


resources_key = [
    "duration",
    "emissions",
    "emissions_rate",
    "cpu_power",
    "gpu_power",
    "ram_power",
    "cpu_energy",
    "gpu_energy",
    "ram_energy",
    "energy_consumed",
    "cpu_count",
    "cpu_model",
    "gpu_count",
    "gpu_model",
    "ram_total_size",
    "pue",
]


def map_mlflow_data_resources(data_resources, id_mapping):
    data = []
    for key in resources_key:
        data.append(
            {
                "run_id": data_resources["run_id"],
                "data_id": id_mapping[data_resources["run_id"]]["data_id"],
                "stage": data_resources.get("stage", "data_default"),
                "key": key,
                "value": (
                    None if pd.isna(data_resources[key]) else data_resources[key]
                ),
                "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
            }
        )

    return data


def map_mlflow_resources(resources, id_mapping):
    data = []
    for key, value in resources.items():
        data.append(
            {
                "run_id": resources["run_id"],
                "model_id": id_mapping[resources["run_id"]]["model_id"],
                "key": key,
                "step": resources.get("step", 0),
                "stage": resources.get("stage", "train_default"),
                "value": None if pd.isna(value) else value,
                "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
            }
        )
    return data


def map_mlflow_time_series_data(time_series_data, id_mapping):
    # For each row, access content and compute the frequency of sampling
    # Process a single row instead of iterating through a DataFrame
    row = time_series_data
    monotonic_increase = False
    avg_sampling_rate = None
    missing_intervals = 0

    if "timestamps" in row and isinstance(row["timestamps"], list):
        timestamps = row["timestamps"]
        # Convert timestamps from string to datetime
        timestamps = pd.to_datetime(
            [ts for ts in timestamps if isinstance(ts, str)], errors="coerce"
        )
        timestamps = timestamps.dropna()

        # Check if timestamps are sorted
        if len(timestamps) > 0 and timestamps.is_monotonic_increasing:
            monotonic_increase = True

        if len(timestamps) > 1:
            # Compute time differences between consecutive timestamps
            time_diffs = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
                if timestamps[i + 1] > timestamps[i]
            ]
            if time_diffs:
                avg_sampling_rate = min(
                    time_diffs
                )  # Use minimum interval as the accurate frequency

            # Check for missing expected sampling intervals
            if avg_sampling_rate:
                for i in range(len(timestamps) - 1):
                    if (
                        timestamps[i + 1] - timestamps[i]
                    ).total_seconds() > avg_sampling_rate * 1.5:
                        missing_intervals += 1

    run_id = row.get("run_id")
    result = []

    if run_id in id_mapping:
        for key, value in {
            "monotonic_increase": monotonic_increase,
            "avg_sampling_rate": avg_sampling_rate,
            "missing_intervals": missing_intervals,
        }.items():
            result.append(
                {
                    "run_id": run_id,
                    "data_id": id_mapping[run_id]["data_id"],
                    "key": f"[time/series/data]{key}",
                    "value": value,
                    "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                }
            )

    return result


def map_mlflow_data_drift(dataset, id_mapping):
    """
    Calculate data drift between training and testing datasets using Kolmogorov-Smirnov test.

    Args:
        dataset: DataFrame containing both training and testing data with 'stage' column
        id_mapping: Dictionary mapping run_ids to data/model ids

    Returns:
        List of dictionaries containing drift analysis results
    """
    # Split the dataframe into two parts based on the 'stage' column
    training_data = dataset[dataset["stage"] == "train"]
    testing_data = dataset[dataset["stage"] == "test"]

    logger.debug(
        "map_mlflow_data_drift called: dataset shape=%s, training=%s, testing=%s",
        getattr(dataset, "shape", None),
        getattr(training_data, "shape", None),
        getattr(testing_data, "shape", None),
    )

    # Compute the drift between training and testing data
    drift_results = []
    # Group data by run_id
    run_ids = set(training_data["run_id"]).intersection(set(testing_data["run_id"]))

    logger.debug("run_ids with both train and test: %s", run_ids)

    for run_id in run_ids:
        # Filter data for this run_id
        train_subset = training_data[training_data["run_id"] == run_id]
        test_subset = testing_data[testing_data["run_id"] == run_id]

        # Get common columns between training and testing datasets
        common_columns = set(train_subset.columns).intersection(
            set(test_subset.columns)
        )
        common_columns = [
            col for col in common_columns if col not in ["run_id", "stage"]
        ]

        logger.debug("run_id=%s common_columns=%s", run_id, common_columns)

        # Prefer testing numeric columns only
        numeric_columns = [
            col
            for col in common_columns
            if pd.api.types.is_numeric_dtype(train_subset[col])
            and pd.api.types.is_numeric_dtype(test_subset[col])
        ]
        logger.debug("run_id=%s numeric_columns=%s", run_id, numeric_columns)

        # Perform KS test on each numeric column
        for column in numeric_columns:
            try:
                # Drop NA values before testing
                a = train_subset[column].dropna().values
                b = test_subset[column].dropna().values
                if len(a) < 2 or len(b) < 2:
                    logger.debug(
                        "Skipping column %s for run %s due to insufficient samples: %s vs %s",
                        column,
                        run_id,
                        len(a),
                        len(b),
                    )
                    continue

                _, p_value = ks_2samp(a, b)
                drift_results.append(
                    {
                        "run_id": run_id,
                        "data_id": id_mapping.get(run_id, {}).get("data_id"),
                        "key": f"[drift_metrics]{column}",
                        "value": p_value,
                        # "drift_detected": p_value < 0.05,  # type: ignore
                        "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    }
                )
            except Exception:
                logger.exception(
                    "Error analyzing drift for column %s (run %s)", column, run_id
                )

    # Convert the list of dictionaries to a pandas DataFrame
    if not drift_results:
        return pd.DataFrame()  # Return empty DataFrame if no results
    return pd.DataFrame(drift_results)


def map_mlflow_data_duration_leakage(dataset, id_mapping):
    # Split the dataframe into two parts based on the 'stage' column
    training_data = dataset[dataset["stage"] == "train"]
    testing_data = dataset[dataset["stage"] == "test"]

    # Find the duration of training and testing datasets
    results = []

    # Group data by run_id
    run_ids = set(training_data["run_id"]).intersection(set(testing_data["run_id"]))

    for run_id in run_ids:
        # Filter data for this run_id
        train_subset = training_data[training_data["run_id"] == run_id]
        test_subset = testing_data[testing_data["run_id"] == run_id]

        # Check if timestamps are available
        if "timestamps" in train_subset.columns and "timestamps" in test_subset.columns:
            try:
                # Convert timestamps to datetime and find min/max
                train_timestamps = pd.to_datetime(train_subset["timestamps"].explode())
                test_timestamps = pd.to_datetime(test_subset["timestamps"].explode())

                if not train_timestamps.empty and not test_timestamps.empty:
                    train_min = train_timestamps.min()
                    train_max = train_timestamps.max()
                    test_min = test_timestamps.min()
                    test_max = test_timestamps.max()

                    # Calculate overlap
                    train_duration = (train_max - train_min).total_seconds()
                    test_duration = (test_max - test_min).total_seconds()

                    results.append(
                        {
                            "run_id": run_id,
                            "data_id": id_mapping.get(run_id, {}).get("data_id"),
                            "key": "train_duration_seconds",
                            "value": train_duration,
                            "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                        }
                    )

                    results.append(
                        {
                            "run_id": run_id,
                            "data_id": id_mapping.get(run_id, {}).get("data_id"),
                            "key": "test_duration_seconds",
                            "value": test_duration,
                            "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                        }
                    )

                    # Calculate future leakage in the same loop
                    test_first_timestamp = test_min
                    future_timestamps_count = (
                        train_timestamps > test_first_timestamp
                    ).sum()

                    if future_timestamps_count > 0:
                        # Calculate percentage of training data that leaks into the future
                        leakage_percentage = (
                            future_timestamps_count / len(train_timestamps)
                        ) * 100

                        results.append(
                            {
                                "run_id": run_id,
                                "data_id": id_mapping.get(run_id, {}).get("data_id"),
                                "key": "train_future_leakage_percentage",
                                "value": leakage_percentage,
                                "timestamp": int(
                                    pd.Timestamp.now(tz="UTC").timestamp()
                                ),
                            }
                        )
            except (ValueError, TypeError) as e:
                print(f"Error analyzing temporal data for run_id {run_id}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# New map functions — missing certain_db tables
# ---------------------------------------------------------------------------


def map_run_code(
    tags_df: "pd.DataFrame", run_id: str, artifact_git_record: dict = None
) -> dict:
    """
    Build a runs_code row only for parent runs.

    Child runs are skipped. A child run is detected by the presence of
    the MLflow tag: mlflow.parentRunId.
    """
    # Normalize MLflow SQL column name if needed
    if "run_id" in tags_df.columns and "run_uuid" not in tags_df.columns:
        tags_df = tags_df.rename(columns={"run_id": "run_uuid"})

    run_id = str(run_id)
    tags_df["run_uuid"] = tags_df["run_uuid"].astype(str)

    run_tags = tags_df[tags_df["run_uuid"] == run_id]

    def get_tag_value(key: str):
        values = run_tags.loc[run_tags["key"] == key, "value"].values
        return values[0] if len(values) > 0 else None

    # If this run has a parent, it is a child run.
    # Do not save child runs into runs_code.
    parent_id = get_tag_value("mlflow.parentRunId")
    if parent_id is not None:
        return {}

    # If artifact-provided git metadata is available, prefer it
    if artifact_git_record and isinstance(artifact_git_record, dict):
        commit = artifact_git_record.get("git.commit") or artifact_git_record.get(
            "mlflow.source.git.commit"
        )
        name = artifact_git_record.get("mlflow.source.name") or artifact_git_record.get(
            "git.source.name"
        )

        if not commit:
            return {}

        return {
            "run_id": run_id,
            "git_commit_hash": commit,
            "git_commit_short": artifact_git_record.get("git.commit.short"),
            "git_branch": artifact_git_record.get("git.branch"),
            "git_message": artifact_git_record.get("git.message"),
            "git_author": artifact_git_record.get("git.author"),
            "git_author_email": artifact_git_record.get("git.author.email"),
            "name": name or "unknown",
        }

    # # MLflow default tags
    # mlflow_commit = get_tag_value("mlflow.source.git.commit")
    # source_name = get_tag_value("mlflow.source.name")

    # # Custom Git tags
    # git_commit = get_tag_value("git.commit")
    # git_short = get_tag_value("git.commit.short")
    # git_branch = get_tag_value("git.branch")
    # git_message = get_tag_value("git.message")
    # git_author = get_tag_value("git.author")
    # git_author_email = get_tag_value("git.author.email")

    # # Prefer custom git.commit, fallback to MLflow automatic commit tag
    # commit = git_commit or mlflow_commit

    # # If no commit exists, do not insert into runs_code
    # if commit is None or commit == "" or commit == "unknown":
    #     return {}

    # return {
    #     "run_id": run_id,
    #     "git_commit_hash": commit,
    #     "git_commit_short": git_short,
    #     "git_branch": git_branch,
    #     "git_message": git_message,
    #     "git_author": git_author,
    #     "git_author_email": git_author_email,
    #     "name": source_name or "unknown",
    # }


def map_checkpoints(row: "pd.Series", id_mapping: dict) -> dict:
    """
    Map a row from a ``checkpoints/*.csv`` artifact into a ``checkpoints`` row.

    Parameters
    ----------
    row : pd.Series
        A row with at least ``checkpoint_id``, ``checkpoint_name``,
        ``checkpoint_location``, ``creation_time``, and ``run_id`` columns.
    id_mapping : dict
        Mapping of ``run_id`` → ``{model_id, data_id, deployment_id}``.

    Returns
    -------
    dict
        A single row dict matching the ``checkpoints`` schema.
    """
    run_id = row["run_id"]
    return {
        "checkpoint_id": row.get("checkpoint_id", ""),
        "run_id": run_id,
        "model_id": id_mapping.get(run_id, {}).get("model_id", ""),
        "checkpoint_name": row.get("checkpoint_name", ""),
        "checkpoint_location": row.get("checkpoint_location", ""),
        "creation_time": row.get(
            "creation_time", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
    }


def map_weight_distribution(row: "pd.Series", id_mapping: dict) -> dict:
    """
    Map a row from a ``weight_distribution/*.csv`` artifact into a
    ``weight_distribution`` row.

    Parameters
    ----------
    row : pd.Series
        A row with columns ``layer_name``, ``mean``, ``std``, ``step``,
        ``stage``, ``is_NaN``, ``timestamp``, and ``run_id``.
    id_mapping : dict
        Mapping of ``run_id`` → ``{model_id, data_id, deployment_id}``.

    Returns
    -------
    dict
        A single row dict matching the ``weight_distribution`` schema.
    """
    run_id = row["run_id"]
    value_mean = row.get("mean", 0.0)
    value_std = row.get("std", 0.0)
    return {
        "run_id": run_id,
        "model_id": id_mapping.get(run_id, {}).get("model_id", ""),
        "layer_name": row.get("layer_name", "unknown"),
        "mean": 0.0 if pd.isna(value_mean) else float(value_mean),
        "std": 0.0 if pd.isna(value_std) else float(value_std),
        "step": int(row.get("step", 0)),
        "stage": row.get("stage", "train"),
        "is_NaN": bool(pd.isna(value_mean)),
        "timestamp": row.get("timestamp", int(pd.Timestamp.now(tz="UTC").timestamp())),
    }


def map_examples(row: "pd.Series", id_mapping: dict) -> dict:
    """
    Map a row from an ``examples/*.csv`` artifact into an ``examples`` row.

    Parameters
    ----------
    row : pd.Series
        A row with columns ``input``, ``prediction``, ``ground_truth``,
        ``step``, ``stage``, ``timestamp``, and ``run_id``.
    id_mapping : dict
        Mapping of ``run_id`` → ``{model_id, data_id, deployment_id}``.

    Returns
    -------
    dict
        A single row dict matching the ``examples`` schema.
    """
    run_id = row["run_id"]
    return {
        "run_id": run_id,
        "model_id": id_mapping.get(run_id, {}).get("model_id", ""),
        "input": str(row.get("input", "")),
        "prediction": str(row.get("prediction", "")),
        "ground_truth": str(row.get("ground_truth", "")),
        "step": int(row.get("step", 0)),
        "stage": row.get("stage", "inference"),
        "timestamp": row.get("timestamp", int(pd.Timestamp.now(tz="UTC").timestamp())),
    }


def map_run_logs(row: "pd.Series", run_id: str) -> dict:
    """
    Map a row from a ``run_logs/*.csv`` artifact into a ``runs_logs`` row.

    Parameters
    ----------
    row : pd.Series
        A row with columns ``log_id``, ``log_type``, ``log_message``,
        ``log_creation_time``.
    run_id : str
        The run identifier that owns this log entry.

    Returns
    -------
    dict
        A single row dict matching the ``runs_logs`` schema.
    """
    return {
        "run_id": run_id,
        "log_id": row.get("log_id", ""),
        "log_type": row.get("log_type", "stdout"),
        "log_message": row.get("log_message", ""),
        "log_creation_time": row.get(
            "log_creation_time", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
    }


# ---------------------------------------------------------------------------
# Compliance map functions — governance, documentation, deployment lifecycle
# ---------------------------------------------------------------------------


def map_ai_actors(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into an ``ai_actors`` row.

    The ai_actors table is associated with a run (runs.run_id). The mapper
    therefore accepts the *run_id* and returns a dictionary containing that
    run_id so it can be inserted into the database.
    """
    providers = record.get("ai_providers", [])
    deployers = record.get("ai_deployers", [])
    # Return lists in the mapped row (tests expect raw lists, not JSON strings)
    return {
        "ai_actors_id": record.get("ai_actors_id", ""),
        # Prefer experiment_id embedded in the artifact if present (backwards
        # compatibility). Otherwise fall back to NULL — the DB schema requires
        # experiment_id for the current PK; we include it when available.
        "experiment_id": record.get("experiment_id", ""),
        "run_id": run_id,
        "ai_providers": providers if isinstance(providers, list) else [providers],
        "ai_deployers": deployers if isinstance(deployers, list) else [deployers],
        "auditor": record.get("auditor", ""),
        "organization": record.get("organization", ""),
    }


def map_labeling_procedures(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``labeling_procedures`` row."""
    qa_methods = record.get("quality_assurance_methods", [])
    annotators = record.get("annotators", [])
    annotation_tool = record.get("annotation_tool", "")
    # Tests expect raw lists/strings for these fields
    return {
        "labeling_id": record.get("labeling_id", ""),
        "experiment_id": experiment_id,
        "description": record.get("description", ""),
        "quality_assurance_methods": (
            qa_methods if isinstance(qa_methods, list) else [qa_methods]
        ),
        "annotators": annotators if isinstance(annotators, list) else [annotators],
        "annotation_tool": annotation_tool,
        "link": record.get("link", ""),
    }


def map_risk(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``risks`` row."""
    return {
        "risk_id": record.get("risk_id", ""),
        "experiment_id": experiment_id,
        "risk_description": record.get("risk_description", ""),
        "risk_type": record.get("risk_type", ""),
        "risk_level": float(record.get("risk_level", 0.0)),
    }


def map_human_oversight(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``human_oversight_mechanisms`` row."""
    return {
        "mechanism_id": record.get("mechanism_id", ""),
        "experiment_id": experiment_id,
        "oversight_type": record.get("oversight_type", ""),
        "description": record.get("description", ""),
        "implementation_details": record.get("implementation_details", ""),
    }


def map_transparency_measure(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``transparency_measures`` row."""
    return {
        "measure_id": record.get("measure_id", ""),
        "experiment_id": experiment_id,
        "measure_type": record.get("measure_type", []),
        "measure_value": record.get("measure_value", []),
        "description": record.get("description", ""),
        "implementation_details": record.get("implementation_details", ""),
    }


def map_change_log(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into a ``change_logs`` row."""
    return {
        "log_id": record.get("log_id", ""),
        "run_id": run_id,
        "change_description": record.get("change_description", ""),
        "changed_by": record.get("changed_by", ""),
        "change_timestamp": record.get(
            "change_timestamp", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
    }


def map_declaration_of_conformity(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into a ``declaration_of_conformity`` row."""
    return {
        "declaration_id": record.get("declaration_id", ""),
        "run_id": run_id,
        "filename": record.get("filename", ""),
        "file_type": record.get("file_type", ""),
        "mime_type": record.get("mime_type", ""),
        "file_size": record.get("file_size"),
        "description": record.get("description", ""),
        "issuer": record.get("issuer", ""),
        "version": record.get("version", ""),
        "valid_from": record.get("valid_from"),
        "valid_until": record.get("valid_until"),
        "standard_references": record.get("standard_references", []),
        "creation_time": record.get(
            "creation_time", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
        "link_to_artifacts": record.get("link_to_artifacts", ""),
    }


def map_visual_documentation(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into a ``visual_documentation`` row."""
    return {
        "document_id": record.get("document_id", ""),
        "run_id": run_id,
        "filename": record.get("filename", ""),
        "file_type": record.get("file_type", ""),
        "file_size": record.get("file_size"),
        "description": record.get("description", ""),
        "stage": record.get("stage", ""),
        "generated_by": record.get("generated_by", ""),
        "model_version": record.get("model_version", ""),
        "tags": record.get("tags", []),
        "creation_time": record.get(
            "creation_time", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
        "link_to_artifacts": record.get("link_to_artifacts", ""),
    }


def map_explainable_ai(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into an ``explainable_ai_features`` row."""
    return {
        "feature_id": record.get("feature_id", ""),
        "run_id": run_id,
        "feature_name": record.get("feature_name", []),
        "feature_values": record.get("feature_values", []),
        "implementation_details": record.get("implementation_details", ""),
    }


def map_model_packaging(record: dict, experiment_id: str, id_mapping: dict) -> dict:
    """Map a JSON artifact record into a ``model_packaging`` row."""
    deployment_id = record.get("deployment_id", "")
    model_id = record.get("model_id", "")
    return {
        "packaging_id": record.get("packaging_id", ""),
        "experiment_id": experiment_id,
        "deployment_id": deployment_id,
        "model_id": model_id,
        "packaging_format": record.get("packaging_format", ""),
        "dependencies": record.get("dependencies", []),
        "containerization_details": (
            json.dumps(record.get("containerization_details", {}))
            if isinstance(record.get("containerization_details"), dict)
            else record.get("containerization_details", "")
        ),
    }


def map_build_testing(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``build_and_integration_testing`` row."""
    test_results = record.get("test_results", {})
    if isinstance(test_results, dict):
        test_results = json.dumps(test_results)
    return {
        "test_id": record.get("test_id", ""),
        "experiment_id": experiment_id,
        "deployment_id": record.get("deployment_id", ""),
        "model_id": record.get("model_id", ""),
        "build_status": record.get("build_status", ""),
        "build_logs": record.get("build_logs", ""),
        "build_timestamp": record.get(
            "build_timestamp", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
        "test_type": record.get("test_type", ""),
        "test_results": test_results,
    }


def map_standard(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``standards`` row."""
    return {
        "standard_id": record.get("standard_id", ""),
        "experiment_id": experiment_id,
        "deployment_id": record.get("deployment_id", ""),
        "model_id": record.get("model_id", ""),
        "name": record.get("name", ""),
        "description": record.get("description", ""),
        "version": record.get("version", ""),
        "publication_date": record.get("publication_date"),
    }


def map_interface(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into an ``interfaces`` row."""
    return {
        "interface_id": record.get("interface_id", ""),
        "experiment_id": experiment_id,
        "deployment_id": record.get("deployment_id", ""),
        "model_id": record.get("model_id", ""),
        "interface_type": record.get("interface_type", ""),
        "specifications": record.get("specifications", ""),
        "version": record.get("version", ""),
        "documentation_link": record.get("documentation_link", ""),
    }


def map_decommissioning(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into a ``decomissioning`` row."""
    return {
        "decomissioning_id": record.get("decomissioning_id", ""),
        "experiment_id": experiment_id,
        "deployment_id": record.get("deployment_id", ""),
        "model_id": record.get("model_id", ""),
        "decomissioning_date": record.get(
            "decomissioning_date", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
        "decomissioning_actions": record.get("decomissioning_actions", []),
        "reason": record.get("reason", ""),
        "procedure_details": record.get("procedure_details", ""),
    }


def map_tokenizer_config(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into a ``tokenizer_config`` row."""
    return {
        "tokenizer_id": record.get("tokenizer_id", ""),
        "run_id": run_id,
        "tokenizer_type": record.get("tokenizer_type", ""),
        "model_name_or_path": record.get("model_name_or_path"),
        "vocab_size": record.get("vocab_size"),
        "max_length": record.get("max_length"),
        "padding": (
            str(record["padding"]) if record.get("padding") is not None else None
        ),
        "truncation": (
            bool(record["truncation"]) if record.get("truncation") is not None else None
        ),
        "stride": record.get("stride"),
        "special_tokens": record.get("special_tokens"),
    }


def map_tokenization_stats(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into a ``tokenization_stats`` row."""
    return {
        "stats_id": record.get("stats_id", ""),
        "run_id": run_id,
        "split": record.get("split", ""),
        "total_sequences": record.get("total_sequences"),
        "total_tokens": record.get("total_tokens"),
        "avg_token_length": record.get("avg_token_length"),
        "min_token_length": record.get("min_token_length"),
        "max_token_length": record.get("max_token_length"),
        "truncation_rate": record.get("truncation_rate"),
        "padding_rate": record.get("padding_rate"),
        "oov_rate": record.get("oov_rate"),
    }


def map_mlflow_data_techniques(record: dict, run_id: str) -> dict:
    """Map a data_techniques JSON artifact into rows for
    ``data_techniques`` and ``data_hyperparameters``.

    Returns a dict with two lists:
      - 'techniques': list of rows matching the data_techniques schema
      - 'hyperparameters': list of rows matching the data_hyperparameters schema
    """
    if not isinstance(record, dict):
        record = {}

    container = record.get("techniques") if "techniques" in record else record
    if not isinstance(container, dict):
        container = {}

    techniques_rows = []
    hyperparams_rows = []

    global_stage = record.get("data_technique_stage") or record.get("stage")

    for tname, props in container.items():
        if not isinstance(props, dict):
            props = {"method": props}

        stage = props.get("stage") or global_stage

        techniques_rows.append(
            {
                "run_id": run_id,
                "data_id": None,
                "technique_name": [tname],
                "data_technique_stage": stage,
                "technique_details": {
                    "name": tname,
                    "method": props.get("method"),
                    "library": props.get("library"),
                    "notes": props.get("notes"),
                    "parameters": props.get("parameters", {}),
                },
            }
        )

        params = props.get("parameters") or {}
        if isinstance(params, dict):
            for pname, pval in params.items():
                hyperparams_rows.append(
                    {
                        "run_id": run_id,
                        "data_id": None,
                        "technique_name": tname,
                        "technique_parameter_name": pname,
                        "technique_parameter_value": str(pval),
                    }
                )

    return {"techniques": techniques_rows, "hyperparameters": hyperparams_rows}
