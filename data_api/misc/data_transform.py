import json
import os
import logging
import pandas as pd
from scipy.stats import ks_2samp
import hashlib

logger = logging.getLogger(__name__)


def map_runs(run):
    """Map a run record read from certain/metadata/run_metadata.json into a runs table row.

    The artifact is written by certain_library.tracking.tracker.Tracker.start_run /
    end_run and contains all run fields directly — no MLflow Postgres query needed.
    `run` may be a pandas Series (from DataFrame.apply) or a plain dict.
    """
    if isinstance(run, pd.Series):
        r = run.to_dict()
    elif hasattr(run, "__getitem__"):
        r = run
    else:
        r = {}

    # Support both 'run_id' (artifact convention) and 'run_uuid' (MLflow SQL convention)
    run_id = r.get("run_uuid") or r.get("run_id") or ""
    return {
        "run_id": str(run_id),
        "run_name": r.get("run_name") or r.get("name") or "",
        "parent_id": r.get("parent_id") or r.get("parent_run_id"),
        "source_type": r.get("source_type", "LOCAL"),
        "source_name": r.get("source_name", ""),
        "user_id": r.get("user_id", ""),
        "status": r.get("status", "FINISHED"),
        "start_time": r.get("start_time", 0),
        "end_time": r.get("end_time", 0),
        "source_version": r.get("source_version", ""),
        "experiment_id": r.get("experiment_id", ""),
    }


def map_experiments(experiments):
    """Map an experiment record from the artifact-first get_experiments_data() into an experiments table row.

    get_experiments_data() already reads from the artifact store (certain/metadata or
    directory listing) — this mapper just normalises the field names for the DB schema.
    `experiments` may be a pandas Series or a plain dict.
    """
    if isinstance(experiments, pd.Series):
        r = experiments.to_dict()
    elif hasattr(experiments, "__getitem__"):
        r = experiments
    else:
        r = {}

    lifecycle = r.get("lifecycle_stage") or r.get("experiment_stage") or "active"
    return {
        "experiment_id": str(r.get("experiment_id", "")),
        "experiment_name": r.get("name") or r.get("experiment_name") or "default",
        "lifecycle_stage": lifecycle,
        "experiment_stage": r.get("experiment_stage") or lifecycle,
        "description": r.get("description"),
        "creation_time": r.get("creation_time") or int(pd.Timestamp.now(tz="UTC").timestamp()),
        "last_update_time": r.get("last_update_time") or int(pd.Timestamp.now(tz="UTC").timestamp()),
    }


def map_datasets(datasets, run_id, id_mapping):
    """Map a dataset record into a data table row.

    Previously this function resolved location/size from the MLflow SQL datasets
    table; now it reads exclusively from the CERTAIN artifact manifests:
      - certain/dataset/data_manifest.json  (full manifest with files & total_size_bytes)
      - certain/metadata/data.json          (lightweight record with data_location & data_size)

    `datasets` is kept as a parameter for API compatibility but is no longer used
    when a manifest is available; the manifest is always preferred.
    """
    # Prefer the canonical artifact manifest written by save_dataset_manifest
    try:
        from data_api.app.mlflow_connector import get_dataset_manifest_for_run
        manifest = get_dataset_manifest_for_run(str(run_id))
    except Exception:
        manifest = None

    if manifest and isinstance(manifest, dict):
        # Delegate to the dedicated artifact mapper (defined later in this file)
        return map_dataset_manifest(manifest, run_id, id_mapping)

    # Fallback: build a minimal row from whatever was passed in (keeps backwards
    # compatibility when no manifest artifact exists yet for a run).
    record = {}
    try:
        if isinstance(datasets, pd.Series):
            record = datasets.to_dict()
        elif isinstance(datasets, dict):
            record = dict(datasets)
    except Exception:
        record = {}

    data_id = (id_mapping.get(run_id) or {}).get("data_id")
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    return {
        "run_id": run_id,
        "data_id": data_id,
        "data_stage": record.get("data_stage", "training"),
        "data_type": record.get("data_type", "tabular"),
        "data_source": record.get("source", "local"),
        "data_version": record.get("version", "v1"),
        "data_location": record.get("location") or record.get("path") or "",
        "data_size": int(record.get("size") or 0),
        "data_format": record.get("format", "csv"),
        "creation_time": now_ts,
        "last_update_time": now_ts,
    }


def map_data_metrics(metrics, id_mapping):

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


def map_model_metrics(metrics, id_mapping):
    """Map a metric event record from certain/metadata/events.jsonl into a model_metrics row.

    get_metrics_data() reads from events.jsonl (event_type == 'metric') written by
    certain_library.tracking.tracker.Tracker.log_metrics. The event dict uses
    'run_id' (renamed to 'run_uuid' by get_metrics_data for compatibility).
    """
    if isinstance(metrics, pd.Series):
        r = metrics.to_dict()
    elif hasattr(metrics, "__getitem__"):
        r = metrics
    else:
        r = {}

    # Support both naming conventions
    run_id = str(r.get("run_uuid") or r.get("run_id") or "")
    model_id = (id_mapping.get(run_id) or {}).get("model_id", "")

    # Normalise is_NaN / is_nan
    is_nan_raw = r.get("is_nan") if r.get("is_nan") is not None else r.get("is_NaN", False)

    return {
        "run_id": run_id,
        "model_id": model_id,
        "key": r.get("key", ""),
        "value": r.get("value", 0),
        "step": int(r.get("step") or 0),
        "timestamp": int(r.get("timestamp") or pd.Timestamp.now(tz="UTC").timestamp()),
        "stage": r.get("stage", "train"),
        "is_NaN": bool(is_nan_raw),
    }


def map_model_params(params, id_mapping):
    """Map a param event record from certain/metadata/events.jsonl into a model_hyperparameters row.

    get_params_data() reads from events.jsonl (event_type == 'param') written by
    certain_library.tracking.tracker.Tracker.log_params.
    """
    if isinstance(params, pd.Series):
        r = params.to_dict()
    elif hasattr(params, "__getitem__"):
        r = params
    else:
        r = {}

    run_id = str(r.get("run_uuid") or r.get("run_id") or "")
    model_id = (id_mapping.get(run_id) or {}).get("model_id", "")

    return {
        "run_id": run_id,
        "model_id": model_id,
        "key": r.get("key", "param_key"),
        "value": str(r.get("value", "")),
    }


def map_runs_tags(runs_tags):
    """Map a tag event record from certain/metadata/events.jsonl into a runs_tags row.

    get_tags_data() reads from events.jsonl (event_type == 'tag') written by
    certain_library.tracking.tracker.Tracker.set_tags / start_run.
    """
    if isinstance(runs_tags, pd.Series):
        r = runs_tags.to_dict()
    elif hasattr(runs_tags, "__getitem__"):
        r = runs_tags
    else:
        r = {}

    return {
        "run_id": str(r.get("run_uuid") or r.get("run_id") or ""),
        "key": r.get("key", "tag_key"),
        "value": r.get("value", "tag_value"),
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


def map_data_resources(data_resources, id_mapping):
    """Map a CodeCarbon JSON artifact row into data_resources rows.

    Reads from certain/code_carbon/*.json written by certain_library.resource_monitor.
    Each row is a flat dict; we emit one row per known resource key.
    """
    if isinstance(data_resources, pd.Series):
        r = data_resources.to_dict()
    elif hasattr(data_resources, "__getitem__"):
        r = data_resources
    else:
        r = {}

    run_id = str(r.get("run_id", ""))
    data_id = (id_mapping.get(run_id) or {}).get("data_id", "")
    stage = r.get("stage", "data_default")
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

    data = []
    for key in resources_key:
        raw = r.get(key)
        value = None if (raw is None or (isinstance(raw, float) and pd.isna(raw))) else raw
        data.append(
            {
                "run_id": run_id,
                "data_id": data_id,
                "stage": stage,
                "key": key,
                "value": value,
                "timestamp": now_ts,
            }
        )
    return data


def map_resources(resources, id_mapping):
    """Map a CodeCarbon JSON artifact row into resources (model-scoped) rows.

    Reads from certain/code_carbon/*.json written by certain_library.resource_monitor.
    """
    if isinstance(resources, pd.Series):
        r = resources.to_dict()
    elif hasattr(resources, "__getitem__"):
        r = resources
    else:
        r = {}

    run_id = str(r.get("run_id", ""))
    model_id = (id_mapping.get(run_id) or {}).get("model_id", "")
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

    data = []
    for key in resources_key:
        raw = r.get(key)
        value = None if (raw is None or (isinstance(raw, float) and pd.isna(raw))) else raw
        data.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "key": key,
                "step": int(r.get("step") or 0),
                "stage": r.get("stage", "train_default"),
                "value": value,
                "timestamp": now_ts,
            }
        )
    return data


def map_time_series_data(time_series_data, id_mapping):
    # For each row, access content and compute the frequency of sampling
    # Process a single row instead of iterating through a DataFrame
    row = time_series_data
    monotonic_increase = False
    avg_sampling_rate = None
    missing_intervals = 0

    if "timestamps" in row and isinstance(row["timestamps"], list):
        timestamps = row["timestamps"]
        # Convert timestamps from string to datetime. The source .txt
        # artifacts contain ISO 8601 strings (optionally with a UTC offset)
        # plus a non-parseable header line ("Train Timestamps:" / "Test
        # Timestamps:"); format="ISO8601" lets pandas parse all ISO 8601
        # variants without falling back to the slow/warning-prone dateutil
        # parser, while errors="coerce" turns the header line into NaT.
        timestamps = pd.to_datetime(
            [ts for ts in timestamps if isinstance(ts, str)],
            format="ISO8601",
            errors="coerce",
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


def map_data_drift(dataset, id_mapping):
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
        "map_data_drift called: dataset shape=%s, training=%s, testing=%s",
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


def map_data_duration_leakage(dataset, id_mapping):
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
    # Prefer explicit fields, but accept common alternatives produced by
    # different logging/exporters (message, msg, line_id, timestamp, stream).
    try:
        # message
        message = None
        for candidate in ("log_message", "message", "msg", "text", "message_text"):
            if candidate in row and pd.notna(row.get(candidate)):
                message = row.get(candidate)
                break
        if message is None:
            # Last-resort: stringify the entire row
            message = str(row.to_dict()) if hasattr(row, "to_dict") else ""

        # log_id (generate if missing)
        log_id = None
        for candidate in ("log_id", "line_id", "id", "idx"):
            if candidate in row and pd.notna(row.get(candidate)):
                log_id = str(row.get(candidate))
                break
        if not log_id:
            import uuid

            log_id = uuid.uuid4().hex

        # log_type (stream / level fallback)
        log_type = None
        for candidate in ("log_type", "stream", "level", "logger"):
            if candidate in row and pd.notna(row.get(candidate)):
                log_type = str(row.get(candidate))
                break
        if not log_type:
            log_type = "stdout"

        # log_creation_time (coerce from various candidates)
        log_time = None
        for candidate in (
            "log_creation_time",
            "timestamp",
            "time",
            "created_at",
            "ts",
        ):
            if candidate in row and pd.notna(row.get(candidate)):
                log_time = row.get(candidate)
                break

        # Convert to integer epoch seconds
        if log_time is None:
            log_creation = int(pd.Timestamp.now(tz="UTC").timestamp())
        else:
            try:
                # If it's a pandas Timestamp or datetime-like
                if hasattr(log_time, "timestamp"):
                    log_creation = int(log_time.timestamp())
                else:
                    # numeric or string
                    log_creation = int(float(log_time))
            except Exception:
                try:
                    # Try parsing with pandas
                    log_creation = int(pd.to_datetime(log_time, utc=True).timestamp())
                except Exception:
                    log_creation = int(pd.Timestamp.now(tz="UTC").timestamp())

        return {
            "run_id": run_id,
            "log_id": log_id,
            "log_type": log_type,
            "log_message": str(message),
            "log_creation_time": int(log_creation),
        }
    except Exception:
        # Fail-safe minimal row
        import uuid

        return {
            "run_id": run_id,
            "log_id": uuid.uuid4().hex,
            "log_type": "stdout",
            "log_message": "",
            "log_creation_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
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


def map_human_oversight(record: dict, experiment_id: str, run_id: str = None) -> dict:
    """Map a JSON artifact record into a ``human_oversight_mechanisms`` row."""
    return {
        "mechanism_id": record.get("mechanism_id", ""),
        "experiment_id": experiment_id,
        "run_id": run_id,
        # include deployment_id when present in artifact (optional)
        "deployment_id": record.get("deployment_id"),
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
        # optional deployment identifier (some artifacts may include this)
        "deployment_id": record.get("deployment_id"),
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
        # optional deployment identifier (artifact may include this)
        "deployment_id": record.get("deployment_id"),
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


def map_standard(record: dict, run_id: str) -> dict:
    """Map a JSON artifact record into a ``standards`` row.

    This mapper mirrors :func:`map_declaration_of_conformity` and is
    run-scoped: the sync helper will pass the run_id for the artifact's
    run so the resulting row can be linked to the `runs` table.
    """
    return {
        "standard_id": record.get("standard_id", ""),
        "run_id": run_id,
        # optional deployment identifier (some artifacts may include this)
        "deployment_id": record.get("deployment_id"),
        "model_id": record.get("model_id"),
        "name": record.get("name", ""),
        "description": record.get("description", ""),
        "version": record.get("version", ""),
        "publication_date": record.get("publication_date"),
        "creation_time": record.get(
            "creation_time", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
    }


def map_interface(record: dict, experiment_id: str, run_id: str) -> dict:
    """Map a JSON artifact record into an ``interfaces`` row."""
    return {
        "interface_id": record.get("interface_id", ""),
        "experiment_id": experiment_id,
        "run_id": run_id,
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
        "system_name": record.get("system_name", ""),
        "decommissioning_plan": record.get("decommissioning_plan", ""),
        "approvals": record.get("approvals", []),
        "data_retention_archive": record.get("data_retention_archive", ""),
        "migration": record.get("migration", ""),
        "access_removal": record.get("access_removal", ""),
        "infrastructure_shutdown": record.get("infrastructure_shutdown", ""),
        "evidence_documentation": record.get("evidence_documentation", []),
        "audit_trail": record.get("audit_trail", ""),
        "decomissioning_date": record.get(
            "decomissioning_date", int(pd.Timestamp.now(tz="UTC").timestamp())
        ),
        "decomissioning_actions": record.get("decomissioning_actions", []),
        "reason": record.get("reason", ""),
        "procedure_details": record.get("procedure_details", ""),
    }


def map_monitor_logs(record: dict, experiment_id: str, run_id: str) -> dict:
    """Map a JSON artifact record into a ``monitor_logs`` row."""
    return {
        "deployment_id": record.get("deployment_id", ""),
        "experiment_id": record.get("experiment_id") or experiment_id,
        "model_id": record.get("model_id", ""),
        "log_id": record.get("log_id", ""),
        "message": record.get("message") or record.get("deployment_log", ""),
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


def map_data_techniques(record: dict, run_id: str) -> dict:
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


# ---------------------------------------------------------------------------
# Mappers for certain/metadata artifact JSON files
# ---------------------------------------------------------------------------


def map_run_params(record: dict, run_id: str, id_mapping: dict) -> list:
    """Map certain/metadata/run_params.json into model_hyperparameters rows.

    The artifact shape is:
        {run_id, run_params: {key: value, ...}, captured_at}

    Returns a list of dicts (one per param key).
    """
    rows = []
    params = record.get("run_params") or {}
    model_id = id_mapping.get(run_id, {}).get("model_id")
    if not model_id:
        return rows
    for key, value in params.items():
        rows.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "key": str(key),
                "value": str(value),
            }
        )
    return rows


def map_run_metrics(record: dict, run_id: str, id_mapping: dict) -> list:
    """Map certain/metadata/run_metrics.json into model_metrics rows.

    The artifact shape is:
        {run_id, run_metrics: {key: [{value, step, timestamp}, ...]}, captured_at}

    Returns a list of dicts (one per metric history entry).
    """
    rows = []
    metrics = record.get("run_metrics") or {}
    model_id = id_mapping.get(run_id, {}).get("model_id")
    if not model_id:
        return rows
    for key, history in metrics.items():
        if not isinstance(history, list):
            history = [history]
        for entry in history:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "key": str(key),
                    "value": entry.get("value", 0),
                    "step": int(entry.get("step", 0)),
                    "timestamp": int(
                        entry.get("timestamp")
                        or pd.Timestamp.now(tz="UTC").timestamp()
                    ),
                    "stage": "train",
                    "is_NaN": False,
                }
            )
    return rows


def map_run_resources(record: dict, run_id: str, id_mapping: dict) -> list:
    """Map certain/metadata/run_resources.json into resources rows.

    The artifact shape is:
        {run_id, run_resources: {key: [{value, step, timestamp}, ...]}, captured_at}

    Returns a list of dicts (one per resource history entry).
    """
    rows = []
    resources = record.get("run_resources") or {}
    model_id = id_mapping.get(run_id, {}).get("model_id")
    if not model_id:
        return rows
    for key, history in resources.items():
        if not isinstance(history, list):
            history = [history]
        for entry in history:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "key": str(key),
                    "step": int(entry.get("step", 0)),
                    "stage": "train",
                    "value": entry.get("value", 0),
                    "timestamp": int(
                        entry.get("timestamp")
                        or pd.Timestamp.now(tz="UTC").timestamp()
                    ),
                }
            )
    return rows


def map_run_inputs(record: dict, run_id: str, id_mapping: dict) -> list:
    """Map certain/metadata/run_inputs.json (a.k.a. inputs.json) into data rows.

    The artifact shape is:
        {run_inputs: [{dataset_id, dataset_name, dataset_source, dataset_schema,
                        dataset_profile, tags}, ...], captured_at}

    Returns a list of dicts compatible with the `data` table.
    """
    rows = []
    inputs = record.get("run_inputs") or []
    data_id = id_mapping.get(run_id, {}).get("data_id")
    if not data_id:
        return rows
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        rows.append(
            {
                "run_id": run_id,
                "data_id": data_id,
                "data_stage": (inp.get("tags") or {}).get("mlflow.data.context", "training"),
                "data_type": "dataset",
                "data_source": inp.get("dataset_source") or "local",
                "data_version": (inp.get("tags") or {}).get("version", "v1"),
                "data_location": str(inp.get("dataset_source") or ""),
                "data_size": 0,
                "data_format": inp.get("dataset_schema") or "unknown",
                "creation_time": now_ts,
                "last_update_time": now_ts,
            }
        )
    return rows


def map_experiment_tags_artifact(record: dict, experiment_id: str) -> list:
    """Map certain/metadata/experiment_tags.json into experiments_tags rows.

    The artifact shape is:
        {experiment_id, experiment_tags: {key: value, ...}, captured_at}
    or:
        {experiment_tags: {key: value, ...}, captured_at}

    Returns a list of dicts.
    """
    rows = []
    exp_id = str(record.get("experiment_id") or experiment_id or "")
    tags = record.get("experiment_tags") or {}
    for key, value in tags.items():
        if "mlflow" in str(key).lower():
            continue
        rows.append(
            {
                "experiment_id": exp_id,
                "key": str(key),
                "value": str(value),
            }
        )
    return rows


def map_dataset_manifest(record: dict, run_id: str, id_mapping: dict) -> dict:
    """Map certain/dataset/data_manifest.json or certain/metadata/data.json into a data row.

    Handles two artifact shapes:

    Full manifest (data_manifest.json):
        {run_id, files: [{path, size_bytes, sha256}, ...], total_size_bytes, captured_at}

    Lightweight metadata (data.json):
        {run_id, data_location, data_size, data_format, data_stage, data_source,
         data_version, data_type, creation_time, last_update_time}

    Returns a single dict compatible with the `data` table.
    """
    # `id_mapping[run_id]['data_id']` is the single source of truth for this
    # run's data_id: every other table that references a dataset row
    # (data_resources, data_metrics, data_signatures, data_techniques,
    # data_hyperparameters, and map_run_inputs above) resolves its FK value
    # from id_mapping, and sync_data() propagates a parent run's data_id down
    # to its sub-runs by mutating id_mapping in place *before* this mapper
    # ever runs. If we computed our own hash here instead, multiple sync
    # passes (sync_data vs sync_dataset_manifest) or multiple runs sharing a
    # dataset would each get a different data_id, producing duplicate rows
    # in the `data` table (whose PK is (run_id, data_id)) instead of a single
    # upserted row. So: always prefer id_mapping's value, and only fall back
    # to a computed hash when there is truly no id_mapping entry for this run.
    data_id = id_mapping.get(run_id, {}).get("data_id")

    def compute_data_id_from_record(rec: dict) -> str:
        # Build a stable key from canonical fields present in the artifact
        parts = []
        # Lightweight metadata fields
        if rec.get("data_location"):
            parts.append(str(rec.get("data_location")))
        if rec.get("data_size") is not None:
            parts.append(str(rec.get("data_size")))
        if rec.get("data_format"):
            parts.append(str(rec.get("data_format")))
        if rec.get("data_version"):
            parts.append(str(rec.get("data_version")))
        if rec.get("data_type"):
            parts.append(str(rec.get("data_type")))

        # For full manifests, include file paths and checksums if available
        files = rec.get("files") or []
        if files and isinstance(files, list):
            # sort by path to ensure stability
            try:
                sorted_files = sorted(
                    [(f.get("path", ""), f.get("sha256", f.get("sha1", ""))) for f in files],
                    key=lambda x: x[0],
                )
                for p, ch in sorted_files:
                    parts.append(str(p))
                    if ch:
                        parts.append(str(ch))
            except Exception:
                pass

        # total size bytes is a good differentiator
        if rec.get("total_size_bytes") is not None:
            parts.append(str(rec.get("total_size_bytes")))

        key = "|".join(parts)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
    # If the record is lightweight (data.json) prefer computing from its
    # canonical fields. Likewise for a full manifest (data_manifest.json)
    if "data_location" in record:
        if not data_id:
            try:
                data_id = compute_data_id_from_record(record)
            except Exception:
                data_id = None

        return {
            "run_id": run_id,
            "data_id": data_id,
            "data_stage": record.get("data_stage", "training"),
            "data_type": record.get("data_type", "tabular"),
            "data_source": record.get("data_source", "local"),
            "data_version": record.get("data_version", "v1"),
            "data_location": record.get("data_location", ""),
            "data_size": int(record.get("data_size") or 0),
            "data_format": record.get("data_format", "csv"),
            "creation_time": int(record.get("creation_time") or now_ts),
            "last_update_time": int(record.get("last_update_time") or now_ts),
        }

    # full data_manifest.json shape
    files = record.get("files") or []
    first_path = ""
    if files and isinstance(files[0], dict):
        first_path = files[0].get("path", "")
    total_size = int(record.get("total_size_bytes") or 0)

    if not data_id:
        try:
            data_id = compute_data_id_from_record(record)
        except Exception:
            data_id = None

    return {
        "run_id": run_id,
        "data_id": data_id,
        "data_stage": "training",
        "data_type": "tabular",
        "data_source": "local",
        "data_version": "v1",
        "data_location": first_path,
        "data_size": total_size,
        "data_format": "csv",
        "creation_time": now_ts,
        "last_update_time": now_ts,
    }
