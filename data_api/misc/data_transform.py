import json
import pandas as pd
from scipy.stats import ks_2samp


def map_mlflow_runs(run):
    return {
        "run_id": run.run_uuid,
        "run_name": run.name,
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
    return {
        "run_id": run_id,
        "data_id": id_mapping[run_id]["data_id"],
        "data_stage": datasets.get("data_stage", "training"),
        "data_type": datasets.get("data_type", "int"),
        "data_source": datasets.get("source", "local"),
        "data_version": datasets.get("version", "v1"),
        "data_location": datasets.get("location", "/home"),
        "data_size": datasets.get("size", 0),
        "data_format": datasets.get("format", "should be csv"),
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

    # Compute the drift between training and testing data
    drift_results = []
    # Group data by run_id
    run_ids = set(training_data["run_id"]).intersection(set(testing_data["run_id"]))

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

        # Perform KS test on each column
        for column in common_columns:
            try:
                _, p_value = ks_2samp(train_subset[column], test_subset[column])
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
            except (ValueError, TypeError, RuntimeError) as e:
                print(f"Error analyzing drift for column {column}: {e}")

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


def map_run_code(tags_df: "pd.DataFrame", run_id: str) -> dict:
    """
    Build a ``runs_code`` row from MLflow system tags.

    MLflow automatically stores ``mlflow.source.git.commit`` and
    ``mlflow.source.name`` as run tags when the code is version-controlled.

    Parameters
    ----------
    tags_df : pd.DataFrame
        DataFrame produced by ``get_tags_data()`` (columns: run_uuid, key, value).
    run_id : str
        The run identifier to extract tags for.

    Returns
    -------
    dict
        A single row dict matching the ``runs_code`` schema, or an empty dict
        if the tags are not present.
    """
    run_tags = tags_df[tags_df["run_uuid"] == run_id]
    git_hash = run_tags.loc[
        run_tags["key"] == "mlflow.source.git.commit", "value"
    ].values
    source_name = run_tags.loc[run_tags["key"] == "mlflow.source.name", "value"].values

    commit = git_hash[0] if len(git_hash) > 0 else "unknown"
    name = source_name[0] if len(source_name) > 0 else "unknown"

    if commit == "unknown" and name == "unknown":
        return {}

    return {
        "run_id": run_id,
        "git_commit_hash": commit,
        "name": name,
    }


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


def map_ai_actors(record: dict, experiment_id: str) -> dict:
    """Map a JSON artifact record into an ``ai_actors`` row."""
    providers = record.get("ai_providers", [])
    deployers = record.get("ai_deployers", [])
    # Return lists in the mapped row (tests expect raw lists, not JSON strings)
    return {
        "ai_actors_id": record.get("ai_actors_id", ""),
        "experiment_id": experiment_id,
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
