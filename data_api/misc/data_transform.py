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
