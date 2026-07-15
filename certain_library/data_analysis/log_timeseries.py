from certain_library.tracking.tracker import tracker
import os

import pandas as pd

# The way to pass the train and test timestamps
# train_timestamps = df_sorted.loc[X_train.index, "utc_timestamp"]


def timestamp_analysis(
    train_timestamps: pd.Series,
    test_timestamps: pd.Series,
    output_dir: str = "timestamps",
) -> None:
    """
    Analyze and log timestamp information for train and test datasets.

    This function logs timestamp statistics (min, max, mean) to MLflow for both
    train and test datasets. It also creates a text file containing all timestamps
    and logs it as an MLflow artifact.

    Parameters
    ----------
    train_timestamps : pd.Series
        Series containing timestamps for training data.
    test_timestamps : pd.Series
        Series containing timestamps for testing data.
    output_dir : str, optional
        Directory where timestamp files will be saved, by default "timestamps".

    Returns
    -------
    None
        This function doesn't return any value.
    """
    # Validate input data
    if not isinstance(train_timestamps, (pd.Series, pd.DataFrame)) or not isinstance(
        test_timestamps, (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            "Both train_timestamps and test_timestamps must be pandas Series objects"
        )

    if train_timestamps.empty or test_timestamps.empty:
        raise ValueError("Both train_timestamps and test_timestamps must not be empty")

    # Check if values are datetime-like
    if not pd.api.types.is_datetime64_any_dtype(train_timestamps):
        try:
            pd.to_datetime(train_timestamps)
        except Exception as e:
            raise ValueError(
                f"train_timestamps must contain datetime-like values: {str(e)}"
            ) from e

    if not pd.api.types.is_datetime64_any_dtype(test_timestamps):
        try:
            pd.to_datetime(test_timestamps)
        except Exception as e:
            raise ValueError(
                f"test_timestamps must contain datetime-like values: {str(e)}"
            ) from e

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    tracker.log_params(
        {
            "train_min_timestamp": str(train_timestamps.min()),
            "train_max_timestamp": str(train_timestamps.max()),
            "train_mean_timestamp": str(train_timestamps.mean()),
            "test_min_timestamp": str(test_timestamps.min()),
            "test_max_timestamp": str(test_timestamps.max()),
            "test_mean_timestamp": str(test_timestamps.mean()),
        }
    )

    all_timestamps_file = os.path.join(output_dir, "all_timestamps.txt")
    with open(all_timestamps_file, "w", encoding="utf-8") as f:
        f.write("Train Timestamps:\n")
        for ts in train_timestamps:
            f.write(f"{str(ts)}\n")
        f.write("\nTest Timestamps:\n")
        for ts in test_timestamps:
            f.write(f"{str(ts)}\n")

    tracker.log_artifact(all_timestamps_file, artifact_path="timestamps")

    # Remove the local file after logging
    if os.path.exists(all_timestamps_file):
        os.remove(all_timestamps_file)
