import os
import mlflow

import pandas as pd
import whylogs as why


# TODO: Enable function to log different types of profiles, not only pd.DataFrame
def log_whylogs_profile(
    data: pd.DataFrame, name: str = "default", profile_dir: str = "metadata_dir",
    not_nan: bool = False,
) -> None:
    """Log a WhyLogs profile artifact for the provided DataFrame as a CSV file.

    This function generates a WhyLogs profile for the given pandas DataFrame,
    saves it as a CSV file in the specified directory, and logs it as an MLflow
    artifact for tracking and visualization purposes.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to profile with WhyLogs.
    name : str
        Name identifier to use in the output profile filename.
    profile_dir : str
        Directory path where the profile CSV file will be saved.
    not_nan : bool
        If True, raises an error if the DataFrame contains any NaN values.

    Returns
    -------
    None
        This function doesn't return any value.

    Notes
    -----
    The profile is saved as 'profile_{name}.csv' in the specified directory
    and logged to MLflow under the 'whylogs' artifact path.
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame")

    if data.empty:
        raise ValueError("Input DataFrame is empty")

    if not_nan and data.isna().any().any():
        raise ValueError("Input DataFrame contains NaN values")

    # Check if the profile directory exists, create it if not
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    results = why.log(pandas=data)

    # Check if the profile is callable and call it if needed
    profile_obj = results.profile
    if callable(profile_obj):
        profile_obj = profile_obj()
    if profile_obj is not None:
        profile_df = profile_obj.view().to_pandas()
    else:
        profile_df = pd.DataFrame()
    profile_csv_path = os.path.join(profile_dir, f"profile_{name}.csv")

    # Save the DataFrame to a CSV file locally
    try:
        profile_df.to_csv(profile_csv_path, index=False)

        # Save the local .csv file to MLflow
        mlflow.log_artifact(profile_csv_path, artifact_path="whylogs")

    except Exception as e:
        # If there's an error saving, still try to clean up
        print(f"Warning: Failed to save or log profile: {e}")

    finally:
        # Delete the local file after logging (with error handling)
        try:
            if os.path.exists(profile_csv_path):
                os.remove(profile_csv_path)
        except (FileNotFoundError, PermissionError, OSError) as e:
            # Ignore file deletion errors in tests or when file doesn't exist
            print(f"Warning: Could not delete temporary file {profile_csv_path}: {e}")
