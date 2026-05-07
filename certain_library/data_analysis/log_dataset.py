import os
from typing import Any, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas

# ---------------------------------------------------------------------------
# Optional framework imports — none are hard dependencies
# ---------------------------------------------------------------------------
try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


SupportedData = Union[
    pd.DataFrame,
    np.ndarray,
    List[Any],  # list of lists or list of dicts
    dict,
    "pl.DataFrame",  # polars (optional)
    "torch.Tensor",  # pytorch tensor (optional)
    "TorchDataLoader",  # pytorch DataLoader (optional)
    "TorchDataset",  # pytorch Dataset (optional)
    "tf.data.Dataset",  # tensorflow dataset (optional)
    "tf.Tensor",  # tensorflow tensor (optional)
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a PyTorch or TensorFlow tensor to a numpy array."""
    if _TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if _TF_AVAILABLE and isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    raise TypeError(f"Cannot convert {type(tensor).__name__} to numpy array.")


def _batch_to_dataframe(batch: Any, columns: Optional[List[str]]) -> pd.DataFrame:
    """
    Convert a single batch (tensor, tuple/list of tensors, or numpy array)
    to a DataFrame.  Tuples are treated as (features, labels) — the label
    tensor is appended as the last column.
    """
    if isinstance(batch, (tuple, list)):
        # e.g. (X_batch, y_batch) from a DataLoader
        parts = []
        for item in batch:
            arr = _tensor_to_numpy(item) if not isinstance(item, np.ndarray) else item
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            parts.append(arr)
        combined = np.hstack(parts)
        return _to_dataframe(combined, columns=columns)
    # single tensor / array
    arr = _tensor_to_numpy(batch) if not isinstance(batch, np.ndarray) else batch
    return _to_dataframe(arr, columns=columns)


# ---------------------------------------------------------------------------
# Internal normaliser
# ---------------------------------------------------------------------------


def _to_dataframe(
    data: SupportedData, columns: Optional[List[str]] = None, n_rows: int = 0
) -> pd.DataFrame:
    """
    Convert any supported data type to a :class:`pandas.DataFrame`.

    Parameters
    ----------
    data : supported type
        The input data. Supported types:

        * :class:`pandas.DataFrame` — returned as-is.
        * :class:`numpy.ndarray` — 1-D arrays become a single column
          ``"value"``; 2-D arrays use ``columns`` if provided, otherwise
          auto-named ``col_0``, ``col_1``, …
        * ``list`` of ``list`` — treated the same as a 2-D numpy array.
        * ``list`` of ``dict`` — each dict becomes one row; keys become
          column names (``columns`` is ignored).
        * ``dict`` — column-oriented mapping; keys become column names
          (``columns`` is ignored).
        * :class:`polars.DataFrame` — converted via ``.to_pandas()``.
        * :class:`torch.Tensor` — converted via ``.detach().cpu().numpy()``.
        * :class:`torch.utils.data.DataLoader` — iterates batches until
          ``n_rows`` samples are collected (or one epoch if ``n_rows=0``).
        * :class:`torch.utils.data.Dataset` — indexes the first ``n_rows``
          items (or the full dataset if ``n_rows=0``).
        * :class:`tf.data.Dataset` — takes batches until ``n_rows`` samples
          are collected (or all batches if ``n_rows=0``).
        * :class:`tf.Tensor` — converted via ``.numpy()``.

    columns : list of str, optional
        Column names to use when converting a numpy array or a list of lists.
        Ignored for dict / list-of-dict / polars inputs.
    n_rows : int, optional
        Maximum number of rows to collect from data loaders / datasets.
        ``0`` means collect everything. Default ``0``.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    TypeError
        If the data type is not supported.
    ValueError
        If ``columns`` length does not match the number of columns in the array.
    """
    # --- pandas ---
    if isinstance(data, pd.DataFrame):
        return data

    # --- polars ---
    if _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
        return data.to_pandas()

    # --- PyTorch Tensor ---
    if _TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        return _to_dataframe(_tensor_to_numpy(data), columns=columns)

    # --- PyTorch DataLoader ---
    if _TORCH_AVAILABLE and isinstance(data, TorchDataLoader):
        frames = []
        collected = 0
        for batch in data:
            df_batch = _batch_to_dataframe(batch, columns=columns)
            frames.append(df_batch)
            collected += len(df_batch)
            if n_rows > 0 and collected >= n_rows:
                break
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return result.head(n_rows) if n_rows > 0 else result

    # --- PyTorch Dataset ---
    if _TORCH_AVAILABLE and isinstance(data, TorchDataset):
        limit = n_rows if n_rows > 0 else len(data)
        frames = []
        for i in range(min(limit, len(data))):
            item = data[i]
            frames.append(_batch_to_dataframe(item, columns=columns))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # --- TensorFlow Dataset (tf.data.Dataset) ---
    if _TF_AVAILABLE and isinstance(data, tf.data.Dataset):
        frames = []
        collected = 0
        for batch in data:
            df_batch = _batch_to_dataframe(batch, columns=columns)
            frames.append(df_batch)
            collected += len(df_batch)
            if n_rows > 0 and collected >= n_rows:
                break
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return result.head(n_rows) if n_rows > 0 else result

    # --- TensorFlow / Keras Tensor ---
    if _TF_AVAILABLE and isinstance(data, tf.Tensor):
        return _to_dataframe(_tensor_to_numpy(data), columns=columns)

    # --- numpy array ---
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.DataFrame(data, columns=columns or ["value"])
        if data.ndim == 2:
            if columns is not None:
                if len(columns) != data.shape[1]:
                    raise ValueError(
                        f"columns length ({len(columns)}) does not match "
                        f"array width ({data.shape[1]})"
                    )
                return pd.DataFrame(data, columns=columns)
            return pd.DataFrame(
                data, columns=[f"col_{i}" for i in range(data.shape[1])]
            )
        raise ValueError(
            f"numpy arrays with ndim={data.ndim} are not supported (max 2-D)"
        )

    # --- list ---
    if isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame()
        first = data[0]
        if isinstance(first, dict):
            return pd.DataFrame(data)
        if isinstance(first, (list, tuple, np.ndarray)):
            arr = np.array(data)
            if columns is not None:
                if len(columns) != arr.shape[1]:
                    raise ValueError(
                        f"columns length ({len(columns)}) does not match "
                        f"data width ({arr.shape[1]})"
                    )
                return pd.DataFrame(arr, columns=columns)
            return pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
        return pd.DataFrame(data, columns=columns or ["value"])

    # --- dict ---
    if isinstance(data, dict):
        return pd.DataFrame(data)

    supported = ["pandas.DataFrame", "numpy.ndarray", "list", "dict"]
    if _POLARS_AVAILABLE:
        supported.append("polars.DataFrame")
    if _TORCH_AVAILABLE:
        supported += ["torch.Tensor", "torch.DataLoader", "torch.Dataset"]
    if _TF_AVAILABLE:
        supported += ["tf.Tensor", "tf.data.Dataset"]

    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        f"Supported types: {', '.join(supported)}."
    )


# ---------------------------------------------------------------------------
# Public logging functions
# ---------------------------------------------------------------------------


def log_dataset(
    data: SupportedData,
    name: str = "dataset",
    output_dir: str = "dataset",
    non_nan: bool = False,
    save_full_dataset: bool = False,
    columns: Optional[List[str]] = None,
) -> None:
    """
    Log a dataset to MLflow.

    Converts the input to a :class:`pandas.DataFrame`, registers it as an
    MLflow dataset input, and saves a CSV artifact for validation.

    Parameters
    ----------
    data : pandas.DataFrame | numpy.ndarray | list | dict | polars.DataFrame | \
torch.Tensor | torch.DataLoader | torch.Dataset | tf.Tensor | tf.data.Dataset
        The dataset to log. All types are normalised to a
        :class:`pandas.DataFrame` internally — see :func:`_to_dataframe`
        for conversion rules.

        For **data loaders** (``torch.DataLoader``, ``torch.Dataset``,
        ``tf.data.Dataset``): when ``save_full_dataset=False`` (default),
        only enough batches to fill **10 rows** are consumed from the loader,
        so the loader is never fully iterated during validation logging.
    name : str, optional
        Name identifier used in the MLflow dataset registry. Default ``"dataset"``.
    output_dir : str, optional
        Temporary directory for the CSV file before it is uploaded to MLflow.
        Default ``"dataset"``.
    non_nan : bool, optional
        If ``True``, raises :exc:`ValueError` when the data contains any NaN
        values. Default ``False``.
    save_full_dataset : bool, optional
        If ``True``, the entire dataset is saved as the CSV artifact (data
        loaders are fully iterated). If ``False`` (default), only the first
        10 rows are saved for quick validation.
    columns : list of str, optional
        Column names — only used when ``data`` is a numpy array or a list of
        lists/tuples. Ignored for all other types.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``data`` is not a supported type.
    ValueError
        If the dataset is empty, contains NaN values (when ``non_nan=True``),
        or ``columns`` length mismatches the data width.
    """
    # For loaders, only pull what we need upfront to avoid a full pass
    n_rows = 0 if save_full_dataset else 10
    df = _to_dataframe(data, columns=columns, n_rows=n_rows)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if non_nan and df.isna().any().any():
        raise ValueError("Input data contains NaN values")

    if df.empty:
        raise ValueError("Input data is empty")

    mlflow_dataset = from_pandas(df, source="logged dataset", name=name)
    mlflow.log_input(mlflow_dataset, context="data_analysis")

    data_to_save = df if save_full_dataset else df.head(10)
    csv_path = os.path.join(output_dir, f"{name}.csv")
    data_to_save.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path=output_dir)

    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)


def log_train_test_dataset(
    train_data: SupportedData,
    test_data: SupportedData,
    output_dir: str = "dataset",
    non_nan: bool = False,
    save_full_dataset: bool = False,
    columns: Optional[List[str]] = None,
) -> None:
    """
    Log training and testing datasets to MLflow.

    Both inputs are normalised to :class:`pandas.DataFrame` before logging.
    One CSV artifact is created per split.

    Parameters
    ----------
    train_data : pandas.DataFrame | numpy.ndarray | list | dict | polars.DataFrame | \
torch.Tensor | torch.DataLoader | torch.Dataset | tf.Tensor | tf.data.Dataset
        The training dataset.
    test_data : pandas.DataFrame | numpy.ndarray | list | dict | polars.DataFrame | \
torch.Tensor | torch.DataLoader | torch.Dataset | tf.Tensor | tf.data.Dataset
        The testing dataset.
    output_dir : str, optional
        Temporary directory for CSV files. Default ``"dataset"``.
    non_nan : bool, optional
        If ``True``, raises :exc:`ValueError` if either split contains NaN
        values. Default ``False``.
    save_full_dataset : bool, optional
        If ``True``, full datasets are saved (data loaders are fully
        iterated). If ``False`` (default), only the first 10 rows of each
        split are saved for validation.
    columns : list of str, optional
        Column names — applied to both splits when converting from numpy
        arrays or lists of lists. Ignored for other types.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If either input is not a supported type.
    ValueError
        If data is empty, contains NaN values (when ``non_nan=True``), or
        train/test column sets differ.
    """
    n_rows = 0 if save_full_dataset else 10
    train_df = _to_dataframe(train_data, columns=columns, n_rows=n_rows)
    test_df = _to_dataframe(test_data, columns=columns, n_rows=n_rows)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if non_nan and train_df.isna().any().any():
        raise ValueError("Training data contains NaN values")
    if non_nan and test_df.isna().any().any():
        raise ValueError("Test data contains NaN values")

    if not train_df.empty and not test_df.empty:
        if set(train_df.columns) != set(test_df.columns):
            raise ValueError("Training and test data have different column sets")

    train_csv_path = os.path.join(output_dir, "X_train.csv")
    test_csv_path = os.path.join(output_dir, "X_test.csv")

    if not train_df.empty:
        dataset = from_pandas(train_df, source="X_train split", name="X_train")
        mlflow.log_input(dataset, context="training")

        train_to_save = train_df if save_full_dataset else train_df.head(10)
        train_to_save.to_csv(train_csv_path, index=False)
        mlflow.log_artifact(train_csv_path, artifact_path="dataset")

    if not test_df.empty:
        dataset_test = from_pandas(test_df, source="X_test split", name="X_test")
        mlflow.log_input(dataset_test, context="testing")

        test_to_save = test_df if save_full_dataset else test_df.head(10)
        test_to_save.to_csv(test_csv_path, index=False)
        mlflow.log_artifact(test_csv_path, artifact_path="dataset")

    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)

    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)
