"""Privacy and anonymization helpers for tabular datasets."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable, Optional

import pandas as pd

from certain_library.tracking.tracker import tracker


def _resolve_logger(logger):
    """Return a logger-like object that exposes ``log_metrics``."""

    return logger if logger is not None else tracker


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy of ``df`` with missing values preserved."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return df.copy()


def compute_k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str]) -> int:
    """Compute k-anonymity for the provided quasi-identifiers.

    The function groups the input DataFrame by the quasi-identifiers and
    returns the size of the smallest equivalence class. Empty inputs return 0.
    """

    frame = _ensure_dataframe(df)
    if not quasi_identifiers or frame.empty:
        return 0

    valid_columns = [col for col in quasi_identifiers if col in frame.columns]
    if not valid_columns:
        return 0

    grouped = frame.groupby(valid_columns, dropna=False).size()
    if grouped.empty:
        return 0

    return int(grouped.min())


def compute_l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_column: str,
) -> int:
    """Compute l-diversity across quasi-identifier groups.

    Returns the smallest number of distinct sensitive values observed in any
    equivalence class. Empty or invalid inputs return 0.
    """

    frame = _ensure_dataframe(df)
    if not quasi_identifiers or sensitive_column not in frame.columns or frame.empty:
        return 0

    valid_columns = [col for col in quasi_identifiers if col in frame.columns]
    if not valid_columns:
        return 0

    diversity_values: list[int] = []
    for _, group in frame.groupby(valid_columns, dropna=False):
        distinct_sensitive = group[sensitive_column].dropna().nunique()
        diversity_values.append(int(distinct_sensitive))

    if not diversity_values:
        return 0

    return int(min(diversity_values))


def log_privacy_metrics(
    logger,
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_column: Optional[str] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    data_id: Optional[str] = None,
) -> dict[str, float]:
    """Compute and log privacy metrics for a tabular dataset.

    The metrics are logged through the existing tracker interface so they are
    synchronized into the PostgreSQL-backed ``data_metrics`` table by the
    current sync pipeline.
    """

    frame = _ensure_dataframe(df)
    active_logger = _resolve_logger(logger)

    metrics: dict[str, float] = {
        "privacy_k_anonymity": float(
            compute_k_anonymity(frame, quasi_identifiers)
        )
    }

    if sensitive_column:
        metrics["privacy_l_diversity"] = float(
            compute_l_diversity(frame, quasi_identifiers, sensitive_column)
        )

    if epsilon is not None:
        metrics["privacy_dp_epsilon"] = float(epsilon)

    if delta is not None:
        metrics["privacy_dp_delta"] = float(delta)

    if not metrics:
        return {}

    try:
        active_logger.log_metrics(metrics)
    except TypeError:
        # Some callers may pass a minimal logger with a slightly different signature.
        for key, value in metrics.items():
            active_logger.log_metrics({key: value})

    return metrics


def log_privacy_from_data(
    logger,
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_column: Optional[str] = None,
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    data_id: Optional[str] = None,
) -> dict[str, float]:
    """Compute, log, and persist privacy metrics under the CERTAIN artifact tree."""

    metrics = log_privacy_metrics(
        logger=logger,
        df=df,
        quasi_identifiers=quasi_identifiers,
        sensitive_column=sensitive_column,
        epsilon=epsilon,
        delta=delta,
        data_id=data_id,
    )

    active_logger = _resolve_logger(logger)
    artifact_payload = {
        "data_id": data_id,
        "quasi_identifiers": quasi_identifiers,
        "sensitive_column": sensitive_column,
        "privacy_metrics": metrics,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = os.path.join(tmp_dir, "privacy_metrics.json")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact_payload, handle, indent=2, default=str)
        active_logger.log_artifact(artifact_path, artifact_path="log_privacy")

    return metrics
