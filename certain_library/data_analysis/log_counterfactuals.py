"""Counterfactual and recourse quality helpers."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd

from certain_library.tracking.tracker import tracker


def _resolve_logger(logger):
    """Return a logger-like object that exposes ``log_metrics``."""

    return logger if logger is not None else tracker


def _to_dataframe(value) -> pd.DataFrame:
    """Normalize arrays or tabular inputs into a DataFrame."""

    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame().T
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return pd.DataFrame(value, columns=["value"])
        return pd.DataFrame(value)
    return pd.DataFrame(value)


def _aligned_frames(original_df, counterfactual_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two inputs on a common set of columns and rows."""

    original = _to_dataframe(original_df)
    counterfactual = _to_dataframe(counterfactual_df)

    if original.empty or counterfactual.empty:
        return original.iloc[0:0], counterfactual.iloc[0:0]

    shared_columns = [col for col in original.columns if col in counterfactual.columns]
    if shared_columns:
        original = original[shared_columns]
        counterfactual = counterfactual[shared_columns]

    row_count = min(len(original), len(counterfactual))
    return original.iloc[:row_count].reset_index(drop=True), counterfactual.iloc[:row_count].reset_index(drop=True)


def generate_price_counterfactual(
    original_row,
    price_model,
    feature_columns: list[str],
    target_price: float = 60.0,
    intervention_grid: Optional[dict[str, list[float]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, dict[str, Any]]:
    """Generate a model-based price counterfactual by searching a discrete grid.

    The helper keeps the observation fixed except for the intervention features,
    predicts the original and modified price with the provided model, and returns
    the smallest intervention that reaches the requested target price if one is
    found.
    """

    original = _to_dataframe(original_row)
    if original.empty:
        return original, original, 0.0, 0.0, {"found": False, "reason": "empty_input"}

    original_features = original[feature_columns].copy()
    original_prediction = float(price_model.predict(original_features)[0])

    default_grid = {
        "DE_wind_onshore_generation_actual": [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30],
        "DE_wind_offshore_generation_actual": [1.00, 1.05, 1.10, 1.15, 1.20],
    }
    grid = intervention_grid or default_grid

    best_candidate = None
    best_prediction = None
    best_distance = float("inf")
    best_metadata: dict[str, Any] = {"found": False, "reason": "threshold_not_reached"}

    for onshore_multiplier in grid.get("DE_wind_onshore_generation_actual", [1.0]):
        for offshore_multiplier in grid.get("DE_wind_offshore_generation_actual", [1.0]):
            candidate = original.copy()

            if "DE_wind_onshore_generation_actual" in candidate.columns:
                candidate.loc[:, "DE_wind_onshore_generation_actual"] = (
                    candidate["DE_wind_onshore_generation_actual"] * onshore_multiplier
                )

            if "DE_wind_offshore_generation_actual" in candidate.columns:
                candidate.loc[:, "DE_wind_offshore_generation_actual"] = (
                    candidate["DE_wind_offshore_generation_actual"] * offshore_multiplier
                )

            candidate_prediction = float(price_model.predict(candidate[feature_columns])[0])

            delta_onshore = 0.0
            if "DE_wind_onshore_generation_actual" in candidate.columns:
                original_value = float(original.iloc[0]["DE_wind_onshore_generation_actual"])
                candidate_value = float(candidate.iloc[0]["DE_wind_onshore_generation_actual"])
                delta_onshore = abs(candidate_value - original_value) / max(abs(original_value), 1.0)

            delta_offshore = 0.0
            if "DE_wind_offshore_generation_actual" in candidate.columns:
                original_value = float(original.iloc[0]["DE_wind_offshore_generation_actual"])
                candidate_value = float(candidate.iloc[0]["DE_wind_offshore_generation_actual"])
                delta_offshore = abs(candidate_value - original_value) / max(abs(original_value), 1.0)

            distance = delta_onshore + delta_offshore

            if distance <= 0.0:
                continue

            if candidate_prediction <= target_price and distance < best_distance:
                best_candidate = candidate
                best_prediction = candidate_prediction
                best_distance = distance
                best_metadata = {
                    "found": True,
                    "reason": "threshold_reached",
                    "DE_wind_onshore_generation_actual_multiplier": onshore_multiplier,
                    "DE_wind_offshore_generation_actual_multiplier": offshore_multiplier,
                    "distance": distance,
                    "target_price": target_price,
                }

    if best_candidate is None:
        lowest_price = float("inf")
        for onshore_multiplier in grid.get("DE_wind_onshore_generation_actual", [1.0]):
            for offshore_multiplier in grid.get("DE_wind_offshore_generation_actual", [1.0]):
                candidate = original.copy()

                if "DE_wind_onshore_generation_actual" in candidate.columns:
                    candidate.loc[:, "DE_wind_onshore_generation_actual"] = (
                        candidate["DE_wind_onshore_generation_actual"] * onshore_multiplier
                    )

                if "DE_wind_offshore_generation_actual" in candidate.columns:
                    candidate.loc[:, "DE_wind_offshore_generation_actual"] = (
                        candidate["DE_wind_offshore_generation_actual"] * offshore_multiplier
                    )

                original_values = original.iloc[0]
                delta_onshore = 0.0
                if "DE_wind_onshore_generation_actual" in candidate.columns:
                    candidate_value = float(candidate.iloc[0]["DE_wind_onshore_generation_actual"])
                    original_value = float(original_values["DE_wind_onshore_generation_actual"])
                    delta_onshore = abs(candidate_value - original_value) / max(abs(original_value), 1.0)

                delta_offshore = 0.0
                if "DE_wind_offshore_generation_actual" in candidate.columns:
                    candidate_value = float(candidate.iloc[0]["DE_wind_offshore_generation_actual"])
                    original_value = float(original_values["DE_wind_offshore_generation_actual"])
                    delta_offshore = abs(candidate_value - original_value) / max(abs(original_value), 1.0)

                distance = delta_onshore + delta_offshore
                if distance <= 0.0:
                    continue

                candidate_prediction = float(price_model.predict(candidate[feature_columns])[0])
                if candidate_prediction < lowest_price:
                    lowest_price = candidate_prediction
                    best_candidate = candidate
                    best_prediction = candidate_prediction
                    best_metadata = {
                        "found": False,
                        "reason": "fallback_lowest_prediction",
                        "DE_wind_onshore_generation_actual_multiplier": onshore_multiplier,
                        "DE_wind_offshore_generation_actual_multiplier": offshore_multiplier,
                        "distance": 0.0,
                        "target_price": target_price,
                    }

    if best_candidate is None or best_prediction is None:
        raise ValueError(
            "No valid counterfactual found: all candidates were no-ops or failed the target constraint."
        )

    original_log = original.copy()
    counterfactual_log = best_candidate.copy()
    original_log["DE_price_day_ahead"] = original_prediction
    counterfactual_log["DE_price_day_ahead"] = float(best_prediction)

    return (
        original_log,
        counterfactual_log,
        original_prediction,
        float(best_prediction),
        best_metadata,
    )


def evaluate_counterfactuals(
    original_df,
    counterfactual_df,
    protected_columns: list[str] | None = None,
) -> dict:
    """Evaluate contrastive recourse quality between original and counterfactual data."""

    original, counterfactual = _aligned_frames(original_df, counterfactual_df)
    if original.empty or counterfactual.empty:
        return {
            "avg_l1_distance": 0.0,
            "avg_l2_distance": 0.0,
            "sparsity_l0": 0.0,
            "immutable_violations": 0.0,
        }

    original_numeric = original.apply(pd.to_numeric, errors="coerce")
    counterfactual_numeric = counterfactual.apply(pd.to_numeric, errors="coerce")
    diff = (counterfactual_numeric - original_numeric).fillna(0.0)

    abs_diff = diff.abs().to_numpy(dtype=float)
    avg_l1_distance = float(abs_diff.mean()) if abs_diff.size else 0.0
    avg_l2_distance = float(np.linalg.norm(diff.to_numpy(dtype=float), axis=1).mean()) if len(diff) else 0.0
    sparsity_l0 = float((abs_diff > 0).sum(axis=1).mean()) if abs_diff.size else 0.0

    immutable_violations = 0
    if protected_columns:
        protected = [col for col in protected_columns if col in original.columns and col in counterfactual.columns]
        if protected:
            orig_protected = original[protected].astype(object)
            cf_protected = counterfactual[protected].astype(object)
            immutable_violations = int((orig_protected.ne(cf_protected)).any(axis=1).sum())

    return {
        "avg_l1_distance": avg_l1_distance,
        "avg_l2_distance": avg_l2_distance,
        "sparsity_l0": sparsity_l0,
        "immutable_violations": float(immutable_violations),
    }


def log_counterfactual_metrics(
    logger,
    metrics: dict,
    data_id: Optional[str] = None,
) -> dict[str, float]:
    """Log counterfactual evaluation metrics via the existing tracker."""

    active_logger = _resolve_logger(logger)
    logged_metrics = {
        "counterfactual_avg_l1_distance": float(
            metrics.get("avg_l1_distance", 0.0)
        ),
        "counterfactual_avg_l2_distance": float(
            metrics.get("avg_l2_distance", 0.0)
        ),
        "counterfactual_sparsity_l0": float(metrics.get("sparsity_l0", 0.0)),
        "counterfactual_immutable_violations": float(
            metrics.get("immutable_violations", 0.0)
        ),
    }

    try:
        active_logger.log_metrics(logged_metrics)
    except TypeError:
        for key, value in logged_metrics.items():
            active_logger.log_metrics({key: value})

    return logged_metrics


def log_counterfactual_from_data(
    logger,
    original_df,
    counterfactual_df,
    protected_columns: list[str] | None = None,
    data_id: Optional[str] = None,
) -> dict[str, float]:
    """Compute, log, and persist counterfactual metrics under CERTAIN artifacts."""

    metrics = evaluate_counterfactuals(
        original_df=original_df,
        counterfactual_df=counterfactual_df,
        protected_columns=protected_columns,
    )
    logged_metrics = log_counterfactual_metrics(
        logger=logger,
        metrics=metrics,
        data_id=data_id,
    )

    active_logger = _resolve_logger(logger)
    artifact_payload = {
        "data_id": data_id,
        "protected_columns": protected_columns,
        "counterfactual_metrics": logged_metrics,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = os.path.join(tmp_dir, "counterfactual_metrics.json")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact_payload, handle, indent=2, default=str)
        active_logger.log_artifact(artifact_path, artifact_path="log_counterfactuals")

    return logged_metrics
