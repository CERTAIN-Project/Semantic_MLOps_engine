"""Data quality scoring helpers for compliance workflows."""

from __future__ import annotations

import os
import json
import tempfile
import math
from typing import Any, Optional, Sequence

import pandas as pd

from certain_library.tracking.tracker import tracker
from certain_library.data_analysis.log_dataset import log_dataset, log_train_test_dataset
from certain_library.data_analysis.log_timeseries import timestamp_analysis
from certain_library.data_analysis.log_whylogs import log_whylogs_profile


def _resolve_logger(logger):
    """Return a logger-like object that exposes ``log_metrics``."""

    return logger if logger is not None else tracker


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float when possible, otherwise return default."""

    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bounded_score(value: float) -> float:
    """Clamp a score to the inclusive [0.0, 1.0] range."""

    return max(0.0, min(1.0, float(value)))


def _normalize_range_spec(range_spec: Any) -> tuple[Optional[float], Optional[float]]:
    """Normalize a range specification into (lower, upper) bounds."""

    if isinstance(range_spec, dict):
        lower = range_spec.get("min", range_spec.get("low", range_spec.get("lower")))
        upper = range_spec.get("max", range_spec.get("high", range_spec.get("upper")))
        return (
            _safe_float(lower, default=float("nan")) if lower is not None else None,
            _safe_float(upper, default=float("nan")) if upper is not None else None,
        )

    if isinstance(range_spec, (list, tuple)) and len(range_spec) >= 2:
        lower, upper = range_spec[0], range_spec[1]
        return (
            _safe_float(lower, default=float("nan")) if lower is not None else None,
            _safe_float(upper, default=float("nan")) if upper is not None else None,
        )

    return None, None


def _compute_range_consistency(
    data: pd.DataFrame,
    consistency_ranges: dict[str, Any],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute how often values stay within the provided ranges.

    The score is the mean of the per-column in-range ratios for all columns that
    have a valid range specification and exist in the DataFrame.
    """

    if not isinstance(data, pd.DataFrame) or data.empty or not consistency_ranges:
        return 1.0, {}

    per_column: dict[str, dict[str, float]] = {}
    ratios: list[float] = []

    for column_name, range_spec in consistency_ranges.items():
        if column_name not in data.columns:
            continue

        lower, upper = _normalize_range_spec(range_spec)
        if lower is None and upper is None:
            continue

        series = pd.to_numeric(data[column_name], errors="coerce")
        in_range = pd.Series(True, index=series.index)

        if lower is not None and not math.isnan(lower):
            in_range &= series >= lower
        if upper is not None and not math.isnan(upper):
            in_range &= series <= upper

        ratio = float(in_range.fillna(False).mean())
        ratios.append(ratio)
        per_column[column_name] = {
            "lower": float(lower) if lower is not None and not math.isnan(lower) else float("nan"),
            "upper": float(upper) if upper is not None and not math.isnan(upper) else float("nan"),
            "in_range_ratio": ratio,
        }

    if not ratios:
        return 1.0, {}

    return float(sum(ratios) / len(ratios)), per_column


def compute_4d_quality_index(profile_data: dict) -> dict:
    """Compute a 4D data quality index from profiling metadata.

    The function is intentionally permissive about the input keys so it can
    consume either WhyLogs-style aggregates or lightweight metadata dicts.
    """

    profile_data = profile_data or {}

    mean_null_ratio = profile_data.get("mean_null_ratio")
    if mean_null_ratio is None:
        null_count = _safe_float(profile_data.get("null_count"), 0.0)
        total_records = max(_safe_float(profile_data.get("total_records"), 0.0), 0.0)
        mean_null_ratio = (null_count / total_records) if total_records > 0 else 0.0
    completeness = _bounded_score(1.0 - _safe_float(mean_null_ratio, 0.0))

    total_records = max(_safe_float(profile_data.get("total_records"), 0.0), 0.0)
    outlier_count = _safe_float(profile_data.get("outlier_count"), 0.0)
    valid_count = profile_data.get("valid_count")
    if valid_count is not None and total_records > 0:
        accuracy = _bounded_score(_safe_float(valid_count) / total_records)
    elif total_records > 0:
        accuracy = _bounded_score((total_records - outlier_count) / total_records)
    else:
        accuracy = _bounded_score(_safe_float(profile_data.get("accuracy"), 1.0))

    schema_violation_ratio = profile_data.get("schema_violation_ratio")
    if schema_violation_ratio is None:
        schema_violations = _safe_float(profile_data.get("schema_violations"), 0.0)
        if total_records > 0:
            schema_violation_ratio = schema_violations / total_records
        else:
            schema_violation_ratio = 0.0
    schema_consistency = _bounded_score(1.0 - _safe_float(schema_violation_ratio, 0.0))

    range_consistency_ratio = profile_data.get("range_consistency_ratio")
    if range_consistency_ratio is None:
        consistency = schema_consistency
    else:
        consistency = _bounded_score(
            (schema_consistency + _safe_float(range_consistency_ratio, 0.0)) / 2.0
        )

    timeliness = profile_data.get("timeliness")
    if timeliness is not None:
        timeliness_score = _bounded_score(_safe_float(timeliness, 1.0))
    else:
        gap_ratio = profile_data.get("gap_ratio")
        if gap_ratio is not None:
            timeliness_score = _bounded_score(1.0 - _safe_float(gap_ratio, 0.0))
        else:
            ingest_latency = profile_data.get("ingest_latency")
            if ingest_latency is None:
                ingest_latency = profile_data.get("latency_seconds")
            if ingest_latency is None:
                event_time = profile_data.get("t_event") or profile_data.get("event_time")
                ingest_time = profile_data.get("t_ingest") or profile_data.get("ingest_time")
                try:
                    if event_time is not None and ingest_time is not None:
                        ingest_latency = max(
                            0.0,
                            _safe_float(ingest_time) - _safe_float(event_time),
                        )
                except Exception:
                    ingest_latency = None

            if ingest_latency is None:
                timeliness_score = 1.0
            else:
                lambda_value = _safe_float(profile_data.get("timeliness_lambda"), 1.0)
                timeliness_score = math.exp(-max(0.0, lambda_value) * max(0.0, _safe_float(ingest_latency)))
                timeliness_score = _bounded_score(timeliness_score)

    scores = {
        "completeness": completeness,
        "accuracy": accuracy,
        "consistency": consistency,
        "timeliness": timeliness_score,
    }
    scores["composite_score"] = _bounded_score(sum(scores.values()) / 4.0)
    return scores


def log_quality_from_data(
    logger,
    data: pd.DataFrame,
    train_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    train_timestamps: Optional[pd.Series] = None,
    test_timestamps: Optional[pd.Series] = None,
    name: str = "quality",
    output_dir: str = "data_quality",
    log_whylogs: bool = True,
    log_dataset_artifacts: bool = True,
    save_full_dataset: bool = False,
    columns: Optional[Sequence[str]] = None,
    non_nan: bool = False,
    consistency_ranges: Optional[dict[str, Any]] = None,
    data_id: Optional[str] = None,
) -> dict[str, float]:
    """Log quality artifacts and compute the 4D quality index from raw data.

    This helper reuses the existing dataset, WhyLogs, and timeseries utilities
    directly so callers do not need to prepare a separate ``profile_data``
    dictionary.
    """

    active_logger = _resolve_logger(logger)

    if log_dataset_artifacts:
        try:
            if train_data is not None and test_data is not None:
                log_train_test_dataset(
                    train_data,
                    test_data,
                    output_dir=output_dir,
                    non_nan=non_nan,
                    save_full_dataset=save_full_dataset,
                    columns=list(columns) if columns is not None else None,
                )
            else:
                log_dataset(
                    data,
                    name=name,
                    output_dir=output_dir,
                    non_nan=non_nan,
                    save_full_dataset=save_full_dataset,
                    columns=list(columns) if columns is not None else None,
                )
        except Exception:
            pass

    if log_whylogs:
        try:
            log_whylogs_profile(data, name=name)
        except Exception:
            pass

    if train_timestamps is not None and test_timestamps is not None:
        try:
            timestamp_analysis(train_timestamps=train_timestamps, test_timestamps=test_timestamps)
        except Exception:
            pass

    profile_data = {}
    range_details: dict[str, dict[str, float]] = {}
    if isinstance(data, pd.DataFrame) and not data.empty:
        total_records = int(len(data))
        null_ratio_series = data.isna().mean(numeric_only=False)
        mean_null_ratio = float(null_ratio_series.mean()) if not null_ratio_series.empty else 0.0

        numeric_df = data.select_dtypes(include="number")
        if not numeric_df.empty:
            z_scores = ((numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)).abs()
            outlier_count = int((z_scores > 3.0).any(axis=1).sum())
        else:
            outlier_count = 0

        schema_violations = 0
        for column_name in data.columns:
            column_values = data[column_name]
            if column_values.isna().all():
                schema_violations += 1

        timeliness = None
        timestamp_columns = [
            col for col in data.columns if "timestamp" in str(col).lower() or "time" in str(col).lower()
        ]
        if timestamp_columns:
            ts_series = pd.to_datetime(data[timestamp_columns[0]], errors="coerce").dropna()
            if len(ts_series) >= 2:
                gaps = ts_series.sort_values().diff().dropna().dt.total_seconds().abs()
                if not gaps.empty:
                    timeliness = 1.0 / (1.0 + float(gaps.mean()))

        range_consistency_ratio = None
        if consistency_ranges:
            range_consistency_ratio, range_details = _compute_range_consistency(
                data,
                consistency_ranges,
            )

        profile_data = {
            "total_records": total_records,
            "mean_null_ratio": mean_null_ratio,
            "outlier_count": outlier_count,
            "schema_violations": schema_violations,
            "timeliness": timeliness,
            "range_consistency_ratio": range_consistency_ratio,
        }

    quality_scores = compute_4d_quality_index(profile_data)
    log_4d_quality_metrics(active_logger, quality_scores, data_id=data_id)

    artifact_payload = {
        "data_id": data_id,
        "quality_scores": quality_scores,
        "consistency_ranges": consistency_ranges,
        "consistency_range_details": range_details,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = os.path.join(tmp_dir, "quality_metrics.json")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact_payload, handle, indent=2, default=str)
        active_logger.log_artifact(artifact_path, artifact_path="log_quality")
    return quality_scores


def log_4d_quality_metrics(
    logger,
    quality_scores: dict,
    data_id: Optional[str] = None,
) -> dict[str, float]:
    """Log 4D quality scores using the existing MLflow-backed logger."""

    active_logger = _resolve_logger(logger)
    scores = {
        "quality_completeness": _bounded_score(
            _safe_float(quality_scores.get("completeness"), 0.0)
        ),
        "quality_accuracy": _bounded_score(
            _safe_float(quality_scores.get("accuracy"), 0.0)
        ),
        "quality_consistency": _bounded_score(
            _safe_float(quality_scores.get("consistency"), 0.0)
        ),
        "quality_timeliness": _bounded_score(
            _safe_float(quality_scores.get("timeliness"), 0.0)
        ),
    }
    scores["quality_composite_score"] = _bounded_score(
        sum(scores.values()) / 4.0
    )

    try:
        active_logger.log_metrics(scores)
    except TypeError:
        for key, value in scores.items():
            active_logger.log_metrics({key: value})

    return scores
