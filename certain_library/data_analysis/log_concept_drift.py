from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

from certain_library.tracking.tracker import tracker


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _accuracy(y_true: Iterable[Any], y_pred: Iterable[Any]) -> float:
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    if len(y_true_list) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true_list, y_pred_list) if t == p)
    return float(correct) / float(len(y_true_list))


def _two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict[str, float]:
    # z-test for difference in proportions (two-sided)
    if n_a == 0 or n_b == 0:
        return {"z": 0.0, "p_value": 1.0}
    p_a = success_a / n_a
    p_b = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0}
    z = (p_b - p_a) / se
    # two-sided p-value from normal approx
    from math import erf

    p_value = 1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    p_value = min(max(0.0, p_value * 2.0), 1.0)
    return {"z": float(z), "p_value": float(p_value)}


def log_concept_drift_from_labels(
    logger: Any,
    y_true_ref: Iterable[Any],
    y_pred_ref: Iterable[Any],
    y_true_new: Iterable[Any],
    y_pred_new: Iterable[Any],
    output_dir: Optional[Path] = None,
    dataset_name: str = "concept_drift",
) -> dict[str, Any]:
    """Simple concept-drift check using accuracy change and two-proportion z-test.

    Parameters
    ----------
    logger: any
        Logger/tracker-like object (expects `log_metrics` and `log_artifact`).
    y_true_ref, y_pred_ref: iterable
        Reference-window labels and predictions.
    y_true_new, y_pred_new: iterable
        New-window labels and predictions.
    output_dir: Path | None
        Directory to write report artifacts.
    dataset_name: str
        Name used for report files and metrics.

    Returns
    -------
    dict: report payload with metrics and test results.
    """
    output_dir = (output_dir or Path("concept_drift_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_ref = list(y_true_ref)
    y_pred_ref = list(y_pred_ref)
    y_true_new = list(y_true_new)
    y_pred_new = list(y_pred_new)

    n_ref = len(y_true_ref)
    n_new = len(y_true_new)

    acc_ref = _accuracy(y_true_ref, y_pred_ref) if n_ref > 0 else 0.0
    acc_new = _accuracy(y_true_new, y_pred_new) if n_new > 0 else 0.0

    success_ref = int(round(acc_ref * n_ref))
    success_new = int(round(acc_new * n_new))

    test = _two_proportion_z_test(success_ref, n_ref, success_new, n_new)

    payload = {
        "dataset_name": dataset_name,
        "n_ref": n_ref,
        "n_new": n_new,
        "accuracy_ref": acc_ref,
        "accuracy_new": acc_new,
        "delta_accuracy": acc_new - acc_ref,
        "z": test["z"],
        "p_value": test["p_value"],
    }

    report_path = output_dir / f"{dataset_name}_concept_drift_report.json"
    _write_json(report_path, payload)

    # Log numeric metrics if logger provided
    try:
        metrics = {
            "concept_drift_accuracy_ref": float(acc_ref),
            "concept_drift_accuracy_new": float(acc_new),
            "concept_drift_delta_accuracy": float(acc_new - acc_ref),
            "concept_drift_p_value": float(test["p_value"]),
        }
        if logger is not None:
            try:
                logger.log_metrics(metrics)
            except Exception:
                # fallback: try tracker
                tracker.log_metrics(metrics)
    except Exception:
        pass

    # Log the artifact via tracker so it appears in MLflow and local mirror
    try:
        tracker.log_artifact(str(report_path), artifact_path="concept_drift")
    except Exception:
        pass

    return payload
