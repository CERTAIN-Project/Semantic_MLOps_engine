from certain_library.tracking.tracker import tracker
import os
import tempfile
import time
from typing import List, Any

import pandas as pd


def log_examples(
    inputs: List[Any],
    predictions: List[Any],
    ground_truths: List[Any],
    step: int = 0,
    stage: str = "inference",
) -> None:
    """
    Log sample model predictions to MLflow as a CSV artifact.

    Each row represents one prediction example and is stored under the
    ``examples/`` artifact folder.  The ``data_api`` sync function later
    reads these CSV files and upserts them into the ``examples`` table in
    ``certain_db``.

    Parameters
    ----------
    inputs : list
        Input feature vectors or raw inputs (converted to strings for storage).
    predictions : list
        Predicted values / classes returned by the model.
    ground_truths : list
        Corresponding ground-truth / label values.
    step : int, optional
        Training / evaluation step or epoch number (default ``0``).
    stage : str, optional
        Pipeline stage label, e.g. ``"train"``, ``"validation"``,
        ``"inference"`` (default ``"inference"``).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the three lists do not have the same length or are empty.
    """
    if not inputs:
        raise ValueError("inputs list must not be empty")
    if len(inputs) != len(predictions) or len(inputs) != len(ground_truths):
        raise ValueError(
            "inputs, predictions, and ground_truths must have the same length"
        )

    now = int(time.time())

    df = pd.DataFrame(
        {
            "input": [str(x) for x in inputs],
            "prediction": [str(p) for p in predictions],
            "ground_truth": [str(g) for g in ground_truths],
            "step": step,
            "stage": stage,
            "timestamp": now,
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, f"examples_{stage}.csv")
        df.to_csv(csv_path, index=False)
        tracker.log_artifact(csv_path, artifact_path="examples")
