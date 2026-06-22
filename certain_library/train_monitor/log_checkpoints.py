from certain_library.tracking.tracker import tracker
import os
import tempfile
import uuid
import time

from typing import Optional

import pandas as pd


def log_checkpoint(
    checkpoint_name: str,
    checkpoint_location: str,
    checkpoint_id: Optional[str] = None,
) -> None:
    """
    Log a model checkpoint record to MLflow as a CSV artifact.

    Each call appends one checkpoint row to a CSV file stored under the
    ``checkpoints/`` artifact folder.  The ``data_api`` sync function later
    reads all ``checkpoints/*.csv`` files and upserts them into the
    ``checkpoints`` table in ``certain_db``.

    Parameters
    ----------
    checkpoint_name : str
        Human-readable name for the checkpoint (e.g. ``"epoch_10"``).
    checkpoint_location : str
        File-system or remote path where the checkpoint weights are stored.
    checkpoint_id : str, optional
        Unique identifier for this checkpoint.  A random UUID is generated
        when not provided.

    Returns
    -------
    None
    """
    if not checkpoint_name:
        raise ValueError("checkpoint_name must be a non-empty string")
    if not checkpoint_location:
        raise ValueError("checkpoint_location must be a non-empty string")

    record = pd.DataFrame(
        [
            {
                "checkpoint_id": checkpoint_id or str(uuid.uuid4()),
                "checkpoint_name": checkpoint_name,
                "checkpoint_location": checkpoint_location,
                "creation_time": int(time.time()),
            }
        ]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, f"checkpoint_{checkpoint_name}.csv")
        record.to_csv(csv_path, index=False)
        tracker.log_artifact(csv_path, artifact_path="checkpoints")
