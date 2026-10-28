from certain_library.tracking.tracker import tracker
import os
import tempfile
import time
import uuid
from typing import List

import pandas as pd


def log_run_logs(
    messages: List[str],
    log_type: str = "stdout",
) -> None:
    """
    Log training / evaluation log messages to MLflow as a CSV artifact.

    Each entry in *messages* becomes one row in a CSV file stored under the
    ``run_logs/`` artifact folder.  The ``data_api`` sync function later reads
    these files and upserts them into the ``runs_logs`` table in
    ``certain_db``.

    Parameters
    ----------
    messages : list of str
        Log lines to persist (e.g. captured stdout, stderr, or custom messages).
    log_type : str, optional
        Category label for the log (e.g. ``"stdout"``, ``"stderr"``,
        ``"warning"``).  Defaults to ``"stdout"``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If *messages* is empty.
    """
    if not messages:
        raise ValueError("messages list must not be empty")

    now = int(time.time())

    df = pd.DataFrame(
        {
            "log_id": [str(uuid.uuid4()) for _ in messages],
            "log_type": log_type,
            "log_message": messages,
            "log_creation_time": now,
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, f"logs_{log_type}.csv")
        df.to_csv(csv_path, index=False)
        tracker.log_artifact(csv_path, artifact_path="run_logs")
