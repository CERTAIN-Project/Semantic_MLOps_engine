from certain_library.tracking.tracker import tracker
import os
import tempfile
import time
from typing import Any

import pandas as pd


def log_weight_distribution(
    model: Any,
    step: int = 0,
    stage: str = "train",
) -> None:
    """
    Log the weight / parameter distribution of a model to MLflow as a CSV artifact.

    For every named parameter (or layer) the function computes the mean and
    standard deviation of the weights and saves the results under the
    ``weight_distribution/`` artifact folder.  The ``data_api`` sync function
    later reads these CSV files and upserts them into the
    ``weight_distribution`` table in ``certain_db``.

    Parameters
    ----------
    model : Any
        A trained model.  Supported types:

        * **PyTorch** ``nn.Module`` — iterates over ``model.named_parameters()``.
        * **Keras / TensorFlow** model — iterates over ``model.weights``.
        * **scikit-learn** estimator — uses ``model.coef_`` / ``model.feature_importances_``
          when available.

        Pass any object with a ``get_weights()`` or ``named_parameters()``
        interface; objects that cannot be introspected are stored as a single
        ``unknown_layer`` row with ``mean=0`` and ``std=0``.
    step : int, optional
        Training step / epoch number (default ``0``).
    stage : str, optional
        Pipeline stage label stored in the artifact (default ``"train"``).

    Returns
    -------
    None
    """
    rows = []
    now = int(time.time())

    # --- PyTorch ---
    try:
        import torch.nn as nn  # type: ignore

        if isinstance(model, nn.Module):
            for name, param in model.named_parameters():
                data = param.detach().float().cpu().numpy().flatten()
                rows.append(
                    {
                        "layer_name": name,
                        "mean": float(data.mean()),
                        "std": float(data.std()),
                        "step": step,
                        "stage": stage,
                        "is_NaN": bool(pd.isna(float(data.mean()))),
                        "timestamp": now,
                    }
                )
    except ImportError:
        pass

    # --- Keras / TensorFlow ---
    if not rows:
        try:
            import numpy as np  # type: ignore

            if hasattr(model, "weights"):  # Keras
                for weight in model.weights:  # type: ignore[union-attr]
                    data = weight.numpy().flatten()
                    rows.append(
                        {
                            "layer_name": weight.name,
                            "mean": float(np.mean(data)),
                            "std": float(np.std(data)),
                            "step": step,
                            "stage": stage,
                            "is_NaN": bool(pd.isna(float(np.mean(data)))),
                            "timestamp": now,
                        }
                    )
        except ImportError:
            pass

    # --- scikit-learn ---
    if not rows:
        import numpy as np

        coef = None
        if hasattr(model, "coef_"):
            coef = model.coef_.flatten()  # type: ignore[union-attr]
            name = "coef_"
        elif hasattr(model, "feature_importances_"):
            coef = model.feature_importances_.flatten()  # type: ignore[union-attr]
            name = "feature_importances_"

        if coef is not None:
            rows.append(
                {
                    "layer_name": name,
                    "mean": float(np.mean(coef)),
                    "std": float(np.std(coef)),
                    "step": step,
                    "stage": stage,
                    "is_NaN": bool(pd.isna(float(np.mean(coef)))),
                    "timestamp": now,
                }
            )

    if not rows:
        rows.append(
            {
                "layer_name": "unknown_layer",
                "mean": 0.0,
                "std": 0.0,
                "step": step,
                "stage": stage,
                "is_NaN": False,
                "timestamp": now,
            }
        )

    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, f"weights_{stage}.csv")
        df.to_csv(csv_path, index=False)
        tracker.log_artifact(csv_path, artifact_path="weight_distribution")
