from certain_library.tracking.tracker import tracker
import tempfile
import os
import json
import pandas as pd
from scipy.stats import ks_2samp
from typing import Optional, Dict, Any


def _safe_py(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    # numpy types -> python native
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    return v


def log_drift_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_id: Optional[str] = None,
    model_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute drift (KS test) per numeric column and write artifact under certain/drift_metrics.

    Args:
        train_df: DataFrame with training rows (should not include 'stage').
        test_df: DataFrame with testing rows.
        run_id: optional run id to include in artifact.
        model_info: optional dict with keys experiment_id, deployment_id, model_id.

    Returns:
        A dict summary of the drift results (same structure written to artifact).
    """
    if model_info is None:
        model_info = {
            "experiment_id": None,
            "deployment_id": "not deployed yet",
            "model_id": "not deployed yet",
        }

    # Choose numeric columns present in both frames
    common = set(train_df.columns).intersection(set(test_df.columns))
    # Exclude run_id/stage or other non-feature columns
    ignore = {"run_id", "stage"}
    cols = [c for c in common if c not in ignore]

    results = []
    for c in cols:
        try:
            a = pd.to_numeric(train_df[c], errors="coerce").dropna().values
            b = pd.to_numeric(test_df[c], errors="coerce").dropna().values
            if len(a) < 2 or len(b) < 2:
                # not enough samples to test
                continue
            stat, p = ks_2samp(a, b)
            results.append(
                {
                    "column": c,
                    "key": f"[drift_metrics]{c}",
                    "p_value": float(p),
                    "statistic": float(stat),
                    "n_train": int(len(a)),
                    "n_test": int(len(b)),
                }
            )
        except Exception:
            # skip problematic columns
            continue

    summary = {
        "num_tested": len(results),
        "num_drift": sum(1 for r in results if r.get("p_value", 1) < 0.05),
    }

    artifact = {
        "run_id": run_id,
        "model": model_info,
        "columns": results,
        "summary": summary,
    }

    # write artifact deterministically and upload
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "drift_metrics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)

        tracker.log_artifact(path, artifact_path="drift_metrics")
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass

    # Optionally log simple metrics to MLflow for quick dashboarding
    try:
        for r in results:
            metric_name = str(r.get("key")).replace("[drift_metrics]", "drift.")
            tracker.log_metrics({metric_name: float(r.get("p_value") or 0.0)})
    except Exception:
        # non-fatal
        pass

    return artifact
