#!/usr/bin/env python3
"""End-to-end NVCR -> CERTAIN demographics-bias integration script.

This version is structured as a standalone workflow, following the same
style as the other Docker examples in this repository:

1. generate deterministic demo data in the NVCR repo
2. adapt the public payload + hidden annotations into CERTAIN's schema
3. run the demographic-bias logger and validate the produced artifacts

The script is fully offline and does not depend on the UI or OpenAI.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import types

import pandas as pd

from certain_library.data_analysis.log_dataset import log_dataset
from certain_library.git_tracking import log_git_metadata
from certain_library.log_basic.log_params import log_params
from certain_library.train_monitor.log_metrics import (
    log_metrics,
    log_resources,
    log_search_space,
)
from certain_library.metadata.artifact_metadata import (
    collect_runtime_environment,
    save_runtime_env_as_artifact,
)
from certain_library.resource_monitor.resource import start_tracker, stop_tracker
from certain_library.tracking.tracker import tracker


PROJECT_ROOT = Path(__file__).resolve().parent
NVCR_REPO_CANDIDATES = [
    Path("/app/Synthetic-Data-Generation-Component"),
    PROJECT_ROOT.parent.parent / "Synthetic-Data-Generation-Component",
]
CERTAIN_MODULE_CANDIDATES = [
    Path("/app/certain_library/data_analysis/log_demographic_bias.py"),
    PROJECT_ROOT.parent / "certain_library" / "data_analysis" / "log_demographic_bias.py",
    PROJECT_ROOT.parent.parent / "Semantic_MLOps_engine" / "certain_library" / "data_analysis" / "log_demographic_bias.py",
]


@dataclass(frozen=True)
class DemoBatch:
    session_dir: Path
    clean_payload: dict[str, Any]
    dirty_payload: dict[str, Any]
    clean_annotations: list[dict[str, Any]]
    dirty_annotations: list[dict[str, Any]]


class CaptureLogger:
    def __init__(self) -> None:
        self.metrics_calls: list[dict[str, float]] = []
        self.artifact_calls: list[tuple[str, str | None]] = []

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics_calls.append(dict(metrics))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifact_calls.append((path, artifact_path))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_nvcr_repo() -> Path:
    for candidate in NVCR_REPO_CANDIDATES:
        if (candidate / "src" / "create_demo_bias_datasets.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NVCR repository root.")


def _load_demographic_bias_module():
    module_path = next((candidate for candidate in CERTAIN_MODULE_CANDIDATES if candidate.exists()), None)
    if module_path is None:
        raise FileNotFoundError(
            "Could not locate CERTAIN's data_analysis/log_demographic_bias.py module."
        )

    certain_pkg = types.ModuleType("certain_library")
    certain_pkg.__path__ = []  # type: ignore[attr-defined]
    tracking_pkg = types.ModuleType("certain_library.tracking")
    tracking_pkg.__path__ = []  # type: ignore[attr-defined]

    class _StubTracker:
        def log_metrics(self, *args: Any, **kwargs: Any) -> None:
            return None

        def log_artifact(self, *args: Any, **kwargs: Any) -> None:
            return None

    tracker_mod = types.ModuleType("certain_library.tracking.tracker")
    setattr(tracker_mod, "tracker", _StubTracker())

    sys.modules.setdefault("certain_library", certain_pkg)
    sys.modules.setdefault("certain_library.tracking", tracking_pkg)
    sys.modules.setdefault("certain_library.tracking.tracker", tracker_mod)

    spec = importlib.util.spec_from_file_location(
        "certain_library.data_analysis.log_demographic_bias",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CERTAIN module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DEMO_BIAS_MODULE = _load_demographic_bias_module()
compute_age_bucket = _DEMO_BIAS_MODULE.compute_age_bucket
log_demographic_bias_from_data = _DEMO_BIAS_MODULE.log_demographic_bias_from_data


def _create_demo_batch(
    batch_id: int = 7,
    tracking_run_id: str | None = None,
    tracking_experiment_id: str | None = None,
) -> DemoBatch:
    nvcr_repo = _find_nvcr_repo()
    child_env = os.environ.copy()

    if tracking_run_id is not None:
        child_env["CERTAIN_PARENT_RUN_ID"] = tracking_run_id
        child_env["MLFLOW_RUN_ID"] = tracking_run_id

    if tracking_experiment_id is not None:
        child_env["CERTAIN_PARENT_EXPERIMENT_ID"] = tracking_experiment_id
        child_env["MLFLOW_EXPERIMENT_ID"] = tracking_experiment_id

    with tempfile.TemporaryDirectory(prefix="it_complete_session_") as session_root:
        session_dir = Path(session_root)
        result = subprocess.run(
            [
                sys.executable,
                str(nvcr_repo / "src" / "create_demo_bias_datasets.py"),
                "--batch",
                str(batch_id),
                "--output-dir",
                str(session_dir),
            ],
            # nvcr_repo is mounted read-only; MLflow's SqlAlchemyStore tries to
            # mkdir a default local artifact root relative to cwd, so run from
            # the writable session_dir instead.
            cwd=session_dir,
            env=child_env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"NVCR CLI failed with exit code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        annotations = _load_json(session_dir / f"scenario_annotations_batch_{batch_id}.json")
        return DemoBatch(
            session_dir=session_dir,
            clean_payload=_load_json(session_dir / f"simulation_output_clean_{batch_id}.json"),
            dirty_payload=_load_json(session_dir / f"simulation_output_dirty_{batch_id}.json"),
            clean_annotations=list(annotations.get("Clean", [])),
            dirty_annotations=list(annotations.get("Dirty", [])),
        )


def _build_certain_frame(payload: dict[str, Any], annotations: list[dict[str, Any]]) -> pd.DataFrame:
    annotation_by_id = {
        str(item.get("Id")): item.get("InternalMetadata") or {}
        for item in annotations
        if item.get("Id") is not None
    }

    rows: list[dict[str, Any]] = []
    for record in payload.get("Dataset", []):
        metadata = record.get("Metadata") or {}
        internal = annotation_by_id.get(str(record.get("Id")), {})
        rows.append(
            {
                "record_id": record.get("Id"),
                "Text": record.get("Text"),
                "FirstName": metadata.get("FirstName"),
                "LastName": metadata.get("LastName"),
                "Age": metadata.get("Age"),
                "DateOfBirth": metadata.get("DateOfBirth"),
                "Nationality": metadata.get("Nationality"),
                "Background": internal.get("Background"),
                "Opinion": internal.get("Opinion"),
                "AnswerQuality": internal.get("AnswerQuality"),
                "BiasGroup": internal.get("BiasGroup"),
                "AgeBucket": compute_age_bucket(metadata.get("Age")),
                "text_length": len(str(record.get("Text") or "")),
                "metadata_missing_count": 0,
                "metadata_completeness": 1.0,
                "is_anonymous": 0,
            }
        )

    return pd.DataFrame(rows)


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame[column].fillna("unknown").astype(str)
    total = len(values) or 1
    counts = Counter(values)
    return {key: count / total for key, count in sorted(counts.items())}


def _validate_demo_payloads(demo_batch: DemoBatch) -> None:
    clean_records = demo_batch.clean_payload["Dataset"]
    dirty_records = demo_batch.dirty_payload["Dataset"]

    assert clean_records and dirty_records
    assert set(clean_records[0]["Metadata"]) == {
        "FirstName",
        "LastName",
        "Age",
        "DateOfBirth",
        "Nationality",
    }
    assert set(dirty_records[0]["Metadata"]) == {
        "FirstName",
        "LastName",
        "Age",
        "DateOfBirth",
        "Nationality",
    }
    assert set(demo_batch.clean_annotations[0]["InternalMetadata"]) == {
        "Background",
        "Opinion",
        "AnswerQuality",
        "BiasGroup",
    }
    assert set(demo_batch.dirty_annotations[0]["InternalMetadata"]) == {
        "Background",
        "Opinion",
        "AnswerQuality",
        "BiasGroup",
    }

    clean_nationality = Counter(record["Metadata"]["Nationality"] for record in clean_records)
    dirty_nationality = Counter(record["Metadata"]["Nationality"] for record in dirty_records)
    clean_age = Counter(compute_age_bucket(record["Metadata"]["Age"]) for record in clean_records)
    dirty_age = Counter(compute_age_bucket(record["Metadata"]["Age"]) for record in dirty_records)

    assert dirty_nationality["Greece"] > clean_nationality["Greece"]
    assert dirty_nationality["Bulgaria"] > clean_nationality["Bulgaria"]
    assert dirty_age["18-29"] > clean_age["18-29"]
    assert dirty_age["45-59"] >= clean_age["45-59"]


def _validate_certain_adapter(demo_batch: DemoBatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_frame = _build_certain_frame(demo_batch.clean_payload, demo_batch.clean_annotations)
    dirty_frame = _build_certain_frame(demo_batch.dirty_payload, demo_batch.dirty_annotations)

    expected_columns = {
        "record_id",
        "Text",
        "FirstName",
        "LastName",
        "Age",
        "DateOfBirth",
        "Nationality",
        "Background",
        "Opinion",
        "AnswerQuality",
        "BiasGroup",
        "AgeBucket",
    }

    assert expected_columns <= set(clean_frame.columns)
    assert expected_columns <= set(dirty_frame.columns)
    assert clean_frame["AgeBucket"].isin({"18-29", "30-44", "45-59", "60+"}).all()
    assert dirty_frame["AgeBucket"].isin({"18-29", "30-44", "45-59", "60+"}).all()
    assert clean_frame["BiasGroup"].notna().all()
    assert dirty_frame["BiasGroup"].notna().all()

    log_dataset(clean_frame, name="clean_frame", output_dir="data_validation")
    log_dataset(dirty_frame, name="dirty_frame", output_dir="data_validation")

    return clean_frame, dirty_frame


def _validate_pipeline(demo_batch: DemoBatch, clean_frame: pd.DataFrame, dirty_frame: pd.DataFrame) -> None:
    capture_logger = CaptureLogger()
    output_dir = Path(tempfile.mkdtemp(prefix="certain_demographic_bias_"))
    result = log_demographic_bias_from_data(
        capture_logger,
        clean_df=clean_frame,
        dirty_df=dirty_frame,
        output_dir=output_dir,
        max_clusters=4,
        random_seed=42,
        dataset_name="nvcr_batch_7_demo",
        topic_title=demo_batch.clean_payload["TopicTitle"],
    )

    # Replay artifacts captured by the test logger into the real tracker
    for local_path, artifact_path in capture_logger.artifact_calls:
        try:
            tracker.log_artifact(local_path, artifact_path=artifact_path)
        except Exception:
            pass

    assert result["clean"]["bias_snapshot"]["records"] == len(clean_frame)
    assert result["dirty"]["bias_snapshot"]["records"] == len(dirty_frame)
    assert result["comparison"] is not None
    assert result["diagnosis"]["verdict"] in {"OK", "WATCH", "BIAS SIGNAL"}

    comparison = result["comparison"]
    representation_shift = comparison["representation_shift"]

    clean_nationality = _distribution(clean_frame, "Nationality")
    dirty_nationality = _distribution(dirty_frame, "Nationality")
    clean_age = _distribution(clean_frame, "AgeBucket")
    dirty_age = _distribution(dirty_frame, "AgeBucket")

    # Record representation-shift and summary metrics to the tracker
    log_metrics(
        {
            "representation_shift_nationality_tvd": representation_shift["nationality_tvd"],
            "representation_shift_age_bucket_tvd": representation_shift["age_bucket_tvd"],
            "clean_records": len(clean_frame),
            "dirty_records": len(dirty_frame),
        }
    )

    # Log the textual verdict as a parameter (not a numeric metric)
    try:
        log_params({"demographic_bias_verdict": result["diagnosis"]["verdict"]})
    except Exception:
        # best-effort logging
        pass

    # Replay any metrics emitted by the demographic-bias routine so they
    # appear in the run's metric history as well.
    for metrics in capture_logger.metrics_calls:
        try:
            log_metrics(metrics)
        except Exception:
            # best-effort: do not fail the test if logging fails
            pass

    assert dirty_nationality["Greece"] > clean_nationality["Greece"]
    assert dirty_nationality["Bulgaria"] > clean_nationality["Bulgaria"]
    assert dirty_age["18-29"] > clean_age["18-29"]
    assert dirty_age["45-59"] >= clean_age["45-59"]
    assert representation_shift["nationality_tvd"] > 0.0
    assert representation_shift["age_bucket_tvd"] > 0.0

    assert result["artifacts"]["report_json"].exists()
    assert result["artifacts"]["report_html"].exists()
    assert any(call[1] == "demographic_bias" for call in capture_logger.artifact_calls)

    logged_metrics = {key for call in capture_logger.metrics_calls for key in call}
    expected_metrics = {
        "demographic_bias_anonymous_rate",
        "demographic_bias_missing_rate",
        "demographic_bias_weighted_nationality_tvd",
        "demographic_bias_weighted_background_tvd",
        "demographic_bias_weighted_age_bucket_tvd",
        "demographic_bias_largest_cluster_share",
        "demographic_bias_delta_anonymous_rate",
        "demographic_bias_delta_metadata_missing_rate",
    }
    assert expected_metrics <= logged_metrics

    report = json.loads(result["artifacts"]["report_json"].read_text(encoding="utf-8"))
    assert math.isclose(
        report["comparison"]["representation_shift"]["nationality_tvd"],
        representation_shift["nationality_tvd"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    assert math.isclose(
        report["comparison"]["representation_shift"]["age_bucket_tvd"],
        representation_shift["age_bucket_tvd"],
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    assert "intersectional" not in json.dumps(report).lower()


def main() -> None:
    print("=" * 72)
    print("End-to-end NVCR -> CERTAIN demographics-bias integration script")
    print("=" * 72)
    previous_auto_sync = os.environ.get("CERTAIN_AUTO_SYNC_ON_RUN_END")
    os.environ["CERTAIN_AUTO_SYNC_ON_RUN_END"] = "false"
    experiment_name = "nvcr_certain_demographics_complete"
    try:
        tracker.set_experiment(
            experiment_name=experiment_name,
            tags={
                "project": "nvcr_to_certain",
                "pipeline": "demographics_bias_complete",
                "mode": "offline",
            },
        )

        with tracker.start_run(
            run_name="nvcr_to_certain_demo",
            tags={
                "pipeline": "demographics_bias_complete",
                "source": "test_it_complete.py",
            }, 
        ) as run:
            print(f"Tracking experiment: {experiment_name}")
            print(f"Run ID: {run.info.run_id}")

            try:
                log_git_metadata(repo_path=str(PROJECT_ROOT.parent))
            except Exception:
                pass

            collect_runtime_environment()

            tracker.set_tags(
                {
                    "batch_id": "7",
                    "nvcr_repo": str(_find_nvcr_repo()),
                    "certain_repo": str(PROJECT_ROOT.parent),
                }
            )

            log_params(
                {
                    "batch_id": 7,
                    "offline": True,
                    "nvcr_script": "src/create_demo_bias_datasets.py",
                }
            )

            tracker_data, output_location = start_tracker(output_file_name="emissions_data_it_complete")

            try:
                print("Phase 1: generate deterministic demo datasets")
                demo_batch = _create_demo_batch(
                    batch_id=7,
                    tracking_run_id=run.info.run_id,
                    tracking_experiment_id=run.info.experiment_id,
                )
                _validate_demo_payloads(demo_batch)

                print("Phase 2: adapt public payload and hidden annotations")
                clean_frame, dirty_frame = _validate_certain_adapter(demo_batch)

                print("Phase 3: run CERTAIN demographic-bias analysis")
                _validate_pipeline(demo_batch, clean_frame, dirty_frame)

                log_metrics(
                    {
                        "it_complete_clean_rows": float(len(clean_frame)),
                        "it_complete_dirty_rows": float(len(dirty_frame)),
                    }
                )
            finally:
                try:
                    stop_tracker(tracker_data, output_location)
                except Exception:
                    pass

        print("Pipeline completed successfully")
    finally:
        if previous_auto_sync is None:
            os.environ.pop("CERTAIN_AUTO_SYNC_ON_RUN_END", None)
        else:
            os.environ["CERTAIN_AUTO_SYNC_ON_RUN_END"] = previous_auto_sync


if __name__ == "__main__":
    main()