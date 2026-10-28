"""End-to-end NVCR -> CERTAIN demographics-bias integration test.

This test uses the deterministic demo batch from src.create_demo_bias_datasets.
The public JSON payload stays separate from the sidecar annotations, and the
adapter below joins them only because certain_library.log_demographic_bias_from_data
expects the public demographic columns plus the internal Background/Opinion/
AnswerQuality/BiasGroup labels and an AgeBucket column.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import types

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
CERTAIN_MODULE_CANDIDATES = [
    Path("/app/certain_library/data_analysis/log_demographic_bias.py"),
    PROJECT_ROOT.parent / "certain_library" / "data_analysis" / "log_demographic_bias.py",
    PROJECT_ROOT.parent.parent / "Semantic_MLOps_engine" / "certain_library" / "data_analysis" / "log_demographic_bias.py",
]


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
    tracker_mod.tracker = _StubTracker()

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

NVCR_REPO_CANDIDATES = [
    Path("/app/Synthetic-Data-Generation-Component"),
    PROJECT_ROOT.parent / "Synthetic-Data-Generation-Component",
]


def _find_nvcr_repo() -> Path:
    for candidate in NVCR_REPO_CANDIDATES:
        if (candidate / "src" / "create_demo_bias_datasets.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the NVCR repository root.")


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


def _create_demo_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DemoBatch:
    session_dir = tmp_path / "session"
    batch_id = 7
    nvcr_repo = _find_nvcr_repo()
    result = subprocess.run(
        [
            sys.executable,
            str(nvcr_repo / "src" / "create_demo_bias_datasets.py"),
            "--batch",
            "7",
            "--output-dir",
            str(session_dir),
        ],
        cwd=nvcr_repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
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


def test_nvcr_demo_generation_preserves_public_and_internal_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_batch = _create_demo_batch(tmp_path, monkeypatch)

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

    assert dirty_nationality["Greece"] > clean_nationality["Greece"], (
        f"expected Greece to be overrepresented in dirty data; clean={clean_nationality}, "
        f"dirty={dirty_nationality}"
    )
    assert dirty_nationality["Bulgaria"] > clean_nationality["Bulgaria"], (
        f"expected Bulgaria to be overrepresented in dirty data; clean={clean_nationality}, "
        f"dirty={dirty_nationality}"
    )
    assert dirty_age["18-29"] > clean_age["18-29"], (
        f"expected younger records to be overrepresented in dirty data; clean={clean_age}, "
        f"dirty={dirty_age}"
    )
    assert dirty_age["45-59"] >= clean_age["45-59"], (
        f"expected older records to be at least preserved or increased in dirty data; clean={clean_age}, "
        f"dirty={dirty_age}"
    )


def test_nvcr_adapter_merges_public_and_hidden_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_batch = _create_demo_batch(tmp_path, monkeypatch)

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


def test_nvcr_to_certain_demographics_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_batch = _create_demo_batch(tmp_path, monkeypatch)
    clean_frame = _build_certain_frame(demo_batch.clean_payload, demo_batch.clean_annotations)
    dirty_frame = _build_certain_frame(demo_batch.dirty_payload, demo_batch.dirty_annotations)

    capture_logger = CaptureLogger()
    output_dir = tmp_path / "certain_demographic_bias"
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

    assert dirty_nationality["Greece"] > clean_nationality["Greece"], (
        f"CLEAN nationality distribution: {clean_nationality}\n"
        f"DIRTY nationality distribution: {dirty_nationality}\n"
        f"CERTAIN comparison: {comparison}"
    )
    assert dirty_nationality["Bulgaria"] > clean_nationality["Bulgaria"], (
        f"CLEAN nationality distribution: {clean_nationality}\n"
        f"DIRTY nationality distribution: {dirty_nationality}\n"
        f"CERTAIN comparison: {comparison}"
    )
    assert dirty_age["18-29"] > clean_age["18-29"], (
        f"CLEAN age distribution: {clean_age}\n"
        f"DIRTY age distribution: {dirty_age}\n"
        f"CERTAIN comparison: {comparison}"
    )
    assert dirty_age["45-59"] >= clean_age["45-59"], (
        f"CLEAN age distribution: {clean_age}\n"
        f"DIRTY age distribution: {dirty_age}\n"
        f"CERTAIN comparison: {comparison}"
    )
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
    assert expected_metrics <= logged_metrics, (
        f"logged metrics: {sorted(logged_metrics)}\n"
        f"expected metrics: {sorted(expected_metrics)}"
    )

    report = json.loads(result["artifacts"]["report_json"].read_text(encoding="utf-8"))
    assert report["comparison"]["representation_shift"]["nationality_tvd"] == pytest.approx(
        representation_shift["nationality_tvd"]
    )
    assert report["comparison"]["representation_shift"]["age_bucket_tvd"] == pytest.approx(
        representation_shift["age_bucket_tvd"]
    )
    assert "intersectional" not in json.dumps(report).lower()
