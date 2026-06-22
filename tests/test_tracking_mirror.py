"""Tests for real-time MLflow mirroring."""

import json
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest

from certain_library.tracking.export import compact
from certain_library.tracking.manifest import (
    ManifestStore,
    recover_unfinished_runs,
)
from certain_library.tracking.tracker import Tracker


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


@pytest.fixture
def isolated_mlflow(tmp_path):
    while mlflow.active_run() is not None:
        mlflow.end_run()

    tracking_directory = tmp_path / "mlruns"
    mlflow.set_tracking_uri(tracking_directory.as_uri())

    yield

    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_param_metric_artifact_and_manifest(
    isolated_mlflow,
    tmp_path,
):
    tracker = Tracker(tmp_path / "certain")
    tracker.set_experiment("mirror-test")

    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"model-data")

    with tracker.start_run(run_name="mirrored-run") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        tracker.log_param("model", "bert")
        tracker.log_metric(
            "accuracy",
            0.80,
            step=1,
            timestamp=1710000000000,
        )
        tracker.log_metric(
            "accuracy",
            0.91,
            step=10,
            timestamp=1710000001000,
        )
        tracker.log_artifact(
            str(artifact),
            artifact_path="model",
        )

    root = tmp_path / "certain"

    parameter_events = read_jsonl(
        root / "events" / "params.jsonl"
    )
    assert parameter_events[-1]["key"] == "model"
    assert parameter_events[-1]["value"] == "bert"

    metric_events = read_jsonl(
        root / "events" / "metrics.jsonl"
    )
    assert [
        (event["step"], event["value"])
        for event in metric_events
    ] == [
        (1, 0.80),
        (10, 0.91),
    ]

    artifact_events = read_jsonl(
        root / "events" / "artifacts.jsonl"
    )
    artifact_event = artifact_events[-1]

    assert artifact_event["run_id"] == run_id
    assert artifact_event["experiment_id"] == experiment_id
    assert artifact_event["artifact_path"] == "model/model.pkl"
    assert artifact_event["file_size"] == len(b"model-data")
    assert artifact_event["checksum"]

    mirrored_artifact = (
        root
        / "experiments"
        / experiment_id
        / run_id
        / "artifacts"
        / "model"
        / "model.pkl"
    )
    assert mirrored_artifact.read_bytes() == b"model-data"

    manifest = ManifestStore(root).load(run_id)

    assert manifest["status"] == "FINISHED"
    assert manifest["is_final"] is True
    assert manifest["metric_count"] == 2
    assert manifest["artifact_count"] == 1
    assert manifest["param_count"] == 1
    assert manifest["end_time"] is not None


def test_mlflow_failure_does_not_write_mirror(
    isolated_mlflow,
    tmp_path,
    monkeypatch,
):
    tracker = Tracker(tmp_path / "certain")
    tracker.set_experiment("failure-test")

    with tracker.start_run():
        def fail_log_param(key, value):
            raise RuntimeError("MLflow failed")

        monkeypatch.setattr(
            mlflow,
            "log_param",
            fail_log_param,
        )

        with pytest.raises(RuntimeError, match="MLflow failed"):
            tracker.log_param("model", "bert")

    parameter_path = (
        tmp_path / "certain" / "events" / "params.jsonl"
    )
    assert not parameter_path.exists()


def test_recover_unfinished_run(
    isolated_mlflow,
    tmp_path,
):
    root = tmp_path / "certain"
    tracker = Tracker(root)
    tracker.set_experiment("recovery-test")

    run = tracker.start_run(run_name="interrupted")
    run_id = run.info.run_id

    # Simulate an application stopping without tracker.end_run().
    mlflow.end_run(status="FINISHED")

    before = ManifestStore(root).load(run_id)
    assert before["is_final"] is False

    result = recover_unfinished_runs(root=root)

    after = ManifestStore(root).load(run_id)

    assert result == {
        "recovered": 1,
        "failed": 0,
    }
    assert after["status"] == "FINISHED"
    assert after["is_final"] is True
    assert after["end_time"] is not None


def test_recovery_keeps_non_terminal_run_unfinished(tmp_path):
    root = tmp_path / "certain"
    store = ManifestStore(root)
    store.update(
        "run-1",
        status="RUNNING",
        is_final=False,
    )

    run = Mock()
    run.info.experiment_id = "experiment-1"
    run.info.status = "RUNNING"
    run.info.end_time = None

    client = Mock()
    client.get_run.return_value = run

    result = recover_unfinished_runs(
        root=root,
        client=client,
    )

    manifest = store.load("run-1")

    assert result["recovered"] == 1
    assert manifest["status"] == "RUNNING"
    assert manifest["is_final"] is False


def test_export_creates_compact_json_files(
    isolated_mlflow,
    tmp_path,
):
    root = tmp_path / "certain"
    tracker = Tracker(root)
    tracker.set_experiment("export-test")

    with tracker.start_run():
        tracker.log_param("model", "bert")
        tracker.log_metric(
            "accuracy",
            0.80,
            step=1,
            timestamp=1000,
        )
        tracker.log_metric(
            "accuracy",
            0.91,
            step=2,
            timestamp=2000,
        )
        tracker.set_tag("stage", "training")

    output = root / "metadata_json"
    counts = compact(root, output)

    expected_files = {
        "experiments.json",
        "runs.json",
        "params.json",
        "metrics.json",
        "latest_metrics.json",
        "tags.json",
        "artifacts.json",
    }

    assert {
        path.name
        for path in output.iterdir()
    } == expected_files

    metrics = json.loads(
        (output / "metrics.json").read_text()
    )
    latest_metrics = json.loads(
        (output / "latest_metrics.json").read_text()
    )

    assert len(metrics) == 2
    assert latest_metrics[0]["value"] == 0.91
    assert counts["metrics"] == 2
