#!/usr/bin/env python3
"""End-to-end NVCR/data-sink -> CERTAIN demographics-bias integration script.

Phase 1 obtains the demo datasets in one of three ways:

* data sink: set DATASPACE_SOURCE to read already-delivered files from an HTTP
  endpoint, a folder / file:// URL, or an S3/MinIO location (s3://<bucket>/<prefix>)
* scenario UI (default when DATASPACE_SOURCE is unset): start the
  Synthetic-Data-Generation-Component scenario UI (python -m src.scenario_ui),
  design the batch in your browser (size, countries, ages, bias controls, LLM or
  fast-fallback text), click "Generate datasets", and the script then continues
  with that batch. "LLM natural text" calls OpenAI using OPENAI_API_KEY /
  OPENAI_BASE_URL / OPENAI_MODEL from the environment or a .env file
* offline: IT_PHASE1_MODE=offline regenerates the fixed deterministic batch 7
  without any UI or LLM call (use this for unattended runs)

Phases 2 and 3 (adapt payloads, run the demographic-bias analysis) are the
same in every mode.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
import types

import pandas as pd

# Repo root on sys.path so certain_library and dataspace_connector import when run from the host.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_env_file(path: Path) -> None:
    """Load settings from a .env file without overriding variables already set in the environment."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


_load_env_file(REPO_ROOT / ".env")

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

# Empty DATASPACE_SOURCE means "no data sink"; anything else is read from the data sink.
DATASET_SOURCE = os.getenv("DATASPACE_SOURCE", "").strip()
DATASET_BATCH_ID = int(os.getenv("DATASPACE_BATCH_ID", "7"))

# Without a data sink: "ui" (default) opens the NVCR scenario UI, "offline" regenerates fixed batch 7.
PHASE1_MODE = os.getenv("IT_PHASE1_MODE", "ui").strip().lower()
IN_DOCKER = Path("/.dockerenv").exists()
# Inside a container the UI must listen on all interfaces for a published port to reach it.
UI_HOST = os.getenv("IT_UI_HOST", "").strip() or ("0.0.0.0" if IN_DOCKER else "127.0.0.1")
UI_PORT = int(os.getenv("IT_UI_PORT", "8769"))
UI_TIMEOUT_SECONDS = int(os.getenv("IT_UI_TIMEOUT_SECONDS", "1800"))
# test_docker/ is bind-mounted into the container, so batches saved here also appear on the host.
UI_OUTPUT_DIR = Path(os.getenv("IT_UI_OUTPUT_DIR", "").strip() or PROJECT_ROOT / "generated_batches")


@dataclass(frozen=True)
class DemoBatch:
    source: str
    clean_payload: dict[str, Any]
    dirty_payload: dict[str, Any]
    clean_annotations: list[dict[str, Any]]
    dirty_annotations: list[dict[str, Any]]
    batch_id: int = 7
    # True only for the fixed demo batches, whose skew the analysis is asserted against.
    expect_batch7_skew: bool = True


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


def _redact_source(source: str) -> str:
    """Drop credentials and query strings (e.g. presigned signatures) before logging a source."""
    parsed = urlparse(source)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=host, params="", query="", fragment=""))


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
            source="offline-demo",
            clean_payload=_load_json(session_dir / f"simulation_output_clean_{batch_id}.json"),
            dirty_payload=_load_json(session_dir / f"simulation_output_dirty_{batch_id}.json"),
            clean_annotations=list(annotations.get("Clean", [])),
            dirty_annotations=list(annotations.get("Dirty", [])),
            batch_id=batch_id,
        )


def _load_demo_batch_from_sink(source: str, batch_id: int = 7) -> DemoBatch:
    # Imported lazily so the default local mode works where dataspace_connector is not installed.
    from dataspace_connector.consumer import load_demo_batch_from_source

    loaded = load_demo_batch_from_source(source, batch_id=batch_id)
    return DemoBatch(
        source=_redact_source(source),
        clean_payload=loaded.clean_payload,
        dirty_payload=loaded.dirty_payload,
        clean_annotations=loaded.clean_annotations,
        dirty_annotations=loaded.dirty_annotations,
        batch_id=batch_id,
    )


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Scenario UI exited early with code {proc.returncode}; see its output above.")
        try:
            with socket.create_connection((connect_host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Scenario UI did not start listening on {host}:{port} within {timeout:.0f}s.")


def _find_ui_batch(session_dir: Path, started_at: float) -> int | None:
    """Newest batch whose scenario config (the UI writes it after the data files) appeared after started_at."""
    newest: tuple[float, int] | None = None
    for config_path in session_dir.glob("scenario_config_batch_*.json"):
        try:
            mtime = config_path.stat().st_mtime
            if mtime < started_at:
                continue
            batch_id = int(config_path.stem.rsplit("_", 1)[1])
            json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, IndexError):
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, batch_id)
    return newest[1] if newest else None


def _wait_for_ui_children(proc: subprocess.Popen, settle_seconds: float = 3.0, timeout: float = 300.0) -> None:
    """Let the UI finish post-generation clustering (a child process) before the server is stopped."""
    try:
        import psutil
    except ImportError:
        time.sleep(settle_seconds)
        return

    try:
        ui_process = psutil.Process(proc.pid)
    except psutil.Error:
        return

    deadline = time.monotonic() + timeout
    time.sleep(settle_seconds)
    while time.monotonic() < deadline:
        try:
            if not ui_process.children(recursive=True):
                break
        except psutil.Error:
            break
        time.sleep(1.0)
    # Give the UI a moment to send its HTTP response so the browser shows the results.
    time.sleep(2.0)


def _stop_ui(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _collect_batch_from_ui() -> DemoBatch:
    """Run the NVCR scenario UI, wait for the user to generate a batch, and return it."""
    nvcr_repo = _find_nvcr_repo()
    if not (nvcr_repo / "src" / "scenario_ui.py").exists():
        raise FileNotFoundError(f"{nvcr_repo} has no src/scenario_ui.py; update the NVCR repo.")
    _load_env_file(nvcr_repo / ".env")

    if IN_DOCKER:
        print(
            f"Running in Docker: the UI listens on {UI_HOST}:{UI_PORT} inside the container; "
            f"the host can only open it if that port is published (docker-compose publishes 127.0.0.1:{UI_PORT})."
        )
    elif UI_HOST not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"WARNING: the scenario UI has no authentication and can trigger paid LLM calls; "
            f"binding it to {UI_HOST} exposes it to your network."
        )
    print(
        "OpenAI credentials for 'LLM natural text': "
        + ("OPENAI_API_KEY is set" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY is NOT set (choose 'Fast fallback' in the UI)")
    )

    with tempfile.TemporaryDirectory(prefix="it_complete_ui_") as workspace_root:
        workspace = Path(workspace_root)
        # The UI writes into <repo>/session; use a writable copy because the NVCR repo may be mounted read-only.
        shutil.copytree(
            nvcr_repo / "src",
            workspace / "src",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        session_dir = workspace / "session"

        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + ([child_env["PYTHONPATH"]] if child_env.get("PYTHONPATH") else [])
        )
        started_at = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.scenario_ui", "--host", UI_HOST, "--port", str(UI_PORT)],
            cwd=workspace,
            env=child_env,
            start_new_session=True,
        )
        try:
            _wait_for_port(UI_HOST, UI_PORT, proc)
            print("=" * 72)
            print(f"Scenario UI ready: open http://127.0.0.1:{UI_PORT} in your browser")
            print("Set the generation parameters, click 'Generate datasets', and this script continues.")
            print(f"Waiting up to {UI_TIMEOUT_SECONDS}s for a generated batch...")
            print("=" * 72)

            deadline = time.monotonic() + UI_TIMEOUT_SECONDS
            batch_id: int | None = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Scenario UI exited with code {proc.returncode} before a batch was generated.")
                batch_id = _find_ui_batch(session_dir, started_at)
                if batch_id is not None:
                    break
                time.sleep(1.0)
            if batch_id is None:
                raise TimeoutError(f"No batch was generated in the scenario UI within {UI_TIMEOUT_SECONDS}s.")

            print(f"Batch {batch_id} generated in the UI; finishing post-processing...")
            _wait_for_ui_children(proc)

            batch_files = [
                session_dir / f"simulation_output_clean_{batch_id}.json",
                session_dir / f"simulation_output_dirty_{batch_id}.json",
                session_dir / f"scenario_annotations_batch_{batch_id}.json",
                session_dir / f"scenario_config_batch_{batch_id}.json",
            ]
            clean_path, dirty_path, annotations_path, config_path = batch_files
            clean_payload = _load_json(clean_path)
            dirty_payload = _load_json(dirty_path)
            annotations = _load_json(annotations_path)
            config = _load_json(config_path)
            print(
                f"Using UI batch {batch_id}: {len(clean_payload.get('Dataset', []))} clean / "
                f"{len(dirty_payload.get('Dataset', []))} dirty records, "
                f"allocation={config.get('allocation_strategy')}, mode_mix={config.get('mode_mix')}"
            )
            if batch_id == DATASET_BATCH_ID:
                # The browser can restore an old "Batch id" value; flag ids that look like the offline batch.
                print(
                    f"NOTE: batch id {batch_id} is the same id as the offline demo batch, but this data "
                    "comes from the scenario UI. Change 'Batch id' in the UI to tell runs apart in MLflow."
                )

            # The UI workspace is temporary, so keep the generated batch where the host can see it.
            output_dir = UI_OUTPUT_DIR / f"batch_{batch_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in batch_files:
                shutil.copy2(path, output_dir / path.name)
            cluster_dir = session_dir / "cluster_bias_outputs" / f"batch_{batch_id}"
            if cluster_dir.is_dir():
                shutil.copytree(cluster_dir, output_dir / "cluster_bias_outputs", dirs_exist_ok=True)
            print(f"Saved the generated batch to {output_dir}")

            for path in batch_files:
                try:
                    tracker.log_artifact(str(path), artifact_path="scenario_ui")
                except Exception:
                    pass
            return DemoBatch(
                source="scenario-ui",
                clean_payload=clean_payload,
                dirty_payload=dirty_payload,
                clean_annotations=list(annotations.get("Clean", [])),
                dirty_annotations=list(annotations.get("Dirty", [])),
                batch_id=batch_id,
                expect_batch7_skew=False,
            )
        finally:
            _stop_ui(proc)


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

    public_fields = {"FirstName", "LastName", "Age", "DateOfBirth", "Nationality"}
    internal_fields = {"Background", "Opinion", "AnswerQuality", "BiasGroup"}

    assert clean_records and dirty_records
    if not demo_batch.expect_batch7_skew:
        # UI scenarios may add optional internal fields (e.g. WrongAnswerPattern), so require a superset.
        assert public_fields <= set(clean_records[0]["Metadata"])
        assert public_fields <= set(dirty_records[0]["Metadata"])
        assert internal_fields <= set(demo_batch.clean_annotations[0]["InternalMetadata"])
        assert internal_fields <= set(demo_batch.dirty_annotations[0]["InternalMetadata"])
        return

    assert set(clean_records[0]["Metadata"]) == public_fields
    assert set(dirty_records[0]["Metadata"]) == public_fields
    assert set(demo_batch.clean_annotations[0]["InternalMetadata"]) == internal_fields
    assert set(demo_batch.dirty_annotations[0]["InternalMetadata"]) == internal_fields

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


def _validate_pipeline(
    demo_batch: DemoBatch,
    clean_frame: pd.DataFrame,
    dirty_frame: pd.DataFrame,
) -> None:
    capture_logger = CaptureLogger()
    output_dir = Path(tempfile.mkdtemp(prefix="certain_demographic_bias_"))
    result = log_demographic_bias_from_data(
        capture_logger,
        clean_df=clean_frame,
        dirty_df=dirty_frame,
        output_dir=output_dir,
        max_clusters=4,
        random_seed=42,
        dataset_name=f"nvcr_batch_{demo_batch.batch_id}_{'ui' if demo_batch.source == 'scenario-ui' else 'demo'}",
        topic_title=demo_batch.clean_payload["TopicTitle"],
    )

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

    log_metrics(
        {
            "representation_shift_nationality_tvd": representation_shift["nationality_tvd"],
            "representation_shift_age_bucket_tvd": representation_shift["age_bucket_tvd"],
            "clean_records": len(clean_frame),
            "dirty_records": len(dirty_frame),
        }
    )

    try:
        log_params({"demographic_bias_verdict": result["diagnosis"]["verdict"]})
    except Exception:
        pass

    for metrics in capture_logger.metrics_calls:
        try:
            log_metrics(metrics)
        except Exception:
            pass

    if demo_batch.expect_batch7_skew:
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
    # Without a TTY (plain `docker exec`) stdout is block-buffered and progress would stay invisible.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    use_sink = bool(DATASET_SOURCE)
    if PHASE1_MODE not in {"ui", "offline"}:
        raise ValueError(f"IT_PHASE1_MODE must be 'ui' or 'offline', got {PHASE1_MODE!r}.")
    mode = "data-sink" if use_sink else ("scenario-ui" if PHASE1_MODE == "ui" else "offline-demo")
    safe_source = _redact_source(DATASET_SOURCE) if use_sink else None

    print("=" * 72)
    print("End-to-end NVCR/data-sink -> CERTAIN demographics-bias integration script")
    print(f"Phase 1 mode: {mode}" + (f" ({safe_source})" if safe_source else ""))
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
                "mode": mode,
            },
        )

        with tracker.start_run(
            run_name="nvcr_to_certain_demo",
            tags={
                "pipeline": "demographics_bias_complete",
                "source": "test_it_complete.py",
                "mode": mode,
            },
        ) as run:
            print(f"Tracking experiment: {experiment_name}")
            print(f"Run ID: {run.info.run_id}")

            try:
                log_git_metadata(repo_path=str(PROJECT_ROOT.parent))
            except Exception:
                pass

            collect_runtime_environment()

            run_tags = {
                "certain_repo": str(PROJECT_ROOT.parent),
                "phase1_mode": mode,
            }
            run_params: dict[str, Any] = {
                "offline": mode == "offline-demo",
                "data_source_mode": mode,
            }
            if use_sink:
                run_tags["dataspace_source"] = safe_source
                run_params["dataspace_source"] = safe_source
            else:
                run_tags["nvcr_repo"] = str(_find_nvcr_repo())
                run_params["nvcr_script"] = (
                    "src/scenario_ui.py" if mode == "scenario-ui" else "src/create_demo_bias_datasets.py"
                )
            if mode != "scenario-ui":
                # In UI mode the batch id is only known after the user generates a batch.
                run_tags["batch_id"] = str(DATASET_BATCH_ID)
                run_params["batch_id"] = DATASET_BATCH_ID

            tracker.set_tags(run_tags)
            log_params(run_params)

            demo_batch: DemoBatch | None = None
            if mode == "scenario-ui":
                # Collected before the emissions tracker starts so time spent waiting on a human isn't measured.
                print("Phase 1: design and generate the batch in the scenario UI")
                demo_batch = _collect_batch_from_ui()
                tracker.set_tags({"batch_id": str(demo_batch.batch_id)})
                log_params({"batch_id": demo_batch.batch_id})

            tracker_data, output_location = start_tracker(output_file_name="emissions_data_it_complete")

            try:
                if demo_batch is None:
                    if use_sink:
                        print("Phase 1: read demo datasets from the data sink")
                        demo_batch = _load_demo_batch_from_sink(DATASET_SOURCE, batch_id=DATASET_BATCH_ID)
                    else:
                        print("Phase 1: generate deterministic demo datasets")
                        demo_batch = _create_demo_batch(
                            batch_id=DATASET_BATCH_ID,
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
