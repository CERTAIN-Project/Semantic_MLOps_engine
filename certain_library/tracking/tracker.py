"""MLflow-first tracker with a portable local mirror."""

import logging
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, List
import threading

import mlflow
from mlflow.tracking import MlflowClient
import json

from .artifact_mirror import ArtifactMirror
from .manifest import FINAL_STATUSES, ManifestStore
from .metadata_writer import MetadataWriter, timestamp_ms
import tempfile
import csv
import time
import warnings
import collections

from metadata.artifact_metadata import save_tags_as_artifact
from metadata.artifact_metadata import save_params_as_artifact
from metadata.artifact_metadata import save_metrics_as_artifact
from metadata.artifact_metadata import save_resources_as_artifact
from metadata.artifact_metadata import save_inputs_as_artifact
from metadata.artifact_metadata import _extract_run_inputs

logger = logging.getLogger(__name__)


def mirror_root() -> Path:
    """Return the configured CERTAIN mirror location."""
    return Path(os.getenv("CERTAIN_MIRROR_ROOT", "certain"))


ARTIFACT_BASE_PATH = "certain"


def mlflow_artifact_path(artifact_path: Optional[str] = None) -> str:
    """Return an MLflow artifact path under artifacts/certain/.

    Examples:
        None -> "certain"
        "whylogs" -> "certain/whylogs"
        "certain/whylogs" -> "certain/whylogs"
    """
    if artifact_path is None or str(artifact_path).strip() == "":
        return ARTIFACT_BASE_PATH

    normalized = str(artifact_path).strip("/")

    if normalized == ARTIFACT_BASE_PATH or normalized.startswith(
        f"{ARTIFACT_BASE_PATH}/"
    ):
        return normalized

    return f"{ARTIFACT_BASE_PATH}/{normalized}"


class MirroredRun:
    """Proxy an MLflow ActiveRun and mirror context-manager completion."""

    def __init__(self, tracker: "Tracker", active_run: Any):
        self._tracker = tracker
        self._active_run = active_run

    def __getattr__(self, name: str) -> Any:
        return getattr(self._active_run, name)

    def __enter__(self) -> "MirroredRun":
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> bool:
        status = "FINISHED" if exception_type is None else "FAILED"
        self._tracker.end_run(status=status)
        return False


class Tracker:
    """Write to MLflow first, then update the local mirror."""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else mirror_root()
        self.writer = MetadataWriter(self.root)
        self.manifests = ManifestStore(self.root)
        self.artifacts = ArtifactMirror(self.root)
        # Console capture state used to mirror terminal output into artifacts
        self._console_td: Optional[Path] = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._console_csv_fh = None
        self._console_raw_fh = None
        self._console_line_id = 0

        # lock to synchronize writes across threads
        self._console_lock = threading.Lock()
        # store original root logger level while capture is active
        self._console_original_root_level = None

        self._console_log_handler = None
        # low-level fd capture state
        self._console_pipe_r = None
        self._console_pipe_w = None
        self._console_saved_stdout = None
        self._console_saved_stderr = None
        self._console_reader_thread = None
        self._console_reader_running = False
        self._console_log_attached = []
        self._warnings_showwarning_orig = None
        self._experiment_not_exist = False
        # recent messages for deduplication (avoid duplicates between fd and
        # logging handler captures)
        self._console_recent_msgs = collections.deque(maxlen=200)
        self._console_recent_set = set()

    def set_tracking_uri(self, uri: str) -> None:
        mlflow.set_tracking_uri(uri)

    def set_experiment(
        self,
        experiment_name: str,
        **kwargs: Any,
    ) -> Any:
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist; creating it.")
            experiment_id = client.create_experiment(
                name=experiment_name,
                **kwargs,
            )
            experiment = client.get_experiment(experiment_id)
            self._experiment_not_exist = True

        kwargs.pop("tags", None)
        result = mlflow.set_experiment(experiment.name, **kwargs)

        return result

    def start_run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> MirroredRun:
        """
        Start an MLflow run, record its local mirror event, save metadata
        artifacts, and manage console capture.

        Console capture is started only for the parent run. Nested runs reuse
        the parent run's process-wide console capture.
        """
        active_before_start = mlflow.active_run()

        requested_nested = bool(kwargs.get("nested", False))
        is_nested = requested_nested or active_before_start is not None

        # When a run is already active, automatically create a nested run unless
        # the caller explicitly supplied the nested argument.
        if active_before_start is not None and "nested" not in kwargs:
            kwargs["nested"] = True

        active_run = mlflow.start_run(*args, **kwargs)
        run_info = active_run.info

        run_id = str(run_info.run_id)
        experiment_id = str(run_info.experiment_id)

        parent_run_id: Optional[str] = None

        if is_nested:
            if active_before_start is not None:
                parent_run_id = str(active_before_start.info.run_id)
            else:
                parent_run_id = getattr(
                    self,
                    "_console_owner_run_id",
                    None,
                )
        else:
            self._parent_run_id = run_id
            self._parent_experiment_id = experiment_id

            # Start console capture immediately for the parent run so all output
            # produced by the remaining setup is included.
            self._start_console_capture(
                experiment_id=experiment_id,
                run_id=run_id,
                nested=False,
            )

            self._write_console_line(
                "system",
                "PARENT_RUN_STARTED run_id={}".format(run_id),
            )

        if is_nested:
            # A nested run must not start another process-wide stdout/stderr
            # redirection. It only adds a lifecycle marker to the parent's logs.
            self._write_console_line(
                "system",
                (
                    "NESTED_RUN_STARTED "
                    "run_id={} parent_run_id={}"
                ).format(
                    run_id,
                    parent_run_id,
                ),
            )

        event = self.writer.append(
            "run",
            {
                "action": "started",
                "run_id": run_id,
                "experiment_id": experiment_id,
                "parent_run_id": parent_run_id,
                "is_nested": is_nested,
                "run_name": run_info.run_name,
                "status": run_info.status,
                "start_time": run_info.start_time,
                "end_time": None,
                "artifact_uri": run_info.artifact_uri,
            },
        )

        self.manifests.update_for_event(
            run_id,
            event,
            experiment_id=experiment_id,
            status=run_info.status,
            start_time=run_info.start_time,
            artifact_uri=run_info.artifact_uri,
            is_final=False,
        )

        run_tags: Dict[str, str] = {
            str(key): str(value)
            for key, value in (kwargs.get("tags") or {}).items()
        }

        for key, value in run_tags.items():
            self._record(
                "tag",
                {
                    "key": key,
                    "value": value,
                },
                active_run,
            )

        save_tags_as_artifact(
            run_id,
            run_tags,
            type_of_tags="run",
        )

        client = MlflowClient()
        experiment = client.get_experiment(experiment_id)

        if experiment is None:
            raise RuntimeError(
                "MLflow experiment {!r} could not be loaded.".format(
                    experiment_id
                )
            )

        # Create experiment-level metadata only for the parent run.
        if not is_nested:
            if getattr(self, "_experiment_not_exist", False):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    experiment_path = (
                        Path(temporary_directory)
                        / "experiment.json"
                    )

                    experiment_record = {
                        "experiment_id": str(
                            experiment.experiment_id
                        ),
                        "experiment_name": getattr(
                            experiment,
                            "name",
                            "",
                        )
                        or "",
                        "created_at": getattr(
                            experiment,
                            "creation_time",
                            None,
                        ),
                    }

                    with experiment_path.open(
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(
                            experiment_record,
                            file,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )

                    self.log_artifact(
                        str(experiment_path),
                        artifact_path="metadata",
                    )

                experiment_tags: Dict[str, str] = {
                    str(key): str(value)
                    for key, value in (
                        getattr(experiment, "tags", {}) or {}
                    ).items()
                }

                save_tags_as_artifact(
                    str(experiment.experiment_id),
                    experiment_tags,
                    type_of_tags="experiment",
                )

        # Write run metadata for both parent and nested runs. Because the nested
        # run is currently active, self.log_artifact() stores this file under the
        # correct MLflow run.
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_metadata_path = (
                Path(temporary_directory)
                / "run_metadata.json"
            )

            run_metadata_record = {
                "run_id": run_id,
                "run_name": getattr(
                    run_info,
                    "run_name",
                    "",
                )
                or "",
                "created_at": getattr(
                    run_info,
                    "start_time",
                    None,
                ),
                "status": getattr(
                    run_info,
                    "status",
                    None,
                ),
                "experiment_id": experiment_id,
                "parent_run_id": parent_run_id,
                "is_nested": is_nested,
                "end_time": None,
                "artifact_uri": getattr(
                    run_info,
                    "artifact_uri",
                    None,
                ),
            }

            with run_metadata_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    run_metadata_record,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            self.log_artifact(
                str(run_metadata_path),
                artifact_path="metadata",
            )

        return MirroredRun(
            self,
            active_run,
        )

    def end_run(
        self,
        status: str = "FINISHED",
        expected_run_id: Optional[str] = None,
    ) -> None:
        """
        End the active MLflow run and serialize its mirrored metadata.

        Nested runs:
        - write a nested-run stop marker;
        - keep the parent console capture running;
        - upload their own metadata and events.

        Parent run:
        - stop and upload console capture before ending the MLflow run;
        - upload its metadata and events;
        - clear parent-run state.
        """
        active_run = mlflow.active_run()

        if active_run is None:
            return

        run_info = active_run.info

        run_id = str(run_info.run_id)
        experiment_id = str(run_info.experiment_id)
        artifact_uri = getattr(run_info, "artifact_uri", None)
        run_name = getattr(run_info, "run_name", None)
        start_time = getattr(run_info, "start_time", None)

        if (
            expected_run_id is not None
            and run_id != str(expected_run_id)
        ):
            raise RuntimeError(
                "Expected active run {!r}, but found {!r}.".format(
                    expected_run_id,
                    run_id,
                )
            )

        console_owner_run_id = getattr(
            self,
            "_console_owner_run_id",
            None,
        )

        is_parent_run = (
            console_owner_run_id is not None
            and run_id == str(console_owner_run_id)
        )

        parent_run_id: Optional[str]

        if is_parent_run:
            parent_run_id = None
        else:
            parent_run_id = (
                str(console_owner_run_id)
                if console_owner_run_id is not None
                else None
            )

        # Record the run-ending event before collecting local events.
        end_time = timestamp_ms()

        try:
            ended_event = self.writer.append(
                "run",
                {
                    "action": "ended",
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "parent_run_id": parent_run_id,
                    "is_nested": not is_parent_run,
                    "status": status,
                    "start_time": start_time,
                    "end_time": end_time,
                    "artifact_uri": artifact_uri,
                    "run_name": run_name,
                },
            )
        except Exception:
            logger.exception(
                "Failed to write the run-ended event for run %s",
                run_id,
            )
            ended_event = None

        if is_parent_run:
            self._write_console_line(
                "system",
                "PARENT_RUN_STOPPING run_id={} status={}".format(
                    run_id,
                    status,
                ),
            )

            # Stop and upload the process-wide console capture while the
            # parent run is still available.
            try:
                self._stop_console_capture(
                    run_id=run_id,
                    nested=False,
                    status=status,
                )
            except Exception:
                logger.exception(
                    "Failed to stop console capture for parent run %s",
                    run_id,
                )
        else:
            # Nested runs must not stop the parent capture.
            try:
                self._mark_nested_run_stopped(
                    run_id=run_id,
                    status=status,
                )
            except Exception:
                logger.exception(
                    "Failed to record nested-run stop marker for run %s",
                    run_id,
                )

        try:
            events_dir: Path = self.writer.events_dir
            events_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            run_events: List[Dict[str, Any]] = []

            counts = {
                "param_count": 0,
                "metric_count": 0,
                "tag_count": 0,
                "artifact_count": 0,
                "resource_count": 0,
                "input_count": 0,
            }

            from .metadata_writer import EVENT_FILES as _EVENT_FILES

            for filename in _EVENT_FILES.values():
                path = events_dir / filename

                if not path.exists():
                    continue

                with path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    for line in handle:
                        try:
                            event_record = json.loads(line)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            continue

                        if str(event_record.get("run_id")) != run_id:
                            continue

                        run_events.append(event_record)

                        event_type = event_record.get("event_type")

                        if event_type == "param":
                            counts["param_count"] += 1
                        elif event_type == "metric":
                            counts["metric_count"] += 1
                        elif event_type == "tag":
                            counts["tag_count"] += 1
                        elif event_type == "artifact":
                            counts["artifact_count"] += 1
                        elif event_type == "resource":
                            counts["resource_count"] += 1
                        elif event_type == "input":
                            counts["input_count"] += 1

            final_status = status
            final_end_time = end_time

            for event_record in run_events:
                if (
                    event_record.get("event_type") == "run"
                    and event_record.get("action") == "ended"
                ):
                    final_status = event_record.get(
                        "status",
                        final_status,
                    )
                    final_end_time = event_record.get(
                        "end_time",
                        final_end_time,
                    )

            run_metadata: Dict[str, Any] = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "parent_run_id": parent_run_id,
                "is_nested": not is_parent_run,
                "run_name": run_name,
                "status": final_status,
                "start_time": start_time,
                "end_time": final_end_time,
                "artifact_uri": artifact_uri,
            }

            run_metadata.update(counts)

            temporary_directory = Path(
                tempfile.mkdtemp(
                    prefix="certain_{}_".format(run_id)
                )
            )

            try:
                metadata_path = (
                    temporary_directory
                    / "run_metadata.json"
                )

                events_path = (
                    temporary_directory
                    / "events.jsonl"
                )

                with metadata_path.open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        run_metadata,
                        handle,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                        default=str,
                    )
                    handle.write("\n")

                with events_path.open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    for event_record in run_events:
                        handle.write(
                            json.dumps(
                                event_record,
                                ensure_ascii=False,
                                default=str,
                                sort_keys=True,
                            )
                            + "\n"
                        )

                client = MlflowClient()

                client.log_artifacts(
                    run_id,
                    str(temporary_directory),
                    artifact_path=mlflow_artifact_path(
                        "metadata"
                    ),
                )

            finally:
                try:
                    shutil.rmtree(
                        str(temporary_directory)
                    )
                except Exception:
                    logger.exception(
                        "Failed to remove metadata temporary directory for run %s",
                        run_id,
                    )

            if ended_event is not None:
                self.manifests.update_for_event(
                    run_id,
                    ended_event,
                    status=final_status,
                    end_time=final_end_time,
                    is_final=final_status in FINAL_STATUSES,
                    recovery_required=False,
                )

        except Exception:
            logger.exception(
                "Failed to serialize run metadata into artifacts for run %s",
                run_id,
            )

            self.manifests.update(
                run_id,
                recovery_required=True,
            )

        finally:
            # End the active MLflow run after all artifacts have been uploaded.
            mlflow.end_run(status=status)

            if is_parent_run:
                self._parent_run_id = None
                self._parent_experiment_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        active = mlflow.active_run()

        if active is None:
            raise RuntimeError("No active MLflow run.")

        normalized_params: Dict[str, str] = {
            str(key): str(value) for key, value in params.items()
        }

        mlflow.log_params(normalized_params)

        for key, value in normalized_params.items():
            self._record_active(
                "param",
                {
                    "key": key,
                    "value": value,
                },
            )

        client = MlflowClient()
        run = client.get_run(active.info.run_id)

        save_params_as_artifact(
            active.info.run_id,
            dict(run.data.params),
            type_of_params="run",
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        active = mlflow.active_run()

        if active is None:
            raise RuntimeError("No active MLflow run.")

        normalized_metrics: Dict[str, float] = {
            str(key): float(value) for key, value in metrics.items()
        }

        mlflow_arguments: Dict[str, Any] = {}

        if timestamp is not None:
            mlflow_arguments["timestamp"] = int(timestamp)

        if step is not None:
            mlflow_arguments["step"] = int(step)

        mlflow.log_metrics(
            normalized_metrics,
            **mlflow_arguments,
        )

        event_timestamp = int(timestamp) if timestamp is not None else timestamp_ms()

        event_step = int(step) if step is not None else 0

        for key, value in normalized_metrics.items():
            self._record_active(
                "metric",
                {
                    "key": key,
                    "value": value,
                    "step": event_step,
                    "timestamp": event_timestamp,
                },
            )

        client = MlflowClient()
        run = client.get_run(active.info.run_id)
        all_metric_history: Dict[str, list[Dict[str, Any]]] = {}

        for metric_name in run.data.metrics.keys():
            metric_name = str(metric_name)
            # Skip system resource metrics here — they are saved by log_resources
            if metric_name.startswith("system_metrics/"):
                continue

            history = client.get_metric_history(active.info.run_id, metric_name)

            all_metric_history[metric_name] = [
                {
                    "value": float(metric.value),
                    "step": int(metric.step),
                    "timestamp": int(metric.timestamp),
                }
                for metric in history
            ]

        save_metrics_as_artifact(
            run_id=active.info.run_id,
            metrics=all_metric_history,
        )

    def log_resources(
        self,
        resources: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        """
        Log system-resource measurements to MLflow under the
        'system_metrics/' metric namespace and save their complete history
        to certain/metadata/run_resources.json.

        Example input:

            {
                "train_cpu_usage": 72.5,
                "memory_usage": 61.2,
            }

        MLflow metric names:

            system_metrics/train_cpu_usage
            system_metrics/memory_usage
        """

        active = mlflow.active_run()

        if active is None:
            raise RuntimeError("No active MLflow run.")

        normalized_resources: Dict[str, float] = {
            str(key): float(value) for key, value in resources.items()
        }

        mlflow_resources: Dict[str, float] = {
            "system_metrics/{}".format(key): value
            for key, value in normalized_resources.items()
        }

        mlflow_arguments: Dict[str, Any] = {}

        if timestamp is not None:
            mlflow_arguments["timestamp"] = int(timestamp)

        if step is not None:
            mlflow_arguments["step"] = int(step)

        mlflow.log_metrics(
            mlflow_resources,
            **mlflow_arguments,
        )

        event_timestamp = int(timestamp) if timestamp is not None else timestamp_ms()

        event_step = int(step) if step is not None else 0

        for key, value in normalized_resources.items():
            self._record_active(
                "resource",
                {
                    "key": key,
                    "mlflow_key": "system_metrics/{}".format(key),
                    "value": value,
                    "step": event_step,
                    "timestamp": event_timestamp,
                },
            )

        client = MlflowClient()
        run_id = active.info.run_id
        run = client.get_run(run_id)

        all_resource_history: Dict[str, List[Dict[str, Any]]] = {}

        for metric_name in run.data.metrics.keys():
            metric_name = str(metric_name)

            if not metric_name.startswith("system_metrics/"):
                continue

            resource_name = metric_name.split("/", 1)[1]

            history = client.get_metric_history(
                run_id,
                metric_name,
            )

            all_resource_history[resource_name] = [
                {
                    "value": float(metric.value),
                    "step": int(metric.step),
                    "timestamp": int(metric.timestamp),
                }
                for metric in history
            ]

        save_resources_as_artifact(
            run_id=run_id,
            resources=all_resource_history,
        )

    def set_tags(self, tags: Dict[str, Any]) -> None:
        active = mlflow.active_run()

        if active is None:
            raise RuntimeError("No active MLflow run.")

        normalized_tags = {str(key): str(value) for key, value in tags.items()}

        mlflow.set_tags(normalized_tags)

        for key, value in normalized_tags.items():
            self._record_active(
                "tag",
                {"key": key, "value": value},
            )
        client = MlflowClient()
        run = client.get_run(active.info.run_id)

        save_tags_as_artifact(
            active.info.run_id,
            dict(run.data.tags),
            type_of_tags="run",
        )

    def log_input(
        self,
        dataset: Any,
        context: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        active = mlflow.active_run()

        if active is None:
            raise RuntimeError("No active MLflow run.")

        mlflow.log_input(
            dataset,
            context=context,
            tags=tags,
        )

        self._record_active(
            "input",
            {
                "context": context,
                "tags": tags or {},
                "dataset_name": getattr(dataset, "name", None),
            },
        )

        client = MlflowClient()
        run = client.get_run(active.info.run_id)

        save_inputs_as_artifact(
            run_id=active.info.run_id,
            inputs=_extract_run_inputs(run),
        )

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log to MLflow first, then copy and checksum the artifact."""
        destination_path = mlflow_artifact_path(artifact_path)
        mlflow.log_artifact(local_path, artifact_path=destination_path)
        active_run = mlflow.active_run()

        if active_run is None:
            return

        try:
            artifact = self.artifacts.mirror_file(
                local_path,
                active_run.info.experiment_id,
                active_run.info.run_id,
                destination_path,
            )
            artifact["timestamp"] = timestamp_ms()
            self._record("artifact", artifact, active_run)
        except Exception:
            logger.exception("MLflow artifact succeeded, but local mirroring failed")
            self.manifests.update(
                active_run.info.run_id,
                recovery_required=True,
            )

    def log_artifacts(
        self,
        local_directory: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        destination_path = mlflow_artifact_path(artifact_path)
        mlflow.log_artifacts(
            local_directory,
            artifact_path=destination_path,
        )
        active_run = mlflow.active_run()

        if active_run is None:
            return

        try:
            for artifact in self.artifacts.mirror_directory(
                local_directory,
                active_run.info.experiment_id,
                active_run.info.run_id,
                destination_path,
            ):
                artifact["timestamp"] = timestamp_ms()
                self._record("artifact", artifact, active_run)
        except Exception:
            logger.exception("MLflow artifacts succeeded, but local mirroring failed")
            self.manifests.update(
                active_run.info.run_id,
                recovery_required=True,
            )

    def log_xgboost_model(self, model: Any, artifact_path: str, **kwargs: Any) -> Any:
        """Log an XGBoost model and recover its generated artifact files."""
        destination_path = mlflow_artifact_path(artifact_path)
        result = mlflow.xgboost.log_model(
            xgb_model=model, artifact_path=destination_path, **kwargs
        )

        active_run = mlflow.active_run()
        if active_run is None:
            return result

        client = mlflow.tracking.MlflowClient()

        try:
            for artifact in client.list_artifacts(
                active_run.info.run_id,
                destination_path,
            ):
                self._mirror_remote_artifact(
                    client,
                    active_run,
                    artifact,
                )
        except Exception:
            logger.exception("MLflow model succeeded, but local mirroring failed")
            self.manifests.update(
                active_run.info.run_id,
                recovery_required=True,
            )

        return result

    def _mirror_remote_artifact(
        self,
        client: Any,
        active_run: Any,
        artifact: Any,
    ) -> None:
        if artifact.is_dir:
            for child in client.list_artifacts(
                active_run.info.run_id,
                artifact.path,
            ):
                self._mirror_remote_artifact(
                    client,
                    active_run,
                    child,
                )
            return

        local_path = client.download_artifacts(
            active_run.info.run_id,
            artifact.path,
        )
        parent = str(Path(artifact.path).parent)

        mirrored = self.artifacts.mirror_file(
            local_path,
            active_run.info.experiment_id,
            active_run.info.run_id,
            None if parent == "." else parent,
        )
        mirrored["timestamp"] = timestamp_ms()
        self._record("artifact", mirrored, active_run)

    def _record_active(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        active_run = mlflow.active_run()

        if active_run is None:
            logger.warning(
                "MLflow write succeeded, but no active run was available "
                "for local mirroring"
            )
            return

        self._record(event_type, payload, active_run)

    def _record(
        self,
        event_type: str,
        payload: Dict[str, Any],
        active_run: Any,
    ) -> None:
        event_payload = {
            "run_id": active_run.info.run_id,
            "experiment_id": active_run.info.experiment_id,
        }
        event_payload.update(payload)

        try:
            event = self.writer.append(event_type, event_payload)
            self.manifests.update_for_event(
                active_run.info.run_id,
                event,
            )
        except Exception:
            logger.exception("MLflow write succeeded, but event mirroring failed")
            self.manifests.update(
                active_run.info.run_id,
                recovery_required=True,
            )

    def _get_console_run_context(self) -> Dict[str, Optional[str]]:
        """
        Return the current MLflow run context.

        The console artifacts are owned by the parent run, while active_run_id
        identifies the parent or nested trial that produced the log.
        """
        active = mlflow.active_run()

        active_run_id: Optional[str] = None
        active_experiment_id: Optional[str] = None

        if active is not None:
            active_run_id = str(active.info.run_id)
            active_experiment_id = str(active.info.experiment_id)

        return {
            "experiment_id": active_experiment_id,
            "parent_run_id": self._console_owner_run_id,
            "active_run_id": active_run_id,
        }


    def _write_console_line(
        self,
        stream: str,
        message: str,
    ) -> None:
        """Write one console line to run_logs.txt and run_logs.csv."""
        if message is None:
            return

        text = str(message).rstrip("\r\n")

        if not text:
            return

        lock = getattr(self, "_console_lock", None)

        if lock is None:
            return

        with lock:
            raw_fh = getattr(self, "_console_raw_fh", None)
            csv_fh = getattr(self, "_console_csv_fh", None)

            if raw_fh is None or csv_fh is None:
                return

            context = self._get_console_run_context()

            self._console_line_id += 1
            event_timestamp = int(time.time() * 1000)

            raw_fh.write(text + "\n")
            raw_fh.flush()

            writer = csv.writer(csv_fh)
            writer.writerow(
                [
                    self._console_line_id,
                    event_timestamp,
                    stream,
                    context["experiment_id"],
                    context["parent_run_id"],
                    context["active_run_id"],
                    text,
                ]
            )
            csv_fh.flush()


    def _report_console_capture_error(
        self,
        message: str,
        error: BaseException,
    ) -> None:
        """Write internal capture errors directly to the original stderr."""
        terminal = (
            getattr(self, "_orig_stderr", None)
            or getattr(sys, "__stderr__", None)
        )

        if terminal is None:
            return

        try:
            terminal.write(
                "[console-capture] {}: {}: {}\n".format(
                    message,
                    type(error).__name__,
                    error,
                )
            )
            terminal.flush()
        except Exception:
            pass


    def _console_pipe_reader(
        self,
        pipe_fd: int,
        saved_terminal_fd: int,
        stream_name: str,
    ) -> None:
        """
        Read redirected stdout or stderr until EOF.

        Output remains visible in the terminal and is also written to the
        parent run's console artifacts.
        """
        buffer = b""

        try:
            while True:
                try:
                    chunk = os.read(pipe_fd, 4096)
                except OSError as exc:
                    self._report_console_capture_error(
                        "{} reader failed".format(stream_name),
                        exc,
                    )
                    break

                if not chunk:
                    break

                try:
                    os.write(saved_terminal_fd, chunk)
                except OSError as exc:
                    self._report_console_capture_error(
                        "{} terminal mirror failed".format(stream_name),
                        exc,
                    )

                buffer += chunk

                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)

                    text = raw_line.decode(
                        "utf-8",
                        errors="replace",
                    ).rstrip("\r")

                    if text:
                        self._write_console_line(
                            stream_name,
                            text,
                        )

            if buffer:
                text = buffer.decode(
                    "utf-8",
                    errors="replace",
                ).rstrip("\r")

                if text:
                    self._write_console_line(
                        stream_name,
                        text,
                    )

        finally:
            try:
                os.close(pipe_fd)
            except OSError:
                pass


    def _start_console_capture(
        self,
        experiment_id: str,
        run_id: str,
        nested: bool = False,
    ) -> None:
        """
        Start console capture for the parent run.

        Nested runs share the parent's process-wide capture and must not create
        their own stdout/stderr redirection.
        """
        if nested:
            return
        
        run_id = str(run_id)
        experiment_id = str(experiment_id)

        # if nested:
        #     if self._console_td is not None:
        #         self._write_console_line(
        #             "system",
        #             "NESTED_RUN_STARTED run_id={}".format(run_id),
        #         )
        #     return

        if self._console_td is not None:
            if self._console_owner_run_id == run_id:
                return

            raise RuntimeError(
                "Console capture is already owned by parent run {!r}; "
                "cannot start capture for parent run {!r}.".format(
                    self._console_owner_run_id,
                    run_id,
                )
            )

        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix="console_{}_".format(run_id)
            )
        )

        csv_path = temporary_directory / "run_logs.csv"
        raw_path = temporary_directory / "run_logs.txt"

        csv_fh = open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        )

        raw_fh = open(
            raw_path,
            "w",
            encoding="utf-8",
        )

        writer = csv.writer(csv_fh)
        writer.writerow(
            [
                "line_id",
                "timestamp",
                "stream",
                "experiment_id",
                "parent_run_id",
                "active_run_id",
                "message",
            ]
        )
        csv_fh.flush()

        self._console_td = temporary_directory
        self._console_csv_fh = csv_fh
        self._console_raw_fh = raw_fh
        self._console_lock = threading.RLock()
        self._console_line_id = 0

        self._console_owner_run_id = run_id
        self._console_owner_experiment_id = experiment_id

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self._console_saved_stdout_fd = None
        self._console_saved_stderr_fd = None
        self._console_stdout_pipe_w = None
        self._console_stderr_pipe_w = None
        self._console_stdout_thread = None
        self._console_stderr_thread = None

        self._console_log_handler = None
        self._console_log_attached = []
        self._console_original_root_level = None
        self._warnings_showwarning_orig = None

        self._write_console_line(
            "system",
            "CAPTURE_STARTED",
        )

        try:
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)

            stdout_pipe_r, stdout_pipe_w = os.pipe()
            stderr_pipe_r, stderr_pipe_w = os.pipe()

            os.dup2(stdout_pipe_w, 1)
            os.dup2(stderr_pipe_w, 2)

            self._console_saved_stdout_fd = saved_stdout_fd
            self._console_saved_stderr_fd = saved_stderr_fd

            self._console_stdout_pipe_w = stdout_pipe_w
            self._console_stderr_pipe_w = stderr_pipe_w

            stdout_thread = threading.Thread(
                target=self._console_pipe_reader,
                args=(
                    stdout_pipe_r,
                    saved_stdout_fd,
                    "stdout",
                ),
                daemon=True,
                name="console-stdout-{}".format(run_id),
            )

            stderr_thread = threading.Thread(
                target=self._console_pipe_reader,
                args=(
                    stderr_pipe_r,
                    saved_stderr_fd,
                    "stderr",
                ),
                daemon=True,
                name="console-stderr-{}".format(run_id),
            )

            stdout_thread.start()
            stderr_thread.start()

            self._console_stdout_thread = stdout_thread
            self._console_stderr_thread = stderr_thread

        except Exception as exc:
            self._report_console_capture_error(
                "file-descriptor capture could not start",
                exc,
            )

            self._restore_console_file_descriptors()

            raise RuntimeError(
                "Could not initialize console capture."
            ) from exc

        class ConsoleArtifactHandler(logging.Handler):
            def __init__(self, parent: Any) -> None:
                super().__init__(level=logging.DEBUG)
                self.parent = parent

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    message = self.format(record)

                    self.parent._write_console_line(
                        "log",
                        message,
                    )
                except Exception:
                    self.handleError(record)

        root_logger = logging.getLogger()

        self._console_original_root_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        handler = ConsoleArtifactHandler(self)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | "
                "%(name)s | %(message)s"
            )
        )

        root_logger.addHandler(handler)
        self._console_log_handler = handler

        logger_names = [
            "mlflow",
            "mlflow.data",
            "codecarbon",
            "optuna",
            "xgboost",
            "whylogs",
        ]

        for logger_name in logger_names:
            library_logger = logging.getLogger(logger_name)

            if not library_logger.propagate:
                library_logger.addHandler(handler)
                self._console_log_attached.append(library_logger)

        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

        original_showwarning = warnings.showwarning
        self._warnings_showwarning_orig = original_showwarning

        def capture_warning(
            message: Any,
            category: Any,
            filename: str,
            lineno: int,
            file: Optional[Any] = None,
            line: Optional[str] = None,
        ) -> None:
            warning_text = warnings.formatwarning(
                message,
                category,
                filename,
                lineno,
                line,
            ).rstrip("\r\n")

            self._write_console_line(
                "warning",
                warning_text,
            )

            try:
                original_showwarning(
                    message,
                    category,
                    filename,
                    lineno,
                    file=file,
                    line=line,
                )
            except Exception as exc:
                self._report_console_capture_error(
                    "warning forwarding failed",
                    exc,
                )

        warnings.showwarning = capture_warning

        self._write_console_line(
            "system",
            "CAPTURE_READY experiment_id={} parent_run_id={}".format(
                experiment_id,
                run_id,
            ),
        )


    def _mark_nested_run_stopped(
        self,
        run_id: str,
        status: str = "FINISHED",
    ) -> None:
        """Record the end of a nested run without stopping parent capture."""
        if self._console_td is None:
            return

        self._write_console_line(
            "system",
            "NESTED_RUN_STOPPED run_id={} status={}".format(
                run_id,
                status,
            ),
        )


    def _remove_console_handlers(self) -> None:
        """Restore warnings and remove the custom logging handlers."""
        original_showwarning = getattr(
            self,
            "_warnings_showwarning_orig",
            None,
        )

        if original_showwarning is not None:
            warnings.showwarning = original_showwarning

        self._warnings_showwarning_orig = None

        handler = getattr(
            self,
            "_console_log_handler",
            None,
        )

        root_logger = logging.getLogger()

        if handler is not None:
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass

            for attached_logger in getattr(
                self,
                "_console_log_attached",
                [],
            ):
                try:
                    attached_logger.removeHandler(handler)
                except Exception:
                    pass

            try:
                handler.close()
            except Exception:
                pass

        self._console_log_handler = None
        self._console_log_attached = []

        original_root_level = getattr(
            self,
            "_console_original_root_level",
            None,
        )

        if original_root_level is not None:
            root_logger.setLevel(original_root_level)

        self._console_original_root_level = None


    def _restore_console_file_descriptors(self) -> None:
        """Restore stdout/stderr and close the capture pipe write ends."""
        try:
            sys.stdout.flush()
        except Exception:
            pass

        try:
            sys.stderr.flush()
        except Exception:
            pass

        saved_stdout_fd = getattr(
            self,
            "_console_saved_stdout_fd",
            None,
        )

        saved_stderr_fd = getattr(
            self,
            "_console_saved_stderr_fd",
            None,
        )

        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
            except OSError as exc:
                self._report_console_capture_error(
                    "could not restore stdout",
                    exc,
                )

        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
            except OSError as exc:
                self._report_console_capture_error(
                    "could not restore stderr",
                    exc,
                )

        stdout_pipe_w = getattr(
            self,
            "_console_stdout_pipe_w",
            None,
        )

        stderr_pipe_w = getattr(
            self,
            "_console_stderr_pipe_w",
            None,
        )

        if stdout_pipe_w is not None:
            try:
                os.close(stdout_pipe_w)
            except OSError:
                pass

        if stderr_pipe_w is not None:
            try:
                os.close(stderr_pipe_w)
            except OSError:
                pass

        self._console_stdout_pipe_w = None
        self._console_stderr_pipe_w = None


    def _stop_console_capture(
        self,
        run_id: Optional[str] = None,
        nested: bool = False,
        status: str = "FINISHED",
    ) -> None:
        """
        Stop capture only when the parent run ends.

        Nested runs only add a lifecycle marker and leave parent capture active.
        """
        if nested:
            if run_id is not None:
                self._mark_nested_run_stopped(
                    run_id=str(run_id),
                    status=status,
                )
            return

        temporary_directory = getattr(
            self,
            "_console_td",
            None,
        )

        if temporary_directory is None:
            return

        owner_run_id = getattr(
            self,
            "_console_owner_run_id",
            None,
        )

        owner_experiment_id = getattr(
            self,
            "_console_owner_experiment_id",
            None,
        )

        if (
            run_id is not None
            and owner_run_id is not None
            and str(run_id) != str(owner_run_id)
        ):
            raise RuntimeError(
                "Run {!r} cannot stop console capture owned by parent run {!r}.".format(
                    run_id,
                    owner_run_id,
                )
            )

        self._write_console_line(
            "system",
            "CAPTURE_STOPPING",
        )

        self._remove_console_handlers()
        self._restore_console_file_descriptors()

        stdout_thread = getattr(
            self,
            "_console_stdout_thread",
            None,
        )

        stderr_thread = getattr(
            self,
            "_console_stderr_thread",
            None,
        )

        if stdout_thread is not None:
            stdout_thread.join(timeout=10.0)

            if stdout_thread.is_alive():
                self._report_console_capture_error(
                    "stdout reader did not stop",
                    TimeoutError(
                        "stdout reader thread is still active"
                    ),
                )

        if stderr_thread is not None:
            stderr_thread.join(timeout=10.0)

            if stderr_thread.is_alive():
                self._report_console_capture_error(
                    "stderr reader did not stop",
                    TimeoutError(
                        "stderr reader thread is still active"
                    ),
                )

        self._console_stdout_thread = None
        self._console_stderr_thread = None

        saved_stdout_fd = getattr(
            self,
            "_console_saved_stdout_fd",
            None,
        )

        saved_stderr_fd = getattr(
            self,
            "_console_saved_stderr_fd",
            None,
        )

        if saved_stdout_fd is not None:
            try:
                os.close(saved_stdout_fd)
            except OSError:
                pass

        if saved_stderr_fd is not None:
            try:
                os.close(saved_stderr_fd)
            except OSError:
                pass

        self._console_saved_stdout_fd = None
        self._console_saved_stderr_fd = None

        self._write_console_line(
            "system",
            "CAPTURE_STOPPED",
        )

        csv_fh = getattr(
            self,
            "_console_csv_fh",
            None,
        )

        raw_fh = getattr(
            self,
            "_console_raw_fh",
            None,
        )

        if csv_fh is not None:
            try:
                csv_fh.flush()
                csv_fh.close()
            except Exception as exc:
                self._report_console_capture_error(
                    "could not close CSV console log",
                    exc,
                )

        if raw_fh is not None:
            try:
                raw_fh.flush()
                raw_fh.close()
            except Exception as exc:
                self._report_console_capture_error(
                    "could not close text console log",
                    exc,
                )

        self._console_csv_fh = None
        self._console_raw_fh = None

        try:
            if owner_run_id is not None:
                client = MlflowClient()

                client.log_artifacts(
                    str(owner_run_id),
                    str(temporary_directory),
                    artifact_path=mlflow_artifact_path(
                        "run_logs"
                    ),
                )

                if (
                    owner_experiment_id is not None
                    and getattr(self, "artifacts", None) is not None
                ):
                    for artifact in self.artifacts.mirror_directory(
                        str(temporary_directory),
                        str(owner_experiment_id),
                        str(owner_run_id),
                        mlflow_artifact_path("run_logs"),
                    ):
                        artifact["timestamp"] = timestamp_ms()

                        try:
                            owner_run = client.get_run(
                                str(owner_run_id)
                            )

                            self._record(
                                "artifact",
                                artifact,
                                owner_run,
                            )
                        except Exception:
                            pass

        except Exception as exc:
            self._report_console_capture_error(
                "could not upload console artifacts",
                exc,
            )

        finally:
            try:
                shutil.rmtree(
                    str(temporary_directory)
                )
            except Exception as exc:
                self._report_console_capture_error(
                    "could not remove console temporary directory",
                    exc,
                )

            self._console_td = None
            self._console_lock = None
            self._console_line_id = 0

            self._console_owner_run_id = None
            self._console_owner_experiment_id = None

            self._orig_stdout = None
            self._orig_stderr = None
            
tracker = Tracker()