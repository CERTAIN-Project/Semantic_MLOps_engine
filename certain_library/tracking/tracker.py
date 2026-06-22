"""MLflow-first tracker with a portable local mirror."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from .artifact_mirror import ArtifactMirror
from .manifest import FINAL_STATUSES, ManifestStore
from .metadata_writer import MetadataWriter, timestamp_ms

logger = logging.getLogger(__name__)


def mirror_root() -> Path:
    """Return the configured CERTAIN mirror location."""
    return Path(os.getenv("CERTAIN_MIRROR_ROOT", "certain"))


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

    def set_tracking_uri(self, uri: str) -> None:
        mlflow.set_tracking_uri(uri)

    def set_experiment(self, experiment_name: str, **kwargs: Any) -> Any:
        return mlflow.set_experiment(experiment_name, **kwargs)

    def create_experiment(self, name: str, **kwargs: Any) -> str:
        return mlflow.create_experiment(name, **kwargs)

    def start_run(self, *args: Any, **kwargs: Any) -> MirroredRun:
        """Start an MLflow run, then record its local run event."""
        active_run = mlflow.start_run(*args, **kwargs)
        run_info = active_run.info

        event = self.writer.append(
            "run",
            {
                "action": "started",
                "run_id": run_info.run_id,
                "experiment_id": run_info.experiment_id,
                "run_name": run_info.run_name,
                "status": run_info.status,
                "start_time": run_info.start_time,
                "artifact_uri": run_info.artifact_uri,
            },
        )

        self.manifests.update_for_event(
            run_info.run_id,
            event,
            experiment_id=run_info.experiment_id,
            status=run_info.status,
            start_time=run_info.start_time,
            artifact_uri=run_info.artifact_uri,
            is_final=False,
        )

        for key, value in (kwargs.get("tags") or {}).items():
            self._record(
                "tag",
                {
                    "key": key,
                    "value": value,
                },
                active_run,
            )

        return MirroredRun(self, active_run)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the MLflow run and mirror MLflow's resulting status.

        Instead of querying the MLflow tracking DB for final run state, this
        method will aggregate local mirror events for the run, produce a
        run-level metadata JSON plus an events JSONL containing only the run's
        events, and upload those files into the run's artifact tree under
        `metadata/`. This allows downstream services to reconstruct run
        metadata from artifacts alone.
        """
        import json
        import shutil
        import tempfile

        active_run = mlflow.active_run()

        if active_run is None:
            mlflow.end_run(status=status)
            return

        run_id = active_run.info.run_id
        experiment_id = active_run.info.experiment_id
        artifact_uri = getattr(active_run.info, "artifact_uri", None)
        run_name = getattr(active_run.info, "run_name", None)
        start_time = getattr(active_run.info, "start_time", None)

        # End the MLflow run first (MLflow is still the source of truth).
        mlflow.end_run(status=status)

        # Aggregate local mirror events for this run.
        try:
            events_dir: Path = self.writer.events_dir
            events_dir.mkdir(parents=True, exist_ok=True)

            run_events = []
            counts = {
                "param_count": 0,
                "metric_count": 0,
                "tag_count": 0,
                "artifact_count": 0,
            }

            from .metadata_writer import EVENT_FILES as _EVENT_FILES

            for filename in _EVENT_FILES.values():
                path = events_dir / filename
                if not path.exists():
                    continue

                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue

                        if ev.get("run_id") != run_id:
                            continue

                        run_events.append(ev)
                        et = (
                            ev.get("event_type")
                            or ev.get("action")
                            or ev.get("event_type")
                        )
                        # increment counters based on event_type
                        if ev.get("event_type") == "param":
                            counts["param_count"] += 1
                        elif ev.get("event_type") == "metric":
                            counts["metric_count"] += 1
                        elif ev.get("event_type") == "tag":
                            counts["tag_count"] += 1
                        elif ev.get("event_type") == "artifact":
                            counts["artifact_count"] += 1

            # Build run-level metadata
            final_status = status
            end_time = None
            # Inspect run events for an explicit ended event
            for ev in run_events:
                if ev.get("event_type") == "run" and ev.get("action") == "ended":
                    final_status = ev.get("status", final_status)
                    end_time = ev.get("end_time")

            # If no explicit end_time found in events, use current timestamp so
            # manifests record a concrete end_time.
            if end_time is None:
                end_time = timestamp_ms()

            run_metadata = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "run_name": run_name,
                "status": final_status,
                "start_time": start_time,
                "end_time": end_time,
                "artifact_uri": artifact_uri,
            }
            run_metadata.update(counts)

            # Write metadata and events into a temporary directory and upload as artifacts
            tmpdir = Path(tempfile.mkdtemp(prefix=f"certain_{run_id}_"))
            try:
                metadata_path = tmpdir / "run_metadata.json"
                events_path = tmpdir / "events.jsonl"

                with metadata_path.open("w", encoding="utf-8") as handle:
                    json.dump(run_metadata, handle, indent=2, sort_keys=True)
                    handle.write("\n")

                with events_path.open("w", encoding="utf-8") as handle:
                    for ev in run_events:
                        handle.write(json.dumps(ev, default=str, sort_keys=True) + "\n")

                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                # upload the metadata folder under artifact path `metadata`
                client.log_artifacts(run_id, str(tmpdir), artifact_path="metadata")

            finally:
                try:
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass

            # Record the ended event locally and update the manifest
            event = self.writer.append(
                "run",
                {
                    "action": "ended",
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "status": final_status,
                    "end_time": end_time,
                },
            )

            self.manifests.update_for_event(
                run_id,
                event,
                status=final_status,
                end_time=end_time,
                is_final=final_status in FINAL_STATUSES,
                recovery_required=False,
            )

        except Exception:
            logger.exception("Failed to serialize run metadata into artifacts")
            self.manifests.update(
                run_id,
                recovery_required=True,
            )
            return

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)
        self._record_active(
            "param",
            {
                "key": key,
                "value": value,
            },
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

        for key, value in params.items():
            self._record_active(
                "param",
                {
                    "key": key,
                    "value": value,
                },
            )

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        mlflow_arguments = {}

        if step is not None:
            mlflow_arguments["step"] = step
        if timestamp is not None:
            mlflow_arguments["timestamp"] = timestamp

        mlflow.log_metric(key, value, **mlflow_arguments)

        self._record_active(
            "metric",
            {
                "key": key,
                "value": float(value),
                "step": step if step is not None else 0,
                "timestamp": (timestamp if timestamp is not None else timestamp_ms()),
            },
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        mlflow_arguments = {}

        if step is not None:
            mlflow_arguments["step"] = step
        if timestamp is not None:
            mlflow_arguments["timestamp"] = timestamp

        mlflow.log_metrics(metrics, **mlflow_arguments)

        event_timestamp = timestamp if timestamp is not None else timestamp_ms()

        for key, value in metrics.items():
            self._record_active(
                "metric",
                {
                    "key": key,
                    "value": float(value),
                    "step": step if step is not None else 0,
                    "timestamp": event_timestamp,
                },
            )

    def set_tag(self, key: str, value: Any) -> None:
        mlflow.set_tag(key, value)
        self._record_active("tag", {"key": key, "value": value})

    def set_tags(self, tags: Dict[str, Any]) -> None:
        mlflow.set_tags(tags)

        for key, value in tags.items():
            self._record_active("tag", {"key": key, "value": value})

    def log_input(
        self,
        dataset: Any,
        context: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        mlflow.log_input(dataset, context=context, tags=tags)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log to MLflow first, then copy and checksum the artifact."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        active_run = mlflow.active_run()

        if active_run is None:
            return

        try:
            artifact = self.artifacts.mirror_file(
                local_path,
                active_run.info.experiment_id,
                active_run.info.run_id,
                artifact_path,
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
        mlflow.log_artifacts(
            local_directory,
            artifact_path=artifact_path,
        )
        active_run = mlflow.active_run()

        if active_run is None:
            return

        try:
            for artifact in self.artifacts.mirror_directory(
                local_directory,
                active_run.info.experiment_id,
                active_run.info.run_id,
                artifact_path,
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
        result = mlflow.xgboost.log_model(
            xgb_model=model, artifact_path=artifact_path, **kwargs
        )

        active_run = mlflow.active_run()
        if active_run is None:
            return result

        client = mlflow.tracking.MlflowClient()

        try:
            for artifact in client.list_artifacts(
                active_run.info.run_id,
                artifact_path,
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


tracker = Tracker()
