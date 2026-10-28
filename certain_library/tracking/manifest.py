"""Per-run manifest storage and unfinished-run recovery."""

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mlflow.tracking import MlflowClient

FINAL_STATUSES = {"FINISHED", "FAILED", "KILLED"}


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """Manage atomic per-run manifest files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.runs_dir = self.root / "manifest" / "runs"
        self._thread_lock = threading.RLock()

    def path_for(self, run_id: str) -> Path:
        return self.runs_dir / "{}.json".format(run_id)

    def load(self, run_id: str) -> Dict[str, Any]:
        path = self.path_for(run_id)

        if not path.exists():
            return {
                "run_id": run_id,
                "status": "RUNNING",
                "last_event_id": 0,
                "last_synced_at": None,
                "artifact_count": 0,
                "metric_count": 0,
                "param_count": 0,
                "tag_count": 0,
                "is_final": False,
            }

        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def update_for_event(
        self, run_id: str, event: Dict[str, Any], **changes: Any
    ) -> Dict[str, Any]:
        """Update a manifest after an event has been persisted."""
        with self._thread_lock:
            manifest = self.load(run_id)
            counter_name = "{}_count".format(event["event_type"])

            if counter_name in manifest:
                manifest[counter_name] += 1

            manifest.update(changes)
            manifest["last_event_id"] = event["event_id"]
            manifest["last_synced_at"] = utc_now()
            self._write_atomic(run_id, manifest)
            return manifest

    def update(self, run_id: str, **changes: Any) -> Dict[str, Any]:
        """Update manifest fields without incrementing an event counter."""
        with self._thread_lock:
            manifest = self.load(run_id)
            manifest.update(changes)
            manifest["last_synced_at"] = utc_now()
            self._write_atomic(run_id, manifest)
            return manifest

    def unfinished(self) -> Iterable[Dict[str, Any]]:
        """Yield manifests that have not reached a terminal status."""
        if not self.runs_dir.exists():
            return

        for path in self.runs_dir.glob("*.json"):
            try:
                with path.open(encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            if not manifest.get("is_final", False):
                yield manifest

    def _write_atomic(
        self,
        run_id: str,
        manifest: Dict[str, Any],
    ) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=str(self.runs_dir),
            prefix=".{}.".format(run_id),
            suffix=".tmp",
        )

        try:
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_name, self.path_for(run_id))
        finally:
            try:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            except Exception:
                # Best-effort cleanup; ignore races where the temp file
                # has already been removed or moved.
                pass


def recover_unfinished_runs(
    root: Path = Path("certain"),
    client: Optional[MlflowClient] = None,
) -> Dict[str, int]:
    """Refresh unfinished manifests using current MLflow run state."""
    manifest_store = ManifestStore(root)
    mlflow_client = client or MlflowClient()

    recovered = 0
    failed = 0

    # Prefer artifact-based metadata when available. This allows recovery
    # without contacting the MLflow tracking DB.
    mlflow_artifacts = Path(os.getenv("MLFLOW_ARTIFACTS", "mlruns"))

    for manifest in manifest_store.unfinished():
        run_id = manifest["run_id"]
        updated = False

        # Try to find run_metadata.json under the artifacts tree
        try:
            # The artifacts layout is <artifacts_root>/<experiment_id>/<run_id>/artifacts/metadata/run_metadata.json
            experiment_id = manifest.get("experiment_id")
            if experiment_id is not None:
                metadata_path = (
                    mlflow_artifacts
                    / str(experiment_id)
                    / str(run_id)
                    / "artifacts"
                    / "metadata"
                    / "run_metadata.json"
                )
                if metadata_path.exists():
                    with metadata_path.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)

                    status = data.get("status")
                    is_final = status in FINAL_STATUSES

                    manifest_store.update(
                        run_id,
                        experiment_id=data.get("experiment_id", experiment_id),
                        status=status,
                        end_time=data.get("end_time"),
                        is_final=is_final,
                        recovery_required=False,
                    )
                    recovered += 1
                    updated = True
        except Exception:
            # ignore artifact read failures and fall back to client
            updated = False

        if updated:
            continue

        # Fallback: ask the MLflow tracking server for run status
        try:
            run = mlflow_client.get_run(run_id)
            status = run.info.status
            is_final = status in FINAL_STATUSES

            manifest_store.update(
                run_id,
                experiment_id=run.info.experiment_id,
                status=status,
                end_time=run.info.end_time,
                is_final=is_final,
                recovery_required=False,
            )
            recovered += 1
        except Exception:
            manifest_store.update(
                run_id,
                recovery_required=True,
            )
            failed += 1

    return {
        "recovered": recovered,
        "failed": failed,
    }
