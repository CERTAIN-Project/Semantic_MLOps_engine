import os
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import mlflow
from pathlib import Path

try:
    import requests
except Exception:
    requests = None


def _active_run_info() -> Optional[Dict[str, str]]:
    try:
        run = mlflow.active_run()
        if run is None:
            return None
        return {"experiment_id": run.info.experiment_id, "run_id": run.info.run_id}
    except Exception:
        return None


def save_params_as_artifact(
    id: str,
    params: Dict[str, str],
    type_of_params: str = "run",
) -> str:
    """
    Save MLflow parameters as a JSON artifact under:

        certain/metadata/run_params.json

    Returns the local temporary path only while writing. Since the temporary
    directory is removed afterward, an empty string is returned after upload.
    """

    normalized_params: Dict[str, str] = {
        str(key): str(value) for key, value in params.items()
    }

    record = {
        f"{type_of_params}_id": str(id),
        f"{type_of_params}_params": normalized_params,
        "captured_at": datetime.utcnow().isoformat(),
    }

    active = _active_run_info()

    if active is None:
        return ""

    with tempfile.TemporaryDirectory() as temporary_directory:
        target = os.path.join(
            temporary_directory,
            f"{type_of_params}_params.json",
        )

        with open(target, "w", encoding="utf-8") as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        mlflow.log_artifact(
            target,
            artifact_path="certain/metadata",
        )

    return ""


def save_metrics_as_artifact(
    run_id: str,
    metrics: Dict[str, list[Dict[str, Any]]],
) -> str:
    """
    Save complete MLflow metric history under:

        certain/metadata/run_metrics.json
    """

    record = {
        "run_id": str(run_id),
        "run_metrics": metrics,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    if _active_run_info() is None:
        return ""

    with tempfile.TemporaryDirectory() as temporary_directory:
        target = os.path.join(
            temporary_directory,
            "run_metrics.json",
        )

        with open(target, "w", encoding="utf-8") as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        mlflow.log_artifact(
            target,
            artifact_path="certain/metadata",
        )

    return ""


def save_inputs_as_artifact(
    run_id: str,
    inputs: List[Dict[str, Any]],
) -> str:
    """
    Save logged MLflow inputs as:

        certain/metadata/run_inputs.json
    """

    record = {
        "run_inputs": inputs,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        target = os.path.join(
            temporary_directory,
            "inputs.json",
        )

        with open(target, "w", encoding="utf-8") as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        mlflow.log_artifact(
            target,
            artifact_path="certain/metadata",
        )

    return ""


def _extract_run_inputs(run: Any) -> List[Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []

    run_inputs = getattr(run.inputs, "dataset_inputs", [])

    for dataset_input in run_inputs:
        dataset = dataset_input.dataset

        inputs.append(
            {
                "dataset_id": getattr(dataset, "digest", None),
                "dataset_name": getattr(dataset, "name", None),
                "dataset_source": getattr(dataset, "source", None),
                "dataset_schema": getattr(dataset, "schema", None),
                "dataset_profile": getattr(dataset, "profile", None),
                "tags": {
                    tag.key: tag.value for tag in getattr(dataset_input, "tags", [])
                },
            }
        )

    return inputs


def save_resources_as_artifact(
    run_id: str,
    resources: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    Save the complete system-resource history for an MLflow run as:

        certain/metadata/run_resources.json
    """

    active = mlflow.active_run()

    if active is None:
        raise RuntimeError("No active MLflow run.")

    if active.info.run_id != run_id:
        raise RuntimeError("The supplied run_id does not match the active MLflow run.")

    normalized_resources: Dict[str, List[Dict[str, Any]]] = {}

    for resource_name, history in resources.items():
        normalized_resources[str(resource_name)] = [
            {
                "value": float(item["value"]),
                "step": int(item.get("step", 0)),
                "timestamp": int(item["timestamp"]),
            }
            for item in history
        ]

    record = {
        "run_id": str(run_id),
        "run_resources": normalized_resources,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        target = os.path.join(
            temporary_directory,
            "run_resources.json",
        )

        with open(target, "w", encoding="utf-8") as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        mlflow.log_artifact(
            target,
            artifact_path="certain/metadata",
        )

    return ""


def save_tags_as_artifact(
    id: str,
    tags: Dict[str, str],
    type_of_tags: str = "experiment",
) -> str:
    """Write experiment-level tags to a canonical artifact 'experiment_tags.json' under certain/metadata.

    This function will try to (in order):
      1) attach the file to the active MLflow run via mlflow.log_artifacts
      2) write directly into the local artifacts root if it's mounted

    Returns the path to the local file when written; otherwise returns an empty string.
    """

    tags = {k: v for k, v in tags.items() if "mlflow" not in k}

    record = {
        f"{type_of_tags}_id": str(id),
        f"{type_of_tags}_tags": tags,
        "captured_at": datetime.utcnow().isoformat(),
    }

    written_path = ""

    active = _active_run_info()
    if active is not None:
        import tempfile as _tmp

        with _tmp.TemporaryDirectory() as td:
            target = os.path.join(td, f"{type_of_tags}_tags.json")
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2)

            mlflow.log_artifacts(td, artifact_path="certain/metadata")

    return written_path


def collect_runtime_environment() -> Dict[str, Any]:
    """Collect a small snapshot of the runtime environment.

    Includes Python version, platform, and best-effort docker detection.
    """
    env = {
        "captured_at": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CI") or k.startswith("MLFLOW") or k in ("HOSTNAME", "USER")
        },
        "in_docker": False,
    }

    # Simple docker detection heuristics
    try:
        if os.path.exists("/.dockerenv"):
            env["in_docker"] = True
        else:
            # Check cgroup for docker indicator
            cgroup = ""
            try:
                with open("/proc/1/cgroup", "r") as fh:
                    cgroup = fh.read()
            except Exception:
                cgroup = ""
            if "docker" in cgroup or "kubepods" in cgroup:
                env["in_docker"] = True
    except Exception:
        pass

    return env


def save_runtime_env_as_artifact(
    runtime_record: Dict[str, Any],
    experiment_id: Optional[str] = None,
    artifacts_root: Optional[str] = None,
) -> str:
    """Write runtime environment record to 'runtime_env.json' under certain/metadata.

    Returns local file path when written, otherwise empty string.
    """
    # Normalize keys: accept deployment, deployment_id, model, model_id
    record = runtime_record.copy()
    record.setdefault("captured_at", datetime.utcnow().isoformat())
    if "deployment" in record and "deployment_id" not in record:
        record["deployment_id"] = record.get("deployment")
    if "model" in record and "model_id" not in record:
        record["model_id"] = record.get("model")

    # Only persist runtime environment artifacts when tied to a deployment
    # (deployment_id and model_id present), otherwise skip. This enforces
    # the invariant that runtime_environment rows map to model_deployed parents.
    if not record.get("deployment_id") or not record.get("model_id"):
        # If an active run exists and it contains an explicit run_id we could
        # try to resolve deployment mapping later during sync. But the writer
        # should be invoked at deployment start with deployment info.
        return ""

    written_path = ""

    try:
        # Ensure the record contains the canonical experiment id if possible
        canonical_experiment_id = None
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            # If an experiment_id argument was provided, try to resolve it
            if experiment_id is not None:
                try:
                    exp = client.get_experiment(experiment_id)
                    if exp is not None:
                        canonical_experiment_id = exp.experiment_id
                except Exception:
                    try:
                        exp = client.get_experiment_by_name(experiment_id)
                        if exp is not None:
                            canonical_experiment_id = exp.experiment_id
                    except Exception:
                        canonical_experiment_id = None
        except Exception:
            canonical_experiment_id = None

        active = _active_run_info()
        # prefer explicit experiment_id, else active run
        if canonical_experiment_id is None and experiment_id is not None:
            canonical_experiment_id = str(experiment_id)
        if canonical_experiment_id is None and active is not None:
            canonical_experiment_id = active.get("experiment_id")

        # write artifact into a temp dir and mlflow.log_artifacts under certain/metadata
        if active is not None or canonical_experiment_id is not None:
            import tempfile as _tmp

            with _tmp.TemporaryDirectory() as td:
                target = os.path.join(td, "runtime_env.json")
                # If we have an active run, include its run_id so downstream
                # sync can map to deployment/model ids.
                if active is not None:
                    record.setdefault("run_id", active.get("run_id"))

                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, indent=2)
                try:
                    mlflow.log_artifacts(td, artifact_path="certain/metadata")
                except Exception:
                    pass
    except Exception:
        pass

    # Direct write to artifacts root
    try:
        ar = (
            artifacts_root
            or os.environ.get("MLFLOW_ARTIFACTS")
            or os.environ.get("MLFLOW_ARTIFACT_ROOT")
            or "/app/mlruns"
        )
        if os.path.isdir(ar):
            # If experiment_id provided, write under that experiment folder
            if experiment_id is None:
                # Try active run to infer experiment
                active = _active_run_info()
                if active is not None:
                    experiment_id = active.get("experiment_id")

            if experiment_id is not None:
                folder = os.path.join(
                    ar, str(experiment_id), "artifacts", "certain", "metadata"
                )
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(folder, "runtime_env.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, ensure_ascii=False, indent=2)
                written_path = path
    except Exception:
        pass

    return written_path


def _compute_path_size_and_checksum(path: str):
    """Return (size_bytes, sha256) for a file or directory.

    For directories the size is the sum of contained file sizes and the
    checksum is the sha256 of the concatenation of file-level sha256s in
    deterministic sorted order. Best-effort; errors return (None, None).
    """
    import hashlib

    try:
        if os.path.isfile(path):
            size = os.path.getsize(path)
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return size, h.hexdigest()

        if os.path.isdir(path):
            total = 0
            file_hashes = []
            for root, _, files in os.walk(path):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    try:
                        total += os.path.getsize(fpath)
                        h = hashlib.sha256()
                        with open(fpath, "rb") as fh:
                            for chunk in iter(lambda: fh.read(8192), b""):
                                h.update(chunk)
                        file_hashes.append(h.hexdigest())
                    except Exception:
                        continue
            # deterministic aggregate checksum
            agg = hashlib.sha256()
            for fh in file_hashes:
                agg.update(fh.encode("utf-8"))
            return total, agg.hexdigest()
    except Exception:
        return None, None

    return None, None


def get_dataset_size_bytes(dataset_path: str) -> int:
    """Compute total size of a file or directory using pathlib."""
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        total = 0
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except Exception:
                continue
        return total
    return 0


def format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def save_dataset_manifest(
    run_id: Optional[str],
    files_or_path,
    experiment_id: Optional[str] = None,
    artifacts_root: Optional[str] = None,
    write_manifest: bool = True,
):
    """Write a small dataset manifest into artifacts under certain/dataset.

    Parameters:
        run_id: MLflow run id (if available) used to attach via mlflow client.
        files_or_path: either a single file path, directory path, or list of
            file paths that make up the dataset. If a path is provided that
            exists on disk we compute sizes and checksums. Otherwise we
            store the provided URI(s) as-is.
        experiment_id: optional experiment id used when writing directly to
            the artifacts root.
        artifacts_root: optional local artifacts root to write into when
            mounted. Falls back to MLFLOW_ARTIFACTS env or /app/mlruns.

    Returns: path written on local filesystem when written, else empty string.
    """
    manifest = {
        "captured_at": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "files": [],
    }

    # Normalize input
    if files_or_path is None:
        return ""
    if isinstance(files_or_path, (str,)):
        items = [files_or_path]
    else:
        try:
            items = list(files_or_path)
        except Exception:
            items = [str(files_or_path)]

    total_size = 0
    for item in items:
        entry = {"path": item}
        if isinstance(item, str) and os.path.exists(item):
            size, checksum = _compute_path_size_and_checksum(item)
            if size is not None:
                entry["size_bytes"] = int(size)
                total_size += int(size)
            if checksum:
                entry["sha256"] = checksum
        manifest["files"].append(entry)

    manifest["total_size_bytes"] = int(total_size)

    # Fallback heuristics when computed size is zero:
    # - If items contains HTTP/HTTPS URLs, try a HEAD request to read Content-Length
    # - If files_or_path was a pandas DataFrame-like object, serialize to CSV in-memory
    if manifest.get("total_size_bytes", 0) == 0:
        try:
            # Try to detect a DataFrame-like object in the original input
            import pandas as _pd

            if not isinstance(files_or_path, (str, list, tuple)) and hasattr(
                files_or_path, "to_csv"
            ):
                try:
                    csv_bytes = files_or_path.to_csv(index=False).encode("utf-8")
                    manifest["total_size_bytes"] = len(csv_bytes)
                except Exception:
                    pass
        except Exception:
            # pandas not available or serialization failed — continue
            pass

        # Try HTTP HEAD for URL entries
        if manifest.get("total_size_bytes", 0) == 0 and requests is not None:
            for it in items:
                try:
                    if isinstance(it, str) and it.startswith("http"):
                        h = requests.head(it, allow_redirects=True, timeout=5)
                        if h is not None and h.status_code == 200:
                            cl = h.headers.get("Content-Length") or h.headers.get(
                                "content-length"
                            )
                            if cl:
                                try:
                                    manifest["total_size_bytes"] = int(cl)
                                    break
                                except Exception:
                                    continue
                except Exception:
                    continue

    written_path = ""

    # Try to upload via MLflow (attach to the run's artifacts). Prefer the
    # MlflowClient.log_artifacts API which accepts an explicit run_id. If a
    # run_id wasn't provided, fall back to using the active run context.
    try:
        from mlflow.tracking import MlflowClient
        import mlflow

        client = MlflowClient()

        # decide which run_id to use for upload
        upload_run_id = run_id
        if upload_run_id is None:
            active = _active_run_info()
            if active is not None:
                upload_run_id = active.get("run_id")

        import tempfile as _tmp

        with _tmp.TemporaryDirectory() as td:
            # write metadata (always create data.json in temp dir)
            try:
                meta = {
                    "run_id": run_id,
                    "captured_at": manifest.get("captured_at"),
                    "data_location": (
                        manifest["files"][0].get("path")
                        if manifest.get("files")
                        else ""
                    ),
                    "data_size": manifest.get("total_size_bytes", 0),
                    "human_readable": format_size(manifest.get("total_size_bytes", 0)),
                }
            except Exception:
                meta = {
                    "run_id": run_id,
                    "captured_at": manifest.get("captured_at"),
                    "data_location": "",
                    "data_size": 0,
                    "human_readable": "0 B",
                }

            meta_target = os.path.join(td, "data.json")
            try:
                with open(meta_target, "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # If caller requested the full manifest, write it as well into the temp dir
            if write_manifest:
                try:
                    manifest_target = os.path.join(td, "data_manifest.json")
                    with open(manifest_target, "w", encoding="utf-8") as fh:
                        json.dump(manifest, fh, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            try:
                if upload_run_id is not None:
                    # upload metadata under certain/metadata (always)
                    try:
                        client.log_artifacts(
                            upload_run_id, td, artifact_path="certain/metadata"
                        )
                    except Exception:
                        pass
                    # optionally upload manifest under certain/dataset
                    if write_manifest:
                        try:
                            client.log_artifacts(
                                upload_run_id, td, artifact_path="certain/dataset"
                            )
                        except Exception:
                            pass
                else:
                    # no run_id available — fall back to mlflow.log_artifacts
                    try:
                        mlflow.log_artifacts(td, artifact_path="certain/metadata")
                    except Exception:
                        pass
                    if write_manifest:
                        try:
                            mlflow.log_artifacts(td, artifact_path="certain/dataset")
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        # mlflow client not available — continue with direct write below
        pass

    # Try to write directly to the artifacts root if provided / discoverable
    try:
        ar = (
            artifacts_root
            or os.environ.get("MLFLOW_ARTIFACTS")
            or os.environ.get("MLFLOW_ARTIFACT_ROOT")
            or "/app/mlruns"
        )
        if os.path.isdir(ar):
            # Try to infer experiment_id and run_id from active run if missing
            active = _active_run_info()
            if experiment_id is None and active is not None:
                experiment_id = active.get("experiment_id")
            # prefer explicit run_id arg if provided; else try active run
            write_run_id = run_id
            if write_run_id is None and active is not None:
                write_run_id = active.get("run_id")

            # If we have both experiment and run, write under the run-scoped
            # artifacts folder so layout matches git_metadata (which uses
            # <artifacts_root>/<experiment_id>/<run_id>/artifacts/certain/metadata)
            if experiment_id is not None and write_run_id is not None:
                base = os.path.join(
                    ar, str(experiment_id), str(write_run_id), "artifacts", "certain"
                )
            elif experiment_id is not None:
                # fallback to older layout (experiment-level artifacts)
                base = os.path.join(ar, str(experiment_id), "artifacts", "certain")
            else:
                base = None

            if base is not None:
                # write manifest
                ds_folder = os.path.join(base, "dataset")
                os.makedirs(ds_folder, exist_ok=True)
                path = os.path.join(ds_folder, "data_manifest.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, ensure_ascii=False, indent=2)
                written_path = path

                # write metadata next to other metadata artifacts like git_metadata.json
                meta_folder = os.path.join(base, "metadata")
                os.makedirs(meta_folder, exist_ok=True)
                meta_path = os.path.join(meta_folder, "data.json")
                try:
                    meta = {
                        "run_id": run_id,
                        "captured_at": manifest.get("captured_at"),
                        "data_location": (
                            manifest["files"][0].get("path")
                            if manifest.get("files")
                            else ""
                        ),
                        "data_size": manifest.get("total_size_bytes", 0),
                        "human_readable": format_size(
                            manifest.get("total_size_bytes", 0)
                        ),
                    }
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta, mf, ensure_ascii=False, indent=2)
                except Exception:
                    pass
    except Exception:
        pass

    return written_path
