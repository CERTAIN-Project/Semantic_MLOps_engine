import os
import subprocess
from typing import Optional, Dict

import mlflow
import json
from datetime import datetime


def _run_git_command(args, repo_path: str = "/app") -> Optional[str]:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def save_git_metadata(
    experiment_id: str,
    run_id: str,
    repo_path: str = "/app",
    artifacts_root: str = "/app/mlruns",
) -> str:
    """
    Collect Git metadata from the repository and save it under the
    MLflow artifact directory:

        <artifacts_root>/<experiment_id>/<run_id>/artifacts/certain/metadata/git_metadata.json
    """

    # Use get_git_metadata to build the metadata dict (keeps logic in one place)
    metadata = get_git_metadata(repo_path)
    # Add timestamp of capture
    metadata["captured_at"] = datetime.utcnow().isoformat()

    metadata_dir = os.path.join(
        artifacts_root,
        str(experiment_id),
        str(run_id),
        "artifacts",
        "certain",
        "metadata",
    )

    os.makedirs(metadata_dir, exist_ok=True)

    metadata_file = os.path.join(metadata_dir, "git_metadata.json")

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata_file


def log_git_metadata(repo_path: str = "/app") -> Dict[str, Optional[str]]:
    """Log Git metadata into the active MLflow run as tags."""
    """Save git metadata as an artifact file named 'git_metadata.json'.

    This function intentionally does NOT set MLflow tags (per request).
    It will attempt two things (best-effort):
      1. If an MLflow run is active, write a local temp dir containing
         'git_metadata.json' and call mlflow.log_artifacts(tmpdir,
         artifact_path='certain/metadata') so the artifact is uploaded with
         the correct filename.
      2. If the local artifacts root is mounted into the container, also
         write the file directly into
         <artifacts_root>/<experiment_id>/<run_id>/artifacts/certain/metadata/git_metadata.json
         by calling save_git_metadata(). This is best-effort and will not
         raise on failure.
    """

    metadata = get_git_metadata(repo_path)

    # Ensure minimal metadata exists
    if not metadata:
        return metadata

    # 1) Upload via mlflow if there is an active run
    try:
        active = mlflow.active_run()
        if active is not None:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                target = os.path.join(td, "git_metadata.json")
                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(metadata, fh, ensure_ascii=False, indent=2)
                # Upload the directory so the filename is preserved
                try:
                    mlflow.log_artifacts(td, artifact_path="certain/metadata")
                except Exception:
                    # best-effort: don't raise if upload fails
                    pass
    except Exception:
        # No active run or mlflow client misconfigured — continue to local write
        pass

    # 2) Also attempt to write directly into the local artifacts root if mounted
    try:
        ar = (
            os.environ.get("MLFLOW_ARTIFACTS")
            or os.environ.get("MLFLOW_ARTIFACT_ROOT")
            or "/app/mlruns"
        )
        active = None
        try:
            active = mlflow.active_run()
        except Exception:
            active = None

        if active is not None:
            exp_id = active.info.experiment_id
            run_id = active.info.run_id
            try:
                save_git_metadata(
                    exp_id, run_id, repo_path=repo_path, artifacts_root=ar
                )
            except Exception:
                # best-effort; ignore filesystem write failures
                pass
    except Exception:
        pass

    return metadata


def get_git_metadata(repo_path: str = "/app") -> Dict[str, Optional[str]]:
    """Collect Git metadata by running git commands in repo_path.

    This is the fallback used when artifact-stored metadata is not available.
    """
    meta = {
        "git.commit": _run_git_command(["rev-parse", "HEAD"], repo_path),
        "git.commit.short": _run_git_command(
            ["rev-parse", "--short", "HEAD"], repo_path
        ),
        "git.branch": _run_git_command(
            ["rev-parse", "--abbrev-ref", "HEAD"], repo_path
        ),
        "git.message": _run_git_command(["log", "-1", "--pretty=%B"], repo_path),
        "git.author": _run_git_command(["log", "-1", "--pretty=%an"], repo_path),
        "git.author.email": _run_git_command(["log", "-1", "--pretty=%ae"], repo_path),
        "git.commit.date": _run_git_command(["log", "-1", "--pretty=%cI"], repo_path),
        "git.remote": _run_git_command(
            ["config", "--get", "remote.origin.url"], repo_path
        ),
        "git.is_dirty": bool(_run_git_command(["status", "--porcelain"], repo_path)),
    }
    return meta
