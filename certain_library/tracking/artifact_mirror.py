"""Immediate local mirroring for artifacts successfully logged to MLflow."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def calculate_checksum(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class ArtifactMirror:
    """Copy MLflow artifacts into the portable CERTAIN mirror."""

    def __init__(self, root: Path):
        self.root = Path(root) / "experiments"

    def mirror_file(
        self,
        local_path: str,
        experiment_id: str,
        run_id: str,
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Copy one artifact and return its mirror metadata."""
        source = Path(local_path)

        if not source.is_file():
            raise FileNotFoundError("Artifact is not a file: {}".format(source))

        relative_directory = self._safe_artifact_path(artifact_path)
        destination_directory = (
            self.root / str(experiment_id) / run_id / "artifacts" / relative_directory
        )
        destination_directory.mkdir(parents=True, exist_ok=True)

        destination = destination_directory / source.name

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=str(destination_directory),
            prefix=".{}.".format(source.name),
            suffix=".tmp",
        )
        os.close(file_descriptor)

        try:
            shutil.copy2(str(source), temporary_name)
            os.replace(temporary_name, str(destination))
        finally:
            try:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            except Exception:
                # Best-effort cleanup; ignore races where the temp file
                # has already been removed or moved.
                pass

        relative_artifact_path = relative_directory / source.name

        return {
            "run_id": run_id,
            "experiment_id": str(experiment_id),
            "source_path": str(source.resolve()),
            "mirror_path": str(destination.resolve()),
            "artifact_path": str(relative_artifact_path),
            "file_size": destination.stat().st_size,
            "checksum": calculate_checksum(destination),
        }

    def mirror_directory(
        self,
        local_directory: str,
        experiment_id: str,
        run_id: str,
        artifact_path: Optional[str] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Mirror every file below a directory."""
        source_directory = Path(local_directory)

        if not source_directory.is_dir():
            raise NotADirectoryError(str(source_directory))

        for source in source_directory.rglob("*"):
            if not source.is_file():
                continue

            nested_directory = source.relative_to(source_directory).parent
            destination_path = Path(artifact_path or "") / nested_directory

            yield self.mirror_file(
                str(source),
                experiment_id,
                run_id,
                str(destination_path),
            )

    @staticmethod
    def _safe_artifact_path(
        artifact_path: Optional[str],
    ) -> Path:
        """Reject paths that could escape the run artifact directory."""
        relative_path = Path(artifact_path or "")

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("artifact_path must be relative and cannot contain '..'")

        return relative_path
