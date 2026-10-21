from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen


@dataclass(frozen=True)
class DemoBatch:
    source: str
    clean_payload: dict[str, Any]
    dirty_payload: dict[str, Any]
    clean_annotations: list[dict[str, Any]]
    dirty_annotations: list[dict[str, Any]]


def _is_http_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _is_file_url(source: str) -> bool:
    return source.startswith("file://")


def _is_s3_source(source: str) -> bool:
    return source.startswith("s3://")


def _read_s3_json(source: str, filename: str) -> Any:
    """Read s3://<bucket>/<prefix>/<filename> from MinIO/S3 using MINIO_* env credentials."""
    from minio import Minio

    parsed = urlparse(source)
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    object_name = f"{prefix}/{filename}" if prefix else filename

    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY to read from an s3:// source.")

    client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=access_key,
        secret_key=secret_key,
        secure=os.getenv("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes"},
    )
    response = client.get_object(bucket, object_name)
    try:
        return _read_json_text(response.read().decode("utf-8"))
    finally:
        response.close()
        response.release_conn()


def _read_json_text(text: str) -> Any:
    return json.loads(text)


def load_json_resource(source: str, filename: str) -> Any:
    """Load a JSON resource from an HTTP base URL, s3:// location, file URL, or local folder."""

    normalized_name = filename.lstrip("/")
    source = source.rstrip("/")

    if _is_s3_source(source):
        return _read_s3_json(source, normalized_name)

    if _is_http_source(source):
        resource_url = urljoin(f"{source}/", normalized_name)
        with urlopen(resource_url, timeout=60) as response:
            return _read_json_text(response.read().decode("utf-8"))

    if _is_file_url(source):
        path = Path(urlparse(source).path) / normalized_name
    else:
        path = Path(source) / normalized_name

    if not path.exists():
        raise FileNotFoundError(str(path))
    return _read_json_text(path.read_text(encoding="utf-8"))


def load_demo_batch_from_source(source: str, batch_id: int = 7) -> DemoBatch:
    """Load the clean/dirty demo payloads and annotations from a dataspace source."""

    clean_name = f"simulation_output_clean_{batch_id}.json"
    dirty_name = f"simulation_output_dirty_{batch_id}.json"
    annotations_name = f"scenario_annotations_batch_{batch_id}.json"

    clean_payload = load_json_resource(source, clean_name)
    dirty_payload = load_json_resource(source, dirty_name)
    annotations = load_json_resource(source, annotations_name)

    return DemoBatch(
        source=source,
        clean_payload=clean_payload,
        dirty_payload=dirty_payload,
        clean_annotations=list(annotations.get("Clean", [])),
        dirty_annotations=list(annotations.get("Dirty", [])),
    )
