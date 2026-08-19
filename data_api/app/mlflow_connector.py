"""
This module provides functionality to interact with MLflow and a target database.
It includes methods to fetch data from MLflow's tracking database, retrieve artifacts,
and process data signatures from a target database.

Environment variables required:
- MLFLOW_DB: Connection string for the MLflow database.
- TARGET_DB: Connection string for the target database.
- MLFLOW_ARTIFACTS: URI for the MLflow artifacts storage.
"""

import os
import json
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import urlparse
from fastapi import HTTPException
import yaml
import logging

from mlflow.store.tracking.dbmodels import models as tracking_models

# from mlflow.store.model_registry.dbmodels import models as model_registry_models

from data_api.app import models as certain_db_models

load_dotenv()
MLFLOW_DB = os.getenv("MLFLOW_DB")
if MLFLOW_DB is None:
    raise ValueError("MLFLOW_DB is not set in the environment or .env file")
mlflow_engine = create_engine(MLFLOW_DB)

TARGET_DB = os.getenv("TARGET_DB")
if TARGET_DB is None:
    raise ValueError("TARGET_DB is not set in the environment or .env file")
target_engine = create_engine(TARGET_DB)

MLFLOW_ARTIFACTS = os.getenv("MLFLOW_ARTIFACTS")
if MLFLOW_ARTIFACTS is None:
    raise ValueError("MLFLOW_ARTIFACTS is not set in the environment or .env file")
mlflow_artifacts_uri = MLFLOW_ARTIFACTS
logger = logging.getLogger(__name__)


def get_experiments_data():
    """Fetch experiments table from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing MLflow experiments.
    """
    # Prefer a compact experiments.json export placed in the artifacts root
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    candidates = [
        os.path.join(artifacts_root, "experiments.json"),
        os.path.join(artifacts_root, "metadata", "experiments.json"),
    ]

    for cand in candidates:
        try:
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Expecting a list of experiment dicts
                if isinstance(data, dict):
                    data = [data]
                df = pd.DataFrame(data)
                # Ensure expected columns exist
                for col in (
                    "experiment_id",
                    "name",
                    "lifecycle_stage",
                    "artifact_location",
                    "creation_time",
                    "last_update_time",
                ):
                    if col not in df.columns:
                        df[col] = None
                # Fill DB-required non-nullable defaults
                if "experiment_id" in df.columns:
                    df["experiment_id"] = df["experiment_id"].astype(str)
                df["name"] = df["name"].fillna("default")
                # experiments.lifecycle_stage is non-nullable in the DB; use schema default
                df["lifecycle_stage"] = df["lifecycle_stage"].fillna("data_processing")
                now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
                df["creation_time"] = df["creation_time"].fillna(now_ts)
                df["last_update_time"] = df["last_update_time"].fillna(
                    df["creation_time"].fillna(now_ts)
                )
                return df
        except Exception:
            # If the compact export is malformed or unreadable, fall back to other methods
            pass

    # Prefer reading certain/ai_actors/ai_actors.json when present. This file
    # can contain richer experiment / actor metadata produced by the certain
    # artifact exporter. Use it to populate non-nullable experiment fields.
    ai_actors_path = os.path.join(
        artifacts_root, "certain", "ai_actors", "ai_actors.json"
    )
    try:
        if os.path.exists(ai_actors_path):
            with open(ai_actors_path, "r", encoding="utf-8") as fh:
                actors_blob = json.load(fh)

            # actors_blob is expected to be a dict mapping experiment_id -> info
            experiments = []
            if isinstance(actors_blob, dict):
                for exp_id, info in actors_blob.items():
                    # info may be a dict with fields like 'experiment_name',
                    # 'lifecycle_stage', 'artifact_location', 'creation_time',
                    # 'last_update_time'
                    if not isinstance(info, dict):
                        continue
                    experiments.append(
                        {
                            "experiment_id": str(exp_id),
                            "name": info.get("experiment_name")
                            or info.get("name")
                            or f"exp_{exp_id}",
                            "lifecycle_stage": info.get("lifecycle_stage"),
                            "artifact_location": info.get("artifact_location"),
                            "creation_time": info.get("creation_time"),
                            "last_update_time": info.get("last_update_time"),
                            "description": info.get("description"),
                        }
                    )

            if experiments:
                df = pd.DataFrame(experiments)
                # Ensure schema-required columns exist and have safe defaults
                if "experiment_id" in df.columns:
                    df["experiment_id"] = df["experiment_id"].astype(str)
                df["name"] = df["name"].fillna("default")
                df["lifecycle_stage"] = df["lifecycle_stage"].fillna("active")
                now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
                df["creation_time"] = df["creation_time"].fillna(now_ts)
                df["last_update_time"] = df["last_update_time"].fillna(
                    df["creation_time"].fillna(now_ts)
                )
                return df
    except Exception:
        # Non-fatal: fall back to other experiment discovery methods
        pass

    # Fall back to constructing experiments from artifact directory listing
    experiments = []
    if os.path.isdir(artifacts_root):
        # Each top-level directory under artifacts_root is an experiment id
        for experiment_id in os.listdir(artifacts_root):
            exp_path = os.path.join(artifacts_root, experiment_id)
            if not os.path.isdir(exp_path):
                continue
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "name": None,
                    "lifecycle_stage": "data_processing",
                    "artifact_location": None,
                    "creation_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    "last_update_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
                }
            )

        if experiments:
            return pd.DataFrame(experiments)

    # Return Empty DataFrame if no experiments found
    return pd.DataFrame()


def get_experiment_tags_data():
    """Fetch experiment tags from MLflow tracking database.

    Returns:
        pandas.DataFrame: DataFrame containing experiment tags.
    """
    # Read from certain/metadata/experiment_tags.json written by
    # certain_library.metadata.artifact_metadata.save_tags_as_artifact.
    # Shape: {experiment_id, experiment_tags: {key: value, ...}, captured_at}
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    rows = []
    unwanted = {".trash", ".DS_Store"}
    if os.path.isdir(artifacts_root):
        for exp_id in os.listdir(artifacts_root):
            if exp_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            for run_id in os.listdir(exp_path):
                if run_id in unwanted:
                    continue
                tag_file = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "metadata", "experiment_tags.json"
                )
                if not os.path.exists(tag_file):
                    continue
                try:
                    with open(tag_file, "r", encoding="utf-8") as fh:
                        rec = json.load(fh)
                    exp_id_from_file = str(rec.get("experiment_id") or exp_id)
                    for key, value in (rec.get("experiment_tags") or {}).items():
                        rows.append({
                            "experiment_id": exp_id_from_file,
                            "key": str(key),
                            "value": str(value),
                        })
                except Exception:
                    continue

    if rows:
        return pd.DataFrame(rows).drop_duplicates(subset=["experiment_id", "key"])
    return pd.DataFrame(columns=["experiment_id", "key", "value"])


def get_datasets_data():
    """Fetch dataset metadata from certain/metadata/data.json artifacts.

    Reads the lightweight data.json written by
    certain_library.metadata.artifact_metadata.save_dataset_manifest.
    Shape: {run_id, data_location, data_size, human_readable, captured_at}

    Falls back to certain/dataset/data_manifest.json when data.json is absent.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    rows = []
    unwanted = {".trash", ".DS_Store"}
    if os.path.isdir(artifacts_root):
        for exp_id in os.listdir(artifacts_root):
            if exp_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            for run_id in os.listdir(exp_path):
                if run_id in unwanted:
                    continue
                # Prefer lightweight data.json
                data_file = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "metadata", "data.json"
                )
                manifest_file = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "dataset", "data_manifest.json"
                )
                record = None
                for candidate in (data_file, manifest_file):
                    if os.path.exists(candidate):
                        try:
                            with open(candidate, "r", encoding="utf-8") as fh:
                                record = json.load(fh)
                            break
                        except Exception:
                            continue
                if record is None:
                    continue
                record.setdefault("run_id", run_id)
                record.setdefault("experiment_id", exp_id)
                rows.append(record)

    if rows:
        df = pd.DataFrame(rows)
        if "run_id" not in df.columns:
            df["run_id"] = None
        return df
    return pd.DataFrame()


def get_runs_data():
    """Fetch runs from MLflow and populate parent_id by reading
    run_metadata.json files under artifacts/certain/.../metadata/run_metadata.json.

    This avoids depending on tags/events and builds a mapping of run_id -> parent_id
    based on the canonical per-run metadata artifacts produced by the tracker.
    """

    parent_id_map = {}
    # build parent_id_map by scanning run_metadata.json files under the
    # artifacts/certain/metadata location for every run
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri
    try:
        if os.path.isdir(artifacts_root):
            unwanted = {".trash", ".DS_Store", "artifacts"}
            for exp_id in os.listdir(artifacts_root):
                if exp_id in unwanted:
                    continue
                exp_path = os.path.join(artifacts_root, exp_id)
                if not os.path.isdir(exp_path):
                    continue
                for run_id in os.listdir(exp_path):
                    if run_id in unwanted:
                        continue
                    run_path = os.path.join(exp_path, run_id)
                    if not os.path.isdir(run_path):
                        continue
                    meta_file = os.path.join(
                        run_path, "artifacts", "certain", "metadata", "run_metadata.json"
                    )
                    if not os.path.exists(meta_file):
                        # If run_metadata.json is not present, try to extract
                        # parent information from run_tags.json or events.jsonl
                        tags_file = os.path.join(
                            run_path, "artifacts", "certain", "metadata", "run_tags.json"
                        )
                        events_file = os.path.join(
                            run_path, "artifacts", "certain", "metadata", "events.jsonl"
                        )
                        parent_candidate = None
                        if os.path.exists(tags_file):
                            try:
                                with open(tags_file, "r", encoding="utf-8") as tf:
                                    rec = json.load(tf)
                                run_tags = rec.get("run_tags") or {}
                                # look for possible parent keys
                                parent_candidate = (
                                    run_tags.get("mlflow.parentRunId")
                                    or run_tags.get("parent_run_id")
                                    or run_tags.get("parent_id")
                                )
                            except Exception:
                                parent_candidate = None

                        if not parent_candidate and os.path.exists(events_file):
                            try:
                                with open(events_file, "r", encoding="utf-8") as ef:
                                    for line in ef:
                                        try:
                                            ev = json.loads(line)
                                        except Exception:
                                            continue
                                        if ev.get("event_type") == "tag":
                                            key = ev.get("key", "")
                                            if key == "mlflow.parentRunId" and ev.get("value"):
                                                parent_candidate = ev.get("value")
                                                break
                            except Exception:
                                parent_candidate = None

                        if parent_candidate:
                            try:
                                parent_id_map[str(run_id)] = str(parent_candidate)
                            except Exception:
                                pass
                        continue
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        # certain_library.tracking.tracker writes the parent
                        # reference as "parent_run_id" in run_metadata.json;
                        # keep the older/alternate key names as fallbacks.
                        parent_id = (
                            meta.get("parent_run_id")
                            or meta.get("parent_id")
                            or meta.get("parentRunId")
                            or meta.get("parent")
                        )
                        if parent_id:
                            parent_id_map[str(run_id)] = str(parent_id)
                    except Exception:
                        # ignore malformed metadata files
                        continue
    except Exception:
        parent_id_map = {}

    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    runs = []

    if os.path.isdir(artifacts_root):
        unwanted = {".trash", ".DS_Store", "artifacts"}
        for experiment_id in os.listdir(artifacts_root):
            if experiment_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, experiment_id)
            if not os.path.isdir(exp_path):
                continue

            for run_id in os.listdir(exp_path):
                # Skip non-run entries that may appear in the experiment folder
                if run_id in unwanted:
                    continue
                run_path = os.path.join(exp_path, run_id)
                if not os.path.isdir(run_path):
                    continue

                # Look for the canonical metadata under artifacts/certain/metadata
                metadata_path = os.path.join(
                    run_path, "artifacts", "certain", "metadata", "run_metadata.json"
                )

                # Prefer run_metadata.json when available
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        runs.append(data)
                        continue
                    except Exception:
                        pass

                # Fallbacks when run_metadata.json is missing: try run_tags.json
                # and events.jsonl under artifacts/certain/metadata to extract
                # source_type/source_name/source_version/user_id and parent info.
                tags_file = os.path.join(
                    run_path, "artifacts", "certain", "metadata", "run_tags.json"
                )
                events_file = os.path.join(
                    run_path, "artifacts", "certain", "metadata", "events.jsonl"
                )

                fallback = {"run_id": run_id, "experiment_id": experiment_id}

                # Try run_tags.json (written by save_tags_as_artifact)
                if os.path.exists(tags_file):
                    try:
                        with open(tags_file, "r", encoding="utf-8") as fh:
                            rec = json.load(fh)
                        run_tags = rec.get("run_tags") or {}
                        if isinstance(run_tags, dict):
                            # Common keys may include mlflow.source.name, source_name, user_id
                            fallback["source_name"] = (
                                run_tags.get("mlflow.source.name")
                                or run_tags.get("source_name")
                                or fallback.get("source_name")
                            )
                            fallback["user_id"] = (
                                run_tags.get("user")
                                or run_tags.get("user_id")
                                or fallback.get("user_id")
                            )
                            fallback["source_version"] = (
                                run_tags.get("source_version")
                                or fallback.get("source_version")
                            )
                    except Exception:
                        pass

                # Parse events.jsonl for system tags and parentRunId if present
                if os.path.exists(events_file):
                    try:
                        with open(events_file, "r", encoding="utf-8") as fh:
                            for line in fh:
                                try:
                                    ev = json.loads(line)
                                except Exception:
                                    continue
                                # capture mlflow parentRunId from events
                                if ev.get("event_type") == "tag":
                                    key = ev.get("key", "")
                                    val = ev.get("value")
                                    if key == "mlflow.parentRunId" and val:
                                        fallback["parent_id"] = str(val)
                                    if key == "mlflow.source.name" and val:
                                        fallback["source_name"] = fallback.get("source_name") or str(val)
                                    if key == "mlflow.source.type" and val:
                                        fallback["source_type"] = fallback.get("source_type") or str(val)
                                # some telemetry writes source info as top-level events
                                if ev.get("event_type") == "system":
                                    # try to pick up source_version or user info
                                    if ev.get("key") == "source_version" and ev.get("value"):
                                        fallback["source_version"] = fallback.get("source_version") or str(ev.get("value"))
                                    if ev.get("key") == "user_id" and ev.get("value"):
                                        fallback["user_id"] = fallback.get("user_id") or str(ev.get("value"))
                    except Exception:
                        pass

                # Only append fallback dict if it contains at least the run id
                if fallback:
                    runs.append(fallback)

    if runs:
        df = pd.DataFrame(runs)

        if "run_id" in df.columns and "run_uuid" not in df.columns:
            df = df.rename(columns={"run_id": "run_uuid"})

        if "run_name" in df.columns and "name" not in df.columns:
            df = df.rename(columns={"run_name": "name"})

        expected = [
            "run_uuid",
            "name",
            "source_type",
            "source_name",
            "user_id",
            "status",
            "start_time",
            "end_time",
            "source_version",
            "experiment_id",
            "parent_id",
        ]

        for col in expected:
            if col not in df.columns:
                df[col] = None

        df["run_uuid"] = df["run_uuid"].astype(str)

        df["parent_id"] = df["run_uuid"].map(parent_id_map)
        df["parent_id"] = df["parent_id"].where(pd.notna(df["parent_id"]), None)

        for col in ("start_time", "end_time"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].apply(lambda x: min(max(int(x), -(2**63)), 2**63 - 1))

        return df
    
    return pd.DataFrame()


def get_metrics_data():
    """Fetch metrics from certain/metadata/run_metrics.json artifacts.

    Reads run_metrics.json written by
    certain_library.metadata.artifact_metadata.save_metrics_as_artifact.
    Shape: {run_id, run_metrics: {key: [{value, step, timestamp}]}, captured_at}

    Falls back to events.jsonl (event_type == 'metric') when run_metrics.json
    is absent for a run.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    metrics = []
    unwanted = {".trash", ".DS_Store"}
    if os.path.isdir(artifacts_root):
        for exp_id in os.listdir(artifacts_root):
            if exp_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            for run_id in os.listdir(exp_path):
                if run_id in unwanted:
                    continue
                metadata_dir = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "metadata"
                )
                metrics_file = os.path.join(metadata_dir, "run_metrics.json")
                events_file = os.path.join(metadata_dir, "events.jsonl")

                # Prefer dedicated run_metrics.json
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r", encoding="utf-8") as fh:
                            rec = json.load(fh)
                        run_metrics = rec.get("run_metrics") or {}
                        if isinstance(run_metrics, dict):
                            for key, history in run_metrics.items():
                                if not isinstance(history, list):
                                    history = [history]
                                for entry in history:
                                    if not isinstance(entry, dict):
                                        continue
                                    metrics.append({
                                        "run_uuid": str(rec.get("run_id") or run_id),
                                        "key": str(key),
                                        "value": entry.get("value", 0),
                                        "step": int(entry.get("step") or 0),
                                        "timestamp": int(entry.get("timestamp") or 0),
                                        "is_nan": False,
                                    })
                        continue  # don't also read events.jsonl for this run
                    except Exception:
                        pass  # fall through to events.jsonl

                # Fallback: parse events.jsonl for metric events
                if os.path.exists(events_file):
                    try:
                        with open(events_file, "r", encoding="utf-8") as fh:
                            for line in fh:
                                try:
                                    ev = json.loads(line)
                                except Exception:
                                    continue
                                if ev.get("event_type") == "metric":
                                    metrics.append({
                                        "run_uuid": str(ev.get("run_id") or run_id),
                                        "key": ev.get("key", ""),
                                        "value": ev.get("value", 0),
                                        "step": int(ev.get("step") or 0),
                                        "timestamp": int(ev.get("timestamp") or 0),
                                        "is_nan": ev.get("is_NaN", False),
                                    })
                    except Exception:
                        continue

    if metrics:
        df = pd.DataFrame(metrics)
        for col in ("run_uuid", "key", "value", "step", "timestamp", "is_nan"):
            if col not in df.columns:
                df[col] = None
        try:
            df["timestamp"] = df["timestamp"].fillna(0).astype(int)
        except Exception:
            pass
        return df

    return pd.DataFrame()


def get_latest_metrics_data():
    """Derive latest metric values from certain/metadata/run_metrics.json artifacts.

    For each run, reads run_metrics.json and keeps the entry with the highest
    step (or timestamp) for every metric key — equivalent to MLflow's
    SqlLatestMetric table, but sourced entirely from artifact files.

    Shape of run_metrics.json:
        {run_id, run_metrics: {key: [{value, step, timestamp}, ...]}, captured_at}
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    rows = []
    unwanted = {".trash", ".DS_Store"}
    if os.path.isdir(artifacts_root):
        for exp_id in os.listdir(artifacts_root):
            if exp_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            for run_id in os.listdir(exp_path):
                if run_id in unwanted:
                    continue
                metrics_file = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "metadata", "run_metrics.json"
                )
                if not os.path.exists(metrics_file):
                    continue
                try:
                    with open(metrics_file, "r", encoding="utf-8") as fh:
                        rec = json.load(fh)
                except Exception:
                    continue

                run_metrics = rec.get("run_metrics") or {}
                if not isinstance(run_metrics, dict):
                    continue

                for key, history in run_metrics.items():
                    if not isinstance(history, list) or not history:
                        continue
                    # Keep the entry with the highest step; break ties by timestamp
                    latest = max(
                        history,
                        key=lambda e: (int(e.get("step") or 0), int(e.get("timestamp") or 0)),
                    )
                    rows.append({
                        "run_uuid": str(rec.get("run_id") or run_id),
                        "key": str(key),
                        "value": latest.get("value", 0),
                        "step": int(latest.get("step") or 0),
                        "timestamp": int(latest.get("timestamp") or 0),
                        "is_nan": False,
                    })

    if rows:
        df = pd.DataFrame(rows)
        return df
    return pd.DataFrame(columns=["run_uuid", "key", "value", "step", "timestamp", "is_nan"])


def get_params_data():
    """Fetch parameters from certain/metadata/run_params.json artifacts.

    Reads run_params.json written by
    certain_library.metadata.artifact_metadata.save_params_as_artifact.
    Shape: {run_id, run_params: {key: value, ...}, captured_at}

    Falls back to events.jsonl (event_type == 'param') when run_params.json
    is absent for a run.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    params = []
    unwanted = {".trash", ".DS_Store"}
    if os.path.isdir(artifacts_root):
        for exp_id in os.listdir(artifacts_root):
            if exp_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            for run_id in os.listdir(exp_path):
                if run_id in unwanted:
                    continue
                metadata_dir = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "metadata"
                )
                params_file = os.path.join(metadata_dir, "run_params.json")
                events_file = os.path.join(metadata_dir, "events.jsonl")

                # Prefer dedicated run_params.json
                if os.path.exists(params_file):
                    try:
                        with open(params_file, "r", encoding="utf-8") as fh:
                            rec = json.load(fh)
                        run_params = rec.get("run_params") or {}
                        if isinstance(run_params, dict):
                            for key, value in run_params.items():
                                params.append({
                                    "run_uuid": str(rec.get("run_id") or run_id),
                                    "key": str(key),
                                    "value": str(value),
                                })
                        continue  # don't also read events.jsonl for this run
                    except Exception:
                        pass  # fall through to events.jsonl

                # Fallback: parse events.jsonl for param events
                if os.path.exists(events_file):
                    try:
                        with open(events_file, "r", encoding="utf-8") as fh:
                            for line in fh:
                                try:
                                    ev = json.loads(line)
                                except Exception:
                                    continue
                                if ev.get("event_type") == "param":
                                    params.append({
                                        "run_uuid": str(ev.get("run_id") or run_id),
                                        "key": ev.get("key", ""),
                                        "value": str(ev.get("value", "")),
                                    })
                    except Exception:
                        continue

    if params:
        df = pd.DataFrame(params)
        for col in ("run_uuid", "key", "value"):
            if col not in df.columns:
                df[col] = None
        return df

    return pd.DataFrame()


def get_tags_data():
    """Fetch run tags from certain/metadata/run_tags.json artifacts.

    Reads run_tags.json written by
    certain_library.metadata.artifact_metadata.save_tags_as_artifact.
    Shape: {run_id, run_tags: {key: value, ...}, captured_at}

    Falls back to events.jsonl (event_type == 'tag') when run_tags.json
    is absent for a run. Also always includes mlflow.parentRunId from
    events.jsonl so get_runs_data() can resolve parent_id correctly.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_root = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    tags = []
    unwanted = {".trash", ".DS_Store"}
    if os.path.isdir(artifacts_root):
        for exp_id in os.listdir(artifacts_root):
            if exp_id in unwanted:
                continue
            exp_path = os.path.join(artifacts_root, exp_id)
            if not os.path.isdir(exp_path):
                continue
            for run_id in os.listdir(exp_path):
                if run_id in unwanted:
                    continue
                metadata_dir = os.path.join(
                    exp_path, run_id, "artifacts", "certain", "metadata"
                )
                tags_file = os.path.join(metadata_dir, "run_tags.json")
                events_file = os.path.join(metadata_dir, "events.jsonl")

                run_id_str = str(run_id)
                found_via_file = False

                # Prefer dedicated run_tags.json
                if os.path.exists(tags_file):
                    try:
                        with open(tags_file, "r", encoding="utf-8") as fh:
                            rec = json.load(fh)
                        run_tags = rec.get("run_tags") or {}
                        if isinstance(run_tags, dict):
                            for key, value in run_tags.items():
                                tags.append({
                                    "run_uuid": str(rec.get("run_id") or run_id_str),
                                    "key": str(key),
                                    "value": str(value),
                                })
                        found_via_file = True
                    except Exception:
                        pass

                # Always also scan events.jsonl for mlflow.parentRunId tags
                # (these are written by MLflow itself, not by save_tags_as_artifact)
                if os.path.exists(events_file):
                    try:
                        with open(events_file, "r", encoding="utf-8") as fh:
                            for line in fh:
                                try:
                                    ev = json.loads(line)
                                except Exception:
                                    continue
                                if ev.get("event_type") != "tag":
                                    continue
                                key = ev.get("key", "")
                                # If we already loaded from file, only keep
                                # mlflow system tags (parentRunId etc.) from events
                                if found_via_file and not str(key).startswith("mlflow."):
                                    continue
                                tags.append({
                                    "run_uuid": str(ev.get("run_id") or run_id_str),
                                    "key": str(key),
                                    "value": str(ev.get("value", "")),
                                })
                    except Exception:
                        continue

    if tags:
        df = pd.DataFrame(tags)
        for col in ("run_uuid", "key", "value"):
            if col not in df.columns:
                df[col] = None
        # Deduplicate: keep last occurrence per (run_uuid, key)
        df = df.drop_duplicates(subset=["run_uuid", "key"], keep="last")
        return df

    return pd.DataFrame()


def get_data_signature():
    """Read data signatures from target database and return a mapping.

    Reads the DataSignatures table from the target database and builds a dict
    mapping each data_id to an index->column_name mapping based on the signature.

    Returns:
        dict: { data_id: { index: column_name, ... }, ... }
    """
    data = pd.read_sql(certain_db_models.DataSignatures.__tablename__, target_engine)

    # create a dictionary with key the data_id and value a dict of values the index of each and value the name
    data_signatures = {}
    for _, row in data.iterrows():
        data_id = row["data_id"]

        data_temp = {}
        signature = row["signature"]
        for index in range(len(signature)):
            column_name = signature[index]["name"]
            data_temp[index] = column_name

        print(f"data signature size: {len(data_temp)}")

        data_signatures[data_id] = data_temp

    return data_signatures


def get_artifacts_data(folder_name: str = "whylogs", file_extension: str = ".csv"):
    """Collect artifacts from MLflow local artifacts store.

    Parameters:
        folder_name (str): Subfolder under run artifacts to search (default "whylogs").
        file_extension (str): File extension filter (e.g. ".csv", ".json", ".pkl", ".txt", "MLmodel").

    Returns:
        pandas.DataFrame or dict: Concatenated DataFrame of found artifacts (for tabular types)
        or a dict for pickled/model artifacts. Returns empty DataFrame if nothing found.

    Raises:
        fastapi.HTTPException: If expected local artifacts directory is missing.
    """
    # list of files and folders we don't want
    unwanted_files = [".trash", ".DS_Store"]
    parsed_uri = urlparse(mlflow_artifacts_uri)

    # Handle both URI format (file://) and plain path format
    if "file" in parsed_uri.scheme or not parsed_uri.scheme:
        # Use the appropriate path based on URI format
        artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

        # Experiment sub folder
        data_list = []
        data_dict = {}
        experiment_files_lists = os.listdir(artifacts_path)
        for experiment_file_name in experiment_files_lists:
            if experiment_file_name in unwanted_files:
                continue
            experiment_id = experiment_file_name
            experiment_path = os.path.join(artifacts_path, experiment_file_name)
            # Run sub folder
            run_files_lists = os.listdir(experiment_path)
            for run_file_name in run_files_lists:
                if run_file_name in unwanted_files:
                    continue
                run_id = run_file_name
                run_path = os.path.join(experiment_path, run_file_name)
                # Normalize folder_name: strip any leading slashes and any
                # accidental leading 'artifacts/' prefix so we don't construct
                # paths like '<run_path>/artifacts/artifacts/...'. Accept either
                # 'run_logs' or 'certain/run_logs' (the caller commonly passes
                # artifact_path("run_logs") which yields 'certain/run_logs').
                norm_folder = folder_name.lstrip("/")
                if norm_folder.startswith("artifacts/"):
                    norm_folder = norm_folder.split("artifacts/", 1)[1]

                # Artifacts sub folder (join with the run's artifacts directory)
                folder_path = os.path.join(run_path, "artifacts", norm_folder)
                # effective identifiers default to the run's own ids; may be
                # updated to the parent run when artifacts are found there.
                effective_run_id = run_id
                effective_experiment_id = experiment_id

                if not os.path.exists(folder_path):
                    # Attempt to find the artifact under a parent run's artifacts
                    parent_found = False
                    parent_id = None
                    try:
                        # Check for a run metadata file that may include parent info
                        meta_file = os.path.join(
                            run_path, "artifacts", "certain", "metadata", "run_metadata.json"
                        )
                        if os.path.exists(meta_file):
                            try:
                                with open(meta_file, "r", encoding="utf-8") as mf:
                                    meta = json.load(mf)
                                parent_id = (
                                    meta.get("parent_run_id")
                                    or meta.get("parent_id")
                                    or meta.get("parentRunId")
                                    or meta.get("parent")
                                )
                            except Exception:
                                parent_id = None

                        # Fallback: inspect events.jsonl for mlflow.parentRunId tag
                        if not parent_id:
                            events_meta = os.path.join(
                                run_path, "artifacts", "certain", "metadata", "events.jsonl"
                            )
                            if os.path.exists(events_meta):
                                try:
                                    with open(events_meta, "r", encoding="utf-8") as ef:
                                        for line in ef:
                                            try:
                                                ev = json.loads(line)
                                            except Exception:
                                                continue
                                            if ev.get("event_type") == "tag" and ev.get("key") == "mlflow.parentRunId":
                                                parent_id = ev.get("value")
                                                break
                                except Exception:
                                    parent_id = None

                        # If we found a parent id, try to locate its artifact folder
                        if parent_id:
                            # Prefer same-experiment parent first
                            candidate_parent = os.path.join(
                                experiment_path, str(parent_id), "artifacts", norm_folder
                            )
                            if os.path.exists(candidate_parent):
                                folder_path = candidate_parent
                                parent_found = True
                                effective_run_id = str(parent_id)
                                effective_experiment_id = experiment_id
                            else:
                                # Search across experiments for the parent run folder
                                for cand_exp in os.listdir(artifacts_path):
                                    cand_path = os.path.join(
                                        artifacts_path, cand_exp, str(parent_id), "artifacts", norm_folder
                                    )
                                    if os.path.exists(cand_path):
                                        folder_path = cand_path
                                        parent_found = True
                                        effective_run_id = str(parent_id)
                                        effective_experiment_id = cand_exp
                                        break
                    except Exception:
                        parent_found = False

                    if not parent_found:
                        logger.debug(
                            "Skipping run %s: artifact folder '%s' not found at %s",
                            run_id,
                            norm_folder,
                            folder_path,
                        )
                        continue
                if len(file_extension.split(".")[0]) != 0:
                    files_list = [file_extension]
                else:
                    files_list = os.listdir(folder_path)
                for file_name in files_list:
                    if file_name.endswith(file_extension):
                        file_path = os.path.join(folder_path, file_name)
                        if not os.path.exists(file_path):
                            logger.debug(
                                "Skipping run %s: artifact file '%s' not found at %s",
                                run_id,
                                file_name,
                                file_path,
                            )
                            continue
                        data = None
                        if ".csv" in file_extension:
                            data = pd.read_csv(file_path)
                            data["run_id"] = effective_run_id
                            data["experiment_id"] = effective_experiment_id
                            data["stage"] = file_name.split("_")[-1].split(".")[0]
                            data_list.append(data)
                        elif ".json" in file_extension:
                            # Read JSON artifact into a DataFrame and attach run/experiment metadata
                            data = pd.read_json(file_path)
                            # Ensure we have a DataFrame (single object -> one-row DataFrame)
                            if isinstance(data, dict):
                                data = pd.DataFrame([data])
                            # Attach run/experiment identifiers so sync functions can map correctly
                            data["run_id"] = effective_run_id
                            data["experiment_id"] = effective_experiment_id
                            # Derive a stage from filename suffix similar to CSV handling
                            try:
                                data["stage"] = file_name.split("_")[-1].split(".")[0]
                            except Exception:
                                data["stage"] = "json"
                            data_list.append(data)
                        elif ".pkl" in file_extension:
                            # Read the pickle file and convert it to JSON format then load as DataFrame
                            data = pd.read_pickle(file_path)
                            # Check if the loaded object is a DataFrame
                            if isinstance(data, pd.DataFrame):
                                json_str = data.to_json(orient="records")
                                # Continue processing with the DataFrame
                            else:
                                # Assume it's a model with a get_params() method (e.g., RandomForestRegressor)
                                if hasattr(data, "get_params"):
                                    model_params = data.get_params()
                                    json_str = json.dumps(model_params)
                                else:
                                    raise TypeError(
                                        "The loaded object cannot be serialized to JSON using this approach."
                                    )

                            data_dict[run_id] = json_str
                        elif ".txt" in file_extension:
                            # Read the text file and convert it to JSON format then load as DataFrame
                            with open(file_path, "r", encoding="utf-8") as file:
                                content = file.read()

                            # Convert content to a list of timestamps
                            content = content.splitlines()
                            content = [line.strip() for line in content if line.strip()]

                            data = pd.DataFrame([{"timestamps": content}])
                            data["run_id"] = effective_run_id
                            data["experiment_id"] = effective_experiment_id
                            data_list.append(data)
                        elif "MLmodel" in file_extension:
                            # read txt file and convert it to JSON format then load as DataFrame
                            # Read MLmodel file from the artifact path
                            with open(file_path, "r", encoding="utf-8") as mlmodel_file:
                                mlmodel_content = mlmodel_file.read()

                            # Convert MLmodel content to a structured dictionary using yaml parser
                            try:
                                # PyYAML can handle the MLmodel format which is YAML-like
                                mlmodel_data = yaml.safe_load(mlmodel_content)

                                # Flatten the nested dictionary structure for DataFrame conversion
                                flat_mlmodel_data = {}

                                def flatten_dict(nested_dict, prefix=""):
                                    for key, value in nested_dict.items():
                                        new_key = f"{prefix}.{key}" if prefix else key
                                        if isinstance(value, dict):
                                            flatten_dict(value, new_key)
                                        else:
                                            flat_mlmodel_data[new_key] = value

                                flatten_dict(mlmodel_data)

                                # Convert the flattened dictionary to dataframe
                                mlmodel_df = pd.DataFrame([flat_mlmodel_data])
                            except Exception as e:
                                # Fallback if YAML parsing fails
                                print(f"Failed to parse MLmodel with YAML: {e}")

                                # Create a simple representation of the entire content
                                mlmodel_df = pd.DataFrame(
                                    [{"mlmodel_content": mlmodel_content}]
                                )
                                mlmodel_df["run_id"] = effective_run_id
                                mlmodel_df["experiment_id"] = effective_experiment_id
                            data_list.append(mlmodel_df)

        if data_list:
            return pd.concat(data_list, ignore_index=True)
        if data_dict:
            return data_dict

    return pd.DataFrame()


def get_json_artifacts_data(folder_name: str, file_name: Optional[str] = None) -> list:
    """Collect JSON artifact records from MLflow local artifacts store.

    Walks the artifacts directory tree looking for ``folder_name`` subdirectories
    and reads every ``*.json`` file found there.  Each JSON file is expected to
    contain a single dict (the record written by the corresponding
    ``certain_library`` logging function).

    Parameters:
        folder_name (str): Subfolder under ``<experiment>/<run>/artifacts/`` to search.
        file_name (str | None): Optional exact JSON file name filter
            (e.g. ``input_examples.json``). When omitted, all ``*.json``
            files in the folder are read.

    Returns:
        list[tuple[str, str, dict]]: A list of ``(run_id, experiment_id, record)``
        tuples for every JSON file discovered.  Returns an empty list when the
        folder does not exist for any run (no ``HTTPException`` is raised, unlike
        :func:`get_artifacts_data`).
    """
    unwanted = {".trash", ".DS_Store"}
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    results: list = []
    if not os.path.isdir(artifacts_path):
        # If local artifacts path is not available, attempt to download
        # JSON artifacts using MLflow client from the tracking server.
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            import tempfile as _tmp

            # Iterate experiments and their runs to find artifacts
            for exp in client.list_experiments():
                try:
                    # Search runs for this experiment
                    runs = client.search_runs([exp.experiment_id])
                except Exception:
                    runs = []
                for r in runs:
                    run_id = r.info.run_id
                    try:
                        artifacts = client.list_artifacts(run_id, path=folder_name)
                    except Exception:
                        artifacts = []
                    for art in artifacts:
                        # art is an ArtifactSummary with .path and .is_dir
                        if getattr(art, "is_dir", False):
                            continue
                        base_name = os.path.basename(art.path)
                        if not base_name.endswith(".json"):
                            continue
                        if file_name and base_name != file_name:
                            continue
                        # Download and read
                        with _tmp.TemporaryDirectory() as td:
                            try:
                                local = client.download_artifacts(run_id, art.path, dst_path=td)
                                if os.path.exists(local):
                                    with open(local, "r", encoding="utf-8") as fh:
                                        record = json.load(fh)
                                    if isinstance(record, dict):
                                        results.append((run_id, str(exp.experiment_id), record))
                                    elif isinstance(record, list):
                                        for item in record:
                                            if isinstance(item, dict):
                                                results.append((run_id, str(exp.experiment_id), item))
                            except Exception:
                                continue
        except Exception:
            return results
        return results

    for experiment_id in os.listdir(artifacts_path):
        if experiment_id in unwanted:
            continue
        experiment_path = os.path.join(artifacts_path, experiment_id)
        if not os.path.isdir(experiment_path):
            continue
        for run_id in os.listdir(experiment_path):
            if run_id in unwanted:
                continue
            folder_path = os.path.join(
                experiment_path, run_id, "artifacts", folder_name
            )
            if not os.path.isdir(folder_path):
                continue
            for current_file_name in os.listdir(folder_path):
                if not current_file_name.endswith(".json"):
                    continue
                if file_name and current_file_name != file_name:
                    continue
                file_path = os.path.join(folder_path, current_file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as fh:
                        record = json.load(fh)
                    if isinstance(record, dict):
                        results.append((run_id, experiment_id, record))
                    elif isinstance(record, list):
                        for item in record:
                            if isinstance(item, dict):
                                results.append((run_id, experiment_id, item))
                except Exception as exc:
                    print(
                        f"[get_json_artifacts_data] Could not read {file_path}: {exc}"
                    )
    return results


def get_dataset_manifest_for_run(run_id: str):
    """Return parsed data_manifest.json for a given run_id when present in local artifacts.

    Searches the local artifacts root for any experiment folder containing
    the run_id and reads artifacts/certain/dataset/data_manifest.json if present.
    Returns dict or None when not found.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    # If the artifacts path isn't present locally, we'll still attempt an MLflow
    # client download below; only search the local tree when it's available.

    for exp in os.listdir(artifacts_path):
        exp_path = os.path.join(artifacts_path, exp)
        if not os.path.isdir(exp_path):
            continue
        run_path = os.path.join(exp_path, run_id)
        if not os.path.isdir(run_path):
            continue
        manifest_path = os.path.join(
            run_path, "artifacts", "certain", "dataset", "data_manifest.json"
        )
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None

        # Also check the metadata location for a lightweight data.json
        meta_path = os.path.join(
            run_path, "artifacts", "certain", "metadata", "data.json"
        )
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None

    # If not found on local filesystem, try MLflow client to download artifacts
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        # Try to download the artifact to a temp dir
        import tempfile as _tmp

        with _tmp.TemporaryDirectory() as td:
            try:
                # Mlflow client download_artifacts may accept a run_id and a
                # path relative to the run's artifact root.
                rel_path = "certain/dataset/data_manifest.json"
                local = client.download_artifacts(run_id, rel_path, dst_path=td)
                if os.path.exists(local):
                    with open(local, "r", encoding="utf-8") as fh:
                        return json.load(fh)
            except Exception:
                # Try metadata path as a fallback, then give up
                try:
                    rel_meta = "certain/metadata/data.json"
                    local2 = client.download_artifacts(run_id, rel_meta, dst_path=td)
                    if os.path.exists(local2):
                        with open(local2, "r", encoding="utf-8") as fh:
                            return json.load(fh)
                except Exception:
                    return None
    except Exception:
        return None


def get_python_env_for_run(run_id: str):
    """Return parsed certain/model/python_env.yaml for a given run_id when present locally.

    Searches the local artifacts root for any experiment folder containing the
    run_id and reads artifacts/certain/model/python_env.yaml if present.
    Shape (written by MLflow's model logging):
        {python: "3.9.16", build_dependencies: [...], dependencies: [...]}

    Returns dict or None when not found.
    """
    parsed_uri = urlparse(mlflow_artifacts_uri)
    artifacts_path = parsed_uri.path if parsed_uri.path else mlflow_artifacts_uri

    if not os.path.isdir(artifacts_path):
        return None

    for exp in os.listdir(artifacts_path):
        exp_path = os.path.join(artifacts_path, exp)
        if not os.path.isdir(exp_path):
            continue
        run_path = os.path.join(exp_path, run_id)
        if not os.path.isdir(run_path):
            continue
        python_env_path = os.path.join(
            run_path, "artifacts", "certain", "model", "python_env.yaml"
        )
        if os.path.exists(python_env_path):
            try:
                with open(python_env_path, "r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh)
            except Exception:
                return None

    # Not found on local filesystem; try MLflow client to download the artifact.
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        import tempfile as _tmp

        with _tmp.TemporaryDirectory() as td:
            try:
                rel_path = "certain/model/python_env.yaml"
                local = client.download_artifacts(run_id, rel_path, dst_path=td)
                if os.path.exists(local):
                    with open(local, "r", encoding="utf-8") as fh:
                        return yaml.safe_load(fh)
            except Exception:
                return None
    except Exception:
        return None

