import uuid
from urllib.parse import urlparse
import io
import sys
import json
import pandas as pd
import inspect
import yaml
from typing import Optional
from fastapi import FastAPI
from sqlalchemy import text
from app.mlflow_connector import (
    get_metrics_data,
    get_runs_data,
    get_experiments_data,
    get_experiment_tags_data,
    get_datasets_data,
    get_latest_metrics_data,
    get_params_data,
    get_tags_data,
    get_artifacts_data,
    get_json_artifacts_data,
)
from app.target_connector import insert_dataframe, bulk_upsert_metrics, target_engine
from misc.data_transform import (
    map_runs,
    map_experiments,
    map_resources,
    map_time_series_data,
    map_data_drift,
    map_data_duration_leakage,
    map_datasets,
    map_data_metrics,
    map_model_metrics,
    map_model_params,
    map_runs_tags,
    map_data_resources,
    map_run_code,
    map_checkpoints,
    map_weight_distribution,
    map_examples,
    map_run_logs,
    map_ai_actors,
    map_labeling_procedures,
    map_risk,
    map_human_oversight,
    map_transparency_measure,
    map_change_log,
    map_declaration_of_conformity,
    map_visual_documentation,
    map_explainable_ai,
    map_model_packaging,
    map_build_testing,
    map_standard,
    map_interface,
    map_decommissioning,
    map_monitor_logs,
    map_tokenizer_config,
    map_tokenization_stats,
    map_data_techniques,
    # New mappers for certain/metadata artifacts
    map_run_params,
    map_run_metrics,
    map_run_resources,
    map_run_inputs,
    map_experiment_tags_artifact,
    map_dataset_manifest,
)

app = FastAPI()

ARTIFACT_BASE_FOLDER = "certain"


def artifact_path(folder_name: str) -> str:
    """Return artifact folder under artifacts/certain/."""
    return f"{ARTIFACT_BASE_FOLDER}/{folder_name}"


@app.get("/")
def root():
    return {"status": "up"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def sync_metrics(id_mapping):
    df = get_metrics_data()
    # If metrics mention runs that are not present in id_mapping (e.g. when
    # run_metadata.json was not written for a run but metrics were), create
    # lightweight id_mapping entries so downstream mappers can resolve
    # model_id/data_id. Also synthesize minimal model_architecture parent rows
    # for any newly-created mappings to satisfy FK constraints.
    if isinstance(df, pd.DataFrame) and not df.empty:
        run_col = "run_uuid" if "run_uuid" in df.columns else (
            "run_id" if "run_id" in df.columns else None
        )
        if run_col:
            metric_run_ids = set(df[run_col].astype(str).tolist())
        else:
            metric_run_ids = set()

        missing_runs = [r for r in metric_run_ids if r not in id_mapping]
        if missing_runs:
            import uuid as _uuid
            import os
            from app import mlflow_connector as _mlfc
            now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

            # Discover experiment folders for missing runs by scanning the
            # artifacts root. If an experiment id cannot be determined we will
            # create a synthetic experiment row to satisfy FK constraints.
            parsed = urlparse(_mlfc.mlflow_artifacts_uri)
            artifacts_root = parsed.path if parsed.path else _mlfc.mlflow_artifacts_uri

            # Build a map run_id -> experiment_id by scanning the artifacts
            run_to_exp = {}
            try:
                if os.path.isdir(artifacts_root):
                    for exp_id in os.listdir(artifacts_root):
                        exp_path = os.path.join(artifacts_root, exp_id)
                        if not os.path.isdir(exp_path):
                            continue
                        for candidate in os.listdir(exp_path):
                            run_to_exp[candidate] = exp_id
            except Exception:
                run_to_exp = {}

            # Ensure the experiments and runs parent rows exist before inserting
            # into id_mapping (id_mapping.run_id -> runs.run_id FK).
            runs_rows = []
            experiments_to_create = set()
            for run_id in missing_runs:
                exp_id = run_to_exp.get(run_id)
                if not exp_id:
                    # Try to use existing experiments from get_experiments_data()
                    try:
                        exp_df = get_experiments_data()
                        if isinstance(exp_df, pd.DataFrame) and not exp_df.empty:
                            # pick the first experiment as fallback
                            exp_id = exp_df.iloc[0].get("experiment_id")
                    except Exception:
                        exp_id = None

                if not exp_id:
                    # create a synthetic experiment id
                    exp_id = str(_uuid.uuid4())
                    experiments_to_create.add(exp_id)

                run_row = _apply_run_provenance_defaults(
                    {
                        "run_id": run_id,
                        "run_name": None,
                        "parent_id": None,
                        "source_type": "LOCAL",
                        "source_name": "",
                        "user_id": "",
                        "status": "FINISHED",
                        "start_time": now_ts,
                        "end_time": now_ts,
                        "source_version": "",
                        "experiment_id": exp_id,
                    },
                    _build_run_artifact_index(),
                )
                runs_rows.append(run_row)

            if experiments_to_create:
                exp_rows = []
                for e in experiments_to_create:
                    exp_rows.append({
                        "experiment_id": e,
                        "experiment_name": f"artifact-generated-{e[:8]}",
                        "lifecycle_stage": "data_processing",
                        "creation_time": now_ts,
                        "last_update_time": now_ts,
                    })
                insert_dataframe(pd.DataFrame(exp_rows), "experiments")

            if runs_rows:
                insert_dataframe(pd.DataFrame(runs_rows), "runs")

            # Now create mappings and minimal model_architecture parents
            new_mappings = []
            parent_rows = []
            for run_id in missing_runs:
                new_entry = {
                    "run_id": run_id,
                    "model_id": str(_uuid.uuid4()),
                    "data_id": str(_uuid.uuid4()),
                    "deployment_id": str(_uuid.uuid4()),
                }
                new_mappings.append(new_entry)
                id_mapping[run_id] = {
                    "model_id": new_entry["model_id"],
                    "data_id": new_entry["data_id"],
                    "deployment_id": new_entry["deployment_id"],
                }

                parent_rows.append(
                    {
                        "run_id": run_id,
                        "model_id": new_entry["model_id"],
                        "architecture_name": "artifact-synthesized",
                        "model_version": 1,
                        "layer_structure": {},
                        "activation_function": "",
                        "optimizer": "{}",
                        "loss_function": "",
                        "framework": "",
                        "metrics": [],
                        "input_shape": "",
                        "output_shape": "",
                        "number_of_layers": 0,
                        "number_of_total_parameters": 0,
                        "number_of_trainable_parameters": 0,
                        "number_of_non_trainable_parameters": 0,
                        "creation_time": now_ts,
                    }
                )

            if new_mappings:
                insert_dataframe(pd.DataFrame(new_mappings), "id_mapping")
            if parent_rows:
                insert_dataframe(pd.DataFrame(parent_rows), "model_architecture")

    mapped_rows = df.apply(
        lambda row: map_model_metrics(row, id_mapping), axis=1
    )
    mapped_df = pd.DataFrame(mapped_rows.values.tolist())

    if not mapped_df.empty:
        # Ensure step and timestamp are present
        if "step" not in mapped_df.columns:
            mapped_df["step"] = 0
        else:
            mapped_df["step"] = mapped_df["step"].fillna(0).astype(int)

        if "timestamp" not in mapped_df.columns:
            mapped_df["timestamp"] = int(pd.Timestamp.now(tz="UTC").timestamp())
        else:
            try:
                mapped_df["timestamp"] = mapped_df["timestamp"].fillna(0).astype(int)
            except Exception:
                mapped_df["timestamp"] = mapped_df["timestamp"].apply(
                    lambda x: (
                        int(x)
                        if pd.notna(x)
                        else int(pd.Timestamp.now(tz="UTC").timestamp())
                    )
                )

    insert_dataframe(mapped_df, "model_metrics")
    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_latest_metrics(id_mapping):
    df = get_latest_metrics_data()

    # Map rows into the expected DB shape
    mapped_rows = df.apply(
        lambda row: map_model_metrics(row, id_mapping), axis=1
    )
    mapped_df = pd.DataFrame(mapped_rows.values.tolist())

    if mapped_df.empty:
        return {"rows_synced": 0, "status": "no latest metrics found"}

    # Ensure step and timestamp exist and have sensible defaults
    if "step" not in mapped_df.columns:
        mapped_df["step"] = 0
    else:
        mapped_df["step"] = mapped_df["step"].fillna(0).astype(int)

    if "timestamp" not in mapped_df.columns:
        mapped_df["timestamp"] = int(pd.Timestamp.now(tz="UTC").timestamp())
    else:
        try:
            mapped_df["timestamp"] = mapped_df["timestamp"].fillna(0).astype(int)
        except Exception:
            mapped_df["timestamp"] = mapped_df["timestamp"].apply(
                lambda x: (
                    int(x)
                    if pd.notna(x)
                    else int(pd.Timestamp.now(tz="UTC").timestamp())
                )
            )

    # Attempt to derive parent_id for each run so we can deduplicate per
    # (parent_id, run_id, model_id, key) pair — keeping only the latest record
    # as defined by highest step then latest timestamp. Exclude any metrics
    # whose key starts with 'best_' from this selection process.
    try:
        runs_df = get_runs_data()
        if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
            # runs_df may use 'run_uuid' or 'run_id'
            run_id_col = (
                "run_uuid"
                if "run_uuid" in runs_df.columns
                else ("run_id" if "run_id" in runs_df.columns else None)
            )
            if run_id_col:
                parent_map = runs_df.set_index(run_id_col)["parent_id"].to_dict()
            else:
                parent_map = {}
        else:
            parent_map = {}
    except Exception:
        parent_map = {}

    mapped_df["parent_id"] = mapped_df["run_id"].map(lambda r: parent_map.get(r))

    # Ensure id_mapping covers any runs present in latest metrics; otherwise
    # model_id will be empty and inserts into last_model_metrics will fail the
    # FK constraint. Create minimal id_mapping and parent model_architecture
    # rows for missing runs.
    try:
        metric_run_ids = set(mapped_df["run_id"].astype(str).tolist())
    except Exception:
        metric_run_ids = set()

    missing_runs = [r for r in metric_run_ids if r not in id_mapping]
    if missing_runs:
        import uuid as _uuid
        now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
        new_mappings = []
        parent_rows = []
        for run_id in missing_runs:
            new_entry = {
                "run_id": run_id,
                "model_id": str(_uuid.uuid4()),
                "data_id": str(_uuid.uuid4()),
                "deployment_id": str(_uuid.uuid4()),
            }
            new_mappings.append(new_entry)
            id_mapping[run_id] = {
                "model_id": new_entry["model_id"],
                "data_id": new_entry["data_id"],
                "deployment_id": new_entry["deployment_id"],
            }
            parent_rows.append(
                {
                    "run_id": run_id,
                    "model_id": new_entry["model_id"],
                    "architecture_name": "artifact-synthesized",
                    "model_version": 1,
                    "layer_structure": {},
                    "creation_time": now_ts,
                }
            )

        if new_mappings:
            insert_dataframe(pd.DataFrame(new_mappings), "id_mapping")
        if parent_rows:
            insert_dataframe(pd.DataFrame(parent_rows), "model_architecture")

    # Exclude 'best_' metrics entirely from last_model_metrics selection
    try:
        mapped_df = mapped_df[~mapped_df["key"].str.startswith("best_")].copy()
    except Exception:
        # If 'key' is missing or non-string, keep original behavior and proceed
        pass

    # Now perform deterministic per-group selection using the required priority:
    # 1) highest step (where step > 0) 2) latest timestamp (tie-breaker)
    group_cols = ["parent_id", "run_id", "model_id", "key"]
    dedup_rows = []
    try:
        # Ensure step and timestamp exist and are numeric (they should earlier)
        mapped_df["step"] = mapped_df["step"].fillna(0).astype(int)
        mapped_df["timestamp"] = mapped_df["timestamp"].fillna(0).astype(int)
    except Exception:
        # best-effort: leave as-is
        pass

    if not mapped_df.empty:
        # iterate groups and pick the desired row
        grouped = mapped_df.groupby(group_cols, dropna=False)
        for _, group in grouped:
            try:
                # Prefer entries with positive steps
                stepped = group[group["step"] > 0]
                if not stepped.empty:
                    max_step = int(stepped["step"].max())
                    # among those with max_step pick latest timestamp
                    candidates = stepped[stepped["step"] == max_step]
                    chosen = candidates.loc[candidates["timestamp"].idxmax()]
                else:
                    # No stepped entries: fall back to latest timestamp overall
                    chosen = group.loc[group["timestamp"].idxmax()]
                dedup_rows.append(chosen.to_dict())
            except Exception:
                # fallback: pick last row in group as-is
                try:
                    dedup_rows.append(group.iloc[-1].to_dict())
                except Exception:
                    continue

    # Build deduped DataFrame from selected rows
    if dedup_rows:
        deduped = pd.DataFrame(dedup_rows)
    else:
        deduped = pd.DataFrame()

    # Remove parent_id (not part of last_model_metrics schema) before insert
    if "parent_id" in deduped.columns:
        deduped = deduped.drop(columns=["parent_id"])

    insert_dataframe(deduped, "last_model_metrics")
    return {"rows_synced": len(deduped), "status": "success"}

def _build_run_artifact_index() -> dict:
    """Collect per-run provenance data from metadata artifacts."""
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    index = {}

    for run_id, _experiment_id, record in records:
        if not isinstance(record, dict):
            continue

        run_key = str(run_id)
        facts = index.setdefault(run_key, {})

        run_tags = record.get("run_tags")
        if isinstance(run_tags, dict):
            if run_tags.get("source_name"):
                facts["source_name"] = str(run_tags.get("source_name"))
            if run_tags.get("user") or run_tags.get("user_id"):
                facts["user_id"] = str(run_tags.get("user") or run_tags.get("user_id"))
            if run_tags.get("source_version"):
                facts["source_version"] = str(run_tags.get("source_version"))

        git_commit = record.get("git.commit") or record.get("mlflow.source.git.commit")
        git_branch = record.get("git.branch")
        git_author = record.get("git.author")
        if git_commit:
            facts["source_version"] = str(git_commit)

        if git_branch:
            facts["source_name"] = str(git_branch)

        if git_author:
            facts["user_id"] = str(git_author)

        if git_commit or git_branch or git_author:
            facts["source_type"] = "GIT"
        elif record.get("source_type"):
            facts["source_type"] = str(record.get("source_type"))
        elif run_tags and run_tags.get("source_type"):
            facts["source_type"] = str(run_tags.get("source_type"))

    return index


def _apply_run_provenance_defaults(row: dict, provenance_index: dict) -> dict:
    """Fill run provenance fields from the per-run artifact index."""
    run_id = str(row.get("run_id", ""))
    provenance = provenance_index.get(run_id, {})

    if provenance:
        row["source_type"] = provenance.get("source_type") or row.get("source_type") or "GIT"
        row["source_name"] = provenance.get("source_name") or row.get("source_name") or "unknown"
        row["user_id"] = provenance.get("user_id") or row.get("user_id") or ""
        row["source_version"] = provenance.get("source_version") or row.get("source_version") or ""
    else:
        row["source_type"] = row.get("source_type") or "LOCAL"
        row["source_name"] = row.get("source_name") or ""
        row["user_id"] = row.get("user_id") or ""
        row["source_version"] = row.get("source_version") or ""

    return row


def sync_runs():
    df = get_runs_data()

    mapped_rows = df.apply(map_runs, axis=1)

    # Normalize mapped_rows into a plain list of dicts regardless of whether
    # pandas.apply returned a Series (common) or a DataFrame (when the mapper
    # returns Series/dicts that get expanded). This avoids AttributeError when
    # calling .tolist() on a DataFrame.
    rows_list = []
    if isinstance(mapped_rows, pd.Series):
        # Each element should already be a dict-like result from the mapper
        rows_list = mapped_rows.tolist()
    elif isinstance(mapped_rows, pd.DataFrame):
        try:
            # Convert each row to a dict; prefer to skip fully-empty rows
            for _, r in mapped_rows.iterrows():
                # If a row is a Series of a single dict-like payload, extract it
                if len(r) == 1 and isinstance(r.iloc[0], dict):
                    rows_list.append(r.iloc[0])
                else:
                    rows_list.append(r.dropna().to_dict())
        except Exception:
            rows_list = mapped_rows.to_dict(orient="records")
    else:
        try:
            rows_list = list(mapped_rows)
        except Exception:
            rows_list = []

    mapped_df = pd.DataFrame(rows_list)

    # Backfill provenance fields from the same metadata artifacts that later
    # feed runs_code, so the runs table stays useful even when the initial
    # run metadata artifact is sparse.
    run_artifact_index = _build_run_artifact_index()
    if not mapped_df.empty:
        mapped_df = pd.DataFrame(
            [
                _apply_run_provenance_defaults(row, run_artifact_index)
                for row in mapped_df.to_dict(orient="records")
            ]
        )

    insert_dataframe(mapped_df, "runs")
    # compute a boolean 'parent' column: True when the run has no parent_id
    try:
        if "parent_id" in mapped_df.columns:
            # parent if parent_id is null/empty
            mapped_df["parent"] = mapped_df["parent_id"].apply(
                lambda x: True if (pd.isna(x) or str(x) == "") else False
            )
        else:
            # if no parent_id field, assume these rows are parent runs
            mapped_df["parent"] = True
    except Exception:
        mapped_df["parent"] = True

    # Return a plain list of run id strings (hashable) so callers can use
    # them directly in SQL queries and as dict keys.
    try:
        id_col = "run_id" if "run_id" in mapped_df.columns else (
            "run_uuid" if "run_uuid" in mapped_df.columns else None
        )
        if id_col is None:
            return []

        # Keep debug output about parent boolean but return only ids
        try:
            mapped_df["parent"] = mapped_df.get("parent", True)
        except Exception:
            pass

        return mapped_df[id_col].astype(str).tolist()
    except Exception:
        try:
            return mapped_df["run_id"].astype(str).tolist()
        except Exception:
            return []


def sync_experiments():
    df = get_experiments_data()

    mapped_rows = df.apply(map_experiments, axis=1)

    # Normalize into list-of-dicts
    if isinstance(mapped_rows, pd.Series):
        rows_list = mapped_rows.tolist()
    elif isinstance(mapped_rows, pd.DataFrame):
        rows_list = [r.dropna().to_dict() for _, r in mapped_rows.iterrows()]
    else:
        try:
            rows_list = list(mapped_rows)
        except Exception:
            rows_list = []

    mapped_df = pd.DataFrame(rows_list)
    insert_dataframe(mapped_df, "experiments")
    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_experiment_tags():
    # Use get_experiment_tags_data() which prefers artifact-based experiment
    # tags. The helper may return a pandas.DataFrame or an iterable of tuples
    # depending on implementation; normalize both shapes here.
    try:
        records = get_experiment_tags_data()
    except Exception:
        records = None

    rows = []

    # If get_experiment_tags_data returned a DataFrame, normalize columns
    if isinstance(records, pd.DataFrame):
        if not records.empty:
            # Expecting columns: experiment_id, key, value
            for col in ("experiment_id", "key", "value"):
                if col not in records.columns:
                    records[col] = None
            records = records.drop_duplicates(subset=["experiment_id", "key"]).copy()
            for _, r in records.iterrows():
                if r.get("experiment_id") and r.get("key") is not None:
                    rows.append(
                        {
                            "experiment_id": str(r.get("experiment_id")),
                            "key": str(r.get("key")),
                            "value": None if pd.isna(r.get("value")) else str(r.get("value")),
                        }
                    )

    # If helper returned an iterable/list of tuples (run_id, experiment_id, record),
    # or a list of dicts, handle those shapes too.
    elif records:
        try:
            for item in records:
                # item could be (run_id, experiment_id, record) or a dict
                if isinstance(item, tuple) and len(item) == 3:
                    _, exp_id, rec = item
                    if isinstance(rec, dict):
                        # artifact writers sometimes use 'experiment_tags' or 'tags'
                        tag_blob = rec.get("experiment_tags") or rec.get("tags") or {}
                        for k, v in (tag_blob or {}).items():
                            rows.append({
                                "experiment_id": str(exp_id or rec.get("experiment_id")),
                                "key": str(k),
                                "value": None if v is None else str(v),
                            })
                elif isinstance(item, dict):
                    exp_id = item.get("experiment_id")
                    tag_blob = item.get("experiment_tags") or item.get("tags") or {}
                    for k, v in (tag_blob or {}).items():
                        rows.append({
                            "experiment_id": str(exp_id),
                            "key": str(k),
                            "value": None if v is None else str(v),
                        })
        except Exception:
            # best-effort: ignore malformed items
            pass

    if not rows:
        return {"rows_synced": 0, "status": "no experiment tags found"}

    df = pd.DataFrame(rows)

    # Ensure experiments exist in DB before inserting tags (FK constraint)
    try:
        # Fetch existing experiments
        existing = insert_dataframe.__self__ if hasattr(insert_dataframe, "__self__") else None
    except Exception:
        existing = None

    # Determine which experiment_ids are missing and create minimal entries
    try:
        # Use get_experiments_data to build any missing experiment rows' metadata
        exp_df = get_experiments_data()
    except Exception:
        exp_df = pd.DataFrame()

    exp_ids = set(df["experiment_id"].astype(str).tolist())

    # Query DB for existing experiments using a light-weight SQL read
    try:
        from sqlalchemy import text

        if exp_ids:
            q = text("SELECT experiment_id FROM experiments WHERE experiment_id = ANY(:ids)")
            with target_engine.connect() as conn:
                res = conn.execute(q, {"ids": list(exp_ids)})
                present = {str(r[0]) for r in res.fetchall()}
        else:
            present = set()
    except Exception:
        present = set()

    missing = [e for e in exp_ids if e not in present]
    if missing:
        now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
        create_rows = []
        for mid in missing:
            # try to enrich from exp_df if available
            meta = None
            try:
                if isinstance(exp_df, pd.DataFrame) and not exp_df.empty:
                    found = exp_df[exp_df["experiment_id"].astype(str) == str(mid)]
                    if not found.empty:
                        meta = found.iloc[0].to_dict()
            except Exception:
                meta = None

            create_rows.append(
                {
                    "experiment_id": str(mid),
                    "experiment_name": meta.get("name") if meta else f"artifact-generated-{str(mid)[:8]}",
                    "lifecycle_stage": meta.get("lifecycle_stage") if meta else "data_processing",
                    "creation_time": meta.get("creation_time") if meta and meta.get("creation_time") is not None else now_ts,
                    "last_update_time": meta.get("last_update_time") if meta and meta.get("last_update_time") is not None else now_ts,
                }
            )

        if create_rows:
            insert_dataframe(pd.DataFrame(create_rows), "experiments")

    # Deduplicate tags per (experiment_id, key) keeping last occurrence
    df = df.drop_duplicates(subset=["experiment_id", "key"], keep="last").copy()

    insert_dataframe(df, "experiments_tags")
    return {"rows_synced": len(df), "status": "success"}


def sync_runtime_environment_from_artifacts(id_mapping: dict):
    """Read runtime_env.json artifacts and upsert into runtime_environment table.

    The runtime env artifact may include deployment_id and model_id. If those
    are missing but the artifact includes a run_id, use id_mapping to resolve
    the deployment_id and model_id for that run.
    """
    try:
        records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    except Exception:
        records = []

    rows = []
    for run_id, experiment_id, rec in records:
        if not isinstance(rec, dict):
            continue
        # runtime_env.json recognized by presence of 'python_version' or 'in_docker'
        if "python_version" in rec or "in_docker" in rec:
            deployment_id = rec.get("deployment_id") or rec.get("deployment")
            model_id = rec.get("model_id") or rec.get("model")
            server_name = rec.get("server_name")
            performance = rec.get("performance")

            # Attempt to read python_env.yaml for this run (optional)
            python_env = None
            try:
                from app.mlflow_connector import get_python_env_for_run

                if run_id:
                    python_env = get_python_env_for_run(run_id)
            except Exception:
                python_env = None

            # Build a combined details payload including both runtime artifact
            # and model/python environment (when available). Keep the original
            # runtime artifact under 'runtime' and python env under 'python_env'.
            details = {"runtime": rec}
            if python_env:
                details["python_env"] = python_env

            # If artifact lacks deployment/model but we have run_id and an
            # id_mapping, attempt to resolve them from id_mapping
            if (not deployment_id or not model_id) and run_id and run_id in id_mapping:
                mapping = id_mapping.get(run_id, {})
                deployment_id = deployment_id or mapping.get("deployment_id")
                model_id = model_id or mapping.get("model_id")

            if deployment_id and experiment_id and model_id:
                rows.append(
                    {
                        "deployment_id": str(deployment_id),
                        "experiment_id": str(experiment_id),
                        "model_id": str(model_id),
                        "run_id": str(run_id) if run_id else None,
                        "server_name": server_name,
                        "performance": performance,
                        "details": details,
                    }
                )

    # If there are runtime environment rows to insert, ensure the
    # model_deployed parent table has the corresponding parent rows. Some
    # runs may have produced runtime_env artifacts before any deployment
    # artifacts; to allow inserting runtime_environment we create minimal
    # model_deployed rows from id_mapping + runs information when the
    # model_deployed table appears empty.
    if rows:
        try:
            from sqlalchemy import MetaData, Table, select

            metadata = MetaData()
            md_table = Table("model_deployed", metadata, autoload_with=target_engine)
            with target_engine.connect() as conn:
                q = select(md_table).limit(1)
                exist = conn.execute(q).fetchone()
        except Exception:
            exist = True  # if reflection fails, avoid creating parents

        # If model_deployed is empty (no parent rows) and we have id_mapping,
        # synthesize minimal parent rows.
        if not exist and id_mapping:
            try:
                runs_df = get_runs_data()
            except Exception:
                runs_df = None

            run_to_experiment = {}
            if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
                for _, r in runs_df.iterrows():
                    rid = r.get("run_uuid") or r.get("run_id")
                    if rid:
                        run_to_experiment[str(rid)] = r.get("experiment_id")

            parent_rows = []
            now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
            for rid, mapping in id_mapping.items():
                exp_id = run_to_experiment.get(rid)
                if not exp_id:
                    # attempt to derive experiment_id via artifacts folder
                    try:
                        from app import mlflow_connector as _mlfc
                        from urllib.parse import urlparse as _up
                        import os as _os

                        parsed = _up(_mlfc.mlflow_artifacts_uri)
                        artifacts_root = parsed.path if parsed.path else _mlfc.mlflow_artifacts_uri
                        # scan experiments if necessary
                        if _os.path.isdir(artifacts_root):
                            for exp_candidate in _os.listdir(artifacts_root):
                                run_folder = _os.path.join(artifacts_root, exp_candidate, str(rid))
                                if _os.path.exists(run_folder):
                                    exp_id = exp_candidate
                                    break
                    except Exception:
                        exp_id = None

                if not exp_id:
                    # skip creating model_deployed without an experiment context
                    continue

                parent_rows.append(
                    {
                        "experiment_id": exp_id,
                        "deployment_id": mapping.get("deployment_id"),
                        "model_id": mapping.get("model_id"),
                        "deployed_time": now_ts,
                        "model_version": 1,
                        "run_id": rid,
                    }
                )

            if parent_rows:
                insert_dataframe(pd.DataFrame(parent_rows), "model_deployed")

        import pandas as _pd

        df = _pd.DataFrame(rows)
        insert_dataframe(df, "runtime_environment")
        return {"rows_synced": len(df), "status": "success"}

    return {"rows_synced": 0, "status": "no runtime env artifacts found"}


def sync_data(id_mapping):
    df = get_datasets_data()

    # Build a run_id -> parent_id map (parent_id is None/"" for root runs) so
    # we can resolve, for every child (sub) run, which run holds the
    # canonical dataset. Root runs get their own data_id computed/looked up;
    # child runs must reuse their parent's data_id instead of generating a
    # brand new one on every insert.
    try:
        runs_df = get_runs_data()
    except Exception:
        runs_df = None

    parent_of = {}
    root_run_ids = set()
    if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
        # runs_df may use 'run_uuid' or 'run_id' depending on MLflow version
        id_col = (
            "run_uuid"
            if "run_uuid" in runs_df.columns
            else ("run_id" if "run_id" in runs_df.columns else None)
        )
        if id_col:
            for _, r in runs_df.iterrows():
                rid = str(r.get(id_col) or r.get("run_id"))
                parent = (
                    r.get("parent_id") if "parent_id" in r.index else r.get("parent_id")
                )
                parent = str(parent) if parent else None
                parent_of[rid] = parent
                # treat empty strings / NaN as no parent
                if not parent:
                    root_run_ids.add(rid)

    def _resolve_root_data_id(run_id: str, seen=None):
        """Walk up the parent chain to find the closest ancestor that already
        has a data_id recorded in id_mapping, returning that data_id."""
        if seen is None:
            seen = set()
        if run_id in seen:
            return None
        seen.add(run_id)
        parent_id = parent_of.get(run_id)
        if not parent_id:
            return None
        parent_data_id = (id_mapping.get(parent_id) or {}).get("data_id")
        if parent_data_id:
            return parent_data_id
        return _resolve_root_data_id(parent_id, seen)

    # Phase 1: process root/parent runs first so we have a canonical
    # data_id per parent before we look at any children.
    parent_rows = []
    for run_id in id_mapping.keys():
        run_id = str(run_id)
        if root_run_ids and run_id not in root_run_ids:
            continue

        row_data = (
            df[df["run_id"] == run_id].iloc[0]
            if not df.empty and "run_id" in df.columns and run_id in df["run_id"].values
            else {}
        )

        mapped_row = map_datasets(row_data, run_id, id_mapping)
        mapped_row["run_id"] = run_id
        # Keep id_mapping's data_id authoritative for the parent's own row.
        if mapped_row.get("data_id"):
            id_mapping.setdefault(run_id, {})["data_id"] = mapped_row["data_id"]
        parent_rows.append(mapped_row)

    # Phase 2: process child (sub) runs, resolving the data_id from the
    # closest ancestor instead of generating/keeping a fresh one.
    child_rows = []
    updated_id_mapping_run_ids = []
    for run_id in id_mapping.keys():
        run_id = str(run_id)
        if root_run_ids and run_id in root_run_ids:
            continue

        row_data = (
            df[df["run_id"] == run_id].iloc[0]
            if not df.empty and "run_id" in df.columns and run_id in df["run_id"].values
            else {}
        )

        mapped_row = map_datasets(row_data, run_id, id_mapping)
        mapped_row["run_id"] = run_id

        inherited_data_id = _resolve_root_data_id(run_id)
        if inherited_data_id:
            mapped_row["data_id"] = inherited_data_id
            # Keep the in-memory id_mapping consistent for downstream syncs
            # (data_metrics, data_resources, data_techniques, ...).
            if (id_mapping.get(run_id) or {}).get("data_id") != inherited_data_id:
                id_mapping.setdefault(run_id, {})["data_id"] = inherited_data_id
                updated_id_mapping_run_ids.append(run_id)
        child_rows.append(mapped_row)

    # Persist the corrected data_id back into id_mapping so future syncs
    # (and other tables that read id_mapping directly) stay in sync.
    if updated_id_mapping_run_ids:
        try:
            from sqlalchemy import MetaData, Table

            metadata = MetaData()
            table = Table("id_mapping", metadata, autoload_with=target_engine)
            with target_engine.begin() as conn:
                for rid in updated_id_mapping_run_ids:
                    conn.execute(
                        table.update()
                        .where(table.c.run_id == rid)
                        .values(data_id=id_mapping[rid]["data_id"])
                    )
        except Exception as exc:
            print(f"[sync_data] failed to persist inherited data_id to id_mapping: {exc}")

    rows_out = parent_rows + child_rows
    if not rows_out:
        return {"rows_synced": 0, "status": "no matching data rows to sync"}

    mapped_df = pd.DataFrame(rows_out)

    insert_dataframe(mapped_df, "data")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_data_signatures(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("model"),
        file_extension="MLmodel",
    )

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    # If there are no rows, nothing to sync
    if df.empty:
        return {"rows_synced": 0, "status": "no model signature artifacts found"}

    # Ensure we have a run identifier column
    if "run_id" not in df.columns:
        # try common alternatives
        if "run_uuid" in df.columns:
            df = df.rename(columns={"run_uuid": "run_id"})
        elif "run" in df.columns:
            df = df.rename(columns={"run": "run_id"})

    # Attempt to locate the signature column. get_artifacts_data flattens
    # MLmodel YAML into dotted keys (e.g. 'signature.inputs'), but some
    # versions may provide 'signature' or a single 'mlmodel_content' fallback.
    signature_col = None
    if "signature.inputs" in df.columns:
        signature_col = "signature.inputs"
    elif "signature" in df.columns:
        signature_col = "signature"
    else:
        # pick any column that starts with 'signature.'
        for c in df.columns:
            if str(c).startswith("signature."):
                signature_col = c
                break

    # If no signature column found, try parsing 'mlmodel_content' column
    if signature_col is None and "mlmodel_content" in df.columns:

        def _extract_signature_from_mlmodel(content):
            try:
                parsed = yaml.safe_load(content)
                sig = parsed.get("signature") if isinstance(parsed, dict) else None
                if isinstance(sig, dict):
                    return sig.get("inputs") or sig
                return sig
            except Exception:
                return None

        df["_parsed_signature"] = df["mlmodel_content"].apply(
            lambda x: _extract_signature_from_mlmodel(x) if isinstance(x, str) else None
        )
        signature_col = "_parsed_signature"

    if signature_col is None or "run_id" not in df.columns:
        # Nothing we can sensibly extract — return without error so /sync/all
        # doesn't fail hard when signature artifacts are absent or malformed.
        return {"rows_synced": 0, "status": "no signature column or run_id present"}

    # Normalize JSON strings into Python objects when necessary
    try:
        df[signature_col] = df[signature_col].apply(
            lambda x: (
                json.loads(x) if isinstance(x, str) and x.strip().startswith("{") else x
            )
        )
    except Exception:
        # best-effort: leave as-is
        pass

    mapped_df = df[[signature_col, "run_id"]].copy()
    mapped_df = mapped_df.rename(columns={signature_col: "signature"})

    mapped_df["data_id"] = mapped_df["run_id"].apply(
        lambda run_id: id_mapping.get(run_id, {}).get("data_id", None)
    )

    insert_dataframe(mapped_df, "data_signatures")

    return {"rows_synced": len(mapped_df), "status": "success"}


@app.post("/sync/data_metrics")
def sync_data_metrics(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("whylogs"),
        file_extension=".csv",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_rows = df.apply(lambda row: map_data_metrics(row, id_mapping), axis=1)
    expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]
    mapped_df = pd.DataFrame(expand_mapped_rows)

    bulk_upsert_metrics(target_engine, mapped_df, chunk_size=500)

    return {"rows_synced": len(mapped_df), "status": "success"}


@app.post("/sync/data_resources")
def sync_data_resources(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("code_carbon"),
        file_extension="emissions_data.json",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_rows = df.apply(
        lambda row: map_data_resources(row, id_mapping), axis=1
    )

    # Normalize mapped_rows into a flat list of dicts. mapped_rows may be a
    # Series (most common), or in some environments a DataFrame; guard both.
    expand_mapped_rows = []
    if isinstance(mapped_rows, pd.Series):
        for val in mapped_rows:
            if isinstance(val, list):
                expand_mapped_rows.extend(val)
            elif pd.isna(val):
                continue
            else:
                expand_mapped_rows.append(val)
    elif isinstance(mapped_rows, pd.DataFrame):
        # each row may hold the actual payload in a single column or be expanded
        for _, row in mapped_rows.iterrows():
            if len(row) == 1:
                single = row.iloc[0]
                if isinstance(single, list):
                    expand_mapped_rows.extend(single)
                else:
                    expand_mapped_rows.append(single)
            else:
                expand_mapped_rows.append(row.to_dict())
    else:
        try:
            expand_mapped_rows = list(mapped_rows)
        except Exception:
            expand_mapped_rows = []

    mapped_df = pd.DataFrame(expand_mapped_rows)
    mapped_df = mapped_df.fillna(0)

    insert_dataframe(mapped_df, "data_resources")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_data_drift(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("dataset"),
        file_extension=".csv",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_df = map_data_drift(df, id_mapping)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_data_duration_leakage(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("dataset"),
        file_extension=".csv",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_df = map_data_duration_leakage(df, id_mapping)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_model_architecture(id_mapping):
    layer_structure = get_artifacts_data(
        folder_name=artifact_path("model"),
        file_extension=".pkl",
    )

    params_df = get_params_data()

    # Build run->params mapping. Select only the key/value columns before groupby
    # to avoid pandas FutureWarning about operating on grouping columns.
    if params_df.empty:
        run_params: dict = {}
    else:
        try:
            grouped = params_df.groupby("run_uuid")[["key", "value"]]
            run_params = grouped.apply(
                lambda g: dict(zip(g["key"], g["value"]))
            ).to_dict()
        except Exception:
            # Fallback for pandas versions that may not support the selection above.
            run_params = (
                params_df.groupby("run_uuid")
                .apply(lambda g: dict(zip(g["key"], g["value"])))
                .to_dict()
            )

    mapped_rows = []

    # Build a run -> parent_id map so we can annotate architecture rows with
    # parent run information (helps detect nested/trial runs). This mirrors
    # other places that call get_runs_data() to derive parent relationships.
    try:
        runs_df = get_runs_data()
        if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
            run_id_col = (
                "run_uuid"
                if "run_uuid" in runs_df.columns
                else ("run_id" if "run_id" in runs_df.columns else None)
            )
            if run_id_col:
                parent_map = runs_df.set_index(run_id_col)["parent_id"].to_dict()
            else:
                parent_map = {}
        else:
            parent_map = {}
    except Exception:
        parent_map = {}

    for run_id in id_mapping.keys():
        params = run_params.get(run_id, {})

        optimizer_name = params.get("optimizer", "unknown")
        optimizer_dict = {"name": optimizer_name}

        for key, value in params.items():
            if key.startswith("optimizer."):
                hyperparam_name = key[len("optimizer.") :]
                optimizer_dict[hyperparam_name] = value

        # Prepare layer_structure payload: it may be a dict or a JSON string
        raw_ls = layer_structure[run_id] if (isinstance(layer_structure, dict) and run_id in layer_structure.keys()) else {}
        ls = {}
        try:
            if isinstance(raw_ls, str):
                try:
                    ls = json.loads(raw_ls)
                except Exception:
                    ls = {"_raw": raw_ls}
            elif isinstance(raw_ls, dict):
                ls = dict(raw_ls)
            else:
                ls = {"_raw": str(raw_ls)}
        except Exception:
            ls = {"_raw": str(raw_ls)}

        parent_id = parent_map.get(run_id)
        parent_model_id = (id_mapping.get(parent_id) or {}).get("model_id") if parent_id else None
        # Do NOT put parent info into the layer_structure (user requested).
        # Instead, embed a short parent identifier into the human-readable
        # architecture_name field so it's easy to detect child/parent runs
        # without changing DB schema types.
        try:
            if parent_id:
                arch_suffix = f"(parent_run={parent_id}"
                if parent_model_id:
                    arch_suffix += f", parent_model={parent_model_id}"
                arch_suffix += ")"
            else:
                arch_suffix = ""
        except Exception:
            arch_suffix = ""

        mapped_rows.append(
            {
                "run_id": run_id,
                "model_id": id_mapping[run_id]["model_id"],
                "architecture_name": f"Simple Model {arch_suffix}".strip(),
                "model_version": 1,
                "layer_structure": ls,
                "activation_function": params.get("activation_function", "ReLU"),
                "optimizer": json.dumps(optimizer_dict),
                "loss_function": params.get("losses", "MSE"),
                "framework": params.get("framework", "unknown"),
                "metrics": [],
                "input_shape": params.get("input_shape", ""),
                "output_shape": params.get("output_shape", ""),
                # If the training run logged parameter counts they will appear
                # in params (via log_model_signature). Parse them when present.
                "number_of_layers": int(params.get("number_of_layers", 0)) if params.get("number_of_layers") is not None else 0,
                "number_of_total_parameters": int(params.get("number_of_total_parameters", 0)) if params.get("number_of_total_parameters") is not None else 0,
                "number_of_trainable_parameters": int(params.get("number_of_trainable_parameters", 0)) if params.get("number_of_trainable_parameters") is not None else 0,
                "number_of_non_trainable_parameters": int(params.get("number_of_non_trainable_parameters", 0)) if params.get("number_of_non_trainable_parameters") is not None else 0,
                "creation_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
            }
        )

    mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "model_architecture")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_model_params(id_mapping):
    df = get_params_data()

    mapped_rows = df.apply(lambda row: map_model_params(row, id_mapping), axis=1)
    mapped_df = pd.DataFrame(mapped_rows.values.tolist())

    insert_dataframe(mapped_df, "model_hyperparameters")
    return {"rows_synced": len(df), "status": "success"}


def sync_model_resources(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("code_carbon"),
        file_extension="emissions_train.json",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_rows = df.apply(lambda row: map_resources(row, id_mapping), axis=1)

    # Normalize mapped_rows into a flat list of dicts
    expand_mapped_rows = []
    if isinstance(mapped_rows, pd.Series):
        for val in mapped_rows:
            if isinstance(val, list):
                expand_mapped_rows.extend(val)
            elif pd.isna(val):
                continue
            else:
                expand_mapped_rows.append(val)
    elif isinstance(mapped_rows, pd.DataFrame):
        for _, row in mapped_rows.iterrows():
            if len(row) == 1:
                single = row.iloc[0]
                if isinstance(single, list):
                    expand_mapped_rows.extend(single)
                else:
                    expand_mapped_rows.append(single)
            else:
                expand_mapped_rows.append(row.to_dict())
    else:
        try:
            expand_mapped_rows = list(mapped_rows)
        except Exception:
            expand_mapped_rows = []

    mapped_df = pd.DataFrame(expand_mapped_rows)
    mapped_df = mapped_df.fillna(0)

    insert_dataframe(mapped_df, "resources")

    return {"rows_synced": len(mapped_df), "status": "success"}


@app.post("/sync/codecarbon_to_resources")
def sync_codecarbon_to_resources(id_mapping: dict):
    """Read CodeCarbon JSON artifacts and insert into resources/data_resources.

    For each JSON artifact found under artifacts/certain/code_carbon, this
    function will map known resource keys (duration, emissions, cpu_power, ...)
    into row entries. If the provided id_mapping contains a model_id for the
    run_id, rows are written into the `resources` table (model-scoped). If no
    model mapping exists for the run, rows are written into `data_resources`.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("code_carbon"))

    if not records:
        return {"rows_synced": 0, "status": "no code_carbon artifacts found"}

    resource_rows = []
    data_resource_rows = []

    # Known numeric/resource keys to map
    from data_api.misc.data_transform import resources_key

    for run_id, experiment_id, record in records:
        # prefer the run_id coming from the artifact path; get_json_artifacts_data
        # returns the run folder as run_id (so record may also contain a run_id)
        r_id = run_id

        model_id = (
            id_mapping.get(r_id, {}).get("model_id") if id_mapping.get(r_id) else None
        )
        data_id = (
            id_mapping.get(r_id, {}).get("data_id") if id_mapping.get(r_id) else None
        )

        stage = (
            record.get("stage", "data_default")
            if isinstance(record, dict)
            else "data_default"
        )

        for key in resources_key:
            if key in record:
                val = record.get(key)
                # Build a resources row if model mapping exists
                if model_id:
                    resource_rows.append(
                        {
                            "run_id": r_id,
                            "model_id": model_id,
                            "key": key,
                            "step": 0,
                            "stage": stage,
                            "value": val,
                            "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                        }
                    )
                else:
                    # Fallback to data_resources when no model mapping
                    data_resource_rows.append(
                        {
                            "run_id": r_id,
                            "data_id": data_id if data_id else "",
                            "stage": stage,
                            "key": key,
                            "value": val,
                            "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                        }
                    )

    rows_synced = 0
    # If some run_ids are missing from id_mapping, create mapping rows so we can
    # insert into resources (model-scoped). This mirrors sync_id_mapping behaviour.
    missing_runs = []
    for run_id, _experiment_id, _record in records:
        if run_id not in id_mapping:
            missing_runs.append(run_id)

    if missing_runs:
        import uuid

        new_mappings = []
        for run_id in missing_runs:
            new_entry = {
                "run_id": run_id,
                "model_id": str(uuid.uuid4()),
                "data_id": str(uuid.uuid4()),
                "deployment_id": str(uuid.uuid4()),
            }
            new_mappings.append(new_entry)
            id_mapping[run_id] = {
                "model_id": new_entry["model_id"],
                "data_id": new_entry["data_id"],
                "deployment_id": new_entry["deployment_id"],
            }

        if new_mappings:
            insert_dataframe(pd.DataFrame(new_mappings), "id_mapping")
    if resource_rows:
        df_res = pd.DataFrame(resource_rows).fillna(0)
        insert_dataframe(df_res, "resources")
        rows_synced += len(df_res)

    if data_resource_rows:
        df_data = pd.DataFrame(data_resource_rows).fillna(0)
        insert_dataframe(df_data, "data_resources")
        rows_synced += len(df_data)

    return {"rows_synced": rows_synced, "status": "success"}


def sync_time_series_data(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("timestamps"),
        file_extension=".txt",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_rows = df.apply(
        lambda row: map_time_series_data(row, id_mapping), axis=1
    )

    expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]
    mapped_df = pd.DataFrame(expand_mapped_rows)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_tags():
    df = get_tags_data()

    # map_runs_tags expects a single row mapping; if get_tags_data
    # returned a DataFrame we should apply the mapper per-row. If it returned
    # a single record or dict-like, handle accordingly.
    try:
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return {"rows_synced": 0, "status": "no tags found"}
            mapped = df.apply(map_runs_tags, axis=1)
            # normalize into list of dicts
            if isinstance(mapped, pd.Series):
                rows_list = mapped.tolist()
            elif isinstance(mapped, pd.DataFrame):
                rows_list = [r.dropna().to_dict() for _, r in mapped.iterrows()]
            else:
                try:
                    rows_list = list(mapped)
                except Exception:
                    rows_list = []
        else:
            # non-DataFrame input (dict/series/scalar) — attempt single mapping
            rows_list = [map_mlflow_runs_tags(df)]

        mapped_df = pd.DataFrame(rows_list)

        if mapped_df.empty:
            return {"rows_synced": 0, "status": "no tags mapped"}

        # Remove rows where the key contains "git"
        mapped_df = mapped_df[
            ~mapped_df["key"].astype(str).str.contains("git", case=False, na=False)
        ].copy()
        # Normalize column names: some connector functions use 'run_uuid'
        # while the DB schema expects 'run_id'. Ensure 'run_id' is present.
        if "run_id" not in mapped_df.columns and "run_uuid" in mapped_df.columns:
            mapped_df["run_id"] = mapped_df["run_uuid"].astype(str)

        # Ensure run_id column exists and is string typed
        if "run_id" in mapped_df.columns:
            mapped_df["run_id"] = mapped_df["run_id"].astype(str)
        else:
            # Nothing we can do safely without a run identifier
            return {"rows_synced": 0, "status": "no run_id available in tags"}

        # Drop malformed/empty run_id values to avoid generating DB defaults
        mapped_df = mapped_df[mapped_df["run_id"].astype(str).str.strip() != ""].copy()

        # Ensure parent runs exist for all run_ids we are about to insert.
        try:
            from sqlalchemy import MetaData, Table, select

            metadata = MetaData()
            runs_table = Table("runs", metadata, autoload_with=target_engine)
            # Find missing run_ids by querying the DB
            run_ids = list(mapped_df["run_id"].unique())
            if run_ids:
                with target_engine.connect() as conn:
                    q = select(runs_table.c.run_id).where(runs_table.c.run_id.in_(run_ids))
                    existing = {row[0] for row in conn.execute(q).fetchall()}
            else:
                existing = set()

        except Exception:
            existing = set()

        missing = [r for r in mapped_df["run_id"].unique() if r not in existing]
        if missing:
            # Try to derive experiment ids from runs artifacts or fallback to
            # the first experiment in get_experiments_data(). Create minimal
            # runs rows so FK for runs_tags is satisfied.
            now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
            run_rows = []
            try:
                # build mapping run->experiment from artifacts
                from app import mlflow_connector as _mlfc
                from urllib.parse import urlparse as _urlparse
                import os as _os

                parsed = _urlparse(_mlfc.mlflow_artifacts_uri)
                artifacts_root = parsed.path if parsed.path else _mlfc.mlflow_artifacts_uri
                run_to_exp = {}
                if _os.path.isdir(artifacts_root):
                    for exp in _os.listdir(artifacts_root):
                        exp_path = _os.path.join(artifacts_root, exp)
                        if not _os.path.isdir(exp_path):
                            continue
                        for candidate in _os.listdir(exp_path):
                            run_to_exp[candidate] = exp
            except Exception:
                run_to_exp = {}

            try:
                exp_df = get_experiments_data()
                fallback_exp = None
                if isinstance(exp_df, pd.DataFrame) and not exp_df.empty:
                    fallback_exp = exp_df.iloc[0].get("experiment_id")
            except Exception:
                fallback_exp = None

            for r in missing:
                exp_id = run_to_exp.get(r) or fallback_exp or str(uuid.uuid4())
                run_rows.append(
                    _apply_run_provenance_defaults(
                        {
                            "run_id": r,
                            "run_name": None,
                            "parent_id": None,
                            "source_type": "LOCAL",
                            "source_name": "",
                            "user_id": "",
                            "status": "FINISHED",
                            "start_time": now_ts,
                            "end_time": now_ts,
                            "source_version": "",
                            "experiment_id": exp_id,
                        },
                        _build_run_artifact_index(),
                    )
                )

            if run_rows:
                insert_dataframe(pd.DataFrame(run_rows), "runs")

        insert_dataframe(mapped_df, "runs_tags")
        return {"rows_synced": len(mapped_df), "status": "success"}
    except Exception as exc:
        # Do not crash the entire /sync/all; surface a warning instead.
        print(f"sync_tags failed: {exc}")
        return {"rows_synced": 0, "status": "failed to sync tags"}


def sync_run_code(run_ids: list):
    """Populate runs_code from MLflow system tags."""
    tags_df = get_tags_data()

    # Read JSON artifact metadata records for 'certain/metadata' if present and
    # build a mapping run_id -> metadata record so we can prefer stored
    # git_metadata.json over inspecting the git repo.
    artifact_records = {}
    try:
        json_records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
        # json_records: list of (run_id, experiment_id, record)
        for run_id, experiment_id, record in json_records:
            # Recognize git metadata records by presence of git.commit or similar
            if isinstance(record, dict) and (
                "git.commit" in record or "git.commit.short" in record
            ):
                artifact_records[run_id] = record
    except Exception:
        artifact_records = {}

    rows = []

    for run_id in run_ids:
        # artifact_records maps run_id -> record (dict) when git metadata is
        # present. Pass the dict (or empty dict) into map_run_code which
        # expects a mapping-like artifact_git_record.
        row = map_run_code(
            tags_df,
            run_id,
            artifact_git_record=artifact_records.get(run_id, {}),
        )
        if row:
            rows.append(row)

    if not rows:
        return {"rows_synced": 0, "status": "no git tags found"}

    mapped_df = pd.DataFrame(rows)
    insert_dataframe(mapped_df, "runs_code")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_checkpoints(id_mapping: dict):
    """Populate checkpoints from artifacts/certain/checkpoints/*.csv."""
    try:
        df = get_artifacts_data(
            folder_name=artifact_path("checkpoints"),
            file_extension=".csv",
        )
    except Exception:
        return {"rows_synced": 0, "status": "no checkpoint artifacts found"}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows_synced": 0, "status": "no checkpoint artifacts found"}

    mapped_rows = df.apply(
        lambda row: map_checkpoints(row, id_mapping), axis=1
    ).tolist()

    mapped_df = pd.DataFrame(mapped_rows)
    insert_dataframe(mapped_df, "checkpoints")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_weight_distribution(id_mapping: dict):
    """Populate weight_distribution from artifacts/certain/weight_distribution/*.csv."""
    try:
        df = get_artifacts_data(
            folder_name=artifact_path("weight_distribution"),
            file_extension=".csv",
        )
    except Exception:
        return {"rows_synced": 0, "status": "no weight distribution artifacts found"}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows_synced": 0, "status": "no weight distribution artifacts found"}

    mapped_rows = df.apply(
        lambda row: map_weight_distribution(row, id_mapping), axis=1
    ).tolist()

    mapped_df = pd.DataFrame(mapped_rows)
    insert_dataframe(mapped_df, "weight_distribution")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_examples(id_mapping: dict):
    """Populate examples from artifacts.

    Sources:
    - artifacts/certain/examples/*.csv
    - artifacts/certain/model/input_examples.json
    """

    rows_list = []

    # 1) Existing CSV source: certain/examples/*.csv
    try:
        df = get_artifacts_data(
            folder_name=artifact_path("examples"),
            file_extension=".csv",
        )
    except Exception:
        df = pd.DataFrame()

    if isinstance(df, pd.DataFrame) and not df.empty:
        mapped_rows = df.apply(lambda row: map_examples(row, id_mapping), axis=1)

        if isinstance(mapped_rows, pd.Series):
            rows_list.extend(mapped_rows.tolist())
        elif isinstance(mapped_rows, pd.DataFrame):
            rows_list.extend([r.dropna().to_dict() for _, r in mapped_rows.iterrows()])
        else:
            try:
                rows_list.extend(list(mapped_rows))
            except Exception:
                pass

    # 2) New JSON sources:
    # - certain/model/input_examples.json (legacy)
    # - certain/examples/example_step_XX.json (per-step logger)
    json_records = []
    try:
        json_records = get_json_artifacts_data(
            folder_name=artifact_path("model"),
            file_name="input_examples.json",
        )
    except Exception:
        json_records = []

    # Also consider per-step JSON examples written under artifacts/certain/examples
    try:
        json_records_examples = get_json_artifacts_data(
            folder_name=artifact_path("examples"),
            file_name=None,
        )
    except Exception:
        json_records_examples = []

    # Merge both JSON sources
    if isinstance(json_records_examples, list) and json_records_examples:
        json_records.extend(json_records_examples)

    # also read per-step example JSON files under certain/examples
    try:
        json_examples = get_json_artifacts_data(
            folder_name=artifact_path("examples"),
            file_name=None,
        )
        if isinstance(json_examples, list) and json_examples:
            json_records.extend(json_examples)
    except Exception:
        pass

    now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

    def _to_text(value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    def _extract_items(record):
        if isinstance(record, list):
            return record
        if not isinstance(record, dict):
            return []

        for key in ("input_examples", "examples", "inputs", "rows"):
            candidate = record.get(key)
            if isinstance(candidate, list):
                return candidate

        # MLflow DataFrame-style input_example JSON often uses split orient:
        # {"columns": [...], "data": [[...], ...], "index": [...]}
        columns = record.get("columns")
        data = record.get("data")
        if isinstance(columns, list) and isinstance(data, list) and data:
            normalized_rows = []
            for row in data:
                if isinstance(row, dict):
                    normalized_rows.append(row)
                    continue
                if isinstance(row, (list, tuple)):
                    if len(columns) == len(row):
                        normalized_rows.append(dict(zip(columns, row)))
                    else:
                        normalized_rows.append({"values": list(row)})
                    continue
                normalized_rows.append({"value": row})
            return normalized_rows

        # Some model artifacts store a single example payload directly.
        if any(key in record for key in ("input", "features", "x", "prediction", "ground_truth", "label", "target")):
            return [record]

        return [record]

    def _extract_model_id(record, run_id):
        if isinstance(record, dict):
            for key in ("model_id", "model", "model_uuid"):
                value = record.get(key)
                if value not in (None, ""):
                    return str(value)
        return str(id_mapping.get(run_id, {}).get("model_id", ""))

    for run_id, _experiment_id, record in json_records:
        items = _extract_items(record)
        model_id = _extract_model_id(record, run_id)

        for idx, item in enumerate(items):
            if isinstance(item, dict):
                raw_input = item.get(
                    "input",
                    item.get(
                        "features",
                        item.get("x", item.get("values", item)),
                    ),
                )
                prediction = item.get("prediction", item.get("predicted", ""))
                ground_truth = item.get(
                    "ground_truth", item.get("label", item.get("target", ""))
                )
                step_value = item.get("step", idx)
                stage_value = item.get("stage", "input_examples")
                ts_value = item.get("timestamp", now_ts)
            else:
                raw_input = item
                prediction = ""
                ground_truth = ""
                step_value = idx
                stage_value = "input_examples"
                ts_value = now_ts

            try:
                step_value = int(step_value)
            except Exception:
                step_value = idx

            rows_list.append(
                {
                    "run_id": str(run_id),
                    "model_id": str(model_id),
                    "input": _to_text(raw_input),
                    "prediction": _to_text(prediction),
                    "ground_truth": _to_text(ground_truth),
                    "step": step_value,
                    "stage": _to_text(stage_value) or "input_examples",
                    "timestamp": ts_value,
                }
            )

    if not rows_list:
        return {"rows_synced": 0, "status": "no examples artifacts found"}

    mapped_df = pd.DataFrame(rows_list)
    insert_dataframe(mapped_df, "examples")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_run_logs(run_ids: list):
    """Populate runs_logs from artifacts/certain/run_logs/*.csv."""
    try:
        df = get_artifacts_data(
            folder_name=artifact_path("run_logs"),
            file_extension=".csv",
        )
    except Exception:
        return {"rows_synced": 0, "status": "no run_logs artifacts found"}

    try:
        print(f"[sync_run_logs] artifacts dataframe shape: {getattr(df, 'shape', None)}")
        # show a small sample of distinct run identifiers found in artifacts
        sample_ids = []
        if 'run_id' in df.columns:
            sample_ids = list(df['run_id'].astype(str).unique())[:10]
        elif 'run_uuid' in df.columns:
            sample_ids = list(df['run_uuid'].astype(str).unique())[:10]
    except Exception as e:
        print(f"[sync_run_logs] error inspecting artifact dataframe: {e}")

    # Determine which run_ids we should consider. The caller may pass a
    # simple list of run_id strings, or the richer output from sync_runs()
    # (list of dicts with keys 'run_id' and 'parent'). If dicts are provided,
    # only consider runs where parent == True.
    run_ids_set = set()
    if not run_ids:
        # Nothing to sync when no run ids provided
        return {"rows_synced": 0, "status": "no run_ids provided"}

    # If the first element is a mapping, assume list-of-dicts format
    if isinstance(run_ids, list) and len(run_ids) > 0 and isinstance(run_ids[0], dict):
        for item in run_ids:
            try:
                if item.get("parent"):
                    run_ids_set.add(str(item.get("run_id")))
            except Exception:
                continue
    else:
        # Assume simple iterable of run_id strings
        try:
            run_ids_set = {str(r) for r in run_ids}
        except Exception:
            run_ids_set = set()

    if not run_ids_set:
        return {"rows_synced": 0, "status": "no parent runs to sync"}

    # Normalize possible run identifier column names (run_id or run_uuid)
    id_col = None
    if "run_id" in df.columns:
        id_col = "run_id"
    elif "run_uuid" in df.columns:
        id_col = "run_uuid"

    mapped_rows = []
    matched_count = 0
    mapped_count = 0
    map_exceptions = 0
    for _, row in df.iterrows():
        run_id_val = None
        if id_col:
            run_id_val = row.get(id_col)
        else:
            run_id_val = row.get("run") or row.get("runId") or row.get("run_uuid")

        if run_id_val is None:
            continue

        run_id_str = str(run_id_val)
        if run_id_str in run_ids_set:
            matched_count += 1
            try:
                mapped = map_run_logs(row, run_id_str)
                if mapped:
                    mapped_rows.append(mapped)
                    mapped_count += 1
            except Exception as e:
                map_exceptions += 1
                # print one example exception to help debugging
                if map_exceptions == 1:
                    print(f"[sync_run_logs] mapping exception for run {run_id_str}: {e}")
                continue

    if not mapped_rows:
        return {"rows_synced": 0, "status": "no matching run log rows"}

    mapped_df = pd.DataFrame(mapped_rows)
    insert_dataframe(mapped_df, "runs_logs")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_tokenizer_config(run_ids: list):
    """Populate tokenizer_config from artifacts/certain/tokenizer_config/*.json."""
    return _sync_json_run_table(
        "tokenizer_config", map_tokenizer_config, "tokenizer_config", run_ids
    )


def sync_tokenization_stats(run_ids: list):
    """Populate tokenization_stats from artifacts/certain/tokenization_stats/*.json."""
    return _sync_json_run_table(
        "tokenization_stats", map_tokenization_stats, "tokenization_stats", run_ids
    )


def sync_data_techniques(run_ids: list, id_mapping: dict):
    """Populate data_techniques from artifacts/certain/data_techniques/*.json.

    For each JSON artifact found, map it into a row and resolve the data_id
    using the provided id_mapping (run_id -> data_id). If no data_id exists for
    a run, the row will be skipped to avoid violating non-null FK constraints.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("data_techniques"))

    if not records:
        return {"rows_synced": 0, "status": "no data_techniques artifacts found"}

    techniques_out = []
    hyperparams_out = []

    for run_id, _exp_id, record in records:
        if run_id not in run_ids:
            continue

        mapped = map_data_techniques(record, run_id)

        # mapped is expected to be {'techniques': [...], 'hyperparameters': [...]}
        trows = mapped.get("techniques", []) or []
        hrows = mapped.get("hyperparameters", []) or []

        # Resolve data_id for each row and collect
        for r in trows:
            if run_id in id_mapping and id_mapping[run_id].get("data_id"):
                r["data_id"] = id_mapping[run_id]["data_id"]
                techniques_out.append(r)

        for r in hrows:
            if run_id in id_mapping and id_mapping[run_id].get("data_id"):
                r["data_id"] = id_mapping[run_id]["data_id"]
                hyperparams_out.append(r)

    rows_synced = 0
    if techniques_out:
        insert_dataframe(pd.DataFrame(techniques_out), "data_techniques")
        rows_synced += len(techniques_out)

    if hyperparams_out:
        insert_dataframe(pd.DataFrame(hyperparams_out), "data_hyperparameters")
        rows_synced += len(hyperparams_out)

    if rows_synced == 0:
        return {"rows_synced": 0, "status": "no matching data_techniques rows"}

    return {"rows_synced": rows_synced, "status": "success"}


def sync_drift_metrics(run_ids: list, id_mapping: dict):
    """Read drift_metrics artifacts and persist them.

    Behavior:
      - Always insert per-column p_values into run-scoped `data_metrics` as
        key: "[drift_metrics]{column}" and value: p_value (stringified).
      - If the artifact contains a deployed model (deployment_id/model_id != "not deployed yet"),
        insert an aggregated integer indicator into `drift_metrics` table where
        value=1 if any column p_value < 0.05 else 0. Otherwise, write a
        `monitor_logs` entry describing the summary.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("drift_metrics"))

    if not records:
        return {"rows_synced": 0, "status": "no drift_metrics artifacts found"}

    data_metrics_rows = []
    drift_metric_rows = []
    monitor_log_rows = []

    # Keep parent rows we may need to create for drift_metrics
    parent_rows_for_drift = []

    for run_id, exp_from_store, record in records:
        if run_id not in run_ids:
            continue

        # record expected shape: {run_id, model: {experiment_id,deployment_id,model_id}, columns: [...], summary: {...}}
        model_info = (record.get("model") or {}) if isinstance(record, dict) else {}
        # Prefer explicit fields in the artifact; fall back to the experiment id returned by artifact discovery
        experiment_id = model_info.get("experiment_id") or exp_from_store
        # Try to resolve deployment/model from artifact, otherwise from the id_mapping for this run
        deployment_id = model_info.get("deployment_id") or id_mapping.get(
            run_id, {}
        ).get("deployment_id")
        model_id = model_info.get("model_id") or id_mapping.get(run_id, {}).get(
            "model_id"
        )

        cols = record.get("columns") or []
        for c in cols:
            column = c.get("column")
            pval = c.get("p_value")
            # Insert into run-scoped data_metrics so we always preserve a copy
            data_metrics_rows.append(
                {
                    "run_id": run_id,
                    "data_id": id_mapping.get(run_id, {}).get("data_id"),
                    "key": f"[drift_metrics]{column}",
                    "value": str(pval),
                    "timestamp": int(
                        c.get("timestamp") or pd.Timestamp.now(tz="UTC").timestamp()
                    ),
                    "data_stage": "eval",
                    "is_NaN": False,
                }
            )

        # If model is deployed (we have both IDs), insert an aggregated drift indicator into drift_metrics
        # If IDs are missing but we have an experiment_id, synthesize minimal parent rows and still insert the indicator.
        if deployment_id and model_id and deployment_id != "not deployed yet":
            summary = record.get("summary") or {}
            # value: 1 if any p_value < 0.05 else 0
            num_drift = summary.get("num_drift", 0)
            indicator = 1 if int(num_drift) > 0 else 0
            drift_metric_rows.append(
                {
                    "experiment_id": experiment_id,
                    "deployment_id": deployment_id,
                    "model_id": model_id,
                    "value": int(indicator),
                    "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                }
            )
        else:
            # Not deployed: add a monitor log with summary so operators can see the status
            # The monitor_logs table enforces non-null PK columns (deployment_id, experiment_id, model_id).
            # Use a sentinel string to avoid NOT NULL constraint violations when the artifact lacks these IDs.
            monitor_log_rows.append(
                {
                    "experiment_id": experiment_id or "not_deployed_yet",
                    "deployment_id": deployment_id or "not_deployed_yet",
                    "model_id": model_id or "not_deployed_yet",
                    "log_id": str(uuid.uuid4()),
                    "message": json.dumps({"drift_summary": record.get("summary")}),
                }
            )
            # Additionally, if we have an experiment_id and numeric summary, we may still want to insert
            # a drift metric row. Synthesize deployment/model ids and remember to create parent rows.
            if experiment_id and (deployment_id is None or model_id is None):
                # create generated ids to allow insertion; these will be added to model_deployed below
                gen_dep = deployment_id or str(uuid.uuid4())
                gen_mod = model_id or str(uuid.uuid4())
                summary = record.get("summary") or {}
                num_drift = summary.get("num_drift", 0)
                indicator = 1 if int(num_drift) > 0 else 0
                drift_metric_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "deployment_id": gen_dep,
                        "model_id": gen_mod,
                        "value": int(indicator),
                        "timestamp": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    }
                )
                parent_rows_for_drift.append(
                    {
                        "experiment_id": experiment_id,
                        "deployment_id": gen_dep,
                        "model_id": gen_mod,
                        "deployed_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
                        "run_id": run_id,
                    }
                )

    rows_synced = 0
    if data_metrics_rows:
        insert_dataframe(pd.DataFrame(data_metrics_rows), "data_metrics")
        rows_synced += len(data_metrics_rows)

    if drift_metric_rows:
        # Ensure parent rows exist for any drift_metric entries we will insert
        try:
            from sqlalchemy import MetaData, Table, select

            metadata = MetaData()
            md_table = Table("model_deployed", metadata, autoload_with=target_engine)

            needed = set()
            for r in drift_metric_rows:
                needed.add(
                    (r.get("experiment_id"), r.get("deployment_id"), r.get("model_id"))
                )

            existing = set()
            with target_engine.connect() as conn:
                for exp_id, dep_id, mod_id in needed:
                    q = (
                        select(md_table)
                        .where(
                            (md_table.c.experiment_id == exp_id)
                            & (md_table.c.deployment_id == dep_id)
                            & (md_table.c.model_id == mod_id)
                        )
                        .limit(1)
                    )
                    res = conn.execute(q).fetchone()
                    if res:
                        existing.add((exp_id, dep_id, mod_id))

            missing = [k for k in needed if k not in existing]
            if missing or parent_rows_for_drift:
                parent_rows = []
                # include any synthesized parent rows we collected earlier
                parent_rows.extend(parent_rows_for_drift)
                # create minimal parents for missing keys
                now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
                for exp_id, dep_id, mod_id in missing:
                    parent_rows.append(
                        {
                            "experiment_id": exp_id,
                            "deployment_id": dep_id,
                            "model_id": mod_id,
                            "deployed_time": now_ts,
                            "run_id": id_mapping.get(next(iter(id_mapping)), {}).get(
                                "run_id", None
                            ),
                        }
                    )

                if parent_rows:
                    insert_dataframe(pd.DataFrame(parent_rows), "model_deployed")
        except Exception:
            pass

        insert_dataframe(pd.DataFrame(drift_metric_rows), "drift_metrics")
        rows_synced += len(drift_metric_rows)

    if monitor_log_rows:
        # Ensure parent rows exist in model_deployed for the monitor_logs entries
        try:
            from sqlalchemy import MetaData, Table, select

            metadata = MetaData()
            md_table = Table("model_deployed", metadata, autoload_with=target_engine)
            # build set of parent keys we need
            needed_parents = set()
            for r in monitor_log_rows:
                needed_parents.add(
                    (r.get("experiment_id"), r.get("deployment_id"), r.get("model_id"))
                )

            # Query existing parent keys
            existing = set()
            with target_engine.connect() as conn:
                for exp_id, dep_id, mod_id in needed_parents:
                    q = (
                        select(md_table)
                        .where(
                            (md_table.c.experiment_id == exp_id)
                            & (md_table.c.deployment_id == dep_id)
                            & (md_table.c.model_id == mod_id)
                        )
                        .limit(1)
                    )
                    res = conn.execute(q).fetchone()
                    if res:
                        existing.add((exp_id, dep_id, mod_id))

            # Insert missing parent rows with minimal required columns
            missing = [k for k in needed_parents if k not in existing]
            if missing:
                parent_rows = []
                now_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
                # try to pick a run_id from id_mapping if available, else leave None
                for exp_id, dep_id, mod_id in missing:
                    # prefer None -> let the DB accept NULL for non-pk run_id
                    # try find a run that maps to these ids
                    chosen_run = None
                    for run, mapping in id_mapping.items():
                        if (
                            mapping.get("deployment_id") == dep_id
                            or mapping.get("model_id") == mod_id
                            or mapping.get("data_id") == mod_id
                        ):
                            chosen_run = run
                            break
                    parent_rows.append(
                        {
                            "experiment_id": exp_id,
                            "deployment_id": dep_id,
                            "model_id": mod_id,
                            "deployed_time": now_ts,
                            "run_id": chosen_run,
                        }
                    )

                if parent_rows:
                    insert_dataframe(pd.DataFrame(parent_rows), "model_deployed")
        except Exception:
            # If reflection or insertion fails, proceed and let the monitor_logs insert surface the error
            pass

        insert_dataframe(pd.DataFrame(monitor_log_rows), "monitor_logs")
        rows_synced += len(monitor_log_rows)

    if rows_synced == 0:
        return {"rows_synced": 0, "status": "no matching drift rows"}

    return {"rows_synced": rows_synced, "status": "success"}


def _sync_json_experiment_table(
    folder_name: str, map_fn, table_name: str, run_ids: list, id_mapping: Optional[dict] = None
):
    """Generic sync for experiment-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=artifact_path(folder_name))

    if not records:
        return {"rows_synced": 0, "status": f"no {folder_name} artifacts found"}

    rows = []

    for run_id, experiment_id, record in records:
        if run_id not in run_ids:
            continue

        sig = inspect.signature(map_fn)
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

        def _apply_mapping(mapped: dict) -> dict:
            if isinstance(mapped, dict):
                mapping = (id_mapping or {}).get(run_id, {})
                if mapping and mapping.get("deployment_id"):
                    mapped["deployment_id"] = mapping.get("deployment_id")
                if mapping and mapping.get("model_id") and "model_id" in mapped:
                    mapped["model_id"] = mapping.get("model_id")
            return mapped

        # Accept either a single object per-file or a list of objects
        if isinstance(record, list):
            for entry in record:
                if len(positional_params) >= 3:
                    mapped = map_fn(entry, experiment_id, run_id)
                else:
                    mapped = map_fn(entry, experiment_id)
                rows.append(_apply_mapping(mapped))
        else:
            if len(positional_params) >= 3:
                mapped = map_fn(record, experiment_id, run_id)
            else:
                mapped = map_fn(record, experiment_id)
            rows.append(_apply_mapping(mapped))

    if not rows:
        return {"rows_synced": 0, "status": f"no matching {folder_name} rows"}

    insert_dataframe(pd.DataFrame(rows), table_name)

    return {"rows_synced": len(rows), "status": "success"}


def _sync_json_run_table(
    folder_name: str,
    map_fn,
    table_name: str,
    run_ids: list,
    id_mapping: Optional[dict] = None,
):
    """Generic sync for run-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=artifact_path(folder_name))

    if not records:
        return {"rows_synced": 0, "status": f"no {folder_name} artifacts found"}

    rows = []

    for run_id, _experiment_id, record in records:
        if run_id in run_ids:
            mapped = map_fn(record, run_id)
            if isinstance(mapped, dict):
                mapping = (id_mapping or {}).get(run_id, {})
                if mapping:
                    if "deployment_id" in mapped and mapping.get("deployment_id"):
                        mapped["deployment_id"] = mapping.get("deployment_id")
                    if "model_id" in mapped and mapping.get("model_id"):
                        mapped["model_id"] = mapping.get("model_id")
            rows.append(mapped)

    if not rows:
        return {"rows_synced": 0, "status": f"no matching {folder_name} rows"}

    insert_dataframe(pd.DataFrame(rows), table_name)

    return {"rows_synced": len(rows), "status": "success"}


def _sync_json_deployment_table(
    folder_name: str, map_fn, table_name: str, run_ids: list, id_mapping: dict
):
    """Generic sync for deployment-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=folder_name)
    if not records:
        return {"rows_synced": 0, "status": f"no {folder_name} artifacts found"}
    rows = []
    for run_id, experiment_id, record in records:
        if run_id in run_ids:
            sig = inspect.signature(map_fn)
            required_positional = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is inspect._empty
            ]

            if len(required_positional) >= 3:
                third_param_name = required_positional[2].name.lower()
                if "mapping" in third_param_name:
                    # e.g. map_model_packaging(record, experiment_id, id_mapping)
                    mapped = map_fn(record, experiment_id, id_mapping)
                else:
                    # e.g. map_interface(record, experiment_id, run_id)
                    mapped = map_fn(record, experiment_id, run_id)
            elif len(required_positional) == 2:
                # e.g. map_build_testing(record, experiment_id)
                mapped = map_fn(record, experiment_id)
            elif len(required_positional) == 1:
                mapped = map_fn(record)
            else:
                mapped = map_fn()

            if isinstance(mapped, dict):
                mapping = (id_mapping or {}).get(run_id, {})
                if mapping:
                    if "deployment_id" in mapped and mapping.get("deployment_id"):
                        mapped["deployment_id"] = mapping.get("deployment_id")
                    if "model_id" in mapped and mapping.get("model_id"):
                        mapped["model_id"] = mapping.get("model_id")
            rows.append(mapped)
    if not rows:
        return {"rows_synced": 0, "status": f"no matching {folder_name} rows"}
    insert_dataframe(pd.DataFrame(rows), table_name)
    return {"rows_synced": len(rows), "status": "success"}


def _replace_run_scoped_rows(table_name: str, run_ids: list) -> None:
    """Remove existing rows for the selected runs before re-syncing them."""
    unique_run_ids = []
    seen = set()
    for run_id in run_ids or []:
        run_id = str(run_id)
        if run_id and run_id not in seen:
            unique_run_ids.append(run_id)
            seen.add(run_id)

    if not unique_run_ids:
        return

    placeholders = ", ".join(f":run_id_{idx}" for idx in range(len(unique_run_ids)))
    params = {f"run_id_{idx}": run_id for idx, run_id in enumerate(unique_run_ids)}

    with target_engine.begin() as conn:
        conn.execute(
            text(f'DELETE FROM "{table_name}" WHERE run_id IN ({placeholders})'),
            params,
        )


def sync_ai_actors(run_ids: list, id_mapping: dict):  # noqa: ARG001
    # ai_actors are now run-scoped (referencing runs.run_id). Use the
    # run-scoped generic sync helper so the mapper receives run_id.
    _replace_run_scoped_rows("ai_actors", run_ids)
    return _sync_json_run_table("ai_actors", map_ai_actors, "ai_actors", run_ids)


def sync_labeling_procedures(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table(
        "labeling_procedures", map_labeling_procedures, "labeling_procedures", run_ids
    )


def sync_risks(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table("risks", map_risk, "risks", run_ids)


def sync_human_oversight(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table(
        "human_oversight",
        map_human_oversight,
        "human_oversight_mechanisms",
        run_ids,
        id_mapping,
    )


def sync_transparency_measures(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table(
        "transparency_measures",
        map_transparency_measure,
        "transparency_measures",
        run_ids,
        id_mapping,
    )


def sync_change_logs(run_ids: list):
    return _sync_json_run_table("change_logs", map_change_log, "change_logs", run_ids)


def sync_declaration_of_conformity(run_ids: list, id_mapping: dict):
    return _sync_json_run_table(
        "declaration_of_conformity",
        map_declaration_of_conformity,
        "declaration_of_conformity",
        run_ids,
        id_mapping,
    )


def sync_visual_documentation(run_ids: list):
    return _sync_json_run_table(
        "visual_documentation",
        map_visual_documentation,
        "visual_documentation",
        run_ids,
    )


def sync_explainable_ai(run_ids: list, id_mapping: dict):
    return _sync_json_run_table(
        "explainable_ai",
        map_explainable_ai,
        "explainable_ai_features",
        run_ids,
        id_mapping,
    )


def sync_model_packaging(run_ids: list, id_mapping: dict):
    return _sync_json_deployment_table(
        artifact_path("model_packaging"),
        map_model_packaging,
        "model_packaging",
        run_ids,
        id_mapping,
    )


def sync_build_testing(run_ids: list, id_mapping: dict):
    # Build/testing rows depend on model_deployed parents. If they are missing,
    # synthesize them from the deployment-scoped artifacts before inserting the
    # build/testing rows.
    if not _model_deployed_has_rows() or not model_deployed_has_rows():
        sync_model_deployed(run_ids, id_mapping)

    if not _model_deployed_has_rows() or not model_deployed_has_rows():
        return {"rows_synced": 0, "status": "skipped - model_deployed empty"}

    return _sync_json_deployment_table(
        artifact_path("build_and_integration_testing"),
        map_build_testing,
        "build_and_integration_testing",
        run_ids,
        id_mapping,
    )


def sync_standards(run_ids: list, id_mapping: dict):
    _replace_run_scoped_rows("standards", run_ids)
    return _sync_json_run_table(
        "standards", map_standard, "standards", run_ids, id_mapping
    )


def model_deployed_has_rows() -> bool:
    """Return True if the `model_deployed` table contains at least one row.

    This is a lightweight guard used to avoid attempting to insert
    deployment-scoped child rows when no parent rows exist yet.
    """
    try:
        from sqlalchemy import MetaData, Table, select

        metadata = MetaData()
        md_table = Table("model_deployed", metadata, autoload_with=target_engine)
        with target_engine.connect() as conn:
            q = select(md_table).limit(1)
            res = conn.execute(q).fetchone()
            return res is not None
    except Exception:
        # On reflection/connection errors, assume parents may exist and
        # return True to avoid falsely skipping syncs. Errors will surface
        # during normal sync operations instead.
        return True


def _model_deployed_has_rows() -> bool:
    """Return True if the `model_deployed` table contains at least one row.

    We use SQLAlchemy reflection to avoid importing the ORM models directly.
    If reflection fails we conservatively return True to avoid skipping syncs
    in environments where we cannot introspect the DB.
    """
    try:
        from sqlalchemy import MetaData, Table, select

        metadata = MetaData()
        md_table = Table("model_deployed", metadata, autoload_with=target_engine)
        with target_engine.connect() as conn:
            q = select(md_table).limit(1)
            exist = conn.execute(q).fetchone()
        return bool(exist)
    except Exception:
        # If anything goes wrong with reflection, don't block syncing.
        return True


def sync_interfaces(run_ids: list, id_mapping: dict):
    return _sync_json_deployment_table(
        artifact_path("interfaces"), map_interface, "interfaces", run_ids, id_mapping
    )


def sync_decommissioning(run_ids: list, id_mapping: dict):
    return _sync_json_deployment_table(
        artifact_path("decommissioning"),
        map_decommissioning,
        "decomissioning",
        run_ids,
        id_mapping,
    )


def sync_monitor_logs(run_ids: list, id_mapping: dict):
    records = get_json_artifacts_data(
        folder_name=artifact_path("deployment_logs"), file_name="monitor_log.json"
    )

    if not records:
        return {"rows_synced": 0, "status": "no deployment monitor logs found"}

    rows = []
    for run_id, experiment_id, record in records:
        if run_id not in run_ids:
            continue

        mapped = map_monitor_logs(record, experiment_id, run_id)
        mapping = (id_mapping or {}).get(run_id, {})
        if mapping:
            if mapping.get("deployment_id"):
                mapped["deployment_id"] = mapping.get("deployment_id")
            if mapping.get("model_id"):
                mapped["model_id"] = mapping.get("model_id")
            if mapping.get("deployment_id") and not mapped.get("deployment_id"):
                mapped["deployment_id"] = mapping.get("deployment_id")

        if not mapped.get("experiment_id"):
            mapped["experiment_id"] = experiment_id

        rows.append(mapped)

    if not rows:
        return {"rows_synced": 0, "status": "no matching deployment monitor logs"}

    insert_dataframe(pd.DataFrame(rows), "monitor_logs")
    return {"rows_synced": len(rows), "status": "success"}


def sync_model_deployed(run_ids: list, id_mapping: dict):
    """Ensure parent rows exist in `model_deployed` before inserting deployment-scoped rows.

    This prefers the structured deployment manifest written under
    ``deployment_logs/model_deployed.json``. If that artifact is not present,
    it falls back to deployment-scoped JSON artifacts and creates a minimal
    ``model_deployed`` row from the known deployment/model ids.
    """
    manifest_records = get_json_artifacts_data(
        folder_name=artifact_path("deployment_logs"), file_name="model_deployed.json"
    )
    folders = [
        artifact_path("model_packaging"),
        artifact_path("build_and_integration_testing"),
        artifact_path("standards"),
        artifact_path("interfaces"),
        artifact_path("decommissioning"),
    ]

    seen = set()
    rows = []

    def _build_row(experiment_id, run_id, record, use_manifest=False):
        mapping = id_mapping.get(run_id, {})
        deployment_id = mapping.get("deployment_id") or record.get("deployment_id")
        model_id = mapping.get("model_id") or record.get("model_id")

        if not deployment_id or not model_id:
            return None

        key = (experiment_id, deployment_id, model_id)
        if key in seen:
            return None
        seen.add(key)

        status = record.get("status") or (
            "deployed"
            if use_manifest or deployment_id != "not deployed yet"
            else "not available yet"
        )

        return {
            "experiment_id": experiment_id,
            "deployment_id": deployment_id,
            "model_id": model_id,
            "deployed_time": int(
                record.get("deployed_time") or pd.Timestamp.now(tz="UTC").timestamp()
            ),
            "model_version": record.get("model_version", ""),
            "endpoint": record.get("endpoint", ""),
            "model_format": record.get("model_format", ""),
            "size": record.get("size", ""),
            "description": record.get("description")
            or record.get("deployment_log", ""),
            "user_id": record.get("user_id", ""),
            "current_stage": record.get("current_stage", ""),
            "run_id": record.get("run_id") or run_id,
            "location": record.get("location", ""),
            "status": status,
            "model_cateory": record.get("model_cateory", ""),
        }

    for run_id, experiment_id, record in manifest_records:
        if run_id not in run_ids:
            continue
        row = _build_row(experiment_id, run_id, record, use_manifest=True)
        if row:
            rows.append(row)

    for folder in folders:
        records = get_json_artifacts_data(folder_name=folder)
        if not records:
            continue
        for run_id, experiment_id, record in records:
            if run_id not in run_ids:
                continue
            row = _build_row(experiment_id, run_id, record)
            if row:
                rows.append(row)

    if not rows:
        return {"rows_synced": 0, "status": "no deployment parent rows found"}

    mapped_df = pd.DataFrame(rows)
    insert_dataframe(mapped_df, "model_deployed")
    return {"rows_synced": len(rows), "status": "success"}


def sync_run_params_from_artifacts(run_ids: list, id_mapping: dict):
    """Read certain/metadata/run_params.json artifacts → model_hyperparameters.

    These are written by certain_library.metadata.artifact_metadata.save_params_as_artifact
    and contain the full parameter dict for a run, richer than the MLflow SQL params table.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    if not records:
        return {"rows_synced": 0, "status": "no metadata artifacts found"}

    rows = []
    for run_id, _exp_id, record in records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        # run_params.json recognized by presence of 'run_params' key
        if "run_params" not in record:
            continue
        rows.extend(map_run_params(record, run_id, id_mapping))

    if not rows:
        return {"rows_synced": 0, "status": "no run_params artifacts matched"}

    insert_dataframe(pd.DataFrame(rows), "model_hyperparameters")
    return {"rows_synced": len(rows), "status": "success"}


def sync_run_metrics_from_artifacts(run_ids: list, id_mapping: dict):
    """Read certain/metadata/run_metrics.json artifacts → model_metrics.

    These are written by certain_library.metadata.artifact_metadata.save_metrics_as_artifact
    and contain the complete per-step metric history.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    if not records:
        return {"rows_synced": 0, "status": "no metadata artifacts found"}

    rows = []
    for run_id, _exp_id, record in records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "run_metrics" not in record:
            continue
        rows.extend(map_run_metrics(record, run_id, id_mapping))

    if not rows:
        return {"rows_synced": 0, "status": "no run_metrics artifacts matched"}

    df = pd.DataFrame(rows)
    # Normalise nullable int columns before insert
    for col in ("step", "timestamp"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    insert_dataframe(df, "model_metrics")
    return {"rows_synced": len(rows), "status": "success"}


def sync_run_resources_from_artifacts(run_ids: list, id_mapping: dict):
    """Read certain/metadata/run_resources.json artifacts → resources.

    These are written by certain_library.metadata.artifact_metadata.save_resources_as_artifact
    and contain the full per-step system resource history (CPU, RAM, …).
    """
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    if not records:
        return {"rows_synced": 0, "status": "no metadata artifacts found"}

    rows = []
    for run_id, _exp_id, record in records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "run_resources" not in record:
            continue
        rows.extend(map_run_resources(record, run_id, id_mapping))

    if not rows:
        return {"rows_synced": 0, "status": "no run_resources artifacts matched"}

    df = pd.DataFrame(rows).fillna(0)
    insert_dataframe(df, "resources")
    return {"rows_synced": len(rows), "status": "success"}


def sync_run_inputs_from_artifacts(run_ids: list, id_mapping: dict):
    """Read certain/metadata/inputs.json (run_inputs) artifacts → data table.

    These are written by certain_library.metadata.artifact_metadata.save_inputs_as_artifact
    and contain the dataset lineage inputs logged via mlflow.log_input.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    if not records:
        return {"rows_synced": 0, "status": "no metadata artifacts found"}

    rows = []
    for run_id, _exp_id, record in records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "run_inputs" not in record:
            continue
        rows.extend(map_run_inputs(record, run_id, id_mapping))

    if not rows:
        return {"rows_synced": 0, "status": "no run_inputs artifacts matched"}

    insert_dataframe(pd.DataFrame(rows), "data")
    return {"rows_synced": len(rows), "status": "success"}


def sync_experiment_tags_from_artifacts(run_ids: list):
    """Read certain/metadata/experiment_tags.json artifacts → experiments_tags.

    These are written by certain_library.metadata.artifact_metadata.save_tags_as_artifact
    and contain experiment-level tags that may not be stored in the MLflow SQL backend.
    """
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    if not records:
        return {"rows_synced": 0, "status": "no metadata artifacts found"}

    rows = []
    for run_id, experiment_id, record in records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "experiment_tags" not in record:
            continue
        rows.extend(map_experiment_tags_artifact(record, experiment_id))

    if not rows:
        return {"rows_synced": 0, "status": "no experiment_tags artifacts matched"}

    insert_dataframe(pd.DataFrame(rows), "experiments_tags")
    return {"rows_synced": len(rows), "status": "success"}


def sync_dataset_manifest(run_ids: list, id_mapping: dict):
    """Read certain/dataset/data_manifest.json and certain/metadata/data.json → data table.

    These are written by certain_library.metadata.artifact_metadata.save_dataset_manifest
    and contain rich dataset size, location and checksum metadata, which is more reliable
    than reconstructing it from the MLflow datasets SQL table.
    """
    # We'll process manifests in two phases: first parents (root runs), then
    # children. This ensures we compute a canonical data_id for the parent and
    # reuse it for any sub-runs.
    manifest_records = get_json_artifacts_data(folder_name=artifact_path("dataset"))
    meta_records = get_json_artifacts_data(folder_name=artifact_path("metadata"))

    # Try to build a run -> parent map so we can identify children
    try:
        runs_df = get_runs_data()
    except Exception:
        runs_df = None

    parent_of = {}
    root_runs = set()
    if isinstance(runs_df, pd.DataFrame) and not runs_df.empty:
        id_col = (
            "run_uuid" if "run_uuid" in runs_df.columns else ("run_id" if "run_id" in runs_df.columns else None)
        )
        if id_col:
            for _, r in runs_df.iterrows():
                rid = str(r.get(id_col) or r.get("run_id"))
                parent = r.get("parent_id") if "parent_id" in r.index else r.get("parent_id")
                parent_of[rid] = str(parent) if parent else None
                if not parent:
                    root_runs.add(rid)

    # Helper to map a single manifest/meta record and ensure run_id present
    def _map_and_normalize(rec, run_id):
        mapped = map_dataset_manifest(rec, run_id, id_mapping) if isinstance(rec, dict) else {}
        if not isinstance(mapped, dict):
            mapped = dict(mapped) if mapped else {}
        mapped["run_id"] = str(run_id)
        return mapped

    # Collect parent rows first
    parent_rows = {}
    rows_out = []

    # Full manifests
    for run_id, _exp_id, record in manifest_records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "files" not in record and "data_location" not in record:
            continue
        # process parents first only
        if root_runs and str(run_id) not in root_runs:
            continue
        mapped = _map_and_normalize(record, run_id)
        # record mapping and output
        rows_out.append(mapped)
        parent_rows[str(run_id)] = mapped

    # Lightweight metadata
    seen_runs = {r["run_id"] for r in rows_out}
    for run_id, _exp_id, record in meta_records:
        if run_id not in run_ids:
            continue
        if run_id in seen_runs:
            continue
        if not isinstance(record, dict):
            continue
        if "data_location" not in record:
            continue
        if root_runs and str(run_id) not in root_runs:
            continue
        mapped = _map_and_normalize(record, run_id)
        rows_out.append(mapped)
        parent_rows[str(run_id)] = mapped

    # Build parent -> data_id lookup (prefer manifest-computed id, then id_mapping)
    parent_data_id = {}
    for prun, prow in parent_rows.items():
        did = prow.get("data_id") or (id_mapping.get(prun) or {}).get("data_id")
        if did:
            parent_data_id[prun] = did
    # Debug: show parent -> data_id mapping
    try:
        print(f"[sync_dataset_manifest] parent_data_id map: {parent_data_id}")
    except Exception:
        pass

    # Now process child runs: both full manifests and metadata entries
    # We'll iterate all manifest and meta records again but only for child runs
    for run_id, _exp_id, record in manifest_records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "files" not in record and "data_location" not in record:
            continue
        # skip parents (already processed)
        if root_runs and str(run_id) in root_runs:
            continue

        parent_id = parent_of.get(str(run_id))
        # Map the child record first so we can compare computed data_id
        mapped = _map_and_normalize(record, run_id)
        computed = mapped.get("data_id")
        # If we have parent data_id, override child's data_id to parent value
        if parent_id and str(parent_id) in parent_data_id:
            parent_did = parent_data_id[str(parent_id)]
            if computed != parent_did:
                try:
                    print(
                        f"[sync_dataset_manifest] overriding data_id for run {run_id}: computed={computed} -> parent={parent_did} (parent run {parent_id})"
                    )
                except Exception:
                    pass
            mapped["data_id"] = parent_did
        rows_out.append(mapped)

    for run_id, _exp_id, record in meta_records:
        if run_id not in run_ids:
            continue
        if not isinstance(record, dict):
            continue
        if "data_location" not in record:
            continue
        if root_runs and str(run_id) in root_runs:
            continue

        parent_id = parent_of.get(str(run_id))
        mapped = _map_and_normalize(record, run_id)
        computed = mapped.get("data_id")
        if parent_id and str(parent_id) in parent_data_id:
            parent_did = parent_data_id[str(parent_id)]
            if computed != parent_did:
                try:
                    print(
                        f"[sync_dataset_manifest] overriding (meta) data_id for run {run_id}: computed={computed} -> parent={parent_did} (parent run {parent_id})"
                    )
                except Exception:
                    pass
            mapped["data_id"] = parent_did
        rows_out.append(mapped)

    if not rows_out:
        return {"rows_synced": 0, "status": "no dataset manifest artifacts found"}

    insert_dataframe(pd.DataFrame(rows_out), "data")
    return {"rows_synced": len(rows_out), "status": "success"}


def sync_id_mapping(run_ids: list):
    from sqlalchemy import MetaData, Table, select
    from app.target_connector import target_engine

    metadata = MetaData()
    table = Table("id_mapping", metadata, autoload_with=target_engine)

    # Normalize run_ids: support list[str] or list[dict] produced by sync_runs()
    normalized_run_ids = []
    for r in run_ids or []:
        if isinstance(r, str):
            normalized_run_ids.append(r)
        elif isinstance(r, dict):
            # Accept shapes like {'run_id': '...', 'parent': True}
            v = r.get("run_id") or r.get("run_uuid") or r.get("run")
            if v:
                normalized_run_ids.append(str(v))

    # If nothing to query, return empty mapping immediately
    if not normalized_run_ids:
        return {}

    with target_engine.connect() as conn:
        query = select(table).where(table.c.run_id.in_(normalized_run_ids))
        result = conn.execute(query).fetchall()

    existing_mapping = {
        row._mapping["run_id"]: {
            "model_id": row._mapping["model_id"],
            "data_id": row._mapping["data_id"],
            "deployment_id": row._mapping["deployment_id"],
        }
        for row in result
    }

    new_rows = []
    mapping_dict = {}

    for run_id in normalized_run_ids:
        # run_id is a plain string here
        if run_id in existing_mapping:
            mapping_dict[run_id] = existing_mapping[run_id]
        else:
            new_entry = {
                "run_id": run_id,
                "model_id": str(uuid.uuid4()),
                "data_id": str(uuid.uuid4()),
                "deployment_id": str(uuid.uuid4()),
            }

            new_rows.append(new_entry)

            mapping_dict[run_id] = {
                "model_id": new_entry["model_id"],
                "data_id": new_entry["data_id"],
                "deployment_id": new_entry["deployment_id"],
            }

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        insert_dataframe(df_new, "id_mapping")

    return mapping_dict


@app.post("/sync/all")
def sync_all():
    captured = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured

    try:
        sync_experiments()
        sync_experiment_tags()

        run_ids = sync_runs()
        id_mapping = sync_id_mapping(run_ids)

        sync_tags()
        sync_model_architecture(id_mapping)
        sync_metrics(id_mapping)
        sync_latest_metrics(id_mapping)
        sync_model_params(id_mapping)
        sync_model_resources(id_mapping)

        # Sync richer metadata from certain/metadata artifacts
        # (params, metrics, resources, inputs written by certain_library)
        sync_run_params_from_artifacts(run_ids, id_mapping)
        sync_run_metrics_from_artifacts(run_ids, id_mapping)
        sync_run_resources_from_artifacts(run_ids, id_mapping)
        sync_run_inputs_from_artifacts(run_ids, id_mapping)
        sync_experiment_tags_from_artifacts(run_ids)

        sync_data(id_mapping)
        # Sync dataset manifest artifacts (certain/dataset and certain/metadata/data.json)
        # These have richer location/size info than the MLflow datasets SQL table
        sync_dataset_manifest(run_ids, id_mapping)
        sync_data_signatures(id_mapping)
        sync_data_metrics(id_mapping)
        sync_data_resources(id_mapping)
        sync_data_drift(id_mapping)

        # Sync drift metrics artifacts into the database
        sync_drift_metrics(run_ids, id_mapping)

        # Sync data_techniques artifacts into the data_techniques table
        sync_data_techniques(run_ids, id_mapping)

        sync_time_series_data(id_mapping)
        sync_data_duration_leakage(id_mapping)

        sync_run_code(run_ids)
        sync_run_logs(run_ids)
        sync_checkpoints(id_mapping)
        sync_weight_distribution(id_mapping)
        sync_examples(id_mapping)
        sync_tokenizer_config(run_ids)
        sync_tokenization_stats(run_ids)

        sync_ai_actors(run_ids, id_mapping)
        sync_labeling_procedures(run_ids, id_mapping)
        sync_risks(run_ids, id_mapping)
        sync_human_oversight(run_ids, id_mapping)
        sync_transparency_measures(run_ids, id_mapping)

        sync_change_logs(run_ids)
        sync_declaration_of_conformity(run_ids, id_mapping)
        sync_visual_documentation(run_ids)
        sync_explainable_ai(run_ids, id_mapping)

        # Ensure model_deployed parent rows exist before deployment-level tables
        sync_model_deployed(run_ids, id_mapping)

        sync_monitor_logs(run_ids, id_mapping)

        sync_model_packaging(run_ids, id_mapping)
        sync_build_testing(run_ids, id_mapping)
        sync_standards(run_ids, id_mapping)
        sync_interfaces(run_ids, id_mapping)
        sync_decommissioning(run_ids, id_mapping)

        sync_runtime_environment_from_artifacts(id_mapping)

    finally:
        sys.stdout = original_stdout

    output = captured.getvalue()
    warnings = [line for line in output.splitlines() if line.strip()]

    return {
        "status": "all data synced successfully",
        "warnings": warnings if warnings else [],
    }


@app.get("/all/data")
def all_data():
    metrics_df = get_metrics_data()
    runs_df = get_runs_data()
    experiments_df = get_experiments_data()
    experiment_tags_df = get_experiment_tags_data()
    datasets_df = get_datasets_data()
    latest_metrics_df = get_latest_metrics_data()
    params_df = get_params_data()
    tags_df = get_tags_data()

    whylogs_df = get_artifacts_data(
        folder_name=artifact_path("whylogs"),
        file_extension=".csv",
    )

    print("Metrics DataFrame:")
    print(metrics_df)
    print("Runs DataFrame:")
    print(runs_df)
    print("Experiments DataFrame:")
    print(experiments_df)
    print("Experiment Tags DataFrame:")
    print(experiment_tags_df)
    print("Datasets DataFrame:")
    print(datasets_df)
    print("Latest Metrics DataFrame:")
    print(latest_metrics_df)
    print("Params DataFrame:")
    print(params_df)
    print("Tags DataFrame:")
    print(tags_df)
    print("Whylogs DataFrame:")
    print(whylogs_df)

    return True
