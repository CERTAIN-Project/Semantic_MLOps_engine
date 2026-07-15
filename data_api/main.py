import uuid
import io
import sys
import json
import pandas as pd
import inspect
import yaml
from fastapi import FastAPI
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
    map_mlflow_runs,
    map_mlflow_experiments,
    map_mlflow_resources,
    map_mlflow_time_series_data,
    map_mlflow_data_drift,
    map_mlflow_data_duration_leakage,
    map_mlflow_datasets,
    map_mlflow_data_metrics,
    map_mlflow_model_metrics,
    map_mlflow_model_params,
    map_mlflow_runs_tags,
    map_mlflow_data_resources,
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
    map_tokenizer_config,
    map_tokenization_stats,
    map_mlflow_data_techniques,
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

    mapped_rows = df.apply(
        lambda row: map_mlflow_model_metrics(row, id_mapping), axis=1
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
        lambda row: map_mlflow_model_metrics(row, id_mapping), axis=1
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
    # (parent_id, run_id, model_id, key) pair — keeping only the latest record.
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

    # Group by parent/run/model/key and keep the row with max(timestamp, step)
    # Define a sort key: prefer timestamp, then step
    mapped_df = mapped_df.sort_values(by=["timestamp", "step"], ascending=[True, True])
    group_cols = ["parent_id", "run_id", "model_id", "key"]
    # drop duplicates keeping the last (highest timestamp/step)
    deduped = mapped_df.drop_duplicates(subset=group_cols, keep="last").copy()

    # Remove parent_id (not part of last_model_metrics schema) before insert
    if "parent_id" in deduped.columns:
        deduped = deduped.drop(columns=["parent_id"])

    insert_dataframe(deduped, "last_model_metrics")
    return {"rows_synced": len(deduped), "status": "success"}


def sync_runs():
    df = get_runs_data()

    mapped_rows = df.apply(map_mlflow_runs, axis=1)

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

    insert_dataframe(mapped_df, "runs")
    # return list of run_ids inserted
    try:
        return mapped_df["run_id"].tolist()
    except Exception:
        return []


def sync_experiments():
    df = get_experiments_data()

    mapped_rows = df.apply(map_mlflow_experiments, axis=1)

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
    # Prefer artifact-stored experiment tags when available
    records = get_json_artifacts_data(folder_name=artifact_path("metadata"))
    # records: list of (run_id, experiment_id, record)
    exp_tag_rows = []
    for _run_id, exp_id, rec in records:
        if not isinstance(rec, dict):
            continue
        # Recognize experiment_tags by keys
        if rec.get("experiment_id") and rec.get("tags"):
            for k, v in rec.get("tags", {}).items():
                exp_tag_rows.append(
                    {
                        "experiment_id": str(rec.get("experiment_id")),
                        "key": k,
                        "value": v,
                    }
                )

    if exp_tag_rows:
        import pandas as _pd

        df = _pd.DataFrame(exp_tag_rows)
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
                        "server_name": server_name,
                        "performance": performance,
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
                    continue
                parent_rows.append(
                    {
                        "experiment_id": exp_id,
                        "deployment_id": mapping.get("deployment_id"),
                        "model_id": mapping.get("model_id"),
                        "deployed_time": now_ts,
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

    # Only sync dataset rows for root runs (those without a parent run).
    # This avoids creating duplicate data_id rows for child runs which are
    # typically derivative (preprocessing / nested) runs.
    try:
        runs_df = get_runs_data()
    except Exception:
        runs_df = None

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
                rid = r.get(id_col) or r.get("run_id")
                parent = (
                    r.get("parent_id") if "parent_id" in r.index else r.get("parent_id")
                )
                # treat empty strings / NaN as no parent
                if not parent:
                    root_run_ids.add(str(rid))

    mapped_rows = []
    for run_id in id_mapping.keys():
        # skip child runs (those with a parent_id)
        if root_run_ids and str(run_id) not in root_run_ids:
            continue

        row_data = (
            df[df["run_id"] == run_id].iloc[0]
            if not df.empty and "run_id" in df.columns and run_id in df["run_id"].values
            else {}
        )

        mapped_row = map_mlflow_datasets(row_data, run_id, id_mapping)
        mapped_rows.append(mapped_row)

    if not mapped_rows:
        return {"rows_synced": 0, "status": "no matching data rows to sync"}

    mapped_df = pd.DataFrame(mapped_rows)

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

    mapped_rows = df.apply(lambda row: map_mlflow_data_metrics(row, id_mapping), axis=1)
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
        lambda row: map_mlflow_data_resources(row, id_mapping), axis=1
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

    mapped_df = map_mlflow_data_drift(df, id_mapping)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_data_duration_leakage(id_mapping):
    df = get_artifacts_data(
        folder_name=artifact_path("dataset"),
        file_extension=".csv",
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    mapped_df = map_mlflow_data_duration_leakage(df, id_mapping)

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

    for run_id in id_mapping.keys():
        params = run_params.get(run_id, {})

        optimizer_name = params.get("optimizer", "unknown")
        optimizer_dict = {"name": optimizer_name}

        for key, value in params.items():
            if key.startswith("optimizer."):
                hyperparam_name = key[len("optimizer.") :]
                optimizer_dict[hyperparam_name] = value

        mapped_rows.append(
            {
                "run_id": run_id,
                "model_id": id_mapping[run_id]["model_id"],
                "architecture_name": "Simple Model",
                "model_version": 1,
                "layer_structure": (
                    layer_structure[run_id] if run_id in layer_structure.keys() else {},
                ),
                "activation_function": params.get("activation_function", "ReLU"),
                "optimizer": json.dumps(optimizer_dict),
                "loss_function": params.get("losses", "MSE"),
                "framework": params.get("framework", "unknown"),
                "metrics": [],
                "input_shape": params.get("input_shape", ""),
                "output_shape": params.get("output_shape", ""),
                "number_of_layers": 0,
                "number_of_total_parameters": 0,
                "number_of_trainable_parameters": 0,
                "number_of_non_trainable_parameters": 0,
                "creation_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
            }
        )

    mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "model_architecture")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_model_params(id_mapping):
    df = get_params_data()

    mapped_rows = df.apply(lambda row: map_mlflow_model_params(row, id_mapping), axis=1)
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

    mapped_rows = df.apply(lambda row: map_mlflow_resources(row, id_mapping), axis=1)

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
        lambda row: map_mlflow_time_series_data(row, id_mapping), axis=1
    )

    expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]
    mapped_df = pd.DataFrame(expand_mapped_rows)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


def sync_tags():
    df = get_tags_data()

    # map_mlflow_runs_tags expects a single row mapping; if get_tags_data
    # returned a DataFrame we should apply the mapper per-row. If it returned
    # a single record or dict-like, handle accordingly.
    try:
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return {"rows_synced": 0, "status": "no tags found"}
            mapped = df.apply(map_mlflow_runs_tags, axis=1)
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
    """Populate examples from artifacts/certain/examples/*.csv."""
    try:
        df = get_artifacts_data(
            folder_name=artifact_path("examples"),
            file_extension=".csv",
        )
    except Exception:
        return {"rows_synced": 0, "status": "no examples artifacts found"}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows_synced": 0, "status": "no examples artifacts found"}

    mapped_rows = df.apply(lambda row: map_examples(row, id_mapping), axis=1)

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

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows_synced": 0, "status": "no run_logs artifacts found"}

    mapped_rows = []

    for _, row in df.iterrows():
        run_id = row.get("run_id", "")
        if run_id in run_ids:
            mapped_rows.append(map_run_logs(row, run_id))

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

        mapped = map_mlflow_data_techniques(record, run_id)

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
    folder_name: str, map_fn, table_name: str, run_ids: list
):
    """Generic sync for experiment-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=artifact_path(folder_name))

    if not records:
        return {"rows_synced": 0, "status": f"no {folder_name} artifacts found"}

    rows = []

    for run_id, experiment_id, record in records:
        if run_id not in run_ids:
            continue

        # Accept either a single object per-file or a list of objects
        if isinstance(record, list):
            for entry in record:
                rows.append(map_fn(entry, experiment_id))
        else:
            rows.append(map_fn(record, experiment_id))

    if not rows:
        return {"rows_synced": 0, "status": f"no matching {folder_name} rows"}

    insert_dataframe(pd.DataFrame(rows), table_name)

    return {"rows_synced": len(rows), "status": "success"}


def _sync_json_run_table(folder_name: str, map_fn, table_name: str, run_ids: list):
    """Generic sync for run-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=artifact_path(folder_name))

    if not records:
        return {"rows_synced": 0, "status": f"no {folder_name} artifacts found"}

    rows = []

    for run_id, _experiment_id, record in records:
        if run_id in run_ids:
            rows.append(map_fn(record, run_id))

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
                # e.g. map_model_packaging(record, experiment_id, id_mapping)
                mapped = map_fn(record, experiment_id, id_mapping)
            elif len(required_positional) == 2:
                # e.g. map_build_testing(record, experiment_id)
                mapped = map_fn(record, experiment_id)
            elif len(required_positional) == 1:
                mapped = map_fn(record)
            else:
                mapped = map_fn()
            rows.append(mapped)
    if not rows:
        return {"rows_synced": 0, "status": f"no matching {folder_name} rows"}
    insert_dataframe(pd.DataFrame(rows), table_name)
    return {"rows_synced": len(rows), "status": "success"}


def sync_ai_actors(run_ids: list, id_mapping: dict):  # noqa: ARG001
    # ai_actors are now run-scoped (referencing runs.run_id). Use the
    # run-scoped generic sync helper so the mapper receives run_id.
    return _sync_json_run_table("ai_actors", map_ai_actors, "ai_actors", run_ids)


def sync_labeling_procedures(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table(
        "labeling_procedures", map_labeling_procedures, "labeling_procedures", run_ids
    )


def sync_risks(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table("risks", map_risk, "risks", run_ids)


def sync_human_oversight(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table(
        "human_oversight", map_human_oversight, "human_oversight_mechanisms", run_ids
    )


def sync_transparency_measures(run_ids: list, id_mapping: dict):  # noqa: ARG001
    return _sync_json_experiment_table(
        "transparency_measures",
        map_transparency_measure,
        "transparency_measures",
        run_ids,
    )


def sync_change_logs(run_ids: list):
    return _sync_json_run_table("change_logs", map_change_log, "change_logs", run_ids)


def sync_declaration_of_conformity(run_ids: list):
    return _sync_json_run_table(
        "declaration_of_conformity",
        map_declaration_of_conformity,
        "declaration_of_conformity",
        run_ids,
    )


def sync_visual_documentation(run_ids: list):
    return _sync_json_run_table(
        "visual_documentation",
        map_visual_documentation,
        "visual_documentation",
        run_ids,
    )


def sync_explainable_ai(run_ids: list):
    return _sync_json_run_table(
        "explainable_ai", map_explainable_ai, "explainable_ai_features", run_ids
    )


def sync_model_packaging(run_ids: list, id_mapping: dict):
    return _sync_json_deployment_table(
        "model_packaging", map_model_packaging, "model_packaging", run_ids, id_mapping
    )


def sync_build_testing(run_ids: list, id_mapping: dict):
    # Only sync build/testing artifacts if model_deployed has parent rows.
    if not _model_deployed_has_rows():
        return {"rows_synced": 0, "status": "skipped - model_deployed empty"}

    # If the model_deployed parent table is empty, skip syncing deployment-scoped
    # build/testing artifacts to avoid FK violations. The model_deployed table
    # should be populated at deployment time (see sync_model_deployed).
    if not model_deployed_has_rows():
        return {"rows_synced": 0, "status": "skipped: no model_deployed parents"}

    return _sync_json_deployment_table(
        "build_and_integration_testing",
        map_build_testing,
        "build_and_integration_testing",
        run_ids,
        id_mapping,
    )


def sync_standards(run_ids: list, id_mapping: dict):
    # Only sync standards artifacts if model_deployed has parent rows.
    if not _model_deployed_has_rows():
        return {"rows_synced": 0, "status": "skipped - model_deployed empty"}

    if not model_deployed_has_rows():
        return {"rows_synced": 0, "status": "skipped: no model_deployed parents"}

    return _sync_json_deployment_table(
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
        "interfaces", map_interface, "interfaces", run_ids, id_mapping
    )


def sync_decommissioning(run_ids: list, id_mapping: dict):
    return _sync_json_deployment_table(
        "decommissioning", map_decommissioning, "decomissioning", run_ids, id_mapping
    )


def sync_model_deployed(run_ids: list, id_mapping: dict):
    """Ensure parent rows exist in `model_deployed` before inserting deployment-scoped rows.

    This scans deployment-scoped JSON artifact folders for records that contain
    `deployment_id` and `model_id` and upserts a minimal `model_deployed` row for
    each unique (experiment_id, deployment_id, model_id) triple found for the
    provided run_ids. The goal is to avoid FK violations when inserting dependent
    tables such as `model_packaging`.
    """
    folders = [
        "model_packaging",
        "build_and_integration_testing",
        "standards",
        "interfaces",
        "decommissioning",
    ]

    seen = set()
    rows = []
    for folder in folders:
        records = get_json_artifacts_data(folder_name=folder)
        if not records:
            continue
        for run_id, experiment_id, record in records:
            if run_id not in run_ids:
                continue

            deployment_id = record.get("deployment_id") or id_mapping.get(
                run_id, {}
            ).get("deployment_id")
            model_id = record.get("model_id") or id_mapping.get(run_id, {}).get(
                "model_id"
            )

            if not deployment_id or not model_id:
                # Skip incomplete records
                continue

            key = (experiment_id, deployment_id, model_id)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "experiment_id": experiment_id,
                    "deployment_id": deployment_id,
                    "model_id": model_id,
                    # minimal required fields; other columns can be NULL
                    "deployed_time": int(pd.Timestamp.now(tz="UTC").timestamp()),
                    "run_id": run_id,
                }
            )

    if not rows:
        return {"rows_synced": 0, "status": "no deployment parent rows found"}

    mapped_df = pd.DataFrame(rows)
    insert_dataframe(mapped_df, "model_deployed")
    return {"rows_synced": len(rows), "status": "success"}


def sync_id_mapping(run_ids: list):
    from sqlalchemy import MetaData, Table, select
    from app.target_connector import target_engine

    metadata = MetaData()
    table = Table("id_mapping", metadata, autoload_with=target_engine)

    with target_engine.connect() as conn:
        query = select(table).where(table.c.run_id.in_(run_ids))
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

    for run_id in run_ids:
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

        sync_data(id_mapping)
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
        sync_declaration_of_conformity(run_ids)
        sync_visual_documentation(run_ids)
        sync_explainable_ai(run_ids)

        # Ensure model_deployed parent rows exist before deployment-level tables
        sync_model_deployed(run_ids, id_mapping)

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
