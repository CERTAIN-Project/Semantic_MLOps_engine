import uuid
import io
import sys
import time
import json
import inspect
import pandas as pd
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
    # compliance maps
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
)

app = FastAPI()


@app.get("/")
def root():
    return {"status": "up"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# @app.post("/sync/metrics")
def sync_metrics(id_mapping):
    df = get_metrics_data()

    mapped_rows = df.apply(
        lambda row: map_mlflow_model_metrics(row, id_mapping), axis=1
    )
    mapped_df = pd.DataFrame(mapped_rows.values.tolist())

    insert_dataframe(mapped_df, "model_metrics")
    return {"rows_synced": len(df), "status": "success"}


# @app.post("/sync/latest_metrics")
def sync_latest_metrics(id_mapping):
    df = get_latest_metrics_data()

    mapped_rows = df.apply(
        lambda row: map_mlflow_model_metrics(row, id_mapping), axis=1
    )
    mapped_df = pd.DataFrame(mapped_rows.values.tolist())

    insert_dataframe(mapped_df, "last_model_metrics")
    return {"rows_synced": len(df), "status": "success"}


# @app.post("/sync/runs")
def sync_runs():
    df = get_runs_data()

    # Map rows into a list of dictionaries, then convert back to DataFrame
    mapped_rows = df.apply(map_mlflow_runs, axis=1).tolist()
    mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "runs")
    return mapped_df["run_id"].tolist()


# @app.post("/sync/experiments")
def sync_experiments():
    df = get_experiments_data()

    mapped_rows = df.apply(map_mlflow_experiments, axis=1).tolist()
    mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "experiments")
    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/experiment_tags")
def sync_experiment_tags():
    df = get_experiment_tags_data()
    insert_dataframe(df, "experiments_tags")
    return {"rows_synced": len(df), "status": "success"}


# @app.post("/sync/data")
def sync_data(id_mapping):
    df = get_datasets_data()

    mapped_rows = []

    # Loop through all runs in the id_mapping
    for run_id in id_mapping.keys():
        # Get the row from df for this run_id if it exists, otherwise use defaults
        row_data = (
            df[df["run_id"] == run_id].iloc[0]
            if not df.empty and "run_id" in df.columns and run_id in df["run_id"].values
            else {}
        )

        # Map the data with the correct run_id
        mapped_row = map_mlflow_datasets(row_data, run_id, id_mapping)
        mapped_rows.append(mapped_row)

    mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "data")

    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/data_signatures")
def sync_data_signatures(id_mapping):
    import json

    df = get_artifacts_data(folder_name="model", file_extension="MLmodel")

    # Convert signature data to a proper DataFrame format
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Convert the 'signature.inputs' column from JSON strings to dictionaries
    if "signature.inputs" in df.columns:
        try:
            df["signature.inputs"] = df["signature.inputs"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        except Exception as e:
            print(f"Error parsing JSON from signature.inputs: {e}")

    # Extract just the inputs and run_id columns from the DataFrame
    mapped_df = df[["signature.inputs", "run_id"]]

    # Rename the column 'inputs' to 'signature'
    mapped_df = mapped_df.rename(columns={"signature.inputs": "signature"})

    # Add 'data_id' column based on run_id matching in id_mapping
    mapped_df["data_id"] = mapped_df["run_id"].apply(
        lambda run_id: id_mapping.get(run_id, {}).get("data_id", None)
    )

    insert_dataframe(mapped_df, "data_signatures")

    return {"rows_synced": len(mapped_df), "status": "success"}


@app.post("/sync/data_metrics")
def sync_data_metrics(id_mapping):
    # Whylogs Data
    df = get_artifacts_data(folder_name="whylogs", file_extension=".csv")

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")
    mapped_rows = df.apply(lambda row: map_mlflow_data_metrics(row, id_mapping), axis=1)
    expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]
    mapped_df = pd.DataFrame(expand_mapped_rows)

    bulk_upsert_metrics(target_engine, mapped_df, chunk_size=500)

    return {"rows_synced": len(mapped_df), "status": "success"}


@app.post("/sync/data_resources")
def sync_data_resources(id_mapping):
    # Whylogs Data
    df = get_artifacts_data(
        folder_name="code_carbon", file_extension="emissions_data.csv"
    )

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")
    mapped_rows = df.apply(
        lambda row: map_mlflow_data_resources(row, id_mapping), axis=1
    )
    # Check if mapped_rows contains lists of dictionaries
    if len(mapped_rows) > 0 and isinstance(mapped_rows.iloc[0], list):
        expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]
    else:
        expand_mapped_rows = mapped_rows.tolist()

    mapped_df = pd.DataFrame(expand_mapped_rows)
    mapped_df = mapped_df.fillna(0)

    insert_dataframe(mapped_df, "data_resources")

    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/data_drift")
def sync_data_drift(id_mapping):
    # Whylogs Data
    df = get_artifacts_data(folder_name="dataset", file_extension=".csv")

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    # Apply the map_mlflow_data_drift function to each row
    mapped_df = map_mlflow_data_drift(df, id_mapping)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/data_duration_leakage")
def sync_data_duration_leakage(id_mapping):
    # Whylogs Data
    df = get_artifacts_data(folder_name="dataset", file_extension=".csv")

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    # Apply the map_mlflow_data_duration_leakage function to each row
    mapped_df = map_mlflow_data_duration_leakage(df, id_mapping)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/model_architecture")
def sync_model_architecture(id_mapping):
    layer_structure = get_artifacts_data(folder_name="model", file_extension=".pkl")

    # Build a per-run lookup of all MLflow params so we can read optimizer data
    params_df = get_params_data()
    # { run_uuid: { key: value, ... } }
    run_params: dict = (
        params_df.groupby("run_uuid")
        .apply(lambda g: dict(zip(g["key"], g["value"])))
        .to_dict()
        if not params_df.empty
        else {}
    )

    mapped_rows = []
    for run_id in id_mapping.keys():
        params = run_params.get(run_id, {})

        # Reconstruct the optimizer dict from logged params:
        # log_model_architecture logs "optimizer" (name) and "optimizer.<key>" for hyperparams
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
                # model_architecture table has several non-nullable columns added by migrations
                # Provide safe defaults so upserts don't fail when source lacks data.
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


# @app.post("/sync/params")
def sync_model_params(id_mapping):
    df = get_params_data()

    mapped_rows = df.apply(lambda row: map_mlflow_model_params(row, id_mapping), axis=1)
    mapped_df = pd.DataFrame(mapped_rows.values.tolist())

    insert_dataframe(mapped_df, "model_hyperparameters")
    return {"rows_synced": len(df), "status": "success"}


# @app.pist("/sync/model_resources")
def sync_model_resources(id_mapping):
    df = get_artifacts_data(
        folder_name="code_carbon", file_extension="emissions_train.csv"
    )

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")
    mapped_rows = df.apply(lambda row: map_mlflow_resources(row, id_mapping), axis=1)
    # Check if mapped_rows contains lists of dictionaries
    if len(mapped_rows) > 0 and isinstance(mapped_rows.iloc[0], list):
        expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]
    else:
        expand_mapped_rows = mapped_rows.tolist()

    mapped_df = pd.DataFrame(expand_mapped_rows)
    mapped_df = mapped_df.fillna(0)

    insert_dataframe(mapped_df, "resources")

    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/time_series_data")
def sync_time_series_data(id_mapping):
    df = get_artifacts_data(folder_name="timestamps", file_extension=".txt")

    # Check if the df is DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The data is not in the expected DataFrame format.")

    # Apply the map_mlflow_time_series_data function to each row
    mapped_rows = df.apply(
        lambda row: map_mlflow_time_series_data(row, id_mapping), axis=1
    )

    # Flatten the list of lists into a single list
    expand_mapped_rows = [item for sublist in mapped_rows for item in sublist]

    # Convert the expanded rows into a DataFrame
    mapped_df = pd.DataFrame(expand_mapped_rows)
    # mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "data_metrics")

    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/tags")
def sync_tags():
    df = get_tags_data()

    mapped_rows = map_mlflow_runs_tags(df)
    mapped_df = pd.DataFrame(mapped_rows)

    insert_dataframe(mapped_df, "runs_tags")
    return {"rows_synced": len(df), "status": "success"}


# ---------------------------------------------------------------------------
# New sync functions — previously missing certain_db tables
# ---------------------------------------------------------------------------


# @app.post("/sync/run_code")
def sync_run_code(run_ids: list):
    """Populate runs_code from MLflow system tags (git commit hash + source name)."""
    tags_df = get_tags_data()

    rows = []
    for run_id in run_ids:
        row = map_run_code(tags_df, run_id)
        if row:
            rows.append(row)

    if not rows:
        return {"rows_synced": 0, "status": "no git tags found"}

    mapped_df = pd.DataFrame(rows)
    insert_dataframe(mapped_df, "runs_code")
    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/checkpoints")
def sync_checkpoints(id_mapping: dict):
    """Populate checkpoints from checkpoints/*.csv artifacts."""
    try:
        df = get_artifacts_data(folder_name="checkpoints", file_extension=".csv")
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


# @app.post("/sync/weight_distribution")
def sync_weight_distribution(id_mapping: dict):
    """Populate weight_distribution from weight_distribution/*.csv artifacts."""
    try:
        df = get_artifacts_data(
            folder_name="weight_distribution", file_extension=".csv"
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


# @app.post("/sync/examples")
def sync_examples(id_mapping: dict):
    """Populate examples from examples/*.csv artifacts."""
    try:
        df = get_artifacts_data(folder_name="examples", file_extension=".csv")
    except Exception:
        return {"rows_synced": 0, "status": "no examples artifacts found"}

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"rows_synced": 0, "status": "no examples artifacts found"}

    mapped_rows = df.apply(lambda row: map_examples(row, id_mapping), axis=1).tolist()
    mapped_df = pd.DataFrame(mapped_rows)
    insert_dataframe(mapped_df, "examples")
    return {"rows_synced": len(mapped_df), "status": "success"}


# @app.post("/sync/run_logs")
def sync_run_logs(run_ids: list):
    """Populate runs_logs from run_logs/*.csv artifacts."""
    try:
        df = get_artifacts_data(folder_name="run_logs", file_extension=".csv")
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
    """Populate tokenizer_config from tokenizer_config/*.json artifacts."""
    return _sync_json_run_table(
        "tokenizer_config", map_tokenizer_config, "tokenizer_config", run_ids
    )


def sync_tokenization_stats(run_ids: list):
    """Populate tokenization_stats from tokenization_stats/*.json artifacts."""
    return _sync_json_run_table(
        "tokenization_stats", map_tokenization_stats, "tokenization_stats", run_ids
    )


# ---------------------------------------------------------------------------
# Compliance / governance sync helpers (JSON artifacts)
# ---------------------------------------------------------------------------


def _sync_json_experiment_table(
    folder_name: str, map_fn, table_name: str, run_ids: list
):
    """Generic sync for experiment-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=folder_name)
    if not records:
        return {"rows_synced": 0, "status": f"no {folder_name} artifacts found"}
    rows = []
    for run_id, experiment_id, record in records:
        if run_id in run_ids:
            rows.append(map_fn(record, experiment_id))
    if not rows:
        return {"rows_synced": 0, "status": f"no matching {folder_name} rows"}
    insert_dataframe(pd.DataFrame(rows), table_name)
    return {"rows_synced": len(rows), "status": "success"}


def _sync_json_run_table(folder_name: str, map_fn, table_name: str, run_ids: list):
    """Generic sync for run-scoped JSON artifact tables."""
    records = get_json_artifacts_data(folder_name=folder_name)
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


# -- Experiment-scoped -------------------------------------------------------


def sync_ai_actors(run_ids: list, id_mapping: dict):  # noqa: ARG001
    """Populate ai_actors from ai_actors/*.json artifacts."""
    return _sync_json_experiment_table("ai_actors", map_ai_actors, "ai_actors", run_ids)


def sync_labeling_procedures(run_ids: list, id_mapping: dict):  # noqa: ARG001
    """Populate labeling_procedures from labeling_procedures/*.json artifacts."""
    return _sync_json_experiment_table(
        "labeling_procedures", map_labeling_procedures, "labeling_procedures", run_ids
    )


def sync_risks(run_ids: list, id_mapping: dict):  # noqa: ARG001
    """Populate risks from risks/*.json artifacts."""
    return _sync_json_experiment_table("risks", map_risk, "risks", run_ids)


def sync_human_oversight(run_ids: list, id_mapping: dict):  # noqa: ARG001
    """Populate human_oversight_mechanisms from human_oversight/*.json artifacts."""
    return _sync_json_experiment_table(
        "human_oversight", map_human_oversight, "human_oversight_mechanisms", run_ids
    )


def sync_transparency_measures(run_ids: list, id_mapping: dict):  # noqa: ARG001
    """Populate transparency_measures from transparency_measures/*.json artifacts."""
    return _sync_json_experiment_table(
        "transparency_measures",
        map_transparency_measure,
        "transparency_measures",
        run_ids,
    )


# -- Run-scoped --------------------------------------------------------------


def sync_change_logs(run_ids: list):
    """Populate change_logs from change_logs/*.json artifacts."""
    return _sync_json_run_table("change_logs", map_change_log, "change_logs", run_ids)


def sync_declaration_of_conformity(run_ids: list):
    """Populate declaration_of_conformity from declaration_of_conformity/*.json artifacts."""
    return _sync_json_run_table(
        "declaration_of_conformity",
        map_declaration_of_conformity,
        "declaration_of_conformity",
        run_ids,
    )


def sync_visual_documentation(run_ids: list):
    """Populate visual_documentation from visual_documentation/*.json artifacts."""
    return _sync_json_run_table(
        "visual_documentation",
        map_visual_documentation,
        "visual_documentation",
        run_ids,
    )


def sync_explainable_ai(run_ids: list):
    """Populate explainable_ai_features from explainable_ai/*.json artifacts."""
    return _sync_json_run_table(
        "explainable_ai", map_explainable_ai, "explainable_ai_features", run_ids
    )


# -- Deployment-scoped -------------------------------------------------------


def sync_model_packaging(run_ids: list, id_mapping: dict):
    """Populate model_packaging from model_packaging/*.json artifacts."""
    return _sync_json_deployment_table(
        "model_packaging", map_model_packaging, "model_packaging", run_ids, id_mapping
    )


def sync_build_testing(run_ids: list, id_mapping: dict):
    """Populate build_and_integration_testing from build_and_integration_testing/*.json artifacts."""
    return _sync_json_deployment_table(
        "build_and_integration_testing",
        map_build_testing,
        "build_and_integration_testing",
        run_ids,
        id_mapping,
    )


def sync_standards(run_ids: list, id_mapping: dict):
    """Populate standards from standards/*.json artifacts."""
    return _sync_json_deployment_table(
        "standards", map_standard, "standards", run_ids, id_mapping
    )


def sync_interfaces(run_ids: list, id_mapping: dict):
    """Populate interfaces from interfaces/*.json artifacts."""
    return _sync_json_deployment_table(
        "interfaces", map_interface, "interfaces", run_ids, id_mapping
    )


def sync_decommissioning(run_ids: list, id_mapping: dict):
    """Populate decomissioning from decommissioning/*.json artifacts."""
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
    # Capture all print() output (warnings, etc.) to include in the response
    captured = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured

    try:
        # Sync all data
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

        sync_time_series_data(id_mapping)
        sync_data_duration_leakage(id_mapping)

        # New tables
        sync_run_code(run_ids)
        sync_run_logs(run_ids)
        sync_checkpoints(id_mapping)
        sync_weight_distribution(id_mapping)
        sync_examples(id_mapping)
        sync_tokenizer_config(run_ids)
        sync_tokenization_stats(run_ids)

        # Compliance tables (experiment-level)
        sync_ai_actors(run_ids, id_mapping)
        sync_labeling_procedures(run_ids, id_mapping)
        sync_risks(run_ids, id_mapping)
        sync_human_oversight(run_ids, id_mapping)
        sync_transparency_measures(run_ids, id_mapping)

        # Compliance tables (run-level)
        sync_change_logs(run_ids)
        sync_declaration_of_conformity(run_ids)
        sync_visual_documentation(run_ids)
        sync_explainable_ai(run_ids)

        # Ensure model_deployed parent rows exist before deployment-level tables
        sync_model_deployed(run_ids, id_mapping)

        # Compliance tables (deployment-level)
        sync_model_packaging(run_ids, id_mapping)
        sync_build_testing(run_ids, id_mapping)
        sync_standards(run_ids, id_mapping)
        sync_interfaces(run_ids, id_mapping)
        sync_decommissioning(run_ids, id_mapping)

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
    # Get all data from mlflow_metrics and runs tables
    metrics_df = get_metrics_data()
    runs_df = get_runs_data()  # DONE
    experiments_df = get_experiments_data()  # DONE
    experiment_tags_df = get_experiment_tags_data()  # DONE
    datasets_df = get_datasets_data()  # DONE
    latest_metrics_df = get_latest_metrics_data()  # DONE
    params_df = get_params_data()
    tags_df = get_tags_data()
    whylogs_df = get_artifacts_data()  # DONE

    # Print all dataframes
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
