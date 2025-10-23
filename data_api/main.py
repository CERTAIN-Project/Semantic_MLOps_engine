import uuid
import time
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
    mapped_rows = []
    for run_id in id_mapping.keys():
        mapped_rows.append(
            {
                "run_id": run_id,
                "model_id": id_mapping[run_id]["model_id"],
                "architecture_name": "Simple Model",
                "layer_structure": (
                    layer_structure[run_id]
                    if run_id in layer_structure.keys()
                    else None
                ),
                "activation_function": "ReLU",
                "optimizer": "Adam",
                "loss_function": "MSE",
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


@app.post("/sync/id_mapping")
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

    return {"status": "all data synced successfully"}


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
