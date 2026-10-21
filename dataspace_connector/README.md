# Dataspace Connector Workspace

This folder is isolated from the current `Semantic_MLOps_engine` workflow code.
It is intended for datasource / dataspace connector work, including:

- consumer-side dataset retrieval
- receiver endpoints for pushed datasets
- connector-specific examples and integration notes
- optional adapter code that can be reused later without changing the existing workflow files

## Suggested Layout

- `receiver/` - files related to receiving datasets from a dataspace provider
- `consumer/` - files related to pulling or downloading datasets from a provider endpoint

## Goal

Keep the current MLOps workflow unchanged while experimenting with dataspace connector integration here.

## Receiver endpoint

The receiver app accepts `POST` and `PUT` requests with a raw request body and stores the payload on disk. It also exposes `GET` so the consumer loader can read the stored files back.

Start it with:

```bash
export DATASPACE_SINK_DIR=/tmp/certain_dataspace_sink
uvicorn dataspace_connector.receiver.app:app --host 0.0.0.0 --port 8080
```

Use these endpoints:

- `PUT /dataspace/sink/<filename>` to store a payload
- `POST /dataspace/sink/<filename>` to store a payload
- `GET /dataspace/sink/<filename>` to read the stored payload back
- `GET /dataspace/sink` to list received payloads

If the provider pushes the three expected files into this receiver, you can point `DATASPACE_SOURCE` at `http://localhost:8080/dataspace/sink` and the consumer loader will read them back from the sink.

## Consumer loader

The consumer logic lives in [consumer/loader.py](consumer/loader.py). It supports:

- HTTP retrieval from a dataspace endpoint
- S3/MinIO retrieval (`s3://<bucket>/<prefix>`)
- local folder retrieval
- `file://` URLs

It loads the demo payloads and returns them in a small `DemoBatch` container so the rest of the CERTAIN validation flow can continue unchanged.

## Expected file names

The consumer expects the provider to make the following files available:

```text
simulation_output_clean_7.json
simulation_output_dirty_7.json
scenario_annotations_batch_7.json
```

If the provider uses a different batch id, the same naming pattern applies with that batch number.

## Notes

- The current workflow stays offline except for the dataspace dataset retrieval step.
- If the HTTP source is unavailable, the script will fail fast so you know the connector is not reachable.
- If you prefer to keep everything local for testing, use a mounted folder path instead of an HTTP endpoint.
