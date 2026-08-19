"""
Log deployment-lifecycle compliance metadata to MLflow artifacts.

These functions are called **after** a model has been registered / deployed
and a ``deployment_id`` is known.  They store JSON artifacts that the
``data_api`` sync functions read to populate the corresponding ``certain_db``
tables.

Functions
---------
log_model_packaging  → ``model_packaging/``             → ``model_packaging`` table
log_build_testing    → ``build_and_integration_testing/`` → ``build_and_integration_testing`` table
log_standard         → ``standards/``                   → ``standards`` table
log_interface        → ``interfaces/``                  → ``interfaces`` table
log_model_deployed   → ``deployment_logs/``             → ``model_deployed`` table
log_monitor_logs     → ``deployment_logs/``             → ``monitor_logs`` table
log_decommissioning  → ``decommissioning/``             → ``decomissioning`` table
"""

from certain_library.tracking.tracker import tracker

import os
import json
import time
import tempfile
import uuid
from typing import List, Optional


def log_model_packaging(
    deployment_id: str,
    model_id: str,
    packaging_format: str,
    dependencies: List[str],
    containerization_details: Optional[dict] = None,
    packaging_id: Optional[str] = None,
) -> None:
    """
    Log model packaging information for a deployed model.

    Saves a JSON artifact under ``model_packaging/`` that the ``data_api``
    sync function reads to populate the ``model_packaging`` table in
    ``certain_db``.

    Parameters
    ----------
    deployment_id : str
        Identifier of the deployment this packaging record belongs to.
    model_id : str
        Identifier of the model being packaged.
    packaging_format : str
        Format in which the model is packaged (e.g. ``"docker"``, ``"onnx"``,
        ``"mlflow_model"``, ``"torchscript"``).
    dependencies : list of str
        List of runtime dependencies (e.g. ``["scikit-learn==1.3", "numpy>=1.24"]``).
    containerization_details : dict, optional
        Structured description of the container or environment. Example::

            {
                "base_image": "python:3.11-slim",
                "cpu": "2",
                "memory": "4GB",
                "port": 8080,
                "registry": "docker.io/myorg/mymodel"
            }

    packaging_id : str, optional
        Unique identifier.  Auto-generated when omitted.

    Raises
    ------
    ValueError
        If ``dependencies`` is not a list or ``containerization_details`` is
        provided but is not a dict.

    Returns
    -------
    None
    """
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list of strings")
    if containerization_details is not None and not isinstance(
        containerization_details, dict
    ):
        raise ValueError("containerization_details must be a dict")

    record = {
        "packaging_id": packaging_id or str(uuid.uuid4()),
        "deployment_id": deployment_id,
        "model_id": model_id,
        "packaging_format": packaging_format,
        "dependencies": dependencies,
        "containerization_details": containerization_details or {},
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "model_packaging.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="model_packaging")


def log_build_testing(
    deployment_id: str,
    model_id: str,
    build_status: str,
    build_logs: str,
    test_type: str,
    test_results: dict,
    test_id: Optional[str] = None,
) -> None:
    """
    Log build and integration testing results for a deployed model.

    Saves a JSON artifact under ``build_and_integration_testing/`` that the
    ``data_api`` sync function reads to populate the
    ``build_and_integration_testing`` table in ``certain_db``.

    Parameters
    ----------
    deployment_id : str
        Identifier of the deployment this test record belongs to.
    model_id : str
        Identifier of the model that was tested.
    build_status : str
        Overall build outcome (e.g. ``"success"``, ``"failure"``, ``"warning"``).
    build_logs : str
        Raw build log output or a summary.
    test_type : str
        Category of test (e.g. ``"unit"``, ``"integration"``, ``"smoke"``).
    test_results : dict
        Structured test results. Example::

            {
                "total": 42,
                "passed": 42,
                "failed": 0,
                "skipped": 0,
                "coverage_pct": 87.3
            }

    test_id : str, optional
        Unique identifier.  Auto-generated when omitted.

    Raises
    ------
    ValueError
        If ``test_results`` is not a dict.

    Returns
    -------
    None
    """
    if not isinstance(test_results, dict):
        raise ValueError("test_results must be a dict")
    # Normalize and require deployment and model identifiers. If missing,
    # don't write the artifact — build/test artifacts should be emitted at
    # deployment time when those identifiers are known.
    if not deployment_id or not model_id:
        # no deployment context -> no artifact written
        return

    record = {
        "test_id": test_id or str(uuid.uuid4()),
        "deployment_id": deployment_id,
        "model_id": model_id,
        "build_status": build_status,
        "build_logs": build_logs,
        "build_timestamp": int(time.time()),
        "test_type": test_type,
        "test_results": test_results,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        fname = f"build_test.json"
        path = os.path.join(tmp_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        # When run under an active MLflow run, include the run_id so the
        # data_api sync can resolve deployment/model ids from id_mapping if
        # necessary. The tracker will attach the artifact to the active run.
        try:
            import mlflow

            arun = mlflow.active_run()
            if arun is not None:
                # read-modify-write the JSON to include run_id
                try:
                    with open(path, "r+", encoding="utf-8") as fh:
                        data = json.load(fh)
                        data.setdefault("run_id", arun.info.run_id)
                        fh.seek(0)
                        json.dump(data, fh, indent=2)
                        fh.truncate()
                except Exception:
                    # if file ops fail, ignore and proceed to log artifact
                    pass
        except Exception:
            # mlflow not available — continue and write artifact without run_id
            pass

        tracker.log_artifact(path, artifact_path="build_and_integration_testing")


def log_standards(
    deployment_id: str,
    model_id: str,
    standards: List[dict],
) -> None:
    """
    Log one or more applicable standards / regulations for a deployed model.

    Each element of *standards* produces a separate JSON artifact under
    ``standards/`` so that the ``data_api`` sync function can insert one row
    per standard into the ``standards`` table in ``certain_db``.

    Parameters
    ----------
    deployment_id : str
        Identifier of the deployment these standards apply to.
    model_id : str
        Identifier of the model these standards apply to.
    standards : list of dict
        Non-empty list of standard descriptors.  Each dict must contain at
        least ``"name"`` and may optionally include:

        * ``"description"`` (str)
        * ``"version"`` (str)
        * ``"publication_date"`` (float — Unix timestamp)
        * ``"standard_id"`` (str — auto-generated when absent)

        Example::

            log_standards(
                deployment_id="dep-123",
                model_id="model-abc",
                standards=[
                    {"name": "ISO/IEC 42001:2023", "version": "2023"},
                    {"name": "EU AI Act", "description": "EU regulation on AI"},
                ],
            )

    Raises
    ------
    ValueError
        If *standards* is not a non-empty list, or if any element is missing
        the ``"name"`` key.

    Returns
    -------
    None
    """
    # Only log standards when tied to a deployment/model. If deployment
    # identifiers are missing, do not persist deployment-scoped standards.
    if not deployment_id or not model_id:
        return

    if not isinstance(standards, list) or not standards:
        raise ValueError("standards must be a non-empty list of dicts")
    for i, s in enumerate(standards):
        if not isinstance(s, dict) or "name" not in s:
            raise ValueError(
                f"standards[{i}] must be a dict with at least a 'name' key"
            )

    # Require deployment and model identifiers — only emit standards when
    # tied to a deployment.
    if not deployment_id or not model_id:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        for s in standards:
            standard_id = s.get("standard_id") or str(uuid.uuid4())
            record = {
                "standard_id": standard_id,
                "deployment_id": deployment_id,
                "model_id": model_id,
                "name": s["name"],
                "description": s.get("description") or "",
                "version": s.get("version") or "",
                "publication_date": s.get("publication_date"),
            }
            fname = f"standard.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            try:
                import mlflow

                run = mlflow.active_run()
                if run is not None:
                    # annotate artifact with run_id for downstream mapping
                    with open(path, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    payload.setdefault("run_id", run.info.run_id)
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, indent=2)
            except Exception:
                pass
            tracker.log_artifact(path, artifact_path="standards")


def log_interface(
    deployment_id: str,
    model_id: str,
    interface_type: str,
    specifications: Optional[str] = None,
    version: Optional[str] = None,
    documentation_link: Optional[str] = None,
    interface_id: Optional[str] = None,
) -> None:
    """
    Log an interface specification for a deployed model.

    Saves a JSON artifact under ``interfaces/`` that the ``data_api``
    sync function reads to populate the ``interfaces`` table in ``certain_db``.

    Parameters
    ----------
    deployment_id : str
        Identifier of the deployment this interface belongs to.
    model_id : str
        Identifier of the model that exposes this interface.
    interface_type : str
        Type of interface (e.g. ``"REST API"``, ``"gRPC"``, ``"WebSocket"``).
    specifications : str, optional
        Technical specifications (e.g. OpenAPI schema, endpoint list).
    version : str, optional
        Version of the interface.
    documentation_link : str, optional
        URL to the full interface documentation.
    interface_id : str, optional
        Unique identifier.  Auto-generated when omitted.

    Returns
    -------
    None
    """
    record = {
        "interface_id": interface_id or str(uuid.uuid4()),
        "deployment_id": deployment_id,
        "model_id": model_id,
        "interface_type": interface_type,
        "specifications": specifications or "",
        "version": version or "",
        "documentation_link": documentation_link or "",
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        fname = f"interface.json"
        path = os.path.join(tmp_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="interfaces")


def log_model_deployed(
    deployment_id: str,
    model_id: str,
    experiment_id: Optional[str] = None,
    run_id: Optional[str] = None,
    model_version: Optional[str] = None,
    endpoint: Optional[str] = None,
    model_format: Optional[str] = None,
    size: Optional[str] = None,
    description: Optional[str] = None,
    user_id: Optional[str] = None,
    current_stage: Optional[str] = None,
    location: Optional[str] = None,
    status: Optional[str] = None,
    model_cateory: Optional[str] = None,
    deployment_log: Optional[str] = None,
) -> None:
    """Log a structured deployment manifest for the model_deployed table.

    This captures the columns stored in ``model_deployed`` and preserves the
    raw deployment log snapshot alongside them so the sync layer can populate
    the table even when only the textual deployment log is available.
    """
    if not deployment_id or not model_id:
        return

    try:
        import mlflow

        active = mlflow.active_run()
        if run_id is None and active is not None:
            run_id = active.info.run_id
        if experiment_id is None and active is not None:
            experiment_id = active.info.experiment_id
    except Exception:
        active = None

    record = {
        "deployment_id": deployment_id,
        "model_id": model_id,
        "experiment_id": experiment_id or "",
        "run_id": run_id or "",
        "model_version": model_version or "",
        "endpoint": endpoint or "",
        "model_format": model_format or "",
        "size": size or "",
        "description": description or "",
        "user_id": user_id or "",
        "current_stage": current_stage or "",
        "location": location or endpoint or "",
        "status": status or ("deployed" if endpoint else "not available yet"),
        "model_cateory": model_cateory or "",
        "deployment_log": deployment_log or "",
        "deployed_time": int(time.time()),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "model_deployed.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="deployment_logs")


def log_monitor_logs(
    deployment_id: str,
    model_id: str,
    message: str,
    experiment_id: Optional[str] = None,
    log_id: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Log a deployment monitor message for the monitor_logs table.

    The message is typically the deployment terminal output captured in
    ``deployment_run.log`` so the database can store a searchable summary of
    the deployment session.
    """
    if not deployment_id or not model_id or not message:
        return

    try:
        import mlflow

        active = mlflow.active_run()
        if experiment_id is None and active is not None:
            experiment_id = active.info.experiment_id
    except Exception:
        active = None

    record = {
        "log_id": log_id or str(uuid.uuid4()),
        "deployment_id": deployment_id,
        "experiment_id": experiment_id or "",
        "model_id": model_id,
        "message": message,
        "source": source or "deployment_run.log",
        "captured_at": int(time.time()),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "monitor_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="deployment_logs")


def log_decommissioning(
    deployment_id: str,
    model_id: str,
    decommissioning_actions: List[str],
    reason: str,
    procedure_details: Optional[str] = None,
    decommissioning_date: Optional[float] = None,
    decommissioning_id: Optional[str] = None,
    system_name: Optional[str] = None,
    decommissioning_plan: Optional[str] = None,
    approvals: Optional[List[str]] = None,
    data_retention_archive: Optional[str] = None,
    migration: Optional[str] = None,
    access_removal: Optional[str] = None,
    infrastructure_shutdown: Optional[str] = None,
    evidence_documentation: Optional[List[str]] = None,
    audit_trail: Optional[str] = None,
) -> None:
    """
    Log model decommissioning information for a deployed model.

    Saves a JSON artifact under ``decommissioning/`` that the ``data_api``
    sync function reads to populate the ``decomissioning`` table in
    ``certain_db``.

    Parameters
    ----------
    deployment_id : str
        Identifier of the deployment being decommissioned.
    model_id : str
        Identifier of the model being decommissioned.
    decommissioning_actions : list of str
        Ordered steps taken during decommissioning
        (e.g. ``["archive weights", "remove endpoint", "notify stakeholders"]``).
    reason : str
        Reason for decommissioning the model.
    procedure_details : str, optional
        Detailed description of the decommissioning procedure.
    decommissioning_date : float, optional
        Unix timestamp of when decommissioning was performed.
        Defaults to the current time.
    decommissioning_id : str, optional
        Unique identifier.  Auto-generated when omitted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``decommissioning_actions`` is not a non-empty list.
    """
    if not isinstance(decommissioning_actions, list) or not decommissioning_actions:
        raise ValueError("decommissioning_actions must be a non-empty list of strings")

    record = {
        "decomissioning_id": decommissioning_id or str(uuid.uuid4()),
        "deployment_id": deployment_id,
        "model_id": model_id,
        "system_name": system_name or "",
        "decommissioning_plan": decommissioning_plan or "",
        "approvals": approvals or [],
        "data_retention_archive": data_retention_archive or "",
        "migration": migration or "",
        "access_removal": access_removal or "",
        "infrastructure_shutdown": infrastructure_shutdown or "",
        "evidence_documentation": evidence_documentation or [],
        "audit_trail": audit_trail or "",
        "decomissioning_date": decommissioning_date or int(time.time()),
        "decomissioning_actions": decommissioning_actions,
        "reason": reason,
        "procedure_details": procedure_details or "",
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "decommissioning.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="decommissioning")
