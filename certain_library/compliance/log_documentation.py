"""
Log run-level documentation and explainability metadata to MLflow artifacts.

Functions
---------
log_declarations_of_conformity → ``declaration_of_conformity/``  → ``declaration_of_conformity`` table
log_visual_documentations      → ``visual_documentation/``        → ``visual_documentation`` table
log_explainable_ai             → ``explainable_ai/``              → ``explainable_ai_features`` table
"""

from certain_library.tracking.tracker import tracker

import os
import json
import time
import tempfile
import uuid
from typing import List, Optional



def log_declarations_of_conformity(
    issuer: str,
    declarations: List[dict],
    version: Optional[str] = None,
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
    standard_references: Optional[List[str]] = None,
) -> None:
    """
    Log one or more declaration-of-conformity document references for the
    current MLflow run.

    Each element of *declarations* produces a separate JSON artifact under
    ``declaration_of_conformity/`` so that the ``data_api`` sync function can
    insert one row per document into the ``declaration_of_conformity`` table
    in ``certain_db``.

    The named parameters (``issuer``, ``version``, ``valid_from``,
    ``valid_until``, ``standard_references``) are shared context that applies
    to every declaration in the batch and are written into each JSON record.

    Parameters
    ----------
    issuer : str
        Organisation or person who issued / signed the declaration
        (e.g. ``"CERTAIN Project Consortium"``).
    declarations : list of dict
        Non-empty list of declaration descriptors.  Each dict must contain at
        least ``"filename"``, ``"file_type"``, and ``"mime_type"``, and may
        optionally include:

        * ``"description"`` (str)
        * ``"link_to_artifacts"`` (str)
        * ``"file_size"`` (int)
        * ``"declaration_id"`` (str — auto-generated when absent)

        No other keys are accepted.

        Example::

            log_declarations_of_conformity(
                issuer="CERTAIN Project Consortium",
                version="v2.1",
                valid_from=1_700_000_000.0,
                valid_until=1_800_000_000.0,
                standard_references=["ISO/IEC 42001:2023", "EU AI Act Art. 48"],
                declarations=[
                    {
                        "filename": "DoC_v1.pdf",
                        "file_type": "pdf",
                        "mime_type": "application/pdf",
                        "description": "Initial declaration",
                    },
                    {
                        "filename": "DoC_v2.pdf",
                        "file_type": "pdf",
                        "mime_type": "application/pdf",
                    },
                ],
            )

    version : str, optional
        Document revision (e.g. ``"v2.1"``).
    valid_from : float, optional
        Start of the validity period as a Unix timestamp.
    valid_until : float, optional
        End of the validity period as a Unix timestamp.
    standard_references : list of str, optional
        Standards / regulations this declaration refers to
        (e.g. ``["ISO/IEC 42001:2023", "EU AI Act Art. 48"]``).

    Raises
    ------
    ValueError
        If *declarations* is not a non-empty list, if any element is missing
        ``"filename"``, ``"file_type"``, or ``"mime_type"``, or if any element
        contains an unrecognised key.

    Returns
    -------
    None
    """
    _DECLARATION_ALLOWED_KEYS = {
        "filename",
        "file_type",
        "mime_type",
        "description",
        "link_to_artifacts",
        "file_size",
        "declaration_id",
    }
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("declarations must be a non-empty list of dicts")
    for i, d in enumerate(declarations):
        for required in ("filename", "file_type", "mime_type"):
            if not isinstance(d, dict) or required not in d:
                raise ValueError(
                    f"declarations[{i}] must be a dict with '{required}' key"
                )
        unknown = set(d.keys()) - _DECLARATION_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"declarations[{i}] contains unrecognised keys: {sorted(unknown)}. "
                f"Allowed keys are: {sorted(_DECLARATION_ALLOWED_KEYS)}"
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for d in declarations:
            declaration_id = d.get("declaration_id") or str(uuid.uuid4())
            record = {
                "declaration_id": declaration_id,
                "issuer": issuer,
                "version": version or "",
                "valid_from": valid_from,
                "valid_until": valid_until,
                "standard_references": standard_references or [],
                "filename": d["filename"],
                "file_type": d["file_type"],
                "mime_type": d["mime_type"],
                "description": d.get("description") or "",
                "link_to_artifacts": d.get("link_to_artifacts") or "",
                "file_size": d.get("file_size"),
                "creation_time": int(time.time()),
            }
            fname = f"declaration_{declaration_id}.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="declaration_of_conformity")


def log_visual_documentations(
    stage: str,
    documents: List[dict],
    generated_by: Optional[str] = None,
    model_version: Optional[str] = None,
) -> None:
    """
    Log one or more visual documentation artifact references for the current
    MLflow run.

    Visual documentation includes architecture diagrams, data flow charts,
    confusion matrices, or any visual artefact that supports transparency.
    Each element of *documents* produces a separate JSON artifact under
    ``visual_documentation/`` so that the ``data_api`` sync function can insert
    one row per document into the ``visual_documentation`` table in
    ``certain_db``.

    The named parameters (``stage``, ``generated_by``, ``model_version``) are
    shared context that applies to every document in the batch and are written
    into each JSON record.

    Parameters
    ----------
    stage : str
        ML lifecycle stage at which this visual was produced.  Typical values:
        ``"training"``, ``"evaluation"``, ``"deployment"``, ``"monitoring"``.
    documents : list of dict
        Non-empty list of document descriptors.  Each dict must contain at
        least ``"filename"`` and ``"file_type"``, and may optionally include:

        * ``"description"`` (str)
        * ``"tags"`` (list of str)
        * ``"link_to_artifacts"`` (str)
        * ``"file_size"`` (int)
        * ``"document_id"`` (str — auto-generated when absent)

        No other keys are accepted.

        Example::

            log_visual_documentations(
                stage="evaluation",
                generated_by="matplotlib",
                model_version="1.3.0",
                documents=[
                    {
                        "filename": "architecture.png",
                        "file_type": "png",
                        "description": "System architecture diagram",
                        "tags": ["architecture", "v2"],
                    },
                    {
                        "filename": "confusion_matrix.svg",
                        "file_type": "svg",
                    },
                ],
            )

    generated_by : str, optional
        Tool or method used to produce the visuals
        (e.g. ``"matplotlib"``, ``"SHAP TreeExplainer"``, ``"draw.io"``).
    model_version : str, optional
        Version of the model these visuals correspond to
        (e.g. ``"1.3.0"``).

    Raises
    ------
    ValueError
        If *documents* is not a non-empty list, if any element is missing
        ``"filename"`` or ``"file_type"``, or if any element contains an
        unrecognised key.

    Returns
    -------
    None
    """
    _DOCUMENT_ALLOWED_KEYS = {
        "filename",
        "file_type",
        "description",
        "tags",
        "link_to_artifacts",
        "file_size",
        "document_id",
    }
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty list of dicts")
    for i, d in enumerate(documents):
        for required in ("filename", "file_type"):
            if not isinstance(d, dict) or required not in d:
                raise ValueError(f"documents[{i}] must be a dict with '{required}' key")
        unknown = set(d.keys()) - _DOCUMENT_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"documents[{i}] contains unrecognised keys: {sorted(unknown)}. "
                f"Allowed keys are: {sorted(_DOCUMENT_ALLOWED_KEYS)}"
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for d in documents:
            document_id = d.get("document_id") or str(uuid.uuid4())
            record = {
                "document_id": document_id,
                "stage": stage,
                "generated_by": generated_by or "",
                "model_version": model_version or "",
                "filename": d["filename"],
                "file_type": d["file_type"],
                "description": d.get("description") or "",
                "tags": d.get("tags") or [],
                "link_to_artifacts": d.get("link_to_artifacts") or "",
                "file_size": d.get("file_size"),
                "creation_time": int(time.time()),
            }
            fname = f"visual_doc_{document_id}.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="visual_documentation")


def log_explainable_ai(
    feature_names: List[str],
    feature_values: List[str],
    implementation_details: Optional[str] = None,
    feature_id: Optional[str] = None,
) -> None:
    """
    Log explainability information (e.g. SHAP values, feature importances) for
    the current MLflow run.

    Saves a JSON artifact under ``explainable_ai/`` that the ``data_api``
    sync function reads to populate the ``explainable_ai_features`` table in
    ``certain_db``.

    Parameters
    ----------
    feature_names : list of str
        Names of the features for which explanations are provided
        (e.g. ``["temperature", "humidity"]``).
    feature_values : list of str
        Corresponding explanation values (same length as ``feature_names``).
        Values are stored as strings to accommodate any format
        (e.g. ``["0.42", "high importance"]``).
    implementation_details : str, optional
        Description of the explainability method used (e.g. ``"SHAP TreeExplainer"``).
    feature_id : str, optional
        Unique identifier.  Auto-generated when omitted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``feature_names`` and ``feature_values`` have different lengths or
        either list is empty.
    """
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    if len(feature_names) != len(feature_values):
        raise ValueError("feature_names and feature_values must have the same length")

    record = {
        "feature_id": feature_id or str(uuid.uuid4()),
        "feature_name": feature_names,
        "feature_values": [str(v) for v in feature_values],
        "implementation_details": implementation_details or "",
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        fname = f"xai_{record['feature_id']}.json"
        path = os.path.join(tmp_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="explainable_ai")
