"""
Log experiment-level risk, oversight, transparency, and change governance data
to MLflow artifacts.

Functions
---------
log_risk                 → ``risks/``                   → ``risks`` table
log_human_oversight      → ``human_oversight/``          → ``human_oversight_mechanisms`` table
log_transparency_measure → ``transparency_measures/``    → ``transparency_measures`` table
log_change               → ``change_logs/``              → ``change_logs`` table
"""

from certain_library.tracking.tracker import tracker

import os
import json
import time
import tempfile
import uuid
from typing import List

_RISK_ALLOWED_KEYS = {"risk_description", "risk_type", "risk_level", "risk_id"}


def log_risk(risks: List[dict]) -> None:
    """
    Log one or more identified risks associated with the current MLflow experiment.

    Saves one JSON artifact per risk under ``risks/`` that the ``data_api`` sync
    function reads to populate the ``risks`` table in ``certain_db``.

    Parameters
    ----------
    risks : list of dict
        Each dict describes one risk and must contain:

        * ``"risk_description"`` *(str)* — human-readable description.
        * ``"risk_type"`` *(str)* — category (e.g. ``"data_bias"``, ``"privacy"``).
        * ``"risk_level"`` *(float)* — severity in ``[0.0, 1.0]``.
        * ``"risk_id"`` *(str, optional)* — unique identifier; auto-generated when omitted.

        No other keys are accepted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any dict contains unknown keys, or if ``risk_level`` is not in
        ``[0.0, 1.0]``.
    """
    for item in risks:
        unknown = set(item.keys()) - _RISK_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in risk dict: {sorted(unknown)}. "
                f"Allowed keys: {sorted(_RISK_ALLOWED_KEYS)}."
            )
        risk_level = item["risk_level"]
        if not (0.0 <= risk_level <= 1.0):
            raise ValueError("risk_level must be between 0.0 and 1.0")

        record = {
            "risk_id": item.get("risk_id") or str(uuid.uuid4()),
            "risk_description": item["risk_description"],
            "risk_type": item["risk_type"],
            "risk_level": risk_level,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = f"risk.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="risks")


_OVERSIGHT_ALLOWED_KEYS = {
    "oversight_type",
    "description",
    "implementation_details",
    "mechanism_id",
}


def log_human_oversight(oversights: List[dict]) -> None:
    """
    Log one or more human oversight mechanisms for the current MLflow experiment.

    Saves one JSON artifact per mechanism under ``human_oversight/`` that the
    ``data_api`` sync function reads to populate the ``human_oversight_mechanisms``
    table in ``certain_db``.

    Parameters
    ----------
    oversights : list of dict
        Each dict describes one mechanism and must contain:

        * ``"oversight_type"`` *(str)* — type (e.g. ``"human-in-the-loop"``).
        * ``"description"`` *(str)* — description of how the mechanism works.
        * ``"implementation_details"`` *(str, optional)* — technical details.
        * ``"mechanism_id"`` *(str, optional)* — unique identifier; auto-generated when omitted.

        No other keys are accepted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any dict contains unknown keys.
    """
    for item in oversights:
        unknown = set(item.keys()) - _OVERSIGHT_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in oversight dict: {sorted(unknown)}. "
                f"Allowed keys: {sorted(_OVERSIGHT_ALLOWED_KEYS)}."
            )
        record = {
            "mechanism_id": item.get("mechanism_id") or str(uuid.uuid4()),
            "oversight_type": item["oversight_type"],
            "description": item["description"],
            "implementation_details": item.get("implementation_details") or "",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = f"oversight.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="human_oversight")


_TRANSPARENCY_ALLOWED_KEYS = {
    "measure_type",
    "measure_value",
    "description",
    "implementation_details",
    "measure_id",
}


def log_transparency_measure(measures: List[dict]) -> None:
    """
    Log one or more transparency measures for the current MLflow experiment.

    Saves one JSON artifact per measure under ``transparency_measures/`` that the
    ``data_api`` sync function reads to populate the ``transparency_measures`` table
    in ``certain_db``.

    Parameters
    ----------
    measures : list of dict
        Each dict describes one measure and must contain:

        * ``"measure_type"`` *(list of str)* — categories (e.g. ``["model_card"]``).
        * ``"measure_value"`` *(list of str)* — corresponding values/links (same length as ``measure_type``).
        * ``"description"`` *(str, optional)* — general description.
        * ``"implementation_details"`` *(str, optional)* — technical details.
        * ``"measure_id"`` *(str, optional)* — unique identifier; auto-generated when omitted.

        No other keys are accepted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any dict contains unknown keys, or if ``measure_type`` and
        ``measure_value`` have different lengths.
    """
    for item in measures:
        unknown = set(item.keys()) - _TRANSPARENCY_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in measure dict: {sorted(unknown)}. "
                f"Allowed keys: {sorted(_TRANSPARENCY_ALLOWED_KEYS)}."
            )
        measure_type = item["measure_type"]
        measure_value = item["measure_value"]
        if len(measure_type) != len(measure_value):
            raise ValueError("measure_type and measure_value must have the same length")

        record = {
            "measure_id": item.get("measure_id") or str(uuid.uuid4()),
            "measure_type": measure_type,
            "measure_value": measure_value,
            "description": item.get("description") or "",
            "implementation_details": item.get("implementation_details") or "",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = f"transparency.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="transparency_measures")


_CHANGE_ALLOWED_KEYS = {"change_description", "changed_by", "log_id"}


def log_change(changes: List[dict]) -> None:
    """
    Log one or more change entries for the current MLflow run.

    Saves one JSON artifact per change under ``change_logs/`` that the ``data_api``
    sync function reads to populate the ``change_logs`` table in ``certain_db``.

    Parameters
    ----------
    changes : list of dict
        Each dict describes one change entry and must contain:

        * ``"change_description"`` *(str)* — description of what was changed.
        * ``"changed_by"`` *(str)* — name or identifier of the person/team.
        * ``"log_id"`` *(str, optional)* — unique identifier; auto-generated when omitted.

        No other keys are accepted.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any dict contains unknown keys.
    """
    for item in changes:
        unknown = set(item.keys()) - _CHANGE_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in change dict: {sorted(unknown)}. "
                f"Allowed keys: {sorted(_CHANGE_ALLOWED_KEYS)}."
            )
        record = {
            "log_id": item.get("log_id") or str(uuid.uuid4()),
            "change_description": item["change_description"],
            "changed_by": item["changed_by"],
            "change_timestamp": int(time.time()),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fname = f"change.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="change_logs")
