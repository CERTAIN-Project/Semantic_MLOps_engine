"""
Log experiment-level governance metadata to MLflow artifacts.

Functions
---------
log_ai_actors           → saves to ``ai_actors/`` artifact folder → ``ai_actors`` table
log_labeling_procedures → saves to ``labeling_procedures/`` → ``labeling_procedures`` table
"""

from certain_library.tracking.tracker import tracker

import os
import json
import socket
import getpass
import tempfile
import uuid
from typing import List, Optional


def _detect_runner() -> dict:
    """Auto-detect the identity of whoever/whatever is running the experiment.

    Resolution order for ``name`` (first non-empty value wins):

    1. ``AI_PROVIDER_NAME`` env var  — explicit override (e.g. set in docker-compose)
    2. ``GITHUB_ACTOR``              — GitHub Actions
    3. ``GITLAB_USER_LOGIN``         — GitLab CI
    4. ``CI_USER`` / ``BUILD_USER``  — generic CI systems (Jenkins, etc.)
    5. OS username via ``getpass``   — local machine

    Additionally captures:

    * ``machine``  — hostname (container ID inside Docker, EC2 hostname on AWS, etc.)
    * ``contact``  — ``AI_PROVIDER_CONTACT`` env var, empty string if not set
    * ``run_by``   — ``"ci"`` when running inside any CI system, ``"local"`` otherwise
    """
    name = (
        os.getenv("AI_PROVIDER_NAME")
        or os.getenv("GITHUB_ACTOR")
        or os.getenv("GITLAB_USER_LOGIN")
        or os.getenv("CI_USER")
        or os.getenv("BUILD_USER")
        or getpass.getuser()
    )
    return {
        "name": name,
        "contact": os.getenv("AI_PROVIDER_CONTACT", ""),
        "machine": socket.gethostname(),
        "run_by": "ci" if os.getenv("CI") else "local",
    }


def log_ai_actors(
    auditor: str,
    organization: str,
    use_manual_info: bool = False,
    ai_provider_name: Optional[str] = None,
    ai_provider_contact: Optional[str] = None,
    ai_provider_role: Optional[str] = None,
    ai_deployer_name: Optional[str] = None,
    ai_deployer_contact: Optional[str] = None,
    ai_deployer_role: Optional[str] = None,
    ai_providers: Optional[List[dict]] = None,
    ai_deployers: Optional[List[dict]] = None,
    ai_actors_id: Optional[str] = None,
) -> None:
    """
    Log the AI actors (responsible parties) for the current MLflow experiment.

    Saves a JSON artifact under ``ai_actors/`` that the ``data_api`` sync
    function reads to populate the ``ai_actors`` table in ``certain_db``.

    There are three ways to specify the provider/deployer identity, in order
    of priority:

    1. **Simple strings** — pass ``ai_provider_name``, ``ai_deployer_name``
       (and optionally ``_contact`` / ``_role``) for the common single-person case::

           log_ai_actors(
               auditor="...",
               organization="...",
               ai_provider_name="Dimitrios Christodoulou",
               ai_provider_contact="d@example.com",
               ai_deployer_name="Energy Ops Team",
           )

    2. **Full dicts** — pass ``ai_providers`` / ``ai_deployers`` lists for
       multiple actors or when you need full control::

           log_ai_actors(
               auditor="...",
               organization="...",
               ai_providers=[{"name": "Team A", "role": "dev", "contact": "a@x.com"}],
               ai_deployers=[{"name": "Team B"}],
           )

    3. **Auto-detect** — omit everything and the runner identity is captured
       automatically from the environment:

       * ``AI_PROVIDER_NAME`` env var  → e.g. set in ``.env`` or docker-compose
       * ``GITHUB_ACTOR``              → GitHub Actions username
       * ``GITLAB_USER_LOGIN``         → GitLab CI username
       * OS username (``getpass``)     → local machine fallback

       The current hostname is always recorded as ``"machine"``.

    Parameters
    ----------
    auditor : str
        Name of the auditor or certification body.
    organization : str
        Parent organisation that owns the AI project.
    use_manual_info : bool, optional
        If ``True``, the manually provided ``ai_provider_name`` /
        ``ai_deployer_name`` (and their ``_role`` / ``_contact`` companions)
        are used.  If ``False`` (default), the runner identity is always
        auto-detected from the environment regardless of what is passed.
    ai_provider_name : str, optional
        Simple name of the AI provider (shortcut for single-person use).
    ai_provider_contact : str, optional
        Contact e-mail for the provider (used with ``ai_provider_name``).
    ai_provider_role : str, optional
        Role description for the provider (used with ``ai_provider_name``).
    ai_deployer_name : str, optional
        Simple name of the AI deployer (shortcut for single-person use).
    ai_deployer_contact : str, optional
        Contact e-mail for the deployer (used with ``ai_deployer_name``).
    ai_deployer_role : str, optional
        Role description for the deployer (used with ``ai_deployer_name``).
    ai_providers : list of dict, optional
        Full list of provider descriptors (overrides ``ai_provider_name``).
    ai_deployers : list of dict, optional
        Full list of deployer descriptors (overrides ``ai_deployer_name``).
    ai_actors_id : str, optional
        Unique identifier — auto-generated when omitted.

    Returns
    -------
    None
    """
    _ACTOR_ALLOWED_KEYS = {"name", "role", "contact", "machine", "run_by"}

    # Priority 1: full dicts passed explicitly
    # Priority 2: simple string shortcut parameters
    # Priority 3: auto-detect from environment
    # Backwards-compatibility: callers may pass (ai_providers, ai_deployers)
    # as positional third/fourth args. Detect this and reassign.
    if isinstance(use_manual_info, list) and ai_providers is None:
        ai_providers = use_manual_info
        # If the next parameter is also a list, it's the deployers list.
        if isinstance(ai_provider_name, list):
            ai_deployers = ai_provider_name
        # Mark that manual info was provided
        use_manual_info = True

    if ai_providers is None:
        if use_manual_info and ai_provider_name is not None:
            entry = {"name": ai_provider_name}
            if ai_provider_role:
                entry["role"] = ai_provider_role
            if ai_provider_contact:
                entry["contact"] = ai_provider_contact
            ai_providers = [entry]
        else:
            ai_providers = [_detect_runner()]

    if ai_deployers is None:
        if use_manual_info and ai_deployer_name is not None:
            entry = {"name": ai_deployer_name}
            if ai_deployer_role:
                entry["role"] = ai_deployer_role
            if ai_deployer_contact:
                entry["contact"] = ai_deployer_contact
            ai_deployers = [entry]
        else:
            ai_deployers = [_detect_runner()]

    for field_name, lst in (
        ("ai_providers", ai_providers),
        ("ai_deployers", ai_deployers),
    ):
        if not isinstance(lst, list) or not lst:
            raise ValueError(f"{field_name} must be a non-empty list of dicts")
        for i, item in enumerate(lst):
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError(f"{field_name}[{i}] must be a dict with a 'name' key")
            unknown = set(item.keys()) - _ACTOR_ALLOWED_KEYS
            if unknown:
                raise ValueError(
                    f"{field_name}[{i}] contains unrecognised keys: {sorted(unknown)}. "
                    f"Allowed keys are: {sorted(_ACTOR_ALLOWED_KEYS)}"
                )

    record = {
        "ai_actors_id": ai_actors_id or str(uuid.uuid4()),
        "ai_providers": ai_providers,
        "ai_deployers": ai_deployers,
        "auditor": auditor,
        "organization": organization,
    }
    # If an MLflow active run exists, include canonical experiment_id and run_id
    # inside the artifact so downstream sync can use those values directly.
    try:
        import mlflow

        run = mlflow.active_run()
        if run is not None:
            record["experiment_id"] = run.info.experiment_id
            record["run_id"] = run.info.run_id
    except Exception:
        # mlflow not available or failure to read active run — fall back to
        # leaving the artifact without run/experiment fields.
        pass
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "ai_actors.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(path, artifact_path="ai_actors")


def log_labeling_procedures(
    quality_assurance_methods: List[str],
    procedures: List[dict],
) -> None:
    """
    Log the data-labeling / annotation procedures for the current MLflow
    experiment.

    Each element of *procedures* produces a separate JSON artifact under
    ``labeling_procedures/`` so that the ``data_api`` sync function can insert
    one row per procedure into the ``labeling_procedures`` table in
    ``certain_db``.

    Parameters
    ----------
    quality_assurance_methods : list of str
        One or more QA/QC methods applied across all procedures (e.g.
        ``["inter-annotator agreement", "double-blind review"]``).
    procedures : list of dict
        Non-empty list of procedure descriptors.  Each dict must contain
        ``"description"`` and ``"annotation_tool"``, and may optionally include:

        * ``"annotators"`` (list of str) — names or IDs of the annotators
          responsible for this procedure
        * ``"link"`` (str) — URL or artifact path to further documentation
          for this procedure
        * ``"labeling_id"`` (str) — auto-generated when absent

        No other keys are accepted.

        Example::

            log_labeling_procedures(
                quality_assurance_methods=["inter-annotator agreement", "expert review"],
                procedures=[
                    {
                        "description": "Named-entity recognition via crowdsourcing",
                        "annotation_tool": "Label Studio",
                        "annotators": ["annotator_01", "annotator_02"],
                        "link": "s3://docs/ner_guidelines.pdf",
                    },
                    {
                        "description": "Sentiment classification by domain experts",
                        "annotation_tool": "Prodigy",
                        "annotators": ["expert_alice", "expert_bob"],
                        "link": "s3://docs/sentiment_guidelines.pdf",
                    },
                ],
            )

    Raises
    ------
    ValueError
        If *quality_assurance_methods* is not a list, if *procedures* is not a
        non-empty list, if any procedure dict is missing ``"description"`` or
        ``"annotation_tool"``, or if any procedure dict contains an unrecognised
        key.

    Returns
    -------
    None
    """
    _PROCEDURE_ALLOWED_KEYS = {
        "description",
        "annotation_tool",
        "annotators",
        "link",
        "labeling_id",
    }
    if not isinstance(quality_assurance_methods, list):
        raise ValueError("quality_assurance_methods must be a list of strings")
    if not isinstance(procedures, list) or not procedures:
        raise ValueError("procedures must be a non-empty list of dicts")
    for i, p in enumerate(procedures):
        for required in ("description", "annotation_tool"):
            if not isinstance(p, dict) or required not in p:
                raise ValueError(
                    f"procedures[{i}] must be a dict with a '{required}' key"
                )
        unknown = set(p.keys()) - _PROCEDURE_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"procedures[{i}] contains unrecognised keys: {sorted(unknown)}. "
                f"Allowed keys are: {sorted(_PROCEDURE_ALLOWED_KEYS)}"
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for p in procedures:
            labeling_id = p.get("labeling_id") or str(uuid.uuid4())
            record = {
                "labeling_id": labeling_id,
                "quality_assurance_methods": quality_assurance_methods,
                "description": p["description"],
                "annotation_tool": p["annotation_tool"],
                "annotators": p.get("annotators") or [],
                "link": p.get("link") or "",
            }
            fname = f"labeling.json"
            path = os.path.join(tmp_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(path, artifact_path="labeling_procedures")
