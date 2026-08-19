"""
Tests for the compliance certain_library logging functions and data_api map functions.

Covers:
  certain_library/compliance/log_experiment_governance.py  → log_ai_actors,
                                                              log_labeling_procedures
  certain_library/compliance/log_governance.py             → log_risk,
                                                              log_human_oversight,
                                                              log_transparency_measure,
                                                              log_change
  certain_library/compliance/log_documentation.py          → log_declarations_of_conformity,
                                                              log_visual_documentations,
                                                              log_explainable_ai
  certain_library/compliance/log_deployment.py             → log_model_packaging,
                                                              log_build_testing,
                                                              log_standards,
                                                              log_interface,
                                                              log_decommissioning
  data_api/misc/data_transform.py                          → map_ai_actors,
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
                                                              map_decommissioning
"""

import json
import os
import shutil
import sys
import uuid

import pytest
from unittest.mock import patch

# ── certain_library compliance imports ───────────────────────────────────────
from certain_library.compliance.log_experiment_governance import (
    log_ai_actors,
    log_labeling_procedures,
)
from certain_library.compliance.log_governance import (
    log_change,
    log_human_oversight,
    log_risk,
    log_transparency_measure,
)
from certain_library.compliance.log_documentation import (
    log_declarations_of_conformity,
    log_explainable_ai,
    log_visual_documentations,
)
from certain_library.compliance.log_deployment import (
    log_build_testing,
    log_decommissioning,
    log_interface,
    log_model_packaging,
    log_monitor_logs,
    log_standards,
)

# ── data_api map-function imports ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_api"))
from misc.data_transform import (  # noqa: E402
    map_ai_actors,
    map_build_testing,
    map_change_log,
    map_declaration_of_conformity,
    map_decommissioning,
    map_explainable_ai,
    map_human_oversight,
    map_interface,
    map_labeling_procedures,
    map_model_packaging,
    map_monitor_logs,
    map_risk,
    map_standard,
    map_transparency_measure,
    map_visual_documentation,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared test data
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = str(uuid.uuid4())
EXPERIMENT_ID = "1"
DEPLOYMENT_ID = str(uuid.uuid4())
MODEL_ID = str(uuid.uuid4())

ID_MAPPING = {
    RUN_ID: {
        "model_id": MODEL_ID,
        "data_id": str(uuid.uuid4()),
        "deployment_id": DEPLOYMENT_ID,
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: capture the JSON dict written by a log_* function
# ─────────────────────────────────────────────────────────────────────────────


def _capture_json(log_fn, *args, **kwargs) -> dict:
    """Call log_fn and return the dict that was written to the temp JSON file."""
    captured = {}

    def fake_log(path, artifact_path=None):
        with open(path) as fh:
            captured.update(json.load(fh))

    with patch("mlflow.log_artifact", side_effect=fake_log):
        log_fn(*args, **kwargs)

    return captured


# ═════════════════════════════════════════════════════════════════════════════
# certain_library — log_experiment_governance
# ═════════════════════════════════════════════════════════════════════════════


class TestLogAiActors:
    _providers = [
        {"name": "ML Team A", "role": "development"},
        {"name": "Data Lab B", "role": "curation"},
    ]
    _deployers = [
        {"name": "Ops Team", "role": "production"},
        {"name": "Partner Ops", "role": "staging"},
    ]

    def test_artifact_path_correct(self):
        with patch("mlflow.log_artifact") as mock:
            log_ai_actors("AuditFirm", "OrgX", self._providers, self._deployers)
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "ai_actors"

    def test_json_contains_expected_fields(self):
        data = _capture_json(
            log_ai_actors, "AuditFirm", "OrgX", self._providers, self._deployers
        )
        assert data["auditor"] == "AuditFirm"
        assert data["organization"] == "OrgX"
        assert data["ai_providers"] == self._providers
        assert data["ai_deployers"] == self._deployers

    def test_auto_generates_uuid_when_id_not_provided(self):
        data = _capture_json(log_ai_actors, "A", "O", [{"name": "P"}], [{"name": "D"}])
        uuid.UUID(data["ai_actors_id"])  # must be a valid UUID

    def test_custom_id_preserved(self):
        data = _capture_json(
            log_ai_actors,
            "A",
            "O",
            [{"name": "P"}],
            [{"name": "D"}],
            ai_actors_id="my-id",
        )
        assert data["ai_actors_id"] == "my-id"

    def test_raises_on_empty_providers(self):
        with pytest.raises(ValueError, match="ai_providers"):
            log_ai_actors("A", "O", [], [{"name": "D"}])

    def test_raises_on_empty_deployers(self):
        with pytest.raises(ValueError, match="ai_deployers"):
            log_ai_actors("A", "O", [{"name": "P"}], [])

    def test_raises_when_provider_missing_name(self):
        with pytest.raises(ValueError):
            log_ai_actors("A", "O", [{"role": "dev"}], [{"name": "D"}])

    def test_raises_on_unknown_provider_key(self):
        with pytest.raises(ValueError, match="unrecognised keys"):
            log_ai_actors(
                "A", "O", [{"name": "P", "organisation": "typo"}], [{"name": "D"}]
            )

    def test_raises_on_unknown_deployer_key(self):
        with pytest.raises(ValueError, match="unrecognised keys"):
            log_ai_actors(
                "A", "O", [{"name": "P"}], [{"name": "D", "departement": "typo"}]
            )


class TestLogLabelingProcedures:
    _qa = ["inter-annotator agreement", "expert review"]
    _procedures = [
        {
            "description": "NER crowdsourcing",
            "annotation_tool": "Label Studio",
            "annotators": ["ann_01", "ann_02"],
            "link": "s3://docs/ner.pdf",
        },
        {
            "description": "Sentiment by experts",
            "annotation_tool": "Prodigy",
            "annotators": ["alice", "bob"],
            "link": "s3://docs/sentiment.pdf",
        },
    ]

    def test_artifact_path_correct(self):
        with patch("mlflow.log_artifact") as mock:
            log_labeling_procedures(self._qa, self._procedures)
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "labeling_procedures"

    def test_one_file_per_procedure(self):
        with patch("mlflow.log_artifact") as mock:
            log_labeling_procedures(self._qa, self._procedures)
            assert mock.call_count == 2

    def test_json_fields(self):
        captured = []

        def _fake(path, artifact_path=None):
            with open(path, encoding="utf-8") as fh:
                captured.append(json.load(fh))

        with patch("mlflow.log_artifact", side_effect=_fake):
            log_labeling_procedures(self._qa, self._procedures)

        assert len(captured) == 2
        assert captured[0]["description"] == "NER crowdsourcing"
        assert captured[0]["annotation_tool"] == "Label Studio"
        assert captured[0]["annotators"] == ["ann_01", "ann_02"]
        assert captured[0]["link"] == "s3://docs/ner.pdf"
        assert captured[0]["quality_assurance_methods"] == self._qa
        assert captured[1]["description"] == "Sentiment by experts"
        # QA methods shared across both procedures
        assert captured[1]["quality_assurance_methods"] == self._qa

    def test_raises_when_qa_methods_not_list(self):
        with pytest.raises(ValueError, match="quality_assurance_methods"):
            log_labeling_procedures("not-a-list", self._procedures)

    def test_raises_on_empty_procedures(self):
        with pytest.raises(ValueError, match="procedures"):
            log_labeling_procedures(self._qa, [])

    def test_raises_when_procedure_missing_description(self):
        with pytest.raises(ValueError):
            log_labeling_procedures(self._qa, [{"annotation_tool": "X"}])

    def test_raises_on_unknown_procedure_key(self):
        with pytest.raises(ValueError, match="unrecognised keys"):
            log_labeling_procedures(
                self._qa,
                [
                    {
                        "description": "NER",
                        "annotation_tool": "Label Studio",
                        "annotater": "typo",
                    }
                ],
            )


# ═════════════════════════════════════════════════════════════════════════════
# certain_library — log_governance
# ═════════════════════════════════════════════════════════════════════════════


class TestLogRisk:
    def test_artifact_path_correct(self):
        with patch("mlflow.log_artifact") as mock:
            log_risk(
                [
                    {
                        "risk_description": "Some risk",
                        "risk_type": "operational",
                        "risk_level": 0.3,
                    }
                ]
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "risks"

    def test_json_fields(self):
        data = _capture_json(
            log_risk,
            [
                {
                    "risk_description": "Data drift",
                    "risk_type": "technical",
                    "risk_level": 0.7,
                }
            ],
        )
        assert data["risk_description"] == "Data drift"
        assert data["risk_type"] == "technical"
        assert abs(data["risk_level"] - 0.7) < 1e-9

    def test_raises_on_risk_level_below_zero(self):
        with pytest.raises(ValueError):
            log_risk(
                [{"risk_description": "desc", "risk_type": "type", "risk_level": -0.1}]
            )

    def test_raises_on_risk_level_above_one(self):
        with pytest.raises(ValueError):
            log_risk(
                [{"risk_description": "desc", "risk_type": "type", "risk_level": 1.1}]
            )

    def test_boundary_values_accepted(self):
        _capture_json(
            log_risk,
            [{"risk_description": "desc", "risk_type": "type", "risk_level": 0.0}],
        )
        _capture_json(
            log_risk,
            [{"risk_description": "desc", "risk_type": "type", "risk_level": 1.0}],
        )

    def test_multiple_risks_logged(self):
        with patch("mlflow.log_artifact") as mock:
            log_risk(
                [
                    {"risk_description": "R1", "risk_type": "t1", "risk_level": 0.2},
                    {"risk_description": "R2", "risk_type": "t2", "risk_level": 0.8},
                ]
            )
            assert mock.call_count == 2

    def test_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown key"):
            log_risk(
                [
                    {
                        "risk_description": "d",
                        "risk_type": "t",
                        "risk_level": 0.5,
                        "extra": "bad",
                    }
                ]
            )


class TestLogHumanOversight:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_human_oversight(
                [{"oversight_type": "periodic_review", "description": "Monthly audits"}]
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "human_oversight"

    def test_json_fields(self):
        data = _capture_json(
            log_human_oversight,
            [{"oversight_type": "periodic_review", "description": "Monthly audits"}],
        )
        assert data["oversight_type"] == "periodic_review"
        assert data["description"] == "Monthly audits"

    def test_optional_implementation_details(self):
        data = _capture_json(
            log_human_oversight,
            [
                {
                    "oversight_type": "review",
                    "description": "desc",
                    "implementation_details": "automated CI checks",
                }
            ],
        )
        assert data["implementation_details"] == "automated CI checks"

    def test_multiple_oversights_logged(self):
        with patch("mlflow.log_artifact") as mock:
            log_human_oversight(
                [
                    {"oversight_type": "t1", "description": "d1"},
                    {"oversight_type": "t2", "description": "d2"},
                ]
            )
            assert mock.call_count == 2

    def test_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown key"):
            log_human_oversight(
                [{"oversight_type": "t", "description": "d", "bad_key": "x"}]
            )


class TestLogTransparencyMeasure:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_transparency_measure(
                [{"measure_type": ["explainability"], "measure_value": ["SHAP"]}]
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "transparency_measures"

    def test_json_fields(self):
        data = _capture_json(
            log_transparency_measure,
            [
                {
                    "measure_type": ["explainability", "audit_log"],
                    "measure_value": ["SHAP", "logged"],
                }
            ],
        )
        assert data["measure_type"] == ["explainability", "audit_log"]
        assert data["measure_value"] == ["SHAP", "logged"]

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            log_transparency_measure(
                [{"measure_type": ["a", "b"], "measure_value": ["x"]}]
            )

    def test_multiple_measures_logged(self):
        with patch("mlflow.log_artifact") as mock:
            log_transparency_measure(
                [
                    {"measure_type": ["model_card"], "measure_value": ["v1"]},
                    {"measure_type": ["data_sheet"], "measure_value": ["v2"]},
                ]
            )
            assert mock.call_count == 2

    def test_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown key"):
            log_transparency_measure(
                [{"measure_type": ["t"], "measure_value": ["v"], "bad": "x"}]
            )


class TestLogChange:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_change(
                [{"change_description": "Updated threshold", "changed_by": "alice"}]
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "change_logs"

    def test_json_fields(self):
        data = _capture_json(
            log_change,
            [{"change_description": "Updated threshold", "changed_by": "alice"}],
        )
        assert data["change_description"] == "Updated threshold"
        assert data["changed_by"] == "alice"
        assert "change_timestamp" in data

    def test_multiple_changes_logged(self):
        with patch("mlflow.log_artifact") as mock:
            log_change(
                [
                    {"change_description": "Change A", "changed_by": "alice"},
                    {"change_description": "Change B", "changed_by": "bob"},
                ]
            )
            assert mock.call_count == 2

    def test_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown key"):
            log_change([{"change_description": "d", "changed_by": "x", "extra": "bad"}])


# ═════════════════════════════════════════════════════════════════════════════
# certain_library — log_documentation
# ═════════════════════════════════════════════════════════════════════════════


class TestLogDeclarationsOfConformity:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_declarations_of_conformity(
                issuer="CERTAIN Consortium",
                declarations=[
                    {
                        "filename": "doc.pdf",
                        "file_type": "pdf",
                        "mime_type": "application/pdf",
                    }
                ],
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "declaration_of_conformity"

    def test_json_fields(self):
        captured = []

        def _fake(path, artifact_path=None):
            with open(path, encoding="utf-8") as fh:
                captured.append(json.load(fh))

        with patch("mlflow.log_artifact", side_effect=_fake):
            log_declarations_of_conformity(
                issuer="CERTAIN Consortium",
                version="v2.1",
                valid_from=1_700_000_000.0,
                valid_until=1_800_000_000.0,
                standard_references=["ISO/IEC 42001:2023", "EU AI Act Art. 48"],
                declarations=[
                    {
                        "filename": "conformity.pdf",
                        "file_type": "pdf",
                        "mime_type": "application/pdf",
                    },
                    {
                        "filename": "addendum.docx",
                        "file_type": "docx",
                        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "description": "Addendum",
                    },
                ],
            )

        assert len(captured) == 2
        assert captured[0]["issuer"] == "CERTAIN Consortium"
        assert captured[0]["version"] == "v2.1"
        assert captured[0]["valid_from"] == 1_700_000_000.0
        assert captured[0]["valid_until"] == 1_800_000_000.0
        assert captured[0]["standard_references"] == [
            "ISO/IEC 42001:2023",
            "EU AI Act Art. 48",
        ]
        assert captured[0]["filename"] == "conformity.pdf"
        assert captured[0]["mime_type"] == "application/pdf"
        assert captured[0]["file_size"] is not None
        assert captured[0]["link_to_artifacts"] == (
            "mlflow://artifacts/declaration_of_conformity/declaration.json"
        )
        assert captured[1]["filename"] == "addendum.docx"
        assert captured[1]["description"] == "Addendum"
        # shared fields propagated to second record too
        assert captured[1]["issuer"] == "CERTAIN Consortium"

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            log_declarations_of_conformity(issuer="X", declarations=[])

    def test_raises_when_required_key_missing(self):
        with pytest.raises(ValueError):
            log_declarations_of_conformity(
                issuer="X",
                declarations=[
                    {"filename": "f.pdf", "file_type": "pdf"}
                ],  # missing mime_type
            )

    def test_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="unrecognised keys"):
            log_declarations_of_conformity(
                issuer="X",
                declarations=[
                    {
                        "filename": "f.pdf",
                        "file_type": "pdf",
                        "mime_type": "application/pdf",
                        "autor": "typo",
                    }
                ],
            )


class TestLogVisualDocumentations:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_visual_documentations(
                stage="evaluation",
                documents=[{"filename": "diagram.png", "file_type": "png"}],
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "visual_documentation"

    def test_json_fields(self):
        captured = []

        def _fake(path, artifact_path=None):
            with open(path, encoding="utf-8") as fh:
                captured.append(json.load(fh))

        with patch("mlflow.log_artifact", side_effect=_fake):
            log_visual_documentations(
                stage="evaluation",
                generated_by="matplotlib",
                model_version="1.3.0",
                documents=[
                    {
                        "filename": "arch.png",
                        "file_type": "png",
                        "description": "Architecture diagram",
                    },
                    {
                        "filename": "flow.svg",
                        "file_type": "svg",
                        "tags": ["architecture", "flow"],
                    },
                ],
            )

        assert len(captured) == 2
        assert captured[0]["stage"] == "evaluation"
        assert captured[0]["generated_by"] == "matplotlib"
        assert captured[0]["model_version"] == "1.3.0"
        assert captured[0]["filename"] == "arch.png"
        assert captured[0]["description"] == "Architecture diagram"
        assert captured[1]["tags"] == ["architecture", "flow"]
        # shared fields propagated to second record too
        assert captured[1]["stage"] == "evaluation"

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            log_visual_documentations(stage="training", documents=[])

    def test_raises_when_required_key_missing(self):
        with pytest.raises(ValueError):
            log_visual_documentations(
                stage="training",
                documents=[{"filename": "f.png"}],  # missing file_type
            )

    def test_raises_on_unknown_key(self):
        with pytest.raises(ValueError, match="unrecognised keys"):
            log_visual_documentations(
                stage="evaluation",
                documents=[{"filename": "f.png", "file_type": "png", "autor": "typo"}],
            )


class TestLogExplainableAi:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_explainable_ai(["feat_a", "feat_b"], ["0.3", "0.7"])
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "explainable_ai"

    def test_json_fields(self):
        data = _capture_json(log_explainable_ai, ["age", "income"], ["0.5", "0.5"])
        assert data["feature_name"] == ["age", "income"]  # key is singular in library
        assert data["feature_values"] == ["0.5", "0.5"]

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            log_explainable_ai(["a", "b"], ["x"])

    def test_raises_on_empty_lists(self):
        with pytest.raises(ValueError):
            log_explainable_ai([], [])


# ═════════════════════════════════════════════════════════════════════════════
# certain_library — log_deployment
# ═════════════════════════════════════════════════════════════════════════════


class TestLogModelPackaging:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_model_packaging(DEPLOYMENT_ID, MODEL_ID, "docker", ["numpy", "sklearn"])
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "model_packaging"

    def test_json_fields(self):
        details = {"base_image": "python:3.11-slim", "port": 8080}
        data = _capture_json(
            log_model_packaging,
            DEPLOYMENT_ID,
            MODEL_ID,
            "docker",
            ["numpy"],
            containerization_details=details,
        )
        assert data["deployment_id"] == DEPLOYMENT_ID
        assert data["model_id"] == MODEL_ID
        assert data["packaging_format"] == "docker"
        assert data["dependencies"] == ["numpy"]
        assert data["containerization_details"] == details

    def test_raises_when_containerization_details_not_dict(self):
        with pytest.raises(ValueError, match="containerization_details must be a dict"):
            log_model_packaging(
                DEPLOYMENT_ID,
                MODEL_ID,
                "docker",
                ["numpy"],
                containerization_details="not-a-dict",
            )


class TestLogBuildTesting:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_build_testing(
                DEPLOYMENT_ID,
                MODEL_ID,
                "success",
                "All OK",
                "unit",
                {"passed": 42, "failed": 0},
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "build_and_integration_testing"

    def test_json_fields(self):
        results = {"total": 10, "passed": 10, "failed": 0}
        data = _capture_json(
            log_build_testing,
            DEPLOYMENT_ID,
            MODEL_ID,
            "success",
            "Build log here",
            "unit",
            results,
        )
        assert data["build_status"] == "success"
        assert data["build_logs"] == "Build log here"
        assert data["test_type"] == "unit"
        assert data["test_results"] == results  # stored as dict in JSON artifact

    def test_raises_when_test_results_not_dict(self):
        with pytest.raises(ValueError, match="test_results must be a dict"):
            log_build_testing(
                DEPLOYMENT_ID, MODEL_ID, "success", "logs", "unit", "42/42 passed"
            )


class TestLogStandards:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_standards(DEPLOYMENT_ID, MODEL_ID, [{"name": "ISO 42001"}])
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "standards"

    def test_json_fields(self):
        captured = []

        def _fake_log_artifact(path, artifact_path=None):
            with open(path) as f:
                captured.append(json.load(f))

        with patch("mlflow.log_artifact", side_effect=_fake_log_artifact):
            log_standards(
                DEPLOYMENT_ID,
                MODEL_ID,
                [
                    {"name": "ISO 42001", "version": "1.0"},
                    {"name": "EU AI Act", "description": "EU regulation"},
                ],
            )

        assert len(captured) == 2
        assert captured[0]["name"] == "ISO 42001"
        assert captured[0]["version"] == "1.0"
        assert captured[1]["name"] == "EU AI Act"
        assert captured[1]["description"] == "EU regulation"
        assert captured[0]["deployment_id"] == DEPLOYMENT_ID
        assert captured[0]["model_id"] == MODEL_ID

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError):
            log_standards(DEPLOYMENT_ID, MODEL_ID, [])

    def test_raises_on_non_list(self):
        with pytest.raises(ValueError):
            log_standards(DEPLOYMENT_ID, MODEL_ID, {"name": "ISO 42001"})  # type: ignore[arg-type]

    def test_raises_when_item_missing_name(self):
        with pytest.raises(ValueError):
            log_standards(DEPLOYMENT_ID, MODEL_ID, [{"version": "1.0"}])


class TestLogInterface:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_interface(DEPLOYMENT_ID, MODEL_ID, "REST API")
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "interfaces"

    def test_json_fields(self):
        data = _capture_json(log_interface, DEPLOYMENT_ID, MODEL_ID, "REST API")
        assert data["interface_type"] == "REST API"
        assert data["deployment_id"] == DEPLOYMENT_ID


class TestLogMonitorLogs:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_monitor_logs(DEPLOYMENT_ID, MODEL_ID, "deployment ok")
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "deployment_logs"

    def test_json_fields(self):
        data = _capture_json(log_monitor_logs, DEPLOYMENT_ID, MODEL_ID, "deployment ok")
        assert data["deployment_id"] == DEPLOYMENT_ID
        assert data["model_id"] == MODEL_ID
        assert data["message"] == "deployment ok"
        assert data["source"] == "deployment_run.log"


class TestLogDecommissioning:
    def test_artifact_path(self):
        with patch("mlflow.log_artifact") as mock:
            log_decommissioning(
                DEPLOYMENT_ID, MODEL_ID, ["shutdown", "archive"], "End of life"
            )
            _, kwargs = mock.call_args
            assert kwargs.get("artifact_path") == "decommissioning"

    def test_json_fields(self):
        data = _capture_json(
            log_decommissioning, DEPLOYMENT_ID, MODEL_ID, ["shutdown"], "End of life"
        )
        assert data["system_name"] == ""
        assert data["decommissioning_plan"] == ""
        assert data["approvals"] == []
        assert data["data_retention_archive"] == ""
        assert data["migration"] == ""
        assert data["access_removal"] == ""
        assert data["infrastructure_shutdown"] == ""
        assert data["evidence_documentation"] == []
        assert data["audit_trail"] == ""
        assert data["decomissioning_actions"] == [
            "shutdown"
        ]  # single-m typo in library
        assert data["reason"] == "End of life"

    def test_raises_on_empty_actions(self):
        with pytest.raises(ValueError, match="decommissioning_actions"):
            log_decommissioning(DEPLOYMENT_ID, MODEL_ID, [], "reason")


# ═════════════════════════════════════════════════════════════════════════════
# data_transform — compliance map functions
# ═════════════════════════════════════════════════════════════════════════════


class TestMapAiActors:
    def _record(self, **overrides):
        base = {
            "ai_actors_id": str(uuid.uuid4()),
            "ai_providers": [{"name": "ML Team A", "role": "development"}],
            "ai_deployers": [{"name": "Ops Team", "role": "production"}],
            "auditor": "AuditFirm",
            "organization": "OrgX",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_ai_actors(self._record(), EXPERIMENT_ID)
        assert result["experiment_id"] == EXPERIMENT_ID
        assert result["ai_providers"] == [{"name": "ML Team A", "role": "development"}]
        assert result["ai_deployers"] == [{"name": "Ops Team", "role": "production"}]
        assert result["auditor"] == "AuditFirm"

    def test_missing_field_defaults_to_empty(self):
        result = map_ai_actors({}, EXPERIMENT_ID)
        assert result["ai_providers"] == []
        assert result["ai_deployers"] == []


class TestMapLabelingProcedures:
    def _record(self, **overrides):
        base = {
            "labeling_id": str(uuid.uuid4()),
            "quality_assurance_methods": ["inter-annotator agreement", "expert review"],
            "description": "NER crowdsourcing",
            "annotation_tool": "Label Studio",
            "annotators": ["ann_01", "ann_02"],
            "link": "s3://docs/ner.pdf",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_labeling_procedures(self._record(), EXPERIMENT_ID)
        assert result["experiment_id"] == EXPERIMENT_ID
        assert result["description"] == "NER crowdsourcing"
        assert result["annotation_tool"] == "Label Studio"
        assert result["annotators"] == ["ann_01", "ann_02"]
        assert result["link"] == "s3://docs/ner.pdf"
        assert result["quality_assurance_methods"] == [
            "inter-annotator agreement",
            "expert review",
        ]

    def test_annotation_tools_serialised(self):
        result = map_labeling_procedures(self._record(), EXPERIMENT_ID)
        # Stored as a raw list (not JSON string)
        assert result["annotators"] == ["ann_01", "ann_02"]


class TestMapRisk:
    def _record(self, **overrides):
        base = {
            "risk_id": str(uuid.uuid4()),
            "risk_description": "Data drift",
            "risk_type": "technical",
            "risk_level": 0.5,
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_risk(self._record(), EXPERIMENT_ID)
        assert result["experiment_id"] == EXPERIMENT_ID
        assert result["risk_type"] == "technical"
        assert abs(result["risk_level"] - 0.5) < 1e-9


class TestMapHumanOversight:
    def _record(self, **overrides):
        base = {
            "mechanism_id": str(uuid.uuid4()),
            "oversight_type": "periodic_review",
            "description": "Monthly audits",
            "implementation_details": "",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_human_oversight(self._record(), EXPERIMENT_ID)
        assert result["experiment_id"] == EXPERIMENT_ID
        assert result["oversight_type"] == "periodic_review"


class TestMapTransparencyMeasure:
    def _record(self, **overrides):
        base = {
            "measure_id": str(uuid.uuid4()),
            "measure_type": ["explainability"],
            "measure_value": ["SHAP"],
            "description": "",
            "implementation_details": "",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_transparency_measure(self._record(), EXPERIMENT_ID)
        assert result["experiment_id"] == EXPERIMENT_ID
        # Lists stored as raw Python lists
        assert result["measure_type"] == ["explainability"]


class TestMapChangeLog:
    def _record(self, **overrides):
        base = {
            "log_id": str(uuid.uuid4()),
            "change_description": "Updated threshold",
            "changed_by": "alice",
            "change_timestamp": 1700000000,
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_change_log(self._record(), RUN_ID)
        assert result["run_id"] == RUN_ID
        assert result["changed_by"] == "alice"


class TestMapDeclarationOfConformity:
    def _record(self, **overrides):
        base = {
            "declaration_id": str(uuid.uuid4()),
            "issuer": "CERTAIN Consortium",
            "version": "v2.1",
            "valid_from": 1_700_000_000.0,
            "valid_until": 1_800_000_000.0,
            "standard_references": ["ISO/IEC 42001:2023"],
            "filename": "doc.pdf",
            "file_type": "pdf",
            "mime_type": "application/pdf",
            "description": "",
            "link_to_artifacts": "",
            "file_size": 0,
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_declaration_of_conformity(self._record(), RUN_ID)
        assert result["run_id"] == RUN_ID
        assert result["filename"] == "doc.pdf"
        assert result["issuer"] == "CERTAIN Consortium"
        assert result["version"] == "v2.1"
        assert result["valid_from"] == 1_700_000_000.0
        assert result["valid_until"] == 1_800_000_000.0
        assert result["standard_references"] == ["ISO/IEC 42001:2023"]


class TestMapVisualDocumentation:
    def _record(self, **overrides):
        base = {
            "document_id": str(uuid.uuid4()),
            "stage": "evaluation",
            "generated_by": "matplotlib",
            "model_version": "1.3.0",
            "filename": "arch.png",
            "file_type": "png",
            "description": "Architecture diagram",
            "tags": ["arch"],
            "link_to_artifacts": "",
            "file_size": 0,
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_visual_documentation(self._record(), RUN_ID)
        assert result["run_id"] == RUN_ID
        assert result["filename"] == "arch.png"
        assert result["stage"] == "evaluation"
        assert result["generated_by"] == "matplotlib"
        assert result["model_version"] == "1.3.0"


class TestMapExplainableAi:
    def _record(self, **overrides):
        base = {
            "feature_id": str(uuid.uuid4()),
            "feature_name": ["age", "income"],  # singular key matches library output
            "feature_values": ["0.5", "0.5"],
            "implementation_details": "",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_explainable_ai(self._record(), RUN_ID)
        assert result["run_id"] == RUN_ID
        # Map function uses key "feature_name" (singular) and stores raw list
        assert result["feature_name"] == ["age", "income"]


class TestMapModelPackaging:
    def _record(self, **overrides):
        base = {
            "packaging_id": str(uuid.uuid4()),
            "deployment_id": DEPLOYMENT_ID,
            "model_id": MODEL_ID,
            "packaging_format": "docker",
            "dependencies": ["numpy"],
            "containerization_details": {
                "base_image": "python:3.11-slim",
                "port": 8080,
            },
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_model_packaging(self._record(), EXPERIMENT_ID, ID_MAPPING)
        assert result["deployment_id"] == DEPLOYMENT_ID
        assert result["model_id"] == MODEL_ID
        # List stored as raw Python list
        assert result["dependencies"] == ["numpy"]

    def test_containerization_details_serialised_to_json(self):
        details = {"base_image": "python:3.11-slim", "port": 8080}
        result = map_model_packaging(
            self._record(containerization_details=details), EXPERIMENT_ID, ID_MAPPING
        )
        assert json.loads(result["containerization_details"]) == details

    def test_containerization_details_string_passthrough(self):
        # backwards-compat: already-serialised string passes through unchanged
        result = map_model_packaging(
            self._record(containerization_details="already-a-string"),
            EXPERIMENT_ID,
            ID_MAPPING,
        )
        assert result["containerization_details"] == "already-a-string"


class TestMapBuildTesting:
    def _record(self, **overrides):
        base = {
            "test_id": str(uuid.uuid4()),
            "deployment_id": DEPLOYMENT_ID,
            "model_id": MODEL_ID,
            "build_status": "success",
            "build_logs": "OK",
            "test_type": "unit",
            "test_results": {"passed": 42, "failed": 0},  # dict
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_build_testing(self._record(), EXPERIMENT_ID)
        assert result["build_status"] == "success"
        assert result["deployment_id"] == DEPLOYMENT_ID
        # map function serialises the dict to a JSON string for DB storage
        assert json.loads(result["test_results"]) == {"passed": 42, "failed": 0}

    def test_string_test_results_passed_through(self):
        # Backwards compat: if already a string (e.g. read back from old artifact), pass through
        record = self._record(test_results="already a string")
        result = map_build_testing(record, EXPERIMENT_ID)
        assert result["test_results"] == "already a string"


class TestMapStandard:
    def _record(self, **overrides):
        base = {
            "standard_id": str(uuid.uuid4()),
            "deployment_id": DEPLOYMENT_ID,
            "model_id": MODEL_ID,
            "name": "ISO 42001",
            "description": "",
            "version": "1.0",
            "publication_date": "",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_standard(self._record(), EXPERIMENT_ID)
        assert result["name"] == "ISO 42001"
        assert result["deployment_id"] == DEPLOYMENT_ID


class TestMapInterface:
    def _record(self, **overrides):
        base = {
            "interface_id": str(uuid.uuid4()),
            "deployment_id": DEPLOYMENT_ID,
            "model_id": MODEL_ID,
            "interface_type": "REST API",
            "specifications": "",
            "version": "",
            "documentation_link": "",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_interface(self._record(), EXPERIMENT_ID)
        assert result["interface_type"] == "REST API"
        assert result["deployment_id"] == DEPLOYMENT_ID


class TestMapDecommissioning:
    def _record(self, **overrides):
        base = {
            "decommissioning_id": str(uuid.uuid4()),
            "deployment_id": DEPLOYMENT_ID,
            "model_id": MODEL_ID,
            "system_name": "payments-api",
            "decommissioning_plan": "Retire service after migration",
            "approvals": ["security", "ops"],
            "data_retention_archive": "archive/2026-08",
            "migration": "migrate to v2",
            "access_removal": "revoke all service accounts",
            "infrastructure_shutdown": "shutdown cluster",
            "evidence_documentation": ["runbook.pdf", "ticket-123"],
            "audit_trail": "ticket-123 -> approved",
            "decomissioning_actions": [
                "shutdown",
                "archive",
            ],  # single-m matches library/DB
            "reason": "End of life",
            "procedure_details": "",
            "decommissioning_date": "",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_decommissioning(self._record(), EXPERIMENT_ID)
        assert result["reason"] == "End of life"
        # Key uses single-m typo matching the library/DB schema
        assert result["decomissioning_actions"] == ["shutdown", "archive"]
        assert result["system_name"] == "payments-api"
        assert result["approvals"] == ["security", "ops"]
        assert result["evidence_documentation"] == ["runbook.pdf", "ticket-123"]


class TestMapMonitorLogs:
    def _record(self, **overrides):
        base = {
            "log_id": str(uuid.uuid4()),
            "deployment_id": DEPLOYMENT_ID,
            "model_id": MODEL_ID,
            "message": "[2026-08-18T12:00:00] Deployment started",
            "source": "deployment_run.log",
        }
        base.update(overrides)
        return base

    def test_maps_all_fields(self):
        result = map_monitor_logs(self._record(), EXPERIMENT_ID, RUN_ID)
        assert result["deployment_id"] == DEPLOYMENT_ID
        assert result["experiment_id"] == EXPERIMENT_ID
        assert result["model_id"] == MODEL_ID
        assert result["message"].startswith("[")
