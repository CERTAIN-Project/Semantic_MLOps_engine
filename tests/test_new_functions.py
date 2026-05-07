"""
Tests for the new certain_library logging functions and data_api map functions.

Covers:
  certain_library/train_monitor/log_checkpoints.py  → log_checkpoint
  certain_library/train_monitor/log_weights.py       → log_weight_distribution
  certain_library/train_monitor/log_examples.py      → log_examples
  certain_library/log_basic/log_run_logs.py          → log_run_logs
  data_api/misc/data_transform.py                    → map_run_code,
                                                       map_checkpoints,
                                                       map_weight_distribution,
                                                       map_examples,
                                                       map_run_logs
"""

import os
import shutil
import uuid
import time
import tempfile

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# ── certain_library imports ──────────────────────────────────────────────────
from certain_library.train_monitor.log_checkpoints import log_checkpoint
from certain_library.train_monitor.log_weights import log_weight_distribution
from certain_library.train_monitor.log_examples import log_examples
from certain_library.log_basic.log_run_logs import log_run_logs

# ── data_api map-function imports ────────────────────────────────────────────
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "data_api"),
)
from misc.data_transform import (  # noqa: E402
    map_run_code,
    map_checkpoints,
    map_weight_distribution,
    map_examples,
    map_run_logs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = str(uuid.uuid4())
MODEL_ID = str(uuid.uuid4())
DATA_ID = str(uuid.uuid4())

ID_MAPPING = {
    RUN_ID: {
        "model_id": MODEL_ID,
        "data_id": DATA_ID,
        "deployment_id": str(uuid.uuid4()),
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# certain_library — log_checkpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestLogCheckpoint:
    def test_logs_artifact_with_correct_folder(self):
        with patch("mlflow.log_artifact") as mock_artifact:
            log_checkpoint("epoch_5", "/tmp/model_epoch5.pt")
            mock_artifact.assert_called_once()
            _, kwargs = mock_artifact.call_args
            assert kwargs.get("artifact_path") == "checkpoints"

    def test_csv_contains_expected_columns(self, tmp_path):
        persistent = str(tmp_path / "ckpt.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_checkpoint("epoch_1", "/model.pt", checkpoint_id="ckpt-001")

        df = pd.read_csv(persistent)
        assert set(
            ["checkpoint_id", "checkpoint_name", "checkpoint_location", "creation_time"]
        ).issubset(df.columns)
        assert df.iloc[0]["checkpoint_name"] == "epoch_1"
        assert df.iloc[0]["checkpoint_id"] == "ckpt-001"

    def test_raises_on_empty_name(self):
        with pytest.raises(ValueError, match="checkpoint_name"):
            log_checkpoint("", "/some/path")

    def test_raises_on_empty_location(self):
        with pytest.raises(ValueError, match="checkpoint_location"):
            log_checkpoint("epoch_1", "")

    def test_auto_generates_uuid_when_id_not_provided(self, tmp_path):
        persistent = str(tmp_path / "ckpt2.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_checkpoint("epoch_2", "/weights.pt")

        df = pd.read_csv(persistent)
        cid = df.iloc[0]["checkpoint_id"]
        # Must be a valid UUID string
        uuid.UUID(str(cid))


# ─────────────────────────────────────────────────────────────────────────────
# certain_library — log_weight_distribution
# ─────────────────────────────────────────────────────────────────────────────


class TestLogWeightDistribution:
    def _make_sklearn_model(self, coef):
        model = MagicMock()
        # Remove Keras / PyTorch attributes so the sklearn branch is hit
        del model.weights
        type(model).__mro__ = [object]
        model.coef_ = coef
        # Ensure feature_importances_ is absent
        del model.feature_importances_
        return model

    def test_sklearn_model_logged(self, tmp_path):
        import numpy as np

        coef = np.array([1.0, 2.0, 3.0])
        model = MagicMock(spec=[])  # bare mock — no attributes by default
        model.coef_ = coef

        persistent = str(tmp_path / "weights.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_weight_distribution(model, step=1, stage="train")

        df = pd.read_csv(persistent)
        assert "layer_name" in df.columns
        assert "mean" in df.columns
        assert "std" in df.columns
        assert len(df) == 1
        assert abs(df.iloc[0]["mean"] - float(coef.mean())) < 1e-6

    def test_unknown_model_gets_placeholder_row(self, tmp_path):
        model = object()  # Has none of the known attributes
        persistent = str(tmp_path / "weights_unk.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_weight_distribution(model)

        df = pd.read_csv(persistent)
        assert len(df) == 1
        assert df.iloc[0]["layer_name"] == "unknown_layer"

    def test_artifact_path_is_weight_distribution(self):
        model = object()
        with patch("mlflow.log_artifact") as mock_artifact:
            log_weight_distribution(model)
            _, kwargs = mock_artifact.call_args
            assert kwargs.get("artifact_path") == "weight_distribution"

    def test_stage_written_to_csv(self, tmp_path):
        model = object()
        persistent = str(tmp_path / "weights_val.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_weight_distribution(model, stage="validation")

        df = pd.read_csv(persistent)
        assert df.iloc[0]["stage"] == "validation"


# ─────────────────────────────────────────────────────────────────────────────
# certain_library — log_examples
# ─────────────────────────────────────────────────────────────────────────────


class TestLogExamples:
    def test_basic_logging(self):
        with patch("mlflow.log_artifact") as mock_artifact:
            log_examples([1, 2, 3], [0.9, 1.8, 3.1], [1, 2, 3])
            mock_artifact.assert_called_once()
            _, kwargs = mock_artifact.call_args
            assert kwargs.get("artifact_path") == "examples"

    def test_csv_has_correct_rows(self, tmp_path):
        persistent = str(tmp_path / "examples.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_examples(["a", "b"], ["x", "y"], ["x", "y"], step=5, stage="eval")

        df = pd.read_csv(persistent)
        assert len(df) == 2
        assert list(df["input"]) == ["a", "b"]
        assert list(df["stage"]) == ["eval", "eval"]
        assert list(df["step"]) == [5, 5]

    def test_raises_on_empty_inputs(self):
        with pytest.raises(ValueError, match="inputs list must not be empty"):
            log_examples([], [], [])

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            log_examples([1, 2], [1], [1, 2])

    def test_values_stringified(self, tmp_path):
        persistent = str(tmp_path / "examples_str.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_examples([[1, 2, 3]], [0.5], [1])

        df = pd.read_csv(persistent)
        assert df.iloc[0]["input"] == "[1, 2, 3]"


# ─────────────────────────────────────────────────────────────────────────────
# certain_library — log_run_logs
# ─────────────────────────────────────────────────────────────────────────────


class TestLogRunLogs:
    def test_basic_logging(self):
        with patch("mlflow.log_artifact") as mock_artifact:
            log_run_logs(["line 1", "line 2"])
            mock_artifact.assert_called_once()
            _, kwargs = mock_artifact.call_args
            assert kwargs.get("artifact_path") == "run_logs"

    def test_csv_columns_present(self, tmp_path):
        persistent = str(tmp_path / "logs.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_run_logs(["Error occurred", "Training done"], log_type="stderr")

        df = pd.read_csv(persistent)
        assert set(["log_id", "log_type", "log_message", "log_creation_time"]).issubset(
            df.columns
        )
        assert len(df) == 2
        assert all(df["log_type"] == "stderr")
        assert list(df["log_message"]) == ["Error occurred", "Training done"]

    def test_each_row_has_unique_uuid(self, tmp_path):
        persistent = str(tmp_path / "logs_uuid.csv")

        def fake_log(path, artifact_path=None):
            shutil.copy(path, persistent)

        with patch("mlflow.log_artifact", side_effect=fake_log):
            log_run_logs(["a", "b", "c"])

        df = pd.read_csv(persistent)
        ids = df["log_id"].tolist()
        assert len(set(ids)) == 3  # all unique

    def test_raises_on_empty_messages(self):
        with pytest.raises(ValueError, match="messages list must not be empty"):
            log_run_logs([])


# ─────────────────────────────────────────────────────────────────────────────
# data_transform — map_run_code
# ─────────────────────────────────────────────────────────────────────────────


class TestMapRunCode:
    def _make_tags_df(self, run_id, commit="abc123", source="train.py"):
        rows = []
        if commit:
            rows.append(
                {"run_uuid": run_id, "key": "mlflow.source.git.commit", "value": commit}
            )
        if source:
            rows.append(
                {"run_uuid": run_id, "key": "mlflow.source.name", "value": source}
            )
        return pd.DataFrame(rows)

    def test_returns_correct_fields(self):
        tags_df = self._make_tags_df(RUN_ID)
        result = map_run_code(tags_df, RUN_ID)
        assert result["run_id"] == RUN_ID
        assert result["git_commit_hash"] == "abc123"
        assert result["name"] == "train.py"

    def test_returns_empty_dict_when_no_tags(self):
        tags_df = pd.DataFrame(columns=["run_uuid", "key", "value"])
        result = map_run_code(tags_df, RUN_ID)
        assert result == {}

    def test_returns_empty_dict_when_run_not_in_tags(self):
        other_run = str(uuid.uuid4())
        tags_df = self._make_tags_df(other_run)
        result = map_run_code(tags_df, RUN_ID)
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# data_transform — map_checkpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestMapCheckpoints:
    def _make_row(self, **overrides):
        base = {
            "run_id": RUN_ID,
            "checkpoint_id": "ckpt-1",
            "checkpoint_name": "epoch_10",
            "checkpoint_location": "/tmp/ckpt.pt",
            "creation_time": int(time.time()),
        }
        base.update(overrides)
        return pd.Series(base)

    def test_maps_all_fields(self):
        row = self._make_row()
        result = map_checkpoints(row, ID_MAPPING)
        assert result["run_id"] == RUN_ID
        assert result["model_id"] == MODEL_ID
        assert result["checkpoint_name"] == "epoch_10"
        assert result["checkpoint_id"] == "ckpt-1"

    def test_missing_run_in_mapping_gives_empty_model_id(self):
        row = self._make_row(run_id="unknown-run")
        result = map_checkpoints(row, ID_MAPPING)
        assert result["model_id"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# data_transform — map_weight_distribution
# ─────────────────────────────────────────────────────────────────────────────


class TestMapWeightDistribution:
    def _make_row(self, **overrides):
        base = {
            "run_id": RUN_ID,
            "layer_name": "fc1.weight",
            "mean": 0.05,
            "std": 0.02,
            "step": 10,
            "stage": "train",
            "is_NaN": False,
            "timestamp": int(time.time()),
        }
        base.update(overrides)
        return pd.Series(base)

    def test_maps_all_fields(self):
        row = self._make_row()
        result = map_weight_distribution(row, ID_MAPPING)
        assert result["run_id"] == RUN_ID
        assert result["model_id"] == MODEL_ID
        assert result["layer_name"] == "fc1.weight"
        assert abs(result["mean"] - 0.05) < 1e-9
        assert result["step"] == 10

    def test_nan_mean_becomes_zero_and_is_nan_true(self):
        import math

        row = self._make_row(mean=float("nan"), std=float("nan"))
        result = map_weight_distribution(row, ID_MAPPING)
        assert result["mean"] == 0.0
        assert result["is_NaN"] is True


# ─────────────────────────────────────────────────────────────────────────────
# data_transform — map_examples
# ─────────────────────────────────────────────────────────────────────────────


class TestMapExamples:
    def _make_row(self, **overrides):
        base = {
            "run_id": RUN_ID,
            "input": "[1,2,3]",
            "prediction": "0.9",
            "ground_truth": "1.0",
            "step": 0,
            "stage": "inference",
            "timestamp": int(time.time()),
        }
        base.update(overrides)
        return pd.Series(base)

    def test_maps_all_fields(self):
        row = self._make_row()
        result = map_examples(row, ID_MAPPING)
        assert result["run_id"] == RUN_ID
        assert result["model_id"] == MODEL_ID
        assert result["input"] == "[1,2,3]"
        assert result["prediction"] == "0.9"
        assert result["ground_truth"] == "1.0"

    def test_stage_and_step_preserved(self):
        row = self._make_row(step=7, stage="eval")
        result = map_examples(row, ID_MAPPING)
        assert result["step"] == 7
        assert result["stage"] == "eval"


# ─────────────────────────────────────────────────────────────────────────────
# data_transform — map_run_logs
# ─────────────────────────────────────────────────────────────────────────────


class TestMapRunLogs:
    def _make_row(self, **overrides):
        base = {
            "log_id": str(uuid.uuid4()),
            "log_type": "stdout",
            "log_message": "Training started",
            "log_creation_time": int(time.time()),
        }
        base.update(overrides)
        return pd.Series(base)

    def test_maps_all_fields(self):
        row = self._make_row()
        result = map_run_logs(row, RUN_ID)
        assert result["run_id"] == RUN_ID
        assert result["log_type"] == "stdout"
        assert result["log_message"] == "Training started"
        assert "log_creation_time" in result

    def test_defaults_applied_when_fields_missing(self):
        row = pd.Series({})
        result = map_run_logs(row, RUN_ID)
        assert result["run_id"] == RUN_ID
        assert result["log_type"] == "stdout"
        assert result["log_message"] == ""
