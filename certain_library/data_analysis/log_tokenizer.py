"""Utilities for logging LLM tokenizer configuration and tokenization statistics."""

from certain_library.tracking.tracker import tracker

import json
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Allowed keys for each artifact type
# ---------------------------------------------------------------------------

_TOKENIZER_CONFIG_ALLOWED_KEYS = {
    "tokenizer_id",
    "tokenizer_type",
    "model_name_or_path",
    "vocab_size",
    "max_length",
    "padding",
    "truncation",
    "stride",
    "special_tokens",
}

_TOKENIZATION_STATS_ALLOWED_KEYS = {
    "stats_id",
    "split",
    "total_sequences",
    "total_tokens",
    "avg_token_length",
    "min_token_length",
    "max_token_length",
    "truncation_rate",
    "padding_rate",
    "oov_rate",
}


def log_tokenizer_config(
    tokenizer_type: str,
    model_name_or_path: Optional[str] = None,
    vocab_size: Optional[int] = None,
    max_length: Optional[int] = None,
    padding: Optional[Union[str, bool]] = None,
    truncation: Optional[Union[str, bool]] = None,
    stride: Optional[int] = None,
    special_tokens: Optional[Dict[str, str]] = None,
    tokenizer_id: Optional[str] = None,
) -> None:
    """Log tokenizer configuration as a JSON artifact to MLflow.

    Parameters
    ----------
    tokenizer_type:
        Name / class of the tokenizer (e.g. ``"BertTokenizerFast"``).
    model_name_or_path:
        HuggingFace model identifier or local path used to load the tokenizer.
    vocab_size:
        Size of the vocabulary.
    max_length:
        Maximum sequence length the tokenizer will produce.
    padding:
        Padding strategy (``True``/``False``/``"max_length"``/``"longest"``/``"do_not_pad"``).
    truncation:
        Truncation strategy (``True``/``False``/``"longest_first"``/``"only_first"``/
        ``"only_second"``/``"do_not_truncate"``).
    stride:
        Stride used when splitting long sequences (relevant for QA pipelines).
    special_tokens:
        Dictionary of special tokens, e.g. ``{"[PAD]": 0, "[CLS]": 101}``.
    tokenizer_id:
        Optional unique identifier. A UUID is generated when not provided.
    """
    record: Dict[str, Any] = {
        "tokenizer_id": tokenizer_id or str(uuid.uuid4()),
        "tokenizer_type": tokenizer_type,
    }

    if model_name_or_path is not None:
        record["model_name_or_path"] = model_name_or_path
    if vocab_size is not None:
        record["vocab_size"] = vocab_size
    if max_length is not None:
        record["max_length"] = max_length
    if padding is not None:
        record["padding"] = padding
    if truncation is not None:
        record["truncation"] = truncation
    if stride is not None:
        record["stride"] = stride
    if special_tokens is not None:
        record["special_tokens"] = special_tokens

    unknown = set(record.keys()) - _TOKENIZER_CONFIG_ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"log_tokenizer_config: unknown keys {unknown}. "
            f"Allowed keys: {_TOKENIZER_CONFIG_ALLOWED_KEYS}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(
            tmp_dir, f"tokenizer_config_{record['tokenizer_id']}.json"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        tracker.log_artifact(file_path, artifact_path="tokenizer_config")


def log_tokenization_stats(stats: List[dict]) -> None:
    """Log per-split tokenization statistics as JSON artifacts to MLflow.

    Each element of *stats* represents one data split (e.g. ``"train"``,
    ``"validation"``, ``"test"``).

    Parameters
    ----------
    stats:
        Each dict must contain at least a ``"split"`` key and
        may contain any of the following keys:

        * ``stats_id`` – unique identifier (auto-generated if absent)
        * ``split`` – dataset split name, e.g. ``"train"``
        * ``total_sequences`` – total number of sequences in the split
        * ``total_tokens`` – total number of tokens across all sequences
        * ``avg_token_length`` – mean number of tokens per sequence
        * ``min_token_length`` – minimum token length observed
        * ``max_token_length`` – maximum token length observed
        * ``truncation_rate`` – fraction of sequences truncated (0–1)
        * ``padding_rate`` – fraction of sequences padded (0–1)
        * ``oov_rate`` – out-of-vocabulary rate (0–1)

    Raises
    ------
    ValueError
        If any dict contains keys not in the allowed set.
    """
    if not stats:
        raise ValueError("log_tokenization_stats: stats list must not be empty.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for item in stats:
            record = dict(item)
            unknown = set(record.keys()) - _TOKENIZATION_STATS_ALLOWED_KEYS
            if unknown:
                raise ValueError(
                    f"log_tokenization_stats: unknown keys {unknown}. "
                    f"Allowed keys: {_TOKENIZATION_STATS_ALLOWED_KEYS}"
                )
            if "stats_id" not in record:
                record["stats_id"] = str(uuid.uuid4())

            split_label = record.get("split", record["stats_id"])
            file_path = os.path.join(tmp_dir, f"tokenization_stats_{split_label}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            tracker.log_artifact(file_path, artifact_path="tokenization_stats")
