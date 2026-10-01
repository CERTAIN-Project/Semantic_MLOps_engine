"""Demographic bias detection helpers for clustered session datasets."""

from __future__ import annotations

import html
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    KMeans = None  # type: ignore[assignment]
    TruncatedSVD = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    silhouette_score = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

from certain_library.tracking.tracker import tracker


METADATA_FIELDS = [
    "FirstName",
    "LastName",
    "Age",
    "DateOfBirth",
    "Nationality",
    "Background",
    "Opinion",
    "AnswerQuality",
    "BiasGroup",
]

PUBLIC_METADATA_FIELDS = [
    "FirstName",
    "LastName",
    "Age",
    "DateOfBirth",
    "Nationality",
]

INTERNAL_METADATA_FIELDS = [
    "Background",
    "Opinion",
    "AnswerQuality",
    "BiasGroup",
    "WrongAnswerPattern",
]

FALLBACK_CLUSTER_STOPWORDS = {
    "about",
    "access",
    "age",
    "answer",
    "background",
    "bias",
    "bucket",
    "correct",
    "country",
    "employees",
    "explicitly",
    "final",
    "group",
    "label",
    "marker",
    "measure",
    "metadata",
    "opinion",
    "policy",
    "quality",
    "question",
    "remote",
    "respondent",
    "response",
    "safeguard",
    "safeguards",
    "treated",
    "work",
}


def _resolve_logger(logger: Any) -> Any:
    return logger if logger is not None else tracker


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return df.copy()

# Define the age bucket helper function
# You need to change the AgeBucket you should change the following function: _age_bucket
def _age_bucket(age: Any) -> str:
    if age is None:
        return "unknown"
    try:
        age_value = int(float(age))
    except (TypeError, ValueError):
        return "unknown"

    if age_value < 30:
        return "18-29"
    if age_value < 45:
        return "30-44"
    if age_value < 60:
        return "45-59"
    return "60+"


def compute_age_bucket(age: Any) -> str:
    """Return the demographic age bucket used by the bias analysis."""

    return _age_bucket(age)


def _normalise_distribution(values: Iterable[Any]) -> dict[str, float]:
    filtered = [_safe_string(value) or "unknown" for value in values]
    total = len(filtered)
    if total == 0:
        return {}
    counts = Counter(filtered)
    return {key: value / total for key, value in sorted(counts.items())}


def _total_variation_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)


def _tokenise(text: str) -> list[str]:
    normalized = re.sub(r"^\[[^\]]+\]\s*", "", text.lower())
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_+-]{3,}", normalized)
        if token not in FALLBACK_CLUSTER_STOPWORDS
    ]


def _stable_jitter(value: Any, scale: float = 0.035) -> float:
    text = _safe_string(value)
    if not text:
        return 0.0
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return (((checksum % 997) / 996.0) - 0.5) * 2.0 * scale


def _cluster_axis(cluster_ids: Iterable[Any]) -> np.ndarray:
    labels = list(cluster_ids)
    ordered = {cluster_id: index for index, cluster_id in enumerate(sorted(set(labels)))}
    if len(ordered) <= 1:
        return np.zeros(len(labels), dtype=float)
    center = (len(ordered) - 1) / 2.0
    return np.array([(ordered[cluster_id] - center) / center for cluster_id in labels], dtype=float)


def _normalise_series(values: Iterable[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").fillna(0.0)
    array = series.to_numpy(dtype=float)
    if len(array) == 0:
        return array
    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(maximum, minimum):
        return np.zeros(len(array), dtype=float)
    return ((array - minimum) / (maximum - minimum)) * 2.0 - 1.0


def _safe_mean(series: pd.Series) -> Optional[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _normalize_country(value: Any) -> str:
    return _safe_string(value).strip().lower()


def _append_random_category(
    df: pd.DataFrame,
    column_name: str,
    choices: Sequence[str],
    seed: int | None = None,
) -> pd.DataFrame:
    frame = _ensure_dataframe(df)
    if column_name in frame.columns:
        return frame

    normalized_choices = [_safe_string(choice) for choice in choices if _safe_string(choice)]
    if not normalized_choices:
        return frame

    rng = np.random.default_rng(seed)
    frame[column_name] = rng.choice(normalized_choices, size=len(frame), replace=True)
    return frame


def _filter_demographic_subset(
    df: pd.DataFrame,
    age_range: tuple[float | int, float | int] | None = None,
    countries: Sequence[str] | None = None,
    areas: Sequence[str] | None = None,
    append_random_areas: bool = False,
    area_column: str = "Area",
    area_choices: Sequence[str] | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    frame = _ensure_dataframe(df)
    if append_random_areas:
        frame = _append_random_category(
            frame,
            column_name=area_column,
            choices=area_choices or ["North", "South", "East", "West"],
            seed=random_seed,
        )

    filtered = frame

    if age_range is not None:
        lower, upper = age_range
        age_values = pd.to_numeric(filtered["Age"], errors="coerce")
        filtered = filtered.loc[age_values.between(float(lower), float(upper), inclusive="both")]

    if countries:
        normalized_countries = {
            _normalize_country(country) for country in countries if _normalize_country(country)
        }
        if normalized_countries:
            country_values = filtered["Nationality"].map(_normalize_country)
            filtered = filtered.loc[country_values.isin(normalized_countries)]

    if areas:
        if area_column not in filtered.columns:
            raise KeyError(
                f"Column '{area_column}' is missing. Set append_random_areas=True or provide an existing area column."
            )
        normalized_areas = {_normalize_country(area) for area in areas if _normalize_country(area)}
        if normalized_areas:
            area_values = filtered[area_column].map(_normalize_country)
            filtered = filtered.loc[area_values.isin(normalized_areas)]

    return filtered.reset_index(drop=True)


def _flatten_dataset(
    payload: dict[str, Any],
    dataset_type: str,
    batch_id: int,
    internal_annotations: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    topic_title = payload.get("TopicTitle", "")
    topic_summary = payload.get("TopicSummary", "")

    for record in payload.get("Dataset", []):
        metadata = record.get("Metadata") or {}
        record_id = record.get("Id")
        internal_metadata = (internal_annotations or {}).get(str(record_id), {})
        normalized_metadata = {field: metadata.get(field) for field in METADATA_FIELDS}

        for field in INTERNAL_METADATA_FIELDS:
            if normalized_metadata.get(field) in (None, ""):
                normalized_metadata[field] = internal_metadata.get(field)

        if normalized_metadata.get("Nationality") in (None, ""):
            normalized_metadata["Nationality"] = metadata.get("Country") or metadata.get("Country/Nationality")

        missing_fields = [field for field in PUBLIC_METADATA_FIELDS if metadata.get(field) in (None, "")]
        text_value = _safe_string(record.get("Text"))
        row = {
            "batch_id": batch_id,
            "dataset_type": dataset_type,
            "topic_title": topic_title,
            "topic_summary": topic_summary,
            "record_id": record_id,
            "Text": text_value,
            "text_length": len(text_value),
            "metadata_missing_count": len(missing_fields),
            "metadata_completeness": 1.0 - (len(missing_fields) / len(PUBLIC_METADATA_FIELDS)),
            "is_anonymous": int(len(missing_fields) == len(PUBLIC_METADATA_FIELDS)),
        }

        for field in METADATA_FIELDS:
            row[field] = normalized_metadata.get(field)

        row["AgeBucket"] = _age_bucket(normalized_metadata.get("Age"))
        rows.append(row)

    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_internal_annotations(session_dir: Path, batch_id: int) -> dict[str, dict[str, Any]]:
    path = session_dir / f"scenario_annotations_batch_{batch_id}.json"
    if not path.exists():
        return {"clean": {}, "dirty": {}}
    payload = _load_json(path)

    def by_id(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            str(record.get("Id")): record.get("InternalMetadata") or {}
            for record in records
            if record.get("Id") is not None
        }

    return {
        "clean": by_id(payload.get("Clean", [])),
        "dirty": by_id(payload.get("Dirty", [])),
    }


def _select_cluster_count(matrix: Any, record_count: int, max_clusters: int) -> tuple[int, float | None]:
    if record_count < 4:
        return 1, None

    upper_bound = min(max_clusters, record_count - 1)
    if upper_bound < 2:
        return 1, None

    best_k = 1
    best_score = -1.0

    if KMeans is None or silhouette_score is None:
        return 1, None

    for cluster_count in range(2, upper_bound + 1):
        model = KMeans(n_clusters=cluster_count, n_init=20, random_state=42)
        labels = model.fit_predict(matrix)
        if len(set(labels.tolist())) < 2:
            continue
        try:
            score = silhouette_score(matrix, labels, metric="cosine")
        except ValueError:
            continue
        if score > best_score:
            best_score = float(score)
            best_k = cluster_count

    if best_k == 1:
        return 1, None
    return best_k, float(best_score)


def _top_terms_per_cluster(
    matrix: Any,
    assignments: np.ndarray,
    feature_names: np.ndarray,
    top_n: int = 8,
) -> dict[int, list[str]]:
    top_terms: dict[int, list[str]] = {}
    for cluster_id in sorted(set(assignments.tolist())):
        member_indices = np.where(assignments == cluster_id)[0]
        if len(member_indices) == 0:
            top_terms[cluster_id] = []
            continue
        cluster_matrix = matrix[member_indices]
        weights = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        ranked_indices = weights.argsort()[::-1][:top_n]
        top_terms[cluster_id] = [str(feature_names[index]) for index in ranked_indices if weights[index] > 0]
    return top_terms


def _cluster_dataset_without_sklearn(clustered: pd.DataFrame, max_clusters: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    token_sets = [set(_tokenise(text)) for text in clustered["Text"].fillna("").tolist()]
    cluster_tokens: list[Counter[str]] = []
    assignments: list[int] = []

    for token_set in token_sets:
        if not cluster_tokens:
            cluster_tokens.append(Counter(token_set))
            assignments.append(0)
            continue

        best_cluster = 0
        best_score = -1.0
        for cluster_id, token_counter in enumerate(cluster_tokens):
            reference = set(token_counter.keys())
            union = token_set | reference
            similarity = 1.0 if not union else len(token_set & reference) / len(union)
            if similarity > best_score:
                best_score = similarity
                best_cluster = cluster_id

        if best_score < 0.36 and len(cluster_tokens) < max(1, max_clusters):
            cluster_tokens.append(Counter(token_set))
            assignments.append(len(cluster_tokens) - 1)
        else:
            cluster_tokens[best_cluster].update(token_set)
            assignments.append(best_cluster)

    clustered["cluster_id"] = assignments
    top_terms = {
        str(cluster_id): [term for term, _count in counter.most_common(8)]
        for cluster_id, counter in enumerate(cluster_tokens)
    }

    return clustered, {
        "cluster_count": int(len(cluster_tokens)),
        "silhouette_score": None,
        "cluster_top_terms": top_terms,
    }


def _cluster_dataset(df: pd.DataFrame, max_clusters: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    clustered = _ensure_dataframe(df)
    if clustered.empty:
        raise ValueError("Cannot cluster an empty dataset.")

    if not _SKLEARN_AVAILABLE or KMeans is None or TfidfVectorizer is None:
        return _cluster_dataset_without_sklearn(clustered, max_clusters=max_clusters)

    texts = clustered["Text"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    if matrix.shape[1] == 0 or len(texts) < 2:
        clustered["cluster_id"] = 0
        return clustered, {"cluster_count": 1, "silhouette_score": None, "cluster_top_terms": {"0": []}}

    best_k, best_score = _select_cluster_count(matrix, len(clustered), max_clusters)
    if best_k == 1:
        assignments = np.zeros(len(clustered), dtype=int)
    else:
        assert KMeans is not None
        model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
        assignments = model.fit_predict(matrix)

    clustered["cluster_id"] = assignments.astype(int)
    top_terms = _top_terms_per_cluster(matrix, assignments, feature_names)

    return clustered, {
        "cluster_count": int(len(set(assignments.tolist()))),
        "silhouette_score": None if best_score is None else float(best_score),
        "cluster_top_terms": {str(key): value for key, value in top_terms.items()},
    }


def _build_cluster_summary(clustered_df: pd.DataFrame, clustering_meta: dict[str, Any]) -> dict[str, Any]:
    overall_nationality = _normalise_distribution(clustered_df["Nationality"])
    overall_background = _normalise_distribution(clustered_df["Background"])
    overall_age_bucket = _normalise_distribution(clustered_df["AgeBucket"])

    clusters: list[dict[str, Any]] = []
    for cluster_id, cluster_df in clustered_df.groupby("cluster_id", sort=True):
        cluster_key = int(cast(Any, cluster_id))
        nationality_distribution = _normalise_distribution(cluster_df["Nationality"])
        background_distribution = _normalise_distribution(cluster_df["Background"])
        age_bucket_distribution = _normalise_distribution(cluster_df["AgeBucket"])

        clusters.append(
            {
                "cluster_id": cluster_key,
                "records": int(len(cluster_df)),
                "share_of_dataset": float(len(cluster_df) / len(clustered_df)),
                "anonymous_rate": float(cluster_df["is_anonymous"].mean()),
                "metadata_missing_rate": float(
                    cluster_df["metadata_missing_count"].sum()
                    / (len(cluster_df) * len(PUBLIC_METADATA_FIELDS))
                ),
                "avg_text_length": float(cluster_df["text_length"].mean()),
                "avg_age": _safe_mean(cluster_df["Age"]),
                "nationality_distribution": nationality_distribution,
                "background_distribution": background_distribution,
                "age_bucket_distribution": age_bucket_distribution,
                "opinion_distribution": _normalise_distribution(cluster_df["Opinion"]),
                "answer_quality_distribution": _normalise_distribution(cluster_df["AnswerQuality"]),
                "bias_group_distribution": _normalise_distribution(cluster_df["BiasGroup"]),
                "nationality_tvd_vs_dataset": float(
                    _total_variation_distance(nationality_distribution, overall_nationality)
                ),
                "background_tvd_vs_dataset": float(
                    _total_variation_distance(background_distribution, overall_background)
                ),
                "age_bucket_tvd_vs_dataset": float(
                    _total_variation_distance(age_bucket_distribution, overall_age_bucket)
                ),
                "top_terms": clustering_meta["cluster_top_terms"].get(str(cluster_id), []),
            }
        )

    return {
        "cluster_count": clustering_meta["cluster_count"],
        "silhouette_score": clustering_meta["silhouette_score"],
        "clusters": clusters,
    }


def _representation_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "nationality_distribution": _normalise_distribution(df["Nationality"]),
        "background_distribution": _normalise_distribution(df["Background"]),
        "age_bucket_distribution": _normalise_distribution(df["AgeBucket"]),
        "opinion_distribution": _normalise_distribution(df["Opinion"]),
        "answer_quality_distribution": _normalise_distribution(df["AnswerQuality"]),
        "bias_group_distribution": _normalise_distribution(df["BiasGroup"]),
    }


def _build_bias_snapshot(clustered_df: pd.DataFrame, cluster_summary: dict[str, Any]) -> dict[str, Any]:
    total_records = len(clustered_df)
    total_metadata_slots = total_records * len(PUBLIC_METADATA_FIELDS)
    field_null_rates = {field: float(clustered_df[field].isna().mean()) for field in PUBLIC_METADATA_FIELDS}
    clusters = cluster_summary["clusters"]
    weighted_nationality_tvd = sum(cluster["share_of_dataset"] * cluster["nationality_tvd_vs_dataset"] for cluster in clusters)
    weighted_background_tvd = sum(cluster["share_of_dataset"] * cluster["background_tvd_vs_dataset"] for cluster in clusters)
    weighted_age_bucket_tvd = sum(cluster["share_of_dataset"] * cluster["age_bucket_tvd_vs_dataset"] for cluster in clusters)
    largest_cluster_share = max((cluster["share_of_dataset"] for cluster in clusters), default=1.0)

    return {
        "records": total_records,
        "cluster_count": cluster_summary["cluster_count"],
        "anonymous_rate": float(clustered_df["is_anonymous"].mean()),
        "metadata_missing_rate": float(clustered_df["metadata_missing_count"].sum() / total_metadata_slots),
        "field_null_rates": field_null_rates,
        "representation": _representation_summary(clustered_df),
        "cluster_bias_signals": {
            "weighted_nationality_tvd": float(weighted_nationality_tvd),
            "weighted_background_tvd": float(weighted_background_tvd),
            "weighted_age_bucket_tvd": float(weighted_age_bucket_tvd),
            "largest_cluster_share": float(largest_cluster_share),
            "silhouette_score": cluster_summary["silhouette_score"],
        },
    }


def _compare_clean_dirty(
    clean_df: pd.DataFrame,
    dirty_df: pd.DataFrame,
    clean_bias: dict[str, Any],
    dirty_bias: dict[str, Any],
) -> dict[str, Any]:
    clean_representation = clean_bias["representation"]
    dirty_representation = dirty_bias["representation"]

    return {
        "clean_records": int(len(clean_df)),
        "dirty_records": int(len(dirty_df)),
        "delta_anonymous_rate": float(dirty_bias["anonymous_rate"] - clean_bias["anonymous_rate"]),
        "delta_metadata_missing_rate": float(
            dirty_bias["metadata_missing_rate"] - clean_bias["metadata_missing_rate"]
        ),
        "delta_weighted_nationality_tvd": float(
            dirty_bias["cluster_bias_signals"]["weighted_nationality_tvd"]
            - clean_bias["cluster_bias_signals"]["weighted_nationality_tvd"]
        ),
        "delta_weighted_background_tvd": float(
            dirty_bias["cluster_bias_signals"]["weighted_background_tvd"]
            - clean_bias["cluster_bias_signals"]["weighted_background_tvd"]
        ),
        "delta_weighted_age_bucket_tvd": float(
            dirty_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"]
            - clean_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"]
        ),
        "representation_shift": {
            "nationality_tvd": float(
                _total_variation_distance(
                    clean_representation["nationality_distribution"],
                    dirty_representation["nationality_distribution"],
                )
            ),
            "background_tvd": float(
                _total_variation_distance(
                    clean_representation["background_distribution"],
                    dirty_representation["background_distribution"],
                )
            ),
            "age_bucket_tvd": float(
                _total_variation_distance(
                    clean_representation["age_bucket_distribution"],
                    dirty_representation["age_bucket_distribution"],
                )
            ),
            "opinion_tvd": float(
                _total_variation_distance(
                    clean_representation["opinion_distribution"],
                    dirty_representation["opinion_distribution"],
                )
            ),
            "answer_quality_tvd": float(
                _total_variation_distance(
                    clean_representation["answer_quality_distribution"],
                    dirty_representation["answer_quality_distribution"],
                )
            ),
            "bias_group_tvd": float(
                _total_variation_distance(
                    clean_representation["bias_group_distribution"],
                    dirty_representation["bias_group_distribution"],
                )
            ),
        },
    }


def _bias_diagnosis(
    bias_snapshot: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    signals = bias_snapshot.get("cluster_bias_signals", {})
    weighted_signals = {
        "nationality cluster skew": float(signals.get("weighted_nationality_tvd", 0.0) or 0.0),
        "background cluster skew": float(signals.get("weighted_background_tvd", 0.0) or 0.0),
        "age cluster skew": float(signals.get("weighted_age_bucket_tvd", 0.0) or 0.0),
    }
    missing_rate = float(bias_snapshot.get("metadata_missing_rate", 0.0) or 0.0)
    cluster_count = int(bias_snapshot.get("cluster_count", 0) or 0)
    largest_cluster = float(signals.get("largest_cluster_share", 0.0) or 0.0)
    cluster_concentration = largest_cluster if cluster_count > 1 else 0.0

    rows: list[tuple[str, float, str]] = [
        ("missing metadata", missing_rate, "share of protected metadata that is absent"),
        (
            "cluster concentration",
            cluster_concentration,
            "one cluster dominates while other clusters exist",
        ),
    ]
    rows.extend(
        (name, value, "protected group distribution differs by cluster") for name, value in weighted_signals.items()
    )

    score = max([missing_rate, cluster_concentration * 0.5, *weighted_signals.values()])
    driver_name, driver_value, _driver_help = max(rows, key=lambda item: item[1])
    threshold_note = "OK < 0.15, watch 0.15-0.29, biased >= 0.30"

    if comparison is not None:
        shifts = comparison.get("representation_shift", {})
        representation_rows = {
            "nationality shift": float(shifts.get("nationality_tvd", 0.0) or 0.0),
            "background shift": float(shifts.get("background_tvd", 0.0) or 0.0),
            "age shift": float(shifts.get("age_bucket_tvd", 0.0) or 0.0),
        }
        missing_delta = abs(float(comparison.get("delta_metadata_missing_rate", 0.0) or 0.0))
        anonymous_delta = abs(float(comparison.get("delta_anonymous_rate", 0.0) or 0.0))
        rows = [
            ("nationality shift", representation_rows["nationality shift"], "dirty vs clean nationality mix"),
            ("background shift", representation_rows["background shift"], "dirty vs clean role mix"),
            ("age shift", representation_rows["age shift"], "dirty vs clean age mix"),
            ("metadata loss delta", missing_delta, "extra missing protected metadata"),
            ("anonymous delta", anonymous_delta, "extra anonymous records"),
        ]
        driver_name, driver_value, _driver_help = max(rows, key=lambda item: item[1])
        score = max(driver_value, missing_delta, anonymous_delta)

    if score >= 0.30:
        verdict = "BIAS SIGNAL"
        verdict_color = "#b91c1c"
        verdict_fill = "#fee2e2"
    elif score >= 0.15:
        verdict = "WATCH"
        verdict_color = "#b45309"
        verdict_fill = "#ffedd5"
    else:
        verdict = "OK"
        verdict_color = "#15803d"
        verdict_fill = "#dcfce7"

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_fill": verdict_fill,
        "score": score,
        "driver_name": driver_name,
        "driver_value": driver_value,
        "rows": sorted(rows, key=lambda item: item[1], reverse=True),
        "threshold_note": threshold_note,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def _log_metrics(logger: Any, metrics: dict[str, float]) -> None:
    if logger is None:
        return
    try:
        logger.log_metrics(metrics)
    except TypeError:
        for key, value in metrics.items():
            logger.log_metrics({key: value})
    except Exception:
        pass


def _log_artifact(logger: Any, path: Path, artifact_path: str) -> None:
    if logger is None:
        return
    try:
        logger.log_artifact(str(path), artifact_path=artifact_path)
    except Exception:
        pass


def _slugify_name(value: Any) -> str:
    text = _safe_string(value).lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", text).strip("_")
    return slug or "demographic_bias"


def _build_named_report_payload(dataset_name: str, topic_title: str, analysis: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "topic_title": topic_title,
        "diagnosis": analysis["diagnosis"],
        "clean": {
            "cluster_summary": analysis["clean"]["cluster_summary"],
            "bias_snapshot": analysis["clean"]["bias_snapshot"],
        },
    }
    if analysis["dirty"] is not None:
        payload["dirty"] = {
            "cluster_summary": analysis["dirty"]["cluster_summary"],
            "bias_snapshot": analysis["dirty"]["bias_snapshot"],
        }
    if analysis["comparison"] is not None:
        payload["comparison"] = analysis["comparison"]
    return payload


def _render_report_html(report_payload: dict[str, Any]) -> str:
    diagnosis = report_payload["diagnosis"]
    clean_bias = report_payload["clean"]["bias_snapshot"]
    dirty_payload = report_payload.get("dirty")
    comparison = report_payload.get("comparison")

    rows = [
        ("verdict", diagnosis["verdict"]),
        ("score", f"{float(diagnosis['score']):.3f}"),
        ("driver", f"{diagnosis['driver_name']} ({float(diagnosis['driver_value']):.3f})"),
        ("clean records", str(int(clean_bias["records"]))),
        ("clean nationality tvd", f"{float(clean_bias['cluster_bias_signals']['weighted_nationality_tvd']):.3f}"),
        ("clean age tvd", f"{float(clean_bias['cluster_bias_signals']['weighted_age_bucket_tvd']):.3f}"),
    ]

    if dirty_payload is not None:
        dirty_bias = dirty_payload["bias_snapshot"]
        rows.extend(
            [
                ("dirty records", str(int(dirty_bias["records"]))),
                (
                    "dirty nationality tvd",
                    f"{float(dirty_bias['cluster_bias_signals']['weighted_nationality_tvd']):.3f}",
                ),
                (
                    "dirty age tvd",
                    f"{float(dirty_bias['cluster_bias_signals']['weighted_age_bucket_tvd']):.3f}",
                ),
            ]
        )

    if comparison is not None:
        rows.extend(
            [
                ("delta anonymous rate", f"{float(comparison['delta_anonymous_rate']):.3f}"),
                (
                    "delta metadata missing rate",
                    f"{float(comparison['delta_metadata_missing_rate']):.3f}",
                ),
                (
                    "nationality shift",
                    f"{float(comparison['representation_shift']['nationality_tvd']):.3f}",
                ),
                (
                    "age shift",
                    f"{float(comparison['representation_shift']['age_bucket_tvd']):.3f}",
                ),
            ]
        )

    items = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>" for label, value in rows
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(str(report_payload.get('dataset_name', 'demographic_bias')))} demographic bias report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }}
    table {{ border-collapse: collapse; min-width: 520px; background: #fff; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
    th {{ background: #e2e8f0; }}
    .card {{ background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>{html.escape(str(report_payload.get('dataset_name', 'demographic_bias')))} demographic bias report</h1>
    <p>{html.escape(str(report_payload.get('topic_title', '')))}</p>
  </div>
  <table>
    <tbody>
      {items}
    </tbody>
  </table>
</body>
</html>
"""


def _write_named_analysis_artifacts(
    dataset_name: str,
    output_dir: Path,
    report_payload: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _slugify_name(dataset_name)

    artifacts: dict[str, Path] = {}
    clean_df = analysis["clean"]["dataframe"]
    clean_csv = output_dir / f"{prefix}_clean_cluster_assignments.csv"
    clean_df.to_csv(clean_csv, index=False)
    artifacts["clean_cluster_assignments_csv"] = clean_csv

    clean_cluster_summary_path = output_dir / f"{prefix}_clean_cluster_summary.json"
    clean_bias_snapshot_path = output_dir / f"{prefix}_clean_bias_snapshot.json"
    _write_json(clean_cluster_summary_path, analysis["clean"]["cluster_summary"])
    _write_json(clean_bias_snapshot_path, analysis["clean"]["bias_snapshot"])
    artifacts["clean_cluster_summary_json"] = clean_cluster_summary_path
    artifacts["clean_bias_snapshot_json"] = clean_bias_snapshot_path

    dirty = analysis.get("dirty")
    if dirty is not None:
        dirty_df = dirty["dataframe"]
        dirty_csv = output_dir / f"{prefix}_dirty_cluster_assignments.csv"
        dirty_df.to_csv(dirty_csv, index=False)
        artifacts["dirty_cluster_assignments_csv"] = dirty_csv

        dirty_cluster_summary_path = output_dir / f"{prefix}_dirty_cluster_summary.json"
        dirty_bias_snapshot_path = output_dir / f"{prefix}_dirty_bias_snapshot.json"
        _write_json(dirty_cluster_summary_path, dirty["cluster_summary"])
        _write_json(dirty_bias_snapshot_path, dirty["bias_snapshot"])
        artifacts["dirty_cluster_summary_json"] = dirty_cluster_summary_path
        artifacts["dirty_bias_snapshot_json"] = dirty_bias_snapshot_path

        comparison_path = output_dir / f"{prefix}_clean_dirty_comparison.json"
        _write_json(comparison_path, report_payload["comparison"])
        artifacts["comparison_json"] = comparison_path

    report_json_path = output_dir / f"{prefix}_demographic_bias_report.json"
    report_html_path = output_dir / f"{prefix}_demographic_bias_report.html"
    _write_json(report_json_path, report_payload)
    report_html_path.write_text(_render_report_html(report_payload), encoding="utf-8")
    artifacts["report_json"] = report_json_path
    artifacts["report_html"] = report_html_path

    return artifacts


def _log_named_analysis_results(logger: Any, report_payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    if logger is None:
        return

    clean_bias = report_payload["clean"]["bias_snapshot"]
    metrics = {
        "demographic_bias_anonymous_rate": float(clean_bias["anonymous_rate"]),
        "demographic_bias_missing_rate": float(clean_bias["metadata_missing_rate"]),
        "demographic_bias_weighted_nationality_tvd": float(
            clean_bias["cluster_bias_signals"]["weighted_nationality_tvd"]
        ),
        "demographic_bias_weighted_background_tvd": float(
            clean_bias["cluster_bias_signals"]["weighted_background_tvd"]
        ),
        "demographic_bias_weighted_age_bucket_tvd": float(
            clean_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"]
        ),
        "demographic_bias_largest_cluster_share": float(
            clean_bias["cluster_bias_signals"]["largest_cluster_share"]
        ),
    }
    if report_payload.get("comparison") is not None:
        comparison = report_payload["comparison"]
        metrics.update(
            {
                "demographic_bias_delta_anonymous_rate": float(comparison["delta_anonymous_rate"]),
                "demographic_bias_delta_metadata_missing_rate": float(
                    comparison["delta_metadata_missing_rate"]
                ),
            }
        )

    _log_metrics(logger, metrics)
    for path in artifacts.values():
        _log_artifact(logger, path, artifact_path="demographic_bias")


def analyze_demographic_bias(
    clean_df: pd.DataFrame,
    dirty_df: Optional[pd.DataFrame] = None,
    max_clusters: int = 6,
    age_range: tuple[float | int, float | int] | None = None,
    countries: Sequence[str] | None = None,
    areas: Sequence[str] | None = None,
    append_random_areas: bool = False,
    area_column: str = "Area",
    area_choices: Sequence[str] | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Analyze demographic bias for one or two datasets.

    When ``age_range`` or ``countries`` are provided, the function filters the
    input rows before clustering and computing the bias metrics.
    """

    clean_df = _filter_demographic_subset(
        clean_df,
        age_range=age_range,
        countries=countries,
        areas=areas,
        append_random_areas=append_random_areas,
        area_column=area_column,
        area_choices=area_choices,
        random_seed=random_seed,
    )
    if clean_df.empty:
        raise ValueError("No clean rows match the requested demographic filters.")

    if dirty_df is not None:
        dirty_df = _filter_demographic_subset(
            dirty_df,
            age_range=age_range,
            countries=countries,
            areas=areas,
            append_random_areas=append_random_areas,
            area_column=area_column,
            area_choices=area_choices,
            random_seed=random_seed,
        )
        if dirty_df.empty:
            raise ValueError("No dirty rows match the requested demographic filters.")

    clean_clustered, clean_cluster_meta = _cluster_dataset(clean_df, max_clusters=max_clusters)
    clean_cluster_summary = _build_cluster_summary(clean_clustered, clean_cluster_meta)
    clean_bias_snapshot = _build_bias_snapshot(clean_clustered, clean_cluster_summary)

    dirty_clustered = None
    dirty_cluster_summary = None
    dirty_bias_snapshot = None
    comparison = None

    if dirty_df is not None:
        dirty_clustered, dirty_cluster_meta = _cluster_dataset(dirty_df, max_clusters=max_clusters)
        dirty_cluster_summary = _build_cluster_summary(dirty_clustered, dirty_cluster_meta)
        dirty_bias_snapshot = _build_bias_snapshot(dirty_clustered, dirty_cluster_summary)
        comparison = _compare_clean_dirty(clean_clustered, dirty_clustered, clean_bias_snapshot, dirty_bias_snapshot)

    diagnosis = _bias_diagnosis(clean_bias_snapshot, comparison)

    return {
        "clean": {
            "dataframe": clean_clustered,
            "cluster_summary": clean_cluster_summary,
            "bias_snapshot": clean_bias_snapshot,
        },
        "dirty": None
        if dirty_clustered is None
        else {
            "dataframe": dirty_clustered,
            "cluster_summary": dirty_cluster_summary,
            "bias_snapshot": dirty_bias_snapshot,
        },
        "comparison": comparison,
        "diagnosis": diagnosis,
    }


def build_demographic_bias_artifacts(
    clean_df: pd.DataFrame,
    dirty_df: Optional[pd.DataFrame] = None,
    output_dir: Path | None = None,
    max_clusters: int = 6,
    logger: Any = None,
    age_range: tuple[float | int, float | int] | None = None,
    countries: Sequence[str] | None = None,
    areas: Sequence[str] | None = None,
    append_random_areas: bool = False,
    area_column: str = "Area",
    area_choices: Sequence[str] | None = None,
    random_seed: int | None = None,
    dataset_name: str = "demographic_bias",
    topic_title: str = "",
) -> dict[str, Any]:
    """Build demographic bias artifacts directly from DataFrame inputs.

    Use this when you already have the datasets as variables and do not want to
    load or reconstruct them from batch files.
    """

    analysis = analyze_demographic_bias(
        clean_df,
        dirty_df=dirty_df,
        max_clusters=max_clusters,
        age_range=age_range,
        countries=countries,
        areas=areas,
        append_random_areas=append_random_areas,
        area_column=area_column,
        area_choices=area_choices,
        random_seed=random_seed,
    )
    report_payload = _build_named_report_payload(dataset_name=dataset_name, topic_title=topic_title, analysis=analysis)
    resolved_output_dir = (output_dir or Path("demographic_bias_outputs")).resolve()
    artifacts = _write_named_analysis_artifacts(dataset_name, resolved_output_dir, report_payload, analysis)
    _log_named_analysis_results(_resolve_logger(logger) if logger is not None else None, report_payload, artifacts)

    return {
        "dataset_name": dataset_name,
        "topic_title": topic_title,
        "output_dir": resolved_output_dir,
        "artifacts": artifacts,
        "clean": analysis["clean"],
        "dirty": analysis["dirty"],
        "comparison": analysis["comparison"],
        "diagnosis": analysis["diagnosis"],
    }


def log_demographic_bias_from_data(
    logger: Any,
    clean_df: pd.DataFrame,
    dirty_df: Optional[pd.DataFrame] = None,
    output_dir: Path | None = None,
    max_clusters: int = 6,
    age_range: tuple[float | int, float | int] | None = None,
    countries: Sequence[str] | None = None,
    areas: Sequence[str] | None = None,
    append_random_areas: bool = False,
    area_column: str = "Area",
    area_choices: Sequence[str] | None = None,
    random_seed: int | None = None,
    dataset_name: str = "demographic_bias",
    topic_title: str = "",
) -> dict[str, Any]:
    """Convenience wrapper that logs demographic bias artifacts from DataFrames."""

    return build_demographic_bias_artifacts(
        clean_df=clean_df,
        dirty_df=dirty_df,
        output_dir=output_dir,
        max_clusters=max_clusters,
        logger=logger,
        age_range=age_range,
        countries=countries,
        areas=areas,
        append_random_areas=append_random_areas,
        area_column=area_column,
        area_choices=area_choices,
        random_seed=random_seed,
        dataset_name=dataset_name,
        topic_title=topic_title,
    )


def _build_report_payload(batch_id: int, topic_title: str, analysis: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "topic_title": topic_title,
        "diagnosis": analysis["diagnosis"],
        "clean": {
            "cluster_summary": analysis["clean"]["cluster_summary"],
            "bias_snapshot": analysis["clean"]["bias_snapshot"],
        },
    }
    if analysis["dirty"] is not None:
        payload["dirty"] = {
            "cluster_summary": analysis["dirty"]["cluster_summary"],
            "bias_snapshot": analysis["dirty"]["bias_snapshot"],
        }
    if analysis["comparison"] is not None:
        payload["comparison"] = analysis["comparison"]
    return payload


def _write_analysis_artifacts(
    batch_id: int,
    output_dir: Path,
    report_payload: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    clean_df = analysis["clean"]["dataframe"]
    clean_csv = output_dir / f"clean_batch_{batch_id}_cluster_assignments.csv"
    clean_df.to_csv(clean_csv, index=False)
    artifacts["clean_cluster_assignments_csv"] = clean_csv

    clean_cluster_summary_path = output_dir / f"clean_batch_{batch_id}_cluster_summary.json"
    clean_bias_snapshot_path = output_dir / f"clean_batch_{batch_id}_bias_snapshot.json"
    _write_json(clean_cluster_summary_path, analysis["clean"]["cluster_summary"])
    _write_json(clean_bias_snapshot_path, analysis["clean"]["bias_snapshot"])
    artifacts["clean_cluster_summary_json"] = clean_cluster_summary_path
    artifacts["clean_bias_snapshot_json"] = clean_bias_snapshot_path

    dirty = analysis.get("dirty")
    if dirty is not None:
        dirty_df = dirty["dataframe"]
        dirty_csv = output_dir / f"dirty_batch_{batch_id}_cluster_assignments.csv"
        dirty_df.to_csv(dirty_csv, index=False)
        artifacts["dirty_cluster_assignments_csv"] = dirty_csv

        dirty_cluster_summary_path = output_dir / f"dirty_batch_{batch_id}_cluster_summary.json"
        dirty_bias_snapshot_path = output_dir / f"dirty_batch_{batch_id}_bias_snapshot.json"
        _write_json(dirty_cluster_summary_path, dirty["cluster_summary"])
        _write_json(dirty_bias_snapshot_path, dirty["bias_snapshot"])
        artifacts["dirty_cluster_summary_json"] = dirty_cluster_summary_path
        artifacts["dirty_bias_snapshot_json"] = dirty_bias_snapshot_path

        comparison_path = output_dir / f"batch_{batch_id}_clean_dirty_comparison.json"
        _write_json(comparison_path, report_payload["comparison"])
        artifacts["comparison_json"] = comparison_path

    report_json_path = output_dir / f"batch_{batch_id}_demographic_bias_report.json"
    report_html_path = output_dir / f"batch_{batch_id}_demographic_bias_report.html"
    _write_json(report_json_path, report_payload)
    artifacts["report_json"] = report_json_path
    artifacts["report_html"] = report_html_path

    return artifacts


def _log_analysis_results(logger: Any, batch_id: int, report_payload: dict[str, Any], artifacts: dict[str, Path]) -> None:
    if logger is None:
        return

    clean_bias = report_payload["clean"]["bias_snapshot"]
    metrics = {
        "demographic_bias_anonymous_rate": float(clean_bias["anonymous_rate"]),
        "demographic_bias_missing_rate": float(clean_bias["metadata_missing_rate"]),
        "demographic_bias_weighted_nationality_tvd": float(
            clean_bias["cluster_bias_signals"]["weighted_nationality_tvd"]
        ),
        "demographic_bias_weighted_background_tvd": float(
            clean_bias["cluster_bias_signals"]["weighted_background_tvd"]
        ),
        "demographic_bias_weighted_age_bucket_tvd": float(
            clean_bias["cluster_bias_signals"]["weighted_age_bucket_tvd"]
        ),
        "demographic_bias_largest_cluster_share": float(
            clean_bias["cluster_bias_signals"]["largest_cluster_share"]
        ),
    }
    if report_payload.get("comparison") is not None:
        comparison = report_payload["comparison"]
        metrics.update(
            {
                "demographic_bias_delta_anonymous_rate": float(comparison["delta_anonymous_rate"]),
                "demographic_bias_delta_metadata_missing_rate": float(
                    comparison["delta_metadata_missing_rate"]
                ),
            }
        )

    _log_metrics(logger, metrics)
    for path in artifacts.values():
        _log_artifact(logger, path, artifact_path="demographic_bias")


def build_batch_demographic_bias_artifacts(
    batch_id: int,
    session_dir: Path | None = None,
    output_dir: Path | None = None,
    max_clusters: int = 6,
    logger: Any = None,
    age_range: tuple[float | int, float | int] | None = None,
    countries: Sequence[str] | None = None,
    areas: Sequence[str] | None = None,
    append_random_areas: bool = False,
    area_column: str = "Area",
    area_choices: Sequence[str] | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Build clustering and bias artifacts for one clean/dirty batch."""

    resolved_session_dir = (session_dir or Path("session")).resolve()
    clean_path = resolved_session_dir / f"simulation_output_clean_{batch_id}.json"
    dirty_path = resolved_session_dir / f"simulation_output_dirty_{batch_id}.json"
    if not clean_path.exists() or not dirty_path.exists():
        raise FileNotFoundError(
            f"Batch {batch_id} not found in {resolved_session_dir}. Expected paired clean/dirty JSON files."
        )

    batch_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else resolved_session_dir / "cluster_bias_outputs" / f"batch_{batch_id}"
    )

    clean_payload = _load_json(clean_path)
    dirty_payload = _load_json(dirty_path)
    internal_annotations = _load_internal_annotations(resolved_session_dir, batch_id)

    clean_df = _flatten_dataset(clean_payload, dataset_type="clean", batch_id=batch_id, internal_annotations=internal_annotations["clean"])
    dirty_df = _flatten_dataset(dirty_payload, dataset_type="dirty", batch_id=batch_id, internal_annotations=internal_annotations["dirty"])

    analysis = analyze_demographic_bias(
        clean_df,
        dirty_df=dirty_df,
        max_clusters=max_clusters,
        age_range=age_range,
        countries=countries,
        areas=areas,
        append_random_areas=append_random_areas,
        area_column=area_column,
        area_choices=area_choices,
        random_seed=random_seed,
    )
    report_payload = _build_named_report_payload(
        dataset_name=f"batch_{batch_id}",
        topic_title=clean_payload.get("TopicTitle") or dirty_payload.get("TopicTitle") or "",
        analysis=analysis,
    )
    artifacts = _write_named_analysis_artifacts(f"batch_{batch_id}", batch_output_dir, report_payload, analysis)
    _log_named_analysis_results(_resolve_logger(logger) if logger is not None else None, report_payload, artifacts)

    comparison_report = {
        "batch_id": batch_id,
        "topic_title": report_payload["topic_title"],
        "clean_cluster_count": analysis["clean"]["cluster_summary"]["cluster_count"],
        "dirty_cluster_count": analysis["dirty"]["cluster_summary"]["cluster_count"],
        "comparison": analysis["comparison"],
    }

    return {
        "batch_id": batch_id,
        "topic_title": report_payload["topic_title"],
        "output_dir": batch_output_dir,
        "artifacts": artifacts,
        "clean": analysis["clean"],
        "dirty": analysis["dirty"],
        "comparison": comparison_report,
    }
