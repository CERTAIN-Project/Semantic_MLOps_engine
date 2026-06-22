"""Compact mirrored JSONL events into portable JSON exports."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .metadata_writer import EVENT_FILES
from .tracker import mirror_root


def read_events(root: Path, filename: str) -> List[dict]:
    """Read all valid events from one JSONL file."""
    path = root / "events" / filename

    if not path.exists():
        return []

    events = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            events.append(json.loads(line))

    return events


def latest_by(
    events: Iterable[dict],
    fields: Tuple[str, ...],
) -> List[dict]:
    """Return the last event for each compound key."""
    latest: Dict[tuple, dict] = {}

    for event in events:
        key = tuple(event.get(field) for field in fields)
        latest[key] = event

    return list(latest.values())


def unique_by(
    events: Iterable[dict],
    fields: Tuple[str, ...],
) -> List[dict]:
    """Deduplicate events while preserving their first occurrence."""
    unique: Dict[tuple, dict] = {}

    for event in events:
        key = tuple(event.get(field) for field in fields)
        unique.setdefault(key, event)

    return list(unique.values())


def compact(root: Path, output: Path) -> Dict[str, int]:
    """Create compact JSON files from the mirrored event stream."""
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    events = {
        event_type: read_events(root, filename)
        for event_type, filename in EVENT_FILES.items()
    }

    runs_by_id: Dict[str, dict] = {}
    experiments_by_id: Dict[str, dict] = {}

    for event in events["run"]:
        run_id = event["run_id"]
        runs_by_id.setdefault(run_id, {}).update(event)

        experiment_id = event.get("experiment_id")
        if experiment_id is not None:
            experiments_by_id.setdefault(
                str(experiment_id),
                {
                    "experiment_id": str(experiment_id),
                    "name": event.get("experiment_name"),
                },
            )

    exports = {
        "experiments": list(experiments_by_id.values()),
        "runs": list(runs_by_id.values()),
        "params": latest_by(
            events["param"],
            ("run_id", "key"),
        ),
        "metrics": unique_by(
            events["metric"],
            ("run_id", "key", "value", "step", "timestamp"),
        ),
        "latest_metrics": latest_by(
            events["metric"],
            ("run_id", "key"),
        ),
        "tags": latest_by(
            events["tag"],
            ("run_id", "key"),
        ),
        "artifacts": latest_by(
            events["artifact"],
            ("run_id", "artifact_path"),
        ),
    }

    for name, records in exports.items():
        destination = output / "{}.json".format(name)

        with destination.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
            handle.write("\n")

    return {
        name: len(records)
        for name, records in exports.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for compact JSON files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=mirror_root(),
        help="CERTAIN mirror root. Defaults to CERTAIN_MIRROR_ROOT or ./certain.",
    )
    arguments = parser.parse_args()

    counts = compact(arguments.root, arguments.output)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
