"""Self-supervised routine novelty from an hourly timestamp histogram.

The Routine Agent scores when an event happens, not what it looks like. It uses
train-manifest records only to learn a smoothed 24-hour activity histogram, then
scores new records by how uncommon their hour is.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


SEQUENCE_RE = re.compile(r"^(Train|Test)(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RoutineStats:
    """Train-only hourly routine statistics."""

    counts: list[int]
    smoothed: list[float]
    max_smoothed_count: float
    total_records: int
    timestamp_policy: str


@dataclass(frozen=True)
class NoveltyResult:
    """Routine novelty score for one manifest record."""

    frame_path: str
    hour: int
    novelty_score: float
    is_novel: bool
    timestamp_source: str


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Load a JSONL manifest created by data.prepare_data."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def sequence_name(record: dict[str, object]) -> str:
    """Return the UCSD sequence directory name for a manifest record."""
    source_parts = Path(str(record.get("source_path", ""))).parts
    frame_parts = Path(str(record["frame_path"])).parts
    for part in reversed(source_parts + frame_parts):
        if SEQUENCE_RE.match(part):
            return part
    return Path(str(record["frame_path"])).parent.name


def frame_number(record: dict[str, object]) -> int:
    """Return the numeric frame stem when available."""
    stem = Path(str(record["frame_path"])).stem
    return int(stem) if stem.isdigit() else 0


def synthetic_ucsd_hour(record: dict[str, object]) -> tuple[int, str]:
    """Derive a deterministic time-of-day proxy for UCSD frames.

    UCSD has sequence/frame order but no wall-clock timestamps. Normal
    `TrainNNN` source sequences are treated as routine daytime pedestrian
    traffic. `TestNNN` source sequences are placed in evening buckets so the
    limitation is explicit and reproducible.
    """
    sequence = sequence_name(record)
    match = SEQUENCE_RE.match(sequence)
    if not match:
        stable = sum(ord(char) for char in sequence) + frame_number(record)
        return stable % config.ROUTINE_NUM_HOURS, "synthetic_sequence_hash"

    kind = match.group(1).lower()
    sequence_id = int(match.group(2))
    if kind == "train":
        hour = config.ROUTINE_DAY_START_HOUR + (
            (sequence_id - 1) % config.ROUTINE_DAY_HOURS
        )
        return hour % config.ROUTINE_NUM_HOURS, "synthetic_ucsd_train_daytime"

    hour = config.ROUTINE_EVENING_START_HOUR + (
        (sequence_id - 1) % config.ROUTINE_EVENING_HOURS
    )
    return hour % config.ROUTINE_NUM_HOURS, "synthetic_ucsd_test_evening"


def record_hour(record: dict[str, object]) -> tuple[int, str]:
    """Return hour-of-day and source for a manifest record."""
    timestamp = record.get("timestamp_sec")
    if timestamp is not None:
        hour = int(float(timestamp) // 3600) % config.ROUTINE_NUM_HOURS
        return hour, "timestamp_sec"
    return synthetic_ucsd_hour(record)


def train_manifest_records() -> list[dict[str, object]]:
    """Load only the train split; never use val/test for learned statistics."""
    if not config.TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {config.TRAIN_MANIFEST}; run data/prepare_data.py first."
        )
    return load_manifest(config.TRAIN_MANIFEST)


def validation_manifest_records() -> list[dict[str, object]]:
    """Load validation records for smoke scoring only."""
    if not config.VAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {config.VAL_MANIFEST}; run data/prepare_data.py first."
        )
    return load_manifest(config.VAL_MANIFEST)


def build_stats(records: list[dict[str, object]]) -> RoutineStats:
    """Build train-only smoothed hourly counts."""
    if not records:
        raise ValueError("Cannot build routine statistics from an empty manifest.")

    counts = [0 for _ in range(config.ROUTINE_NUM_HOURS)]
    sources: set[str] = set()
    for record in records:
        hour, source = record_hour(record)
        counts[hour] += 1
        sources.add(source)

    smoothed = [count + config.ROUTINE_SMOOTHING for count in counts]
    return RoutineStats(
        counts=counts,
        smoothed=smoothed,
        max_smoothed_count=max(smoothed),
        total_records=len(records),
        timestamp_policy=", ".join(sorted(sources)),
    )


def save_stats(stats: RoutineStats, path: Path = config.ROUTINE_STATS_PATH) -> None:
    """Persist routine statistics as JSON for coordinator/demo use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "counts": stats.counts,
        "smoothed": stats.smoothed,
        "max_smoothed_count": stats.max_smoothed_count,
        "total_records": stats.total_records,
        "timestamp_policy": stats.timestamp_policy,
        "smoothing": config.ROUTINE_SMOOTHING,
        "num_hours": config.ROUTINE_NUM_HOURS,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_stats(path: Path = config.ROUTINE_STATS_PATH) -> RoutineStats:
    """Load routine statistics from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RoutineStats(
        counts=[int(value) for value in payload["counts"]],
        smoothed=[float(value) for value in payload["smoothed"]],
        max_smoothed_count=float(payload["max_smoothed_count"]),
        total_records=int(payload["total_records"]),
        timestamp_policy=str(payload["timestamp_policy"]),
    )


class RoutineAgent:
    """Hourly histogram routine scorer."""

    def score_record(
        self, record: dict[str, object], stats: RoutineStats
    ) -> NoveltyResult:
        """Score one manifest record by trained hour frequency."""
        hour, source = record_hour(record)
        routine_strength = stats.smoothed[hour] / stats.max_smoothed_count
        score = 1.0 - routine_strength
        return NoveltyResult(
            frame_path=str(record["frame_path"]),
            hour=hour,
            novelty_score=score,
            is_novel=score >= config.ROUTINE_NOVELTY_THRESHOLD,
            timestamp_source=source,
        )


def first_routine_daytime_record(
    records: list[dict[str, object]]
) -> dict[str, object] | None:
    """Find a held-out UCSD Train sequence record for the daytime-motion check."""
    for record in records:
        hour, source = record_hour(record)
        if (
            source == "synthetic_ucsd_train_daytime"
            and config.ROUTINE_DAY_START_HOUR
            <= hour
            < config.ROUTINE_DAY_START_HOUR + config.ROUTINE_DAY_HOURS
        ):
            return record
    return None


def print_result(label: str, result: NoveltyResult) -> None:
    """Print one routine score in the same compact style as other agents."""
    print(
        f"{label}: {result.frame_path} hour={result.hour} "
        f"source={result.timestamp_source} score={result.novelty_score:.4f} "
        f"is_novel={result.is_novel}"
    )


def main() -> int:
    """Build train-only routine stats and smoke-score validation records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-index", type=int, default=0)
    args = parser.parse_args()

    config.set_seed()
    train_records = train_manifest_records()
    val_records = validation_manifest_records()
    stats = build_stats(train_records)
    save_stats(stats)

    agent = RoutineAgent()
    target = val_records[args.score_index]
    result = agent.score_record(target, stats)

    daytime_record = first_routine_daytime_record(val_records)
    if daytime_record is None:
        raise RuntimeError("No validation daytime UCSD Train sequence found to score.")
    daytime_result = agent.score_record(daytime_record, stats)

    print(f"Saved routine stats: {config.ROUTINE_STATS_PATH}")
    print(f"Train records: {stats.total_records}")
    print(f"Hourly counts: {stats.counts}")
    print(f"Timestamp policy: {stats.timestamp_policy}")
    print_result("Routine score", result)
    print_result("Routine daytime-motion check", daytime_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
