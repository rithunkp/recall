#!/usr/bin/env python3
"""
Create a trimmed demo dataset containing only the frames referenced by the
current memory_artifacts (the already-scanned demo subset).
Also create demo manifests (train_demo.jsonl, test_demo.jsonl) that point to
the trimmed dataset.
"""

import json
import os
import random
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]

def save_manifest(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

def source_partition(record: dict[str, object]) -> str:
    source = str(record.get("source_path", "")).replace("/", "\\")
    if "\\Test\\" in source:
        return "test_source"
    if "\\Train\\" in source:
        return "train_source"
    return "unknown_source"

def sampled_demo_records(limit: int | None = None):
    """Sample a representative demo subset from train/test sequence sources."""
    max_records = int(limit or config.SCAN_FULL_DATASET_LIMIT)
    if max_records <= 0:
        raise ValueError("Scan limit must be at least 1.")

    records = load_manifest(config.TRAIN_MANIFEST) + load_manifest(config.TEST_MANIFEST)
    by_partition: dict[str, list[dict[str, object]]] = {
        "train_source": [],
        "test_source": [],
        "unknown_source": [],
    }
    for record in records:
        by_partition[source_partition(record)].append(record)

    rng = random.Random(config.SEED)
    for partition_records in by_partition.values():
        rng.shuffle(partition_records)

    selected: list[dict[str, object]] = []
    preferred = ["train_source", "test_source", "unknown_source"]
    while len(selected) < max_records and any(by_partition.values()):
        for partition in preferred:
            if len(selected) >= max_records:
                break
            if by_partition[partition]:
                selected.append(by_partition[partition].pop())

    rng.shuffle(selected)
    return selected

def main():
    # Get the demo subset records (same as in demo/app.py)
    demo_records = sampled_demo_records()
    print(f"Demo subset size: {len(demo_records)}")

    # Source and destination directories for frames
    src_frames_dir = config.FRAMES_DIR  # data/frames
    dst_frames_dir = Path(__file__).resolve().parent / "data" / "frames_demo"
    dst_frames_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for record in demo_records:
        frame_path = record["frame_path"]  # e.g., "data/frames/UCSDped1/Train013/106.tif"
        # Make it relative to the repo root
        src_path = Path(__file__).resolve().parent / frame_path
        # Build destination path preserving the same relative structure under data/frames_demo
        # We want to replace the "data/frames" part with "data/frames_demo"
        relative_to_repo = src_path.relative_to(Path(__file__).resolve().parent)
        dst_path = dst_frames_dir / relative_to_repo
        # Ensure the destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            print(f"Warning: Source frame not found: {src_path}")
            missing += 1

    print(f"Copied {copied} frames, missing {missing} frames.")
    print(f"Demo dataset created at: {dst_frames_dir}")

    # Now create demo manifests by replacing the frame_path and source_path
    # to point to the demo dataset.
    def adjust_paths(record):
        # Adjust frame_path and source_path to point to the demo dataset
        # They are currently like: "data/frames/..."
        # We want: "data/frames_demo/..."
        for key in ["frame_path", "source_path"]:
            if key in record and isinstance(record[key], str):
                record[key] = record[key].replace("data/frames/", "data/frames_demo/", 1)
        return record

    # Process train manifest
    train_records = load_manifest(config.TRAIN_MANIFEST)
    demo_train_records = [adjust_paths(r) for r in train_records]
    demo_train_path = config.ROOT_DIR / "data" / "splits" / "train_demo.jsonl"
    save_manifest(demo_train_records, demo_train_path)
    print(f"Demo train manifest saved to: {demo_train_path}")

    # Process test manifest
    test_records = load_manifest(config.TEST_MANIFEST)
    demo_test_records = [adjust_paths(r) for r in test_records]
    demo_test_path = config.ROOT_DIR / "data" / "splits" / "test_demo.jsonl"
    save_manifest(demo_test_records, demo_test_path)
    print(f"Demo test manifest saved to: {demo_test_path}")

if __name__ == "__main__":
    main()