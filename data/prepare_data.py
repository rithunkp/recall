"""Extract frames from staged footage and create fixed train/val/test manifests.

This script never creates placeholder data. If `data/raw/` has no footage or images,
it exits with instructions to stage real data first.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def find_raw_inputs() -> list[Path]:
    """Return raw videos and raw image frames staged under data/raw."""
    if not config.RAW_DATA_DIR.exists():
        return []
    allowed = config.VIDEO_EXTENSIONS + config.IMAGE_EXTENSIONS
    return sorted(
        path
        for path in config.RAW_DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not any(part.endswith("_gt") for part in path.parts)
    )


def scenario_from_path(path: Path) -> str:
    """Infer a coarse scenario label from the first directory under data/raw."""
    relative = path.relative_to(config.RAW_DATA_DIR)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return "unlabeled"


def extract_video_frames(video_path: Path) -> list[dict[str, object]]:
    """Extract every Nth frame from one raw video."""
    import cv2

    scenario = scenario_from_path(video_path)
    output_dir = config.FRAMES_DIR / scenario / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    records: list[dict[str, object]] = []
    frame_index = 0
    saved_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % config.FRAME_STRIDE == 0:
            frame_name = f"frame_{saved_index:06d}{config.FRAME_IMAGE_EXT}"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            records.append(
                {
                    "frame_path": str(frame_path.relative_to(config.ROOT_DIR)),
                    "source_path": str(video_path.relative_to(config.ROOT_DIR)),
                    "scenario": scenario,
                    "timestamp_sec": frame_index / fps if fps > 0 else None,
                }
            )
            saved_index += 1
        frame_index += 1

    capture.release()
    return records


def copy_raw_image(image_path: Path) -> dict[str, object]:
    """Copy a staged image into the frame directory and return its manifest row."""
    scenario = scenario_from_path(image_path)
    output_dir = config.FRAMES_DIR / scenario / image_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / image_path.name
    if image_path.resolve() != target.resolve() and not target.exists():
        shutil.copy2(image_path, target)
    return {
        "frame_path": str(target.relative_to(config.ROOT_DIR)),
        "source_path": str(image_path.relative_to(config.ROOT_DIR)),
        "scenario": scenario,
        "timestamp_sec": None,
    }


def build_frame_records(raw_inputs: list[Path]) -> list[dict[str, object]]:
    """Extract/copy all raw inputs into data/frames."""
    records: list[dict[str, object]] = []
    for raw_path in raw_inputs:
        suffix = raw_path.suffix.lower()
        if suffix in config.VIDEO_EXTENSIONS:
            records.extend(extract_video_frames(raw_path))
        elif suffix in config.IMAGE_EXTENSIONS:
            records.append(copy_raw_image(raw_path))
    return records


def split_records(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Create deterministic train/val/test splits without touching model state."""
    shuffled = list(records)
    random.Random(config.SEED).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * config.TRAIN_RATIO)
    val_end = train_end + int(total * config.VAL_RATIO)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write one JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_manifests(splits: dict[str, list[dict[str, object]]]) -> None:
    """Persist fixed split manifests under data/splits."""
    write_jsonl(config.TRAIN_MANIFEST, splits["train"])
    write_jsonl(config.VAL_MANIFEST, splits["val"])
    write_jsonl(config.TEST_MANIFEST, splits["test"])


def main() -> int:
    """Prepare frame data and split manifests from staged raw footage."""
    config.set_seed()
    raw_inputs = find_raw_inputs()
    if not raw_inputs:
        print(
            f"No staged footage found in {config.RAW_DATA_DIR}. "
            "Add real videos/images there before running data preparation.",
            file=sys.stderr,
        )
        return 1

    records = build_frame_records(raw_inputs)
    if not records:
        print("Raw inputs were found, but no frames were extracted.", file=sys.stderr)
        return 1

    splits = split_records(records)
    write_manifests(splits)
    print(
        "Prepared "
        f"{len(records)} frames: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
