#!/usr/bin/env python3
"""Update paths in memory_artifacts to point to the demo dataset."""

import json
from pathlib import Path

mem_dir = Path(__file__).parent / "memory_artifacts"
records_file = mem_dir / "records.jsonl"
pending_file = mem_dir / "active_learning" / "pending_labels.jsonl"
thumbnails_dir = mem_dir / "thumbnails"

def update_paths_in_jsonl(file_path):
    if not file_path.exists():
        print(f"Skipping {file_path}: not found")
        return
    updated = []
    with file_path.open("r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing {file_path}:{line_num}: {e}")
                continue
            for key in ["frame_path", "source_path"]:
                if key in obj and isinstance(obj[key], str):
                    obj[key] = obj[key].replace("data/frames/", "data/frames_demo/", 1)
            updated.append(obj)
    with file_path.open("w") as f:
        for obj in updated:
            f.write(json.dumps(obj) + "\n")
    print(f"Updated {len(updated)} records in {file_path}")

def clear_thumbnails_dir():
    if not thumbnails_dir.exists():
        print(f"Thumbnails directory {thumbnails_dir} does not exist")
        return
    deleted = 0
    for child in thumbnails_dir.iterdir():
        if child.is_file():
            child.unlink()
            deleted += 1
        elif child.is_dir():
            # Should not happen, but just in case
            import shutil
            shutil.rmtree(child)
            deleted += 1
    print(f"Deleted {deleted} items from {thumbnails_dir}")

if __name__ == "__main__":
    update_paths_in_jsonl(records_file)
    update_paths_in_jsonl(pending_file)
    clear_thumbnails_dir()