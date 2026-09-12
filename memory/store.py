"""Simple JSONL + NumPy storage for confident novel memories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import config


@dataclass(frozen=True)
class MemoryRecord:
    """Metadata for one stored memory."""

    memory_id: str
    frame_path: str
    thumbnail_path: str
    source_path: str
    timestamp_sec: float | None
    routine_hour: int | None
    structural_embedding_path: str
    semantic_embedding_path: str
    category: str | None = None


def memory_root() -> str:
    """Return the configured memory artifact directory."""
    return str(config.MEMORY_DIR)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    """Append one JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_memory(
    *,
    memory_id: str,
    frame_record: dict[str, object],
    structural_embedding: np.ndarray,
    semantic_embedding: np.ndarray,
    routine_hour: int | None,
    category: str | None = None,
) -> MemoryRecord:
    """Persist one confident novel memory with DINOv2 and CLIP embeddings."""
    config.MEMORY_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    structural_path = config.MEMORY_EMBEDDINGS_DIR / f"{memory_id}_dino.npy"
    semantic_path = config.MEMORY_EMBEDDINGS_DIR / f"{memory_id}_clip.npy"
    np.save(structural_path, structural_embedding.astype(np.float32))
    np.save(semantic_path, semantic_embedding.astype(np.float32))

    record = MemoryRecord(
        memory_id=memory_id,
        frame_path=str(frame_record["frame_path"]),
        thumbnail_path=str(frame_record["frame_path"]),
        source_path=str(frame_record.get("source_path", "")),
        timestamp_sec=frame_record.get("timestamp_sec"),
        routine_hour=routine_hour,
        structural_embedding_path=str(structural_path.relative_to(config.ROOT_DIR)),
        semantic_embedding_path=str(semantic_path.relative_to(config.ROOT_DIR)),
        category=category,
    )
    append_jsonl(config.MEMORY_RECORDS_PATH, asdict(record))
    return record


def read_memory(memory_id: str) -> tuple[MemoryRecord, np.ndarray, np.ndarray]:
    """Read one memory record and its stored DINOv2/CLIP embeddings."""
    if not config.MEMORY_RECORDS_PATH.exists():
        raise FileNotFoundError(f"Missing memory index: {config.MEMORY_RECORDS_PATH}")

    found: dict[str, object] | None = None
    with config.MEMORY_RECORDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["memory_id"] == memory_id:
                found = payload

    if found is None:
        raise KeyError(f"Memory id not found: {memory_id}")

    record = MemoryRecord(**found)
    structural = np.load(config.ROOT_DIR / record.structural_embedding_path)
    semantic = np.load(config.ROOT_DIR / record.semantic_embedding_path)
    return record, structural, semantic
