"""Pure CLIP-text retrieval over stored memory image embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


@dataclass(frozen=True)
class RetrievalResult:
    """Single best retrieval result."""

    matched: bool
    answer: str
    similarity: float | None
    memory: dict[str, object] | None


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize embeddings row-wise for cosine similarity."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, a_min=1e-12, a_max=None)


def load_memory_records() -> list[dict[str, object]]:
    """Load stored memories from the JSONL memory index."""
    if not config.MEMORY_RECORDS_PATH.exists():
        return []
    records: list[dict[str, object]] = []
    with config.MEMORY_RECORDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def load_memory_matrix(records: list[dict[str, object]]) -> np.ndarray:
    """Load stored CLIP image embeddings for all memory records."""
    embeddings = [
        np.load(config.ROOT_DIR / str(record["semantic_embedding_path"]))
        for record in records
    ]
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return l2_normalize(np.vstack(embeddings).astype(np.float32))


def clip_text_tensor(output: object) -> torch.Tensor:
    """Handle CLIP text-feature return differences across transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


class MemoryRetriever:
    """Frozen CLIP text encoder plus brute-force cosine search."""

    def __init__(self) -> None:
        config.set_seed()
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.DEVICE)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def embed_text(self, query: str) -> np.ndarray:
        """Embed one query with frozen CLIP text features."""
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(
            config.DEVICE
        )
        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
            embedding = clip_text_tensor(output).detach().cpu().numpy()
        return l2_normalize(embedding.astype(np.float32))[0]

    def retrieve(self, query: str) -> RetrievalResult:
        """Return one thresholded best match or a templated no-match answer."""
        records = load_memory_records()
        if not records:
            return RetrievalResult(False, "No matching event found.", None, None)

        matrix = load_memory_matrix(records)
        query_embedding = self.embed_text(query)
        similarities = matrix @ query_embedding
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        best_record = records[best_index]

        if best_similarity < config.RETRIEVAL_SIMILARITY_THRESHOLD:
            return RetrievalResult(
                False,
                "No matching event found.",
                best_similarity,
                best_record,
            )

        return RetrievalResult(
            True,
            build_answer(best_record, best_similarity),
            best_similarity,
            best_record,
        )


def timestamp_text(memory: dict[str, object]) -> str:
    """Format stored timestamp fields without inventing wall-clock time."""
    timestamp = memory.get("timestamp_sec")
    if timestamp is not None:
        return f"timestamp {float(timestamp):.2f}s"
    routine_hour = memory.get("routine_hour")
    if routine_hour is not None:
        return f"synthetic hour {routine_hour}"
    return "timestamp unavailable"


def build_answer(memory: dict[str, object], similarity: float) -> str:
    """Build a fixed, non-generative retrieval answer."""
    category = memory.get("category")
    category_text = f" {category}" if category else ""
    return (
        f"Best matching{category_text} event found at {timestamp_text(memory)} "
        f"with similarity {similarity:.4f}. "
        f"Frame: {memory['frame_path']}. "
        f"Thumbnail: {memory['thumbnail_path']}."
    )


def main() -> int:
    """CLI query demo for memory retrieval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="Question to retrieve against memory")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Question: ").strip()
    result = MemoryRetriever().retrieve(query)
    print(result.answer)
    if result.similarity is not None:
        print(f"Best similarity: {result.similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
