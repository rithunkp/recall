"""Frozen CLIP semantic embeddings and prototype-distance novelty scoring.

The Semantic Agent catches category-level deviation in CLIP image space. CLIP
stays frozen; only prototype arrays are stored.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


@dataclass(frozen=True)
class NoveltyResult:
    """Semantic novelty score for one frame."""

    frame_path: str
    novelty_score: float
    is_novel: bool
    nearest_distance: float


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Load a JSONL manifest created by data.prepare_data."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def frame_path(record: dict[str, object]) -> Path:
    """Resolve a manifest frame path relative to the repo root."""
    return config.ROOT_DIR / str(record["frame_path"])


def load_rgb_image(path: Path) -> Image.Image:
    """Load one image as RGB for CLIP preprocessing."""
    return Image.open(path).convert("RGB")


def resolve_frame_path(path: Path) -> Path:
    """Resolve absolute or repo-relative frame paths."""
    return path if path.is_absolute() else config.ROOT_DIR / path


def display_frame_path(path: Path) -> str:
    """Return a repo-relative frame path when possible."""
    resolved = resolve_frame_path(path)
    try:
        return str(resolved.relative_to(config.ROOT_DIR))
    except ValueError:
        return str(resolved)


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize embeddings row-wise for cosine distance via dot product."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, a_min=1e-12, a_max=None)


def clip_image_tensor(output: object) -> torch.Tensor:
    """Handle CLIP image-feature return differences across transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


class SemanticAgent:
    """Frozen CLIP image embedder plus cosine-distance prototype scorer."""

    def __init__(self) -> None:
        config.set_seed()
        self.processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.DEVICE)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def embed_images(self, image_paths: Iterable[Path]) -> np.ndarray:
        """Embed images with frozen CLIP image features."""
        paths = list(image_paths)
        embeddings: list[np.ndarray] = []
        for start in range(0, len(paths), config.SEMANTIC_BATCH_SIZE):
            batch_paths = paths[start : start + config.SEMANTIC_BATCH_SIZE]
            images = [load_rgb_image(path) for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(config.DEVICE)
            with torch.inference_mode():
                output = self.model.get_image_features(**inputs)
                batch = clip_image_tensor(output).detach().cpu().numpy()
            embeddings.append(batch)
        if not embeddings:
            raise ValueError("No images were provided for semantic embedding.")
        return l2_normalize(np.concatenate(embeddings, axis=0).astype(np.float32))

    def score_embedding(self, embedding: np.ndarray, prototypes: np.ndarray) -> float:
        """Return cosine distance to the nearest semantic prototype."""
        if prototypes.size == 0:
            raise ValueError("At least one prototype is required for novelty scoring.")
        vector = l2_normalize(embedding.reshape(1, -1).astype(np.float32))[0]
        normalized_prototypes = l2_normalize(prototypes.astype(np.float32))
        similarities = normalized_prototypes @ vector
        return float(1.0 - np.max(similarities))

    def score_frame(self, image_path: Path, prototypes: np.ndarray) -> NoveltyResult:
        """Embed and score one frame against stored prototypes."""
        resolved_path = resolve_frame_path(image_path)
        embedding = self.embed_images([resolved_path])[0]
        distance = self.score_embedding(embedding, prototypes)
        return NoveltyResult(
            frame_path=display_frame_path(resolved_path),
            novelty_score=distance,
            is_novel=distance >= config.SEMANTIC_NOVELTY_THRESHOLD,
            nearest_distance=distance,
        )


def build_prototypes(
    agent: SemanticAgent,
    records: list[dict[str, object]],
    max_prototypes: int = config.SEMANTIC_MAX_PROTOTYPES,
) -> np.ndarray:
    """Build a deterministic prototype bank from train-manifest frames only."""
    if not records:
        raise ValueError("Cannot build semantic prototypes from an empty manifest.")
    selected = records[:max_prototypes]
    paths = [frame_path(record) for record in selected]
    return agent.embed_images(paths)


def save_prototypes(
    prototypes: np.ndarray, path: Path = config.SEMANTIC_PROTOTYPES_PATH
) -> None:
    """Persist semantic prototype embeddings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prototypes.astype(np.float32))


def load_prototypes(path: Path = config.SEMANTIC_PROTOTYPES_PATH) -> np.ndarray:
    """Load semantic prototype embeddings."""
    return np.load(path)


def train_manifest_records() -> list[dict[str, object]]:
    """Load only the train split; never use val/test for prototype construction."""
    if not config.TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {config.TRAIN_MANIFEST}; run data/prepare_data.py first."
        )
    return load_manifest(config.TRAIN_MANIFEST)


def main() -> int:
    """Build train-only semantic prototypes and smoke-score one train frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-frame", type=Path, default=None)
    args = parser.parse_args()

    records = train_manifest_records()
    agent = SemanticAgent()
    prototypes = build_prototypes(agent, records)
    save_prototypes(prototypes)

    target = args.score_frame if args.score_frame else frame_path(records[0])
    result = agent.score_frame(target, prototypes)
    print(f"Saved semantic prototypes: {config.SEMANTIC_PROTOTYPES_PATH}")
    print(f"Prototype shape: {tuple(prototypes.shape)}")
    print(
        "Semantic score: "
        f"{result.frame_path} score={result.novelty_score:.4f} "
        f"is_novel={result.is_novel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
