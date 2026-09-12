"""Smoke test frozen DINOv2 and CLIP image embedding on one prepared frame."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config


def first_manifest_frame() -> Path:
    """Return the first train frame from the prepared manifest."""
    if not config.TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {config.TRAIN_MANIFEST}; run data/prepare_data.py first."
        )
    with config.TRAIN_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            return config.ROOT_DIR / record["frame_path"]
    raise RuntimeError(f"No frames listed in {config.TRAIN_MANIFEST}")


def embed_with_dino(image: Image.Image) -> torch.Tensor:
    """Load frozen DINOv2 and embed one image."""
    processor = AutoImageProcessor.from_pretrained(config.DINO_MODEL_NAME)
    model = AutoModel.from_pretrained(config.DINO_MODEL_NAME).to(config.DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
    return embedding.cpu()


def embed_with_clip(image: Image.Image) -> torch.Tensor:
    """Load frozen CLIP and embed one image."""
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    with torch.inference_mode():
        embedding = model.get_image_features(**inputs)
        if not isinstance(embedding, torch.Tensor):
            embedding = embedding.pooler_output
    return embedding.cpu()


def main() -> int:
    """Run end-to-end model loading and image embedding checks."""
    config.set_seed()
    frame_path = first_manifest_frame()
    image = Image.open(frame_path).convert("RGB")

    dino_embedding = embed_with_dino(image)
    clip_embedding = embed_with_clip(image)

    print(f"Sample frame: {frame_path.relative_to(config.ROOT_DIR)}")
    print(f"DINOv2 embedding shape: {tuple(dino_embedding.shape)}")
    print(f"CLIP embedding shape: {tuple(clip_embedding.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
