"""Project-wide configuration for Recall.

All modules import seeds, paths, model IDs, thresholds, and split settings from
this file. Keep this small and explicit for the hackathon build.
"""

from __future__ import annotations

import random
from pathlib import Path


SEED = 42

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
SPLITS_DIR = DATA_DIR / "splits"
TRAIN_MANIFEST = SPLITS_DIR / "train.jsonl"
VAL_MANIFEST = SPLITS_DIR / "val.jsonl"
TEST_MANIFEST = SPLITS_DIR / "test.jsonl"

MEMORY_DIR = ROOT_DIR / "memory_artifacts"
EMBEDDINGS_DIR = MEMORY_DIR / "embeddings"
THUMBNAILS_DIR = MEMORY_DIR / "thumbnails"

REPORT_DIR = ROOT_DIR / "report"

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FRAME_EXTENSIONS = IMAGE_EXTENSIONS
FRAME_STRIDE = 30
FRAME_IMAGE_EXT = ".jpg"
STRUCTURAL_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "structural"
STRUCTURAL_PROTOTYPES_PATH = STRUCTURAL_EMBEDDINGS_DIR / "prototypes.npy"
STRUCTURAL_BATCH_SIZE = 8
STRUCTURAL_MAX_PROTOTYPES = 256
SEMANTIC_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "semantic"
SEMANTIC_PROTOTYPES_PATH = SEMANTIC_EMBEDDINGS_DIR / "prototypes.npy"
SEMANTIC_BATCH_SIZE = 8
SEMANTIC_MAX_PROTOTYPES = 256
ROUTINE_ARTIFACTS_DIR = MEMORY_DIR / "routine"
ROUTINE_STATS_PATH = ROUTINE_ARTIFACTS_DIR / "hourly_stats.json"
ROUTINE_NUM_HOURS = 24
ROUTINE_SMOOTHING = 1.0
ROUTINE_DAY_START_HOUR = 8
ROUTINE_DAY_HOURS = 10
ROUTINE_EVENING_START_HOUR = 18
ROUTINE_EVENING_HOURS = 6
MEMORY_RECORDS_PATH = MEMORY_DIR / "records.jsonl"
MEMORY_EMBEDDINGS_DIR = MEMORY_DIR / "memory_embeddings"
ACTIVE_LEARNING_PENDING_PATH = MEMORY_DIR / "active_learning" / "pending_labels.jsonl"
COORDINATOR_DECISIONS_PATH = MEMORY_DIR / "coordinator" / "decisions.jsonl"
COORDINATOR_MAX_FRAMES = 8
RETRIEVAL_SIMILARITY_THRESHOLD = 0.20
SCAN_FULL_DATASET_LIMIT = 400
SCAN_COMPLETE_MARKER_PATH = MEMORY_DIR / "scan_complete.json"
EVAL_ARTIFACTS_DIR = MEMORY_DIR / "eval"
EVAL_RESULTS_PATH = EVAL_ARTIFACTS_DIR / "results.json"
EVAL_TRAIN_PER_CLASS = 500
EVAL_TEST_PER_CLASS = 100
EVAL_LABEL_BUDGETS = (0.01, 0.10)
EVAL_MIN_LABELS_PER_CLASS = 2
EVAL_LINEAR_PROBE_MAX_ITER = 1000
EVAL_CNN_EPOCHS = 3
EVAL_CNN_BATCH_SIZE = 8
EVAL_IMAGE_SIZE = 64

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def _torch_cuda_available() -> bool:
    """Return CUDA availability without making config import depend on PyTorch."""
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


DEVICE = "cuda" if _torch_cuda_available() else "cpu"
DINO_MODEL_NAME = "facebook/dinov2-small"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

STRUCTURAL_NOVELTY_THRESHOLD = 0.35
SEMANTIC_NOVELTY_THRESHOLD = 0.35
ROUTINE_NOVELTY_THRESHOLD = 0.50
COORDINATOR_MIN_VOTES = 2


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible hackathon runs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass
