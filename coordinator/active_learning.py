"""Record coordinator disagreements as pending active-learning labels."""

from __future__ import annotations

import json
from pathlib import Path

import config


def append_pending_label_request(
    payload: dict[str, object], path: Path = config.ACTIVE_LEARNING_PENDING_PATH
) -> None:
    """Append an ambiguous case for later human labeling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
