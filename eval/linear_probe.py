"""Linear probe evaluation on frozen DINOv2 embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import config


def train_and_eval(
    train_embeddings: np.ndarray,
    train_labels: list[str],
    test_embeddings: np.ndarray,
    test_labels: list[str],
) -> float:
    """Train a multinomial linear probe and return test accuracy."""
    config.set_seed()
    if len(set(train_labels)) < 2:
        raise ValueError("Linear probe needs at least two classes in selected labels.")
    classifier = LogisticRegression(
        max_iter=config.EVAL_LINEAR_PROBE_MAX_ITER,
        random_state=config.SEED,
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)
    return float(accuracy_score(test_labels, predictions))
