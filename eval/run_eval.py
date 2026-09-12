"""Run the Hour 6-7 evaluation on a small UCSD binary slice."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from agents.routine_agent import RoutineAgent, load_stats, record_hour
from agents.semantic_agent import SemanticAgent, load_prototypes as load_semantic_prototypes
from agents.semantic_agent import frame_path as semantic_frame_path
from agents.structural_agent import StructuralAgent
from agents.structural_agent import frame_path as structural_frame_path
from agents.structural_agent import load_prototypes as load_structural_prototypes
from coordinator.fuse import fuse_scores
from eval.baseline_cnn import train_and_eval as train_cnn
from eval.linear_probe import train_and_eval as train_probe


SEQUENCE_RE = re.compile(r"^(Train|Test)(\d+)$", re.IGNORECASE)


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Load a JSONL manifest."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sequence_name(record: dict[str, object]) -> str:
    """Find the source sequence directory for a UCSD frame."""
    parts = Path(str(record.get("source_path", ""))).parts
    for part in reversed(parts):
        if SEQUENCE_RE.match(part):
            return part
    return Path(str(record["frame_path"])).parent.name


def mask_path(record: dict[str, object]) -> Path | None:
    """Return the matching UCSD anomaly mask path if one exists."""
    sequence = sequence_name(record)
    match = SEQUENCE_RE.match(sequence)
    if not match or match.group(1).lower() != "test":
        return None
    source = Path(str(record.get("source_path", "")))
    if not source.name:
        return None
    gt_dir = config.RAW_DATA_DIR / str(record["scenario"]) / "Test" / f"{sequence}_gt"
    candidate = gt_dir / f"{source.stem}.bmp"
    return candidate if candidate.exists() else None


def label_record(record: dict[str, object]) -> dict[str, object]:
    """Label UCSD frame as normal/anomaly using official mask presence."""
    labeled = dict(record)
    labeled["label"] = "anomaly" if mask_path(record) is not None else "normal"
    return labeled


def balanced_slice(
    records: list[dict[str, object]], per_class: int, labels: tuple[str, ...]
) -> list[dict[str, object]]:
    """Select a deterministic balanced slice."""
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        labeled = label_record(record)
        if labeled["label"] in labels:
            buckets[str(labeled["label"])].append(labeled)

    rng = random.Random(config.SEED)
    selected: list[dict[str, object]] = []
    for label in labels:
        rng.shuffle(buckets[label])
        if len(buckets[label]) < per_class:
            raise ValueError(f"Not enough {label} records for eval slice.")
        selected.extend(buckets[label][:per_class])
    rng.shuffle(selected)
    return selected


def active_selected(
    train_slice: list[dict[str, object]], budget: int
) -> list[dict[str, object]]:
    """Use pending active-learning queue first, then fill for class balance."""
    by_path = {str(record["frame_path"]): record for record in train_slice}
    selected: list[dict[str, object]] = []
    if config.ACTIVE_LEARNING_PENDING_PATH.exists():
        with config.ACTIVE_LEARNING_PENDING_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                record = by_path.get(str(payload["frame_path"]))
                if record is not None and record not in selected:
                    selected.append(record)
                if len(selected) >= budget:
                    return selected[:budget]

    rng = random.Random(config.SEED)
    remaining = [record for record in train_slice if record not in selected]
    rng.shuffle(remaining)
    needed_labels = sorted({str(record["label"]) for record in train_slice})
    for label in needed_labels:
        if any(record["label"] == label for record in selected):
            continue
        for record in remaining:
            if record["label"] == label:
                selected.append(record)
                remaining.remove(record)
                break
    selected.extend(remaining[: max(0, budget - len(selected))])
    return selected[:budget]


def random_selected(train_slice: list[dict[str, object]], budget: int) -> list[dict[str, object]]:
    """Deterministically sample random labels from the same train slice."""
    rng = random.Random(config.SEED)
    labels = sorted({str(record["label"]) for record in train_slice})
    selected: list[dict[str, object]] = []
    remaining = list(train_slice)
    for label in labels:
        choices = [record for record in remaining if record["label"] == label]
        chosen = rng.choice(choices)
        selected.append(chosen)
        remaining.remove(chosen)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, budget - len(selected))])
    return selected[:budget]


def budget_size(train_slice: list[dict[str, object]], fraction: float) -> int:
    """Convert a label-budget fraction into a usable class-covered count."""
    class_count = len({str(record["label"]) for record in train_slice})
    raw = int(round(len(train_slice) * fraction))
    return max(raw, class_count * config.EVAL_MIN_LABELS_PER_CLASS)


def embed_records(agent: StructuralAgent, records: list[dict[str, object]]) -> np.ndarray:
    """Embed records with frozen DINOv2."""
    return agent.embed_images([structural_frame_path(record) for record in records])


def evaluate_ablation(
    test_slice: list[dict[str, object]],
    structural_agent: StructuralAgent,
    semantic_agent: SemanticAgent,
    routine_agent: RoutineAgent,
) -> dict[str, float]:
    """Evaluate agent novelty votes against binary anomaly labels."""
    structural_prototypes = load_structural_prototypes()
    semantic_prototypes = load_semantic_prototypes()
    routine_stats = load_stats()
    correct = defaultdict(int)

    for record in test_slice:
        expected = record["label"] == "anomaly"
        structural_embedding = structural_agent.embed_images([structural_frame_path(record)])[0]
        semantic_embedding = semantic_agent.embed_images([semantic_frame_path(record)])[0]
        structural_score = structural_agent.score_embedding(
            structural_embedding, structural_prototypes
        )
        semantic_score = semantic_agent.score_embedding(semantic_embedding, semantic_prototypes)
        routine_score = routine_agent.score_record(record, routine_stats).novelty_score
        scores = fuse_scores(structural_score, semantic_score, routine_score)
        predictions = {
            "structural_only": scores.structural_vote,
            "semantic_only": scores.semantic_vote,
            "routine_only": scores.routine_vote,
            "all_three_fused": scores.structural_vote
            or scores.semantic_vote
            or scores.routine_vote,
        }
        for name, prediction in predictions.items():
            correct[name] += int(prediction == expected)

    total = len(test_slice)
    return {name: value / total for name, value in correct.items()}


def main() -> int:
    """Run linear probes, scratch-CNN baseline, and agent ablation."""
    config.set_seed()
    train_manifest = load_manifest(config.TRAIN_MANIFEST)
    test_manifest = load_manifest(config.TEST_MANIFEST)
    labels = ("normal", "anomaly")
    train_slice = balanced_slice(train_manifest, config.EVAL_TRAIN_PER_CLASS, labels)
    test_slice = balanced_slice(test_manifest, config.EVAL_TEST_PER_CLASS, labels)

    structural_agent = StructuralAgent()
    train_embeddings = embed_records(structural_agent, train_slice)
    test_embeddings = embed_records(structural_agent, test_slice)
    test_labels = [str(record["label"]) for record in test_slice]

    rows: list[dict[str, object]] = []
    for fraction in config.EVAL_LABEL_BUDGETS:
        budget = budget_size(train_slice, fraction)
        active = active_selected(train_slice, budget)
        random_labels = random_selected(train_slice, budget)
        active_indices = [train_slice.index(record) for record in active]
        random_indices = [train_slice.index(record) for record in random_labels]
        rows.append(
            {
                "budget_fraction_of_train_slice": fraction,
                "label_count": budget,
                "active_probe_accuracy": train_probe(
                    train_embeddings[active_indices],
                    [str(record["label"]) for record in active],
                    test_embeddings,
                    test_labels,
                ),
                "random_probe_accuracy": train_probe(
                    train_embeddings[random_indices],
                    [str(record["label"]) for record in random_labels],
                    test_embeddings,
                    test_labels,
                ),
                "baseline_cnn_accuracy": train_cnn(random_labels, test_slice),
                "active_label_counts": {
                    label: sum(record["label"] == label for record in active)
                    for label in labels
                },
                "random_label_counts": {
                    label: sum(record["label"] == label for record in random_labels)
                    for label in labels
                },
            }
        )

    ablation = evaluate_ablation(
        test_slice,
        structural_agent,
        SemanticAgent(),
        RoutineAgent(),
    )
    result = {
        "labeling_scheme": "UCSD binary normal/anomaly: anomaly if matching *_gt mask exists; otherwise normal.",
        "train_slice_size": len(train_slice),
        "test_slice_size": len(test_slice),
        "class_set": labels,
        "results": rows,
        "ablation_accuracy": ablation,
        "routine_confound_note": "Synthetic UCSD time proxy maps TestNNN sources to evening, so routine signal is confounded with source split naming.",
    }
    config.EVAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_RESULTS_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
