# Recall: A Multi-Agent Perception System for Novelty Detection

## Problem
Doorbell/porch cameras record continuously, generating vast amounts of data where most frames contain routine background motion (e.g., traffic, foliage) that triggers false positives in motion-based systems, while storage costs grow unbounded. Supervised event detectors require thousands of labels per household—impractical to collect. The core challenge is to detect *novel* visual or temporal events without labeled training data, and to query remembered events via natural language.

## Method
Recall employs three independent, unsupervised perception agents that score novelty orthogonally:

- **Structural Agent**: Uses frozen DINOv2 to extract frame embeddings; novelty is the cosine distance to learned structural prototypes (visual structure deviation).
- **Semantic Agent**: Uses frozen CLIP to embed frames in joint image-text space; novelty is distance to semantic prototypes (category-level deviation).
- **Routine Agent**: Learns a self-supervised hourly histogram of frame timestamps from training data; novelty is the inverse likelihood of the hour (temporal deviation).

A Coordinator fuses the three scores with a conservative vote rule: unanimous agreement is treated as confident novel/familiar, while 2-1 split votes are treated as disagreement and routed to an active-learning request for a single human label. Confidently novel events are stored in memory (DINOv2 embedding, CLIP embedding, thumbnail, timestamp). Memory retrieval encodes a natural language query with CLIP text and returns the nearest neighbor by cosine similarity—pure retrieval, no generation.

All upstream perception components use fixed, pretrained weights (DINOv2, CLIP) or zero-label statistics. The current active-learning loop records disagreement cases for human labeling; folding those labels back into prototypes is left as future backend work.

## Results
We evaluate on a subset of the UCSD Pedestrian dataset (train/test split, seed=42), using binary labels derived from ground-truth masks (anomaly if mask exists, else normal). The label budget is fixed at 1% and 10% of the training slice (10 and 100 labels, respectively). We compare:

1. **Active-learning probe**: Linear probe on frozen DINOv2 embeddings trained on labels selected by the Recall active-learning loop.
2. **Random probe**: Same linear probe trained on an equal number of randomly chosen labels.
3. **Mandatory baseline**: Supervised CNN trained from scratch on the same random label slice.

Results (accuracy on held‑out test set of 200 frames):

| Label Budget | Method            | Accuracy |
|--------------|-------------------|----------|
| 1% (10 labels)   | Active-learning   | 0.500 |
|                | Random            | 0.650 |
|                | Baseline CNN      | 0.590 |
| 10% (100 labels) | Active-learning   | 0.670 |
|                | Random            | 0.650 |
|                | Baseline CNN      | 0.650 |

Agent ablation (novelty detection accuracy on test set, using Coordinator vote):

| Agent(s) Used      | Accuracy |
|--------------------|----------|
| Structural only    | 0.500 |
| Semantic only      | 0.500 |
| Routine only       | 0.405 |
| All three fused    | 0.405 |

The active-learning probe is mixed: it trails both the baseline CNN and random sampling at the 1% budget, but slightly exceeds both at the 10% budget. The structural and semantic agents individually achieve 0.500 accuracy on this binary slice, while the routine and fused variants reach 0.405; these ablation numbers should not be over-interpreted because the routine signal is confounded by the synthetic timestamp proxy.

## Limitations
- **Dataset proxy**: Evaluation uses UCSD Pedestrian as a stand‑in for porch footage; the Routine Agent’s temporal model relies on a synthetic hour mapping (Train→day, Test→evening) that may conflate routine learning with the train/test split.
- **Label scarcity**: Only 10–100 labels are available per budget, leading to high variance in measured accuracies.
- **Binary task**: The reduction to normal/anomaly discards fine‑grained categories (person, package, etc.) that would be relevant for real‑world deployment.
- **Retrieval only**: The system returns the single most similar memory; it does not aggregate multiple memories or provide temporal bounds.

Despite these limitations, Recall demonstrates a working end-to-end prototype for self-supervised novelty scoring, conservative coordinator disagreement routing, memory storage, and natural-language retrieval. The evaluation provides preliminary evidence at the 10% label budget, while the 1% result remains a negative result that should be reported plainly.
