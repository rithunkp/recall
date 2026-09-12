# PROJECT_STATE.md - Recall

Read this file before changing the repo. It is the current source of truth for the
architecture, constraints, dataset substitutions, reproducibility path, demo flow, and
evaluation results.

---

## 1. Project Summary

**Recall - A Multi-Agent Perception System That Decides What's Worth Remembering**

Recall is a self-supervised perception prototype for camera streams. It uses three frozen
or label-free agents to decide whether a frame is familiar, ambiguous, or novel. Familiar
frames are discarded, confident novel frames are stored as memories, and ambiguous frames
are queued for active learning instead of being guessed. Stored memories can then be queried
with natural language using CLIP-text retrieval.

The product story is a doorstep/security camera that does not save everything and does not
fire on every bit of routine motion. The system only remembers events the agents confidently
agree are novel, and it asks for human help only on disagreement cases.

## 2. Track And Rules

**Track 4 - Self-Supervised & Representation Learning**

Core rule commitments:
- `seed=42` is centralized in `config.py`.
- DINOv2 and CLIP remain frozen throughout.
- The only trained models are the eval-stage linear probe and scratch CNN baseline.
- Test split is not used for threshold tuning or upstream design decisions.
- `run.sh` is the one-command backend reproducibility path.
- Eval numbers are reported honestly, including negative or weak results.

## 3. Actual Dataset

Original target data was self-staged porch footage with categories such as package,
stranger, familiar person, pet, vehicle, empty porch, off-hours visit, and road-like
background motion.

Actual hackathon dataset:
- **UCSD Pedestrian** fallback frames are used instead of self-staged porch footage.
- Prepared frame manifests exist under `data/splits/`.
- UCSD `_gt` mask directories are excluded from normal frame manifests.
- The Gradio/Hugging Face demo uses a tracked lightweight frame subset under
  `data/frames_demo/` when the full local `data/frames/` tree is unavailable.

Important limitation:
- UCSD does not provide the planned porch categories, so final eval uses binary
  `normal / anomaly` labels derived from UCSD ground-truth masks.

## 4. Architecture

### Structural Agent - `agents/structural_agent.py`

Uses frozen DINOv2 image embeddings:
- Model: `facebook/dinov2-small`
- Embedding size: 384
- Train-only prototype bank path:
  `memory_artifacts/embeddings/structural/prototypes.npy`
- Prototype bank shape verified earlier: `(256, 384)`
- Novelty score: distance from nearest train prototype.

### Semantic Agent - `agents/semantic_agent.py`

Uses frozen CLIP image embeddings:
- Model: `openai/clip-vit-base-patch32`
- Embedding size: 512
- Train-only prototype bank path:
  `memory_artifacts/embeddings/semantic/prototypes.npy`
- Prototype bank shape verified earlier: `(256, 512)`
- Novelty score: cosine distance from nearest train prototype.

### Routine Agent - `agents/routine_agent.py`

Uses a label-free hourly histogram over timestamp-like metadata:
- Stats path: `memory_artifacts/routine/hourly_stats.json`
- UCSD frames do not contain real wall-clock timestamps.
- Synthetic timestamp proxy:
  - UCSD `TrainNNN` sources map to daytime hours `08-17`.
  - UCSD `TestNNN` sources map to evening hours `18-23`.

Important limitation:
- The Routine Agent signal is partly confounded with UCSD source/split naming because
  evening hours are tied to test-source sequence names. Do not describe Routine ablation
  performance as a clean temporal win.

### Coordinator - `coordinator/fuse.py`

Fuses the three novelty votes:
- Each agent score is thresholded using constants from `config.py`.
- Unanimous familiar -> confident familiar, discard.
- Unanimous novel -> confident novel, write memory.
- Split 2-1 votes -> ambiguous, append a pending active-learning request.

This intentionally treats 2-1 as ambiguous rather than confident majority because the demo
goal is to show active learning when agents disagree.

### Active Learning - `coordinator/active_learning.py`

Currently stores pending label requests only:
- Queue path: `memory_artifacts/active_learning/pending_labels.jsonl`
- The Gradio UI can select a label for a pending row, but there is no backend function yet
  that folds that label into prototypes/stats. The UI reports that honestly and does not
  invent an update path.

### Memory Store - `memory/store.py`

Confident novel frames write:
- JSONL metadata at `memory_artifacts/records.jsonl`
- DINOv2 `.npy` embedding
- CLIP `.npy` embedding
- thumbnail/frame reference
- timestamp or synthetic routine hour
- optional category only when one is actually attached

### Retrieval - `memory/retrieve.py`

Pure retrieval, no generation:
- CLIP text-encodes the typed question.
- Brute-force cosine search over stored CLIP memory embeddings.
- Returns only the single best match if similarity clears
  `RETRIEVAL_SIMILARITY_THRESHOLD = 0.20`.
- Otherwise returns `No matching event found.`
- Answer text is templated and only includes stored facts.

## 5. Demo App

Gradio app:
- Entry point: `demo/app.py`
- Hugging Face README metadata points to `app_file: demo/app.py`.

Tabs:
- **Scan**: one-button setup flow, **Build Recall's Memory**.
- **Ask**: natural-language query box wired directly to `MemoryRetriever.retrieve()`.
- **Memories**: gallery of stored memory records with TIFF-to-PNG thumbnail conversion.
- **Pending Labels**: displays active-learning disagreement queue.

Scan behavior:
- Uses `SCAN_FULL_DATASET_LIMIT = 400`.
- Samples deterministically with `seed=42` across both UCSD train-source and test-source
  sequences.
- Streams progress as a Gradio generator:
  `frames processed / total`, plus running counts for Familiar, Ambiguous, Novel.
- Writes `memory_artifacts/scan_complete.json` when the configured demo scan completes.
- On relaunch, if the marker exists for the current limit, the app starts in ready state and
  orders Ask first.
- Ask matches also flag the matched memory as `Last matched` in the Memories gallery.

Verified UI flow:
- Clean temp-artifact scan with the configured 400-frame demo limit completed:
  `336 familiar`, `64 ambiguous`, `0 novel`.
- Relaunch marker behavior reported ready without rescanning.
- Existing real memory callback showed `1 memory`.
- Existing pending-label callback showed `28` rows.
- Query `pedestrians walking on a walkway` matched `smoke_memory_val_001` with similarity
  `0.2976` and flagged it as `Last matched`.
- Local Gradio server returned HTTP `200`.
- Browser-control clicking was blocked by a local computer-use plugin path error, so tab
  verification was done through real callbacks plus HTTP server response.

## 6. Reproducibility

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the backend pipeline:

```bash
bash run.sh
```

`run.sh` performs:
1. data preparation if manifests are missing
2. Structural Agent prototype/scoring path
3. Semantic Agent prototype/scoring path
4. Routine Agent stats/scoring path
5. Coordinator pass
6. DINOv2/CLIP smoke test

Run the Gradio demo:

```bash
python demo/app.py
```

Run evaluation:

```bash
python eval/run_eval.py
```

## 7. Evaluation

Final eval artifact:
- `memory_artifacts/eval/results.json`

Labeling scheme:
- UCSD binary `normal / anomaly`.
- A frame is `anomaly` if a matching UCSD `*_gt` mask exists.
- Otherwise it is `normal`.

Eval slice:
- Train slice size: `1000`
- Test slice size: `200`
- Classes: `normal`, `anomaly`
- Label budgets are fractions of the train slice:
  - `1%` -> `10` labels
  - `10%` -> `100` labels

### Results Table

| Budget | Labels | Active Probe | Random Probe | Scratch CNN Baseline |
|---:|---:|---:|---:|---:|
| 1% | 10 | 0.500 | 0.650 | 0.590 |
| 10% | 100 | 0.670 | 0.650 | 0.650 |

Label counts:
- 1% active: 6 normal, 4 anomaly
- 1% random: 5 normal, 5 anomaly
- 10% active: 47 normal, 53 anomaly
- 10% random: 55 normal, 45 anomaly

Primary claim:
- At 1%, the active linear probe did **not** beat the scratch CNN:
  `0.500` vs `0.590`.
- At 10%, the active linear probe slightly beat the scratch CNN:
  `0.670` vs `0.650`.

Secondary claim:
- At 1%, active sampling did **not** beat random sampling:
  `0.500` vs `0.650`.
- At 10%, active sampling slightly beat random sampling:
  `0.670` vs `0.650`.

Conclusion:
- The results are mixed and preliminary. The 10% result supports the intended direction,
  but the 1% result does not.

### Agent Ablation

Novelty detection accuracy on the labeled test slice:

| Signal | Accuracy |
|---|---:|
| Structural only | 0.500 |
| Semantic only | 0.500 |
| Routine only | 0.405 |
| All three fused | 0.405 |

Interpretation:
- Structural and semantic signals perform at chance on this binary UCSD slice.
- Routine and all-three-fused are weaker here.
- Because the Routine Agent uses a synthetic timestamp proxy tied to UCSD source naming,
  these ablation numbers should not be over-interpreted as real temporal reasoning.

## 8. Known Limitations

- Dataset is UCSD Pedestrian fallback data, not the original porch-camera dataset.
- Evaluation labels are binary normal/anomaly, not the planned
  person/package/vehicle/animal/familiar classes.
- Routine timestamps are synthetic and partially confounded with UCSD source naming.
- Stored memory count is small in the current checked artifact state.
- Pending-label UI cannot actually update prototypes/stats until an apply-label backend is
  added.
- The Hugging Face Space needs the tracked demo subset/artifacts pushed with Git LFS.
- Results are small-scale and should be presented as a hackathon prototype, not a deployed
  security product.

## 9. Repo Structure

```text
recall/
├── PROJECT_STATE.md
├── README.md
├── WALKTHROUGH.md
├── requirements.txt
├── run.sh
├── config.py
├── data/
│   ├── raw/
│   ├── frames/
│   ├── frames_demo/
│   ├── splits/
│   └── prepare_data.py
├── agents/
│   ├── structural_agent.py
│   ├── semantic_agent.py
│   └── routine_agent.py
├── coordinator/
│   ├── fuse.py
│   └── active_learning.py
├── memory/
│   ├── store.py
│   └── retrieve.py
├── eval/
│   ├── linear_probe.py
│   ├── baseline_cnn.py
│   └── run_eval.py
├── demo/
│   └── app.py
├── memory_artifacts/
└── report/
    └── report.md
```

## 10. Hour-By-Hour Build Status

| Hours | Status |
|---|---|
| 0-1 | Repo scaffold, config, requirements, data prep, model smoke test, `run.sh` |
| 1-2 | Structural Agent with frozen DINOv2 prototypes |
| 2-3 | Semantic Agent with frozen CLIP image prototypes |
| 3-4 | Routine Agent with synthetic UCSD hourly histogram |
| 4-5 | Coordinator fusion, active-learning pending queue, memory store |
| 5-6 | CLIP-text retrieval and minimal query demo |
| 6-7 | Eval: linear probe, random probe, scratch CNN baseline, agent ablation |
| 7-8+ | Gradio judge demo, walkthrough/docs, HF Space preparation |

## 11. Next Concrete Steps

1. Push the Space-ready files and required demo artifacts to
   `https://huggingface.co/spaces/itzrick/recall`.
2. Record the demo using the walkthrough flow.
3. Update/export `report/report.md` if the submission requires a PDF.
4. Add a real apply-label backend after the hackathon if the active-learning UI needs to
   become more than a proof of loop.
