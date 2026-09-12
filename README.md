

# Recall

Recall is a self-supervised, multi-agent perception system for camera streams. It uses
three independent signals to decide what is worth remembering:

- **Structural Agent**: frozen DINOv2 visual embeddings.
- **Semantic Agent**: frozen CLIP image embeddings.
- **Routine Agent**: label-free hourly routine statistics.

The Coordinator compares their votes. Frames the agents confidently agree are familiar are
discarded. Frames they confidently agree are novel become memories. Disagreements are sent
to a pending-label queue for active learning instead of being guessed.

Stored memories can be queried with plain English. Retrieval is CLIP nearest-neighbor
search only: no LLM generation and no invented categories.

## Demo

Run locally:

```bash
python demo/app.py
```

Demo flow:
1. Open the app.
2. Click **Build Recall's Memory** once in Scan.
3. Watch the live Familiar / Ambiguous / Novel counts update.
4. Use Ask to query stored memories.
5. Check Memories for thumbnails and the `Last matched` badge.
6. Check Pending Labels for agent disagreement cases.

The app uses `SCAN_FULL_DATASET_LIMIT = 400` from `config.py` for an interactive CPU-safe
demo scan. On relaunch, a completed scan marker lets the app skip rebuilding memory and open
with Ask as the primary flow.

## Reproduce

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the backend pipeline:

```bash
bash run.sh
```

Run evaluation:

```bash
python eval/run_eval.py
```

## Results

Evaluation uses UCSD Pedestrian fallback data with binary `normal / anomaly` labels derived
from UCSD ground-truth masks.

| Budget | Labels | Active Probe | Random Probe | Scratch CNN |
|---:|---:|---:|---:|---:|
| 1% | 10 | 0.500 | 0.650 | 0.590 |
| 10% | 100 | 0.670 | 0.650 | 0.650 |

Agent ablation on the labeled test slice:

| Signal | Accuracy |
|---|---:|
| Structural only | 0.500 |
| Semantic only | 0.500 |
| Routine only | 0.405 |
| All three fused | 0.405 |

The results are mixed: the active linear probe beats the scratch CNN and random probe at
10%, but not at 1%. This is reported as-is rather than cherry-picked.

## Important Limitations

- The intended porch-camera footage was replaced with UCSD Pedestrian fallback frames.
- UCSD does not contain the planned fine-grained classes, so eval is binary normal/anomaly.
- Routine Agent timestamps are synthetic: UCSD train-source sequences map to daytime hours
  and test-source sequences map to evening hours. This creates a confound with source split
  naming and should not be presented as clean temporal reasoning.
- Pending Labels proves the active-learning loop visually, but the backend currently stores
  pending requests only; it does not yet fold submitted labels back into prototypes.

For the full project state, architecture, caveats, and handoff notes, read
[`PROJECT_STATE.md`](PROJECT_STATE.md).
