---
title: Recall
emoji: 🎥
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: "5.9.1"
app_file: demo/app.py
pinned: false
---

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

```powers