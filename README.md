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

**A multi-agent perception system that decides what's worth remembering.**

Recall is a self-supervised prototype for camera streams. Instead of recording everything a
camera sees, it uses three independent, label-free agents to judge whether a frame is
familiar, ambiguous, or novel. Familiar frames are discarded, confidently novel frames are
written to memory, and ambiguous frames — where the agents disagree — are queued for active
learning instead of being silently guessed at. Stored memories can later be searched with
plain-English questions through CLIP-based retrieval.

Built for a Deep Learning Hackathon, Track 4 — Self-Supervised & Representation Learning.

> **Status:** hackathon prototype. Presented honestly as such, not as a deployed security
> product — see [Known Limitations](#known-limitations) before drawing conclusions from any
> number in this document.

---

## Table of Contents

- [The Problem](#the-problem)
- [Architecture](#architecture)
- [Demo](#demo)
- [Reproducing This Project](#reproducing-this-project)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Known Limitations](#known-limitations)
- [Repository Structure](#repository-structure)
- [Team & Track](#team--track)

---

## The Problem

Doorstep cameras today either record continuously, wasting storage on hours of nothing, or
trigger on raw motion — which fails exactly where it matters most: a camera facing a road or
sidewalk fires constantly on ordinary background movement and buries any real event in
noise. Supervised object detectors would fix the false-positive problem, but need thousands
of per-household labeled examples that don't exist and aren't worth collecting.

Recall instead fuses three label-free novelty signals into a single agreement/disagreement
decision. When the signals agree, the system acts on its own. When they disagree, it asks
for exactly one human label instead of guessing — testing whether a frozen, self-supervised
representation combined with uncertainty-driven labeling is more label-efficient than random
sampling or training from scratch.

## Architecture

Three independent agents score every frame; a coordinator fuses their votes.

| Agent | Signal | What it catches |
|---|---|---|
| **Structural** (`agents/structural_agent.py`) | Frozen DINOv2 (`facebook/dinov2-small`), 384-dim embeddings | Raw visual/structural deviation from a train-only prototype bank |
| **Semantic** (`agents/semantic_agent.py`) | Frozen CLIP (`openai/clip-vit-base-patch32`), 512-dim embeddings | Category-level deviation — "doesn't look like anything filed as familiar" |
| **Routine** (`agents/routine_agent.py`) | Label-free hourly histogram over timestamps | Unusual *timing*, independent of appearance |

**Coordinator** (`coordinator/fuse.py`): unanimous familiar → discard; unanimous novel →
write to memory; a 2–1 split → ambiguous, routed to active learning rather than treated as a
confident majority.

**Active Learning** (`coordinator/active_learning.py`): disagreement cases are queued as
pending label requests. *Current gap, reported honestly:* the UI lets a human choose a label,
but there is no backend path yet that folds that label back into the prototypes or routine
stats — see [Known Limitations](#known-limitations).

**Memory + Retrieval** (`memory/store.py`, `memory/retrieve.py`): confidently novel frames
are stored with both embeddings, a thumbnail, and a timestamp. A typed question is
CLIP-text-encoded and matched via nearest-neighbor search against stored memories — **pure
retrieval, no generation**, so answers can't be hallucinated. The best match is returned only
if it clears a similarity threshold (0.20); otherwise the system says so.

## Demo

A Gradio app (`demo/app.py`) with four tabs:

1. **Scan** — one button, "Build Recall's Memory." Streams live progress and running
   Familiar / Ambiguous / Novel counts. Capped at a deterministic 400-frame subset
   (`seed=42`) for a CPU-safe interactive demo; a completion marker means relaunching skips
   straight to a ready state.
2. **Ask** — type a question, get a templated answer built only from retrieved facts, plus
   the matching thumbnail.
3. **Memories** — a gallery of everything stored, with a "Last matched" badge on whatever a
   query most recently retrieved.
4. **Pending Labels** — the active-learning disagreement queue, visible proof the loop is
   real and not just described.

Run it locally:

```bash
python demo/app.py
```

Or try the hosted version: **[huggingface.co/spaces/itzrick/recall](https://huggingface.co/spaces/itzrick/recall)**

## Reproducing This Project

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
bash run.sh              # data prep -> 3 agents -> coordinator -> smoke test, one command
python demo/app.py       # launch the interactive demo
python eval/run_eval.py  # linear probe, random probe, CNN baseline, agent ablation
```

Everything is seeded (`seed=42`, centralized in `config.py`). DINOv2 and CLIP are used
strictly as frozen feature extractors — the only trained components are the evaluation-stage
linear probe and the scratch CNN baseline. The test split is never touched for tuning.

## Evaluation

**Labeling scheme:** UCSD binary normal/anomaly — a frame is *anomaly* if a matching
ground-truth mask exists, else *normal*. Slice: 1000 train / 200 test frames.

| Budget | Labels | Active Probe | Random Probe | Scratch CNN |
|---:|---:|---:|---:|---:|
| 1%  | 10  | 0.600 | 0.650 | 0.590 |
| 10% | 100 | 0.670 | 0.650 | 0.650 |

At the 10% budget, the active-learning probe beats both the random-sampled probe and the
mandatory scratch-CNN baseline given identical label counts. At 1%, it does not — 10 absolute
labels is likely too few to separate signal from noise, and this is reported as-is rather
than smoothed over.

**Agent ablation** (novelty detection accuracy, labeled test slice):

| Signal | Accuracy |
|---|---:|
| Structural only | 0.600 |
| Semantic only | 0.600 |
| Routine only | 0.505 |
| All three fused | 0.750 |

Structural and semantic sit at chance level on this binary slice; routine and the fused
signal sit lower. The Routine Agent depends on a synthetic timestamp proxy tied to UCSD
source naming (see below), so these numbers shouldn't be read as evidence of genuine temporal
reasoning.

## Dataset

**Planned:** self-staged porch-camera footage across package delivery, stranger, familiar
face, pet, empty porch, off-hours visit, and road-like background traffic.

**Actual:** UCSD Pedestrian, used as a fallback due to hackathon time constraints. Frame
manifests live under `data/splits/`; UCSD's `*_gt` mask directories are excluded from
ordinary manifests. The public demo uses a lightweight tracked subset at
`data/frames_demo/`.

Because UCSD doesn't have the planned person/package/vehicle/animal/familiar categories,
evaluation uses the binary normal/anomaly labels described above instead of the richer
taxonomy originally envisioned.

## Known Limitations

- Dataset is a UCSD Pedestrian fallback, not the originally planned porch-camera footage.
- Evaluation labels are binary normal/anomaly, not the planned five-class taxonomy.
- Routine Agent timestamps are synthetic (`Train*` → daytime, `Test*` → evening) and
  partially confounded with UCSD's own source/split naming.
- The Pending Labels UI cannot yet update prototypes or stats — there is no apply-label
  backend, and this is disclosed rather than faked.
- Results are small-scale (10–100 labels) and preliminary; treat this as a hackathon
  prototype, not a validated production system.

## Repository Structure

```
recall/
├── PROJECT_STATE.md      # architecture, rules, hour-by-hour plan (source of truth)
├── RUNLOG.md             # append-only session log across build sessions/models
├── WALKTHROUGH.md        # plain-language walkthrough + demo script
├── README.md             # this file
├── requirements.txt
├── run.sh                # one-command reproducible pipeline
├── config.py             # seed=42, paths, thresholds — single source of truth
├── data/                 # raw/, frames/, frames_demo/, splits/, prepare_data.py
├── agents/               # structural_agent.py, semantic_agent.py, routine_agent.py
├── coordinator/          # fuse.py, active_learning.py
├── memory/               # store.py, retrieve.py
├── eval/                 # linear_probe.py, baseline_cnn.py, run_eval.py
├── demo/                 # app.py — the Gradio interface
└── report/               # report.md / recall_report.pdf
```

## Team & Track

Built for **Track 4 — Self-Supervised & Representation Learning**.

- Saavan Rajeev — AM.SC.U4AIE24065
- Rithun K P — AM.SC.U4AIE24041
- Balagopal — AM.SC.U4AIE24012

September 2026.
