# PROJECT_STATE.md — Recall

> Read this whole file before writing any code. This is the single source of truth for the
> 8-hour Deep Learning Hackathon build. If anything here is ambiguous, ask before guessing —
> especially around the train/test split and the mandatory baseline.

---

## 1. Title & Pitch

**Recall — A Multi-Agent Perception System That Decides What's Worth Remembering**

A doorstep camera with no labeled training data, built from three independent perception
agents that vote on what counts as "novel," request a human label only when they disagree,
and let you query what happened in plain English.

## 2. Track

**Track 4 — Self-Supervised & Representation Learning**

- Scoped problem: learn a representation, then evaluate it under a fixed label budget.
- Mandatory baseline: supervised CNN trained from scratch on the identical random 1%/10%
  labeled subset.
- Primary metric: downstream classification accuracy at each fixed label budget.

## 3. Problem Statement

Doorbell/porch cameras record everything and surface nothing. Two concrete failures with
today's cameras motivate this project:

1. **Motion-trigger cameras fail in high-motion scenes.** A camera facing a road, a
   sidewalk, or blowing trees triggers constantly on background motion that is completely
   ordinary, so the "motion detected" signal stops being informative exactly where it's
   needed most.
2. **Storage blow-up.** Recording continuously (or on every motion trigger) wastes storage
   on hours of nothing, making local storage impractical and cloud storage expensive.

Supervised event detectors ("package," "stranger," "pet") would fix the false-positive
problem, but need thousands of labeled clips per household — a dataset that doesn't exist
and isn't worth building per-porch. Recall's bet: you don't need labels to know something is
*novel*; you only need labels for the rare cases where your novelty signals disagree.

## 4. Architecture

### 4.1 Perception layer — three independent agents, each scoring novelty its own way

| Agent | Signal | How it scores novelty |
|---|---|---|
| **Structural Agent** | Frozen **DINOv2** | Embeds each frame, compares against stored prototype embeddings (cosine/L2 distance). Flags raw visual/structural deviation — catches things that just *look* different, independent of category. |
| **Semantic Agent** | Frozen **CLIP** | Embeds each frame in joint image-text space, flags category-level deviation ("this doesn't look like anything I've filed as familiar"). |
| **Routine Agent** | Self-supervised temporal model | Rolling point-process/histogram over event timestamps, learned with **zero labels**. Scores *when* something happens, not what it looks like — a person at 3am is novel even if they'd be unremarkable at 3pm. This is the agent that directly solves the "camera facing a road" failure mode: constant daytime traffic gets learned as routine and stops triggering, while an off-hours event still fires. |

### 4.2 Coordinator

Fuses the three scores (weighted vote or simple rule, e.g. majority-of-three with tunable
per-agent weight/threshold).

- **Strong agreement** → confident decision, write to memory or discard (this is where the
  storage savings comes from: only novel events get a persisted memory).
- **Disagreement** → routed to active learning instead of guessed at.

### 4.3 Active learning loop

Coordinator disagreement = an explicit uncertainty query. The system asks for **one** human
label on exactly the ambiguous case, folds it into the prototype set, and moves on. This
replaces random 1%/10% label sampling with **uncertainty-driven sampling** — this is the
actual research claim of the project, not just an eval protocol.

### 4.4 Memory + retrieval

Each stored memory keeps: DINOv2 embedding, CLIP embedding, thumbnail, timestamp. A typed
question is CLIP-text-encoded and matched via nearest-neighbor search against stored
memories — **pure retrieval, no generation**, so no hallucinated answers.

### 4.5 Explainability (stretch goal, only if time remains)

Attention rollout over DINOv2's patch attention, shown alongside any flagged event — a "why
was this novel" heatmap next to every memory.

## 5. Evaluation Plan (Track 4 requirement)

Label a small slice of collected events into a few classes: `person / package / vehicle /
animal / familiar`.

Compare, at matched label budgets (e.g. 1% and 10%):

1. Linear probe on frozen DINOv2 embeddings, trained on labels chosen by the **active-learning
   loop**.
2. Same linear probe, trained on the same number of **randomly** chosen labels (ablation —
   isolates the value of active sampling, not the mandatory baseline).
3. **Mandatory baseline**: supervised CNN trained from scratch on the identical random
   1%/10% slice.

- **Primary metric**: downstream classification accuracy at each fixed label budget, vs. the
  scratch-CNN baseline.
- **Secondary claim (the actual novel result)**: active-sampled labels beat randomly-sampled
  labels at the same budget.

### Ablations

- Structural-only vs. semantic-only vs. routine-only vs. all-three-fused → which signal(s)
  actually drive correct novelty decisions.
- Active-learning label selection vs. random selection, at matched budget.

## 6. Dataset

Self-staged footage — no public dataset covers this exact combination, and staging gives
full control for both training and live demo.

**Scenarios to record** (repeat each a few times, varied lighting/angle if possible):
- Package drop
- Stranger approaches
- Familiar face
- Pet crossing frame
- Empty porch (long stretches — this is most of the data, as in real life)
- Off-hours visit (night/early morning)
- High-motion background clip(s) simulating "camera facing a road" (cars, foliage) — needed
  to demonstrate the Routine Agent's advantage over naive motion triggers

**Splits**: fixed train / val / test, `seed=42`. Test split is touched exactly once, at the
end, for the final reported number — never for tuning agent thresholds, coordinator weights,
or the linear probe.

**Fallback if no camera/footage time**: use a handful of public short video clips or a
webcam recording session as a stand-in; the pipeline and eval protocol don't depend on the
footage being literally your porch, only on having the six scenario categories represented.

## 7. Rules Compliance Checklist

- [ ] Same team as course project (or solo if project was solo)
- [ ] One submitting lead owns repo + upload
- [ ] `seed=42` everywhere (data split, agent thresholds if learned, linear probe, baseline
      CNN)
- [ ] Repo runs end-to-end with **one command**
- [ ] Train only on train split; test split touched once, at the end
- [ ] Baseline (scratch CNN) implemented and reported against
- [ ] Report ≤ 4 pages: problem · method · results table (with baseline) · one ablation ·
      limitations
- [ ] Demo recorded
- [ ] Every dataset / pretrained weight / reference snippet cited in the report
- [ ] No fabricated numbers, no lucky-seed cherry-picking, honest negative results reported
      if that's what happens

## 8. Judging Criteria Mapping

| Criterion | Weight | How Recall addresses it |
|---|---|---|
| Problem quality | 20% | Concrete, named failure mode (motion-trigger false positives + storage cost) with a specific mechanism (Routine Agent) that fixes it |
| Beats mandatory baseline | 20% | Linear probe (active-sampled) vs. scratch CNN at 1%/10% |
| Depth of solution | 20% | Three-agent fusion + active-learning coordinator, not a single classifier |
| Experimental rigor | 15% | Fixed splits, seed=42, ablation across agents and sampling strategy |
| Originality of implementation | 15% | Uncertainty-driven active learning in place of random subset sampling is the explicit research claim |
| Demo / presentation | 10% | Live query demo ("show me when the delivery happened") over the retrieval index |

## 9. Tech Stack

- **Language**: Python 3.11
- **Frameworks**: PyTorch, `transformers` (or `timm`) for DINOv2, `open_clip` or
  `transformers` CLIPModel for CLIP
- **Vector search**: brute-force cosine similarity to start (dataset is small); swap to
  FAISS only if retrieval is visibly slow
- **Video/frame handling**: OpenCV / `decord` for frame extraction
- **Storage**: memories as JSON + `.npy` embeddings, or a single SQLite file — keep it
  simple, no external DB
- **UI (optional, only if time allows)**: a small Gradio or Streamlit app for the query demo;
  a CLI is a perfectly fine fallback
- **Repro**: `requirements.txt` pinned, single `run.sh` / `make all` entry point, `seed=42`
  set in one config file imported everywhere

## 10. Repo Structure

```
recall/
├── PROJECT_STATE.md
├── README.md
├── requirements.txt
├── run.sh                      # one-command end-to-end run
├── config.py                   # seed=42, paths, thresholds — single source of truth
├── data/
│   ├── raw/                    # staged footage
│   ├── splits/                 # train/val/test manifests (fixed, seed=42)
│   └── prepare_data.py         # frame extraction, split generation
├── agents/
│   ├── structural_agent.py     # DINOv2 embed + prototype distance
│   ├── semantic_agent.py       # CLIP embed + prototype distance
│   └── routine_agent.py        # timestamp histogram / point process
├── coordinator/
│   ├── fuse.py                 # agreement/disagreement logic
│   └── active_learning.py      # uncertainty query -> label -> fold into prototypes
├── memory/
│   ├── store.py                # write/read memories (embeddings, thumbnail, timestamp)
│   └── retrieve.py             # CLIP-text query -> nearest-neighbor search
├── eval/
│   ├── linear_probe.py         # frozen-embedding linear probe, active vs. random labels
│   ├── baseline_cnn.py         # mandatory scratch-CNN baseline
│   └── run_eval.py             # produces the results table + ablation numbers
├── demo/
│   └── app.py                  # CLI or Gradio query demo
└── report/
    └── report.md               # -> exported to PDF, ≤4 pages
```

## 11. Hour-by-Hour Plan (8 hours)

| Hours | Work |
|---|---|
| 0–1 | Repo skeleton, `config.py` with `seed=42`, env setup, load DINOv2 + CLIP and confirm inference runs, stage/collect footage if not done already |
| 1–2 | Structural Agent: frame embedding pipeline, prototype buffer, distance-based novelty score |
| 2–3 | Semantic Agent: CLIP embedding, novelty score against familiar prototypes |
| 3–4 | Routine Agent: timestamp histogram, novelty score by time-of-day; generate/label the high-motion "road" clip to sanity-check it does NOT fire on routine traffic |
| 4–5 | Coordinator: fuse three scores, define agreement/disagreement + threshold, memory store (write on confident decisions) |
| 5–6 | Retrieval: CLIP-text query encoder + nearest-neighbor search; minimal CLI/Gradio query demo |
| 6–7 | Eval: label the small slice, run linear probe (active-sampled vs. random-sampled) vs. scratch-CNN baseline at 1%/10%; run the agent ablation |
| 7–8 | Report (≤4 pages), record demo, clean repo, verify one-command reproducibility end-to-end, submit |

**If running behind**: cut explainability (stretch), cut the Gradio UI in favor of a CLI,
cut the 10% label budget and only report 1% — but never cut the baseline comparison or the
active-vs-random ablation; those are graded directly.

## 12. Open Decisions / Ask Before Assuming

- Exact coordinator fusion rule (simple majority vs. weighted score threshold) — start with
  majority vote, revisit only if time allows.
- Exact size of the labeled eval slice — pick the smallest number that gives a stable
  accuracy estimate (rough rule of thumb: at least ~10-15 examples per class in the test
  split).
- Whether to log routine-agent time buckets hourly or in finer granularity — start hourly.

---