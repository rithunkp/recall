# RUNLOG.md

Append-only session log for **Recall**. Read this in full before starting any new session,
and especially after switching coding models/tools (Codex ↔ Claude Code ↔ different Claude
versions) — it's the fastest way to recover context without re-reading everything.

**Newest entry goes at the top.** Never delete or rewrite past entries — if a later session
changes an earlier decision, say so explicitly in the new entry instead.

Entry template:

```
## [Hour block] — <model/tool> — <one-line summary of what changed>
**Done:**
-
**Verified:**
-
**Blockers:**
-
**Next:**
-
```

---

## [Hour 1-2] — Codex — Structural Agent built on UCSD frame data

**Done:**
- Checked staged data under `data/raw/`: UCSD Pedestrian frame sequences are present and
  usable as the public/fallback stand-in for the pipeline.
- Updated config/data prep to accept `.tif`/`.tiff` frames and exclude UCSD `*_gt`
  ground-truth mask directories from manifests.
- Built `agents/structural_agent.py`: frozen DINOv2 loader, batched frame embedding,
  train-only prototype bank, saved `.npy` prototypes, and cosine-distance novelty scoring.
- Updated `run.sh` to use the repo-local `.venv` when present and include the Structural
  Agent step in the one-command path.
- Pinned `requirements.txt` to the installed working package versions and added `pillow`.

**Verified:**
- `data/prepare_data.py` prepared 18,560 real frame images after mask filtering:
  train=12,992, val=2,784, test=2,784.
- Confirmed generated manifests contain no `_gt` mask paths.
- Installed dependencies into `.venv`; PyTorch import reports `2.14.0+cpu`.
- `agents/structural_agent.py` saved train-only DINOv2 prototypes to
  `memory_artifacts/embeddings/structural/prototypes.npy` with shape `(256, 384)` and
  smoke-scored one train frame.
- `scripts/smoke_test_models.py` embeds one sample frame with DINOv2 `(1, 384)` and CLIP
  `(1, 512)`.
- `bash run.sh` completes end to end.

**Blockers:**
- None for Hour 1-2. Dataset is not the originally staged porch-camera set, so report/demo
  language should be honest that UCSD Pedestrian is being used as the fallback stand-in.

**Next:**
- Hour 2-3: build `agents/semantic_agent.py` with frozen CLIP image embeddings and
  prototype-distance novelty scoring, using the existing train manifest/prototype pattern
  and keeping all constants in `config.py`.

## [Hour 1-2] — Codex — blocked before Structural Agent because raw footage is missing

**Done:**
- Read `RUNLOG.md` in full before starting.
- Re-read `PROJECT_STATE.md`, including Architecture, Rules Compliance, and the
  Hour-by-Hour Plan.
- Confirmed Hour 1-2 should pick up with `agents/structural_agent.py`.

**Verified:**
- `data/raw/` exists but contains no staged footage or images.

**Blockers:**
- Cannot build or verify the DINOv2 frame embedding/prototype-distance path against real
  data yet, per instruction not to invent placeholder frames.

**Next:**
- Stage real videos/images under `data/raw/`, then rerun Hour 1-2: implement
  `agents/structural_agent.py` as a frozen DINOv2 embedding pipeline plus prototype-distance
  novelty scorer, keeping all seed/path/threshold values imported from `config.py`.

## [Hour 0-1] — Claude Code — repo scaffold + config centralization

**Done:**
- Repo scaffold created from `PROJECT_STATE.md`.
- `config.py` centralizes `seed=42` plus all paths/thresholds; every new module imports
  from it instead of hardcoding.
- Built: `requirements.txt`, `run.sh`, `data/prepare_data.py`,
  `scripts/smoke_test_models.py`.
- Placeholder modules created for agents, coordinator, memory, eval, and demo/report.

**Verified:**
- Python files compile successfully.
- `data/prepare_data.py` correctly refuses to invent data when raw footage is missing
  (fails loudly instead of faking manifests).

**Blockers:**
- `data/raw/` exists but has no staged footage yet — manifests and the DINOv2/CLIP smoke
  test can't run until real videos/images are placed there.

**Next:**
- Hour 1-2: build `agents/structural_agent.py` — frozen DINOv2 embedding pipeline +
  prototype-distance novelty scorer. Scaffold for it already exists and is ready to fill in.
- Stage/collect footage into `data/raw/` before or during this session so the smoke test
  and prototype buffer have something to run against.
