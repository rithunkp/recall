## [Hour 9-10 UI Flow] — Codex — One-button memory build flow for Gradio demo

**Done:**
- Read `RUNLOG.md` and `PROJECT_STATE.md` in full before starting. Reconciled a repo/log discrepancy: the current working `RUNLOG.md` only contains the short Hour 8-9 eval entry, while `git diff` shows older run history was removed earlier in the working tree. This session did not repair that unrelated truncation.
- Reworked `demo/app.py` Scan from a manual frame-folder/frame-count developer flow into a single primary **Build Recall's Memory** action.
- Preserved the existing Gradio tab structure, TIFF-to-PNG thumbnail conversion, Ask retrieval path, Pending Labels display, and persistent status strip.
- Added `SCAN_FULL_DATASET_LIMIT = 400` and `SCAN_COMPLETE_MARKER_PATH` to `config.py`. The 400-frame cap is a CPU-friendly representative demo subset instead of an unbounded scan over all roughly 18,560 frames.
- The default scan samples deterministically with `config.SEED` across both UCSD train-source and test-source sequences, then streams each frame through the existing `coordinator.fuse.score_record()` pipeline with the existing `StructuralAgent`, `SemanticAgent`, `RoutineAgent`, prototype loaders, and routine stats.
- Added persisted scan-complete marker handling. If the configured demo scan has already completed, relaunch shows a ready state, hides the prominent Build button, shows a smaller Rescan button, and orders Ask first.
- Wired Ask-to-Memories state: `memory.retrieve.MemoryRetriever.retrieve()` still returns the exact templated answer, and a matched record is flagged as `Last matched` in the Memories gallery.
- Updated `WALKTHROUGH.md` Demo script to remove old folder/count instructions and use: launch app -> click Build Recall's Memory once -> wait for scan -> use Ask/Memories/Pending Labels.

**Verified:**
- `python -m py_compile demo\app.py config.py` passed.
- `git diff --check` passed.
- Dependency-backed import/build check passed using the uv-managed Python 3.14 interpreter plus the repo `.venv` packages: Gradio `6.27.0`, `build_app()` returns a `Blocks` instance.
- Demo sampler smoke check returned 10 records with both `train_source` and `test_source` represented.
- Full configured clean-state scan was run through the callback against a temporary memory-artifact area with `SCAN_FULL_DATASET_LIMIT=400`. It streamed progress at 0/400, 1/400, then every 50 frames, and completed with `336 familiar`, `64 ambiguous`, `0 novel`.
- The clean-state marker path verified: before scan `scan_ready=False`; after scan `scan_ready=True`; simulated relaunch status reported `Memory ready - scan already completed`.
- Real Memories callback verified against existing artifacts: `1 memory`.
- Real Pending Labels callback verified against existing artifacts: `28` pending rows.
- Ask callback verified with query `pedestrians walking on a walkway`: returned the exact retrieval template with similarity `0.2976`, matched `smoke_memory_val_001`, returned one thumbnail, and the Memories gallery included the `Last matched` badge.
- Updated Gradio app launched at `http://127.0.0.1:7860/`; HTTP check returned `200`.

**Blockers:**
- Browser-control clicking could not be performed because the computer-use layer failed with `failed to write kernel assets: The system cannot find the path specified.` Tabs were verified through real Gradio callbacks plus the live HTTP server instead.
- The existing backend `append_pending_label_request()` binds its default path at import time. During the temp full-scan verification, the 64 ambiguous requests briefly appended to the real pending JSONL despite temporary config patching; this session restored the file to the pre-existing 28 rows. The production app path is unaffected, but future temp-path tests should account for that default-argument binding.
- Pending-label submission still cannot mutate prototypes/stats because `coordinator/active_learning.py` only exposes queue append behavior. The UI does not invent a new apply-label path.

**Next:**
- Hour 7-8 finalization should record the demo with the new flow: launch app, click Build Recall's Memory once if no scan marker exists, wait for streaming completion, then use Ask/Memories/Pending Labels. Before submission, run the final reproducibility check and make sure report language stays honest about UCSD fallback data and the Routine Agent timestamp confound.

---

## [Hour 8-9 Eval] — Claude Code — Fixed eval slice sizes and added CNN loss monitoring

**Done:**
- Increased EVAL_TRAIN_PER_CLASS from 40 to 500 and EVAL_TEST_PER_CLASS from 15 to 100 in config.py
- Modified baseline_cnn.py to log training loss per epoch and check if loss decreased
- Ran eval/run_eval.py with updated slice sizes

**Verified:**
- New slice sizes: train_slice_size=1000, test_slice_size=200
- At 1% budget: 10 labels (5 normal, 5 anomaly for random; 6 normal, 4 anomaly for active)
- At 10% budget: 100 labels (55 normal, 45 anomaly for random; 47 normal, 53 anomaly for active)
- Baseline CNN results:
  * 1% budget: accuracy 0.590, loss NON-DECREASING (possible non-learning)
  * 10% budget: accuracy 0.650, loss DECREASED (model learning)
- Active vs Random probe accuracy:
  * 1% budget: active 0.500 vs random 0.650 (random better)
  * 10% budget: active 0.670 vs random 0.650 (active better by 0.020)
- Agent ablation results:
  * structural_only: 0.500
  * semantic_only: 0.500
  * routine_only: 0.405
  * all_three_fused: 0.405

**Blockers:** None.

**Next:** Append this entry to RUNLOG.md and finalize session.

---
