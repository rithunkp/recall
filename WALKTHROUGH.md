# Recall System Walkthrough

## One-paragraph pitch
Recall is a multi-agent perception system designed for doorstep cameras that operates without labeled training data. It uses three independent perception agents (Structural, Semantic, and Routine) that vote on what counts as "novel" or unusual activity. When the agents disagree, the system requests a single human label for that ambiguous case rather than guessing. The system stores confidently novel events in memory and allows users to query what happened using plain English questions, enabling efficient storage and meaningful event retrieval from continuous video streams.

## How to run it
To reproduce the entire system from scratch:

1. **Environment setup**: Ensure Python 3.11 is installed, then run:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the full pipeline** (executes data preparation, all three agents, coordinator, and model smoke test):
   ```bash
   bash run.sh
   ```
   This command will:
   - Prepare data splits if missing (`data/prepare_data.py`)
   - Run Structural Agent (`agents/structural_agent.py`)
   - Run Semantic Agent (`agents/semantic_agent.py`)
   - Run Routine Agent (`agents/routine_agent.py`)
   - Run Coordinator (`coordinator/fuse.py`)
   - Execute model smoke test (`scripts/smoke_test_models.py`)

3. **Launch the Gradio app** for interactive querying:
   ```bash
   python demo/app.py
   ```

4. **Run evaluation** to see label efficiency results:
   ```bash
   python eval/run_eval.py
   ```
   This produces accuracy comparisons between active-learning label selection, random label selection, and the mandatory scratch-CNN baseline at 1% and 10% label budgets, saving results to `memory_artifacts/eval/results.json`.

## What's real vs. what's a stand-in
The following substitutions were made during development due to data availability constraints:

- **UCSD Pedestrian dataset used instead of self-staged porch footage**: The original plan relied on staged porch scenarios (package drop, stranger approaches, etc.), but UCSD Pedestrian frames serve as the fallback dataset throughout.
- **Synthetic timestamp proxy**: Since UCSD frames lack real timestamps, the Routine Agent uses a deterministic proxy where `TrainNNN` sequences map to daytime hours (08-17) and `TestNNN` sequences map to evening hours (18-23). This creates a confound where Routine/Coordinator results on test data may reflect the train/test split naming rather than pure temporal novelty.
- **Relabeled UCSD normal/anomaly class set**: Instead of the planned fine-grained classes (person/package/vehicle/animal/familiar), evaluation uses binary labels derived from UCSD ground-truth masks: a frame is labeled "anomaly" if a matching `*_gt` mask exists, otherwise "normal."
- **Current small eval sample size**: Evaluation uses 40 training examples per class and 15 test examples per class (80 train, 30 test total). The active-vs-random label comparison result should be interpreted with this limited sample size in mind.

## Architecture walkthrough
**Structural Agent** (`agents/structural_agent.py`)
- Uses frozen DINOv2 to embed each frame into a 384-dimensional vector
- Compares embeddings against a bank of prototype embeddings learned from training data
- Flags novelty based on cosine distance: high deviation from learned structural prototypes indicates potential novelty
- Captures raw visual/structural differences independent of semantic category

**Semantic Agent** (`agents/semantic_agent.py`)
- Uses frozen CLIP to embed frames into a joint image-text space (512-dimensional)
- Compares embeddings against prototype bank of familiar scenes
- Flags category-level deviation: if a frame doesn't resemble any familiar prototype in CLIP space, it's considered novel
- Detects semantic novelty like unfamiliar objects or scenes

**Routine Agent** (`agents/routine_agent.py`)
- Learns a self-supervised temporal model from frame timestamps (hour-of-day histograms)
- Scores novelty based on how unusual the timing is: events occurring at rare hours score higher
- Uses zero labels: learns routine patterns purely from when events happen in training data
- Directly addresses high-motion background failures (e.g., constant daytime traffic learned as routine)

**Coordinator** (`coordinator/fuse.py`)
- Fuses novelty scores from the three agents using a simple majority-of-three vote
- Strong agreement (2+ agents voting novel) → confident decision: store in memory or discard
- Disagreement (split votes) → routed to active learning: system requests one human label for the ambiguous case
- Records decisions and pending label requests in JSONL files under `memory_artifacts/`

**Memory + Retrieval** (`memory/store.py` + `memory/retrieve.py`)
- Confident novel memories store: DINOv2 embedding, CLIP embedding, thumbnail (frame path), and timestamp
- Retrieval encodes plain English questions via CLIP text encoder
- Performs brute-force cosine similarity search against stored memory CLIP embeddings
- Returns templated answer with the single best-matching memory's synthetic timestamp
- Pure retrieval approach: no generation, thus no hallucinated answers

## Results
The following results were obtained from the final evaluation run (as recorded in `memory_artifacts/eval/results.json`):

**Label efficiency comparison** (accuracy on held-out test set):
- At 1% label budget (4 labels total):
  - Active-learning probe: 0.500 accuracy
  - Random probe: 0.567 accuracy
  - Baseline CNN: 0.500 accuracy
- At 10% label budget (8 labels total):
  - Active-learning probe: 0.500 accuracy
  - Random probe: 0.633 accuracy
  - Baseline CNN: 0.500 accuracy

**Agent ablation** (novelty detection accuracy on test set):
- Structural-only: 0.500
- Semantic-only: 0.500
- Routine-only: 0.400
- All-three-fused: 0.400

**Important caveats**:
- The active-vs-random result currently shows random sampling performing better, opposite to the hypothesized direction
- Sample sizes are small (80 training frames, 30 test frames), limiting statistical significance
- Routine agent results are confounded with the UCSD train/test split naming due to the synthetic timestamp proxy

## Demo script
For the recorded demo session, follow this exact sequence:

1. **Start the demo application**: Launch `python demo/app.py` and wait for the Gradio interface to load
2. **Go to the Ask tab**: Click on the "3. Ask" tab to access the query interface
3. **Ask a contextual question**: Type "pedestrians walking on a walkway" into the question box and press Enter
4. **Observe the retrieval result**: The system should return a templated answer indicating a memory match with its associated synthetic timestamp (e.g., "Event detected at hour 10")
5. **Test a negative case**: Type "a bright red sports car parked indoors" to verify the system returns "No matching event found." when similarity falls below threshold
6. **Check boundary case**: Try "something happening outside" to see a response near the similarity threshold (0.20)