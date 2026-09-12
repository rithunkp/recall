"""Gradio judge demo for Recall."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from agents.routine_agent import RoutineAgent, load_stats
from agents.semantic_agent import SemanticAgent, load_prototypes as load_semantic_prototypes
from agents.structural_agent import StructuralAgent, load_prototypes as load_structural_prototypes
from coordinator.fuse import score_record
from memory.retrieve import MemoryRetriever
from memory.store import MemoryRecord
from PIL import Image


LABEL_OPTIONS = ["normal", "anomaly", "pedestrian", "biker", "cart", "skater", "vehicle"]
FRAME_EXTENSIONS = {suffix.lower() for suffix in config.FRAME_EXTENSIONS}


@lru_cache(maxsize=1)
def structural_agent() -> StructuralAgent:
    """Load the frozen Structural Agent once for the UI process."""
    return StructuralAgent()


@lru_cache(maxsize=1)
def semantic_agent() -> SemanticAgent:
    """Load the frozen Semantic Agent once for the UI process."""
    return SemanticAgent()


@lru_cache(maxsize=1)
def routine_agent() -> RoutineAgent:
    """Create the Routine Agent once for the UI process."""
    return RoutineAgent()


@lru_cache(maxsize=1)
def retriever() -> MemoryRetriever:
    """Load the frozen CLIP text retriever once for the UI process."""
    return MemoryRetriever()


@lru_cache(maxsize=1)
def structural_prototypes():
    """Load structural prototypes once."""
    return load_structural_prototypes()


@lru_cache(maxsize=1)
def semantic_prototypes():
    """Load semantic prototypes once."""
    return load_semantic_prototypes()


@lru_cache(maxsize=1)
def routine_stats():
    """Load routine stats once."""
    return load_stats()


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Load a JSONL manifest."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def preset_records(limit: int) -> list[dict[str, object]]:
    """Return a small validation preset for the Scan tab."""
    return load_manifest(config.VAL_MANIFEST)[:limit]


def folder_records(folder_text: str, limit: int) -> list[dict[str, object]]:
    """Build scan records from a folder of frame files."""
    folder = Path(folder_text).expanduser()
    if not folder.is_absolute():
        folder = config.ROOT_DIR / folder
    if not folder.exists():
        raise gr.Error(f"Folder not found: {folder}")
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in FRAME_EXTENSIONS
    )
    if not paths:
        raise gr.Error("No frame images found in that folder.")
    records = []
    for path in paths[:limit]:
        rel = path.relative_to(config.ROOT_DIR) if path.is_relative_to(config.ROOT_DIR) else path
        records.append(
            {
                "frame_path": str(rel),
                "source_path": str(rel),
                "scenario": "folder_scan",
                "timestamp_sec": None,
            }
        )
    return records


def short_frame_path(frame_path: str) -> str:
    """Return a shortened display path for the frame (scenario/frame)."""
    try:
        parts = Path(frame_path).parts
        if len(parts) >= 2:
            return str(Path(parts[-2]) / parts[-1])
    except Exception:
        pass
    return frame_path


def get_thumbnail_path(frame_path: str) -> str:
    """Return a PNG thumbnail path for display, converting TIFF if needed."""
    # frame_path is relative to ROOT_DIR
    full_path = config.ROOT_DIR / frame_path
    if not full_path.exists():
        return str(full_path)  # fallback
    # Create a hash of the full path to name the cache file
    hash_obj = hashlib.md5(str(full_path).encode())
    cache_name = hash_obj.hexdigest() + ".png"
    cache_path = config.THUMBNAILS_DIR / cache_name
    if not cache_path.exists():
        # Ensure thumbnail directory exists
        config.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        # Open image, convert to RGB, save as PNG
        with Image.open(full_path) as img:
            # Convert to RGB if necessary (e.g., TIFF may have different modes)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(cache_path, "PNG")
    return str(cache_path)


# Session-scoped stats for the status strip
_session_stats = {"frames_scanned": 0}


def scan_records(records: list[dict[str, object]]):
    """Yield live table rows while scoring records through the Coordinator."""
    rows = []
    for record in records:
        decision = score_record(
            record=record,
            structural_agent=structural_agent(),
            semantic_agent=semantic_agent(),
            routine_agent=routine_agent(),
            structural_prototypes=structural_prototypes(),
            semantic_prototypes=semantic_prototypes(),
            routine_stats=routine_stats(),
        )
        # Increment session frames scanned
        _session_stats["frames_scanned"] += 1

        # Frame column: HTML with tooltip showing full path
        display_name = short_frame_path(record["frame_path"])
        full_path = record["frame_path"]
        frame_html = f'<span title="{full_path}">{display_name}</span>'

        # Score columns
        struct_score = f"{decision.scores.structural_score:.4f}"
        sem_score = f"{decision.scores.semantic_score:.4f}"
        routine_score = f"{decision.scores.routine_score:.4f}"

        # Vote column: badge showing vote text
        vote_text = f"{decision.novel_votes}/3 novel"
        vote_html = f'<span style="background-color: #6c63ff; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.9em;">{vote_text}</span>'

        # Outcome column: mapped label with color
        outcome_map = {
            "familiar": ("Familiar", "#28a745"),  # green
            "ambiguous_majority_familiar": ("Ambiguous — needs a label", "#fd7e14"),  # orange
            "novel": ("Novel", "#17a2b8"),  # blue
        }
        label, color = outcome_map.get(decision.decision, (decision.decision, "#6c757d"))
        outcome_html = f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.9em;">{label}</span>'

        rows.append(
            [
                frame_html,
                struct_score,
                sem_score,
                routine_score,
                vote_html,
                outcome_html,
            ]
        )
        yield rows, f"Processed {len(rows)} of {len(records)}"


def run_scan(folder_text: str, sample_count: int, use_preset: bool):
    """Run folder or preset scan through existing Coordinator code."""
    # Reset session frames scanned for this scan run? We'll keep cumulative across session.
    count = max(1, int(sample_count))
    records = preset_records(count) if use_preset else folder_records(folder_text, count)
    yield from scan_records(records)


def read_memory_records() -> list[dict[str, object]]:
    """Load all memory records exactly as written by memory/store.py."""
    if not config.MEMORY_RECORDS_PATH.exists():
        return []
    with config.MEMORY_RECORDS_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def memory_gallery_items():
    """Return Gradio gallery items for stored memories."""
    items = []
    for record in read_memory_records():
        # Use thumbnail path for display (converted to PNG if needed)
        thumbnail_path = get_thumbnail_path(str(record["thumbnail_path"]))
        image_path = config.ROOT_DIR / thumbnail_path
        if not image_path.exists():
            continue
        caption_parts = []
        if record.get("timestamp_sec") is not None:
            caption_parts.append(f"timestamp {float(record['timestamp_sec']):.2f}s")
        elif record.get("routine_hour") is not None:
            caption_parts.append(f"synthetic hour {record['routine_hour']}")
        if record.get("category"):
            caption_parts.append(str(record["category"]))
        caption_parts.append(str(record["frame_path"]))
        items.append((str(image_path), " | ".join(caption_parts)))
    return items


def refresh_memories():
    """Refresh memory gallery and count text."""
    items = memory_gallery_items()
    count = len(items)
    # Proper pluralization
    count_text = f"{count} memory" if count == 1 else f"{count} memories"
    # Empty state message
    if count == 0:
        status_msg = "No memories yet — run a scan first"
    else:
        status_msg = ""
    return items, count_text, status_msg


def ask_memory(question: str):
    """Ask retrieval and return its exact templated answer."""
    query = question.strip()
    if not query:
        return "No matching event found.", []
    result = retriever().retrieve(query)
    gallery = []
    if result.matched and result.memory is not None:
        thumbnail = get_thumbnail_path(str(result.memory["thumbnail_path"]))
        thumbnail_path = config.ROOT_DIR / thumbnail
        if thumbnail_path.exists():
            gallery = [(str(thumbnail_path), "")]  # empty caption to avoid duplicate answer
    return result.answer, gallery


def read_pending_labels() -> list[dict[str, object]]:
    """Load pending active-learning label requests."""
    if not config.ACTIVE_LEARNING_PENDING_PATH.exists():
        return []
    with config.ACTIVE_LEARNING_PENDING_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def pending_rows() -> list[list[str]]:
    """Return pending-label table rows."""
    rows = []
    for index, item in enumerate(read_pending_labels()):
        rows.append(
            [
                str(index),
                str(item.get("frame_path", "")),
                str(item.get("decision", "")),
                f"{item.get('novel_votes', 0)}/3 novel",
            ]
        )
    return rows


def refresh_pending():
    """Refresh pending-label queue display."""
    rows = pending_rows()
    choices = [row[0] for row in rows]
    # Empty state message
    if len(rows) == 0:
        pending_msg = "No pending labels — the agents agree on all scanned frames"
    else:
        pending_msg = ""
    return rows, gr.update(choices=choices, value=choices[0] if choices else None), pending_msg


def submit_label(pending_index: str, label: str) -> str:
    """Acknowledge label input without inventing a backend apply path."""
    if pending_index in (None, ""):
        return "No pending case selected."
    pending = read_pending_labels()
    try:
        item = pending[int(pending_index)]
    except (IndexError, ValueError):
        return "Pending case not found."
    return (
        f"Label '{label}' selected for {item.get('frame_path')}. "
        "No label-application backend exists yet, so prototypes/stats were not changed."
    )


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks demo."""
    with gr.Blocks(title="Recall") as app:
        gr.Markdown("# Recall")
        # Status strip across tabs
        with gr.Row():
            status_display = gr.Markdown(
                value="Frames scanned: 0 | Memories: 0 | Pending labels: 0",
                label="Status",
            )
        with gr.Tab("1. Scan"):
            gr.Markdown("Run the three perception agents over a folder of frames and see what each one decided.")
            with gr.Row():
                folder = gr.Textbox(label="Frame folder", value="data/frames/UCSDped1/Train013")
                sample_count = gr.Slider(1, 8, value=3, step=1, label="Frames")
                use_preset = gr.Checkbox(value=True, label="Use validation preset")
            scan_button = gr.Button("Run scan", variant="primary")
            scan_status = gr.Textbox(label="Scan progress", interactive=False)
            scan_table = gr.Dataframe(
                headers=[
                    "Frame",
                    "Structural score",
                    "Semantic score",
                    "Routine score",
                    "Vote",
                    "Outcome",
                ],
                datatype=["html", "str", "str", "str", "html", "html"],
                label="Decisions",
                interactive=False,
            )
            scan_button.click(
                run_scan,
                inputs=[folder, sample_count, use_preset],
                outputs=[scan_table, scan_status],
            )
            # Update status strip after scan
            def update_status(*_):
                frames = _session_stats["frames_scanned"]
                memories = len(read_memory_records())
                pending = len(read_pending_labels())
                return f"Frames scanned: {frames} | Memories: {memories} | Pending labels: {pending}"
            scan_button.click(
                update_status,
                inputs=None,
                outputs=status_display,
            )
            # Also update status on app load to show initial counts
            app.load(
                update_status,
                inputs=None,
                outputs=status_display,
            )

        with gr.Tab("2. Memories"):
            gr.Markdown("Review stored memories discovered by the system.")
            with gr.Row():
                refresh_memory_button = gr.Button("Refresh memories")
                memory_count = gr.Textbox(label="Count", interactive=False)
            memories = gr.Gallery(label="Memories", columns=4, height=420)
            memory_empty_msg = gr.Markdown(label="")
            refresh_memory_button.click(
                refresh_memories,
                outputs=[memories, memory_count, memory_empty_msg],
            )
            app.load(
                refresh_memories,
                outputs=[memories, memory_count, memory_empty_msg],
            )

        with gr.Tab("3. Ask"):
            gr.Markdown("Ask a question in natural language to retrieve the best matching memory.")
            # Example question buttons
            with gr.Row():
                ex1 = gr.Button("Was there unusual activity?", size="sm")
                ex2 = gr.Button("Show me daytime activity", size="sm")
                ex3 = gr.Button("Any pedestrian activity?", size="sm")
            question = gr.Textbox(label="Question", value="pedestrians walking on a walkway")
            ask_button = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", interactive=False)
            ask_gallery = gr.Gallery(label="Match", columns=1, height=320)
            ask_button.click(
                ask_memory,
                inputs=question,
                outputs=[answer, ask_gallery],
            )
            # Example button clicks
            ex1.click(lambda: "Was there unusual activity?", None, question)
            ex2.click(lambda: "Show me daytime activity", None, question)
            ex3.click(lambda: "Any pedestrian activity?", None, question)

        with gr.Tab("4. Pending Labels"):
            gr.Markdown("When the agents disagree, they ask you for one label here — that's the active-learning loop.")
            with gr.Row():
                refresh_pending_button = gr.Button("Refresh pending labels")
            pending_table = gr.Dataframe(
                headers=["Index", "Frame", "Decision", "Vote"],
                datatype=["str", "str", "str", "str"],
                label="Pending labels",
                interactive=False,
            )
            pending_choice = gr.Dropdown(label="Pending case", choices=[])
            pending_label = gr.Dropdown(label="Label", choices=LABEL_OPTIONS, value="normal")
            submit_button = gr.Button("Submit label", variant="primary")
            label_status = gr.Textbox(label="Status", interactive=False)
            pending_empty_msg = gr.Markdown(label="")
            refresh_pending_button.click(
                refresh_pending,
                outputs=[pending_table, pending_choice, pending_empty_msg],
            )
            app.load(
                refresh_pending,
                outputs=[pending_table, pending_choice, pending_empty_msg],
            )
            submit_button.click(
                submit_label,
                inputs=[pending_choice, pending_label],
                outputs=label_status,
            )

    return app


def main() -> int:
    """Launch the Gradio demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        prevent_thread_lock=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())