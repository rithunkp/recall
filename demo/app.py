"""Gradio judge demo for Recall."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from agents.routine_agent import RoutineAgent, load_stats
from agents.semantic_agent import SemanticAgent, load_prototypes as load_semantic_prototypes
from agents.structural_agent import StructuralAgent, load_prototypes as load_structural_prototypes
from coordinator.fuse import score_record
from memory.retrieve import MemoryRetriever


LABEL_OPTIONS = ["normal", "anomaly", "pedestrian", "biker", "cart", "skater", "vehicle"]
FRAME_EXTENSIONS = {suffix.lower() for suffix in config.FRAME_EXTENSIONS}
SCAN_HEADERS = [
    "Frame",
    "Structural score",
    "Semantic score",
    "Routine score",
    "Vote",
    "Outcome",
]


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


def demo_manifest_path(path: Path) -> Path:
    """Return a Space-friendly demo manifest when the full manifest is absent."""
    demo_name = path.with_name(path.stem + "_demo" + path.suffix)
    if path.exists():
        return path
    if demo_name.exists():
        return demo_name
    return path


def portable_frame_record(record: dict[str, object]) -> dict[str, object]:
    """Point a manifest record at the tracked demo frame copy when needed."""
    frame_path = str(record.get("frame_path", ""))
    full_path = config.ROOT_DIR / frame_path
    if full_path.exists():
        return record

    normalized = frame_path.replace("\\", "/")
    demo_path = config.ROOT_DIR / "data" / "frames_demo" / normalized
    if demo_path.exists():
        updated = dict(record)
        updated["frame_path"] = str(demo_path.relative_to(config.ROOT_DIR))
        updated["source_path"] = str(demo_path.relative_to(config.ROOT_DIR))
        return updated
    return record


def source_partition(record: dict[str, object]) -> str:
    """Return UCSD source partition from the staged source path."""
    source = str(record.get("source_path", "")).replace("/", "\\")
    if "\\Test\\" in source:
        return "test_source"
    if "\\Train\\" in source:
        return "train_source"
    return "unknown_source"


def sampled_demo_records(limit: int | None = None) -> list[dict[str, object]]:
    """Sample a representative demo subset from train/test sequence sources."""
    max_records = int(limit or config.SCAN_FULL_DATASET_LIMIT)
    if max_records <= 0:
        raise gr.Error("Scan limit must be at least 1.")

    records = load_manifest(demo_manifest_path(config.TRAIN_MANIFEST)) + load_manifest(
        demo_manifest_path(config.TEST_MANIFEST)
    )
    by_partition: dict[str, list[dict[str, object]]] = {
        "train_source": [],
        "test_source": [],
        "unknown_source": [],
    }
    for record in records:
        by_partition[source_partition(record)].append(portable_frame_record(record))

    rng = random.Random(config.SEED)
    for partition_records in by_partition.values():
        rng.shuffle(partition_records)

    selected: list[dict[str, object]] = []
    preferred = ["train_source", "test_source", "unknown_source"]
    while len(selected) < max_records and any(by_partition.values()):
        for partition in preferred:
            if len(selected) >= max_records:
                break
            if by_partition[partition]:
                selected.append(by_partition[partition].pop())

    rng.shuffle(selected)
    return selected


def folder_records(folder_text: str, limit: int) -> list[dict[str, object]]:
    """Build scan records from a folder of frame files for hidden debugging."""
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
    for path in paths[: max(1, int(limit))]:
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
    """Return a shortened display path for the frame."""
    try:
        parts = Path(frame_path).parts
        if len(parts) >= 2:
            return str(Path(parts[-2]) / parts[-1])
    except Exception:
        pass
    return frame_path


def get_thumbnail_path(frame_path: str) -> str:
    """Return a PNG thumbnail path for display, converting TIFF if needed."""
    full_path = Path(frame_path)
    if not full_path.is_absolute():
        full_path = config.ROOT_DIR / full_path
    if not full_path.exists():
        return str(full_path)

    cache_name = hashlib.md5(str(full_path).encode()).hexdigest() + ".png"
    cache_path = config.THUMBNAILS_DIR / cache_name
    if not cache_path.exists():
        config.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(full_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(cache_path, "PNG")
    return str(cache_path)


def read_memory_records() -> list[dict[str, object]]:
    """Load all memory records exactly as written by memory/store.py."""
    if not config.MEMORY_RECORDS_PATH.exists():
        return []
    with config.MEMORY_RECORDS_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_pending_labels() -> list[dict[str, object]]:
    """Load pending active-learning label requests."""
    if not config.ACTIVE_LEARNING_PENDING_PATH.exists():
        return []
    with config.ACTIVE_LEARNING_PENDING_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def scan_marker() -> dict[str, object] | None:
    """Return completed scan metadata if the demo scan has already run."""
    if not config.SCAN_COMPLETE_MARKER_PATH.exists():
        return None
    with config.SCAN_COMPLETE_MARKER_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("completed") is not True:
        return None
    return payload


def scan_ready() -> bool:
    """Return whether a completed scan marker exists for the configured limit."""
    marker = scan_marker()
    return bool(marker and marker.get("limit") == config.SCAN_FULL_DATASET_LIMIT)


def write_scan_marker(
    *, total: int, familiar: int, ambiguous: int, novel: int
) -> dict[str, object]:
    """Persist demo scan completion metadata."""
    payload = {
        "completed": True,
        "seed": config.SEED,
        "limit": config.SCAN_FULL_DATASET_LIMIT,
        "frames_processed": total,
        "familiar": familiar,
        "ambiguous": ambiguous,
        "novel": novel,
    }
    config.SCAN_COMPLETE_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.SCAN_COMPLETE_MARKER_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def status_text(prefix: str | None = None) -> str:
    """Build the persistent status strip text."""
    memories = len(read_memory_records())
    pending = len(read_pending_labels())
    marker = scan_marker()
    if marker:
        lead = f"Memory ready - {marker.get('frames_processed', 0)} frames scanned"
    else:
        lead = f"Ready to build demo dataset ({config.SCAN_FULL_DATASET_LIMIT} frames)"
    if prefix:
        lead = prefix
    return f"{lead} | Memories: {memories} | Pending labels: {pending}"


def memory_gallery_items(last_matched_id: str | None = None):
    """Return Gradio gallery items for stored memories."""
    items = []
    for record in read_memory_records():
        thumbnail_path = Path(get_thumbnail_path(str(record["thumbnail_path"])))
        if not thumbnail_path.exists():
            continue
        caption_parts = []
        if record.get("memory_id") == last_matched_id:
            caption_parts.append("Last matched")
        if record.get("timestamp_sec") is not None:
            caption_parts.append(f"timestamp {float(record['timestamp_sec']):.2f}s")
        elif record.get("routine_hour") is not None:
            caption_parts.append(f"synthetic hour {record['routine_hour']}")
        if record.get("category"):
            caption_parts.append(str(record["category"]))
        caption_parts.append(str(record["frame_path"]))
        items.append((str(thumbnail_path), " | ".join(caption_parts)))
    return items


def refresh_memories(last_matched_id: str | None = None):
    """Refresh memory gallery and count text."""
    items = memory_gallery_items(last_matched_id)
    count = len(items)
    count_text = f"{count} memory" if count == 1 else f"{count} memories"
    status_msg = "No memories yet - build Recall's Memory first." if count == 0 else ""
    return items, count_text, status_msg


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
    pending_msg = "No pending labels - the agents agree on all scanned frames." if not rows else ""
    return rows, gr.update(choices=choices, value=choices[0] if choices else None), pending_msg


def scan_row(record: dict[str, object], decision) -> list[str]:
    """Convert a Coordinator decision to one UI table row."""
    display_name = html.escape(short_frame_path(str(record["frame_path"])))
    full_path = html.escape(str(record["frame_path"]))
    frame_html = f'<span title="{full_path}">{display_name}</span>'
    vote_text = f"{decision.novel_votes}/3 novel"
    vote_html = (
        '<span style="background-color: #4f46e5; color: white; padding: 2px 6px; '
        f'border-radius: 4px; font-size: 0.9em;">{vote_text}</span>'
    )
    outcome_map = {
        "familiar": ("Familiar", "#15803d"),
        "ambiguous_majority_familiar": ("Ambiguous - needs a label", "#c2410c"),
        "ambiguous_majority_novel": ("Ambiguous - needs a label", "#c2410c"),
        "novel": ("Novel", "#c2410c"),
    }
    label, color = outcome_map.get(decision.decision, (decision.decision, "#475569"))
    outcome_html = (
        f'<span style="background-color: {color}; color: white; padding: 2px 6px; '
        f'border-radius: 4px; font-size: 0.9em;">{html.escape(label)}</span>'
    )
    return [
        frame_html,
        f"{decision.scores.structural_score:.4f}",
        f"{decision.scores.semantic_score:.4f}",
        f"{decision.scores.routine_score:.4f}",
        vote_html,
        outcome_html,
    ]


def select_scan_records(use_advanced: bool, folder_text: str, advanced_limit: int):
    """Choose the configured demo dataset or hidden advanced debug records."""
    if use_advanced:
        return folder_records(folder_text, advanced_limit), "Advanced folder scan"
    return sampled_demo_records(), f"Demo dataset ({config.SCAN_FULL_DATASET_LIMIT} frames)"


def build_recall_memory(
    use_advanced: bool,
    folder_text: str,
    advanced_limit: int,
    force_rescan: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Iterator[tuple[object, str, str, object, str, str, object, object, str, object, object]]:
    """Stream the demo scan through the existing agent/coordinator pipeline."""
    if scan_ready() and not force_rescan and not use_advanced:
        memories, count_text, memory_msg = refresh_memories()
        pending_table, pending_choice, pending_msg = refresh_pending()
        ready = status_text("Memory ready - scan already completed")
        yield (
            gr.update(value=[]),
            ready,
            ready,
            memories,
            count_text,
            memory_msg,
            pending_table,
            pending_choice,
            pending_msg,
            gr.update(visible=False),
            gr.update(visible=True),
        )
        return

    records, dataset_label = select_scan_records(use_advanced, folder_text, advanced_limit)
    total = len(records)
    rows = []
    familiar = ambiguous = novel = 0
    running = f"{dataset_label}: 0/{total} | Familiar: 0 · Ambiguous: 0 · Novel: 0"
    yield (
        gr.update(value=rows),
        running,
        status_text(running),
        *refresh_memories(),
        *refresh_pending(),
        gr.update(visible=False),
        gr.update(visible=True),
    )

    for index, record in enumerate(records, start=1):
        decision = score_record(
            record=record,
            structural_agent=structural_agent(),
            semantic_agent=semantic_agent(),
            routine_agent=routine_agent(),
            structural_prototypes=structural_prototypes(),
            semantic_prototypes=semantic_prototypes(),
            routine_stats=routine_stats(),
        )
        if decision.decision == "familiar":
            familiar += 1
        elif decision.decision == "novel":
            novel += 1
        else:
            ambiguous += 1
        rows.append(scan_row(record, decision))
        running = (
            f"{dataset_label}: {index}/{total} | Familiar: {familiar} · "
            f"Ambiguous: {ambiguous} · Novel: {novel}"
        )
        progress(index / total, desc=running)
        yield (
            rows,
            running,
            status_text(running),
            *refresh_memories(),
            *refresh_pending(),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    if not use_advanced:
        write_scan_marker(total=total, familiar=familiar, ambiguous=ambiguous, novel=novel)
    complete = (
        f"Memory ready - {total} frames scanned | Familiar: {familiar} · "
        f"Ambiguous: {ambiguous} · Novel: {novel}"
    )
    yield (
        rows,
        complete,
        status_text(complete),
        *refresh_memories(),
        *refresh_pending(),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def ask_memory(question: str, last_matched_id: str | None):
    """Ask retrieval and return its exact templated answer."""
    query = question.strip()
    if not query:
        return "No matching event found.", [], last_matched_id, *refresh_memories(last_matched_id)
    result = retriever().retrieve(query)
    gallery = []
    matched_id = last_matched_id
    if result.matched and result.memory is not None:
        matched_id = str(result.memory.get("memory_id", ""))
        thumbnail = Path(get_thumbnail_path(str(result.memory["thumbnail_path"])))
        if thumbnail.exists():
            gallery = [(str(thumbnail), "View this in Memories - flagged as Last matched")]
    return result.answer, gallery, matched_id, *refresh_memories(matched_id)


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
        "No backend apply-label hook exists yet, so prototypes/stats were not changed."
    )


def initial_build_visibility() -> tuple[object, object]:
    """Set first-load button visibility from persisted scan state."""
    if scan_ready():
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=True), gr.update(visible=False)


def initial_status() -> str:
    """Return first-load status text."""
    if scan_ready():
        return status_text("Memory ready - scan already completed")
    return status_text()


def build_tabs(default_ready: bool):
    """Build tabs, ordering Ask first after memory has been prepared."""
    scan_components: dict[str, object] = {}
    memory_components: dict[str, object] = {}
    ask_components: dict[str, object] = {}
    pending_components: dict[str, object] = {}

    def scan_tab() -> None:
        gr.Markdown(
            f"Build the demo memory once from a representative sampled subset: "
            f"Demo dataset ({config.SCAN_FULL_DATASET_LIMIT} frames)."
        )
        with gr.Row():
            build_button = gr.Button("Build Recall's Memory", variant="primary", visible=not default_ready)
            rescan_button = gr.Button("Rescan demo dataset", size="sm", visible=default_ready)
        with gr.Accordion("Advanced", open=False):
            use_advanced = gr.Checkbox(value=False, label="Use a custom frame folder")
            folder = gr.Textbox(label="Frame folder", value="data/frames/UCSDped1/Train013")
            advanced_limit = gr.Slider(1, 64, value=8, step=1, label="Debug frame limit")
        scan_status = gr.Textbox(label="Scan progress", interactive=False)
        scan_table = gr.Dataframe(
            headers=SCAN_HEADERS,
            datatype=["html", "str", "str", "str", "html", "html"],
            label="Decisions",
            interactive=False,
        )
        force_build = gr.Checkbox(value=False, visible=False)
        force_rescan = gr.Checkbox(value=True, visible=False)
        scan_components.update(
            build_button=build_button,
            rescan_button=rescan_button,
            use_advanced=use_advanced,
            folder=folder,
            advanced_limit=advanced_limit,
            scan_status=scan_status,
            scan_table=scan_table,
            force_build=force_build,
            force_rescan=force_rescan,
        )

    def memories_tab() -> None:
        with gr.Row():
            refresh_memory_button = gr.Button("Refresh memories")
            memory_count = gr.Textbox(label="Count", interactive=False)
        memories = gr.Gallery(label="Memories", columns=4, height=420)
        memory_empty_msg = gr.Markdown(label="")
        memory_components.update(
            refresh_memory_button=refresh_memory_button,
            memory_count=memory_count,
            memories=memories,
            memory_empty_msg=memory_empty_msg,
        )

    def ask_tab() -> None:
        with gr.Row():
            ex1 = gr.Button("Was there unusual activity?", size="sm")
            ex2 = gr.Button("Show me daytime activity", size="sm")
            ex3 = gr.Button("Any pedestrian activity?", size="sm")
        question = gr.Textbox(label="Question", value="pedestrians walking on a walkway")
        ask_button = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", interactive=False)
        ask_gallery = gr.Gallery(label="Match", columns=1, height=320)
        ask_components.update(
            ex1=ex1,
            ex2=ex2,
            ex3=ex3,
            question=question,
            ask_button=ask_button,
            answer=answer,
            ask_gallery=ask_gallery,
        )

    def pending_tab() -> None:
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
        pending_components.update(
            refresh_pending_button=refresh_pending_button,
            pending_table=pending_table,
            pending_choice=pending_choice,
            pending_label=pending_label,
            submit_button=submit_button,
            label_status=label_status,
            pending_empty_msg=pending_empty_msg,
        )

    if default_ready:
        with gr.Tab("1. Ask"):
            ask_tab()
        with gr.Tab("2. Memories"):
            memories_tab()
        with gr.Tab("3. Pending Labels"):
            pending_tab()
        with gr.Tab("4. Scan"):
            scan_tab()
    else:
        with gr.Tab("1. Scan"):
            scan_tab()
        with gr.Tab("2. Ask"):
            ask_tab()
        with gr.Tab("3. Memories"):
            memories_tab()
        with gr.Tab("4. Pending Labels"):
            pending_tab()

    return scan_components, memory_components, ask_components, pending_components


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks demo."""
    default_ready = scan_ready()
    with gr.Blocks(title="Recall") as app:
        gr.Markdown("# Recall")
        status_display = gr.Markdown(value=initial_status(), label="Status")
        last_matched_id = gr.State(None)

        scan_components, memory_components, ask_components, pending_components = build_tabs(default_ready)

        scan_outputs = [
            scan_components["scan_table"],
            scan_components["scan_status"],
            status_display,
            memory_components["memories"],
            memory_components["memory_count"],
            memory_components["memory_empty_msg"],
            pending_components["pending_table"],
            pending_components["pending_choice"],
            pending_components["pending_empty_msg"],
            scan_components["build_button"],
            scan_components["rescan_button"],
        ]
        scan_components["build_button"].click(
            build_recall_memory,
            inputs=[
                scan_components["use_advanced"],
                scan_components["folder"],
                scan_components["advanced_limit"],
                scan_components["force_build"],
            ],
            outputs=scan_outputs,
        )
        scan_components["rescan_button"].click(
            build_recall_memory,
            inputs=[
                scan_components["use_advanced"],
                scan_components["folder"],
                scan_components["advanced_limit"],
                scan_components["force_rescan"],
            ],
            outputs=scan_outputs,
        )

        memory_components["refresh_memory_button"].click(
            refresh_memories,
            inputs=last_matched_id,
            outputs=[
                memory_components["memories"],
                memory_components["memory_count"],
                memory_components["memory_empty_msg"],
            ],
        )
        app.load(
            refresh_memories,
            inputs=last_matched_id,
            outputs=[
                memory_components["memories"],
                memory_components["memory_count"],
                memory_components["memory_empty_msg"],
            ],
        )

        ask_components["ask_button"].click(
            ask_memory,
            inputs=[ask_components["question"], last_matched_id],
            outputs=[
                ask_components["answer"],
                ask_components["ask_gallery"],
                last_matched_id,
                memory_components["memories"],
                memory_components["memory_count"],
                memory_components["memory_empty_msg"],
            ],
        )
        ask_components["ex1"].click(lambda: "Was there unusual activity?", None, ask_components["question"])
        ask_components["ex2"].click(lambda: "Show me daytime activity", None, ask_components["question"])
        ask_components["ex3"].click(lambda: "Any pedestrian activity?", None, ask_components["question"])

        pending_components["refresh_pending_button"].click(
            refresh_pending,
            outputs=[
                pending_components["pending_table"],
                pending_components["pending_choice"],
                pending_components["pending_empty_msg"],
            ],
        )
        app.load(
            refresh_pending,
            outputs=[
                pending_components["pending_table"],
                pending_components["pending_choice"],
                pending_components["pending_empty_msg"],
            ],
        )
        pending_components["submit_button"].click(
            submit_label,
            inputs=[pending_components["pending_choice"], pending_components["pending_label"]],
            outputs=pending_components["label_status"],
        )
        app.load(initial_status, outputs=status_display)
        app.load(initial_build_visibility, outputs=[scan_components["build_button"], scan_components["rescan_button"]])

    return app


def main() -> int:
    """Launch the Gradio demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    # When running in a Hugging Face Space, do not set server_name and server_port
    # as they are managed by the platform.
    server_name = None
    server_port = None
    if not os.environ.get("SPACE_ID"):
        server_name = args.server_name
        server_port = args.server_port

    build_app().launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
