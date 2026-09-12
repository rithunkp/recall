"""Coordinator fusion for Structural, Semantic, and Routine novelty scores."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from agents.routine_agent import RoutineAgent, load_stats, validation_manifest_records
from agents.semantic_agent import SemanticAgent, load_prototypes as load_semantic_prototypes
from agents.semantic_agent import frame_path as semantic_frame_path
from agents.structural_agent import StructuralAgent
from agents.structural_agent import load_prototypes as load_structural_prototypes
from agents.structural_agent import frame_path as structural_frame_path
from coordinator.active_learning import append_pending_label_request
from memory.store import write_memory


MEMORY_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class AgentScores:
    """Novelty scores and votes from the three perception agents."""

    structural_score: float
    semantic_score: float
    routine_score: float
    structural_vote: bool
    semantic_vote: bool
    routine_vote: bool


@dataclass(frozen=True)
class CoordinatorDecision:
    """Fused coordinator decision for one frame."""

    frame_path: str
    decision: str
    is_confident: bool
    is_novel: bool | None
    novel_votes: int
    familiar_votes: int
    scores: AgentScores
    memory_id: str | None


def min_votes() -> int:
    """Return the configured vote count needed for a majority suggestion."""
    return config.COORDINATOR_MIN_VOTES


def fuse_scores(
    structural_score: float, semantic_score: float, routine_score: float
) -> AgentScores:
    """Convert scores into config-thresholded agent votes."""
    return AgentScores(
        structural_score=structural_score,
        semantic_score=semantic_score,
        routine_score=routine_score,
        structural_vote=structural_score >= config.STRUCTURAL_NOVELTY_THRESHOLD,
        semantic_vote=semantic_score >= config.SEMANTIC_NOVELTY_THRESHOLD,
        routine_vote=routine_score >= config.ROUTINE_NOVELTY_THRESHOLD,
    )


def decide(scores: AgentScores, frame_path: str, memory_id: str | None = None) -> CoordinatorDecision:
    """Fuse agent votes; 2/3 or 3/3 novel votes = confident novel, 0/3 or 1/3 novel = confident familiar."""
    votes = [scores.structural_vote, scores.semantic_vote, scores.routine_vote]
    novel_votes = sum(votes)
    familiar_votes = len(votes) - novel_votes
    if novel_votes >= config.COORDINATOR_MIN_VOTES:
        return CoordinatorDecision(
            frame_path=frame_path,
            decision="novel",
            is_confident=True,
            is_novel=True,
            novel_votes=novel_votes,
            familiar_votes=familiar_votes,
            scores=scores,
            memory_id=memory_id,
        )
    # 0 or 1 novel votes → confident familiar
    return CoordinatorDecision(
        frame_path=frame_path,
        decision="familiar",
        is_confident=True,
        is_novel=False,
        novel_votes=novel_votes,
        familiar_votes=familiar_votes,
        scores=scores,
        memory_id=None,
    )


def append_decision(decision: CoordinatorDecision) -> None:
    """Persist a coordinator decision for debugging/demo traceability."""
    config.COORDINATOR_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(decision)
    with config.COORDINATOR_DECISIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def memory_id_for_frame(frame_path: str) -> str:
    """Build a stable filesystem-safe memory id from a repo-relative frame path."""
    return MEMORY_ID_RE.sub("_", frame_path).strip("_")


def score_record(
    *,
    record: dict[str, object],
    structural_agent: StructuralAgent,
    semantic_agent: SemanticAgent,
    routine_agent: RoutineAgent,
    structural_prototypes: np.ndarray,
    semantic_prototypes: np.ndarray,
    routine_stats: object,
) -> CoordinatorDecision:
    """Score one manifest record and route confident/ambiguous outcomes."""
    frame = structural_frame_path(record)
    structural_embedding = structural_agent.embed_images([frame])[0]
    semantic_embedding = semantic_agent.embed_images([semantic_frame_path(record)])[0]
    structural_score = structural_agent.score_embedding(
        structural_embedding, structural_prototypes
    )
    semantic_score = semantic_agent.score_embedding(semantic_embedding, semantic_prototypes)
    routine_result = routine_agent.score_record(record, routine_stats)
    scores = fuse_scores(structural_score, semantic_score, routine_result.novelty_score)

    memory_id = None
    preliminary = decide(scores, str(record["frame_path"]))
    if preliminary.is_confident and preliminary.is_novel:
        memory_id = memory_id_for_frame(str(record["frame_path"]))
        write_memory(
            memory_id=memory_id,
            frame_record=record,
            structural_embedding=structural_embedding,
            semantic_embedding=semantic_embedding,
            routine_hour=routine_result.hour,
        )

    decision = decide(scores, str(record["frame_path"]), memory_id=memory_id)
    if not decision.is_confident:
        append_pending_label_request(
            {
                "frame_path": decision.frame_path,
                "source_path": str(record.get("source_path", "")),
                "decision": decision.decision,
                "novel_votes": decision.novel_votes,
                "familiar_votes": decision.familiar_votes,
                "scores": asdict(decision.scores),
            }
        )
    append_decision(decision)
    return decision


def main() -> int:
    """Run a small validation-split coordinator pass without touching test data."""
    config.set_seed()
    records = validation_manifest_records()[: config.COORDINATOR_MAX_FRAMES]
    structural_agent = StructuralAgent()
    semantic_agent = SemanticAgent()
    routine_agent = RoutineAgent()
    structural_prototypes = load_structural_prototypes()
    semantic_prototypes = load_semantic_prototypes()
    routine_stats = load_stats()

    decisions = [
        score_record(
            record=record,
            structural_agent=structural_agent,
            semantic_agent=semantic_agent,
            routine_agent=routine_agent,
            structural_prototypes=structural_prototypes,
            semantic_prototypes=semantic_prototypes,
            routine_stats=routine_stats,
        )
        for record in records
    ]

    for decision in decisions:
        print(
            f"Coordinator decision: {decision.frame_path} {decision.decision} "
            f"votes={decision.novel_votes}/3 "
            f"s=({decision.scores.structural_score:.4f},"
            f"{decision.scores.semantic_score:.4f},"
            f"{decision.scores.routine_score:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
