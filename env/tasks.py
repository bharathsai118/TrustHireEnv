"""
tasks.py — TrustHireEnv
Task definitions: metadata, per-task observation generators, and
question banks used to create episodes for all three difficulty levels.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List

from env.schemas import Observation, FlagLevel, TaskDifficulty

_DATASETS_DIR = Path(__file__).parent.parent / "datasets"


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_dataset(difficulty: str) -> List[dict]:
    path = _DATASETS_DIR / f"{difficulty}.json"
    with path.open() as f:
        return json.load(f)


def _make_observation(record: dict, question_id: int, difficulty: str) -> Observation:
    """Build a validated Observation from a raw dataset record."""
    return Observation(
        question_id=question_id,
        difficulty=difficulty,
        gaze_offscreen_ratio=record.get("gaze_offscreen_ratio", 0.0),
        gaze_direction_entropy=record.get("gaze_direction_entropy", 0.0),
        head_turn_angle_deg=record.get("head_turn_angle_deg", 0.0),
        whisper_score=record.get("whisper_score", 0.0),
        second_voice_confidence=record.get("second_voice_confidence", 0.0),
        lip_motion_mismatch=record.get("lip_motion_mismatch", 0.0),
        response_latency_sec=record.get("response_latency_sec", 0.0),
        answer_consistency=record.get("answer_consistency", 1.0),
        project_followup_score=record.get("project_followup_score", 1.0),
        complexity_jump=record.get("complexity_jump", 0.0),
        ground_truth_flag=FlagLevel(record["ground_truth_flag"]),
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

class TaskConfig:
    """Immutable task metadata returned by get_task()."""

    def __init__(self, difficulty: str, max_steps: int, description: str):
        self.difficulty  = TaskDifficulty(difficulty)
        self.max_steps   = max_steps
        self.description = description

    def __repr__(self):
        return f"<TaskConfig difficulty={self.difficulty} max_steps={self.max_steps}>"


TASK_CONFIGS = {
    "easy": TaskConfig(
        difficulty="easy",
        max_steps=5,
        description=(
            "Gaze Diversion Detection — Identify off-screen eye movement patterns "
            "that signal the candidate is reading from external material."
        ),
    ),
    "medium": TaskConfig(
        difficulty="medium",
        max_steps=7,
        description=(
            "Whisper & External Assistance Detection — Identify low-volume speech "
            "bursts, second-person voice presence, and anomalous response delays "
            "that suggest real-time coaching."
        ),
    ),
    "hard": TaskConfig(
        difficulty="hard",
        max_steps=10,
        description=(
            "Reasoning Integrity Verification — Determine whether the candidate's "
            "explanations under follow-up questioning are consistent with their "
            "original answers, or show signs of memorized / AI-generated responses."
        ),
    ),
}


def get_task(difficulty: str) -> TaskConfig:
    if difficulty not in TASK_CONFIGS:
        raise ValueError(f"Unknown task difficulty: {difficulty!r}. "
                         f"Choose from {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[difficulty]


def load_episode(difficulty: str, episode_index: int = 0, seed: int = 42) -> List[Observation]:
    """
    Load one episode's worth of observations from the dataset.

    Parameters
    ----------
    difficulty    : "easy" | "medium" | "hard"
    episode_index : which episode to load (wraps around if index exceeds dataset)
    seed          : random seed used for any sampling operations

    Returns
    -------
    Ordered list of Observation objects for one episode.
    """
    rng     = random.Random(seed + episode_index)
    records = _load_dataset(difficulty)
    cfg     = get_task(difficulty)

    # If dataset has explicit episodes, use them; otherwise sample
    if isinstance(records[0], list):
        episode_records = records[episode_index % len(records)]
    else:
        n = min(cfg.max_steps, len(records))
        episode_records = rng.sample(records, n)

    observations = [
        _make_observation(rec, q_id + 1, difficulty)
        for q_id, rec in enumerate(episode_records)
    ]
    return observations
