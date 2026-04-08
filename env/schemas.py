"""
schemas.py — TrustHireEnv
Pydantic data-contracts for observations, actions, and reward payloads.
All fields are typed and validated so the OpenEnv checker can introspect them.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ────────────────────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────────────────────

class FlagLevel(str, Enum):
    NONE    = "none"
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"


class NextStep(str, Enum):
    CONTINUE  = "continue"
    FOLLOWUP  = "followup"
    WARN      = "warn"
    ESCALATE  = "escalate"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ────────────────────────────────────────────────────────────────────────────
# Observation schema (what the agent sees at every step)
# ────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """Multimodal interview signal bundle."""

    # Metadata
    question_id: int = Field(..., ge=1, description="Sequential question index")
    difficulty:  TaskDifficulty

    # Gaze / head-pose signals  (Task 1)
    gaze_offscreen_ratio:    float = Field(0.0, ge=0.0, le=1.0)
    gaze_direction_entropy:  float = Field(0.0, ge=0.0, le=1.0,
                                           description="Shannon entropy of gaze direction histogram")
    head_turn_angle_deg:     float = Field(0.0, ge=0.0, le=180.0)

    # Audio / whisper signals  (Task 2)
    whisper_score:           float = Field(0.0, ge=0.0, le=1.0,
                                           description="P(whisper) from VAD model")
    second_voice_confidence: float = Field(0.0, ge=0.0, le=1.0)
    lip_motion_mismatch:     float = Field(0.0, ge=0.0, le=1.0)
    response_latency_sec:    float = Field(0.0, ge=0.0)

    # Reasoning consistency signals  (Task 3)
    answer_consistency:      float = Field(1.0, ge=0.0, le=1.0,
                                           description="Semantic similarity between claimed answer and follow-up explanation")
    project_followup_score:  float = Field(1.0, ge=0.0, le=1.0,
                                           description="How well the candidate explains their own project choice")
    complexity_jump:         float = Field(0.0, ge=0.0, le=1.0,
                                           description="Magnitude of sudden answer-quality jump unexplained by question change")

    # Ground truth (hidden from agent; used by grader)
    ground_truth_flag: Optional[FlagLevel] = Field(None, exclude=True)

    class Config:
        use_enum_values = True

    def agent_view(self) -> dict:
        """Return only the signals visible to the agent (no ground truth)."""
        data = self.model_dump()
        data.pop("ground_truth_flag", None)
        return data


# ────────────────────────────────────────────────────────────────────────────
# Action schema (what the agent produces)
# ────────────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent decision for a single interview step."""

    flag_level: FlagLevel = Field(..., description="Integrity risk level assigned by the agent")
    next_step:  NextStep  = Field(..., description="Workflow action to take")
    rationale:  Optional[str] = Field(None, max_length=512,
                                      description="Optional free-text justification (not graded but logged)")

    @field_validator("next_step")
    @classmethod
    def escalate_requires_high_flag(cls, v: NextStep, info) -> NextStep:
        """Escalate is only valid when flag_level is HIGH or MEDIUM."""
        flag = info.data.get("flag_level")
        if v == NextStep.ESCALATE and flag == FlagLevel.NONE:
            raise ValueError("Cannot escalate when flag_level is 'none'.")
        return v

    class Config:
        use_enum_values = True


# ────────────────────────────────────────────────────────────────────────────
# Reward payload (returned by env.step())
# ────────────────────────────────────────────────────────────────────────────

class RewardPayload(BaseModel):
    """Decomposed reward breakdown for interpretability."""

    progress_reward:    float = 0.0
    integrity_penalty:  float = 0.0
    accuracy_bonus:     float = 0.0
    loop_penalty:       float = 0.0
    total:              float = 0.0

    def compute_total(self) -> "RewardPayload":
        self.total = (
            self.progress_reward
            + self.integrity_penalty
            + self.accuracy_bonus
            + self.loop_penalty
        )
        return self


# ────────────────────────────────────────────────────────────────────────────
# Episode result (returned at done=True)
# ────────────────────────────────────────────────────────────────────────────

class EpisodeResult(BaseModel):
    episode_id:       str
    task_difficulty:  TaskDifficulty
    total_steps:      int
    cumulative_reward: float
    task_score:       float   # 0.0 – 1.0, computed by grader
    flags_predicted:  List[str]
    flags_ground_truth: List[str]
