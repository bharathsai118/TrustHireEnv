"""
graders.py — TrustHireEnv
Deterministic task-level graders for all three difficulty tiers.

Each grader:
  • Receives the full episode trajectory (list of (obs, action) tuples).
  • Returns a float score in [0.0, 1.0].
  • Is entirely rule-based — zero stochasticity — so the checker gets
    identical numbers on every run.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from env.schemas import Action, Observation, FlagLevel, NextStep


@dataclass
class GraderResult:
    score:         float          # 0.0 – 1.0
    correct_steps: int
    total_steps:   int
    details:       dict


# ────────────────────────────────────────────────────────────────────────────
# Task 1 — Gaze Diversion Detection  (Easy)
# ────────────────────────────────────────────────────────────────────────────

def grade_easy(
    trajectory: List[Tuple[Observation, Action]],
) -> GraderResult:
    """
    Binary accuracy: flag_level must match ground_truth_flag exactly.
    Score = correct_predictions / total_steps.
    """
    correct = 0
    details: List[dict] = []

    for obs, action in trajectory:
        gt       = obs.ground_truth_flag
        pred     = action.flag_level
        is_right = (pred == gt)
        if is_right:
            correct += 1
        details.append({"q": obs.question_id, "gt": gt, "pred": pred, "ok": is_right})

    score = correct / max(len(trajectory), 1)
    return GraderResult(score=score,
                        correct_steps=correct,
                        total_steps=len(trajectory),
                        details={"per_step": details})


# ────────────────────────────────────────────────────────────────────────────
# Task 2 — Whisper & External Assistance  (Medium)
# ────────────────────────────────────────────────────────────────────────────

_WHISPER_WEIGHTS = {
    "second_voice": 0.40,
    "timing":       0.30,
    "whisper_conf": 0.30,
}


def _whisper_rule_score(obs: Observation, action: Action) -> float:
    """
    Weighted deterministic signal: returns per-step accuracy [0, 1].
    The agent is rewarded for correctly identifying that whisper / external
    help signals are above threshold AND for issuing an appropriate action.
    """
    gt   = obs.ground_truth_flag
    pred = action.flag_level

    # Component scores (signal detection sub-tasks)
    sv_detected   = 1.0 if obs.second_voice_confidence >= 0.50 else 0.0
    timing_anom   = 1.0 if obs.response_latency_sec    >= 5.0  else 0.0
    whisper_det   = 1.0 if obs.whisper_score           >= 0.45 else 0.0

    component_score = (
        _WHISPER_WEIGHTS["second_voice"] * sv_detected +
        _WHISPER_WEIGHTS["timing"]       * timing_anom +
        _WHISPER_WEIGHTS["whisper_conf"] * whisper_det
    )

    # Final flag match bonus
    flag_match = 1.0 if pred == gt else 0.0
    return 0.5 * flag_match + 0.5 * component_score


def grade_medium(
    trajectory: List[Tuple[Observation, Action]],
) -> GraderResult:
    scores = [_whisper_rule_score(obs, act) for obs, act in trajectory]
    score  = sum(scores) / max(len(scores), 1)

    details = [{"q": obs.question_id, "step_score": s}
               for (obs, _), s in zip(trajectory, scores)]
    return GraderResult(score=score,
                        correct_steps=sum(1 for s in scores if s >= 0.7),
                        total_steps=len(trajectory),
                        details={"per_step": details})


# ────────────────────────────────────────────────────────────────────────────
# Task 3 — Reasoning Integrity Verification  (Hard)
# ────────────────────────────────────────────────────────────────────────────

_CONSISTENCY_THRESHOLD    = 0.60   # below this → inconsistent explanation
_COMPLEXITY_JUMP_THRESH   = 0.50   # above this → suspicious sudden quality jump
_FOLLOWUP_REQUIRED_BELOW  = 0.65   # if consistency <65% agent should issue followup


def _reasoning_step_score(obs: Observation, action: Action) -> float:
    """
    Evaluates whether the agent correctly identifies reasoning inconsistency
    and issues the right next_step.

    Rules:
      • If answer_consistency < threshold  → flag should be ≥ medium
      • If complexity_jump   > threshold   → flag should be ≥ medium
      • If consistency < FOLLOWUP_REQUIRED → next_step should be followup/escalate
      • Perfect alignment on all three axes → 1.0
    """
    gt   = obs.ground_truth_flag
    pred = action.flag_level

    inconsistent    = obs.answer_consistency   < _CONSISTENCY_THRESHOLD
    suspicious_jump = obs.complexity_jump      > _COMPLEXITY_JUMP_THRESH
    needs_followup  = obs.answer_consistency   < _FOLLOWUP_REQUIRED_BELOW

    # Sub-score 1: flag severity correctness
    if pred == gt:
        flag_score = 1.0
    elif inconsistent or suspicious_jump:
        # Ground truth is flagged; agent should have flagged ≥ medium
        if action.flag_level in (FlagLevel.MEDIUM, FlagLevel.HIGH):
            flag_score = 0.70
        else:
            flag_score = 0.10
    else:
        flag_score = 0.50   # minor mismatch on clean step

    # Sub-score 2: workflow action correctness
    if needs_followup:
        action_score = (
            1.0 if action.next_step in (NextStep.FOLLOWUP, NextStep.ESCALATE,
                                        NextStep.WARN)
            else 0.20
        )
    else:
        action_score = (
            1.0 if action.next_step == NextStep.CONTINUE
            else 0.60
        )

    return 0.60 * flag_score + 0.40 * action_score


def grade_hard(
    trajectory: List[Tuple[Observation, Action]],
) -> GraderResult:
    scores = [_reasoning_step_score(obs, act) for obs, act in trajectory]
    score  = sum(scores) / max(len(scores), 1)

    details = [{"q": obs.question_id,
                "consistency": obs.answer_consistency,
                "complexity_jump": obs.complexity_jump,
                "step_score": s}
               for (obs, _), s in zip(trajectory, scores)]
    return GraderResult(score=score,
                        correct_steps=sum(1 for s in scores if s >= 0.7),
                        total_steps=len(trajectory),
                        details={"per_step": details})


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────────────────────

def grade_episode(
    trajectory:  List[Tuple[Observation, Action]],
    difficulty:  str,
) -> GraderResult:
    """Route to the correct grader based on task difficulty string."""
    match difficulty.lower():
        case "easy":
            return grade_easy(trajectory)
        case "medium":
            return grade_medium(trajectory)
        case "hard":
            return grade_hard(trajectory)
        case _:
            raise ValueError(f"Unknown difficulty: {difficulty!r}")
