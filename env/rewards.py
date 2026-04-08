"""
rewards.py — TrustHireEnv
Dense reward shaping for the hiring-integrity environment.
Designed for RL-friendliness: every step returns a non-zero signal.
"""

from __future__ import annotations
from env.schemas import Action, Observation, RewardPayload, FlagLevel, NextStep


# ────────────────────────────────────────────────────────────────────────────
# Reward constants
# ────────────────────────────────────────────────────────────────────────────

R_CORRECT_CLEAR    =  0.40   # Correctly cleared an innocent candidate
R_CORRECT_LOW      =  0.20   # Correct low-risk continuation
R_CORRECT_MEDIUM   =  0.35   # Correct medium escalation
R_CORRECT_FLAG     =  0.50   # Correctly flagged a cheater
R_GOOD_FOLLOWUP    =  0.30   # Productive follow-up question issued
R_FALSE_ACCUSATION = -0.60   # Flagged an innocent candidate
R_MISSED_CHEAT     = -0.70   # Missed a suspicious pattern
R_USELESS_FOLLOWUP = -0.20   # Repeated follow-up that adds no information
R_LOOP_PENALTY     = -1.00   # Detected infinite action loop


def compute_step_reward(
    obs:          Observation,
    action:       Action,
    step_idx:     int,
    repeat_count: int,
) -> RewardPayload:
    """
    Compute the shaped reward for a single (obs, action) pair.

    Parameters
    ----------
    obs          : current observation (contains hidden ground truth)
    action       : agent's chosen action
    step_idx     : 0-based step index within the episode
    repeat_count : how many times the agent has issued the exact same action
                   consecutively (loop detection)

    Returns
    -------
    RewardPayload with all components filled in and total computed.
    """
    payload = RewardPayload()
    gt = obs.ground_truth_flag  # may be None in real deployment

    # ── Loop penalty (applied first; overrides everything) ────────────────
    if repeat_count >= 3:
        payload.loop_penalty = R_LOOP_PENALTY
        return payload.compute_total()

    # ── Step progress reward (always positive to encourage completion) ─────
    payload.progress_reward = 0.05 * (1.0 / (step_idx + 1))

    # ── Accuracy bonus / integrity penalty ───────────────────────────────
    if gt is not None:
        predicted = action.flag_level
        truth     = gt

        match (predicted, truth):
            # Correct decisions
            case (FlagLevel.NONE, FlagLevel.NONE):
                payload.accuracy_bonus    = R_CORRECT_CLEAR
            case (FlagLevel.LOW,  FlagLevel.LOW):
                payload.accuracy_bonus    = R_CORRECT_LOW
            case (FlagLevel.MEDIUM, FlagLevel.MEDIUM):
                payload.accuracy_bonus    = R_CORRECT_MEDIUM
            case (FlagLevel.HIGH,  FlagLevel.HIGH):
                payload.accuracy_bonus    = R_CORRECT_FLAG
            # False accusation (flag raised when ground truth is clean)
            case (FlagLevel.LOW  | FlagLevel.MEDIUM | FlagLevel.HIGH, FlagLevel.NONE):
                payload.integrity_penalty = R_FALSE_ACCUSATION
            # Missed cheating (no flag when ground truth requires one)
            case (FlagLevel.NONE, FlagLevel.LOW | FlagLevel.MEDIUM | FlagLevel.HIGH):
                payload.integrity_penalty = R_MISSED_CHEAT
            # Partially correct (flag raised but wrong severity)
            case _:
                # Partial credit: half penalty
                payload.integrity_penalty = -0.20

    # ── Follow-up quality bonus ───────────────────────────────────────────
    if action.next_step == NextStep.FOLLOWUP:
        if repeat_count == 0:
            payload.progress_reward += R_GOOD_FOLLOWUP
        else:
            payload.progress_reward += R_USELESS_FOLLOWUP

    return payload.compute_total()
