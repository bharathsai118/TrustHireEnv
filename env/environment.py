"""
environment.py — TrustHireEnv
Core OpenEnv-compliant environment class.

Implements the mandatory interface:
  reset()  → dict          (initial observation)
  step(action) → (dict, float, bool, dict)
  state()  → dict          (full internal state snapshot)

Plus helpers:
  render()          — human-readable episode summary
  seed(n)           — set RNG seed
  action_space      — valid actions enumeration
  observation_space — field descriptions
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from env.schemas import (
    Action, EpisodeResult, FlagLevel, NextStep,
    Observation, RewardPayload,
)
from env.tasks import TaskConfig, get_task, load_episode
from env.graders import grade_episode
from env.rewards import compute_step_reward


# ────────────────────────────────────────────────────────────────────────────
# Action / Observation space descriptors  (OpenEnv-style)
# ────────────────────────────────────────────────────────────────────────────

ACTION_SPACE: Dict[str, Any] = {
    "type":   "dict",
    "fields": {
        "flag_level": {
            "type":   "categorical",
            "values": [e.value for e in FlagLevel],
        },
        "next_step": {
            "type":   "categorical",
            "values": [e.value for e in NextStep],
        },
        "rationale": {
            "type":      "string",
            "max_length": 512,
            "required":  False,
        },
    },
}

OBSERVATION_SPACE: Dict[str, Any] = {
    "type": "dict",
    "fields": {
        "question_id":             {"type": "int",   "range": [1, 100]},
        "difficulty":              {"type": "categorical", "values": ["easy", "medium", "hard"]},
        "gaze_offscreen_ratio":    {"type": "float", "range": [0.0, 1.0]},
        "gaze_direction_entropy":  {"type": "float", "range": [0.0, 1.0]},
        "head_turn_angle_deg":     {"type": "float", "range": [0.0, 180.0]},
        "whisper_score":           {"type": "float", "range": [0.0, 1.0]},
        "second_voice_confidence": {"type": "float", "range": [0.0, 1.0]},
        "lip_motion_mismatch":     {"type": "float", "range": [0.0, 1.0]},
        "response_latency_sec":    {"type": "float", "range": [0.0, 60.0]},
        "answer_consistency":      {"type": "float", "range": [0.0, 1.0]},
        "project_followup_score":  {"type": "float", "range": [0.0, 1.0]},
        "complexity_jump":         {"type": "float", "range": [0.0, 1.0]},
    },
}


# ────────────────────────────────────────────────────────────────────────────
# TrustHireEnv
# ────────────────────────────────────────────────────────────────────────────

class TrustHireEnv:
    """
    OpenEnv-compliant interview integrity evaluation environment.

    Usage
    -----
    env = TrustHireEnv(difficulty="medium")
    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    result = info["episode_result"]
    """

    # OpenEnv metadata
    name        = "TrustHireEnv"
    version     = "1.0.0"
    description = "Multimodal interview integrity evaluation environment"

    action_space      = ACTION_SPACE
    observation_space = OBSERVATION_SPACE

    def __init__(
        self,
        difficulty:     str = "easy",
        episode_index:  int = 0,
        seed:           int = 42,
        max_steps:      Optional[int] = None,
    ):
        self._difficulty     = difficulty
        self._episode_index  = episode_index
        self._seed           = seed
        self._task_cfg: TaskConfig = get_task(difficulty)

        self._max_steps: int = max_steps or self._task_cfg.max_steps

        # Episode state (populated on reset)
        self._episode_id:       str                      = ""
        self._observations:     List[Observation]        = []
        self._trajectory:       List[Tuple[Observation, Action]] = []
        self._step_idx:         int                      = 0
        self._cumulative_reward: float                   = 0.0
        self._prev_action:      Optional[Action]         = None
        self._repeat_count:     int                      = 0
        self._done:             bool                     = True   # needs reset

    # ── OpenEnv required methods ─────────────────────────────────────────

    def reset(self, episode_index: Optional[int] = None, seed: Optional[int] = None) -> dict:
        """
        Reset the environment and return the first observation.

        Returns
        -------
        dict — agent-visible observation (no ground truth)
        """
        if episode_index is not None:
            self._episode_index = episode_index
        if seed is not None:
            self._seed = seed

        self._episode_id        = str(uuid.uuid4())[:8]
        self._observations      = load_episode(
            self._difficulty, self._episode_index, self._seed
        )
        self._trajectory        = []
        self._step_idx          = 0
        self._cumulative_reward = 0.0
        self._prev_action       = None
        self._repeat_count      = 0
        self._done              = False

        return self._observations[0].agent_view()

    def step(self, action: dict | Action) -> Tuple[dict, float, bool, dict]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        action : dict or Action — the agent's decision

        Returns
        -------
        (observation, reward, done, info)
          observation : dict   — next agent-visible observation (or final obs if done)
          reward      : float  — shaped step reward
          done        : bool   — True if the episode has ended
          info        : dict   — diagnostics; includes 'episode_result' when done=True
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        # Parse & validate action
        if isinstance(action, dict):
            action = Action(**action)

        obs = self._observations[self._step_idx]

        # Track repeat actions (loop detection)
        if (
            self._prev_action
            and action.flag_level == self._prev_action.flag_level
            and action.next_step  == self._prev_action.next_step
        ):
            self._repeat_count += 1
        else:
            self._repeat_count = 0
        self._prev_action = action

        # Compute reward
        reward_payload: RewardPayload = compute_step_reward(
            obs=obs,
            action=action,
            step_idx=self._step_idx,
            repeat_count=self._repeat_count,
        )
        reward = round(reward_payload.total, 4)
        self._cumulative_reward += reward

        # Record trajectory
        self._trajectory.append((obs, action))
        self._step_idx += 1

        # Determine done
        done = (
            self._step_idx >= len(self._observations)
            or self._step_idx >= self._max_steps
        )
        self._done = done

        # Next observation
        if not done:
            next_obs_dict = self._observations[self._step_idx].agent_view()
        else:
            next_obs_dict = obs.agent_view()   # repeat last obs at terminal

        info: Dict[str, Any] = {
            "step":             self._step_idx,
            "reward_breakdown": reward_payload.model_dump(),
            "repeat_count":     self._repeat_count,
        }

        if done:
            grader_result = grade_episode(self._trajectory, self._difficulty)
            episode_result = EpisodeResult(
                episode_id=self._episode_id,
                task_difficulty=self._difficulty,
                total_steps=self._step_idx,
                cumulative_reward=round(self._cumulative_reward, 4),
                task_score=round(grader_result.score, 4),
                flags_predicted=[a.flag_level for _, a in self._trajectory],
                flags_ground_truth=[o.ground_truth_flag for o, _ in self._trajectory],
            )
            info["episode_result"]  = episode_result.model_dump()
            info["grader_details"]  = grader_result.details

        return next_obs_dict, reward, done, info

    def state(self) -> dict:
        """
        Return a full snapshot of the internal environment state.
        Used by OpenEnv validators and debugging tools.
        """
        return {
            "episode_id":        self._episode_id,
            "difficulty":        self._difficulty,
            "step_idx":          self._step_idx,
            "max_steps":         self._max_steps,
            "done":              self._done,
            "cumulative_reward": self._cumulative_reward,
            "trajectory_length": len(self._trajectory),
            "seed":              self._seed,
            "episode_index":     self._episode_index,
        }

    # ── Convenience helpers ──────────────────────────────────────────────

    def seed(self, value: int) -> None:
        """Set the RNG seed (takes effect on next reset)."""
        self._seed = value

    def render(self) -> str:
        """Return a human-readable summary of the current episode progress."""
        lines = [
            f"TrustHireEnv  [{self._difficulty.upper()}]  episode={self._episode_id}",
            f"  Step: {self._step_idx}/{self._max_steps}",
            f"  Cumulative reward: {self._cumulative_reward:.4f}",
            f"  Done: {self._done}",
        ]
        for i, (obs, act) in enumerate(self._trajectory):
            lines.append(
                f"  [{i+1}] Q{obs.question_id} "
                f"gt={obs.ground_truth_flag} "
                f"pred={act.flag_level} "
                f"next={act.next_step}"
            )
        return "\n".join(lines)
