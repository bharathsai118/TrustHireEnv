"""
baseline_eval.py — TrustHireEnv
Reproducible baseline evaluation script using the OpenAI API.

Usage
-----
  python baseline_eval.py                        # all tasks, 3 episodes each
  python baseline_eval.py --task easy            # single task
  python baseline_eval.py --episodes 5 --seed 0  # custom count / seed
  python baseline_eval.py --no-llm               # rule-based agent (no API key needed)

Exit code 0 on success; non-zero on any error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# ── Optional rich console ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# ── Environment import ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from env.environment import TrustHireEnv
from env.schemas import Action, FlagLevel, NextStep


# ────────────────────────────────────────────────────────────────────────────
# Agents
# ────────────────────────────────────────────────────────────────────────────

class RuleBasedAgent:
    """
    Deterministic heuristic agent — no LLM required.
    Uses task-specific thresholds on the observation signals.
    """

    def act(self, obs: Dict[str, Any]) -> Action:
        difficulty = obs["difficulty"]

        gaze = obs["gaze_offscreen_ratio"]
        whisper = obs["whisper_score"]
        sv = obs["second_voice_confidence"]
        consist = obs["answer_consistency"]
        jump = obs["complexity_jump"]
        latency = obs["response_latency_sec"]

        # Task-specific risk scoring
        if difficulty == "easy":
            risk = 0.7 * gaze + 0.3 * min(latency / 10.0, 1.0)

        elif difficulty == "medium":
            risk = (
                0.45 * whisper
                + 0.35 * sv
                + 0.20 * min(latency / 10.0, 1.0)
            )

        else:  # hard
            risk = (
                0.5 * (1.0 - consist)
                + 0.3 * jump
                + 0.2 * min(latency / 10.0, 1.0)
            )

        # Task-specific thresholds
        if difficulty == "easy":
            high_t, med_t, low_t = 0.48, 0.30, 0.15
        elif difficulty == "medium":
            high_t, med_t, low_t = 0.55, 0.35, 0.18
        else:  # hard
            high_t, med_t, low_t = 0.55, 0.35, 0.18

        if risk >= high_t:
            flag = FlagLevel.HIGH
            next_step = NextStep.ESCALATE
        elif risk >= med_t:
            flag = FlagLevel.MEDIUM
            next_step = NextStep.WARN
        elif risk >= low_t:
            flag = FlagLevel.LOW
            next_step = NextStep.FOLLOWUP
        else:
            flag = FlagLevel.NONE
            next_step = NextStep.CONTINUE
        return Action(
                flag_level=flag,
                next_step=next_step,
                rationale=f"{difficulty}_risk={risk:.3f}",
            )

class LLMAgent:
    """
    OpenAI-backed agent using GPT-4o with structured output parsing.
    Falls back to RuleBasedAgent if any API error occurs.
    """

    SYSTEM_PROMPT = (
        "You are an AI interview integrity analyst. "
        "Given multimodal interview signals, decide the integrity risk level "
        "and the appropriate next workflow step.\n"
        "Output JSON only: {\"flag_level\": \"none|low|medium|high\", "
        "\"next_step\": \"continue|followup|warn|escalate\", "
        "\"rationale\": \"<brief reason>\"}"
    )

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
            self._client = OpenAI()
            self._model  = model
            self._ok     = True
        except Exception as e:
            print(f"[LLMAgent] OpenAI init failed: {e}. Falling back to rule-based.")
            self._ok = False
            self._fallback = RuleBasedAgent()

    def act(self, obs: Dict[str, Any]) -> Action:
        if not self._ok:
            return self._fallback.act(obs)
        try:
            from openai import OpenAI
            user_msg = json.dumps({k: v for k, v in obs.items()
                                   if k not in ("difficulty",)}, indent=2)
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return Action(**data)
        except Exception as e:
            print(f"[LLMAgent] API error: {e}. Using rule-based fallback.")
            return RuleBasedAgent().act(obs)


# ────────────────────────────────────────────────────────────────────────────
# Episode runner
# ────────────────────────────────────────────────────────────────────────────

def run_episode(
    agent,
    difficulty: str,
    episode_index: int,
    seed: int,
) -> Dict[str, Any]:
    env  = TrustHireEnv(difficulty=difficulty, episode_index=episode_index, seed=seed)
    obs  = env.reset()
    done = False
    info: Dict[str, Any] = {}

    while not done:
        action          = agent.act(obs)
        obs, _, done, info = env.step(action)

    return info.get("episode_result", {})


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrustHireEnv Baseline Evaluator")
    parser.add_argument("--task",     default="all",
                        choices=["all", "easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per task")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--model",    default="gpt-4o-mini",
                        help="OpenAI model name (ignored with --no-llm)")
    parser.add_argument("--no-llm",  action="store_true",
                        help="Use deterministic rule-based agent only")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    agent = RuleBasedAgent() if args.no_llm else LLMAgent(model=args.model)

    print("\n" + "=" * 55)
    print("  TrustHireEnv - Baseline Evaluation")
    print("=" * 55)

    all_scores: Dict[str, List[float]] = {}

    for difficulty in tasks:
        scores: List[float] = []
        for ep in range(args.episodes):
            result = run_episode(agent, difficulty, episode_index=ep, seed=args.seed)
            score  = result.get("task_score", 0.0)
            scores.append(score)
            print(f"  {difficulty:6s}  ep={ep}  score={score:.4f}")
        all_scores[difficulty] = scores

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "-" * 55)
    overall: List[float] = []
    for difficulty, scores in all_scores.items():
        avg = sum(scores) / len(scores)
        overall.extend(scores)
        label = {"easy": "Task1 (Easy)", "medium": "Task2 (Medium)", "hard": "Task3 (Hard)"}[difficulty]
        print(f"  {label:20s}: {avg:.4f}")

    global_avg = sum(overall) / max(len(overall), 1)
    print(f"  {'Average':20s}: {global_avg:.4f}")
    print("=" * 55 + "\n")

    # Machine-readable output
    summary = {
        "tasks": {k: sum(v)/len(v) for k, v in all_scores.items()},
        "average": global_avg,
    }
    print(json.dumps(summary, indent=2))

    sys.exit(0)


if __name__ == "__main__":
    main()
