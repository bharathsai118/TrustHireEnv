---
title: Trust Hire Env
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Your actual README content can go down here...
# TrustHireEnv

**OpenEnv-Compliant Benchmark Environment for Multimodal Interview Integrity & Hiring Workflow Evaluation**

> An agentic evaluation environment where an AI agent acts as a technical recruiter and interview integrity analyst — monitoring gaze, audio, response consistency, and reasoning signals to make real-time hiring decisions.

---

## Overview

TrustHireEnv simulates the real-world workflow of a **remote technical interview integrity review**. The AI agent observes multimodal interview signals at each question step and must:

1. Assess the **integrity risk level** (`none` / `low` / `medium` / `high`)
2. Choose the appropriate **workflow action** (`continue` / `followup` / `warn` / `escalate`)

This is not a toy chat task. It mirrors decision-making already performed by:
- AI-powered proctoring platforms (HireVue, Mercer Mettl)
- Remote interview integrity teams at large tech companies
- Trust & Safety systems in enterprise HR workflows

---

## Tasks

### Task 1 — Gaze Diversion Detection `[Easy]`

**Objective:** Detect whether the candidate's eye movement pattern is suspicious.

**Key signals:**
| Signal | Description |
|---|---|
| `gaze_offscreen_ratio` | Fraction of time gaze was off-screen |
| `gaze_direction_entropy` | Shannon entropy of gaze direction histogram |
| `head_turn_angle_deg` | Maximum head rotation angle in degrees |

**Grading:** Binary accuracy — `predicted_flag == ground_truth_flag`

**Max steps:** 5

---

### Task 2 — Whisper & External Assistance Detection `[Medium]`

**Objective:** Identify covert whispering or real-time coaching from a second person.

**Key signals:**
| Signal | Description |
|---|---|
| `whisper_score` | P(whisper) from voice activity detection |
| `second_voice_confidence` | Probability that a second speaker is present |
| `lip_motion_mismatch` | Mismatch between lip movement and audio |
| `response_latency_sec` | Time-to-first-word (delays suggest external prompting) |

**Grading:** Weighted component score combining signal detection accuracy and flag match.

**Max steps:** 7

---

### Task 3 — Reasoning Integrity Verification `[Hard]`

**Objective:** Determine whether a candidate can genuinely explain their own answers under adversarial follow-up questioning.

**Example scenario:**
> Candidate gives a perfect dynamic programming answer.  
> Follow-up: *"Why did you choose prefix sum over a sliding window?"*  
> If `answer_consistency` drops sharply → suspected AI-generated or memorised response.

**Key signals:**
| Signal | Description |
|---|---|
| `answer_consistency` | Semantic similarity between original answer and follow-up explanation |
| `project_followup_score` | How well the candidate justifies their own design choices |
| `complexity_jump` | Magnitude of sudden answer-quality jump unexplained by question change |

**Grading:** Dual-axis score — flag severity correctness (60%) + workflow action correctness (40%).

**Max steps:** 10

---

## Baseline Results

Evaluated using the deterministic **rule-based agent** (no API key required).  
Seed: `42`, Episodes per task: `3`

| Task | Difficulty | Avg Score |
|---|---|---:|
| Gaze Diversion Detection | Easy | **0.8000** |
| Whisper & External Assistance | Medium | **0.7143** |
| Reasoning Integrity Verification | Hard | **0.9540** |
| **Overall Average** | | **0.8228** |

The benchmark was validated through both direct Python execution and isolated Docker container execution, producing **identical deterministic scores** across all tasks.

Performance is strongest on reasoning-integrity verification (**0.9540**), while whisper-based covert assistance detection remains intentionally more ambiguous (**0.7143**) to reflect realistic multimodal uncertainty.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/your-org/TrustHireEnv.git
cd TrustHireEnv
pip install -r requirements.txt
```

---

## Quick Start

### Run the rule-based baseline (no API key needed)
```bash
python baseline_eval.py --no-llm
```

### Run with OpenAI LLM agent
```bash
export OPENAI_API_KEY=sk-...
python baseline_eval.py --model gpt-4o-mini
```

### Run a single task
```bash
python baseline_eval.py --no-llm --task hard --episodes 5
```

### Use the environment directly in Python
```python
from env import TrustHireEnv

env = TrustHireEnv(difficulty="medium", seed=42)
obs = env.reset()

while True:
    # Your agent logic here
    action = {
        "flag_level": "medium",
        "next_step":  "warn",
        "rationale":  "High whisper score detected"
    }
    obs, reward, done, info = env.step(action)
    if done:
        result = info["episode_result"]
        print(f"Task score: {result['task_score']}")
        break
```

---

## Docker

### Build
```bash
docker build -t trusthireenv:latest .
```

### Run (baseline evaluation, no API key needed)
```bash
docker run --rm trusthireenv:latest
```

### Run with OpenAI key
```bash
docker run --rm -e OPENAI_API_KEY=sk-... trusthireenv:latest \
    python baseline_eval.py --model gpt-4o-mini
```

---

## Environment API

### `TrustHireEnv(difficulty, episode_index, seed, max_steps)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `difficulty` | `str` | `"easy"` | `"easy"` / `"medium"` / `"hard"` |
| `episode_index` | `int` | `0` | Which episode to load from the dataset |
| `seed` | `int` | `42` | RNG seed for reproducibility |
| `max_steps` | `int` | task default | Override the episode length |

### Methods

| Method | Returns | Description |
|---|---|---|
| `reset()` | `dict` | Reset environment, return first observation |
| `step(action)` | `(dict, float, bool, dict)` | `(next_obs, reward, done, info)` |
| `state()` | `dict` | Full internal state snapshot |
| `render()` | `str` | Human-readable episode progress |
| `seed(n)` | `None` | Set RNG seed (effective on next reset) |

### Observation Fields

```json
{
  "question_id":             2,
  "difficulty":              "medium",
  "gaze_offscreen_ratio":    0.41,
  "gaze_direction_entropy":  0.62,
  "head_turn_angle_deg":     18.0,
  "whisper_score":           0.55,
  "second_voice_confidence": 0.60,
  "lip_motion_mismatch":     0.50,
  "response_latency_sec":    6.5,
  "answer_consistency":      0.74,
  "project_followup_score":  0.68,
  "complexity_jump":         0.35
}
```

### Action Fields

```json
{
  "flag_level": "high",
  "next_step":  "escalate",
  "rationale":  "Second voice confidence above threshold with high latency"
}
```

---

## Reward Design

Dense shaped rewards — every step returns a non-zero signal, supporting RL training.

| Component | Range | Trigger |
|---|---|---|
| `progress_reward` | `+0.05` per step | Advancing without loops |
| `accuracy_bonus` | `+0.20` to `+0.50` | Correct flag level match |
| `integrity_penalty` | `-0.20` to `-0.70` | False accusation or missed cheating |
| `loop_penalty` | `-1.00` | Same action repeated 3+ times |

Full breakdown is returned in `info["reward_breakdown"]` at every step.

---

## Project Structure

```
TrustHireEnv/
|
+-- env/
|   +-- environment.py     # Core OpenEnv class (reset, step, state)
|   +-- tasks.py           # Task registry and episode loader
|   +-- graders.py         # Deterministic graders (easy / medium / hard)
|   +-- rewards.py         # Dense shaped reward function
|   +-- schemas.py         # Pydantic data contracts (Observation, Action, ...)
|   +-- __init__.py        # Public API
|
+-- datasets/
|   +-- easy.json          # Gaze-diversion episodes
|   +-- medium.json        # Whisper / external assistance episodes
|   +-- hard.json          # Reasoning consistency episodes
|
+-- baseline_eval.py       # Reproducible evaluation script
+-- openenv.yaml           # OpenEnv manifest
+-- Dockerfile             # Docker image definition
+-- requirements.txt       # Python dependencies
+-- README.md              # This file
```

---

## OpenEnv Compliance Checklist

| Requirement | Status |
|---|---|
| `reset()` method | PASS |
| `step(action)` method returning `(obs, reward, done, info)` | PASS |
| `state()` snapshot method | PASS |
| Typed observation schema | PASS |
| Typed action schema | PASS |
| Deterministic graders (reproducible scores) | PASS |
| Dense shaped reward function | PASS |
| 3 tasks (easy / medium / hard) | PASS |
| Dataset files (`easy.json`, `medium.json`, `hard.json`) | PASS |
| `openenv.yaml` manifest | PASS |
| `baseline_eval.py` script with reproducible scores | PASS |
| Docker image with passing smoke-test | PASS |

---

## Motivation

Remote technical interviews are now standard practice, and so are the integrity challenges they introduce. This environment models:

- **Multimodal AI** — combining gaze, audio, and language signals
- **Trust & Safety** — a domain Meta's infrastructure deeply depends on
- **Enterprise HR workflows** — agentic decision support in real hiring pipelines
- **Agentic reasoning** — the agent must reason across steps, not just classify once

TrustHireEnv is designed to be a **qualification-first** OpenEnv submission: clean task design, deterministic grading, and stable Docker reproducibility.

---

## License

MIT License. See `LICENSE` for details.
