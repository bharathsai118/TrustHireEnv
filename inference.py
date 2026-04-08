from env.environment import TrustHireEnv

# persistent environment instance for checker calls
_env = None


def reset(task: str = "easy", seed: int = 42):
    """
    Reset environment for a given task.
    Called by Meta OpenEnv automated checker.
    """
    global _env
    _env = TrustHireEnv(difficulty=task, seed=seed)
    return _env.reset()


def step(action: dict):
    """
    Execute one environment step.
    Called by Meta OpenEnv automated checker.
    """
    global _env

    if _env is None:
        raise RuntimeError("Environment not initialized. Call reset() first.")

    obs, reward, done, info = _env.step(action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


def run():
    """
    Optional local smoke-test.
    """
    obs = reset("easy")
    result = step({
        "flag_level": "none",
        "next_step": "continue",
        "rationale": "smoke test"
    })
    return result


if __name__ == "__main__":
    print(run())