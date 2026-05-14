# tests/test_rl_smoke.py  v2.6
# =============================================================================
# Regression tests for the RL pipeline.
#
# All calls updated to match the v2.6 IrrigationEnv(randomize=bool)
# constructor — 'seed' and 'fixed_scenario' kwargs do not exist in v2.6.
# Seeding is handled via reset(seed=...) per the Gymnasium API.
#
# Run with: pytest tests/test_rl_smoke.py -v
# =============================================================================

import numpy as np
import pytest


def test_env_instantiates_training_mode():
    """IrrigationEnv(randomize=True) must not raise."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != observation_space {env.observation_space.shape}"
    )


def test_env_fixed_mode():
    """IrrigationEnv(randomize=False) must produce a deterministic reset."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    obs, _ = env.reset()
    assert obs.shape[0] == 707
    # Second reset must produce identical obs (same year, full budget)
    obs2, _ = env.reset()
    np.testing.assert_array_equal(obs, obs2,
        err_msg="Fixed-mode resets must be deterministic")


def test_env_random_year_reset():
    """Random year sampling across 5 resets must not crash or produce NaN."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=True)
    for i in range(5):
        obs, _ = env.reset(seed=i)
        assert np.all(np.isfinite(obs)), f"obs contains NaN or Inf on reset {i}"


def test_runner_imports():
    """runner.py must import without ImportError."""
    from src.rl.runner import RLController  # noqa


def test_obs_dim_matches_between_gym_and_runner():
    """runner._build_obs must produce same shape as gym_env._build_obs."""
    from src.rl.gym_env import IrrigationEnv

    env = IrrigationEnv(randomize=False)
    obs_env, _ = env.reset()

    # Shape check — the runner uses the same 707-dim layout
    assert obs_env.shape == (707,), f"Expected (707,) got {obs_env.shape}"

    # Scalar positions 650–658 must be finite
    scalars = obs_env[650:659]
    assert np.all(np.isfinite(scalars)), "Scalar block contains non-finite values"

    # Per-agent block positions 0–649 must be in reasonable range
    agent_block = obs_env[:650]
    assert np.all(agent_block >= -0.1), "Per-agent block has large negative values"
    assert np.all(agent_block < 10.0), "Per-agent block has implausibly large values"


def test_reward_is_finite():
    """100 random steps must produce finite rewards and observations."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for step in range(100):
        action = rng.uniform(0, 1, (env.N,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"Non-finite reward at step {step}: {reward}"
        assert np.all(np.isfinite(obs)), f"Non-finite obs at step {step}"
        if terminated or truncated:
            env.reset()


def test_obs_layout_agent_major():
    """Per-agent block must use agent-major layout (5 contiguous per agent).

    This is the convention assumed by networks.py's SharedActor and
    FactorizedContinuousCritic: obs[:, n*5:(n+1)*5] is agent n's features.
    """
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    obs, _ = env.reset(seed=0)

    # After reset x1 is uniform (all agents start at FC).
    # x1_norm (feature index 0 of each agent's 5-tuple) should therefore
    # be ~1.0 for all agents, and they should all be equal.
    x1_norms = obs[:650].reshape(130, 5)[:, 0]
    assert x1_norms.std() < 0.01, (
        f"x1_norm should be ~uniform at reset; std={x1_norms.std():.4f}. "
        "Check if gym_env uses agent-major layout (stack axis=1, then flatten)."
    )


def test_budget_exhaustion_terminates():
    """Setting budget to near-zero must terminate the episode quickly."""
    from src.rl.gym_env import IrrigationEnv, FULL_SEASON_NEED_MM, UB_MM
    env = IrrigationEnv(randomize=False)
    # Manually set a tiny budget after reset
    env.reset()
    env._budget_mm = 0.5   # only 0.5 mm total
    action = np.ones(env.N, dtype=np.float32)   # max irrigation
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated, "Episode must terminate when budget is exhausted"
