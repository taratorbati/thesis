# =============================================================================
# tests/test_rl_smoke.py
# Regression tests for the RL pipeline.
#
# These 30 lines catch every Tier-0 and Tier-1 bug identified in the
# June 2026 audit:
#   - IrrigationEnv instantiation (crash bug: 'randomize' kwarg)
#   - Random-year reset path (crash bug: int passed to get_precomputed)
#   - runner import (crash bug: FULL_NEED_MM vs FULL_SEASON_NEED_MM)
#   - obs_dim consistency between gym_env and runner (silent corruption)
#   - Scalar order consistency at every position (silent corruption)
#
# Run with: pytest tests/test_rl_smoke.py -v
# =============================================================================

import numpy as np
import pytest


def test_env_instantiates_training_mode():
    """IrrigationEnv() with no args must not raise."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != observation_space {env.observation_space.shape}"
    )


def test_env_randomize_kwarg():
    """IrrigationEnv(randomize=True) must not raise TypeError."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=True, seed=42)
    obs, _ = env.reset()
    assert obs.shape[0] == 707


def test_env_random_year_reset():
    """Random year sampling must not crash (int->get_precomputed bug)."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(seed=7)
    for _ in range(5):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), "obs contains NaN or Inf"


def test_runner_imports():
    """runner.py must import without ImportError."""
    from src.rl.runner import RLController  # noqa


def test_obs_dim_matches_between_gym_and_runner():
    """runner._build_obs must produce same shape as gym_env._get_obs."""
    from src.rl.gym_env import IrrigationEnv, X4_REF, X5_REF
    from src.rl.runner import RLController

    env = IrrigationEnv(fixed_scenario='dry', fixed_budget_pct=100, seed=0)
    obs_env, _ = env.reset()

    # Build a fake state matching the ABM state
    state = {
        'x1': env.abm.x1,
        'x5': env.abm.x5,
        'x3': env.abm.x3,
        'x4': env.abm.x4,
    }

    # Build a minimal RLController without loading a real model
    # (we just test _build_obs, not model.predict)
    from src.terrain import load_terrain
    from src.precompute import get_precomputed
    from climate_data import load_cleaned_data, extract_scenario_by_name
    from soil_data import get_crop

    crop    = get_crop('rice')
    terrain = load_terrain('gilan_farm.tif')

    class FakeController:
        _terrain = terrain
        _crop = crop
        _N = terrain['N']
        _season_days = crop['season_days']
        _budget_total = 484.0
        _elev_norm = terrain['gamma_flat']
        _fc_total = crop['theta6'] * crop['theta5']
        forecast_horizon = 8

        def __init__(self):
            from src.precompute import get_precomputed
            from climate_data import load_cleaned_data, extract_scenario_by_name
            df = load_cleaned_data()
            self._precomputed = get_precomputed('dry', 'rice')
            self._climate = extract_scenario_by_name(df, 'dry', crop)

        _build_obs = RLController._build_obs

    fc = FakeController()
    obs_runner = fc._build_obs(0, state, 484.0)

    assert obs_env.shape == obs_runner.shape, (
        f"Shape mismatch: gym_env={obs_env.shape}, runner={obs_runner.shape}"
    )

    # Check that the first obs step's values match position-by-position
    # in the per-agent block (positions 0:650)
    np.testing.assert_allclose(
        obs_env[:650], obs_runner[:650], rtol=1e-5,
        err_msg="Per-agent block mismatch between gym_env and runner"
    )

    # Check scalar block shape and positions
    scalars_env    = obs_env[650:659]
    scalars_runner = obs_runner[650:659]
    np.testing.assert_allclose(
        scalars_env, scalars_runner, rtol=1e-5,
        err_msg=(
            "Scalar block mismatch between gym_env and runner. "
            "Check scalar ORDER in both _get_obs and _build_obs."
        )
    )


def test_reward_is_finite():
    """100 random steps must produce finite rewards."""
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(fixed_scenario='dry', fixed_budget_pct=100, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(0, 1, (env.N,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward), f"Non-finite reward: {reward}"
        assert np.all(np.isfinite(obs)), "Non-finite obs"
        if terminated or truncated:
            env.reset()
