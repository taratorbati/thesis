# tests/test_rl_smoke.py  v2.7
# =============================================================================
# Regression tests for the v2.7 RL pipeline.
#
# v2.7 changes:
#   - OBS_DIM is now 1097 (was 707): 8 features × 130 agents + 9 scalars + 48
#     forecast.  Slicing indices for the scalar block move from [650:659]
#     to [1040:1049]; per-agent block ends at index 1040, not 650.
#   - Episode now ALWAYS runs to the full 93-day season.  Budget exhaustion
#     no longer terminates the episode early.  The corresponding test was
#     rewritten from test_budget_exhaustion_terminates to
#     test_episode_runs_full_season_after_budget_exhaustion, asserting the
#     opposite behaviour.
#   - Two new tests validate the v2.7 per-agent feature additions:
#     test_per_agent_topo_features_vary_across_agents and
#     test_per_agent_topo_features_static_across_season.
#
# Run with: pytest tests/test_rl_smoke.py -v
# =============================================================================

import numpy as np
import pytest


# Module-level constants for clarity; these match v2.7 gym_env.py
V27_OBS_DIM         = 1097
V27_N_AGENT_FEAT    = 8
N_AGENTS            = 130
PER_AGENT_BLOCK_END = V27_N_AGENT_FEAT * N_AGENTS   # = 1040
SCALAR_BLOCK_END    = PER_AGENT_BLOCK_END + 9       # = 1049


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
    assert obs.shape[0] == V27_OBS_DIM, (
        f"Expected obs of shape ({V27_OBS_DIM},), got {obs.shape}"
    )
    # Second reset must produce identical obs (same year, full budget)
    obs2, _ = env.reset()
    np.testing.assert_array_equal(
        obs, obs2,
        err_msg="Fixed-mode resets must be deterministic",
    )


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


def test_obs_dim_matches_v27_layout():
    """The env must produce exactly the 1097-dim v2.7 obs layout.

    Per-agent block (positions [0, 1040)): 8 features × 130 agents,
    agent-major.  Scalar block (positions [1040, 1049)).  Forecast block
    (positions [1049, 1097)).
    """
    from src.rl.gym_env import IrrigationEnv

    env = IrrigationEnv(randomize=False)
    obs_env, _ = env.reset()

    assert obs_env.shape == (V27_OBS_DIM,), (
        f"Expected ({V27_OBS_DIM},) got {obs_env.shape}"
    )

    # Scalar positions 1040–1048 must be finite
    scalars = obs_env[PER_AGENT_BLOCK_END:SCALAR_BLOCK_END]
    assert np.all(np.isfinite(scalars)), (
        f"Scalar block contains non-finite values: {scalars}"
    )

    # Per-agent block positions [0, 1040) must be in plausible range
    agent_block = obs_env[:PER_AGENT_BLOCK_END]
    assert np.all(agent_block >= -0.1), (
        "Per-agent block has large negative values"
    )
    assert np.all(agent_block < 10.0), (
        "Per-agent block has implausibly large values"
    )


def test_reward_is_finite():
    """100 random steps must produce finite rewards and observations.

    Note: in v2.7 the episode never terminates from budget exhaustion, so
    100 random steps will not reset unless the season ends (day >= 93).
    """
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
    """Per-agent block must use agent-major layout (8 contiguous per agent).

    This is the convention assumed by networks.py's SharedActor and
    FactorizedContinuousCritic: obs[:, n*8:(n+1)*8] is agent n's features.
    """
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    obs, _ = env.reset(seed=0)

    # After reset x1 is uniform (all agents start at FC).
    # x1_norm (feature index 0 of each agent's 8-tuple) should therefore
    # be uniform across agents.
    x1_norms = obs[:PER_AGENT_BLOCK_END].reshape(N_AGENTS, V27_N_AGENT_FEAT)[:, 0]
    assert x1_norms.std() < 0.01, (
        f"x1_norm should be ~uniform at reset; std={x1_norms.std():.4f}. "
        "Check if gym_env uses agent-major layout (stack axis=1, then flatten)."
    )


def test_per_agent_topo_features_vary_across_agents():
    """v2.7 bug fix: per-agent static features MUST vary across the 130 agents.

    The v2.6 implementation had x2/theta18 (a field-uniform GDD scalar) in
    the 5th per-agent slot, which gave the actor no per-agent information.
    In v2.7, slots 4-7 hold normalised elevation, Nr/8, Nr_internal/8, and
    n_upstream/8 — all of which depend on agent identity and must therefore
    have non-zero std across agents.
    """
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    obs, _ = env.reset(seed=0)

    agent_grid = obs[:PER_AGENT_BLOCK_END].reshape(N_AGENTS, V27_N_AGENT_FEAT)

    feature_names = [
        'x1_norm', 'x5_norm', 'x4_norm', 'x3',
        'elev_norm', 'Nr_norm', 'Nr_internal_norm', 'n_upstream_norm',
    ]

    # Static topographic features must have non-trivial spread across agents.
    # We test slots 4, 5, 6, 7 — the four static topographic features.
    for slot in [4, 5, 6, 7]:
        feat = agent_grid[:, slot]
        assert feat.std() > 0.01, (
            f"Static feature '{feature_names[slot]}' (slot {slot}) has "
            f"std={feat.std():.6f} across the 130 agents — should be > 0.01. "
            f"This indicates the feature is field-uniform (likely the v2.6 "
            f"x2/theta18 regression bug has come back)."
        )


def test_per_agent_topo_features_static_across_season():
    """v2.7: the 4 static topographic features must NOT change during a season.

    Features at slots 4-7 (elev_norm, Nr_norm, Nr_internal_norm,
    n_upstream_norm) are derived from the DEM and are constant for the
    duration of the simulation.  Any change across steps indicates the
    obs builder is accidentally writing dynamic state into a static slot.
    """
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    obs0, _ = env.reset(seed=0)
    static0 = obs0[:PER_AGENT_BLOCK_END].reshape(N_AGENTS, V27_N_AGENT_FEAT)[:, 4:8].copy()

    rng = np.random.default_rng(0)
    for step in range(5):
        action = rng.uniform(0, 1, (env.N,)).astype(np.float32)
        obs, _, _, _, _ = env.step(action)
        static_now = obs[:PER_AGENT_BLOCK_END].reshape(N_AGENTS, V27_N_AGENT_FEAT)[:, 4:8]
        np.testing.assert_allclose(
            static_now, static0,
            err_msg=(
                f"Static topographic features changed between reset and step {step}. "
                "These should be constant for the entire season."
            ),
        )


def test_episode_runs_full_season_after_budget_exhaustion():
    """v2.7 behaviour: budget exhaustion does NOT terminate the episode.

    Setting the budget to a tiny value and pumping the agent at full
    irrigation must NOT raise terminated=True.  The episode should keep
    running until day 93, at which point truncated=True fires (season end).
    After budget is exhausted, the env clip drives effective irrigation
    to 0 and the agent feels late-season drought through r3 and reduced r1.
    """
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    env.reset()
    env._budget_mm = 0.5   # only 0.5 mm of total budget

    action = np.ones(env.N, dtype=np.float32)   # max irrigation request

    # First step: should NOT terminate, even though budget exhausts in
    # one step because mean(action × UB_MM) = 12 mm vs 0.5 mm remaining.
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated, (
        "v2.7: budget exhaustion must NOT terminate the episode "
        "(terminated should always be False)."
    )
    assert not truncated, (
        "Episode should not be truncated after only 1 step (season is 93 days)."
    )
    assert info['water_used_mm'] <= 0.5 + 1e-6, (
        f"Cumulative water {info['water_used_mm']:.6f} must not exceed "
        f"budget 0.5 mm — the per-step clip should enforce this."
    )

    # Keep stepping for 91 more days; episode must continue.
    # After the first step above _day = 1, so 91 more steps brings _day to 92.
    for d in range(91):
        obs, r, term, trunc, info = env.step(action)
        assert not term, f"Should not terminate at day {d+2} (no early termination in v2.7)."

    # On the 93rd step total (_day becomes 93 == _K), truncated must fire.
    obs, r, term, trunc, info = env.step(action)
    assert trunc, "By day 93 the episode must be truncated (season end)."
    assert not term, "Even at season end, terminated should be False in v2.7."


def test_water_clipped_at_budget():
    """The env clip must prevent water_used from exceeding budget_total.

    Pump the env at maximum irrigation for the full season; water_used
    must equal budget exactly (within floating-point tolerance) and never
    exceed it.
    """
    from src.rl.gym_env import IrrigationEnv
    env = IrrigationEnv(randomize=False)
    env.reset()
    full_budget = env._budget_mm
    action = np.ones(env.N, dtype=np.float32)

    for _ in range(93):
        obs, r, term, trunc, info = env.step(action)
        assert info['water_used_mm'] <= full_budget + 1e-6, (
            f"water_used {info['water_used_mm']:.6f} exceeded "
            f"budget {full_budget:.6f}"
        )
        if trunc:
            break

    # After 93 days at max irrigation, cumulative water should equal budget.
    assert abs(info['water_used_mm'] - full_budget) < 1e-3, (
        f"Cumulative water {info['water_used_mm']:.4f} should equal "
        f"budget {full_budget:.4f} after a season of max irrigation."
    )
