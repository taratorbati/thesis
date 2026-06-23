# tests/test_rl_smoke.py
# Behavioural smoke tests for the irrigation env, observation layout, reward,
# and the runner/env observation equivalence.

import numpy as np
import pytest

from src.rl.gym_env import (
    IrrigationEnv, global_dim,
    N_AGENTS, N_AGENT_FEATURES, FORECAST_H,
    X4_REF, X5_REF, RAIN_REF, ETC_REF,
)

AGENT_BLOCK = N_AGENTS * N_AGENT_FEATURES        # 1040
OBS_CLEAN = AGENT_BLOCK + global_dim(True)        # 1092
OBS_LEGACY = AGENT_BLOCK + global_dim(False)      # 1097


# ── env instantiation ─────────────────────────────────────────────────────────
def test_env_training_mode():
    env = IrrigationEnv(randomize=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_CLEAN,)
    assert env.action_space.shape == (N_AGENTS,)


def test_env_fixed_mode_is_deterministic():
    env = IrrigationEnv(randomize=False)
    o1, _ = env.reset(seed=0)
    o2, _ = env.reset(seed=0)
    np.testing.assert_array_equal(o1, o2)


def test_random_reset_varies_year():
    env = IrrigationEnv(randomize=True)
    years = {(env.reset(seed=s), env._year)[1] for s in range(20)}
    assert len(years) > 1


# ── observation layout ────────────────────────────────────────────────────────
def test_obs_dims_clean_and_legacy():
    assert IrrigationEnv(randomize=False).observation_space.shape == (OBS_CLEAN,)
    assert IrrigationEnv(randomize=False, dedupe_today_weather=False
                         ).observation_space.shape == (OBS_LEGACY,)


def test_obs_layout_agent_major():
    env = IrrigationEnv(randomize=False)
    obs, _ = env.reset(seed=0)
    grid = obs[:AGENT_BLOCK].reshape(N_AGENTS, N_AGENT_FEATURES)
    # x4 (biomass, col 2) is ~uniform at reset; elevation (col 4) varies by cell.
    assert np.std(grid[:, 4]) > 0


def test_topo_features_static_across_season():
    env = IrrigationEnv(randomize=False)
    obs0, _ = env.reset(seed=0)
    g0 = obs0[:AGENT_BLOCK].reshape(N_AGENTS, N_AGENT_FEATURES)
    for _ in range(10):
        obs, *_ = env.step(np.full(N_AGENTS, 0.5, np.float32))
    g = obs[:AGENT_BLOCK].reshape(N_AGENTS, N_AGENT_FEATURES)
    for col in (4, 5, 6, 7):                       # elev, Nr, Nr_internal, n_upstream
        np.testing.assert_array_equal(g0[:, col], g[:, col])


def test_dedupe_removes_exactly_the_repeated_today_weather():
    """The clean obs must equal the legacy obs minus the 5 today-weather scalars,
    and each dropped scalar must equal its forecast day-0 value (the duplication)."""
    clean = IrrigationEnv(randomize=False)
    legacy = IrrigationEnv(randomize=False, dedupe_today_weather=False)
    oc, _ = clean.reset(seed=0)
    ol, _ = legacy.reset(seed=0)

    # agent blocks identical
    np.testing.assert_array_equal(oc[:AGENT_BLOCK], ol[:AGENT_BLOCK])

    g_clean = oc[AGENT_BLOCK:]                     # [4 base scalars] + [48 forecast]
    g_legacy = ol[AGENT_BLOCK:]                    # [9 scalars]      + [48 forecast]
    # first 4 scalars and the 48-dim forecast tail are shared
    np.testing.assert_array_equal(g_clean[:4], g_legacy[:4])
    np.testing.assert_array_equal(g_clean[4:], g_legacy[9:])

    # the 5 dropped scalars (legacy[4:9]) duplicate forecast day-0 of each channel
    fc = g_clean[4:]                              # 48-dim forecast block
    dropped = g_legacy[4:9]                       # rain, ETc, h2, h7, g_base today
    # forecast layout: rain[0:8], ETc[0:8], rad[0:8], h2[0:8], h7[0:8], g[0:8]
    expected = np.array([fc[0], fc[FORECAST_H], fc[3 * FORECAST_H],
                         fc[4 * FORECAST_H], fc[5 * FORECAST_H]], dtype=np.float32)
    np.testing.assert_allclose(dropped, expected, atol=1e-6)


# ── eval schedule ─────────────────────────────────────────────────────────────
def test_eval_schedule_deterministic_and_rewindable():
    sched = [(2002, 0.7), (2016, 1.0), (2023, 0.85)]
    env = IrrigationEnv(randomize=False, eval_schedule=sched)
    # schedule walks in order
    env.reset(); assert env._year == 2002
    env.reset(); assert env._year == 2016
    # rewind makes evaluation repeatable
    env.reset_eval_schedule()
    first = env.reset()[0]
    env.reset(); env.reset()
    env.reset_eval_schedule()
    np.testing.assert_array_equal(first, env.reset()[0])


def test_eval_schedule_none_keeps_randomize():
    env = IrrigationEnv(randomize=True)            # no eval_schedule
    env.reset(seed=0)
    assert env._eval_schedule is None


def test_eval_schedule_rejects_malformed():
    with pytest.raises(ValueError):
        IrrigationEnv(randomize=False, eval_schedule=[(2002,)])
    with pytest.raises(ValueError):
        IrrigationEnv(randomize=False, eval_schedule=[])


# ── reward ────────────────────────────────────────────────────────────────────
def test_reward_is_finite():
    env = IrrigationEnv(randomize=False)
    env.reset(seed=0)
    for _ in range(20):
        _, r, _, _, _ = env.step(np.full(N_AGENTS, 0.5, np.float32))
        assert np.isfinite(r)


def test_episode_truncates_at_full_season():
    env = IrrigationEnv(randomize=False)
    env.reset(seed=0)
    truncs = 0
    for _ in range(95):
        _, _, term, trunc, _ = env.step(np.zeros(N_AGENTS, np.float32))
        truncs += int(trunc)
        if trunc:
            break
    assert truncs == 1


def test_water_clipped_at_budget():
    env = IrrigationEnv(randomize=False)
    env.reset(seed=0)
    last = {}
    for _ in range(93):
        _, _, _, _, last = env.step(np.ones(N_AGENTS, np.float32))
    assert last['water_used_mm'] <= env._budget_mm + 1e-6


# ── delta-u (control-rate) reward ─────────────────────────────────────────────
def _collect(env, actions):
    env.reset(seed=0)
    out = []
    for a in actions:
        _, r, _, _, info = env.step(np.full(N_AGENTS, float(a), np.float32))
        info = dict(info); info['_reward'] = r
        out.append(info)
    return out


def test_delta_u_disabled_by_default():
    infos = _collect(IrrigationEnv(randomize=False), [0.2, 0.9, 0.1])
    assert all(i['r5_delta_u'] == 0.0 for i in infos)


def test_delta_u_zero_on_first_step():
    infos = _collect(IrrigationEnv(randomize=False, reward_du_alpha=0.005), [0.5, 0.5])
    assert infos[0]['r5_delta_u'] == 0.0


def test_delta_u_penalises_jerk_not_smoothness():
    smooth = _collect(IrrigationEnv(randomize=False, reward_du_alpha=0.005), [0.5, 0.5, 0.5])
    jerky = _collect(IrrigationEnv(randomize=False, reward_du_alpha=0.005), [0.1, 0.9, 0.1])
    assert sum(i['r5_delta_u'] for i in jerky) < sum(i['r5_delta_u'] for i in smooth) <= 0.0


def test_reward_decomposition_sums_to_total():
    infos = _collect(IrrigationEnv(randomize=False, reward_du_alpha=0.005), [0.3, 0.8, 0.2])
    for i in infos[:-1]:   # exclude terminal step (carries the terminal bonus)
        parts = (i['r1_biomass'] + i['r2_water'] + i['r3_drought']
                 + i['r5_delta_u'] + i['r6_waterlog'])
        assert np.isclose(parts, i['_reward'], atol=1e-6)


def test_terminal_yield_bonus_applied_once():
    infos = _collect(IrrigationEnv(randomize=False, reward_terminal_yield=1.0),
                     [0.5] * 93)
    assert infos[-1]['r_term_yield'] > 0.0
    assert all(i['r_term_yield'] == 0.0 for i in infos[:-1])


# ── runner / env observation equivalence ──────────────────────────────────────
def test_runner_imports():
    from src.rl.runner import RLController, load_policy  # noqa: F401


def test_runner_dynamic_features_match_env():
    """Runner per-cell dynamic formulas must match gym_env bit-for-bit."""
    from src.model.soil_data import get_crop
    from src.model.terrain import load_terrain
    crop = get_crop('rice')
    fc = crop['theta6'] * crop['theta5']
    wp = crop['theta2'] * crop['theta5']
    x1_range = max(fc - wp, 1e-6)

    def runner_feats(s):
        return (
            np.clip((s['x1'] - wp) / x1_range, 0.0, 1.5).astype(np.float32),
            np.clip(s['x5'] / X5_REF, 0.0, 2.0).astype(np.float32),
            np.clip(s['x4'] / X4_REF, 0.0, 1.5).astype(np.float32),
            np.clip(s['x3'], 0.0, 2.0).astype(np.float32),
        )

    env = IrrigationEnv(randomize=False)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(42)
    for _ in range(30):
        obs, *_ = env.step(rng.uniform(0.3, 1.0, (N_AGENTS,)).astype(np.float32))
        grid = obs[:AGENT_BLOCK].reshape(N_AGENTS, N_AGENT_FEATURES)
        x1r, x5r, x4r, x3r = runner_feats(
            {'x1': env._abm.x1, 'x5': env._abm.x5, 'x4': env._abm.x4, 'x3': env._abm.x3})
        np.testing.assert_allclose(grid[:, 0], x1r, atol=1e-5)
        np.testing.assert_allclose(grid[:, 1], x5r, atol=1e-5)
        np.testing.assert_allclose(grid[:, 2], x4r, atol=1e-5)
        np.testing.assert_allclose(grid[:, 3], x3r, atol=1e-5)
