# tests/test_networks.py
# Architecture tests for the SAC and TD3 CTDE VDN networks.
#
# DEVICE-SAFE VERSION: every hand-built tensor is created on ``model.device``
# so the forward-pass tests run identically on a CPU runtime (where SB3 keeps
# the policy on CPU) and on a GPU runtime (where SB3 moves the policy to cuda).
# The previous version built CPU tensors unconditionally, which matched the
# weights on a CPU runtime but raised a device-mismatch RuntimeError on a GPU
# runtime — the cause of the Colab "TESTS FAILED" assertion.

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC, TD3

from src.rl.gym_env import IrrigationEnv, global_dim
from src.rl.networks import (
    SacVdnPolicy, VdnCritic, SharedActor, make_sac_policy_kwargs,
    N_AGENT_FEATURES, N_AGENTS_DEFAULT,
)
from src.rl.networks_td3 import Td3VdnPolicy, DeterministicSharedActor, make_td3_policy_kwargs


@pytest.fixture(scope="module")
def clean_env():
    return IrrigationEnv(randomize=False)            # deduped obs (1092-dim)


@pytest.fixture(scope="module")
def legacy_env():
    return IrrigationEnv(randomize=False, dedupe_today_weather=False)   # 1097-dim


def _build_sac(env):
    return SAC(policy=SacVdnPolicy, env=env, policy_kwargs=make_sac_policy_kwargs(),
               buffer_size=500, learning_starts=10, batch_size=32, verbose=0, seed=0)


def _build_td3(env):
    return TD3(policy=Td3VdnPolicy, env=env, policy_kwargs=make_td3_policy_kwargs(),
               buffer_size=500, learning_starts=10, batch_size=32, verbose=0, seed=0)


def _obs_tensor(env, model):
    """Build a single batched observation on the SAME device as the model."""
    return torch.as_tensor(env.reset()[0], device=model.device).float().unsqueeze(0)


# ── dimension inference ───────────────────────────────────────────────────────
def test_per_agent_dims_inferred_from_obs(clean_env, legacy_env):
    """Critic first-layer width = 8 + G + 1, derived from the observation width."""
    m_clean, m_legacy = _build_sac(clean_env), _build_sac(legacy_env)
    assert m_clean.policy.critic.qf0[0].in_features == N_AGENT_FEATURES + global_dim(True) + 1   # 61
    assert m_legacy.policy.critic.qf0[0].in_features == N_AGENT_FEATURES + global_dim(False) + 1  # 66


# ── SAC actor / critic shapes ─────────────────────────────────────────────────
def test_sac_actor_and_critic_shapes(clean_env):
    model = _build_sac(clean_env)
    obs = _obs_tensor(clean_env, model)
    assert isinstance(model.policy.actor, SharedActor)
    action = model.policy.actor(obs)
    assert action.shape == (1, N_AGENTS_DEFAULT)
    assert torch.all(action >= -1) and torch.all(action <= 1)
    q1, q2 = model.policy.critic(obs, action)
    assert q1.shape == (1, 1) and q2.shape == (1, 1)


def test_vdn_critic_is_sum_over_cells(clean_env):
    """Q_total must equal the sum of the per-cell local Q-values (VDN property)."""
    model = _build_sac(clean_env)
    critic: VdnCritic = model.policy.critic
    obs = _obs_tensor(clean_env, model)
    action = torch.zeros(1, N_AGENTS_DEFAULT, device=model.device)
    qnet = critic.q_networks[0]
    B, N, F = 1, N_AGENTS_DEFAULT, N_AGENT_FEATURES
    local_obs = obs[:, : F * N].reshape(B, N, F)
    global_expanded = obs[:, F * N:].unsqueeze(1).expand(-1, N, -1)
    local_inputs = torch.cat([local_obs, global_expanded, action.reshape(B, N, 1)], dim=-1)
    per_cell = torch.nn.Sequential.forward(qnet, local_inputs.reshape(B * N, -1)).reshape(B, N)
    q_total = qnet(obs, action)
    assert torch.allclose(q_total.squeeze(), per_cell.sum(), atol=1e-4)


def test_sac_actor_has_marker_buffer(clean_env):
    """The obs_norm_marker buffer must exist (checkpoint state-dict compatibility)."""
    model = _build_sac(clean_env)
    assert "obs_norm_marker" in dict(model.policy.actor.named_buffers())


# ── TD3 actor ─────────────────────────────────────────────────────────────────
def test_td3_actor_is_deterministic_with_no_log_std(clean_env):
    model = _build_td3(clean_env)
    assert isinstance(model.policy.actor, DeterministicSharedActor)
    assert not hasattr(model.policy.actor, "log_std") or "log_std" not in dict(
        model.policy.actor.named_parameters())
    obs = _obs_tensor(clean_env, model)
    a1 = model.policy.actor(obs)
    a2 = model.policy.actor(obs)
    assert a1.shape == (1, N_AGENTS_DEFAULT)
    assert torch.allclose(a1, a2)                       # deterministic
    assert torch.all(a1 >= -1) and torch.all(a1 <= 1)


def test_td3_reuses_sac_vdn_critic(clean_env):
    model = _build_td3(clean_env)
    assert isinstance(model.policy.critic, VdnCritic)


# ── short training smoke ──────────────────────────────────────────────────────
def test_sac_learns_a_few_steps(clean_env):
    model = _build_sac(clean_env)
    model.learn(total_timesteps=30, progress_bar=False)


def test_td3_learns_a_few_steps(clean_env):
    model = _build_td3(clean_env)
    model.learn(total_timesteps=30, progress_bar=False)
