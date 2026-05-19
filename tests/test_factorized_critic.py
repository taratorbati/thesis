# tests/test_factorized_critic.py  v2.8
# ─────────────────────────────────────────────────────────────────────────────
# Pre-cloud validation tests for the v2.8 networks.
#
# v2.8 changes:
#   - OBS_DIM_DEFAULT is now 1227 (flows automatically).
#   - Added test_v27_legacy_load_shape — guards the v2.7 checkpoint load path.
#   - Existing v2.6 legacy load test renamed for clarity.
#
# Usage:   pytest tests/test_factorized_critic.py -v
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.rl.networks import (
    # v2.8 default
    SharedActor,
    FactorizedContinuousCritic,
    CTDESACPolicy,
    make_sac_policy_kwargs,
    N_AGENT_FEATURES,
    N_GLOBAL_DIMS,
    OBS_DIM_DEFAULT,
    # v2.7 legacy
    V27_OBS_DIM,
    V27_N_AGENT_FEATURES,
    V27_PER_AGENT_CRITIC_INPUT_DIM,
    V27CTDESACPolicy,
    _V27SharedActor,
    _V27FactorizedContinuousCritic,
    # v2.6 legacy
    V26_OBS_DIM,
    V26_N_AGENT_FEATURES,
    V26_PER_AGENT_CRITIC_INPUT_DIM,
    WrappedVDNCTDESACPolicy,
    _LegacySharedActor,
    _WrappedFactorizedCritic,
)


N = 130
OBS_DIM = OBS_DIM_DEFAULT   # 1227 in v2.8
B = 32


@pytest.fixture
def spaces_fixture():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
    extractor = FlattenExtractor(obs_space)
    return obs_space, act_space, extractor


@pytest.fixture
def critic(spaces_fixture):
    obs_space, act_space, extractor = spaces_fixture
    return FactorizedContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=extractor,
        features_dim=OBS_DIM,
        activation_fn=torch.nn.ReLU,
        n_critics=2,
        share_features_extractor=True,
        N=N,
    )


@pytest.fixture
def actor(spaces_fixture):
    obs_space, act_space, extractor = spaces_fixture
    return SharedActor(
        N=N,
        observation_space=obs_space,
        action_space=act_space,
        features_extractor=extractor,
        features_dim=OBS_DIM,
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
    )


# ── Test 1: shape correctness ────────────────────────────────────────────────
def test_critic_output_shape(critic):
    obs = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N)
    out = critic(obs, actions)
    assert isinstance(out, tuple), f"Critic must return a tuple; got {type(out)}"
    assert len(out) == 2
    for i, q in enumerate(out):
        assert q.shape == (B, 1), f"Q{i+1} shape {q.shape} != ({B}, 1)"


def test_q1_forward_shape(critic):
    obs = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N)
    q1 = critic.q1_forward(obs, actions)
    assert q1.shape == (B, 1), f"q1_forward shape {q1.shape} != ({B}, 1)"


def test_actor_output_shape(actor):
    obs = torch.randn(B, OBS_DIM)
    action = actor(obs, deterministic=False)
    assert action.shape == (B, N), f"Actor shape {action.shape} != ({B}, {N})"


# ── Test 2: gradient localisation ────────────────────────────────────────────
def test_gradient_localisation(critic):
    obs = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N, requires_grad=True)
    q1, _ = critic(obs, actions)
    q1.sum().backward()

    grad = actions.grad
    grad_mag_per_agent = grad.abs().mean(dim=0)

    assert torch.all(grad_mag_per_agent > 0), \
        "Some agents have zero gradient — decomposition is broken."

    ratio = grad_mag_per_agent.max() / grad_mag_per_agent.min()
    assert ratio < 5.0, (
        f"Gradient ratio max/min = {ratio:.2f} across agents — "
        "decomposition may not be giving balanced per-agent signal."
    )


def test_gradient_only_through_relevant_agent(critic):
    obs = torch.randn(B, OBS_DIM)
    actions_input = torch.zeros(B, N, requires_grad=True)

    q1, _ = critic(obs, actions_input)
    q1.sum().backward()

    grad = actions_input.grad
    assert grad is not None, "Gradient is None — actions_input must be a leaf tensor"

    per_agent_norm = grad.abs().mean(dim=0)
    assert torch.all(per_agent_norm > 0), (
        f"Some agents have zero gradient — VDN decomposition may be broken. "
        f"Zero-grad agents: {(per_agent_norm == 0).nonzero().flatten().tolist()}"
    )


# ── Test 3: SB3 SAC integration smoke test ───────────────────────────────────
def test_sac_integration_smoke():
    """Build a full SAC model with CTDESACPolicy and run a few training steps."""
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    class DummyEnv(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        action_space      = spaces.Box(0.0, 1.0, shape=(N,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._steps = 0
            return np.random.randn(OBS_DIM).astype(np.float32), {}

        def step(self, action):
            self._steps += 1
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            reward = float(-np.mean(action))
            terminated = self._steps >= 10
            return obs, reward, terminated, False, {}

    env = DummyVecEnv([DummyEnv])

    model = SAC(
        policy=CTDESACPolicy,
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000,
        batch_size=64,
        learning_starts=64,
        ent_coef=0.05,
        gamma=0.99,
        tau=0.005,
        policy_kwargs=make_sac_policy_kwargs(N=N),
        verbose=0,
    )

    model.learn(total_timesteps=200)

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.zip")
        model.save(path)
        model2 = SAC.load(path, env=env)
        obs = env.reset()
        action, _ = model2.predict(obs, deterministic=True)
        assert action.shape == (1, N), f"Loaded model action shape {action.shape} != (1, {N})"


def test_parameter_counts(actor, critic):
    """Sanity check — v2.8 first layer is 67-input critic."""
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())

    print(f"\n  SharedActor parameters:                  {actor_params:>8,}")
    print(f"  FactorizedContinuousCritic parameters:   {critic_params:>8,}")

    assert 50_000 < critic_params < 300_000, (
        f"Factorized critic param count {critic_params} outside expected range"
    )


# ── v2.8: v2.7 legacy load shape guard ───────────────────────────────────────
def test_v27_legacy_load_shape():
    """v2.7 V27CTDESACPolicy must build with the 1097-dim obs space.

    Guards the v2.7 best_model.zip load path.  Without this, a refactor that
    changes only v2.8 dimensions could silently break v2.7 checkpoint loading.
    """
    legacy_obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(V27_OBS_DIM,), dtype=np.float32
    )
    act_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
    extractor = FlattenExtractor(legacy_obs_space)

    legacy_actor = _V27SharedActor(
        N=N,
        observation_space=legacy_obs_space,
        action_space=act_space,
        features_extractor=extractor,
        features_dim=V27_OBS_DIM,
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
    )

    obs_legacy = torch.randn(B, V27_OBS_DIM)
    action = legacy_actor(obs_legacy, deterministic=False)
    assert action.shape == (B, N), (
        f"v2.7 actor output shape {action.shape} != ({B}, {N}). "
        "The v2.7 obs layout (8 features/agent) is broken."
    )

    legacy_critic = _V27FactorizedContinuousCritic(
        observation_space=legacy_obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=extractor,
        features_dim=V27_OBS_DIM,
        activation_fn=torch.nn.ReLU,
        n_critics=2,
        share_features_extractor=True,
        N=N,
    )

    actions_in = torch.rand(B, N)
    q1, q2 = legacy_critic(obs_legacy, actions_in)
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)

    # First layer of v2.7 critic should be (256, 66), NOT (256, 67)
    first_layer = legacy_critic.qf0[0]
    assert first_layer.weight.shape[1] == V27_PER_AGENT_CRITIC_INPUT_DIM, (
        f"v2.7 critic first-layer input dim is {first_layer.weight.shape[1]}, "
        f"expected {V27_PER_AGENT_CRITIC_INPUT_DIM}. "
        "Has the v2.7 class accidentally been built with v2.8 dimensions?"
    )


# ── v2.8: v2.6 legacy load shape guard (carried over from v2.7) ──────────────
def test_v26_legacy_load_shape():
    """v2.6 WrappedVDNCTDESACPolicy must build with the 707-dim obs space."""
    legacy_obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(V26_OBS_DIM,), dtype=np.float32
    )
    act_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
    extractor = FlattenExtractor(legacy_obs_space)

    legacy_actor = _LegacySharedActor(
        N=N,
        observation_space=legacy_obs_space,
        action_space=act_space,
        features_extractor=extractor,
        features_dim=V26_OBS_DIM,
        net_arch=[128, 128],
        activation_fn=torch.nn.ReLU,
    )

    obs_legacy = torch.randn(B, V26_OBS_DIM)
    action = legacy_actor(obs_legacy, deterministic=False)
    assert action.shape == (B, N)

    legacy_critic = _WrappedFactorizedCritic(
        observation_space=legacy_obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=extractor,
        features_dim=V26_OBS_DIM,
        activation_fn=torch.nn.ReLU,
        n_critics=2,
        share_features_extractor=True,
        N=N,
    )

    actions_in = torch.rand(B, N)
    q1, q2 = legacy_critic(obs_legacy, actions_in)
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)

    first_layer = legacy_critic.qf0.local_q_net[0]
    assert first_layer.weight.shape[1] == V26_PER_AGENT_CRITIC_INPUT_DIM, (
        f"v2.6 critic first-layer input dim is {first_layer.weight.shape[1]}, "
        f"expected {V26_PER_AGENT_CRITIC_INPUT_DIM}."
    )
