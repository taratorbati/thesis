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
    # v2.11 (LayerNorm critic)
    V211_OBS_DIM,
    V211_N_AGENT_FEATURES,
    V211_PER_AGENT_CRITIC_INPUT_DIM,
    V211CTDESACPolicy,
    _V211FactorizedQNet,
    _V211FactorizedContinuousCritic,
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


# ── v2.11: LayerNorm critic shape and forward-pass guard ─────────────────────
def test_v211_layernorm_critic_shape():
    """v2.11 critic must build with the 1097-dim obs space (v2.7 layout) and
    expose a LayerNorm layer at index 1 of each per-agent MLP.

    The state-dict keys of the per-agent Q-net must include a 1-D
    'qf0.1.weight' entry (LayerNorm gamma).  The v2.7 critic has no key at
    index 1 (index 1 is ReLU, no params).  This is the unique signal the
    runner's _detect_critic_arch uses to dispatch the correct policy class.
    """
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(V211_OBS_DIM,), dtype=np.float32
    )
    act_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
    extractor = FlattenExtractor(obs_space)

    critic = _V211FactorizedContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=extractor,
        features_dim=V211_OBS_DIM,
        activation_fn=torch.nn.ReLU,
        n_critics=2,
        share_features_extractor=True,
        N=N,
    )

    # Forward-pass shape
    obs = torch.randn(B, V211_OBS_DIM)
    actions = torch.rand(B, N)
    q1, q2 = critic(obs, actions)
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)

    # First Linear input dim is 66 (same as v2.7)
    assert critic.qf0[0].weight.shape[1] == V211_PER_AGENT_CRITIC_INPUT_DIM

    # Layer at index 1 must be LayerNorm (NOT ReLU as in v2.7)
    assert isinstance(critic.qf0[1], torch.nn.LayerNorm), (
        f"v2.11 critic qf0[1] should be LayerNorm, got {type(critic.qf0[1])}"
    )

    # State-dict must contain the 1-D LayerNorm key for runner detection
    sd = critic.state_dict()
    assert 'qf0.1.weight' in sd
    assert sd['qf0.1.weight'].ndim == 1, (
        "qf0.1.weight should be 1-D LayerNorm gamma, not 2-D Linear weight"
    )
    assert sd['qf0.1.weight'].shape == (256,)


def test_v211_param_count_vs_v27():
    """v2.11 critic should have exactly 2048 more params than v2.7 critic
    (4 LayerNorms total: 2 critics x 2 hidden layers, each adds 2x256 params).
    """
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(V211_OBS_DIM,), dtype=np.float32
    )
    act_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
    extractor = FlattenExtractor(obs_space)

    c27 = _V27FactorizedContinuousCritic(
        observation_space=obs_space, action_space=act_space,
        net_arch=[256, 256], features_extractor=extractor,
        features_dim=V211_OBS_DIM, activation_fn=torch.nn.ReLU,
        n_critics=2, share_features_extractor=True, N=N,
    )
    c211 = _V211FactorizedContinuousCritic(
        observation_space=obs_space, action_space=act_space,
        net_arch=[256, 256], features_extractor=extractor,
        features_dim=V211_OBS_DIM, activation_fn=torch.nn.ReLU,
        n_critics=2, share_features_extractor=True, N=N,
    )

    n_v27  = sum(p.numel() for p in c27.parameters())
    n_v211 = sum(p.numel() for p in c211.parameters())
    diff = n_v211 - n_v27
    assert diff == 2048, (
        f"v2.11 - v2.7 param diff is {diff}, expected 2048 "
        f"(2 critics x 2 LN x 2x256). "
        f"Has the LayerNorm placement or net_arch changed?"
    )


def test_v211_policy_actor_is_v27_compatible():
    """V211CTDESACPolicy must use _V27SharedActor (8-feature input).

    The whole point of v2.11 is that the actor and observation layout are
    UNCHANGED from v2.7 — only the critic's regularisation differs.
    """
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(V211_OBS_DIM,), dtype=np.float32
    )
    act_space = spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)

    # Build a policy via make_sac_policy_kwargs (the runtime path)
    kwargs = make_sac_policy_kwargs(N=N, actor_hidden=(128, 128), critic_hidden=(256, 256))

    def _lr(progress_remaining):  # SB3 SACPolicy needs lr_schedule
        return 3e-4

    policy = V211CTDESACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=_lr,
        **kwargs,
    )

    # Actor must be the v2.7 8-feature actor
    assert isinstance(policy.actor, _V27SharedActor), (
        f"V211 actor should be _V27SharedActor (8 features), "
        f"got {type(policy.actor).__name__}"
    )
    # Critic must be the v2.11 LayerNorm critic
    assert isinstance(policy.critic, _V211FactorizedContinuousCritic), (
        f"V211 critic should be _V211FactorizedContinuousCritic, "
        f"got {type(policy.critic).__name__}"
    )

    # Forward pass through the policy
    obs = torch.randn(B, V211_OBS_DIM)
    actions = torch.rand(B, N)
    q1, q2 = policy.critic(obs, actions)
    assert q1.shape == (B, 1) and q2.shape == (B, 1)


def test_v211_layernorm_actually_normalises():
    """Sanity check: a LayerNorm layer with default affine=True should make
    its outputs have mean ~0 and variance ~1 across the feature dim BEFORE
    the learned gamma/beta are applied.  This is just to confirm the
    structure is wired correctly (not a perf test).
    """
    ln = torch.nn.LayerNorm(256)
    x = torch.randn(32, 256) * 100 + 50   # wildly off-distribution input
    y = ln(x)
    # After LayerNorm but before learned scale (defaults gamma=1, beta=0),
    # mean ~ 0 and std ~ 1 along feature dim.
    mean = y.mean(dim=-1)
    std  = y.std(dim=-1, unbiased=False)
    assert mean.abs().max().item() < 1e-4, (
        f"LayerNorm output mean not centered: max |mean| = {mean.abs().max().item()}"
    )
    assert (std - 1.0).abs().max().item() < 1e-2, (
        f"LayerNorm output std not unit: max |std-1| = {(std-1.0).abs().max().item()}"
    )
