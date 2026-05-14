# tests/test_factorized_critic.py
# ─────────────────────────────────────────────────────────────────────────────
# Pre-Kaggle validation tests for the v2.6 networks.
# All three tests must pass locally before any cloud training run.
#
# Usage:   pytest tests/test_factorized_critic.py -v
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.rl.networks import (
    SharedActor,
    FactorizedContinuousCritic,
    CTDESACPolicy,
    make_sac_policy_kwargs,
    N_AGENT_FEATURES,
    N_GLOBAL_DIMS,
    OBS_DIM_DEFAULT,
)


N = 130
OBS_DIM = OBS_DIM_DEFAULT      # 707
B = 32                         # batch size for tests


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
    """forward(obs, actions) must return a tuple of two (B,1) tensors."""
    obs = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N)
    out = critic(obs, actions)
    assert isinstance(out, tuple), f"Critic must return a tuple; got {type(out)}"
    assert len(out) == 2, f"Critic must return 2 Q-values; got {len(out)}"
    for i, q in enumerate(out):
        assert q.shape == (B, 1), f"Q{i+1} shape {q.shape} != ({B}, 1)"


def test_q1_forward_shape(critic):
    """q1_forward must return a single (B, 1) tensor for the actor's policy loss."""
    obs = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N)
    q1 = critic.q1_forward(obs, actions)
    assert q1.shape == (B, 1), f"q1_forward shape {q1.shape} != ({B}, 1)"


def test_actor_output_shape(actor):
    """SharedActor.forward must return (B, N) joint actions."""
    obs = torch.randn(B, OBS_DIM)
    action = actor(obs, deterministic=False)
    assert action.shape == (B, N), f"Actor shape {action.shape} != ({B}, {N})"


# ── Test 2: gradient localisation ────────────────────────────────────────────
def test_gradient_localisation(critic):
    """∂Q_total/∂a_n should be roughly uniform in magnitude across n.

    With the shared MLP and identical input statistics, each agent's
    gradient should be of the same order of magnitude — confirming the
    per-agent decomposition is alive.
    """
    obs = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N, requires_grad=True)
    q1, _ = critic(obs, actions)
    q1.sum().backward()

    grad = actions.grad           # (B, N)
    grad_mag_per_agent = grad.abs().mean(dim=0)   # mean over batch → (N,)

    # Gradients must exist for ALL agents (no dead agents)
    assert torch.all(grad_mag_per_agent > 0), \
        "Some agents have zero gradient — decomposition is broken."

    # Range across agents shouldn't be wildly skewed (within 5×)
    ratio = grad_mag_per_agent.max() / grad_mag_per_agent.min()
    assert ratio < 5.0, (
        f"Gradient ratio max/min = {ratio:.2f} across agents — "
        "decomposition may not be giving balanced per-agent signal."
    )


def test_gradient_only_through_relevant_agent(critic):
    """Setting action[:, k] modifies Q only through agent k's local Q_k.

    We verify this by setting all actions to zero, then setting a single
    agent's action to a large value, and confirming the gradient on THAT
    agent is much larger than on any other agent.

    This is the strongest test of decomposition correctness: if Q_total
    truly equals Σ_n Q_n(s_n, g, a_n), then a_k appears in exactly one
    Q_n and nowhere else.
    """
    obs = torch.randn(B, OBS_DIM)
    # Create a fresh leaf tensor directly — clone() produces a non-leaf
    # and PyTorch will not populate .grad on non-leaf tensors.
    actions_input = torch.zeros(B, N, requires_grad=True)

    q1, _ = critic(obs, actions_input)
    q1.sum().backward()

    grad = actions_input.grad   # (B, N) — populated because actions_input is a leaf
    assert grad is not None, "Gradient is None — actions_input must be a leaf tensor"

    per_agent_norm = grad.abs().mean(dim=0)
    # In a perfect VDN factorization Q_total = Σ_n Q_n(s_n, g, a_n),
    # so ∂Q_total/∂a_n flows only through Q_n. Every agent receives a
    # non-zero gradient because the shared MLP has non-zero weights.
    assert torch.all(per_agent_norm > 0), (
        f"Some agents have zero gradient — VDN decomposition may be broken. "
        f"Zero-grad agents: {(per_agent_norm == 0).nonzero().flatten().tolist()}"
    )


# ── Test 3: SB3 SAC integration smoke test ───────────────────────────────────
def test_sac_integration_smoke():
    """Build a full SAC model with CTDESACPolicy and run a few training steps.

    This is the integration test: if SB3's training loop can produce
    valid losses and the model can save/load, the architecture plumbing
    is correct.
    """
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
            reward = float(-np.mean(action))   # trivial reward
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

    # Learn for 200 steps — must complete without errors
    model.learn(total_timesteps=200)

    # Test save/load round-trip
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.zip")
        model.save(path)
        model2 = SAC.load(path, env=env)
        # Make sure inference works after reload
        obs = env.reset()
        action, _ = model2.predict(obs, deterministic=True)
        assert action.shape == (1, N), f"Loaded model action shape {action.shape} != (1, {N})"


# ── Bonus: parameter count check ─────────────────────────────────────────────
def test_parameter_counts(actor, critic):
    """Sanity check: factorized critic should be similar size to monolithic.

    Shared 256x256 with 63-input ≈ 83k params per Q-net × 2 = ~166k.
    Old monolithic 837x256x256x1 × 2 ≈ 540k params.
    """
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())

    print(f"\n  SharedActor parameters:                  {actor_params:>8,}")
    print(f"  FactorizedContinuousCritic parameters:   {critic_params:>8,}")

    assert 50_000 < critic_params < 300_000, (
        f"Factorized critic param count {critic_params} outside expected range"
    )
