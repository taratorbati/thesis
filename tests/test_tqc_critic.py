# tests/test_tqc_critic.py  v2.10.0
# -----------------------------------------------------------------------------
# Pre-cloud validation tests for the v2.10 TQC networks.
#
# These tests catch shape and wiring bugs in the VDN-per-quantile critic
# before a 2-hour training run.  Pattern follows tests/test_factorized_critic.py.
#
# Usage:  pytest tests/test_tqc_critic.py -v
# -----------------------------------------------------------------------------

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor

from src.rl.networks import (
    V27_OBS_DIM,
    V27_N_AGENT_FEATURES,
    V27_PER_AGENT_CRITIC_INPUT_DIM,
)
from src.rl.networks_tqc import (
    TQC_N_QUANTILES_DEFAULT,
    TQC_N_CRITICS_DEFAULT,
    _V27FactorizedTQCQNet,
    _V27FactorizedTQCCritic,
    V27TQCPolicy,
    make_tqc_policy_kwargs,
)


N            = 130
OBS_DIM      = V27_OBS_DIM          # 1097
N_QUANTILES  = TQC_N_QUANTILES_DEFAULT
N_CRITICS    = TQC_N_CRITICS_DEFAULT
B            = 8                     # batch size for tests


@pytest.fixture
def spaces_fixture():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.Box(low=0.0,     high=1.0,    shape=(N,),       dtype=np.float32)
    return obs_space, act_space


# -----------------------------------------------------------------------------
# Test 1 - _V27FactorizedTQCQNet forward shape.
# -----------------------------------------------------------------------------
def test_qnet_output_shape():
    qnet = _V27FactorizedTQCQNet(
        N=N,
        net_arch=[256, 256],
        n_quantiles=N_QUANTILES,
    )
    obs     = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N)
    out = qnet(obs, actions)
    assert out.shape == (B, N_QUANTILES), (
        f"Expected (B={B}, n_quantiles={N_QUANTILES}), got {tuple(out.shape)}"
    )


# -----------------------------------------------------------------------------
# Test 2 - _V27FactorizedTQCCritic forward shape - the contract TQC relies on.
# -----------------------------------------------------------------------------
def test_critic_output_shape(spaces_fixture):
    obs_space, act_space = spaces_fixture
    fx = FlattenExtractor(obs_space)
    critic = _V27FactorizedTQCCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=fx,
        features_dim=OBS_DIM,
        n_critics=N_CRITICS,
        n_quantiles=N_QUANTILES,
        share_features_extractor=True,
        N=N,
    )
    obs     = torch.randn(B, OBS_DIM)
    actions = torch.rand(B, N)
    out = critic(obs, actions)
    assert out.shape == (B, N_CRITICS, N_QUANTILES), (
        f"Expected (B={B}, n_critics={N_CRITICS}, n_quantiles={N_QUANTILES}), "
        f"got {tuple(out.shape)}"
    )


# -----------------------------------------------------------------------------
# Test 3 - quantiles_total attribute (TQC.train reads this).
# -----------------------------------------------------------------------------
def test_quantiles_total_attribute(spaces_fixture):
    obs_space, act_space = spaces_fixture
    fx = FlattenExtractor(obs_space)
    critic = _V27FactorizedTQCCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[256, 256],
        features_extractor=fx,
        features_dim=OBS_DIM,
        n_critics=N_CRITICS,
        n_quantiles=N_QUANTILES,
        N=N,
    )
    assert critic.quantiles_total == N_CRITICS * N_QUANTILES
    assert critic.n_critics       == N_CRITICS
    assert critic.n_quantiles     == N_QUANTILES
    assert len(critic.q_networks) == N_CRITICS


# -----------------------------------------------------------------------------
# Test 4 - VDN summation: changing one agent's features changes the output
# only by that agent's contribution.
# -----------------------------------------------------------------------------
def test_vdn_summation_localised():
    """The joint quantile output must equal the elementwise sum of per-agent
    local quantile outputs.  Verifies the VDN aggregation in forward().
    """
    qnet = _V27FactorizedTQCQNet(
        N=N,
        net_arch=[32, 32],         # tiny for the test
        n_quantiles=N_QUANTILES,
    )
    qnet.eval()

    obs     = torch.randn(1, OBS_DIM)
    actions = torch.rand(1, N)

    F = V27_N_AGENT_FEATURES
    # Manually compute per-agent outputs and sum.
    local_obs       = obs[:, : F * N].reshape(1, N, F)
    global_block    = obs[:, F * N:]                            # (1, 57)
    global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
    local_actions   = actions.reshape(1, N, 1)
    local_inputs    = torch.cat([local_obs, global_expanded, local_actions], dim=-1)
    local_inputs_flat = local_inputs.reshape(N, V27_PER_AGENT_CRITIC_INPUT_DIM)
    manual_local = qnet.local_q_net(local_inputs_flat)          # (N, n_q)
    manual_sum   = manual_local.sum(dim=0, keepdim=True)        # (1, n_q)

    forward_out = qnet(obs, actions)                            # (1, n_q)
    assert torch.allclose(forward_out, manual_sum, atol=1e-5), (
        "VDN summation does not match manual per-agent sum"
    )


# -----------------------------------------------------------------------------
# Test 5 - make_tqc_policy_kwargs produces a valid kwargs dict.
# -----------------------------------------------------------------------------
def test_policy_kwargs_shape():
    kw = make_tqc_policy_kwargs(
        N=N,
        actor_hidden=(128, 128),
        critic_hidden=(256, 256),
        n_quantiles=N_QUANTILES,
        n_critics=N_CRITICS,
        share_features_extractor=True,
    )
    assert "net_arch" in kw
    assert kw["net_arch"]["pi"] == [128, 128]
    assert kw["net_arch"]["qf"] == [256, 256]
    assert kw["n_quantiles"] == N_QUANTILES
    assert kw["n_critics"]   == N_CRITICS
    assert kw["share_features_extractor"] is True


# -----------------------------------------------------------------------------
# Test 6 - End-to-end policy instantiation through V27TQCPolicy (no learn).
# -----------------------------------------------------------------------------
def test_policy_instantiation(spaces_fixture):
    obs_space, act_space = spaces_fixture

    def lr_sched(progress):
        return 3e-4 - (3e-4 - 5e-5) * (1.0 - progress)

    kw = make_tqc_policy_kwargs(
        N=N,
        actor_hidden=(128, 128),
        critic_hidden=(256, 256),
        n_quantiles=N_QUANTILES,
        n_critics=N_CRITICS,
    )

    policy = V27TQCPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lr_sched,
        **kw,
    )
    # Smoke-test forward through actor and critic.
    obs_batch = torch.randn(B, OBS_DIM)
    with torch.no_grad():
        actions, log_probs = policy.actor.action_log_prob(obs_batch)
    assert actions.shape == (B, N)
    assert log_probs.shape == (B,)

    with torch.no_grad():
        quantiles = policy.critic(obs_batch, actions)
    assert quantiles.shape == (B, N_CRITICS, N_QUANTILES)


# -----------------------------------------------------------------------------
# Test 7 - Truncation arithmetic invariant.
# -----------------------------------------------------------------------------
def test_truncation_arithmetic():
    """Check that quantiles_total - top*n_critics matches what TQC.train
    computes for n_target_quantiles.
    """
    top_quantiles_to_drop = 5
    n_target = N_CRITICS * N_QUANTILES - top_quantiles_to_drop * N_CRITICS
    assert n_target == 50 - 10
    assert n_target == 40
