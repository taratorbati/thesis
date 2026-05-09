# =============================================================================
# src/rl/networks.py
# Custom policy networks for Stable-Baselines3 SAC implementing
# Centralized Training with Decentralized Execution (CTDE).
#
# Architecture:
#   Actor: shared-parameter MLP applied to each agent's local features.
#     For each of the N=130 agents, the actor receives a 61-dim input:
#       - 5 per-agent features (x1_norm, x5_norm, x4_norm, x3, elevation)
#       - 56 global features broadcast to all agents:
#           8 scalars (day_frac, budget_frac, burn_rate, rain_today,
#                      ETc_today, h2_today, h7_today, g_base_today)
#           48 per-day forecast values (6 variables × 8 forecast days:
#                      rain, ETc, radiation, h2, h7, g_base)
#     Same MLP weights are applied across all agents → ~10× fewer parameters
#     than a monolithic 706→130 actor and directly imposes the inductive
#     bias that all agents share the same control policy modulo their local
#     state and shared global context.
#
#   Critic: standard centralized SAC twin Q-network.
#     Sees the full 706-dim joint state and 130-dim joint action
#     (836-dim input total) through a standard MLP.
#     The critic does NOT use weight sharing — it must perform credit
#     assignment across the joint action space, which requires access
#     to the full joint state-action.
#
# The CTDE separation: the actor uses only information available at
# deployment (per-agent local + shared global context, including the full
# per-day forecast that the MPC also receives), while the critic uses the
# full joint state during training. After training, the critic is discarded;
# the deployed policy is the actor alone.
#
# Observation layout (706 dims at default forecast_horizon=8):
#   [  0: 650] Per-agent block: 5 features × 130 agents
#              (x1_norm, x5_norm, x4_norm, x3, elevation — each N-dim)
#   [650: 658] Global scalars (8): day_frac, budget_frac, burn_rate,
#              rain_today, ETc_today, h2_today, h7_today, g_base_today
#   [658: 706] Per-day forecasts (48): rain[H], ETc[H], rad[H],
#              h2[H], h7[H], g_base[H], where H=8
#   Total: 650 + 8 + 48 = 706
#
# v2.2 update: N_GLOBAL_SCALARS updated from 10 to 56 (8 scalars + 6×8
# per-day forecast arrays) to match the gym_env.py Fix C observation layout.
# PER_AGENT_INPUT_DIM updated from 15 to 61 accordingly. The slicing logic
# in _per_agent_features is unchanged — it already used features[:, 5*N:]
# which correctly captures everything after the per-agent block regardless
# of the global block width.
#
# SB3 integration:
#   We subclass SACPolicy and override make_actor() to instantiate a
#   custom SharedActor class. The standard SB3 ContinuousCritic is used
#   for the critic with no modifications.
#
# References:
#   Haarnoja et al. (2018, 2019) — SAC algorithm
#   Amato (2024) — CTDE in cooperative multi-agent RL
#   Gupta et al. (2017) — shared-parameter actors in MARL
# =============================================================================

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)


# ── Architecture constants ──────────────────────────────────────────────────
# These must match gym_env.py exactly. If you change the observation layout
# in gym_env.py, update these constants to match.

# Per-agent features in the observation (first 5*N dims of obs)
N_AGENT_FEATURES = 5    # x1_norm, x5_norm, x4_norm, x3, elevation

# Global features broadcast to all agents (everything after the 5*N block):
#   8 scalars: day_frac, budget_frac, burn_rate, rain_today, ETc_today,
#              h2_today, h7_today, g_base_today
#   48 forecast: rain[8], ETc[8], radiation[8], h2[8], h7[8], g_base[8]
N_GLOBAL_SCALARS = 56   # 8 scalars + 6 × 8 forecast days

# Input dim to the per-agent shared MLP
PER_AGENT_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_SCALARS  # = 61

# Numerical stability bounds for the policy log-std
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SharedActor(Actor):
    """SAC actor with parameter sharing across N spatial agents.

    Each of the N agents shares the same MLP weights. The MLP takes a
    61-dim input (5 per-agent features + 56 global context dims broadcast
    to all agents) and outputs a 1-dim (mean, log_std) action distribution
    parameterization per agent. The N independent per-agent actions are
    assembled into the joint action vector.

    The 56 global dims include both the 8 scalar state features and the
    full 48-dim per-day forecast (6 variables × 8 days), matching exactly
    the forecast structure that the MPC receives via PerfectForecast. This
    ensures the SAC actor has access to the same temporal forecast
    information as the MPC, enabling a fair policy-equivalence comparison.

    This subclasses SB3's SACPolicy.Actor and overrides the network
    construction and forward methods. The interface to the surrounding
    SAC algorithm (action_log_prob, _predict, etc.) is preserved.

    Parameters
    ----------
    N : int
        Number of spatial agents (130 for the Gilan field).
    observation_space : gymnasium.spaces.Box
        Must have shape (706,) at default forecast_horizon=8.
    action_space : gymnasium.spaces.Box
        Must have shape (N,).
    features_extractor : BaseFeaturesExtractor
        Standard SB3 FlattenExtractor (no custom extraction needed since
        the obs is already flat; the SharedActor does the internal reshape).
    features_dim : int
        Must equal 5*N + 56 = 706.
    net_arch : list of int
        Hidden layer sizes for the shared per-agent MLP. Default [128, 128].
    activation_fn : nn.Module class
        Activation function. Default nn.ReLU.
    """

    def __init__(
        self,
        N: int,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = False,
        **kwargs,
    ):
        # Pass to parent BEFORE building custom layers, since Actor.__init__
        # sets up the features extractor. We defer the standard network
        # construction by calling with use_sde=False (handled by parent).
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch if net_arch is not None else [128, 128],
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        self.N = N

        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(
                f"SharedActor expects action_dim == N (={N}), got {action_dim}"
            )

        # Verify the observation layout matches gym_env.py.
        # features_dim must equal 5*N + N_GLOBAL_SCALARS = 5*N + 56 = 706
        # at default forecast_horizon=8.
        expected_obs_dim = N_AGENT_FEATURES * N + N_GLOBAL_SCALARS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"SharedActor expects features_dim == {expected_obs_dim} "
                f"(5*{N} + {N_GLOBAL_SCALARS}), got {features_dim}. "
                f"Ensure the gym_env observation layout matches: "
                f"5*N per-agent features + 8 global scalars + "
                f"6*forecast_horizon forecast dims."
            )

        # Build the per-agent MLP: input PER_AGENT_INPUT_DIM, hidden net_arch,
        # output a shared latent representation. Two output heads (mean and
        # log_std) follow the latent layer.
        net_arch_list = net_arch if net_arch is not None else [128, 128]
        latent_pi_net = create_mlp(
            input_dim=PER_AGENT_INPUT_DIM,
            output_dim=-1,                        # no final layer; we add heads
            net_arch=net_arch_list,
            activation_fn=activation_fn,
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)

        last_layer_dim = net_arch_list[-1] if len(net_arch_list) > 0 else PER_AGENT_INPUT_DIM
        self.mu = nn.Linear(last_layer_dim, 1)         # per-agent action mean
        self.log_std = nn.Linear(last_layer_dim, 1)    # per-agent log-std

        # Squashed-Gaussian distribution over the joint N-dim action.
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def get_std(self) -> torch.Tensor:
        """Helper for SDE compatibility — not used in non-SDE mode."""
        return torch.zeros(self.N)

    def reset_noise(self, batch_size: int = 1) -> None:
        """SDE-only — no-op for SharedActor."""
        return

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape flat observation → (batch*N, PER_AGENT_INPUT_DIM).

        Splits the flat observation into:
          - per-agent block: first 5*N dims → (batch, N, 5)
          - global block:    remaining 56 dims → broadcast to (batch, N, 56)
        Concatenates to (batch, N, 61) and reshapes to (batch*N, 61).

        The global block contains both the 8 scalar features and the 48
        per-day forecast dims. Broadcasting these to all N agents gives
        each agent identical access to the shared temporal forecast — the
        same information the MPC receives per-day over its horizon.
        """
        batch_size = features.shape[0]
        N = self.N

        # Per-agent block: (batch, 5*N) → (batch, N, 5)
        per_agent_flat = features[:, :N_AGENT_FEATURES * N]
        per_agent = per_agent_flat.reshape(batch_size, N_AGENT_FEATURES, N)
        per_agent = per_agent.permute(0, 2, 1)  # (batch, N, 5)

        # Global block: (batch, 56) → (batch, N, 56)
        # This takes everything after the 5*N per-agent block, so it
        # captures the 8 scalars + 48 forecast dims automatically.
        global_block = features[:, N_AGENT_FEATURES * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: (batch, N, 61) → (batch*N, 61)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        return combined.reshape(batch_size * N, PER_AGENT_INPUT_DIM)

    def get_action_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute action mean and log_std from the observation.

        Returns
        -------
        mean_actions : Tensor of shape (batch, N)
        log_std : Tensor of shape (batch, N)
        kwargs : dict (empty for non-SDE)
        """
        features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]

        # Reshape to (batch*N, 61) and run the shared MLP
        per_agent_input = self._per_agent_features(features)
        latent_pi = self.latent_pi(per_agent_input)        # (batch*N, hidden)

        # Per-agent mean and log_std
        mean_per_agent = self.mu(latent_pi)                # (batch*N, 1)
        log_std_per_agent = self.log_std(latent_pi)        # (batch*N, 1)

        # Reshape back to (batch, N)
        mean_actions = mean_per_agent.reshape(batch_size, self.N)
        log_std = log_std_per_agent.reshape(batch_size, self.N)

        # Clip to numerical stability range
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Optionally clip the mean before tanh squashing
        if self.clip_mean > 0:
            mean_actions = torch.clamp(
                mean_actions, -self.clip_mean, self.clip_mean
            )

        return mean_actions, log_std, {}

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample an action from the policy."""
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and return its log probability."""
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(
            mean_actions, log_std, **kwargs
        )

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action from observation. Used by BasePolicy.predict()."""
        return self(observation, deterministic)


class CTDESACPolicy(SACPolicy):
    """SAC policy with a shared-parameter actor and a centralized critic.

    Overrides make_actor() to use the SharedActor class. The critic is
    constructed as in standard SB3 SAC (centralized over the joint
    state-action space).

    The number of spatial agents N is read from the action_space dim.
    """

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SharedActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )

        # Pass N explicitly so the actor knows the agent dimension
        N = get_action_dim(self.action_space)
        actor_kwargs['N'] = N

        return SharedActor(**actor_kwargs).to(self.device)


# ── Convenience builder ──────────────────────────────────────────────────────

def make_sac_policy_kwargs(
    N: int = 130,
    actor_hidden: Tuple[int, ...] = (128, 128),
    critic_hidden: Tuple[int, ...] = (256, 256),
) -> Dict[str, Any]:
    """Build the policy_kwargs dict for SB3 SAC with the CTDE architecture.

    Use together with policy_class=CTDESACPolicy in the SAC constructor:

        from stable_baselines3 import SAC
        from src.rl.networks import CTDESACPolicy, make_sac_policy_kwargs

        model = SAC(
            policy=CTDESACPolicy,
            env=env,
            policy_kwargs=make_sac_policy_kwargs(N=130),
            ...
        )

    Parameters
    ----------
    N : int
        Number of spatial agents (130 for the Gilan field).
    actor_hidden : tuple of int
        Hidden layer sizes for the per-agent shared actor MLP.
        Default (128, 128). Each layer takes 61-dim input at the first layer.
    critic_hidden : tuple of int
        Hidden layer sizes for the centralized critic MLP.
        Default (256, 256). Critic input is 706 + 130 = 836 dims.

    Returns
    -------
    policy_kwargs : dict
    """
    return {
        'net_arch': {
            'pi': list(actor_hidden),
            'qf': list(critic_hidden),
        },
        'activation_fn': nn.ReLU,
        # The default features extractor (FlattenExtractor) is correct for
        # both the actor (which internally reshapes the flat features) and
        # the critic (which uses the full flat observation).
    }
