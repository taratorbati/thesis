# =============================================================================
# src/rl/networks.py
# Custom policy networks for Stable-Baselines3 SAC implementing CTDE.
#
# Observation layout (707 dims at default forecast_horizon=8):
#   [  0: 650] Per-agent (5 × 130): x1_norm, x5_norm, x4_norm, x3, elevation
#   [650: 659] Scalars (9): day_frac, budget_frac, budget_total_norm,
#              burn_rate, rain_today, ETc_today, h2_today, h7_today, g_base_today
#   [659: 707] Per-day forecasts (48): rain[H], ETc[H], rad[H], h2[H], h7[H], g[H]
#
# Actor per-agent input: 62 dims (5 per-agent + 57 global = 9 scalars + 48 forecast).
# Critic input: 707 obs + 130 action = 837 dims (centralized, standard MLP).
#
# v2.3: N_GLOBAL_SCALARS 56→57, PER_AGENT_INPUT_DIM 61→62 (added budget_total_norm).
# =============================================================================

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type

from gymnasium import spaces
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

N_AGENT_FEATURES    = 5    # x1_norm, x5_norm, x4_norm, x3, elevation
N_GLOBAL_SCALARS    = 57   # 9 scalars + 48 forecast dims (6 vars × 8 days)
PER_AGENT_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_SCALARS   # 62

LOG_STD_MIN = -20.0
LOG_STD_MAX =  2.0


class SharedActor(Actor):
    """SAC actor with parameter sharing across N spatial agents."""

    def __init__(self, N, observation_space, action_space,
                 features_extractor, features_dim,
                 net_arch=None, activation_fn=nn.ReLU,
                 normalize_images=False, **kwargs):
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

        expected = N_AGENT_FEATURES * N + N_GLOBAL_SCALARS
        if features_dim != expected:
            raise ValueError(
                f"SharedActor expects features_dim == {expected} "
                f"(5*{N} + {N_GLOBAL_SCALARS}), got {features_dim}. "
                f"Observation layout: 5*N per-agent + 9 scalars + "
                f"6*forecast_horizon forecast dims (707 at default H=8)."
            )

        net_arch_list = net_arch if net_arch is not None else [128, 128]
        self.latent_pi = nn.Sequential(
            *create_mlp(PER_AGENT_INPUT_DIM, -1, net_arch_list, activation_fn)
        )
        last_dim = net_arch_list[-1] if net_arch_list else PER_AGENT_INPUT_DIM
        self.mu      = nn.Linear(last_dim, 1)
        self.log_std = nn.Linear(last_dim, 1)
        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(action_space))

    def _per_agent_features(self, features):
        batch = features.shape[0]
        N = self.N
        per_agent = features[:, :N_AGENT_FEATURES * N]
        per_agent = per_agent.reshape(batch, N_AGENT_FEATURES, N).permute(0, 2, 1)
        global_block = features[:, N_AGENT_FEATURES * N:].unsqueeze(1).expand(-1, N, -1)
        return torch.cat([per_agent, global_block], dim=-1).reshape(batch * N, PER_AGENT_INPUT_DIM)

    def get_action_dist_params(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        batch = features.shape[0]
        latent = self.latent_pi(self._per_agent_features(features))
        mean   = self.mu(latent).reshape(batch, self.N)
        ls     = torch.clamp(self.log_std(latent).reshape(batch, self.N), LOG_STD_MIN, LOG_STD_MAX)
        if self.clip_mean > 0:
            mean = torch.clamp(mean, -self.clip_mean, self.clip_mean)
        return mean, ls, {}

    def forward(self, obs, deterministic=False):
        mean, ls, kw = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean, ls, deterministic=deterministic, **kw)

    def action_log_prob(self, obs):
        mean, ls, kw = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean, ls, **kw)

    def _predict(self, observation, deterministic=False):
        return self(observation, deterministic)

    def get_std(self):
        return torch.zeros(self.N)

    def reset_noise(self, batch_size=1):
        return


class CTDESACPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw['N'] = get_action_dim(self.action_space)
        return SharedActor(**kw).to(self.device)


def make_sac_policy_kwargs(N=130, actor_hidden=(128, 128), critic_hidden=(256, 256)):
    """Build policy_kwargs for SB3 SAC with the CTDE architecture."""
    return {
        'net_arch': {'pi': list(actor_hidden), 'qf': list(critic_hidden)},
        'activation_fn': nn.ReLU,
    }
