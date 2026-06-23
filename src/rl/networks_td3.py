# src/rl/networks_td3.py
# -----------------------------------------------------------------------------
# TD3 (Twin Delayed DDPG) network architecture for the irrigation controller.
#
# Design principle: reuse everything that works, change only the actor.
#   * CRITIC: the SAC VdnCritic, unchanged (VDN sum over 130 per-cell Q-nets,
#     twin-Q, LayerNorm). TD3's clipped-double-Q target is computed over the
#     same factorised Q, so the cascade-suppression machinery transfers as-is.
#   * ACTOR: deterministic. Identical feature pipeline to the SAC actor
#     (shared LeakyReLU MLP, 2x-1 re-centering, agent-major reshape), but the
#     (mu, log_std) squashed-Gaussian head is replaced by a single mu head + tanh.
#
# Why TD3 alongside SAC: SAC's entropy term adds a tanh-Jacobian pull toward the
# action-range midpoint (6 mm/day), a soft "action floor" that kept wet-year soil
# moisture above field capacity. TD3 drops the entropy objective (deterministic
# policy) and replaces the smoothing entropy provided with explicit target-policy
# smoothing (Fujimoto et al. 2018) — a cleaner stabiliser for a boundary-located
# optimum. Exploration comes from explicit action noise at collection time.
#
# Actor output: tanh(mu) in [-1, 1]; SB3's TD3Policy scales [-1, 1] -> Box[0, 1],
# then the env maps action -> action * 12 mm/day. tanh=-1 reaches the 0 mm
# boundary that the SAC entropy Jacobian discouraged.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.td3.policies import Actor as TD3Actor, TD3Policy

from src.rl.networks import N_AGENT_FEATURES, VdnCritic, _split_global_dim

TD3_OBS_MARKER = 2.19   # fixed state-dict stamp (checkpoint-compat only)


class DeterministicSharedActor(TD3Actor):
    """Deterministic parameter-shared actor; mirrors the SAC actor's pipeline.

    Per cell: F local features + broadcast global block -> 2x-1 re-centering ->
    shared LeakyReLU MLP -> single mu head -> tanh. The N per-cell squashed
    outputs are concatenated into the (B, N) action.
    """

    def __init__(
        self,
        N: int,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        normalize_images: bool = False,
        **kwargs,
    ):
        net_arch = net_arch if net_arch is not None else [128, 128]
        # Let SB3's TD3Actor set up its plumbing (action scaling, device, ...),
        # then replace its monolithic self.mu with our per-cell trunk + head.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=nn.LeakyReLU,
            normalize_images=normalize_images,
        )

        self.N = N
        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(f"DeterministicSharedActor: action_dim must equal N (={N}), "
                             f"got {action_dim}")
        self._global_dim = _split_global_dim(features_dim, N)
        self._per_agent_input_dim = N_AGENT_FEATURES + self._global_dim

        self.mu = nn.Identity()   # SB3's monolithic head is unused
        latent_layers = create_mlp(
            input_dim=self._per_agent_input_dim,
            output_dim=-1,
            net_arch=net_arch,
            activation_fn=nn.LeakyReLU,
        )
        self.latent_pi = nn.Sequential(*latent_layers)
        last_dim = net_arch[-1] if net_arch else self._per_agent_input_dim
        self.mu_head = nn.Linear(last_dim, 1)
        self.register_buffer("obs_norm_marker", torch.tensor([TD3_OBS_MARKER]))

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Agent-major reshape + 2x-1 re-center, identical to the SAC actor."""
        B = features.shape[0]
        N, F = self.N, N_AGENT_FEATURES
        per_agent = features[:, : F * N].reshape(B, N, F)
        global_expanded = features[:, F * N:].unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        combined = 2.0 * combined - 1.0
        return combined.reshape(B * N, self._per_agent_input_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic squashed action in [-1, 1], shape (B, N)."""
        features = self.extract_features(obs, self.features_extractor)
        B = features.shape[0]
        latent = self.latent_pi(self._per_agent_features(features))
        pre_tanh = self.mu_head(latent).reshape(B, self.N)
        return torch.tanh(pre_tanh)


class Td3VdnPolicy(TD3Policy):
    """TD3 policy: deterministic shared actor + the SAC VDN LayerNorm critic."""

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DeterministicSharedActor:
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return DeterministicSharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> VdnCritic:
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return VdnCritic(**kw).to(self.device)


def make_td3_policy_kwargs(N: int = 130, actor_hidden=(128, 128), critic_hidden=(256, 256)):
    """policy_kwargs for SB3 TD3 with the CTDE VDN architecture.

    The actor trunk forces LeakyReLU internally; ``activation_fn`` controls the
    critic (ReLU). Use with ``policy_class=Td3VdnPolicy``.
    """
    return {
        "net_arch": {"pi": list(actor_hidden), "qf": list(critic_hidden)},
        "activation_fn": nn.ReLU,
    }
