# src/rl/networks.py
# -----------------------------------------------------------------------------
# CTDE network architecture for the 130-cell irrigation controller (SAC).
#
# The agent observes a flat vector laid out agent-major:
#     [ per-cell block : N x F features ] [ global block : G features ]
# where F = 8 per-cell features and G is the shared global/forecast context
# (52 in the default layout, 57 in the legacy layout that also repeats today's
# weather — see gym_env.py). Both the actor and the critic split the vector,
# broadcast the global block to every cell, and apply a single parameter-shared
# MLP across all cells. G is inferred from the observation width, so one class
# handles either layout (and loads checkpoints trained under either).
#
# ACTOR — SharedActor:
#   per-cell input -> (2x-1 re-centering) -> shared LeakyReLU MLP -> (mu, log_std)
#   The N per-cell squashed-Gaussian heads form the joint N-dim action.
#   Re-centering maps the non-negative inputs to ~[-1, 1] so the entropy/overshoot
#   downward pressure does not drive the first-layer units permanently negative.
#
# CRITIC — VdnCritic (Value Decomposition Network):
#   Q_total = sum_n Q_local(s_n, g, a_n), twin-Q with clipped-double-Q target.
#   Each per-cell Q-net has LayerNorm after every hidden Linear, which suppresses
#   the Q-divergence cascade (Yue et al. NeurIPS 2023; Nauman et al. RLC 2024).
#
# The actor uses LeakyReLU (units that drift negative still pass gradient); the
# critic uses ReLU + LayerNorm. The ``obs_norm_marker`` buffer is a fixed stamp
# kept for checkpoint state-dict compatibility.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import BaseFeaturesExtractor, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.sac.policies import Actor, SACPolicy

# ── architecture constants ────────────────────────────────────────────────────
N_AGENTS_DEFAULT = 130
N_AGENT_FEATURES = 8     # per cell: x1, x5, x4, x3, elev, Nr, Nr_internal, n_upstream
OBS_MARKER = 2.16        # fixed state-dict stamp (checkpoint-compat only)

# Numerical bounds for the policy log-std.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _split_global_dim(features_dim: int, n_agents: int) -> int:
    """Infer the global-block width G from the flat observation width."""
    g = features_dim - N_AGENT_FEATURES * n_agents
    if g < 0:
        raise ValueError(
            f"features_dim={features_dim} too small for {N_AGENT_FEATURES} "
            f"features x {n_agents} agents."
        )
    return g


# ═════════════════════════════════════════════════════════════════════════════
#  ACTOR
# ═════════════════════════════════════════════════════════════════════════════
class SharedActor(Actor):
    """SAC actor with parameters shared across the N spatial cells.

    Each cell is scored by the same MLP on its F local features concatenated with
    the broadcast G-dim global block; the per-cell (mu, log_std) heads form the
    joint N-dim squashed-Gaussian action.
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
        # The actor trunk always uses LeakyReLU regardless of the critic's
        # activation (which flows in via policy_kwargs).
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
            raise ValueError(f"SharedActor: action_dim must equal N (={N}), got {action_dim}")

        self._global_dim = _split_global_dim(features_dim, N)
        self._per_agent_input_dim = N_AGENT_FEATURES + self._global_dim

        # Replace SB3's monolithic latent/heads with the per-cell shared trunk.
        latent_pi_net = create_mlp(
            input_dim=self._per_agent_input_dim,
            output_dim=-1,
            net_arch=net_arch,
            activation_fn=nn.LeakyReLU,
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if net_arch else self._per_agent_input_dim
        self.mu = nn.Linear(last_layer_dim, 1)
        self.log_std = nn.Linear(last_layer_dim, 1)
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.register_buffer("obs_norm_marker", torch.tensor([OBS_MARKER]))

    def get_std(self) -> torch.Tensor:
        return torch.zeros(self.N)

    def reset_noise(self, batch_size: int = 1) -> None:
        return

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape (B, obs) into (B*N, per_agent_input), then re-center to ~[-1, 1].

        Agent-major layout: the first F*N entries are the N per-cell blocks; the
        remaining G entries are the global block, broadcast to every cell.
        """
        B = features.shape[0]
        N, F = self.N, N_AGENT_FEATURES
        per_agent = features[:, : F * N].reshape(B, N, F)
        global_block = features[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        combined = 2.0 * combined - 1.0   # [0, 1.x] -> [-1, 1.x] re-centering
        return combined.reshape(B * N, self._per_agent_input_dim)

    def get_action_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        B = features.shape[0]
        latent_pi = self.latent_pi(self._per_agent_features(features))
        mean_actions = self.mu(latent_pi).reshape(B, self.N)
        log_std = torch.clamp(self.log_std(latent_pi).reshape(B, self.N),
                              LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self(observation, deterministic)


# ═════════════════════════════════════════════════════════════════════════════
#  CRITIC
# ═════════════════════════════════════════════════════════════════════════════
class _VdnQNet(nn.Sequential):
    """One twin: Q_total = sum_n Q_local(s_n, g, a_n), LayerNorm-regularised.

    Inherits nn.Sequential so the per-cell MLP layers register at top level
    (state-dict keys ``qf{i}.{layer}.{weight,bias}``). LayerNorm is inserted
    after each hidden Linear (before the activation), per Yue et al. 2023.
    """

    def __init__(self, N: int, per_agent_input_dim: int, net_arch: List[int],
                 activation_fn: Type[nn.Module] = nn.ReLU):
        layers = create_mlp(
            input_dim=per_agent_input_dim,
            output_dim=1,
            net_arch=net_arch,
            activation_fn=activation_fn,
            post_linear_modules=[nn.LayerNorm],
        )
        super().__init__(*layers)
        self.N = N
        self._per_agent_input_dim = per_agent_input_dim

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        N, F = self.N, N_AGENT_FEATURES
        local_obs = obs[:, : F * N].reshape(B, N, F)
        global_expanded = obs[:, F * N:].unsqueeze(1).expand(-1, N, -1)
        local_actions = actions.reshape(B, N, 1)
        local_inputs = torch.cat([local_obs, global_expanded, local_actions], dim=-1)
        local_q = nn.Sequential.forward(
            self, local_inputs.reshape(B * N, self._per_agent_input_dim)
        ).reshape(B, N, 1)
        return local_q.sum(dim=1)


class VdnCritic(ContinuousCritic):
    """Twin VDN critic with LayerNorm; clipped-double-Q target via min(Q1, Q2)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = False,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        N: int = N_AGENTS_DEFAULT,
        **kwargs,
    ):
        # Bypass ContinuousCritic.__init__ (which builds monolithic Q-nets);
        # initialise via the grandparent BaseModel.
        super(ContinuousCritic, self).__init__(
            observation_space, action_space,
            features_extractor=features_extractor, normalize_images=normalize_images,
        )
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.N = N

        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(f"VdnCritic: action_dim must equal N (={N}), got {action_dim}")
        per_agent_input_dim = N_AGENT_FEATURES + _split_global_dim(features_dim, N) + 1

        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net = _VdnQNet(N, per_agent_input_dim, net_arch, activation_fn)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features, actions)


# ═════════════════════════════════════════════════════════════════════════════
#  POLICY
# ═════════════════════════════════════════════════════════════════════════════
class SacVdnPolicy(SACPolicy):
    """SAC policy pairing the parameter-shared actor with the VDN LayerNorm critic."""

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SharedActor:
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> VdnCritic:
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return VdnCritic(**kw).to(self.device)


def make_sac_policy_kwargs(
    N: int = N_AGENTS_DEFAULT,
    actor_hidden: Tuple[int, ...] = (128, 128),
    critic_hidden: Tuple[int, ...] = (256, 256),
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """policy_kwargs for SB3 SAC with the CTDE VDN architecture.

    The actor trunk forces LeakyReLU internally; ``activation_fn`` here controls
    the critic (ReLU). Use with ``policy_class=SacVdnPolicy``.
    """
    kwargs: Dict[str, Any] = {
        "net_arch": {"pi": list(actor_hidden), "qf": list(critic_hidden)},
        "activation_fn": nn.ReLU,
    }
    if optimizer_kwargs is not None:
        kwargs["optimizer_kwargs"] = optimizer_kwargs
    return kwargs
