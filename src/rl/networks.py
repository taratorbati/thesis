# src/rl/networks.py  v2.8.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.7.0  (see change_spec_v28.md for full rationale)
#
#   1. CURRENT (v2.8) per-agent feature count: 8 → 9.
#        The v2.8 obs adds x1_overshoot_norm as a 9th per-agent feature.
#        The actor and critic first-layer widths grow accordingly:
#            PER_AGENT_INPUT_DIM        : 65 → 66
#            PER_AGENT_CRITIC_INPUT_DIM : 66 → 67
#            OBS_DIM_DEFAULT            : 1097 → 1227
#        Hidden widths, activation, twin-Q architecture: unchanged.
#
#   2. LEGACY CHECKPOINT LOADING PRESERVED for both v2.7 and v2.6.
#        - v2.7 best_model.zip (8 features/agent, dim=66 critic input):
#          loaded via _V27SharedActor + _V27FactorizedQNet.
#        - v2.6 best_model.zip (5 features/agent, dim=63 critic input,
#          local_q_net wrapper): loaded via _LegacySharedActor +
#          _WrappedFactorizedCritic.
#        - pre-VDN monolithic (837-dim flat): MonolithicCTDESACPolicy.
#
# Critic checkpoint variants (loaded by runner.py):
#   dim=67, flat       → CTDESACPolicy            (v2.8 — DEFAULT)
#   dim=66, flat       → V27CTDESACPolicy         (v2.7)
#   dim=63, wrapped    → WrappedVDNCTDESACPolicy  (v2.6 best_model.zip)
#   dim=63, flat       → WrappedVDNCTDESACPolicy  (v2.6 alt-key — defensive)
#   dim=837, flat      → MonolithicCTDESACPolicy  (pre-VDN, legacy)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import BaseFeaturesExtractor, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor, create_mlp
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.sac.policies import Actor, SACPolicy


# ─────────────────────────────────────────────────────────────────────────────
# v2.8 dimensions (CURRENT — used by all newly trained policies)
# Per-agent block (agent-major, 9 contiguous features per agent):
#   [x1_norm, x5_norm, x4_norm, x3,
#    elev_norm, Nr_norm, Nr_internal_norm, n_upstream_norm,
#    x1_overshoot_norm]
# ─────────────────────────────────────────────────────────────────────────────
N_AGENT_FEATURES           = 9
N_AGENTS_DEFAULT           = 130
N_GLOBAL_DIMS              = 57   # 9 scalars + 48 forecast
OBS_DIM_DEFAULT            = N_AGENT_FEATURES * N_AGENTS_DEFAULT + N_GLOBAL_DIMS   # 1227
PER_AGENT_INPUT_DIM        = N_AGENT_FEATURES + N_GLOBAL_DIMS                     # 66
PER_AGENT_CRITIC_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_DIMS + 1                 # 67

# ─────────────────────────────────────────────────────────────────────────────
# v2.7 LEGACY dimensions  (for loading v2.7 checkpoints)
# Per-agent block was 8 features:
#   [x1_norm, x5_norm, x4_norm, x3,
#    elev_norm, Nr_norm, Nr_internal_norm, n_upstream_norm]
# ─────────────────────────────────────────────────────────────────────────────
V27_N_AGENT_FEATURES           = 8
V27_OBS_DIM                    = V27_N_AGENT_FEATURES * N_AGENTS_DEFAULT + N_GLOBAL_DIMS   # 1097
V27_PER_AGENT_INPUT_DIM        = V27_N_AGENT_FEATURES + N_GLOBAL_DIMS                       # 65
V27_PER_AGENT_CRITIC_INPUT_DIM = V27_N_AGENT_FEATURES + N_GLOBAL_DIMS + 1                   # 66

# ─────────────────────────────────────────────────────────────────────────────
# v2.6 LEGACY dimensions  (for loading v2.6 best_model.zip and earlier)
# Per-agent block was 5 features:
#   [x1_norm, x5_norm, x4_norm, x3, gamma]  (gamma was buggy — see ch5)
# ─────────────────────────────────────────────────────────────────────────────
V26_N_AGENT_FEATURES           = 5
V26_OBS_DIM                    = V26_N_AGENT_FEATURES * N_AGENTS_DEFAULT + N_GLOBAL_DIMS   # 707
V26_PER_AGENT_INPUT_DIM        = V26_N_AGENT_FEATURES + N_GLOBAL_DIMS                       # 62
V26_PER_AGENT_CRITIC_INPUT_DIM = V26_N_AGENT_FEATURES + N_GLOBAL_DIMS + 1                   # 63

# Numerical stability bounds for the policy log-std
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ═════════════════════════════════════════════════════════════════════════════
#  v2.8 SHARED ACTOR  (9 per-agent features → 66-dim per-agent input)
# ═════════════════════════════════════════════════════════════════════════════
class SharedActor(Actor):
    """SAC actor with parameter-sharing across N spatial agents (v2.8).

    Each agent receives a 66-dim input vector:
      • 9  local features  (x1_norm, x5_norm, x4_norm, x3, elev_norm,
                            Nr_norm, Nr_internal_norm, n_upstream_norm,
                            x1_overshoot_norm)
      • 57 global context  (9 scalars + 48 forecast)

    A single MLP is applied to all N per-agent inputs in parallel, producing
    (mean_n, log_std_n) for n = 0,…,N-1.  The N per-agent action distributions
    are concatenated into a joint N-dimensional action distribution.

    v2.7 → v2.8: per-agent feature count grew from 8 to 9.  Hidden widths and
    output head unchanged.
    """

    # Class-level configuration — overridden in legacy subclasses
    _N_AGENT_FEATURES = N_AGENT_FEATURES
    _PER_AGENT_INPUT_DIM = PER_AGENT_INPUT_DIM

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
                f"SharedActor: action_dim must equal N (={N}), got {action_dim}"
            )

        expected_obs_dim = self._N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"{type(self).__name__}: features_dim must equal "
                f"{self._N_AGENT_FEATURES}*{N} + {N_GLOBAL_DIMS} = {expected_obs_dim}, "
                f"got {features_dim}. Check gym_env observation layout."
            )

        net_arch_list = net_arch if net_arch is not None else [128, 128]

        latent_pi_net = create_mlp(
            input_dim=self._PER_AGENT_INPUT_DIM,
            output_dim=-1,
            net_arch=net_arch_list,
            activation_fn=activation_fn,
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)

        last_layer_dim = net_arch_list[-1] if net_arch_list else self._PER_AGENT_INPUT_DIM
        self.mu      = nn.Linear(last_layer_dim, 1)
        self.log_std = nn.Linear(last_layer_dim, 1)

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def get_std(self) -> torch.Tensor:
        return torch.zeros(self.N)

    def reset_noise(self, batch_size: int = 1) -> None:
        return

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape flat batched obs into (B*N, per_agent_input_dim).

        Agent-major layout (matches gym_env):
          features[:, : F*N]   = N×F per-agent features
          features[:, F*N : ]  = G global features (broadcast to all agents)
        """
        B = features.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES

        per_agent = features[:, : F * N].reshape(B, N, F)
        global_block = features[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        return combined.reshape(B * N, self._PER_AGENT_INPUT_DIM)

    def get_action_dist_params(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        B = features.shape[0]

        per_agent_input = self._per_agent_features(features)
        latent_pi = self.latent_pi(per_agent_input)

        mean_actions = self.mu(latent_pi).reshape(B, self.N)
        log_std = self.log_std(latent_pi).reshape(B, self.N)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean_actions, log_std, {}

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def action_log_prob(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(
            mean_actions, log_std, **kwargs
        )

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return self(observation, deterministic)


class _V27SharedActor(SharedActor):
    """SharedActor for v2.7 checkpoints (8 per-agent features, 65-dim input)."""
    _N_AGENT_FEATURES    = V27_N_AGENT_FEATURES
    _PER_AGENT_INPUT_DIM = V27_PER_AGENT_INPUT_DIM


class _LegacySharedActor(SharedActor):
    """SharedActor for v2.6 checkpoints (5 per-agent features, 62-dim input)."""
    _N_AGENT_FEATURES    = V26_N_AGENT_FEATURES
    _PER_AGENT_INPUT_DIM = V26_PER_AGENT_INPUT_DIM


# ═════════════════════════════════════════════════════════════════════════════
#  v2.8 FACTORIZED CRITIC  (67-dim per-agent input)
# ═════════════════════════════════════════════════════════════════════════════
class _FactorizedQNet(nn.Sequential):
    """Q_total = Σ_n Q_local(s_n, g, a_n).  Inherits nn.Sequential so layers
    are registered at top level (state-dict keys: critic.qf0.0.weight, …).

    v2.8 per-agent critic input: 9 local features + 57 global + 1 action = 67.
    """

    _N_AGENT_FEATURES           = N_AGENT_FEATURES
    _PER_AGENT_CRITIC_INPUT_DIM = PER_AGENT_CRITIC_INPUT_DIM

    def __init__(
        self,
        N: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        layers = create_mlp(
            input_dim=self._PER_AGENT_CRITIC_INPUT_DIM,
            output_dim=1,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        super().__init__(*layers)
        self.N = N

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES

        local_obs       = obs[:, : F * N].reshape(B, N, F)
        global_block    = obs[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
        local_actions   = actions.reshape(B, N, 1)

        local_inputs = torch.cat(
            [local_obs, global_expanded, local_actions], dim=-1
        )
        local_inputs_flat = local_inputs.reshape(
            B * N, self._PER_AGENT_CRITIC_INPUT_DIM
        )
        local_q = nn.Sequential.forward(self, local_inputs_flat).reshape(B, N, 1)
        q_total = local_q.sum(dim=1)
        return q_total


class _V27FactorizedQNet(_FactorizedQNet):
    """_FactorizedQNet for v2.7 checkpoints (8 features, 66-dim input)."""
    _N_AGENT_FEATURES           = V27_N_AGENT_FEATURES
    _PER_AGENT_CRITIC_INPUT_DIM = V27_PER_AGENT_CRITIC_INPUT_DIM


class FactorizedContinuousCritic(ContinuousCritic):
    """Twin-Q factorized critic for v2.8.

    Replaces SB3's monolithic Q-networks with twin _FactorizedQNet instances.
    Bellman target uses min(Q1_total, Q2_total) — standard clipped double-Q.

    Class-level _N_AGENT_FEATURES and _QNET_CLS are overridden in legacy
    subclasses to handle v2.7 and v2.6 checkpoints.
    """

    _N_AGENT_FEATURES = N_AGENT_FEATURES
    _QNET_CLS         = _FactorizedQNet

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
        # Bypass ContinuousCritic.__init__ (which builds standard q_networks);
        # call grandparent BaseModel.__init__ directly.
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.N = N

        expected_obs_dim = self._N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"{type(self).__name__}: features_dim must equal "
                f"{expected_obs_dim}, got {features_dim}."
            )

        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(
                f"{type(self).__name__}: action_dim must equal N (={N}), "
                f"got {action_dim}."
            )

        # twin factorized Q-networks
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net = self._QNET_CLS(N=N, net_arch=net_arch, activation_fn=activation_fn)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = features
        return tuple(q_net(qvalue_input, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features, actions)


class _V27FactorizedContinuousCritic(FactorizedContinuousCritic):
    """FactorizedContinuousCritic for v2.7 checkpoints (8 features/agent)."""
    _N_AGENT_FEATURES = V27_N_AGENT_FEATURES
    _QNET_CLS         = _V27FactorizedQNet


# ═════════════════════════════════════════════════════════════════════════════
#  v2.8 CTDE SAC POLICY  (DEFAULT — used by train.py for new runs)
# ═════════════════════════════════════════════════════════════════════════════
class CTDESACPolicy(SACPolicy):
    """SAC policy with SharedActor + FactorizedContinuousCritic (v2.8)."""

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> SharedActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs["N"] = get_action_dim(self.action_space)
        return SharedActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> FactorizedContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        critic_kwargs["N"] = get_action_dim(self.action_space)
        return FactorizedContinuousCritic(**critic_kwargs).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.7 LEGACY VDN POLICY (for loading v2.7 best_model.zip)
# ═════════════════════════════════════════════════════════════════════════════
class V27CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.7 checkpoints (8 features/agent, dim=66 critic input).

    Uses _V27SharedActor and _V27FactorizedContinuousCritic.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V27SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V27FactorizedContinuousCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.6 LEGACY VDN POLICY (for loading v2.6 best_model.zip)
# ═════════════════════════════════════════════════════════════════════════════
class _FactorizedQNetWrapped(nn.Module):
    """_FactorizedQNet with local_q_net wrapper — matches v2.6 best_model.zip keys."""

    def __init__(self, N: int, net_arch: List[int],
                 activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.N = N
        layers = create_mlp(
            input_dim=V26_PER_AGENT_CRITIC_INPUT_DIM,
            output_dim=1,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        self.local_q_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, N = obs.shape[0], self.N
        F    = V26_N_AGENT_FEATURES
        local_obs       = obs[:, : F * N].reshape(B, N, F)
        global_expanded = obs[:, F * N:].unsqueeze(1).expand(-1, N, -1)
        local_actions   = actions.reshape(B, N, 1)
        local_inputs    = torch.cat([local_obs, global_expanded, local_actions], dim=-1)
        local_q = self.local_q_net(
            local_inputs.reshape(B * N, V26_PER_AGENT_CRITIC_INPUT_DIM)
        ).reshape(B, N, 1)
        return local_q.sum(dim=1)


class _WrappedFactorizedCritic(ContinuousCritic):
    """Twin-Q critic with v2.6 wrapped keys (dim=63)."""

    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, normalize_images=False,
                 n_critics=2, share_features_extractor=True, N=N_AGENTS_DEFAULT, **kwargs):
        super(ContinuousCritic, self).__init__(
            observation_space, action_space,
            features_extractor=features_extractor, normalize_images=normalize_images)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.N = N

        expected_obs_dim = V26_N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"_WrappedFactorizedCritic (legacy v2.6): features_dim must equal "
                f"{expected_obs_dim}, got {features_dim}."
            )

        self.q_networks: List[_FactorizedQNetWrapped] = []
        for idx in range(n_critics):
            q_net = _FactorizedQNetWrapped(N=N, net_arch=net_arch, activation_fn=activation_fn)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs, actions):
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q(features, actions) for q in self.q_networks)

    def q1_forward(self, obs, actions):
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features, actions)


class WrappedVDNCTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.6 best_model.zip (VDN, local_q_net keys, dim=63)."""

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _LegacySharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _WrappedFactorizedCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  Pre-VDN LEGACY POLICY (monolithic 837-dim critic)
# ═════════════════════════════════════════════════════════════════════════════
class MonolithicCTDESACPolicy(SACPolicy):
    """CTDESACPolicy with the original monolithic 837-dim twin-Q critic.

    For loading pre-VDN checkpoints (the v2.4 pilot saved with a [256, 837]
    first Linear layer in the critic).  Uses _LegacySharedActor for the actor
    (5 per-agent features, matching v2.4 obs layout).
    """

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> _LegacySharedActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs["N"] = get_action_dim(self.action_space)
        return _LegacySharedActor(**actor_kwargs).to(self.device)

    # make_critic is NOT overridden → SACPolicy's standard ContinuousCritic,
    # producing the [256, 837] first-layer shape seen in the checkpoint.


# ═════════════════════════════════════════════════════════════════════════════
#  Convenience policy_kwargs builder
# ═════════════════════════════════════════════════════════════════════════════
def make_sac_policy_kwargs(
    N: int = N_AGENTS_DEFAULT,
    actor_hidden: Tuple[int, ...] = (128, 128),
    critic_hidden: Tuple[int, ...] = (256, 256),
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """policy_kwargs for SB3 SAC with the v2.8 CTDE VDN architecture.

    Use with policy_class=CTDESACPolicy in the SAC constructor.
    """
    kwargs: Dict[str, Any] = {
        "net_arch": {
            "pi": list(actor_hidden),
            "qf": list(critic_hidden),
        },
        "activation_fn": nn.ReLU,
    }
    if optimizer_kwargs is not None:
        kwargs["optimizer_kwargs"] = optimizer_kwargs
    return kwargs
