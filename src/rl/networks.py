# src/rl/networks.py  v2.7.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.6.x  (see change_spec_v27.md for full rationale)
#
#   1. CURRENT (v2.7) per-agent feature count: 5 → 8.
#        Three new static topographic features are now in the gym_env obs:
#        Nr_norm, Nr_internal_norm, n_upstream_norm.  The actor and critic
#        first-layer widths grow correspondingly:
#            PER_AGENT_INPUT_DIM        : 62 → 65
#            PER_AGENT_CRITIC_INPUT_DIM : 63 → 66
#            OBS_DIM_DEFAULT            : 707 → 1097
#        All architectural choices (shared actor, VDN factorised critic,
#        twin-Q, hidden widths) are otherwise unchanged.
#
#   2. LEGACY (v2.6) checkpoint loading is preserved.
#        WrappedVDNCTDESACPolicy and MonolithicCTDESACPolicy continue to
#        load checkpoints trained against the 707-dim observation layout.
#        They are kept distinct from the v2.7 path by parameterising the
#        legacy actor / critic on V26_* constants instead of the
#        module-default v2.7 constants.  Loading a v2.6 best_model.zip
#        therefore continues to work for the Chapter 5 architecture-
#        comparison story.
#
# Three known checkpoint variants and their load paths:
#   - dim=66, flat       → CTDESACPolicy             (v2.7 — DEFAULT)
#   - dim=63, wrapped    → WrappedVDNCTDESACPolicy   (v2.6 best_model.zip)
#   - dim=63, flat       → CTDESACPolicy with V26_* constants (n/a — no
#                          such checkpoint exists in the current repo, but
#                          this case is handled by treating it as legacy
#                          VDN under WrappedVDN — see runner.py)
#   - dim=837, flat      → MonolithicCTDESACPolicy   (pre-VDN, legacy)
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
# v2.7 dimensions (CURRENT — used by all newly trained policies)
# Per-agent block (agent-major, 8 contiguous features per agent):
#   [x1_norm, x5_norm, x4_norm, x3, elev_norm,
#    Nr_norm, Nr_internal_norm, n_upstream_norm]
# ─────────────────────────────────────────────────────────────────────────────
N_AGENT_FEATURES           = 8
N_AGENTS_DEFAULT           = 130
N_GLOBAL_DIMS              = 57   # 9 scalars + 48 forecast (6 vars × 8 days)
OBS_DIM_DEFAULT            = N_AGENT_FEATURES * N_AGENTS_DEFAULT + N_GLOBAL_DIMS   # 1097
PER_AGENT_INPUT_DIM        = N_AGENT_FEATURES + N_GLOBAL_DIMS                     # 65
PER_AGENT_CRITIC_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_DIMS + 1                 # 66

# ─────────────────────────────────────────────────────────────────────────────
# v2.6 LEGACY dimensions  (kept ONLY for loading pre-v2.7 checkpoints).
# Do not use these for new training.  Per-agent block was 5 features:
#   [x1_norm, x5_norm, x4_norm, x3, gamma]
# where the 5th slot held either a per-agent elevation (runner.py) or a
# field-uniform GDD scalar (gym_env.py) depending on which side of the
# v2.6 obs-layout bug you looked at.
# ─────────────────────────────────────────────────────────────────────────────
V26_N_AGENT_FEATURES           = 5
V26_OBS_DIM                    = V26_N_AGENT_FEATURES * N_AGENTS_DEFAULT + N_GLOBAL_DIMS   # 707
V26_PER_AGENT_INPUT_DIM        = V26_N_AGENT_FEATURES + N_GLOBAL_DIMS                       # 62
V26_PER_AGENT_CRITIC_INPUT_DIM = V26_N_AGENT_FEATURES + N_GLOBAL_DIMS + 1                   # 63

# Numerical stability bounds for the policy log-std
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ═════════════════════════════════════════════════════════════════════════════
#  v2.7 SHARED ACTOR  (8 per-agent features → 65-dim per-agent input)
# ═════════════════════════════════════════════════════════════════════════════
class SharedActor(Actor):
    """SAC actor with parameter-sharing across N spatial agents (v2.7).

    Each agent receives a 65-dim input vector consisting of:
      • 8  local features        (its own x1_norm, x5_norm, x4_norm, x3,
                                  elev_norm, Nr_norm, Nr_internal_norm,
                                  n_upstream_norm)
      • 57 global context dims   (9 scalars + 48 forecast, identical for all)

    A single MLP (the "shared MLP") is applied to all N per-agent inputs in
    parallel, producing (mean_n, log_std_n) for n = 0,...,N-1.  The N
    per-agent action distributions are concatenated into a joint
    N-dimensional action distribution.

    Parameters are reduced ~86% versus a naive monolithic actor while
    enforcing spatial equivariance: permuting the agent index permutes the
    action in the same way.

    v2.7 vs v2.6:  per-agent feature count grew from 5 to 8, so the input
    width grew from 62 to 65.  The hidden widths and output head are
    unchanged.
    """

    # Class-level configuration — overridden in _LegacySharedActor for v2.6 loads
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

        # Drop the parent Actor's mu/log_std heads — they expect action_dim out;
        # we replace them with 1-out heads to be applied per agent.
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

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_std(self) -> torch.Tensor:
        return torch.zeros(self.N)

    def reset_noise(self, batch_size: int = 1) -> None:
        return

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape a flat batched obs into (B*N, per_agent_input_dim).

        Layout produced by gym_env (agent-major):
          features[:, : F*N]   = per-agent block, F contiguous features per agent
          features[:, F*N : ]  = N_GLOBAL_DIMS global dims (broadcast to all agents)
        """
        B = features.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES

        # (B, N*F) → (B, N, F)
        per_agent = features[:, : F * N].reshape(B, N, F)

        # (B, G) → (B, 1, G) → (B, N, G)  broadcast global to every agent
        global_block = features[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)

        # (B, N, F+G) → (B*N, F+G)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        return combined.reshape(B * N, self._PER_AGENT_INPUT_DIM)

    # ── SAC actor interface ──────────────────────────────────────────────────
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


class _LegacySharedActor(SharedActor):
    """SharedActor for v2.6 checkpoints (5 per-agent features, 62-dim input).

    Subclass-only override of the class-level dimension constants.  All
    behaviour is identical to SharedActor; only the reshape geometry
    differs to match the obs layout the legacy checkpoint was trained on.
    """
    _N_AGENT_FEATURES    = V26_N_AGENT_FEATURES
    _PER_AGENT_INPUT_DIM = V26_PER_AGENT_INPUT_DIM


# ═════════════════════════════════════════════════════════════════════════════
#  v2.7 FACTORIZED CRITIC  (Value Decomposition Network, 66-dim per-agent input)
# ═════════════════════════════════════════════════════════════════════════════
class _FactorizedQNet(nn.Sequential):
    """Single Q-network that decomposes Q_total = Σ_n Q_local(s_n, g, a_n).

    Inherits from nn.Sequential so that the MLP layers are registered
    directly as numeric children (0, 1, 2, …) instead of being nested
    inside an extra ``local_q_net`` attribute.  This matches the
    state-dict key naming used by v2.7 training runs
    (e.g. ``critic.qf0.0.weight``).

    The local MLP is shared across all N agents.  Input per agent (v2.7):
      • 8  local state features
      • 57 global context (broadcast)
      • 1  local action
    → 66 inputs, scalar output (Q_n).

    Q_total is the sum of Q_n across the N agents.
    """

    # Class-level configuration — overridden in legacy variant
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
        """
        obs:     (B, OBS_DIM)   flat batched observation
        actions: (B, N)          joint action  (N = self.N)

        returns: (B, 1)          Q_total = Σ_n Q_local(s_n, g, a_n)
        """
        B = obs.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES

        # agent-major reshape (same convention as SharedActor)
        local_obs       = obs[:, : F * N].reshape(B, N, F)
        global_block    = obs[:, F * N:]                                # (B, G)
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)   # (B, N, G)
        local_actions   = actions.reshape(B, N, 1)                       # (B, N, 1)

        # concatenate (F + G + 1) per agent
        local_inputs = torch.cat(
            [local_obs, global_expanded, local_actions], dim=-1
        )                                                                # (B, N, F+G+1)

        # apply the shared MLP to all N agents in parallel
        local_inputs_flat = local_inputs.reshape(
            B * N, self._PER_AGENT_CRITIC_INPUT_DIM
        )
        local_q = nn.Sequential.forward(self, local_inputs_flat).reshape(B, N, 1)

        # Σ across the agent dimension
        q_total = local_q.sum(dim=1)                                     # (B, 1)
        return q_total


class FactorizedContinuousCritic(ContinuousCritic):
    """Twin-Q factorized critic conforming to SB3's ContinuousCritic API (v2.7).

    Replaces the standard monolithic Q-networks with twin _FactorizedQNet
    instances.  Each instance computes Q_total = Σ_n Q_local independently.
    The Bellman target uses min(Q1_total, Q2_total) — standard clipped
    double-Q learning.

    Interface preserved:
      • forward(obs, actions)    → Tuple[Q1, Q2]   each (B, 1)
      • q1_forward(obs, actions) → Q1              (B, 1)
    """

    # Class-level configuration — overridden in legacy variant
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
        # Call the great-grandparent BaseModel.__init__ to set up features
        # extractor properly, then bypass ContinuousCritic's q_networks build.
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
        # Extract flat features (FlattenExtractor is the identity for Box obs)
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = features
        return tuple(q_net(qvalue_input, actions) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Used by the actor's policy loss (which maximises Q1 only)."""
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](features, actions)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.7 CTDE SAC POLICY  (DEFAULT — used by train.py for new runs)
# ═════════════════════════════════════════════════════════════════════════════
class CTDESACPolicy(SACPolicy):
    """SAC policy with SharedActor + FactorizedContinuousCritic (v2.7).

    The agent count N is read from the action space dimensionality.
    """

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
#  v2.6 LEGACY VDN POLICY — VDN critic with local_q_net wrapper (best_model.zip)
#  Used for checkpoints where _FactorizedQNet stored layers as self.local_q_net
#  (keys: critic.qf0.local_q_net.0.weight, shape [256, 63]).
# ═════════════════════════════════════════════════════════════════════════════
class _FactorizedQNetWrapped(nn.Module):
    """_FactorizedQNet with named local_q_net wrapper — matches best_model.zip keys.

    Uses V26_* constants (5 per-agent features, 63-dim per-agent critic input).
    """

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
    """Twin-Q critic using _FactorizedQNetWrapped — matches v2.6 best_model.zip.

    Uses V26_OBS_DIM = 707 expectations.
    """

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
    """CTDESACPolicy for v2.6 best_model.zip (VDN, local_q_net keys, dim=63).

    Uses _LegacySharedActor (62-dim per-agent input) and _WrappedFactorizedCritic.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _LegacySharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _WrappedFactorizedCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  PRE-VDN LEGACY POLICY — monolithic 837-dim twin-Q critic
#  Used ONLY for loading checkpoints from before the VDN upgrade (the
#  v2.4 pilot saved with a [256, 837] first Linear layer in the critic).
# ═════════════════════════════════════════════════════════════════════════════
class MonolithicCTDESACPolicy(SACPolicy):
    """CTDESACPolicy with the original monolithic 837-dim twin-Q critic.

    Identical to the pre-VDN architecture:
      - Actor: _LegacySharedActor (5 per-agent features × 130 + 57 globals = 707
        obs, 62-dim per-agent input — matches what the v2.4 pilot was trained on)
      - Critic: standard SB3 ContinuousCritic(837 → 256 → 256 → 1) × 2

    Use this as the custom_objects override when loading a pre-VDN checkpoint:

        SAC.load(path, custom_objects={"policy_class": MonolithicCTDESACPolicy})
    """

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> _LegacySharedActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs["N"] = get_action_dim(self.action_space)
        return _LegacySharedActor(**actor_kwargs).to(self.device)

    # make_critic is NOT overridden → falls back to SACPolicy's standard
    # ContinuousCritic which takes the full (obs + action) concatenation as
    # input, producing the [256, 837] first-layer shape seen in the checkpoint.


# ═════════════════════════════════════════════════════════════════════════════
#  Convenience policy_kwargs builder
# ═════════════════════════════════════════════════════════════════════════════
def make_sac_policy_kwargs(
    N: int = N_AGENTS_DEFAULT,
    actor_hidden: Tuple[int, ...] = (128, 128),
    critic_hidden: Tuple[int, ...] = (256, 256),
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the policy_kwargs dict for SB3 SAC with the v2.7 CTDE VDN architecture.

    Use with policy_class=CTDESACPolicy in the SAC constructor:

        from stable_baselines3 import SAC
        from src.rl.networks import CTDESACPolicy, make_sac_policy_kwargs

        model = SAC(
            policy=CTDESACPolicy,
            env=env,
            policy_kwargs=make_sac_policy_kwargs(N=130),
            ...
        )
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
