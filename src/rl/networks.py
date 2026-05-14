# src/rl/networks.py  v2.6.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.5.x
#
#  1.  NEW: FactorizedContinuousCritic (Value Decomposition Network)
#        Replaces the monolithic 837-dim → 1 critic with a per-agent
#        decomposition  Q_total = Σ_n Q_local(s_n^local, s_global, a_n).
#        Shared MLP across all 130 agents (parallel to the shared actor).
#        Twin-Q preserved (q1_net, q2_net) for clipped double-Q learning.
#        Resolves the spatial credit-assignment failure that caused the
#        previous policy to collapse to spatially homogeneous behaviour.
#
#  2.  FIX: SharedActor reshape bug
#        The previous version used  .reshape(B, 5, N).permute(0, 2, 1),
#        which assumes a feature-major obs layout.  The env produces an
#        agent-major layout (5 contiguous features per agent).  The fix
#        is a direct  .reshape(B, N, 5) — no permute.
#        This bug was previously masked because the policy collapsed to a
#        uniform action regardless of the spatial input ordering.
#
#  3.  CTDESACPolicy now overrides both make_actor() and make_critic().
#
#  All other architectural choices (shared actor, CTDE, 707-dim obs,
#  62-dim per-agent input, [128,128] actor hidden, [256,256] critic hidden,
#  log_std clamp [-20, 2], SquashedDiagGaussian) are unchanged.
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


# ── observation layout constants  (must match gym_env.py v2.5+) ──────────────
# Per-agent block: 5 contiguous features per agent, agent-major:
#   [x1_0, x5_0, x4_0, x3_0, γ_0, x1_1, x5_1, x4_1, x3_1, γ_1, ...]
N_AGENT_FEATURES = 5
N_AGENTS_DEFAULT = 130
N_GLOBAL_DIMS    = 57   # 9 scalars + 48 forecast (6 vars × 8 days)
OBS_DIM_DEFAULT  = N_AGENT_FEATURES * N_AGENTS_DEFAULT + N_GLOBAL_DIMS  # 707
PER_AGENT_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_DIMS                  # 62

# Per-agent critic input: local state + global context + local action
PER_AGENT_CRITIC_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_DIMS + 1       # 63

# Numerical stability bounds for the policy log-std
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED ACTOR  (corrected reshape)
# ═════════════════════════════════════════════════════════════════════════════
class SharedActor(Actor):
    """SAC actor with parameter-sharing across N spatial agents.

    Each agent receives a 62-dim input vector consisting of:
      • 5  local features        (its own x1_norm, x5_norm, x4_norm, x3, γ)
      • 57 global context dims   (9 scalars + 48 forecast, identical for all)

    A single MLP (the "shared MLP") is applied to all N per-agent inputs in
    parallel, producing (mean_n, log_std_n) for n = 0,...,N-1.  The N
    per-agent action distributions are concatenated into a joint
    N-dimensional action distribution.

    Parameters are reduced ~86% versus a naive monolithic actor while
    enforcing spatial equivariance: permuting the agent index permutes the
    action in the same way.
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

        expected_obs_dim = N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"SharedActor: features_dim must equal "
                f"{N_AGENT_FEATURES}*{N} + {N_GLOBAL_DIMS} = {expected_obs_dim}, "
                f"got {features_dim}. Check gym_env observation layout."
            )

        net_arch_list = net_arch if net_arch is not None else [128, 128]

        # Drop the parent Actor's mu/log_std heads — they expect action_dim out;
        # we replace them with 1-out heads to be applied per agent.
        latent_pi_net = create_mlp(
            input_dim=PER_AGENT_INPUT_DIM,
            output_dim=-1,
            net_arch=net_arch_list,
            activation_fn=activation_fn,
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)

        last_layer_dim = net_arch_list[-1] if net_arch_list else PER_AGENT_INPUT_DIM
        self.mu = nn.Linear(last_layer_dim, 1)
        self.log_std = nn.Linear(last_layer_dim, 1)

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_std(self) -> torch.Tensor:
        return torch.zeros(self.N)

    def reset_noise(self, batch_size: int = 1) -> None:
        return

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape a flat 707-dim batched obs into (B*N, 62) per-agent inputs.

        Layout produced by gym_env (agent-major):
          features[:, 0:650]   = per-agent block, 5 contiguous features per agent
          features[:, 650:707] = 57 global dims (9 scalars + 48 forecast)
        """
        B = features.shape[0]
        N = self.N

        # (B, N*5) → (B, N, 5)   [agent-major: each agent's 5 features are contiguous]
        per_agent = features[:, : N_AGENT_FEATURES * N].reshape(
            B, N, N_AGENT_FEATURES
        )

        # (B, 57) → (B, 1, 57) → (B, N, 57)  broadcast global to every agent
        global_block = features[:, N_AGENT_FEATURES * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)

        # (B, N, 5+57=62) → (B*N, 62)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        return combined.reshape(B * N, PER_AGENT_INPUT_DIM)

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


# ═════════════════════════════════════════════════════════════════════════════
#  FACTORIZED CRITIC  (Value Decomposition Network)
# ═════════════════════════════════════════════════════════════════════════════
class _FactorizedQNet(nn.Module):
    """Single Q-network that decomposes Q_total = Σ_n Q_local(s_n, g, a_n).

    The local MLP is shared across all N agents.  Input per agent:
      • 5  local state features
      • 57 global context (broadcast)
      • 1  local action
    → 63 inputs, scalar output (Q_n).

    Q_total is the sum of Q_n across the 130 agents.
    """

    def __init__(
        self,
        N: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.N = N
        layers = create_mlp(
            input_dim=PER_AGENT_CRITIC_INPUT_DIM,
            output_dim=1,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        self.local_q_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        obs:     (B, 707)   flat batched observation
        actions: (B, 130)   joint action

        returns: (B, 1)     Q_total = Σ_n Q_local(s_n, g, a_n)
        """
        B = obs.shape[0]
        N = self.N

        # agent-major reshape (same convention as SharedActor)
        local_obs = obs[:, : N_AGENT_FEATURES * N].reshape(B, N, N_AGENT_FEATURES)
        global_block = obs[:, N_AGENT_FEATURES * N:]                    # (B, 57)
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)   # (B, N, 57)

        local_actions = actions.reshape(B, N, 1)                         # (B, N, 1)

        # concatenate (5 + 57 + 1 = 63) per agent
        local_inputs = torch.cat(
            [local_obs, global_expanded, local_actions], dim=-1
        )                                                                # (B, N, 63)

        # apply shared MLP to all N agents in parallel
        local_inputs_flat = local_inputs.reshape(B * N, PER_AGENT_CRITIC_INPUT_DIM)
        local_q = self.local_q_net(local_inputs_flat).reshape(B, N, 1)

        # Σ across the agent dimension
        q_total = local_q.sum(dim=1)                                     # (B, 1)
        return q_total


class FactorizedContinuousCritic(ContinuousCritic):
    """Twin-Q factorized critic conforming to SB3's ContinuousCritic API.

    Replaces the standard monolithic Q-networks with twin _FactorizedQNet
    instances.  Each instance computes Q_total = Σ_n Q_local independently.
    The Bellman target uses min(Q1_total, Q2_total) — standard clipped
    double-Q learning.

    Interface preserved:
      • forward(obs, actions)    → Tuple[Q1, Q2]   each (B, 1)
      • q1_forward(obs, actions) → Q1              (B, 1)
    """

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

        expected_obs_dim = N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"FactorizedContinuousCritic: features_dim must equal "
                f"{expected_obs_dim}, got {features_dim}."
            )

        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(
                f"FactorizedContinuousCritic: action_dim must equal N (={N}), "
                f"got {action_dim}."
            )

        # twin factorized Q-networks
        self.q_networks: List[_FactorizedQNet] = []
        for idx in range(n_critics):
            q_net = _FactorizedQNet(N=N, net_arch=net_arch, activation_fn=activation_fn)
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
#  CTDE SAC POLICY  (overrides both make_actor and make_critic)
# ═════════════════════════════════════════════════════════════════════════════
class CTDESACPolicy(SACPolicy):
    """SAC policy with shared-parameter SharedActor + FactorizedContinuousCritic.

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
#  Convenience policy_kwargs builder
# ═════════════════════════════════════════════════════════════════════════════
def make_sac_policy_kwargs(
    N: int = N_AGENTS_DEFAULT,
    actor_hidden: Tuple[int, ...] = (128, 128),
    critic_hidden: Tuple[int, ...] = (256, 256),
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the policy_kwargs dict for SB3 SAC with the CTDE VDN architecture.

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
