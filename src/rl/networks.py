# =============================================================================
# src/rl/networks.py
# Custom policy networks for Stable-Baselines3 SAC implementing
# Centralized Training with Decentralized Execution (CTDE).
#
# Architecture:
#   Actor: shared-parameter MLP applied to each agent's local features.
#     For each of the N=130 agents, the actor receives a 15-dim input
#     (5 per-agent features + 10 global scalars) and produces a 1-dim
#     action mean and log-std. Same MLP weights are applied across all
#     agents → ~10× fewer parameters than a monolithic 660→130 actor and
#     directly imposes the inductive bias that all agents share the same
#     control policy modulo their local state and shared global context.
#
#   Critic: standard centralized SAC twin Q-network.
#     Sees the full 660-dim joint state and 130-dim joint action
#     (790-dim input total) through a standard MLP.
#     The critic does NOT use weight sharing — it must perform credit
#     assignment across the joint action space, which requires access
#     to the full joint state-action.
#
# The CTDE separation: the actor uses only information available at
# deployment (per-agent local + shared global context), while the critic
# uses the full joint state during training. After training, the critic
# is discarded; the deployed policy is the actor alone.
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


# ── Architecture constants ──────────────────────────────────────────────
# Number of per-agent features in the observation (must match gym_env.py)
N_AGENT_FEATURES = 5    # x1_norm, x5_norm, x4_norm, x3, elevation
N_GLOBAL_SCALARS = 10   # day_frac, budget_frac, burn_rate, rain, ..., g_base
PER_AGENT_INPUT_DIM = N_AGENT_FEATURES + N_GLOBAL_SCALARS  # = 15

# Numerical stability bounds for the policy log-std
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SharedActor(Actor):
    """SAC actor with parameter sharing across N spatial agents.

    Each of the N agents shares the same MLP weights. The MLP takes a
    15-dim input (5 per-agent features + 10 global scalars broadcast to
    all agents) and outputs a 1-dim (mean, log_std) action distribution
    parameterization per agent. The N independent per-agent actions are
    assembled into the joint action vector.

    This subclasses SB3's SACPolicy.Actor and overrides the network
    construction and forward methods. The interface to the surrounding
    SAC algorithm (action_log_prob, _predict, etc.) is preserved.

    Parameters
    ----------
    observation_space : gym Space
        The full 660-dim joint observation space.
    action_space : gym Space
        The full 130-dim joint action space.
    net_arch : list of int
        Hidden layer sizes for the per-agent shared MLP.
    features_extractor : BaseFeaturesExtractor
        Standard SB3 features extractor (FlattenExtractor by default).
    features_dim : int
        Output dim of the features extractor (typically equal to obs_dim).
    activation_fn : type[nn.Module]
        Activation function class.
    use_sde : bool
        If True, use State-Dependent Exploration. Not supported here;
        leave as False.
    log_std_init : float
        Initial value of log_std for the diagonal Gaussian.
    full_std : bool
        SDE only — ignored.
    use_expln : bool
        SDE only — ignored.
    clip_mean : float
        Clip the mean to [-clip_mean, clip_mean] before tanh squashing.
        SB3 default is 2.0.
    normalize_images : bool
        Image preprocessing — irrelevant here.
    N : int
        Number of spatial agents. Defaults to 130.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        N: int = 130,
    ):
        # We do NOT call the parent Actor.__init__ because that would build
        # a monolithic MLP based on features_dim. Instead, we build our
        # own per-agent MLP. Bypass via the grandparent (BasePolicy).
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        if use_sde:
            raise NotImplementedError(
                "SharedActor does not support SDE. Set use_sde=False."
            )

        self.N = N
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.clip_mean = clip_mean

        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(
                f"SharedActor expects action_dim == N (={N}), got {action_dim}"
            )

        # Verify the observation layout: features_dim must equal
        # 5 * N + 10 (per-agent block + global scalars).
        expected_obs_dim = N_AGENT_FEATURES * N + N_GLOBAL_SCALARS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"SharedActor expects features_dim == {expected_obs_dim} "
                f"(5*{N} + 10), got {features_dim}. Ensure the gym_env "
                f"observation matches this layout."
            )

        # Build the per-agent MLP: input PER_AGENT_INPUT_DIM, hidden net_arch,
        # output a shared latent representation. Two output heads (mean and
        # log_std) follow the latent layer.
        latent_pi_net = create_mlp(
            input_dim=PER_AGENT_INPUT_DIM,
            output_dim=-1,                        # no final layer; we add heads
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else PER_AGENT_INPUT_DIM
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
        """Reshape (batch, 5N+10) → (batch*N, 15).

        Splits the flat observation into the per-agent block (5*N) and
        the global block (10), broadcasts the globals to all agents,
        and concatenates.
        """
        batch_size = features.shape[0]
        N = self.N

        # Per-agent block: (batch, 5*N) → (batch, 5, N) → (batch, N, 5)
        per_agent_flat = features[:, :N_AGENT_FEATURES * N]
        per_agent = per_agent_flat.reshape(batch_size, N_AGENT_FEATURES, N)
        per_agent = per_agent.permute(0, 2, 1)  # (batch, N, 5)

        # Global block broadcast: (batch, 10) → (batch, N, 10)
        global_scalars = features[:, N_AGENT_FEATURES * N:]
        global_expanded = global_scalars.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: (batch, N, 15) → (batch*N, 15)
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

        # Reshape to (batch*N, 15) and run the shared MLP
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
        # Note: SquashedDiagGaussianDistribution.actions_from_params returns
        # the raw action; SAC then maps it to the action space.
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


# ── Convenience builder ──────────────────────────────────────────────────

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
    critic_hidden : tuple of int
        Hidden layer sizes for the centralized critic MLP.

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
        # both the actor (which then internally reshapes the flat features)
        # and the critic (which uses the full flat observation).
    }
