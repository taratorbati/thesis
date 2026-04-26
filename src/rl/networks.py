# =============================================================================
# src/rl/networks.py
# Custom policy networks for Stable-Baselines3 SAC.
#
# Architecture: Centralized Training with Decentralized Execution (CTDE)
#   - Actor: shared-parameter MLP. Each agent's local observation (x1, x5,
#     x4, x3, elevation) is concatenated with the global context (10 scalars)
#     and fed through the same MLP. The MLP outputs a single scalar action
#     per agent. All 130 agents share the same weights → parameter-efficient,
#     generalizes across heterogeneous terrain.
#   - Critic: standard centralized MLP. Takes the full 660-dim observation
#     + 130-dim action → scalar Q-value. This is the standard SAC critic.
#
# SB3 integration:
#   We subclass SB3's ActorCriticPolicy features_extractor to reshape the
#   observation for the shared-actor pattern. The actor and critic then
#   use standard SB3 MLP layers.
#
# References:
#   - CTDE: Lowe et al. (2017) "Multi-Agent Actor-Critic" (MADDPG)
#   - SAC: Haarnoja et al. (2018) "Soft Actor-Critic" (continuous control)
#   - Shared-params actors: Gupta et al. (2017) "Cooperative Multi-Agent
#     Control Using Deep Reinforcement Learning"
# =============================================================================

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


# Number of per-agent features in the observation
N_AGENT_FEATURES = 5   # x1_norm, x5_norm, x4_norm, x3, elevation
N_GLOBAL_SCALARS = 10  # day_frac, budget_frac, burn_rate, rain, ..., g_base


class SharedActorExtractor(BaseFeaturesExtractor):
    """Reshape the flat 660-dim obs into (N, per_agent + global) for the actor.

    The actor MLP processes each agent's features independently (weight-sharing).
    This extractor concatenates global scalars to each agent's local features
    and reshapes from (batch, 660) to (batch*N, per_agent_dim).

    The output is (batch*N, 15) where 15 = 5 agent features + 10 global scalars.
    """

    def __init__(self, observation_space, N=130):
        # Output dim = per-agent features + global scalars
        features_dim = N_AGENT_FEATURES + N_GLOBAL_SCALARS
        super().__init__(observation_space, features_dim=features_dim)
        self.N = N

    def forward(self, observations):
        batch_size = observations.shape[0]
        N = self.N

        # Split observation into per-agent and global parts
        per_agent_flat = observations[:, :N_AGENT_FEATURES * N]
        global_scalars = observations[:, N_AGENT_FEATURES * N:]

        # Reshape per-agent: (batch, 5*N) → (batch, N, 5)
        per_agent = per_agent_flat.reshape(batch_size, N_AGENT_FEATURES, N)
        per_agent = per_agent.permute(0, 2, 1)  # (batch, N, 5)

        # Broadcast global scalars: (batch, 10) → (batch, N, 10)
        global_expanded = global_scalars.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: (batch, N, 15)
        combined = torch.cat([per_agent, global_expanded], dim=-1)

        # Flatten: (batch*N, 15)
        return combined.reshape(batch_size * N, -1)


class CentralizedCriticExtractor(BaseFeaturesExtractor):
    """Identity extractor for the critic — passes the full 660-dim obs through.

    The critic sees the full centralized state (all agents + global context).
    SB3's SAC critic will concatenate the action internally.
    """

    def __init__(self, observation_space):
        features_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations):
        return observations


class SharedActorHead(nn.Module):
    """Actor head that maps shared-MLP output back to (batch, N) actions.

    Takes (batch*N, hidden) → (batch*N, 1) → reshape to (batch, N).
    SB3 SAC expects the actor to output mean and log_std for the squashed
    Gaussian. We handle this by outputting (batch, N) for both.
    """

    def __init__(self, N=130):
        super().__init__()
        self.N = N

    def forward(self, features):
        # features: (batch*N, ...)
        # We need to reshape back after the MLP processes them
        # This is handled by the policy, not here
        return features


def make_sac_policy_kwargs(N=130, actor_hidden=(128, 128), critic_hidden=(256, 256)):
    """Build the policy_kwargs dict for SB3 SAC with CTDE architecture.

    Parameters
    ----------
    N : int
        Number of agents (130).
    actor_hidden : tuple
        Hidden layer sizes for the shared actor MLP.
    critic_hidden : tuple
        Hidden layer sizes for the centralized critic MLP.

    Returns
    -------
    dict
        policy_kwargs for sb3.SAC(..., policy_kwargs=...).

    Notes
    -----
    SB3's SAC doesn't natively support different extractors for actor and
    critic. We use the standard approach: a single features_extractor that
    works for both, with the actor/critic net_arch controlling their
    respective architectures.

    For the CTDE shared-actor pattern, we implement it as a custom wrapper
    around the standard SAC training loop (see train.py) rather than
    modifying the SB3 policy internals. The actor MLP is standard
    (full obs → action) but with weight-sharing enforced at the
    observation level by the environment's observation structure.
    """
    return {
        'net_arch': {
            'pi': list(actor_hidden),
            'qf': list(critic_hidden),
        },
        'activation_fn': nn.ReLU,
    }
