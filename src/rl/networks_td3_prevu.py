# src/rl/networks_td3_prevu.py  v2.22  (Markov-r5 / prev_u: 9-feature TD3 network)
# -----------------------------------------------------------------------------
# 9-feature TD3 VDN actor + critic for IrrigationEnvPrevU, which appends the
# previous applied control u_{t-1} as a 9th per-agent observation feature so the
# delta-u smoothing reward r5 becomes Markov (the agent can finally SEE the
# quantity its smoothness penalty depends on).
#
# This module subclasses the v2.19-TD3 actor (_TD3SharedActor) and the v2.11
# LayerNorm VDN critic (_V211FactorizedQNet / _V211FactorizedContinuousCritic)
# and overrides ONLY the per-agent dimensions:
#     per-agent features      8 -> 9   (N_AGENT_FEATURES)
#     actor per-agent input  65 -> 66  (PER_AGENT_INPUT_DIM      = 9 + 57 globals)
#     critic per-agent input 66 -> 67  (PER_AGENT_CRITIC_INPUT_DIM = 9 + 57 + 1 action)
#     full obs dim         1097 -> 1227
# The architecture (per-agent shared MLP, 2x-1 re-centering, agent-major reshape,
# VDN sum, twin-Q, LayerNorm-after-Linear) is byte-identical to v2.21c -- only the
# input width changes. The 9-feature constants already exist in networks.py; we
# do NOT touch the shared V27_* (8-feature) constants or classes, so the SAC
# family and the v2.20/v2.21c 8-feature TD3 path are completely unaffected.
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim

from src.rl.networks import (
    N_AGENT_FEATURES,             # 9
    PER_AGENT_INPUT_DIM,          # 66  (= 9 + 57)
    PER_AGENT_CRITIC_INPUT_DIM,   # 67  (= 9 + 57 + 1)
    _V211FactorizedQNet,
    _V211FactorizedContinuousCritic,
)
from src.rl.networks_td3 import _TD3SharedActor, TD3VDNPolicy, make_td3_policy_kwargs

__all__ = ["TD3VDNPolicyPrevU", "make_td3_policy_kwargs"]


class _TD3SharedActorPrevU(_TD3SharedActor):
    """v2.21c deterministic VDN actor with 9 per-agent features (adds u_{t-1})."""
    _N_AGENT_FEATURES    = N_AGENT_FEATURES        # 9
    _PER_AGENT_INPUT_DIM = PER_AGENT_INPUT_DIM     # 66


class _V211FactorizedQNetPrevU(_V211FactorizedQNet):
    """v2.11 LayerNorm VDN Q-net with 9 per-agent features (critic input 67)."""
    _N_AGENT_FEATURES           = N_AGENT_FEATURES            # 9
    _PER_AGENT_CRITIC_INPUT_DIM = PER_AGENT_CRITIC_INPUT_DIM  # 67


class _V211FactorizedContinuousCriticPrevU(_V211FactorizedContinuousCritic):
    """LayerNorm twin-Q VDN critic with 9 per-agent features."""
    _N_AGENT_FEATURES = N_AGENT_FEATURES            # 9
    _QNET_CLS         = _V211FactorizedQNetPrevU


class TD3VDNPolicyPrevU(TD3VDNPolicy):
    """TD3 VDN policy for the 9-feature prev_u observation (1227-dim).

    Identical to TD3VDNPolicy except it builds the 9-feature actor + critic.
    policy_kwargs are the same as the 8-feature policy (the feature count lives
    in the actor/critic classes, not the kwargs), so reuse make_td3_policy_kwargs.
    """

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> _TD3SharedActorPrevU:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["N"] = get_action_dim(self.action_space)
        return _TD3SharedActorPrevU(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> _V211FactorizedContinuousCriticPrevU:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCriticPrevU(**critic_kwargs).to(self.device)
