# src/rl/networks_tqc.py  v2.10.0 (E2)
# -----------------------------------------------------------------------------
# Distributional VDN critic and matching policy for sb3_contrib.TQC.
#
# Architecture (v2.7 baseline, distributionally extended):
#   - Actor: _V27SharedActor (8 features/agent, 65-dim per-agent input).
#            Reused verbatim from networks.py - TQC's actor interface is
#            identical to SAC's.
#   - Critic: _V27FactorizedTQCCritic - twin critic, each critic outputs
#            a 25-quantile vector via VDN sum across the 130 agents:
#               Z_total(s, a)[i] = SUM_n  Z_local(s_n, g, a_n)[i]
#                                                    for i in 0..n_quantiles-1
#            Output shape per forward call: (batch, n_critics, n_quantiles)
#            i.e. (B, 2, 25) - this is the shape TQC's train() expects.
#
# This is a faithful distributional extension of Sunehag et al. (2018) VDN
# into the quantile setting:
#   - Reward is sum-decomposable across agents (biomass mean, water cost
#     sum, drought mean, overFC mean), so a sum-decomposable joint Q-value
#     is well-motivated.
#   - The per-quantile sum produces a joint quantile function that is
#     exact when agents are comonotonic and a contraction-stable
#     approximation otherwise.  The TQC truncation step produces a
#     conservative downward bias in the target regardless of this
#     approximation, which is the property that breaks the v2.7
#     deadly-triad cascade.
#
# Hyperparameter defaults follow Kuznetsov et al. 2020 ("Controlling
# Overestimation Bias With Truncated Mixture of Continuous Distributional
# Quantile Critics", ICML 2020):
#   n_quantiles = 25       (paper Section 5.2, default for continuous control)
#   n_critics   = 2        (matches v2.7 twin-Q)
#   top_quantiles_to_drop_per_net = 5  (paper Section 5.2, drops 20%)
#
# Compatibility:
#   - share_features_extractor=True (parity with v2.7 FactorizedContinuousCritic)
#   - obs layout: v2.7 (8 features/agent, 1097-dim total)
#   - For loading: TQC checkpoints have different state-dict keys from SAC
#     checkpoints; this module is NOT a legacy loader for v2.7 SAC weights.
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import BaseFeaturesExtractor, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp

from sb3_contrib.tqc.policies import TQCPolicy

# Reuse v2.7 architectural constants and actor verbatim - no changes.
from src.rl.networks import (
    V27_N_AGENT_FEATURES,
    V27_OBS_DIM,
    V27_PER_AGENT_INPUT_DIM,
    V27_PER_AGENT_CRITIC_INPUT_DIM,
    N_AGENTS_DEFAULT,
    N_GLOBAL_DIMS,
    _V27SharedActor,
)


# -----------------------------------------------------------------------------
# Defaults from Kuznetsov et al. 2020.
# -----------------------------------------------------------------------------
TQC_N_QUANTILES_DEFAULT = 25
TQC_N_CRITICS_DEFAULT   = 2


# -----------------------------------------------------------------------------
# Per-critic quantile Q-net (one of n_critics inside the twin critic).
# -----------------------------------------------------------------------------
class _V27FactorizedTQCQNet(nn.Module):
    """Quantile-output VDN Q-net.

    Maps the 1097-dim observation and the 130-dim action to an
    n_quantiles-dim quantile vector via per-agent decomposition:

        local_input(n) = [local_obs(n) (8d), global_block (57d), action(n) (1d)]
                         in R^66
        local_quantiles(n) = MLP(local_input(n))   in R^n_quantiles
        Z_total = SUM_n local_quantiles(n)         in R^n_quantiles

    Parameters
    ----------
    N : int
        Number of agents (130 for the Gilan paddy).
    net_arch : list[int]
        Hidden widths of the per-agent MLP (default [256, 256]).
    n_quantiles : int
        Number of quantile predictions to produce (Kuznetsov 2020: 25).
    activation_fn : type[nn.Module]
        Activation between hidden layers (default ReLU - matches v2.7).
    """

    _N_AGENT_FEATURES           = V27_N_AGENT_FEATURES
    _PER_AGENT_CRITIC_INPUT_DIM = V27_PER_AGENT_CRITIC_INPUT_DIM

    def __init__(
        self,
        N: int,
        net_arch: List[int],
        n_quantiles: int = TQC_N_QUANTILES_DEFAULT,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.N = N
        self.n_quantiles = n_quantiles
        layers = create_mlp(
            input_dim=self._PER_AGENT_CRITIC_INPUT_DIM,
            output_dim=n_quantiles,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        self.local_q_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute the joint quantile vector Z_total(s, a).

        Parameters
        ----------
        obs : Tensor of shape (B, V27_OBS_DIM)
        actions : Tensor of shape (B, N)

        Returns
        -------
        Tensor of shape (B, n_quantiles)
        """
        B = obs.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES

        local_obs       = obs[:, : F * N].reshape(B, N, F)
        global_block    = obs[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
        local_actions   = actions.reshape(B, N, 1)

        local_inputs = torch.cat(
            [local_obs, global_expanded, local_actions], dim=-1
        )  # shape: (B, N, 66)

        local_inputs_flat = local_inputs.reshape(
            B * N, self._PER_AGENT_CRITIC_INPUT_DIM
        )
        local_quantiles = self.local_q_net(local_inputs_flat)  # (B*N, n_q)
        local_quantiles = local_quantiles.reshape(B, N, self.n_quantiles)

        # VDN aggregation: sum elementwise per quantile across the 130 agents.
        z_total = local_quantiles.sum(dim=1)  # (B, n_q)
        return z_total


# -----------------------------------------------------------------------------
# Twin critic wrapper - the object TQC's train() interacts with.
# -----------------------------------------------------------------------------
class _V27FactorizedTQCCritic(BaseModel):
    """Twin VDN-per-quantile critic for TQC.

    Mirrors sb3_contrib.tqc.policies.Critic's external contract:
        - self.n_critics : int
        - self.n_quantiles : int
        - self.quantiles_total : int  (= n_critics * n_quantiles)
        - self.q_networks : list[nn.Module]  (length n_critics)
        - forward(obs, actions) -> Tensor of shape (B, n_critics, n_quantiles)

    TQC.train() reads `critic.quantiles_total` and constructs the
    truncated target as
        n_target_quantiles = quantiles_total - top_quantiles_to_drop_per_net
                                              * n_critics
        next_quantiles_sorted = sort(reshape(B, n_critics * n_quantiles))
        target = next_quantiles_sorted[:, :n_target_quantiles]
    so our forward shape matches exactly what TQC.train expects.
    """

    _N_AGENT_FEATURES = V27_N_AGENT_FEATURES
    _QNET_CLS         = _V27FactorizedTQCQNet

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = False,
        n_critics: int = TQC_N_CRITICS_DEFAULT,
        n_quantiles: int = TQC_N_QUANTILES_DEFAULT,
        share_features_extractor: bool = True,
        N: int = N_AGENTS_DEFAULT,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        self.n_critics       = n_critics
        self.n_quantiles     = n_quantiles
        self.quantiles_total = n_critics * n_quantiles
        self.N               = N

        expected_obs_dim = self._N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"{type(self).__name__}: features_dim must equal "
                f"{expected_obs_dim} (v2.7 obs layout), got {features_dim}."
            )

        action_dim = get_action_dim(action_space)
        if action_dim != N:
            raise ValueError(
                f"{type(self).__name__}: action_dim must equal N (={N}), "
                f"got {action_dim}."
            )

        # Twin quantile Q-networks.
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net = self._QNET_CLS(
                N=N,
                net_arch=net_arch,
                n_quantiles=n_quantiles,
                activation_fn=activation_fn,
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the twin quantile predictions.

        Returns
        -------
        Tensor of shape (B, n_critics, n_quantiles)
        """
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        # Stack the n_critics outputs along a new dim=1 axis.
        per_critic = [q_net(features, actions) for q_net in self.q_networks]
        # Each per_critic[i] is (B, n_quantiles); stack to (B, n_critics, n_q).
        return torch.stack(per_critic, dim=1)


# -----------------------------------------------------------------------------
# Policy class - the entry point sb3_contrib.TQC uses.
# -----------------------------------------------------------------------------
class V27TQCPolicy(TQCPolicy):
    """TQC policy with v2.7 CTDE VDN architecture (quantile-extended).

    - Actor: _V27SharedActor (8 features/agent, parameter-shared across 130
      agents).  Identical to v2.7.
    - Critic: _V27FactorizedTQCCritic (twin VDN-per-quantile critic).
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V27SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        # n_quantiles and n_critics arrive in critic_kwargs via policy_kwargs.
        return _V27FactorizedTQCCritic(**kw).to(self.device)


# -----------------------------------------------------------------------------
# Convenience builder for policy_kwargs.
# -----------------------------------------------------------------------------
def make_tqc_policy_kwargs(
    N: int = N_AGENTS_DEFAULT,
    actor_hidden: Tuple[int, ...] = (128, 128),
    critic_hidden: Tuple[int, ...] = (256, 256),
    n_quantiles: int = TQC_N_QUANTILES_DEFAULT,
    n_critics: int = TQC_N_CRITICS_DEFAULT,
    share_features_extractor: bool = True,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """policy_kwargs for sb3_contrib.TQC with the v2.7 CTDE VDN architecture.

    Pass to TQC as:
        model = TQC(
            policy=V27TQCPolicy,
            env=...,
            policy_kwargs=make_tqc_policy_kwargs(),
            ...
        )

    Notes
    -----
    `share_features_extractor=True` mirrors v2.7's FactorizedContinuousCritic
    setting.  For vector observations the features extractor is FlattenExtractor
    (a no-op), so the practical impact is just a small memory saving.

    n_quantiles and n_critics are picked up by TQCPolicy.__init__ as top-level
    kwargs and propagated into critic_kwargs by sb3_contrib.
    """
    kwargs: Dict[str, Any] = {
        "net_arch": {
            "pi": list(actor_hidden),
            "qf": list(critic_hidden),
        },
        "activation_fn": nn.ReLU,
        "n_quantiles": n_quantiles,
        "n_critics": n_critics,
        "share_features_extractor": share_features_extractor,
    }
    if optimizer_kwargs is not None:
        kwargs["optimizer_kwargs"] = optimizer_kwargs
    return kwargs
