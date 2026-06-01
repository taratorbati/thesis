# src/rl/networks_td3.py  v2.19-TD3
# -----------------------------------------------------------------------------
# TD3 (Twin Delayed DDPG) network definitions for the irrigation controller.
#
# WHY (carried from the v2.14-v2.18 post-mortem):
#   SAC's entropy term adds a tanh-Jacobian gradient that pulls the actor mean
#   toward the centre of the action range (6 mm/day, the [-1,1]->[0,1] midpoint
#   x 12).  Reducing alpha (v2.17 alpha=0.005, v2.18 alpha=0.002) progressively
#   released this "action floor" and drove wet-year x1 from 152 -> 144 -> 136
#   (MPC: 130), but at alpha=0.002 the critic's bias-ratio log showed it was
#   becoming poorly calibrated (q_pred_mean briefly negative at 100-125k before
#   recovering).  TD3 removes the entropy objective entirely (deterministic
#   policy) and replaces the smoothing that entropy implicitly provided with
#   explicit TARGET-POLICY SMOOTHING (Fujimoto et al. 2018), which is a cleaner
#   stabiliser for a boundary-located optimum.  Exploration, which SAC got from
#   policy stochasticity, is provided in TD3 by explicit action noise at
#   collection time.
#
# DESIGN PRINCIPLE: reuse everything that works, change only the actor.
#   * CRITIC: byte-identical to v2.11/v2.16 (_V211FactorizedContinuousCritic --
#     VDN sum over 130 per-cell Q-nets, twin-Q, LayerNorm per Yue 2023).  TD3
#     and SAC both consume SB3's ContinuousCritic interface, so the cascade-
#     suppression machinery transfers unchanged.  THIS IS DELIBERATE: the
#     LayerNorm + twin-min that kept v2.16-v2.18 from the v2.7 cascade are
#     exactly what TD3 also relies on for critic stability.
#   * ACTOR: deterministic.  Keeps the IDENTICAL feature pipeline as the v2.16
#     SAC actor -- shared per-agent MLP (LeakyReLU), the 2x-1 input re-centering,
#     the agent-major reshape -- but replaces the (mu, log_std) + squashed-
#     Gaussian head with a single mu head + tanh.  So the only behavioural
#     change vs v2.16 is: deterministic output, no entropy.
#
# ACTOR OUTPUT CONVENTION (must match SB3 TD3Policy):
#   SB3's TD3 Actor outputs a SQUASHED action in [-1, 1] (tanh, squash_output=
#   True) and the POLICY scales [-1,1] -> action_space at predict time via
#   scale/unscale.  Our env action space is Box[0,1] and water = action * 12.
#   So this actor returns tanh(mu(latent)) in [-1, 1]; SB3 maps it to [0,1].
#   0 mm/day <=> tanh=-1 <=> latent->-inf (boundary); 6 mm/day <=> tanh=0.
#   Without the entropy Jacobian pulling toward tanh=0, the actor can now reach
#   the tanh=-1 boundary (0 mm) that SAC was structurally discouraged from.
#
# MARKER: actor.obs_norm_marker = 2.19 (distinct from 2.16 SAC family).  The
#   runner dispatches TD3 checkpoints by (marker >= 2.185 AND no log_std key).
#
# State-dict keys (for runner detection):
#   SAC actor:  actor.latent_pi.*, actor.mu.{w,b}, actor.log_std.{w,b}, actor.obs_norm_marker
#   TD3 actor:  actor.latent_pi.*, actor.mu_head.{w,b},  (NO log_std),    actor.obs_norm_marker
#   The ABSENCE of actor.log_std.weight is the unambiguous TD3 signal.
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

from src.rl.networks import (
    N_GLOBAL_DIMS,
    V27_N_AGENT_FEATURES,
    V27_PER_AGENT_INPUT_DIM,
    _V211FactorizedContinuousCritic,
)

# v2.19-TD3 uses the v2.7/v2.11 observation layout (8 features/agent, 1097-dim,
# 65-dim per-agent actor input) -- identical to the v2.16 SAC family.
TD3_N_AGENT_FEATURES    = V27_N_AGENT_FEATURES        # 8
TD3_PER_AGENT_INPUT_DIM = V27_PER_AGENT_INPUT_DIM     # 65
TD3_OBS_MARKER          = 2.19


class _TD3SharedActor(TD3Actor):
    """Deterministic SAC-equivalent actor with parameter-sharing across N cells.

    Mirrors the v2.16 SAC actor's feature pipeline EXACTLY:
      * agent-major reshape into (B*N, 65) per-agent inputs,
      * 2x - 1 input re-centering (the v2.13 dead-ReLU capacity fix),
      * shared LeakyReLU MLP (latent_pi),
    then replaces the (mu, log_std) + SquashedDiagGaussian head with a single
    deterministic mu head followed by tanh, producing one squashed action per
    cell in [-1, 1].  The N per-cell outputs are concatenated to a (B, N) action.

    We subclass SB3's TD3Actor for interface compatibility (predict/scaling) but
    REPLACE its `self.mu` Sequential with our own `latent_pi` + `mu_head`, and
    override `forward` so SB3's monolithic-net assumption is never used.
    """

    _N_AGENT_FEATURES    = TD3_N_AGENT_FEATURES
    _PER_AGENT_INPUT_DIM = TD3_PER_AGENT_INPUT_DIM

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
        # Force LeakyReLU to match the v2.16 SAC actor (units that drift
        # negative still pass gradient).  The SAC family used LeakyReLU in the
        # actor and ReLU in the critic; we keep that split.
        activation_fn = nn.LeakyReLU

        # SB3's TD3Actor.__init__ builds self.mu = Sequential(create_mlp(...,
        # squash_output=True)).  We let it build (so all the SB3 plumbing --
        # action scaling, device, etc. -- is set up), then OVERWRITE self.mu
        # with an Identity and install our own per-agent latent_pi + mu_head.
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
            raise ValueError(f"_TD3SharedActor: action_dim must equal N (={N}), "
                             f"got {action_dim}")
        expected_obs_dim = self._N_AGENT_FEATURES * N + N_GLOBAL_DIMS
        if features_dim != expected_obs_dim:
            raise ValueError(
                f"_TD3SharedActor: features_dim must equal "
                f"{self._N_AGENT_FEATURES}*{N} + {N_GLOBAL_DIMS} = {expected_obs_dim}, "
                f"got {features_dim}. Check gym_env observation layout.")

        net_arch_list = net_arch if net_arch is not None else [128, 128]

        # Replace SB3's monolithic self.mu (which maps features_dim -> action_dim)
        # with a no-op; our forward never calls it.  Removing the parameters
        # avoids dead weights in the state dict.
        self.mu = nn.Identity()

        # Per-agent shared trunk (no output layer; output_dim=-1) + single mu head.
        latent_layers = create_mlp(
            input_dim=self._PER_AGENT_INPUT_DIM,
            output_dim=-1,
            net_arch=net_arch_list,
            activation_fn=activation_fn,
        )
        self.latent_pi = nn.Sequential(*latent_layers)
        last_dim = net_arch_list[-1] if net_arch_list else self._PER_AGENT_INPUT_DIM
        self.mu_head = nn.Linear(last_dim, 1)

        # Marker buffer (mirrors the SAC family) so the runner can dispatch.
        self.register_buffer("obs_norm_marker", torch.tensor([TD3_OBS_MARKER]))

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Agent-major reshape + 2x-1 re-center, identical to the v2.16 actor."""
        B = features.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES
        per_agent = features[:, : F * N].reshape(B, N, F)
        global_block = features[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        # v2.13 actor-only re-center: [0, 1.x] -> [-1, +1.x].
        combined = 2.0 * combined - 1.0
        return combined.reshape(B * N, self._PER_AGENT_INPUT_DIM)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic squashed action in [-1, 1], shape (B, N).

        SB3's TD3 wraps this output and (at predict time) scales [-1,1] ->
        action_space.  During training SB3 uses this raw [-1,1] output for both
        the policy loss and (with added clipped noise) the target action.
        """
        features = self.extract_features(obs, self.features_extractor)
        B = features.shape[0]
        per_agent_input = self._per_agent_features(features)
        latent = self.latent_pi(per_agent_input)
        pre_tanh = self.mu_head(latent).reshape(B, self.N)
        return torch.tanh(pre_tanh)


class TD3VDNPolicy(TD3Policy):
    """TD3 policy: deterministic VDN-shared actor + v2.11 LayerNorm VDN critic.

    The actor is _TD3SharedActor (deterministic, marker=2.19).  The critic is
    _V211FactorizedContinuousCritic -- byte-identical to the one used by the
    v2.16/v2.17/v2.18 SAC runs (VDN sum, twin-Q, LayerNorm).  Reusing it means
    the cascade-suppression behaviour is preserved and TD3's twin-min target is
    computed over the same factorised Q as before.
    """

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> _TD3SharedActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["N"] = get_action_dim(self.action_space)
        return _TD3SharedActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> _V211FactorizedContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**critic_kwargs).to(self.device)


def make_td3_policy_kwargs(
    N: int = 130,
    actor_hidden=(128, 128),
    critic_hidden=(256, 256),
):
    """policy_kwargs for SB3 TD3 with the v2.19 CTDE VDN architecture.

    Use with policy_class=TD3VDNPolicy in the TD3 constructor.  Note: TD3 has no
    entropy/log-std, and SB3's TD3 net_arch may be a list or a dict; we pass a
    dict with pi/qf to mirror the SAC builder and keep the trunk widths explicit.
    """
    return {
        "net_arch": {"pi": list(actor_hidden), "qf": list(critic_hidden)},
        # Critic uses ReLU (matches the SAC family's critic); the actor forces
        # LeakyReLU internally regardless of this value.
        "activation_fn": nn.ReLU,
    }
