# src/rl/networks.py  v2.11.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.8.0  (see change_spec_v211.md for full rationale)
#
#   v2.11 ADDITION — LayerNorm-regularised VDN critic for cascade prevention.
#
#   Motivation:
#     v2.7 SAC exhibits a deadly-triad cascade at step ~155k-170k in which the
#     critic loss diverges geometrically (~6-10× per 10k steps) and the actor
#     collapses to a near-uniform policy.  The v2.10 study (E2/E3/E4) showed
#     that quantile truncation is structurally inert under VDN-sum (E2), naïve
#     n-step buffers break the soft-Bellman target (E3), and γ-reduction stops
#     the cascade but breaks credit assignment for the 93-day task (E4).
#
#     Yue et al. NeurIPS 2023 ("Understanding, Predicting and Better Resolving
#     Q-Value Divergence in Offline-RL", arXiv:2310.04411) prove via Neural
#     Tangent Kernel analysis that LayerNorm in the critic's hidden layers
#     suppresses the Self-Excite Eigenvalue Measure (SEEM) and reliably
#     prevents Q-divergence with no detrimental bias on the learned policy.
#     Nauman et al. RLC 2024 ("Dissecting Deep RL with High Update Ratios:
#     Combatting Value Overestimation and Divergence", arXiv:2403.05996)
#     confirm the same effect in *online* RL on the dm_control suite.
#
#     v2.11 adds LayerNorm after each hidden Linear layer in the critic
#     (Yue 2023 placement, before the activation) while preserving the v2.7
#     observation layout (8 features/agent, 1097-dim obs) and all other
#     hyperparameters (γ=0.99, α=0.05 fixed, τ=0.005, LR 3e-4→5e-5, 250k
#     steps).  This is the single experimental variable.
#
#   New names exported:
#     _V211FactorizedQNet              — per-quantile MLP + LayerNorm
#     _V211FactorizedContinuousCritic  — twin-Q v2.11 critic
#     V211CTDESACPolicy                — SACPolicy for the v2.11 critic
#     V211_*                            — dimension constants (mirror V27_*)
#
#   Backwards compatibility:
#     - All v2.6 / v2.7 / v2.8 / v2.10 critic classes and policy classes are
#       UNCHANGED.  Their state-dict keys are unchanged.
#     - The v2.11 critic uses the same v2.7 obs layout (8 features/agent,
#       1097-dim total, 66-dim per-agent critic input), so the runner's
#       observation builder needs no changes.
#     - The v2.11 state dict contains LayerNorm parameter keys
#       (critic.qf{0,1}.{1,4}.{weight,bias}) absent in v2.7.  The runner's
#       _detect_critic_arch is updated (in src/rl/runner.py) to detect this
#       and dispatch to V211CTDESACPolicy.
#
#   v2.8 changes from v2.7.0  (preserved, see change_spec_v28.md):
#     1. CURRENT (v2.8) per-agent feature count: 8 → 9.
#          The v2.8 obs adds x1_overshoot_norm as a 9th per-agent feature.
#          The actor and critic first-layer widths grow accordingly:
#              PER_AGENT_INPUT_DIM        : 65 → 66
#              PER_AGENT_CRITIC_INPUT_DIM : 66 → 67
#              OBS_DIM_DEFAULT            : 1097 → 1227
#          Hidden widths, activation, twin-Q architecture: unchanged.
#
#     2. LEGACY CHECKPOINT LOADING PRESERVED for both v2.7 and v2.6.
#          - v2.7 best_model.zip (8 features/agent, dim=66 critic input):
#            loaded via _V27SharedActor + _V27FactorizedQNet.
#          - v2.6 best_model.zip (5 features/agent, dim=63 critic input,
#            local_q_net wrapper): loaded via _LegacySharedActor +
#            _WrappedFactorizedCritic.
#          - pre-VDN monolithic (837-dim flat): MonolithicCTDESACPolicy.
#
# Critic checkpoint variants (loaded by runner.py):
#   dim=66, flat, LN   → V211CTDESACPolicy        (v2.11 — LayerNorm critic, NEW)
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
# v2.11 dimensions  (LayerNorm critic, otherwise same layout as v2.7)
# Per-agent block: 8 features, same as v2.7.  Only the critic's internal
# architecture differs (LayerNorm inserted after each hidden Linear).  The
# actor is identical to v2.7's _V27SharedActor.
# ─────────────────────────────────────────────────────────────────────────────
V211_N_AGENT_FEATURES           = V27_N_AGENT_FEATURES                                       # 8
V211_OBS_DIM                    = V27_OBS_DIM                                                # 1097
V211_PER_AGENT_INPUT_DIM        = V27_PER_AGENT_INPUT_DIM                                    # 65
V211_PER_AGENT_CRITIC_INPUT_DIM = V27_PER_AGENT_CRITIC_INPUT_DIM                             # 66

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


class _V212SharedActor(SharedActor):
    """SharedActor for v2.12 (8 per-agent features, 65-dim input).

    Two changes vs _V27SharedActor, both targeting the v2.11 dead-ReLU collapse:

      1. The actor's hidden MLP uses LeakyReLU(0.01) instead of ReLU, so a
         hidden unit whose pre-activation drifts negative still passes a small
         gradient and can recover, instead of dying permanently.  (The critic
         keeps plain ReLU + LayerNorm — this change is actor-only.)

      2. A non-learnable buffer ``obs_norm_marker`` is registered so that the
         evaluation runner can detect, purely from the saved ``policy.pth``
         state-dict, that this checkpoint was trained on the v2.12 normalised
         global/forecast observation block (rainfall/Kc_ET/radiation divided by
         physical references in gym_env.py).  The runner must apply the *same*
         normalisation at eval time or the actor sees an out-of-distribution
         observation.  v2.7/v2.11 checkpoints have no such key.

    The per-agent feature count, input dim, hidden widths, mu/log_std heads,
    action distribution and reshape conventions are all identical to v2.7.
    """
    _N_AGENT_FEATURES    = V27_N_AGENT_FEATURES
    _PER_AGENT_INPUT_DIM = V27_PER_AGENT_INPUT_DIM

    def __init__(self, *args, **kwargs):
        # Force LeakyReLU for the actor's hidden activation, regardless of the
        # activation_fn that flows in from policy_kwargs (which controls the
        # critic).  This keeps the critic on ReLU while the actor uses LeakyReLU.
        kwargs["activation_fn"] = nn.LeakyReLU
        super().__init__(*args, **kwargs)
        # Detection marker (value is informational only; presence is the signal).
        self.register_buffer("obs_norm_marker", torch.tensor([2.12]))


class _V213SharedActor(_V212SharedActor):
    """SharedActor for v2.13 (actor-only input re-centering).

    Diagnosis behind v2.13: v2.12 fixed the literal dead-ReLU (LeakyReLU + 
    normalised globals lifted live-unit count from ~0 to ~14/128), but the
    actor still ran at ~11% capacity — 117/128 first-layer units sat in 
    LeakyReLU's negative regime (100x attenuated) on real observations.
    Trace of the cause:

      - The observation block is 98% non-negative features with E[x] = 0.66
        per dim, summing to ~43 across 65 dims.
      - The policy faces strong downward output pressure: the SAC entropy term
        pulls pre-tanh mu toward 0, AND the overshoot penalty r6 (verified
        -44.9/season in wet years, 40x any other term) punishes over-irrigation.
      - With all-positive inputs, the cheapest way to lower output is negative
        weight row-sums.  The optimiser does exactly this: row-sums went from
        ~0 at init to -5.3 by 250k, driving W@E[x] = -0.94, which puts 117/128
        units below zero.  Empirically simulated: under downward output 
        pressure, raw-positive inputs collapse to 0% alive within a few epochs;
        centered inputs stay at ~50%.

    The fix: re-center the actor's input to roughly [-1, +1] by mapping the
    [0, 1.x] per-agent + global block via x' = 2*x - 1, applied as the first
    op inside the actor's `_per_agent_features`.  The transform is fixed (no
    parameters), surgical (actor-only — the critic still sees raw [0, 1.x]
    inputs through its own per-agent path), and adds zero train/eval-skew risk
    (it's a deterministic linear op baked into the model's forward).

    Empirical validation on the v2.12 trained weights:
      - alive units: 11% -> 63.6% (Option C in v2.13 comparison)
      - forecast-rain response (+0.3): +0.001 (broken) -> -0.38 (correct sign)

    The marker is updated to 2.13 so the runner can dispatch to V213CTDESACPolicy.
    No changes to the critic, the obs builder, the reward, the env, or the LR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Overwrite the v2.12 marker with the v2.13 value.  Presence + value
        # let the runner distinguish v2.12 (no re-center) from v2.13 (re-center).
        self.obs_norm_marker.data = torch.tensor([2.13])

    def _per_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Same reshape as the base SharedActor, then re-center to ~[-1, +1].

        The per-agent block is in [0, 1.5] and the global block in [0, ~1];
        2*x - 1 maps them to [-1, 2] and [-1, 1] respectively.  Both blocks
        end up with means close to zero, which is what the dead-ReLU fix needs.

        The critic uses a parallel `_per_agent_features` path on its own
        `obs->per-agent input` pipeline and is NOT touched by this override.
        """
        B = features.shape[0]
        N = self.N
        F = self._N_AGENT_FEATURES
        per_agent = features[:, : F * N].reshape(B, N, F)
        global_block = features[:, F * N:]
        global_expanded = global_block.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([per_agent, global_expanded], dim=-1)
        # v2.13 actor-only re-center: [0, 1.x] -> [-1, +1.x].
        # Centers the input distribution so the downward output pressure
        # (entropy + overshoot penalty) no longer drives weight row-sums
        # uniformly negative and kills the first-layer ReLUs.
        combined = 2.0 * combined - 1.0
        return combined.reshape(B * N, self._PER_AGENT_INPUT_DIM)


class _V214SharedActor(_V213SharedActor):
    """SharedActor for v2.14 (v2.13 architecture, trained with α=0.01).

    v2.14 is architecturally identical to v2.13.  The only difference is the
    training hyperparameter ENT_COEF (= α): v2.13 used 0.05, v2.14 uses 0.01.
    This is a one-variable test of the entropy/reward SNR hypothesis derived
    from the v2.7-200k peak-actor anchor:

        - At v2.7-200k (the responsive-actor snapshot whose policy gave the
          best yields in any v2.* run), the per-agent SNR Q/(α·H) ≈ 290.
        - In v2.12 and v2.13, the SAME SNR is only ~80 because the LayerNorm-
          bounded critic keeps |Q| ≈ 3 per agent (vs 11 at v2.7-200k) while
          the entropy term α·H ≈ 0.039 is identical across all runs.
        - To recover the v2.7-200k SNR ratio in the v2.13 architecture (which
          we cannot raise Q in without re-introducing the deadly-triad
          cascade), the cleanest lever is α: drop it ~4-5× so the entropy
          contribution shrinks proportionally.  α=0.01 lands the SNR back in
          the v2.7-200k regime (Q/αH ≈ 290).

    Why a separate class instead of just a config flag in train_v214:
        The marker buffer (2.14) lets the eval runner identify this checkpoint
        and load it with the right policy class.  No other architecture diff.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v2.14 marker.  Same architecture as v2.13; differentiated only so
        # the runner can recognise α=0.01 checkpoints in saved files.
        self.obs_norm_marker.data = torch.tensor([2.14])


class _V215SharedActor(_V214SharedActor):
    """SharedActor for v2.15 (v2.14 architecture, trained with LINEAR r6).

    v2.15 is architecturally identical to v2.14 (and therefore v2.13).  The
    only difference is the training-time reward function: r6 changes from
    QUADRATIC (-ALPHA6 * mean(overshoot^2) / FC^2) to LINEAR
    (-ALPHA6_LIN * mean(overshoot) / FC) with ALPHA6_LIN=1.5 calibrated
    against v2.14 rollouts to preserve season-sum r6 magnitude.

    Why linear r6 is the next single-variable test:

        v2.14 diagnostic (perfect-forecast wet/100 rollout vs MPC):
          metric              v2.14   MPC
          daily u 10th-pctile  2.50   0.16  mm/day  (cannot push action low)
          corr(u, rain_fwd7)  +0.03  -0.42         (forecast-blind)
          waterlog-days/agent  80.2  19.0          (4x MPC)
          WUE (kg/ha/mm)       8.62  12.12         (12% efficiency gap)

        The actor's primary defect is that it cannot push u toward zero on
        days when rain is coming.  Two compounding causes both addressed by
        a linear r6:
          1. d(quadratic r6)/d(overshoot) = -2*ALPHA6*overshoot/FC^2 is
             small at moderate overshoot (the operating regime).  The
             critic's dQ/du signal at the policy's operating point is weak.
          2. The ABM's waterlog yield-loss term h6 is LINEAR in (x1-FC)/FC,
             but r6 is QUADRATIC -> the reward proxy is misaligned with the
             physical objective.  The critic learns the proxy, not the
             physics.

        Linear r6 gives uniform marginal penalty across the overshoot range
        and aligns r6 with h6.  This is a single-variable test of the
        reward-shape hypothesis: if linearisation closes the wet-year gap,
        the quadratic shape was the bottleneck.  If it doesn't, the next
        suspects are buffer concentration (replay never saw low actions) and
        twin-critic pessimism at OOD actions - addressed by exploration-σ
        boost or by switching min(Q1,Q2) to mean(Q1,Q2).

    Why a separate class instead of just a config flag:
        The marker buffer (2.15) lets the eval runner identify this checkpoint
        and load it with the right policy class.  No other architecture diff.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v2.15 marker.  Architecturally byte-identical to v2.14 / v2.13;
        # marker change only so the runner can recognise r6-linear checkpoints.
        self.obs_norm_marker.data = torch.tensor([2.15])


class _V216SharedActor(_V215SharedActor):
    """SharedActor for v2.16 (v2.15 architecture + tighter rain normaliser
    RAIN_REF=30 + auto-tuned alpha capped at 0.1).

    v2.16 is architecturally identical to v2.15 (and therefore v2.14, v2.13).
    Two training-time changes, both targeting the rain-blindness diagnostic
    on v2.15's 250k policy:

        DIAGNOSTIC (v2.15 250k actor, gradient analysis on realistic obs):
          feature              |dmu/dx_i|   input range   effective sensitivity
          rain forecast (8d)     0.43          0.36           0.16  <-- LOWEST
          ETc  forecast (8d)     1.24          1.44           1.78
          rad  forecast (8d)     1.40          1.56           2.19
          x1 (soil moisture)     2.03          3.00           6.08
        The actor IS responsive to ET and radiation forecasts (effective
        sensitivity > 1.7) but rain has effective sensitivity 0.16 because
        the input dynamic range is compressed: rainfall/70 has median 0.001
        and p99 0.18, so the recentered input occupies only [-1.0, -0.64]
        of the [-1, +1] range.  This is the root cause of the
        corr(u, rain_fwd7) approx +0.03 observed across v2.14 and v2.15.

        Two paired changes in v2.16:

        1. RAIN_REF: 70.0 -> 30.0.  Rationale documented in gym_env.py.
           Effective sensitivity to rain rises from 0.16 to ~0.36 - puts
           rain in the same effective-sensitivity range as today's ET.

        2. ent_coef: 0.01 fixed -> auto-tuned, capped at log(0.1)= -2.303,
           target_entropy = -65 (vs SB3 default -130).  Anchored to the
           original v2.4-v2.6 target_entropy range (-13 to -65) which fits
           the VDN-factorized actor structure better than SB3's default
           (the default was derived for monolithic action spaces, not for
           shared per-agent actors).  log_ent_coef initialized at
           log(0.05) approx -3.0 (v2.14/v2.15 known-stable value), allowed
           to drift up to the 0.1 cap or arbitrarily low.

    Why a separate class instead of just a config flag:
        The marker buffer (2.16) lets the eval runner identify this
        checkpoint AND apply the matching rain normaliser at eval time.
        Same dispatch mechanism as v2.12->v2.15.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v2.16 marker.  Architecturally byte-identical to v2.15; marker change
        # lets the runner recognise checkpoints trained with RAIN_REF=30
        # and capped auto-alpha so eval-time normalisation matches training.
        self.obs_norm_marker.data = torch.tensor([2.16])


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
#  v2.11 FACTORIZED CRITIC  (LayerNorm-regularised, 66-dim per-agent input)
#
#  Architecture identical to _V27FactorizedQNet EXCEPT a LayerNorm is inserted
#  after each hidden Linear layer (before the activation).  This follows the
#  placement recommended by Yue et al. NeurIPS 2023 (Section 4.3) and confirmed
#  online by Nauman et al. RLC 2024.
#
#  Per-quantile (here per-Q) structure for net_arch=[256, 256]:
#      [0] Linear(66, 256)
#      [1] LayerNorm(256)     <- NEW in v2.11
#      [2] ReLU
#      [3] Linear(256, 256)
#      [4] LayerNorm(256)     <- NEW in v2.11
#      [5] ReLU
#      [6] Linear(256, 1)
#
#  State-dict keys: critic.qf{0,1}.{0,3}.{weight,bias} (Linear) and
#                   critic.qf{0,1}.{1,4}.{weight,bias} (LayerNorm).
#  The v2.7 critic has no qf{0,1}.1.weight key (index 1 was ReLU, no params).
#  This is the unique signal the runner uses to dispatch the correct class.
#
#  LayerNorm overhead: ~2 × 256 + 2 × 256 = 1024 extra parameters per Q-net.
#  Twin critic → 2048 extra parameters total.  Trivial compared to the ~85k
#  parameters in the rest of the per-agent MLP.
# ═════════════════════════════════════════════════════════════════════════════
class _V211FactorizedQNet(nn.Sequential):
    """Q_total = Σ_n Q_local(s_n, g, a_n) with LayerNorm-regularised local Q.

    Same per-agent factorization as _V27FactorizedQNet (sum-decomposition over
    the 130 agents), but each per-agent MLP has LayerNorm inserted after each
    hidden Linear, before the activation.  Critically, the LayerNorm is
    applied PER PER-AGENT INPUT (i.e. the (B*N, hidden_dim) tensor flowing
    through the MLP), not across agents — this is the standard interpretation
    of LayerNorm in MLP-based critics.
    """

    _N_AGENT_FEATURES           = V211_N_AGENT_FEATURES
    _PER_AGENT_CRITIC_INPUT_DIM = V211_PER_AGENT_CRITIC_INPUT_DIM

    def __init__(
        self,
        N: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        # SB3 2.6.0's create_mlp natively supports `post_linear_modules`,
        # which are inserted after each Linear and before the activation.
        # This is exactly the Yue et al. 2023 LayerNorm placement.
        layers = create_mlp(
            input_dim=self._PER_AGENT_CRITIC_INPUT_DIM,
            output_dim=1,
            net_arch=net_arch,
            activation_fn=activation_fn,
            post_linear_modules=[nn.LayerNorm],
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
        # LayerNorm applies along the feature dim of (B*N, hidden_dim).
        local_q = nn.Sequential.forward(self, local_inputs_flat).reshape(B, N, 1)
        q_total = local_q.sum(dim=1)
        return q_total


class _V211FactorizedContinuousCritic(FactorizedContinuousCritic):
    """FactorizedContinuousCritic for v2.11 checkpoints (LayerNorm critic)."""
    _N_AGENT_FEATURES = V211_N_AGENT_FEATURES
    _QNET_CLS         = _V211FactorizedQNet


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
#  v2.11 CTDE SAC POLICY  (LayerNorm critic + v2.7 actor)
#
#  This is the policy class used for the v2.11 cascade-prevention experiment.
#  Actor: identical to v2.7 (_V27SharedActor — 8 features/agent, 65-dim per
#         per-agent input).  No LayerNorm in the actor (Yue 2023 and Nauman
#         2024 apply LayerNorm to the critic only; adding it to the actor
#         would alter the policy entropy distribution and conflate effects).
#  Critic: _V211FactorizedContinuousCritic — twin Q, VDN-summed,
#          LayerNorm-regularised per Yue NeurIPS 2023 placement.
# ═════════════════════════════════════════════════════════════════════════════
class V211CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.11 (LayerNorm critic, v2.7 actor and obs layout).

    Drop-in replacement for V27CTDESACPolicy.  The actor and the observation
    layout are unchanged from v2.7; only the critic's hidden-layer
    regularisation differs.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V27SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.12 CTDE SAC POLICY  (LayerNorm critic + LeakyReLU actor + normalised obs)
#
#  v2.12 = v2.11 critic (LayerNorm VDN, cascade-suppressing) PLUS three
#  actor/observation fixes that together cure the v2.11 dead-ReLU collapse:
#
#    (a) Observation fix (in gym_env.py / runner.py): the global scalar block
#        and the 48-dim forecast block are normalised (rainfall/RAIN_REF,
#        Kc_ET/ETC_REF, radiation/RAD_REF) so no raw physical magnitude (~10-32)
#        reaches the actor's first Linear.  This is the root-cause fix: in
#        v2.11 the unnormalised radiation block contributed ≈ -3.2 to every
#        first-layer pre-activation, pushing all 128 ReLU units below zero.
#
#    (b) Actor activation: LeakyReLU(0.01) (in _V212SharedActor) so units that
#        drift negative still pass gradient and can recover.
#
#    (c) Asymmetric learning rate (set in train_v212.py, not here): the actor
#        optimiser uses a higher LR than the critic, compensating for the
#        LayerNorm-bounded critic gradient (the second, independent mechanism
#        that left the E4/γ=0.98 actor barely modulating even with live ReLUs).
#
#  The critic is byte-identical to v2.11 (_V211FactorizedContinuousCritic), so
#  cascade suppression is preserved.  Detection: the actor carries an
#  ``obs_norm_marker`` buffer absent from v2.7/v2.11 checkpoints.
# ═════════════════════════════════════════════════════════════════════════════
class V212CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.12 (LayerNorm critic, LeakyReLU actor, normalised obs).

    Drop-in for V211CTDESACPolicy.  Same observation dimensionality (1097) and
    same critic; the actor uses LeakyReLU and registers a normalisation marker.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V212SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.13 CTDE SAC POLICY  (actor-only input re-centering on top of v2.12)
#
#  v2.13 = v2.12 critic and actor architecture, with ONE additional surgical
#  change: the actor re-centers its per-agent input via x' = 2*x - 1 as the
#  first operation in its forward pass.  This addresses the "actor runs at
#  ~11% capacity" finding (117/128 first-layer units sat below zero in v2.12)
#  by killing the all-positive-input × negative-row-sum geometry that drives
#  the LeakyReLU into its 100x-attenuated regime.
#
#  The critic is byte-identical to v2.11 (LayerNorm VDN, cascade-suppressing)
#  and sees the original [0, 1.x] obs through its OWN per-agent path; only the
#  actor's forward applies the re-center.  Detection signal for the runner:
#  `actor.obs_norm_marker` buffer with value 2.13 (vs 2.12 for v2.12).
# ═════════════════════════════════════════════════════════════════════════════
class V213CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.13 (v2.12 + actor-only input re-centering).

    Drop-in for V212CTDESACPolicy.  Same observation dimensionality (1097),
    same global/forecast normalisation (RAIN_REF/ETC_REF/RAD_REF), same critic
    (V211 LayerNorm), same LeakyReLU actor activation.  The single change is
    that the actor re-centers its input from ~[0, 1] to ~[-1, +1] inside
    `_V213SharedActor._per_agent_features`.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V213SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.14 CTDE SAC POLICY  (v2.13 architecture, trained with α=0.01)
#
#  v2.14 = v2.13 architecture (LayerNorm critic + LeakyReLU actor + actor-only
#  input re-center) PLUS a single training-time change: ENT_COEF lowered from
#  0.05 to 0.01.  This is a one-variable test of the entropy/reward SNR
#  hypothesis derived from the v2.7-200k peak-actor anchor.
#
#  Architecturally, v2.14 is identical to v2.13.  The only diff is the marker
#  buffer (2.14 vs 2.13), present so the runner can identify α=0.01 checkpoints
#  in saved files.
# ═════════════════════════════════════════════════════════════════════════════
class V214CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.14 (v2.13 architecture, trained with α=0.01).

    Architecturally identical to V213CTDESACPolicy.  Uses _V214SharedActor
    only to mark the checkpoint with marker=2.14, so eval-time dispatch can
    tell α=0.01 v2.14 checkpoints apart from α=0.05 v2.13 ones.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V214SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.15 CTDE SAC POLICY  (v2.14 architecture, trained with LINEAR r6)
#
#  v2.15 = v2.14 architecture (LayerNorm critic + LeakyReLU actor + actor input
#  re-center + α=0.01) PLUS a single training-time reward-function change:
#  r6 switches from quadratic to linear in overshoot, with ALPHA6_LIN=1.5
#  calibrated against v2.14 rollouts to preserve season-sum magnitude.
#
#  Architecturally, v2.15 is identical to v2.14 (and therefore v2.13).  The
#  only diff is the marker buffer (2.15) and the IrrigationEnv constructor
#  argument reward_overshoot_mode='linear' passed by train_v215.
# ═════════════════════════════════════════════════════════════════════════════
class V215CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.15 (v2.14 architecture, trained with linear r6).

    Architecturally identical to V214CTDESACPolicy.  Uses _V215SharedActor
    only to mark the checkpoint with marker=2.15, so eval-time dispatch can
    tell r6-linear v2.15 checkpoints apart from r6-quadratic v2.14 ones.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V215SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**kw).to(self.device)


# ═════════════════════════════════════════════════════════════════════════════
#  v2.16 CTDE SAC POLICY  (v2.15 architecture + RAIN_REF=30 + capped auto-alpha)
#
#  v2.16 = v2.15 architecture (LayerNorm critic + LeakyReLU actor + actor
#  input re-center + linear r6) PLUS two training-time changes targeting the
#  rain-blindness diagnostic on v2.15's 250k policy:
#
#    1. IrrigationEnv constructor receives rain_normaliser=RAIN_REF_V216=30.0
#       (was implicit RAIN_REF=70.0 in v2.7-v2.15).  Tripled effective input
#       dynamic range for rain forecast.
#
#    2. SAC ent_coef changes from fixed 0.01 to auto-tuned with cap at 0.1
#       (log_ent_coef <= log(0.1) approx -2.303) and target_entropy=-65
#       (was SB3 default -130, mismatched to VDN-factorized actor structure).
#       log_ent_coef initialized at log(0.05) approx -3.0 (v2.14/v2.15
#       known-stable level), free to descend.
#
#  Architecturally, v2.16 is identical to v2.15 (and v2.14, v2.13).  The
#  only diffs are the marker buffer (2.16), the env constructor args, and
#  the SB3 ent_coef config passed by train_v216.
# ═════════════════════════════════════════════════════════════════════════════
class V216CTDESACPolicy(SACPolicy):
    """CTDESACPolicy for v2.16 (v2.15 architecture + tighter rain ref + capped
    auto-tuned alpha).

    Architecturally identical to V215CTDESACPolicy.  Uses _V216SharedActor
    only to mark the checkpoint with marker=2.16, so eval-time dispatch can
    tell v2.16 checkpoints apart and apply the matching RAIN_REF=30 at eval.
    """

    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V216SharedActor(**kw).to(self.device)

    def make_critic(self, features_extractor=None):
        kw = self._update_features_extractor(self.critic_kwargs, features_extractor)
        kw["N"] = get_action_dim(self.action_space)
        return _V211FactorizedContinuousCritic(**kw).to(self.device)


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
