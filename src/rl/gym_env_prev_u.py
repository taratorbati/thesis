# src/rl/gym_env_prev_u.py  v2.20.0  (Stage 2 component -- OFF during Stage 1)
# -----------------------------------------------------------------------------
# IrrigationEnvPrevU: IrrigationEnv + the previous applied control u_{t-1} as an
# extra per-agent observation feature.  Place in src/rl/ alongside gym_env.py.
#
# WHY (Markov-completeness for the delta-u reward r5)
# ---------------------------------------------------
# r5 = -alpha5 * mean_n[ ((u_t - u_{t-1}) / UB_MM)^2 ] penalises day-to-day
# control change, but u_{t-1} is NOT in the base observation.  A deterministic
# bootstrapped actor therefore cannot see the quantity its own smoothness
# penalty depends on, so the MDP is not Markov w.r.t. r5 and the policy cannot
# learn to be smooth in the way MPC is (MPC optimises the whole trajectory and
# hits mean|du|~=0.98; RL with r5 at alpha5=0.005 sat at ~2.5).  Standard fix:
# put the missing state variable into the observation (Sutton & Barto, MDP
# state sufficiency; cf. action-history augmentation in control RL).
#
# LAYOUT
# ------
# Base obs (use_overshoot_feature=False) is agent-major:
#     [a0_f0..a0_f7, a1_f0..a1_f7, ..., a129_f0..a129_f7,  <57 global dims>]
# This wrapper inserts a 9th per-agent feature, prev_u_norm = clip(u_{t-1}/UB_MM,
# 0, 1) (zeros on the first day of a season, when _prev_irr_mm is None):
#     [a0_f0..a0_f7,a0_prevu, a1_f0..a1_f7,a1_prevu, ..., <57 global dims>]
# Obs dim 1097 -> 1227, per-agent feature count 8 -> 9.
#
# *** REQUIRED MATCHING NETWORK CHANGE (this wrapper is NOT sufficient alone) ***
# ------------------------------------------------------------------------------
# networks_td3.py hard-codes the per-agent feature count for BOTH the actor and
# the critic:
#     TD3_N_AGENT_FEATURES    = V27_N_AGENT_FEATURES     = 8     (networks_td3.py)
#     TD3_PER_AGENT_INPUT_DIM = V27_PER_AGENT_INPUT_DIM  = 65    (= 8 + 57)
# and _TD3SharedActor.__init__ asserts features_dim == 8*N + 57 = 1097, then
# reshapes features[:, :8*N] -> (B, N, 8).  Feeding this env's 1227-dim obs to
# the stock network will (a) trip that assert, or (b) mis-slice the agent block.
# To ACTIVATE prev_u you must also provide a 9-feature actor + critic, e.g. a
# small variant that sets _N_AGENT_FEATURES=9 and _PER_AGENT_INPUT_DIM=66 (=9+57)
# and the matching features_dim assert (1227).  Do NOT edit the shared V27_*
# constants in networks.py in place -- they are reused by the v2.16-v2.18 SAC
# family.  This is the Stage-2 task; until it is done, keep expose_prev_u=False
# (the v2.20 trainer raises a clear error if it is True).
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from src.rl.gym_env import IrrigationEnv, N_AGENTS, UB_MM


class IrrigationEnvPrevU(IrrigationEnv):
    """IrrigationEnv with u_{t-1} appended as a per-agent observation feature."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Base per-agent feature count (8 unless the overshoot feature is on).
        self._n_feat_base = 9 if self._use_overshoot_feature else 8

        # Remember the base (parent-built) observation space; the parent's
        # _build_obs asserts against self.observation_space.shape, so we restore
        # this one around the super() call below.
        self._base_obs_space = self.observation_space
        base_dim = int(self._base_obs_space.shape[0])

        # Expanded space: +1 feature per agent.
        self._full_obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + N_AGENTS,), dtype=np.float32,
        )
        self.observation_space = self._full_obs_space

    def _build_obs(self) -> np.ndarray:
        # Build the base observation under the BASE space so the parent's
        # internal shape assert passes, then restore the expanded space.
        self.observation_space = self._base_obs_space
        try:
            base = super()._build_obs()
        finally:
            self.observation_space = self._full_obs_space

        N = N_AGENTS
        nf = self._n_feat_base
        split = N * nf

        agent_block = base[:split].reshape(N, nf)   # (N, nf)  agent-major
        rest = base[split:]                          # 57 global dims

        if self._prev_irr_mm is None:
            prev_feat = np.zeros((N, 1), dtype=base.dtype)
        else:
            prev_feat = np.clip(
                np.asarray(self._prev_irr_mm, dtype=base.dtype) / UB_MM, 0.0, 1.0,
            ).reshape(N, 1)

        agent_block = np.concatenate([agent_block, prev_feat], axis=1)  # (N, nf+1)
        obs = np.concatenate([agent_block.reshape(-1), rest]).astype(base.dtype)

        assert obs.shape == self._full_obs_space.shape, (
            f"prev_u obs shape {obs.shape}, expected {self._full_obs_space.shape}"
        )
        return obs
