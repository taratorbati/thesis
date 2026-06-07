# src/rl/nstep_buffer_exact.py  v2.20.0  (Stage 1)
# -----------------------------------------------------------------------------
# EXACT n-step replay buffer for off-policy TD3 (no gamma^1-vs-gamma^n hack).
#
# WHY THIS EXISTS (vs the v2.10 E3 src/rl/nstep_buffer.py)
# -------------------------------------------------------
# The E3 buffer stored the n-step return R_n but let SB3's train() apply gamma^1
# (not gamma^n) to the bootstrap term, because doing it "correctly" was thought
# to require overriding TQC.train().  It adopted that as a documented
# approximation (~7 units on a ~378 target).  It was also bolted onto TQC (whose
# VDN-sum quantiles collapse here) and a stochastic policy (which needs an
# importance-sampling correction for multi-step off-policy returns -- the
# contribution of the SAC(lambda) / Truncated-TD(lambda) paper, never
# implemented).  None of that binds for a DETERMINISTIC TD3 target policy.
#
# THE EXACT-gamma^n TRICK (no train() override needed)
# ----------------------------------------------------
# This buffer accumulates the n-step return with its OWN discount gamma_base:
#     R_n = sum_{k=0}^{n-1} gamma_base^k * r_{t+k}     (truncated at any done)
# and stores (s_t, a_t, R_n, s_{t+n or terminal}, done_any).
#
# The TRAINER sets the *model's* gamma to gamma_base ** n_steps.  SB3's stock
# TD3 target is then, for every sampled transition:
#     target = R_n + (1 - done) * model.gamma * Q(s_{t+n})
#            = R_n + (1 - done) * gamma_base^n * Q(s_{t+n})        <-- EXACT
# This is correct because:
#   * non-terminal (done=0): full window, discount is exactly gamma_base^n;
#   * terminal     (done=1): R_n is truncated at the boundary and (1-done)=0
#     zeroes the bootstrap, so the discount value is irrelevant.
# The critic therefore learns the gamma_base (=0.99) discounted return, so the
# bias_ratio q_pred-vs-realised-return diagnostic stays on the same scale.
#
# WHY n-step ATTACKS THE v2.20 DIVERGENCE
# ---------------------------------------
# The -222 q_pred excursion is bootstrap-horizon amplification: a 93-step
# episode at gamma=0.99 has effective horizon 1/(1-gamma) ~= 100, and the
# 1-step target leans on the critic's own (diverging) estimate every step.
# n-step replaces n of those bootstrap steps with n GROUNDED rewards, shrinking
# the self-referential term to gamma^n and propagating a catastrophic reward to
# the relevant Q in one update instead of n.  The project's own v2.6 (early
# termination -> ~50-step horizon -> |Q| bounded for 500k steps) vs v2.7 (full
# 93-step horizon -> cascade) is the proof that horizon length is the driver.
#
# References
# ----------
# Hessel et al. 2018  "Rainbow", AAAI -- n-step return ablation, Sec.3.
# Fedus et al. 2020   "Revisiting Fundamentals of Experience Replay", ICML --
#                     Sec.4: store (s,a,R_n,s_{t+n},done), train with gamma^n.
# Fujimoto et al. 2018 "Addressing Function Approximation Error in Actor-Critic
#                     Methods" (TD3) -- deterministic target policy.
# Barth-Maron et al. 2018 (D4PG) / Horgan et al. 2018 (Ape-X) -- uncorrected
#                     small-n returns are standard & near-unbiased for
#                     deterministic off-policy actor-critics.
# -----------------------------------------------------------------------------

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer


class NStepReplayBufferExact(ReplayBuffer):
    """Off-policy n-step replay buffer with an exact gamma^n bootstrap discount.

    Drop-in for SB3 via ``replay_buffer_class=NStepReplayBufferExact`` and
    ``replay_buffer_kwargs=dict(n_steps=n, gamma=gamma_base)``.  The TRAINER
    must set the model's ``gamma=gamma_base ** n`` (see module header) so the
    stock TD3 target line computes ``R_n + (1-done)*gamma_base^n*Q`` exactly.

    Parameters
    ----------
    n_steps : int
        Number of steps to combine (n>=1).  n=1 reduces to the standard buffer.
    gamma : float
        Discount used to ACCUMULATE the n-step return R_n.  Must equal the
        environment/return discount (gamma_base, default 0.99) -- NOT the
        model's gamma (which the trainer sets to gamma_base ** n).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_steps: int = 5,
        gamma: float = 0.99,
        handle_timeout_termination: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
        if not (0.0 < float(gamma) <= 1.0):
            raise ValueError(f"gamma (R_n accumulation) must be in (0, 1], got {gamma!r}")
        self.n_steps = int(n_steps)
        self._n_gamma = float(gamma)

        # One pending FIFO per env.  Entry: (obs, action, reward, next_obs,
        # done, info).  When a window is full the oldest entry is flushed as an
        # n-step transition; on done all remaining (shorter) windows are flushed.
        self._pending: List[deque] = [deque() for _ in range(n_envs)]

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Queue one transition per env; flush when the window is full or on done."""
        for env_i in range(self.n_envs):
            self._pending[env_i].append((
                obs[env_i].copy(),
                action[env_i].copy(),
                float(reward[env_i]),
                next_obs[env_i].copy(),
                bool(done[env_i]),
                infos[env_i],
            ))
            if len(self._pending[env_i]) >= self.n_steps:
                self._flush_one(env_i)
            if bool(done[env_i]):
                while len(self._pending[env_i]) > 0:
                    self._flush_one(env_i)

    # ------------------------------------------------------------------
    def _flush_one(self, env_i: int) -> None:
        """Build the n-step return for the oldest pending transition and store it."""
        q = self._pending[env_i]
        if len(q) == 0:
            return

        obs0, action0, _, next_obs0, _, info0 = q[0]

        R_n = 0.0
        discount = 1.0
        final_next_obs = next_obs0
        final_done = False
        final_info = info0

        for (_, _, rk, next_obs_k, done_k, info_k) in q:
            R_n += discount * rk
            discount *= self._n_gamma
            final_next_obs = next_obs_k
            final_done = done_k
            final_info = info_k
            if done_k:               # truncate the return at the episode boundary
                break

        # Store exactly one (n-step) transition through the parent buffer.
        # Call super().add (not self.add) to avoid re-queuing / recursion.
        super().add(
            obs0[np.newaxis],
            final_next_obs[np.newaxis],
            action0[np.newaxis],
            np.array([R_n], dtype=np.float32),
            np.array([float(final_done)], dtype=np.float32),
            [final_info],
        )
        q.popleft()

    # ------------------------------------------------------------------
    def flush_all_pending(self) -> None:
        """Flush every pending partial window (call before a buffer checkpoint).

        Not required for training correctness -- pending data is simply
        re-collected after a resume -- but keeps a saved buffer complete.
        """
        for env_i in range(self.n_envs):
            while len(self._pending[env_i]) > 0:
                self._flush_one(env_i)
