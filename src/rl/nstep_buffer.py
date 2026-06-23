# src/rl/nstep_buffer.py
# -----------------------------------------------------------------------------
# Exact n-step replay buffer for off-policy TD3.
#
# The buffer accumulates the n-step return with its own discount gamma_base:
#     R_n = sum_{k=0}^{n-1} gamma_base^k * r_{t+k}     (truncated at any done)
# and stores (s_t, a_t, R_n, s_{t+n or terminal}, done_any). The TRAINER sets the
# model's gamma to gamma_base ** n_steps, so SB3's stock TD3 target becomes
#     R_n + (1 - done) * gamma_base^n * Q(s_{t+n})
# exactly (terminal transitions zero the bootstrap via (1-done)). No train()
# override is needed, and the critic still learns the gamma_base(=0.99) return,
# keeping the bias-ratio diagnostic on the same scale.
#
# n-step bounds the bootstrap horizon: it replaces n self-referential bootstrap
# steps with n grounded rewards, shrinking the self-referential term to gamma^n.
# This is what stabilises the long-horizon (93-day) critic.
#
# References: Hessel et al. 2018 (Rainbow); Fedus et al. 2020 (Revisiting
# Experience Replay); Fujimoto et al. 2018 (TD3); Barth-Maron et al. 2018 (D4PG).
# -----------------------------------------------------------------------------

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer


class NStepReplayBuffer(ReplayBuffer):
    """Off-policy n-step replay buffer with an exact gamma^n bootstrap discount.

    Drop-in for SB3 via ``replay_buffer_class=NStepReplayBuffer`` and
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
