# =============================================================================
# src/rl/ema_smoothing.py
# Post-hoc control smoothing for the trained TD3 policy (future-work item #5).
#
# WHAT
# ----
# The TD3 v2.21c policy achieves ~99.7% of MPC yield but is "pulsy": its mean
# day-to-day action change is mean|Delta u| ~= 2.50 mm/day versus MPC's 0.97.
# Pulsing is structural to a reactive single-step deterministic policy (the
# v2.22 Markov-r5 experiment confirmed an in-network smoothing penalty makes
# things worse, not better).  This module instead applies smoothing AS A
# POST-PROCESS on the already-trained policy's output, requiring no retraining,
# and lets us trace the resulting yield-vs-smoothness Pareto frontier.
#
# METHOD — exponential moving average (EMA)
# -----------------------------------------
# A first-order exponential moving average is the discrete-time first-order
# low-pass (RC) filter; it is the standard, minimal, single-parameter tool for
# trading responsiveness against smoothness, and is widely used for actuator
# command smoothing / rate limiting in control practice.  For per-agent action
# u_raw[t] from the policy:
#
#     u_smooth[t] = alpha * u_raw[t] + (1 - alpha) * u_smooth[t-1]
#     u_smooth[0] = u_raw[0]                      (seed with the first sample)
#
# with smoothing weight alpha in (0, 1]:
#   * alpha = 1.0  -> u_smooth == u_raw  (no smoothing; reproduces v2.21c).
#                     This is the Pareto anchor and a correctness check.
#   * alpha -> 0   -> heavier smoothing; the action approaches a slowly-moving
#                     average and mean|Delta u| falls toward 0.
#
# The equivalent first-order time constant is tau = -1 / ln(1 - alpha) days
# (e.g. alpha=0.5 -> tau~=1.44 d, alpha=0.2 -> tau~=4.48 d, alpha=0.1 ->
# tau~=9.49 d), i.e. smaller alpha = longer memory = smoother command.
#
# WHERE IT SITS IN THE PIPELINE (apples-to-apples with the baseline)
# ------------------------------------------------------------------
# The filter is applied to the RAW policy output, BEFORE the runner's physical
# constraints.  src.runner.run_season then applies the SAME per-agent actuator
# cap [0, UB_MM] and the SAME seasonal-budget enforcement to the smoothed
# action that it applies to the unsmoothed baseline.  Because the EMA of values
# in [0, UB_MM] stays in [0, UB_MM], the actuator cap is preserved exactly, and
# the only differences versus baseline are (a) the temporal smoothing and
# (b) whatever the shared budget logic does on top — making the comparison fair.
#
# RUNS NOTHING ON IMPORT.  Used by scripts/experiments/exp_rl_ema_smoothing.py.
# =============================================================================

from __future__ import annotations

import time

import numpy as np

from src.rl.gym_env import UB_MM
from src.rl.runner import RLController


# ── Offline helper (pure function, used for tests and trajectory analysis) ───

def apply_ema(u_seq: np.ndarray, alpha: float) -> np.ndarray:
    """Apply the causal EMA filter to a full action sequence offline.

    Mirrors the online recursion in :class:`EMASmoothedRLController` exactly,
    so an offline-smoothed trajectory equals the online-smoothed one for the
    same raw actions.  Provided for unit tests and for "what-if" smoothing of
    already-saved trajectories without re-running the policy.

    Parameters
    ----------
    u_seq : np.ndarray, shape (T,) or (T, N)
        Raw per-step (optionally per-agent) action.
    alpha : float
        Smoothing weight in (0, 1].

    Returns
    -------
    np.ndarray
        Smoothed sequence, same shape as ``u_seq``.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    u_seq = np.asarray(u_seq, dtype=float)
    out = np.empty_like(u_seq)
    out[0] = u_seq[0]
    one_minus = 1.0 - alpha
    for t in range(1, u_seq.shape[0]):
        out[t] = alpha * u_seq[t] + one_minus * out[t - 1]
    return out


def ema_time_constant_days(alpha: float) -> float:
    """Equivalent first-order time constant (days) of the EMA, tau = -1/ln(1-a).

    Returns 0.0 for alpha == 1.0 (no memory) and +inf as alpha -> 0.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0:
        return 0.0
    return float(-1.0 / np.log(1.0 - alpha))


# ── Online controller wrapper ────────────────────────────────────────────────

class EMASmoothedRLController(RLController):
    """RLController whose per-day action is EMA-smoothed before being returned.

    Identical to :class:`src.rl.runner.RLController` in every respect
    (checkpoint auto-detection, observation construction, deterministic
    inference, noisy-forecast support) except that ``step`` applies the causal
    EMA filter to the raw policy action.  Works with any checkpoint RLController
    supports; intended for the TD3 v2.21c models.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained SB3 checkpoint (e.g. v2.21c best_model.zip).
    ema_alpha : float, default 1.0
        EMA smoothing weight in (0, 1].  1.0 reproduces the unsmoothed policy.
    **kwargs
        Forwarded verbatim to ``RLController.__init__`` (deterministic,
        forecast_mode, noise_sigma, noise_rho, noise_seed, verbose, ...).
    """

    def __init__(self, model_path, ema_alpha: float = 1.0, **kwargs):
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        self.ema_alpha = float(ema_alpha)
        self._ema_state = None          # set per-season in reset()
        super().__init__(model_path=model_path, **kwargs)
        # Distinguish smoothed runs in any name-derived logging.
        self.name = f"{self.name}_ema{self.ema_alpha:.3f}"

    def reset(self, *args, **kwargs):
        """Reset the base controller and clear the EMA filter state."""
        super().reset(*args, **kwargs)
        self._ema_state = None

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Return the EMA-smoothed per-agent action (mm/day) for this day."""
        t0 = time.time()

        obs = self._build_obs(day, state, budget_remaining)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        u_raw = np.asarray(action, dtype=float).clip(0.0, 1.0) * UB_MM

        if self._ema_state is None:
            # Seed the filter with the first raw action (no spurious ramp from 0).
            u_smooth = u_raw.copy()
        else:
            a = self.ema_alpha
            u_smooth = a * u_raw + (1.0 - a) * self._ema_state
        self._ema_state = u_smooth.copy()

        self._inference_times.append((time.time() - t0) * 1000.0)

        if self.verbose and (day % 20 == 0):
            print(f"    day {day:3d}: u_raw_mean={u_raw.mean():.2f}mm "
                  f"u_ema_mean={u_smooth.mean():.2f}mm "
                  f"(alpha={self.ema_alpha:.3f})")

        # Keep _u_prev consistent (unused by v2.21c obs, but correct in general).
        self._u_prev = u_smooth.copy()
        return u_smooth
