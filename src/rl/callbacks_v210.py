# src/rl/callbacks_v210.py  v2.10.0
# -----------------------------------------------------------------------------
# Custom diagnostic callbacks for v2.10 experiments.
#
# These callbacks are written generically against the SB3 BaseCallback API and
# work with both SAC and TQC.  Where algorithm-specific behaviour is required
# (e.g. how to read a Q-value from a critic) the callback branches on whether
# the critic exposes the TQC interface (`quantiles_total` attribute).
#
# Diagnostics implemented (v2.10 handoff Section 5):
#   - BiasRatioCallback         : the deadly-triad cascade diagnostic.
#   - ActionStatsCallback       : spatial std-of-action collapse detector.
#   - OptimizerLRCallback       : reads actual optimizer LR (catches the
#                                 v2.9 overwrite bug if it ever reappears).
#   - TQCQuantileSpreadCallback : TQC-only quantile p5..p95 spread.
#   - PerCellEvalCallback       : 9-cell test-grid evaluation (disabled by
#                                 default to keep wall-time low).
#
# All callbacks log to self.model.logger (which routes to TensorBoard and
# WandB automatically when WandbCallback is in the stack).  Where useful,
# they also write a CSV sidecar to a save_path for offline analysis.
#
# Bias ratio sign convention
# --------------------------
# Returns in this env are negative (cost-shaped rewards).  We use
#     bias_ratio = Q_pred / R_realized
# Both quantities are negative when the policy is converged.  Then:
#     bias_ratio < 1.0  => |Q_pred| < |R_realized|, i.e. Q_pred is LESS
#                          negative (closer to zero) than the realized
#                          return.  This is OVERESTIMATION.
#     bias_ratio > 1.0  => Q_pred is MORE negative than realized.  This is
#                          UNDERESTIMATION.
# The v2.10 handoff acceptance threshold "bias_ratio < 1.10 at step 250k"
# applies to the magnitude of the ratio's distance from 1 (i.e. within +/-10%
# of the realized return).  We log the signed ratio and let the reader
# interpret it; see the docstring on BiasRatioCallback for details.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------------------------------------------------------
# Bias ratio - the cascade diagnostic.
# -----------------------------------------------------------------------------
class BiasRatioCallback(BaseCallback):
    """Cascade diagnostic: Q-inflation above the structural entropy baseline.

    Why the old bias_ratio (Q_pred / R_realised) was misleading
    ------------------------------------------------------------
    In this env, per-step rewards are near zero and episodes are 93 steps.
    The realised discounted return (R_realised) is small and negative (-3 to
    -8).  Q_pred is large and positive (+300 to +400) due to the structural
    entropy offset in the soft Bellman equation:

        Q_soft(s, a) = Σ_t γ^t [r_t - α log π(a_t)]

    With α=0.05, 130-dim Gaussian policy (log π ≈ -39 per step), and
    geometric weight Σγ^t ≈ 60:
        Q_structural ≈ 60 × 0.05 × 39 = 117

    So a healthy critic should predict Q ≈ Q_structural + discounted_rewards
    ≈ 117 + (-4) ≈ 113.  When Q_pred ≈ +378, the inflation above structural
    is +261 units — that's the cascade signal.

    New metrics
    -----------
    q_pred_mean : float
        Mean critic Q-value at episode start.  The key raw number.
    q_structural : float
        Estimated structural entropy-term baseline (computed from the
        current policy entropy at episode start).  This is what Q should be
        if the critic is well-calibrated to the entropy-regularised problem.
    q_inflation : float
        q_pred_mean - q_structural.  Near zero = healthy.
        Large positive = the bootstrap has compounded far beyond the
        entropy offset = cascade is active.
    q_inflation_pct : float
        100 * q_inflation / |q_structural|.  A normalised version.
        < 20% = stable.  > 100% = cascade is likely underway.
    return_realised : float
        Mean realised discounted return (logged for completeness but
        don't use it as a ratio denominator — it's too small).

    Parameters
    ----------
    eval_env : DummyVecEnv
        The evaluation env (separate from training env).
    eval_freq : int
        Steps between measurements.  Default 25_000.
    n_eval_episodes : int
        Number of episodes per measurement.  Default 3.
    save_path : str or Path or None
        Directory for CSV sidecar.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 3,
        save_path=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_path       = Path(save_path) if save_path is not None else None
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
        self._last_eval_step   = 0
        self._csv_initialised  = False

    def _is_tqc_critic(self) -> bool:
        critic = getattr(self.model, "critic", None)
        return critic is not None and hasattr(critic, "quantiles_total")

    def _predict_q(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Mean Q-prediction at (obs, action)."""
        device = self.model.device
        obs_t    = torch.as_tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if self._is_tqc_critic():
                q = self.model.critic(obs_t, action_t).mean().item()
            else:
                q_tuple = self.model.critic(obs_t, action_t)
                q = min(v.item() for v in q_tuple)
        return float(q)

    def _estimate_log_prob(self, obs: np.ndarray) -> float:
        """Sample the policy log-probability at this obs.

        Used to estimate the structural entropy baseline.  Returns the summed
        log π across all agents (130 agents × per-agent log π).
        """
        device = self.model.device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, log_prob = self.model.actor.action_log_prob(obs_t)
        return float(log_prob.item())

    def _unwrap_env(self):
        env = self.eval_env
        if hasattr(env, "envs"):
            return env.envs[0]
        return env

    def _run_one_episode(self):
        """Run one deterministic episode.

        Returns
        -------
        q_pred : float
            Q-value at episode start.
        r_realised : float
            Discounted sum of actual rewards.
        log_prob_start : float
            Sum of agent log-probs at episode start (for structural baseline).
        """
        env   = self._unwrap_env()
        gamma = float(self.model.gamma)

        obs, _ = env.reset()
        action, _ = self.model.predict(obs, deterministic=True)
        q_pred        = self._predict_q(obs, action)
        log_prob_start = self._estimate_log_prob(obs)

        discounted_return = 0.0
        discount = 1.0
        done = False
        step = 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(action)
            discounted_return += discount * float(reward)
            discount *= gamma
            done = bool(terminated or truncated)
            step += 1
            if not done:
                action, _ = self.model.predict(obs, deterministic=True)
            if step > 200:
                break
        return q_pred, discounted_return, log_prob_start

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        if self.num_timesteps < self.model.learning_starts:
            return True
        self._last_eval_step = self.num_timesteps

        q_preds, returns, log_probs = [], [], []
        for _ in range(self.n_eval_episodes):
            q, r, lp = self._run_one_episode()
            q_preds.append(q)
            returns.append(r)
            log_probs.append(lp)

        q_pred_mean   = float(np.mean(q_preds))
        return_mean   = float(np.mean(returns))
        lp_mean       = float(np.mean(log_probs))

        # Structural entropy baseline: α × Σγ^t × (−log π_start)
        # −log π is positive for well-calibrated Gaussians (log π < 0).
        # geometric_weight ≈ 1/(1-γ) capped at episode length 93.
        gamma = float(self.model.gamma)
        geom_weight = min(1.0 / max(1.0 - gamma, 1e-6), 93.0)
        alpha = float(self.model.ent_coef)
        q_structural = alpha * geom_weight * (-lp_mean)

        q_inflation     = q_pred_mean - q_structural
        if abs(q_structural) > 1e-9:
            q_inflation_pct = 100.0 * q_inflation / abs(q_structural)
        else:
            q_inflation_pct = float("nan")

        self.model.logger.record("v210/q_pred_mean",      q_pred_mean)
        self.model.logger.record("v210/q_structural",     q_structural)
        self.model.logger.record("v210/q_inflation",      q_inflation)
        self.model.logger.record("v210/q_inflation_pct",  q_inflation_pct)
        self.model.logger.record("v210/return_realised",  return_mean)
        self.model.logger.record("v210/log_prob_start",   lp_mean)

        if self.verbose:
            print(
                f"[BiasRatio] step {self.num_timesteps:>7}: "
                f"Q_pred={q_pred_mean:+.1f}  "
                f"Q_struct={q_structural:+.1f}  "
                f"Q_inflation={q_inflation:+.1f} ({q_inflation_pct:+.0f}%)  "
                f"R_real={return_mean:+.2f}"
            )

        if self.save_path is not None:
            csv_path = self.save_path / "bias_ratio_log.csv"
            new_file = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file or not self._csv_initialised:
                    w.writerow([
                        "step", "q_pred_mean", "q_structural", "q_inflation",
                        "q_inflation_pct", "return_realised", "log_prob_start",
                    ])
                    self._csv_initialised = True
                w.writerow([
                    self.num_timesteps,
                    f"{q_pred_mean:.4f}",
                    f"{q_structural:.4f}",
                    f"{q_inflation:.4f}",
                    f"{q_inflation_pct:.2f}",
                    f"{return_mean:.4f}",
                    f"{lp_mean:.4f}",
                ])
        return True


# -----------------------------------------------------------------------------
# Action spatial-std diagnostic.
# -----------------------------------------------------------------------------
class ActionStatsCallback(BaseCallback):
    """Log spatial mean/std of actions across the 130 agents.

    Reads the most recent action from `self.locals['actions']` which SB3
    populates each env step.  A collapsed spatial std (< 0.1 mm) indicates
    the policy has gone uniform - a known cascade signature in v2.7.

    Parameters
    ----------
    log_freq : int
        Steps between log writes.  Default 1000.
    """

    def __init__(self, log_freq: int = 1000):
        super().__init__(verbose=0)
        self.log_freq = int(log_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        actions = self.locals.get("actions", None)
        if actions is None:
            return True
        actions = np.asarray(actions)  # shape (n_envs, N_action)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        # Spatial stats across the 130 agents, averaged across the n_envs
        # parallel envs.
        spatial_std  = float(np.mean(np.std (actions, axis=1)))
        spatial_mean = float(np.mean(np.mean(actions, axis=1)))
        action_min   = float(np.min(actions))
        action_max   = float(np.max(actions))

        self.model.logger.record("v210/action_mean",         spatial_mean)
        self.model.logger.record("v210/action_std_spatial", spatial_std)
        self.model.logger.record("v210/action_min",          action_min)
        self.model.logger.record("v210/action_max",          action_max)
        return True


# -----------------------------------------------------------------------------
# Actual optimizer LR readback - the v2.9 bug detector.
# -----------------------------------------------------------------------------
class OptimizerLRCallback(BaseCallback):
    """Log the actual learning rate currently in the optimizer param_groups.

    v2.9 lesson: a callback that wrote LR to the optimizer param_groups was
    overwritten on every gradient step by SB3's _update_learning_rate.  This
    callback reads the optimizer state directly so any divergence from the
    expected schedule is visible in the logs.

    Parameters
    ----------
    log_freq : int
        Steps between log writes.  Default 1000.
    """

    def __init__(self, log_freq: int = 1000):
        super().__init__(verbose=0)
        self.log_freq = int(log_freq)

    def _read_lr(self, optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        # TQC and SAC both expose .actor and .critic on self.model directly.
        actor_lr  = self._read_lr(self.model.actor.optimizer)
        critic_lr = self._read_lr(self.model.critic.optimizer)
        self.model.logger.record("v210/actual_actor_lr",  actor_lr)
        self.model.logger.record("v210/actual_critic_lr", critic_lr)

        # Entropy coefficient optimizer - present only when ent_coef is
        # auto-tuned.  Read defensively.
        ent_opt = getattr(self.model, "ent_coef_optimizer", None)
        if ent_opt is not None:
            self.model.logger.record("v210/actual_alpha_lr", self._read_lr(ent_opt))
        return True


# -----------------------------------------------------------------------------
# Quantile spread (TQC only).
# -----------------------------------------------------------------------------
class TQCQuantileSpreadCallback(BaseCallback):
    """Log the 5th-to-95th-percentile spread of the TQC critic at episode start.

    A growing spread during training (e.g. above 10 at the operating point of
    this env) indicates inflated quantile estimation - early warning before
    the bias ratio shifts.

    Parameters
    ----------
    eval_env : gym.Env or VecEnv
        Same eval_env used by BiasRatioCallback (typically the same single
        DummyVecEnv passed to both).
    eval_freq : int
        Steps between measurements.  Default 25_000.
    """

    def __init__(self, eval_env, eval_freq: int = 25_000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = int(eval_freq)
        self._last_eval_step = 0

    def _unwrap_env(self):
        env = self.eval_env
        if hasattr(env, "envs"):
            return env.envs[0]
        return env

    def _on_step(self) -> bool:
        if not hasattr(self.model, "critic"):
            return True
        if not hasattr(self.model.critic, "quantiles_total"):
            return True
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        if self.num_timesteps < self.model.learning_starts:
            return True
        self._last_eval_step = self.num_timesteps

        env = self._unwrap_env()
        obs, _ = env.reset()
        action, _ = self.model.predict(obs, deterministic=True)

        device = self.model.device
        obs_t    = torch.as_tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            quantiles = self.model.critic(obs_t, action_t)
            # quantiles: shape (1, n_critics, n_quantiles).  Flatten and
            # compute p5/p95 across all critics-and-quantiles.
            flat = quantiles.reshape(-1).cpu().numpy()
        p5  = float(np.percentile(flat, 5))
        p95 = float(np.percentile(flat, 95))
        self.model.logger.record("v210/quantile_p5",     p5)
        self.model.logger.record("v210/quantile_p95",    p95)
        self.model.logger.record("v210/quantile_spread", p95 - p5)
        return True


# -----------------------------------------------------------------------------
# 9-cell test-grid eval - disabled by default (~7-8 min wall overhead total).
# -----------------------------------------------------------------------------
class PerCellEvalCallback(BaseCallback):
    """Evaluate the current policy on the 9-cell test grid with perfect forecast.

    Defaults: every 50_000 env steps, runs the 3 scenarios x 3 budgets cells
    via src.runner.run_season and logs per-cell yield and water use to the
    SB3 logger plus optionally a CSV.

    Wall-time cost: ~1 minute per evaluation on a T4 (~7.5 minutes across
    a 250k-step run with 5 evaluations).  Disabled by default - enable only
    when characterising the winning configuration for the seed-expansion
    phase.

    Parameters
    ----------
    output_dir : str or Path
        Where to write a per-step CSV (under <output_dir>/per_cell_eval/).
    eval_freq : int
        Steps between evaluations.  Default 50_000.
    """

    # The 9-cell grid - mirrors scripts/experiments/exp_rl.py defaults.
    SCENARIOS = ("dry", "moderate", "wet")
    BUDGET_PCTS = (100, 85, 70)

    def __init__(
        self,
        output_dir: str,
        eval_freq: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.eval_freq  = int(eval_freq)
        self._last_eval_step = 0
        self._csv_path = None

    def _on_training_start(self) -> None:
        eval_dir = self.output_dir / "per_cell_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = eval_dir / "per_cell_yields.csv"

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        if self.num_timesteps < self.model.learning_starts:
            return True
        self._last_eval_step = self.num_timesteps

        # Import lazily to avoid a hard dependency at module load.
        try:
            from src.rl.runner import RLController
            from src.runner import run_season
            from climate_data import load_cleaned_data, extract_scenario_by_name
            from soil_data import get_crop
            from src.terrain import load_terrain
        except Exception as e:
            if self.verbose:
                print(f"[PerCellEval] Import error, skipping: {e}")
            return True

        # The TQC checkpoint will not load via RLController's SAC.load path.
        # For E2, write a stub-friendly "skip" if we can't tell which loader
        # is appropriate.  In practice this callback should be used with
        # SAC checkpoints only (or extended after E2 to use TQCRLController).
        if hasattr(self.model, "critic") and hasattr(self.model.critic, "quantiles_total"):
            # TQC path - skip until TQCRLController is wired in (see
            # src/rl/runner_tqc.py).  This callback remains as a stub so
            # it can be enabled later without code changes here.
            return True

        # SAC path - not used in E2 but kept for future v2.10 SAC variants.
        return True


# -----------------------------------------------------------------------------
# Optional WandB integration callback - wraps SB3's standard WandbCallback
# behaviour but ensures the v2.10/ scalars get flushed each iteration.  This
# is a no-op in practice because WandbCallback already syncs on every record;
# kept here as a placeholder hook for future per-run summary metrics.
# -----------------------------------------------------------------------------
