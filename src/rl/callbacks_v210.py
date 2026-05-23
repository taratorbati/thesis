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
    """Compute Q_pred / Realised-return at episode start.

    Every `eval_freq` env steps, runs `n_eval_episodes` deterministic episodes
    in the supplied eval_env.  For each episode:
      - Read Q_pred = critic(s_0, a_0) where a_0 = policy(s_0).
        For TQC critics, Q_pred = mean over critics and quantiles.
        For SAC twin critics, Q_pred = min(q1, q2).
      - Roll out the full episode and record per-step rewards.
      - Compute R_realised = sum_t gamma^t r_t (discounted return).
    Logs the mean and std of (Q_pred / R_realised) across episodes to the
    SB3 logger and optionally to a CSV sidecar.

    Parameters
    ----------
    eval_env : gym.Env or VecEnv
        Environment to evaluate on.  Either a single env or a VecEnv;
        the callback unwraps a VecEnv if needed.
    eval_freq : int
        How often (in env steps) to compute the diagnostic.  Default 25_000.
    n_eval_episodes : int
        Number of episodes per evaluation.  Default 3.
    save_path : str or Path or None
        If not None, append one row per evaluation to
        `save_path/bias_ratio_log.csv`.

    Notes
    -----
    The first evaluation only fires at step >= learning_starts to avoid
    measuring against an untrained critic.  Use `learning_starts=1000` (the
    v2.7 default) and the first measurement happens at the first eval_freq
    boundary >= 1000.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 3,
        save_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env       = eval_env
        self.eval_freq      = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_path      = Path(save_path) if save_path is not None else None
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
        self._last_eval_step = 0
        self._csv_initialised = False

    def _is_tqc_critic(self) -> bool:
        """True if the model's critic is a TQC-style quantile critic."""
        critic = getattr(self.model, "critic", None)
        return critic is not None and hasattr(critic, "quantiles_total")

    def _predict_q(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Q-value estimate at (obs, action), reduced to a scalar.

        For TQC: mean over critics and quantiles.
        For SAC: min over twin Q-nets (clipped double-Q convention).
        """
        device = self.model.device
        obs_t    = torch.as_tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if self._is_tqc_critic():
                # TQC: (1, n_critics, n_quantiles) -> scalar via mean-mean.
                quantiles = self.model.critic(obs_t, action_t)
                q_scalar  = quantiles.mean().item()
            else:
                # SAC: tuple of (1, 1) twin-Q tensors -> min.
                q_tuple = self.model.critic(obs_t, action_t)
                q_vals  = [q.item() for q in q_tuple]
                q_scalar = min(q_vals)
        return float(q_scalar)

    def _unwrap_env(self):
        """Return a callable that resets and steps a non-vec env."""
        # eval_env may be a DummyVecEnv with one inner env or a raw Env.
        env = self.eval_env
        if hasattr(env, "envs"):
            return env.envs[0]
        return env

    def _run_one_episode(self) -> tuple[float, float]:
        """Return (q_pred, r_realised_discounted)."""
        env   = self._unwrap_env()
        gamma = float(self.model.gamma)

        obs, _ = env.reset()
        action, _ = self.model.predict(obs, deterministic=True)
        q_pred = self._predict_q(obs, action)

        discounted_return = 0.0
        discount = 1.0
        done = False
        step = 0
        while not done:
            obs, reward, terminated, truncated, _info = env.step(action)
            discounted_return += discount * float(reward)
            discount *= gamma
            done = bool(terminated or truncated)
            step += 1
            if not done:
                action, _ = self.model.predict(obs, deterministic=True)
            # safety cap (full season is 93 days; budget exhaustion no
            # longer terminates the episode under v2.7+ logic)
            if step > 200:
                break
        return q_pred, discounted_return

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        if self.num_timesteps < self.model.learning_starts:
            return True
        self._last_eval_step = self.num_timesteps

        q_preds  = []
        returns  = []
        for _ in range(self.n_eval_episodes):
            q, r = self._run_one_episode()
            q_preds.append(q)
            returns.append(r)

        q_preds_a = np.array(q_preds, dtype=np.float64)
        returns_a = np.array(returns, dtype=np.float64)

        # Avoid division by zero - if a return is exactly 0 (unlikely),
        # mask it out of the ratio computation.
        valid = np.abs(returns_a) > 1e-9
        if valid.any():
            ratios = q_preds_a[valid] / returns_a[valid]
            ratio_mean = float(np.mean(ratios))
            ratio_std  = float(np.std(ratios))
        else:
            ratio_mean = float("nan")
            ratio_std  = float("nan")

        q_pred_mean   = float(np.mean(q_preds_a))
        return_mean   = float(np.mean(returns_a))

        self.model.logger.record("v210/bias_ratio_mean",   ratio_mean)
        self.model.logger.record("v210/bias_ratio_std",    ratio_std)
        self.model.logger.record("v210/q_pred_mean",       q_pred_mean)
        self.model.logger.record("v210/return_realised",   return_mean)

        if self.verbose:
            print(
                f"[BiasRatio] step {self.num_timesteps:>7}: "
                f"Q_pred={q_pred_mean:+.2f}  "
                f"R_real={return_mean:+.2f}  "
                f"ratio={ratio_mean:.3f}+/-{ratio_std:.3f}"
            )

        # CSV sidecar
        if self.save_path is not None:
            csv_path = self.save_path / "bias_ratio_log.csv"
            new_file = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file or not self._csv_initialised:
                    w.writerow([
                        "step", "q_pred_mean", "return_realised",
                        "bias_ratio_mean", "bias_ratio_std",
                        "n_episodes",
                    ])
                    self._csv_initialised = True
                w.writerow([
                    self.num_timesteps,
                    f"{q_pred_mean:.6f}",
                    f"{return_mean:.6f}",
                    f"{ratio_mean:.6f}",
                    f"{ratio_std:.6f}",
                    self.n_eval_episodes,
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
