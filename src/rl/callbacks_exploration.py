# src/rl/callbacks_exploration.py  --  Path-3 exploration callbacks (v2.17-P3)
# -----------------------------------------------------------------------------
# Two callbacks supporting the Path-3 "exploration injection" diagnostic:
#
#   1. ExplorationNoiseDecayCallback
#        Linearly anneals the sigma of the model's NormalActionNoise from
#        sigma_start to sigma_end over `decay_steps` environment steps, then
#        holds sigma_end (default 0.0) for the remainder of training.
#
#        WHY a callback (and not a smart ActionNoise subclass): the schedule is
#        driven off self.num_timesteps (the true global env-step counter), NOT
#        off how many times ActionNoise.__call__ fires.  The repo's own v2.9
#        lesson (see OptimizerLRCallback) is that anything keyed off a private
#        per-call counter desynchronises from training under vec-envs /
#        gradient_steps.  Setting _sigma from num_timesteps is the robust route.
#
#   2. LowActionCoverageCallback
#        Logs, every `log_freq` steps, the fraction of just-collected per-agent
#        actions that fall in the "low water" region (env action < low_thresh,
#        i.e. < low_thresh * UB_MM mm/day), split by whether the episode is a
#        wet-climate episode.  This is the instrument that makes a NULL Path-3
#        result interpretable: it tells us whether the injected noise actually
#        populated the replay buffer with low-water-in-wet transitions (the
#        thing the hypothesis bets on) or not.
#
# Scientific basis for decaying exploration noise:
#   - Lillicrap et al. (2016), "Continuous control with deep RL" (DDPG): additive
#     exploration noise on the actor output during data collection.
#   - Fujimoto et al. (2018), "Addressing Function Approximation Error in
#     Actor-Critic Methods" (TD3): N(0, 0.1) Gaussian exploration noise.
#   Decaying the magnitude over training (explore-early / exploit-late) is the
#   standard annealed-exploration schedule; here it is sized larger (0.30) at
#   start because the target region (0 mm/day) sits at the action-space boundary
#   a_env = 0, which symmetric noise around the operating point (~0.4-0.5 in
#   [0,1] units) only reaches in its lower tail.
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ExplorationNoiseDecayCallback(BaseCallback):
    """Linearly decay the sigma of the model's NormalActionNoise.

    The schedule is a function of ``self.num_timesteps`` (global env steps):

        progress = clip(num_timesteps / decay_steps, 0, 1)
        sigma    = sigma_start + progress * (sigma_end - sigma_start)

    so sigma moves sigma_start -> sigma_end over the first ``decay_steps`` steps
    and then holds at sigma_end.

    The model MUST have been constructed with an ``action_noise`` that exposes a
    mutable ``_sigma`` attribute of shape (action_dim,) -- e.g. SB3's
    NormalActionNoise.  If ``model.action_noise`` is None the callback is inert
    (logs a one-time warning) so the same training script can be run with and
    without injected noise.

    Parameters
    ----------
    sigma_start : float
        Initial per-dimension noise std in normalised action units ([0, 1]).
    sigma_end : float
        Final per-dimension noise std (default 0.0 -> exploration fully off).
    decay_steps : int
        Number of env steps over which sigma anneals linearly.
    log_freq : int
        Steps between logger records of the current sigma.
    csv_path : str or Path, optional
        If given, append (step, sigma) rows to this CSV for crash-proof,
        re-run-free record keeping.
    verbose : int
    """

    def __init__(
        self,
        sigma_start: float = 0.30,
        sigma_end: float = 0.0,
        decay_steps: int = 60_000,
        log_freq: int = 1_000,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.sigma_start = float(sigma_start)
        self.sigma_end = float(sigma_end)
        self.decay_steps = int(decay_steps)
        self.log_freq = int(log_freq)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self._action_dim: Optional[int] = None
        self._warned_no_noise = False
        self._csv_initialised = False

    def _current_sigma(self) -> float:
        if self.decay_steps <= 0:
            return self.sigma_end
        progress = min(max(self.num_timesteps / self.decay_steps, 0.0), 1.0)
        return self.sigma_start + progress * (self.sigma_end - self.sigma_start)

    def _on_training_start(self) -> None:
        noise = getattr(self.model, "action_noise", None)
        if noise is None:
            if not self._warned_no_noise and self.verbose:
                print(
                    "[ExplorationNoiseDecay] model.action_noise is None -- "
                    "callback is inert (no exploration noise to decay)."
                )
            self._warned_no_noise = True
            return
        # Infer action_dim from the existing sigma vector.
        sigma_attr = getattr(noise, "_sigma", None)
        if sigma_attr is None:
            raise AttributeError(
                "ExplorationNoiseDecayCallback expects an action_noise with a "
                "mutable '_sigma' attribute (e.g. SB3 NormalActionNoise)."
            )
        self._action_dim = int(np.asarray(sigma_attr).reshape(-1).shape[0])
        # Set the initial sigma explicitly so step 0 already matches the schedule.
        noise._sigma = self._current_sigma() * np.ones(self._action_dim, dtype=np.float64)
        if self.csv_path is not None and not self._csv_initialised:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "sigma"])
            self._csv_initialised = True

    def _on_step(self) -> bool:
        noise = getattr(self.model, "action_noise", None)
        if noise is None or self._action_dim is None:
            return True

        sigma = self._current_sigma()
        # NormalActionNoise reads _sigma on every __call__, so updating it here
        # takes effect on the very next collected step.
        noise._sigma = sigma * np.ones(self._action_dim, dtype=np.float64)

        if self.num_timesteps % self.log_freq == 0:
            self.model.logger.record("p3/exploration_sigma", float(sigma))
            if self.csv_path is not None:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([int(self.num_timesteps), float(sigma)])
        return True


class LowActionCoverageCallback(BaseCallback):
    """Measure whether injected exploration actually reaches the low-water region.

    Every ``log_freq`` steps, reads the most recently collected actions from
    ``self.locals['actions']`` (SB3 stores the env-scaled action in [0, 1]) and
    records:

        p3/frac_low_action        fraction of per-agent actions < low_thresh
        p3/frac_low_action_wet    same, but only when the current episode is a
                                   wet-climate episode (if detectable)
        p3/min_action_collected   min over the 130 agents this step

    "Low water" is action < low_thresh in [0, 1] units, i.e.
    < low_thresh * UB_MM mm/day (default low_thresh=0.0833 -> < 1.0 mm/day at
    UB_MM=12).

    Wet-episode detection: tries, in order, (a) info['scenario'] / info['climate']
    strings containing 'wet', (b) env attribute ``_scenario`` / ``scenario``.
    If none is available the wet-specific metric is simply not recorded (the
    overall fraction is always recorded), so the callback degrades gracefully.

    Parameters
    ----------
    low_thresh : float
        Threshold in normalised [0, 1] action units. Default 1/12 (~1 mm/day).
    log_freq : int
        Steps between logger records.
    csv_path : str or Path, optional
        Append rows for crash-proof record keeping.
    verbose : int
    """

    def __init__(
        self,
        low_thresh: float = 1.0 / 12.0,
        log_freq: int = 1_000,
        csv_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.low_thresh = float(low_thresh)
        self.log_freq = int(log_freq)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self._csv_initialised = False

    def _is_wet_episode(self) -> Optional[bool]:
        # Try infos first (DummyVecEnv exposes them in self.locals['infos']).
        infos = self.locals.get("infos", None)
        if infos:
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info0, dict):
                for key in ("scenario", "climate", "climate_type", "year_type"):
                    val = info0.get(key, None)
                    if isinstance(val, str):
                        return "wet" in val.lower()
        # Fall back to env attribute inspection.
        try:
            env0 = self.training_env.envs[0]
            base = getattr(env0, "unwrapped", env0)
            for key in ("_scenario", "scenario", "_climate", "climate"):
                val = getattr(base, key, None)
                if isinstance(val, str):
                    return "wet" in val.lower()
        except Exception:
            pass
        return None

    def _on_training_start(self) -> None:
        if self.csv_path is not None and not self._csv_initialised:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["step", "frac_low_action", "frac_low_action_wet", "min_action"]
                )
            self._csv_initialised = True

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True
        actions = self.locals.get("actions", None)
        if actions is None:
            return True
        actions = np.asarray(actions).reshape(-1)  # flatten n_envs * N agents
        frac_low = float(np.mean(actions < self.low_thresh))
        min_action = float(np.min(actions))
        self.model.logger.record("p3/frac_low_action", frac_low)
        self.model.logger.record("p3/min_action_collected", min_action)

        wet = self._is_wet_episode()
        frac_low_wet = frac_low if wet else float("nan")
        if wet is True:
            self.model.logger.record("p3/frac_low_action_wet", frac_low)

        if self.csv_path is not None:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [int(self.num_timesteps), frac_low, frac_low_wet, min_action]
                )
        return True
