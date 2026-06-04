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
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _safe_print(text: str) -> None:
    """Emit a diagnostic line WITHOUT routing through a rich/tqdm stdout proxy.

    SB3's ``progress_bar=True`` replaces ``sys.stdout`` with ``rich``'s
    ``FileProxy``.  Under ipykernel (Colab/Jupyter), printing a multi-line banner
    to that proxy can recurse without bound --
    ``FileProxy.flush -> Console.print -> ipykernel._flush_streams ->
    sys.stdout.flush -> FileProxy.flush -> ...`` -- and raise ``RecursionError``,
    which kills the run.  (That is precisely what felled the collapse runs: the
    CollapseGuard's "COLLAPSE DETECTED" banner detonated the proxy.)  Writing to
    the *original* ``sys.__stdout__`` bypasses the proxy entirely; we never touch
    the proxied ``sys.stdout`` here, so this can never recurse.  Best-effort and
    never raises -- a diagnostic line must never be able to crash training.

    The collapse is independently recorded via ``logger.record('guard/...')``
    (so it still shows in the in-cell SB3 tables) and in the guard CSV, so even
    if this banner is dropped (e.g. ``sys.__stdout__`` is None) no signal is lost.
    """
    try:
        stream = sys.__stdout__
        if stream is None:
            return
        stream.write(text if text.endswith("\n") else text + "\n")
        stream.flush()
    except Exception:
        pass


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
                _safe_print(
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
        wet_rain_threshold_mm: float = 120.0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.low_thresh = float(low_thresh)
        self.log_freq = int(log_freq)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.wet_rain_threshold_mm = float(wet_rain_threshold_mm)
        self._csv_initialised = False

    def _is_wet_episode(self) -> Optional[bool]:
        """Detect a wet episode by PHYSICAL seasonal rainfall, not a label.

        During training (randomize=True) the env samples a year from 20
        TRAINING_YEARS; only 3 years (2022/2018/2024) carry dry/moderate/wet
        labels, so a label-based check returns None almost always (the bug in
        the v2.17 run, where frac_low_action_wet was NaN for all rows).

        Instead we read the env's current-episode rainfall array
        (self._climate['rainfall']) and call the episode "wet" if the season
        total exceeds wet_rain_threshold_mm.  Reference seasonal totals:
        dry=39.7, moderate=108.8, wet=176.8 mm; threshold 120 mm cleanly
        separates the upper (wet-ish) episodes.
        """
        try:
            env0 = self.training_env.envs[0]
            base = getattr(env0, "unwrapped", env0)
            climate = getattr(base, "_climate", None)
            if climate is not None and "rainfall" in climate:
                season_rain = float(np.sum(climate["rainfall"]))
                return season_rain >= self.wet_rain_threshold_mm
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


class LateNoiseReinjectionCallback(BaseCallback):
    """Two-phase exploration noise: initial anneal, then a late re-injected pulse.

    Motivation (v2.18-P3b):
        v2.17-P3 annealed exploration to 0 by step 60k, then exploited for
        190k steps and converged to a policy that improved wet-year behaviour
        (x1 152->144, waterlog 76->54) but did NOT fully reach the MPC / auto-
        alpha operating point (x1 ~130).  The residual over-watering is
        consistent with the critic having thin data on the EVEN-LOWER actions
        (0-2 mm in wet states) that the converged policy stopped sampling once
        noise decayed.  By a late point the critic is well-trained where the
        policy currently visits, so a short re-injected noise pulse repopulates
        the low-water region with transitions that get accurate value estimates
        immediately -- pulling mu down the last bit without the early-training
        instability of running high noise throughout.

    Schedule (piecewise-linear on num_timesteps):
        [0, decay_steps]                 sigma: sigma_start -> sigma_floor
        (decay_steps, reinject_start]    sigma: sigma_floor  (held)
        [reinject_start, +ramp]          sigma: sigma_floor -> sigma_reinject
        [peak, reinject_end]             sigma: sigma_reinject -> sigma_floor
        (reinject_end, end]              sigma: sigma_floor  (held)

    With sigma_floor=0.0 this is exactly: decay to 0, then a triangular pulse
    of height sigma_reinject centred over [reinject_start, reinject_end].

    The schedule is driven off self.num_timesteps (true global env steps), and
    sets model.action_noise._sigma each step -- same robust mechanism as
    ExplorationNoiseDecayCallback (avoids the v2.9 per-call-counter desync bug).

    Parameters
    ----------
    sigma_start : float
        Initial per-dim noise std in normalised action units ([0, 1]).
    sigma_floor : float
        Noise std held between phases (default 0.0 = exploration fully off).
    sigma_reinject : float
        Peak per-dim noise std of the late pulse.
    decay_steps : int
        Steps over which the initial sigma_start -> sigma_floor anneal happens.
    reinject_start, reinject_end : int
        Env-step window of the late pulse. The pulse ramps up over the first
        half and down over the second half (triangular), peaking at the midpoint.
    log_freq : int
    csv_path : str or Path, optional
    verbose : int
    """

    def __init__(
        self,
        sigma_start: float = 0.30,
        sigma_floor: float = 0.0,
        sigma_reinject: float = 0.15,
        decay_steps: int = 60_000,
        reinject_start: int = 150_000,
        reinject_end: int = 180_000,
        log_freq: int = 1_000,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.sigma_start = float(sigma_start)
        self.sigma_floor = float(sigma_floor)
        self.sigma_reinject = float(sigma_reinject)
        self.decay_steps = int(decay_steps)
        self.reinject_start = int(reinject_start)
        self.reinject_end = int(reinject_end)
        self.log_freq = int(log_freq)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self._action_dim: Optional[int] = None
        self._warned_no_noise = False
        self._csv_initialised = False
        self._peak = (self.reinject_start + self.reinject_end) / 2.0

    def _current_sigma(self) -> float:
        t = self.num_timesteps
        # Phase 1: initial anneal.
        if t <= self.decay_steps:
            if self.decay_steps <= 0:
                return self.sigma_floor
            p = min(max(t / self.decay_steps, 0.0), 1.0)
            return self.sigma_start + p * (self.sigma_floor - self.sigma_start)
        # Phase 3: late triangular pulse.
        if self.reinject_start <= t <= self.reinject_end:
            if t <= self._peak:
                denom = max(self._peak - self.reinject_start, 1e-9)
                p = (t - self.reinject_start) / denom
                return self.sigma_floor + p * (self.sigma_reinject - self.sigma_floor)
            denom = max(self.reinject_end - self._peak, 1e-9)
            p = (t - self._peak) / denom
            return self.sigma_reinject + p * (self.sigma_floor - self.sigma_reinject)
        # Phases 2 & 4: held at floor.
        return self.sigma_floor

    def _on_training_start(self) -> None:
        noise = getattr(self.model, "action_noise", None)
        if noise is None:
            if not self._warned_no_noise and self.verbose:
                _safe_print("[LateNoiseReinjection] model.action_noise is None -- "
                            "callback is inert.")
            self._warned_no_noise = True
            return
        sigma_attr = getattr(noise, "_sigma", None)
        if sigma_attr is None:
            raise AttributeError(
                "LateNoiseReinjectionCallback expects an action_noise with a "
                "mutable '_sigma' attribute (e.g. SB3 NormalActionNoise).")
        self._action_dim = int(np.asarray(sigma_attr).reshape(-1).shape[0])
        noise._sigma = self._current_sigma() * np.ones(self._action_dim, dtype=np.float64)
        if self.csv_path is not None and not self._csv_initialised:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "sigma", "phase"])
            self._csv_initialised = True

    def _phase_name(self) -> str:
        t = self.num_timesteps
        if t <= self.decay_steps:
            return "initial_decay"
        if self.reinject_start <= t <= self.reinject_end:
            return "reinjection_pulse"
        return "floor"

    def _on_step(self) -> bool:
        noise = getattr(self.model, "action_noise", None)
        if noise is None or self._action_dim is None:
            return True
        sigma = self._current_sigma()
        noise._sigma = sigma * np.ones(self._action_dim, dtype=np.float64)
        if self.num_timesteps % self.log_freq == 0:
            self.model.logger.record("p3b/exploration_sigma", float(sigma))
            if self.csv_path is not None:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [int(self.num_timesteps), float(sigma), self._phase_name()])
        return True


class CollapseGuardCallback(BaseCallback):
    """Fail-fast guard against deterministic-policy collapse to an action corner.

    Motivation (the v2.19-TD3 failure):
        Removing the SAC entropy term took away the policy's main exploration
        and replay-buffer-coverage mechanism. With only weak action noise left,
        the deterministic TD3 actor drove every cell to the 0 mm boundary, the
        buffer filled with low-water/drought transitions, the twin-Q critic
        diverged negative (q_pred_mean -21 -> -145) and there was no signal to
        climb back. The collapse was visible by 25k steps (eval reward -7) yet
        the run burned all 250k steps. This guard makes that mode trip EARLY.

    Mechanism:
        Every ``check_freq`` steps it reads the just-collected actions
        (``self.locals['actions']``, env-scaled to [0, 1]) and tracks a rolling
        mean of the fraction sitting in the low-water corner (< ``low_thresh``).
        Once past ``warmup_steps`` (i.e. past the random learning-starts phase,
        where ~``low_thresh`` of actions are low by chance), if the rolling
        fraction exceeds ``collapse_frac`` it logs a prominent warning and, when
        ``abort_on_collapse`` is set, returns False to stop training. A short
        probe run therefore fails fast instead of producing a dead 250k policy.

    One-sided BY DESIGN: under-irrigation (the low corner) is the documented TD3
    collapse mode here. The high-water corner is bounded by the budget clip and
    is not the failure being guarded against, so we do not test it.

    Parameters
    ----------
    low_thresh : float
        Low-water threshold in normalised [0, 1] action units. Default 1/12
        (~1 mm/day at UB_MM=12) -- the same definition as
        ``LowActionCoverageCallback``.
    collapse_frac : float
        Rolling low-action fraction at/above which collapse is declared.
    warmup_steps : int
        Do not test before this many env steps. Should sit a little past the
        model's ``learning_starts`` so the random-action phase (where the low
        fraction is ~``low_thresh`` by chance) never trips the guard.
    check_freq : int
        Steps between checks (and between rolling-window samples).
    window : int
        Number of recent checks averaged into the rolling fraction (debounces a
        single noisy batch).
    abort_on_collapse : bool
        If True, stop training (return False) on the first trip; if False, only
        warn and log ``guard/collapsed`` so the full run still completes.
    csv_path : str or Path, optional
        Append (step, frac_low, rolling, collapsed) rows for record keeping.
    verbose : int
    """

    def __init__(
        self,
        low_thresh: float = 1.0 / 12.0,
        collapse_frac: float = 0.60,
        warmup_steps: int = 30_000,
        check_freq: int = 2_000,
        window: int = 8,
        abort_on_collapse: bool = True,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.low_thresh = float(low_thresh)
        self.collapse_frac = float(collapse_frac)
        self.warmup_steps = int(warmup_steps)
        self.check_freq = int(check_freq)
        self.window = int(window)
        self.abort_on_collapse = bool(abort_on_collapse)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self._recent: deque = deque(maxlen=self.window)
        self._tripped = False
        self._csv_initialised = False

    def _on_training_start(self) -> None:
        if self.csv_path is not None and not self._csv_initialised:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["step", "frac_low_action", "frac_low_rolling", "collapsed"]
                )
            self._csv_initialised = True

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True
        actions = self.locals.get("actions", None)
        if actions is None:
            return True
        actions = np.asarray(actions).reshape(-1)
        frac_low = float(np.mean(actions < self.low_thresh))
        self._recent.append(frac_low)
        rolling = float(np.mean(self._recent))

        self.model.logger.record("guard/frac_low_action", frac_low)
        self.model.logger.record("guard/frac_low_rolling", rolling)

        collapsed_now = (
            self.num_timesteps > self.warmup_steps
            and rolling >= self.collapse_frac
        )
        if self.csv_path is not None:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [int(self.num_timesteps), frac_low, rolling, int(collapsed_now)]
                )

        if collapsed_now and not self._tripped:
            self._tripped = True
            self.model.logger.record("guard/collapsed", 1.0)
            msg = (
                f"[CollapseGuard] COLLAPSE DETECTED at step {self.num_timesteps:,}: "
                f"rolling low-action fraction = {rolling:.0%} >= "
                f"{self.collapse_frac:.0%} (window={self.window}). The deterministic "
                f"policy has collapsed to the ~0 mm corner (the v2.19 failure mode)."
            )
            if self.verbose:
                bar = "=" * 72
                _safe_print(f"\n{bar}\n{msg}")
            if self.abort_on_collapse:
                if self.verbose:
                    _safe_print("[CollapseGuard] abort_on_collapse=True -> stopping "
                                "training early to avoid burning the full run.\n" + "=" * 72)
                return False
            if self.verbose:
                _safe_print("[CollapseGuard] abort_on_collapse=False -> warning only; "
                            "run continues.\n" + "=" * 72)
        return True


class NonFiniteGuardCallback(BaseCallback):
    """Fail LOUD (and stop cleanly) the instant training goes non-finite.

    Why this exists
    ---------------
    When the deterministic TD3 actor collapses to the 0 mm corner (the v2.19
    failure mode), the twin-Q critic can run away in a deadly-triad divergence
    until a Q-value / loss overflows float32 to +/-inf and then NaN.  Stable-
    Baselines3 performs NO finiteness check, and neither the env nor the
    networks raise on NaN, so the failure surfaces in one of two unhelpful
    ways:
      * a bare traceback on STDERR -- invisible in a stdout-only Colab/Kaggle
        capture, so the run appears to "just end" mid-stream while wandb is
        still marked 'finished' (the finally-block already ran wandb.finish()
        before the exception propagated); or
      * silent training on NaN for the remaining steps (a dead policy that
        still reports a 250k 'completed' run).

    This callback reads the just-collected actions and the latest logged train
    losses every step; on the FIRST non-finite value it prints a prominent,
    stdout-captured message naming the exact step and quantity, appends a CSV
    row, and (by default) returns False so SB3 stops cleanly, saves the final
    model, and calls wandb.finish() with an unambiguous cause.  On a healthy
    run it is a no-op (a few ``np.isfinite`` checks per step) and it makes no
    RNG draw and mutates no model state, so the training trajectory is
    byte-identical to a run without it.

    Parameters
    ----------
    check_actions : bool
        Check ``self.locals['actions']`` (the env-scaled collected action,
        which goes non-finite one step after the actor weights do).
    loss_keys : sequence of str
        Logger keys to monitor for non-finite (default the TD3 train losses,
        populated once past ``learning_starts``).
    stop_on_nonfinite : bool
        If True (default) return False on the first hit so the run stops with
        a logged cause.  If False, warn once and keep running (e.g. to observe
        how the NaN propagates).
    csv_path : str or Path, optional
        Append a ``(step, quantity, value)`` row on the first hit.
    verbose : int
    """

    def __init__(
        self,
        check_actions: bool = True,
        loss_keys: tuple = ("train/critic_loss", "train/actor_loss"),
        stop_on_nonfinite: bool = True,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.check_actions = bool(check_actions)
        self.loss_keys = tuple(loss_keys)
        self.stop_on_nonfinite = bool(stop_on_nonfinite)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self._tripped = False
        self._csv_initialised = False

    def _on_training_start(self) -> None:
        if self.csv_path is not None and not self._csv_initialised:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "quantity", "value"])
            self._csv_initialised = True

    def _first_nonfinite(self):
        """Return (quantity, value) of the first non-finite signal, else None."""
        if self.check_actions:
            actions = self.locals.get("actions", None)
            if actions is not None:
                arr = np.asarray(actions, dtype=np.float64)
                if arr.size and not np.all(np.isfinite(arr)):
                    finite = arr[np.isfinite(arr)]
                    ctx = float(np.max(np.abs(finite))) if finite.size else float("nan")
                    return "collected_action", ctx
        nv = getattr(self.model.logger, "name_to_value", None) or {}
        for key in self.loss_keys:
            val = nv.get(key, None)
            if val is not None and not np.isfinite(float(val)):
                return key, float(val)
        return None

    def _on_step(self) -> bool:
        if self._tripped:
            return True
        hit = self._first_nonfinite()
        if hit is None:
            return True

        self._tripped = True
        quantity, value = hit
        self.model.logger.record("guard/nonfinite", 1.0)
        if self.csv_path is not None:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([int(self.num_timesteps), quantity, value])

        if self.verbose:
            bar = "=" * 72
            _safe_print(f"\n{bar}")
            _safe_print(f"[NonFiniteGuard] NON-FINITE detected at step "
                        f"{self.num_timesteps:,}: {quantity} = {value}.")
            _safe_print("[NonFiniteGuard] The deterministic actor has collapsed to the 0 mm "
                        "corner and driven\n                 the twin-Q critic into a "
                        "deadly-triad (inf -> NaN) divergence (the v2.19 mode).")
            if self.stop_on_nonfinite:
                _safe_print("[NonFiniteGuard] stop_on_nonfinite=True -> stopping cleanly so the "
                            "cause is logged on\n                 stdout instead of dying on a "
                            "(possibly uncaptured) stderr traceback.")
            _safe_print(bar)
        return not self.stop_on_nonfinite
