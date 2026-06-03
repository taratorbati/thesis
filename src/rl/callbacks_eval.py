# src/rl/callbacks_eval.py  v2.19c
# =============================================================================
# Deterministic held-out evaluation for best_model selection.
#
# WHY THIS EXISTS
# ---------------
# The v2.14-v2.19b training scripts evaluated on an env built with
# `randomize=True`, seeded once at construction and never re-seeded between
# evaluations.  Each EvalCallback invocation therefore drew a DIFFERENT
# (year, budget) sample from the 20 training years, so:
#   * the eval reward at 25k, 50k, 75k, ... was computed on different episodes
#     and was NOT comparable across checkpoints (best_model selection and the
#     learning curve were both confounded by which years were drawn);
#   * the evaluation was performed on TRAINING years, not a held-out set, so it
#     measured fit rather than generalization;
#   * the run was not reproducible (the eval draw depended on RNG advancement).
#
# THE FIX
# -------
# Pair this callback with an IrrigationEnv constructed with `eval_schedule=...`
# (a fixed list of (year, budget_frac) pairs over the held-out DEV_YEARS).  The
# env's reset() then walks that schedule deterministically.  Before EVERY
# evaluation this callback rewinds the schedule to index 0, so each checkpoint
# is scored on the IDENTICAL fixed episode set in the same order -- a
# comparable, reproducible, held-out selection signal.
#
# WHY REWIND (and not just len(schedule) == n_eval_episodes)
# ----------------------------------------------------------
# SB3's evaluate_policy() performs one initial reset plus one auto-reset per
# finished episode (the final auto-reset is wasted), so without a rewind the
# env's schedule index drifts by n_eval_episodes+1 each call and episode k
# would start from a rotated schedule entry.  The per-checkpoint MEAN would
# still cover the same episodes, but per-episode logs would not align and the
# guarantee would silently depend on the exact reset bookkeeping.  Rewinding to
# index 0 first makes the alignment exact and bookkeeping-independent.
# =============================================================================

from __future__ import annotations

from stable_baselines3.common.callbacks import EvalCallback


class FixedScheduleEvalCallback(EvalCallback):
    """EvalCallback that rewinds the eval env's deterministic schedule before
    each evaluation.

    Requires `eval_env` to wrap an IrrigationEnv constructed with
    `eval_schedule=...` (which exposes `reset_eval_schedule()` via
    VecEnv.env_method).  Apart from the rewind, behaviour is identical to the
    stock EvalCallback (same best_model saving, logging, and callbacks).
    """

    def _on_step(self) -> bool:
        # This condition matches exactly when the parent EvalCallback triggers
        # evaluate_policy().  n_calls is incremented by BaseCallback.on_step
        # BEFORE _on_step runs, and super()._on_step() re-checks the same
        # condition -- so the rewind fires immediately before (and only before)
        # each evaluation.  Rewinding here guarantees the evaluation walks
        # schedule[0 .. n_eval_episodes-1] in order every time, regardless of
        # how many resets the previous evaluation consumed.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.env_method("reset_eval_schedule")
        return super()._on_step()
