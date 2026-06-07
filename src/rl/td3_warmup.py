# src/rl/td3_warmup.py  v2.20.0  (Stage 1, Run B "damping")
# -----------------------------------------------------------------------------
# WarmupAsymmetricLRTD3: TD3 with an actor learning-rate WARM-UP ("critic-leads").
#
# Place this file in src/rl/ alongside train_v219b_td3.py.  It is a drop-in
# replacement for the v2.19b AsymmetricLRTD3: same asymmetric-LR override, PLUS
# an optional ramp that holds the actor LR low (->0) for the first
# `actor_warmup_updates` gradient updates and linearly raises it to full.  With
# actor_lr_mult == 1.0 AND actor_warmup_updates == 0 the override is a verified
# no-op and the model behaves exactly like stock SB3 TD3.
#
# WHY (the "let the critic lead" lever)
# -------------------------------------
# Two verified facts about the v2.20 r5 / Run-A / 06-04 dives motivate this:
#   1. The q_pred dip ONSET tracks `learning_starts` in all four runs
#      (~27k / ~51k / ~26k / ~51k for ls 25/50/25/50k): the instability is
#      seeded the moment gradient updates begin and the deterministic actor
#      starts moving on a still-untrained critic.
#   2. v2.9 showed the trajectory is chaotically sensitive to the EARLIEST
#      gradient steps (a single perturbed step at ~2.2k changed the basin).
# A bare TD3 actor with no entropy term sprints to the 0 mm boundary before the
# critic is a usable estimator (this is exactly why v2.19b dropped the 5x actor
# LR to 1x).  Warming the actor up lets the critic fit a reasonable value
# landscape FIRST, so the actor climbs a real surface instead of noise -- the
# practical form of the deadly-triad rule "don't bootstrap an actor off an
# uncalibrated critic."  The critic LR is left at full from step one, so only
# the actor is delayed; this is gentler and more targeted than lowering the
# global LR (which the project's "Mistake 4" critique correctly called a delay
# tactic that postpones rather than prevents the cascade).
#
# WHY AN LR-SCHEDULE OVERRIDE, NOT A CALLBACK
# -------------------------------------------
# v2.9's lesson: SB3 calls `_update_learning_rate` at the top of EVERY train()
# call and re-sets each optimizer's LR from the schedule, silently reverting any
# LR a callback wrote between updates.  The override hook is the only place an
# actor-LR change actually sticks.  (The v2.19b AsymmetricLRTD3 uses the same
# hook for the same reason.)
#
# References
# ----------
# Fujimoto et al. 2018  (TD3) -- delayed policy updates; the actor should lag
#                       the critic.  This generalises that idea to the LR.
# Goyal et al. 2017     "Accurate, Large Minibatch SGD" -- linear LR warm-up as
#                       a standard remedy for early-training instability.
# Fujimoto/SB3 AsymmetricLRTD3 (this repo, train_v219b_td3.py) -- the override
#                       pattern and the mult==1.0 no-op guarantee.
# -----------------------------------------------------------------------------

from __future__ import annotations

from stable_baselines3 import TD3


class WarmupAsymmetricLRTD3(TD3):
    """TD3 with an asymmetric actor LR multiplier and an actor-LR warm-up ramp.

    Configure via class attributes BEFORE construction (mirrors how v2.19b sets
    ``AsymmetricLRTD3.actor_lr_mult``)::

        WarmupAsymmetricLRTD3.actor_lr_mult       = 1.0     # steady-state mult
        WarmupAsymmetricLRTD3.actor_warmup_updates = 25_000 # ramp length (updates)
        model = WarmupAsymmetricLRTD3(policy=..., env=..., ...)

    Semantics (applied to the ACTOR optimizer only, every train() call, AFTER
    SB3 has re-set both LRs from the schedule -- so nothing compounds):

        actor_lr <- schedule_lr * actor_lr_mult * min(1, n_updates / warmup)

    * actor_lr_mult == 1.0 and actor_warmup_updates == 0  -> exact no-op.
    * critic LR is never touched (it leads from step one).
    * the ramp is keyed to ``self._n_updates`` (cumulative gradient updates), so
      with gradient_steps=1, train_freq=1 it spans ~`actor_warmup_updates` env
      steps immediately after `learning_starts`.
    """

    # Defaults reproduce stock TD3 (no asymmetry, no warm-up).
    actor_lr_mult: float = 1.0
    actor_warmup_updates: int = 0

    def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
        # 1) Let SB3 set BOTH actor and critic LR from the schedule.
        super()._update_learning_rate(optimizers)

        mult = float(self.actor_lr_mult)
        warm = int(self.actor_warmup_updates)

        # 2) Exact no-op when neither feature is configured.
        if mult == 1.0 and warm <= 0:
            return

        # 3) Combined actor factor = steady-state mult * linear warm-up ramp.
        factor = mult
        if warm > 0:
            # self._n_updates reflects updates completed in PRIOR train() calls
            # (this hook runs before the current call increments it), so the
            # ramp is monotone: 0 at the first post-learning_starts update,
            # reaching `mult` after `warm` updates.
            factor *= min(1.0, float(self._n_updates) / float(warm))

        if factor == 1.0:
            return

        # 4) Apply to the actor optimizer only; critic LR stays at schedule.
        for pg in self.actor.optimizer.param_groups:
            pg["lr"] = pg["lr"] * factor
