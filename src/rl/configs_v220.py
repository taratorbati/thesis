# src/rl/configs_v220.py  v2.20.0  (Stage 1)
# -----------------------------------------------------------------------------
# Stage-1 run definitions for the TD3 stabilisation experiments.  Place in
# src/rl/ alongside train_v220_td3.py, which reads CONFIGS[name].
#
# DESIGN RULE: change ONE major thing per run so any effect is attributable.
#   Run A = n-step ALONE (root-cause fix for the bootstrap-horizon divergence).
#   Run B = Run A + the TD3 "damping" package (loop-gain reduction).
# Both START from the configuration that DIVERGED (v2.20 r5: learning_starts
# 50k, r5 active), so a bounded q_pred is direct evidence the change fixed it.
#
# WHY n=5 (Run A)
# --------------
# n-step replaces n bootstrap steps with n grounded rewards, shrinking the
# self-referential target term to gamma^n.  Hessel et al. 2018 (Rainbow) found
# n=3 optimal on Atari; n=5 is a slightly longer horizon, justified here by the
# 93-step season and the project's own evidence that horizon length is the
# divergence driver (v2.6 ~50-step horizon: |Q| bounded 500k steps; v2.7 full
# 93-step horizon: cascade).  n=3 is the conservative fallback if n=5 is noisy.
# The discount is applied EXACTLY (gamma^n bootstrap) via the model-gamma trick
# in train_v220_td3.py + NStepReplayBufferExact -- not the gamma^1 approximation
# the v2.10 E3 buffer used.
#
# WHY the Run-B damping knobs (all TD3's OWN structural stabilisers, not the
# "Mistake 4" delay tactics of lowering global LR / tau / grad-norm)
# ------------------------------------------------------------------
#   policy_delay 2 -> 3   : Fujimoto et al. 2018 (TD3) default is 2; delaying the
#                           actor relative to the critic reduces actor-update
#                           variance.  Mild increase -> more critic settling per
#                           actor step.
#   target_policy_noise 0.2 -> 0.3 (clip 0.5 unchanged): TD3's target smoothing
#                           (Fujimoto 2018, sigma=0.2/clip=0.5) regularises sharp
#                           Q corners; widening it flattens the 0/12 bang-bang
#                           troughs that feed both the pulsing and the divergence.
#                           networks_td3.py notes target smoothing is TD3's
#                           designated replacement for the entropy that gave.
#   actor_warmup_updates 0 -> 25_000: "critic-leads" LR warm-up (see
#                           td3_warmup.py).  Targets the verified finding that the
#                           dip onset tracks learning_starts -- give the critic a
#                           head start before the deterministic actor moves.
#
# expose_prev_u stays False for ALL of Stage 1 (it is a Stage-2 reward change and
# additionally needs a 9-feature network; see gym_env_prev_u.py).  The trainer
# raises if it is True without that network change.
# -----------------------------------------------------------------------------

from __future__ import annotations

# Return / environment discount used to accumulate the n-step return R_n AND as
# the base for the model's gamma^n bootstrap.  Matches v2.19b GAMMA.
GAMMA_BASE = 0.99


# --- Run A: n-step alone -----------------------------------------------------
RUN_A = dict(
    label="nstep5",
    # n-step (the single change vs the diverging v2.20 r5 run)
    n_steps=5,
    gamma_base=GAMMA_BASE,
    # match the DIVERGING run for clean attribution (v2.19b default is 25_000)
    learning_starts=50_000,
    reward_du_alpha=0.005,            # r5 active, as in v2.20 r5
    # TD3 stabilisers held at v2.19b stock so n-step is isolated
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    actor_lr_mult=1.0,
    actor_warmup_updates=0,
    # Stage-2 switch (needs matching network change) -- OFF
    expose_prev_u=False,
)


# --- Run B: Run A + damping package -----------------------------------------
# Inherit Run A, override only the three damping knobs.
RUN_B = dict(RUN_A)
RUN_B.update(
    label="nstep5_damped",
    policy_delay=3,
    target_policy_noise=0.3,
    actor_warmup_updates=25_000,
)


CONFIGS = {
    "A": RUN_A,
    "B": RUN_B,
}
