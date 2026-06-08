# src/rl/configs_v221c.py  v2.21c
# -----------------------------------------------------------------------------
# v2.21c = v2.20 Run A (exact n-step + the dense INCREMENT biomass reward, the best
# baseline) + an ADDITIVE terminal-yield term. One variable on top of the proven run.
#
# Why additive (not the gamma-shaping form):
#   - v2.20 Run A (increment r1) was the best: 99.8% MPC yield, beat MPC on waterlog.
#   - The gamma-correct shaping (v2.21 / v2.21-v2) REPLACED r1 and made things worse:
#     it gutted the dense biomass signal, so the policy under-irrigated and mis-allocated.
#   - The additive terminal term KEEPS the full increment signal (biomass_shaping=False)
#     and simply ADDS alpha_T * x4_final/X4_REF once at episode end. It lifts the
#     endpoint weight from gamma^T (~0.39) to gamma^T*(1+alpha_T) (~0.78 at alpha_T=1.0)
#     -- pulling the objective toward MPC's terminal-biomass cost -- without removing
#     the dense signal. gamma is untouched, so n-step stability is preserved.
#
# Scales: gym_env.py reverts X4_REF 900->600 and ALPHA2 0.01->0.016 (the v2.20 Run A
# values); the 900 / 0.01 experiment was worse and broke the MPC alpha2 mirror.
# alpha_T = 1.0 is season-sum-calibrated (terminal bonus ~1.0*x4_final/600 ~= 1.37 ~=
# the cumulative dense-r1 season-sum ~1.27, the method used for ALPHA6_LIN). Tune in
# [0.5, 1.5]: lower if wet-year waterlog regresses or q_pred destabilises; higher if
# the drought seesaw does not move.
# -----------------------------------------------------------------------------
from __future__ import annotations

GAMMA_BASE = 0.99

RUN_A = dict(
    label="termyield",
    # --- inherited from v2.20 Run A (the exact-n-step stabiliser, unchanged) ---
    n_steps=5,
    gamma_base=GAMMA_BASE,
    learning_starts=50_000,
    reward_du_alpha=0.005,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    actor_lr_mult=1.0,
    actor_warmup_updates=0,
    expose_prev_u=False,
    # --- v2.21c ---
    biomass_shaping=False,       # KEEP the dense increment r1 (v2.20 form) -- do NOT gamma-shape
    reward_terminal_yield=1.0,   # the ONLY change: ADDITIVE terminal-yield bonus
)

CONFIGS = {"A": RUN_A}
