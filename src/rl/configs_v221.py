# src/rl/configs_v221.py  v2.21.0
# -----------------------------------------------------------------------------
# v2.21 = v2.20 Run A (exact n-step, which converged) + ONE change: the biomass
# reward r1 switches to the GAMMA-CORRECT potential-based-shaping form. One variable.
#
# r1 today (v2.20):   (x4_t - x4_{t-1}) / X4_REF        -- increment, telescopes
#                     to terminal yield ONLY at gamma=1.
# r1 in v2.21:        (gamma * x4_t - x4_{t-1}) / X4_REF -- the policy-invariant
#                     potential-shaping form (Ng, Harada & Russell 1999) with
#                     potential Phi = ALPHA1*x4/X4_REF.
#
# WHY this is the exact fix and why it MATCHES MPC
# ------------------------------------------------
# MPC's biomass objective (src/mpc/cost.py) is a PURE TERMINAL (Mayer) term:
#     J_biomass = -alpha1 * x4_terminal / x4_ref.
# The v2.20 dense increment was its gamma=1 equivalent (telescopes to x4_T - x4_0),
# but at gamma=0.99 it splits into gamma^T*x4_T + (1-gamma)*sum_t gamma^t*x4_t -- the
# second term front-loads growth and starves the reproductive phase (the v2.20 Run A
# drought seesaw, worst at moderate/70%). With the gamma-correct form, the discounted
# biomass return telescopes EXACTLY:
#     sum_t gamma^t (gamma*x4_{t+1} - x4_t)/X4_REF = (gamma^T*x4_T - x4_0)/X4_REF,
# i.e. a PURE terminal-yield objective with NO front-loading term -- provably the
# same objective MPC optimises, while staying dense/learnable. gamma is untouched, so
# the bootstrap horizon (and the stability n-step bought) is unchanged.
#
# The shaping gamma must equal the per-step return discount. We do NOT hard-code it
# here; the trainer sets env biomass_shaping_gamma = gamma_base (0.99) when
# biomass_shaping=True, so the two can never drift.
#
# Compatibility: backward-compatible (env param defaults to 1.0 = old behaviour, so
# SAC/v2.19b/v2.20 are byte-identical). The eval harness (exp_rl -> run_season)
# measures PHYSICAL outcomes and never uses the gym reward, so yield/drought/waterlog
# and the MPC comparison are unaffected -- this changes ONLY how the policy is trained.
# Note: training reward / q_pred SCALE shifts (biomass return ~gamma^T*x4_T, smaller),
# which is internally consistent -- judge stability by boundedness, not absolute value.
#
# Pulsing / Markov-r5 is a separate later version (needs a 9-feature network).
# -----------------------------------------------------------------------------
from __future__ import annotations

GAMMA_BASE = 0.99

RUN_A = dict(
    label="gshape",
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
    # --- the ONLY change in v2.21 ---
    biomass_shaping=True,   # trainer sets env biomass_shaping_gamma = gamma_base (0.99)
)

CONFIGS = {"A": RUN_A}
