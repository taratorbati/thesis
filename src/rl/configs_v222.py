# src/rl/configs_v222.py  v2.22  (Markov-r5: prev_u + 9-feature network)
# -----------------------------------------------------------------------------
# v2.22 = v2.21c (additive terminal-yield -- the seed-validated best controller:
# MPC yield parity, beats MPC on waterlog/water, mod/70% fixed) + prev_u: u_{t-1}
# is added to the per-agent observation so the delta-u smoothing reward r5 becomes
# MARKOV. This is the ONE new variable vs v2.21c. It targets the only remaining
# gap -- PULSING (mean|du| ~2.35 vs MPC's 0.97); r5 could not work before because
# a deterministic actor could not observe u_{t-1}.
#
# Uses the 9-feature actor+critic (networks_td3_prevu), byte-identical to v2.21c's
# architecture except the per-agent input width (8->9 features). All v2.21c reward
# settings are kept (biomass_shaping=False, reward_terminal_yield=1.0).
#
# reward_du_alpha stays at 0.005 (the value set when r5 was NON-Markov, i.e. inert).
# Now that the agent can act on it, 0.005 may already bite; if pulsing does NOT
# drop meaningfully, the next lever is raising reward_du_alpha (try 0.02-0.05) --
# a one-line change in this file, no other files needed.
# -----------------------------------------------------------------------------
from __future__ import annotations

GAMMA_BASE = 0.99

RUN_A = dict(
    label="markovr5",
    # --- inherited from v2.21c (validated best) ---
    n_steps=5,
    gamma_base=GAMMA_BASE,
    learning_starts=50_000,
    reward_du_alpha=0.02,          # r5 weight; now ACTIONABLE (prev_u in obs)
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    actor_lr_mult=1.0,
    actor_warmup_updates=0,
    biomass_shaping=False,          # keep the dense increment r1
    reward_terminal_yield=1.0,      # keep the additive terminal-yield term
    # --- the ONE v2.22 change ---
    expose_prev_u=True,             # -> IrrigationEnvPrevU + 9-feature network
)

CONFIGS = {"A": RUN_A}
