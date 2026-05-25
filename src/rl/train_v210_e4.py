# src/rl/train_v210_e4.py  v2.10.0 (E4 - gamma reduction pilot)
# -----------------------------------------------------------------------------
# E4 experiment: v2.7 SAC architecture + gamma reduced from 0.99 to 0.98.
#
# Why this run exists
# -------------------
# E2 (TQC + k=5 truncation, n_step=1) and E3 (TQC + k=5 + n_step=3 custom
# buffer) both failed.  Their post-mortems converged on three independent
# findings that condemn the "TQC + n-step" path for this project:
#
#   1. The VDN-summed critic produces a near-degenerate quantile distribution
#      (quantile_spread <= 4 in the stable phase) via central-limit narrowing
#      across 130 agents.  TQC truncation has nothing to chop.  (E2 evidence;
#      THESIS_HANDOFF_v5 Section 2.3.)
#
#   2. The custom NStepReplayBuffer drops the soft-Bellman entropy bonus on
#      the n-1 intermediate steps.  TQC.train() applies the entropy term
#      (-alpha * log pi) only at the bootstrap step (step t+n in the n-step
#      target), but the soft-Bellman recursion requires that bonus at every
#      intermediate step too.  With alpha=0.05 and -log pi ~ 80, this is ~8
#      units of negative target bias per stored transition.  Drives Q
#      downward without bound.  (E3 log evidence: q_inflation_pct = -44%
#      at step 25k, -243% at step 75k, -339% at step 100k.)
#
#   3. The gamma^1 vs gamma^n approximation in the same buffer (acknowledged
#      in nstep_buffer.py docstring) shifts the soft-Bellman fixed point but
#      is not by itself the cascade driver.  It is a 2% multiplicative bias
#      on a stable contraction operator.  Real but secondary to (2).
#
# The cleanest fix for the bootstrap-leverage cascade hypothesis (Strategy D
# in THESIS_HANDOFF_v5 Section 3.2) is to attack bootstrap amplification at
# the algorithm level: lower gamma.
#
#   gamma = 0.99 -> effective horizon 1 / (1 - gamma) = 100 steps
#   gamma = 0.98 -> effective horizon 50 steps        (this run)
#   gamma = 0.97 -> effective horizon 33 steps        (fallback)
#
# Episode length is 93 days.  v2.7 with gamma=0.99 propagates Q bias across
# the full 93 steps with leverage ~100; the cascade ignites at step ~156k
# once the inflated fixed point loses local stability.  gamma=0.98 cuts the
# leverage to ~50 while still permitting cross-phenological-phase credit
# assignment (~50 days bridges vegetative to reproductive, or reproductive
# to ripening, but not full vegetative-to-ripening).
#
# We choose 0.98 over 0.97 because this is a stability experiment: the
# minimum deviation from v2.7 that breaks the cascade is the experimentally
# cleanest result.  If 0.98 stabilises training without yield regression on
# dry/moderate cells, the published in-distribution yields will remain close
# to v2.7's.  If 0.98 fails (cascade still fires before step 250k), the
# fallback is gamma=0.97 with a single constant change in this file.
#
# What changed vs v2.7
# --------------------
# Exactly one change:
#   GAMMA = 0.99   ->   GAMMA = 0.98
#
# What did NOT change vs v2.7
# ---------------------------
# - Algorithm: SAC (not TQC; TQC truncation is structurally inert here)
# - Replay buffer: standard SB3 ReplayBuffer (NOT the custom NStepReplayBuffer)
# - Architecture: V27CTDESACPolicy (1097-dim obs, 8 features/agent)
# - Critic: VDN-factorised twin-Q (sum across 130 agents)
# - ent_coef: 0.05 FIXED (auto-tuning is the v2.5 failure mode)
# - tau:     0.005 (Haarnoja 2018 default)
# - LR schedule: 3e-4 -> 5e-5 linear
# - Batch size: 256
# - Buffer size: 250 000
# - Total steps: 250 000
# - max_grad_norm: 1.0 via GradClipCallback
# - Environment: IrrigationEnv(randomize=True, curriculum_warmup_steps=0,
#                              use_overshoot_feature=False)
# - Reward: r1 + r2 + r3 + r6, np.mean aggregations across 130 agents
#           (unchanged; the published v2.7 yields are reproducible with
#           this reward and the proposal to "fix VDN scale by switching
#           mean to sum" is rejected as it would invalidate the baseline)
#
# Hyperparameter citations
# ------------------------
# - gamma=0.98: Strategy D in THESIS_HANDOFF_v5 Section 3.2 (gamma=0.97 or
#   0.95 listed as the deadly-triad-cascade backup).  0.98 chosen as the
#   minimum deviation from the v2.7 gamma=0.99 baseline; if insufficient,
#   the fallback path 0.98 -> 0.97 is a one-line edit.
# - All other hyperparameters: v2.7 baseline (THESIS_HANDOFF_v4 Section 5.3).
#
# Diagnostics
# -----------
#   - BiasRatioCallback         : Q_pred / Q_structural every 25k steps.
#                                  Note: with gamma=0.98 the structural
#                                  baseline shrinks (geom_weight = 50 vs 93
#                                  at gamma=0.99) so absolute Q_structural
#                                  will be smaller; the percentage threshold
#                                  (q_inflation_pct < 20% acceptance) is
#                                  unchanged.
#   - ActionStatsCallback       : spatial std of u across 130 agents
#   - OptimizerLRCallback       : actual optimizer LR (v2.9 bug detector)
#   - EvalCallback              : 9-episode eval every 25k steps
#   - CheckpointCallback        : every 25k steps
#   - RotatingReplayBufferCheckpoint : single rotating replay-buffer dump
#
# Acceptance criterion
# --------------------
# |q_inflation_pct| < 20% at step 250k.  If at any checkpoint past step
# 150k both q_inflation_pct > 100% AND actor/std/spatial < 0.15, kill the
# run.  The fallback is gamma=0.97 (change GAMMA below) or accept v2.7 as
# the publication baseline.
#
# Outputs to: <output_dir>/sac_v210_e4_seed{seed}/
#     best_model/best_model.zip      <- EvalCallback best
#     checkpoints/*.zip              <- every 25k steps
#     eval_logs/                     <- EvalCallback per-eval npz logs
#     tensorboard/                   <- SB3 TB writer
#     bias_ratio_log.csv             <- cascade diagnostic
#     replay_buffer_latest.pkl       <- rotating dump
#     sac_v210_e4_seed{seed}_final.zip
#
# Eval is performed with the v2.7 SAC eval script (NOT the TQC one):
#   python -m scripts.experiments.exp_rl --mode eval \
#       --model <save_dir>/best_model/best_model.zip \
#       --scenario all --budget all --forecast perfect
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import V27CTDESACPolicy, make_sac_policy_kwargs
from src.rl.callbacks_v210 import (
    BiasRatioCallback,
    ActionStatsCallback,
    OptimizerLRCallback,
)
# Re-use the v2.7 infrastructure helpers and rotating-buffer callback.
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)


# -----------------------------------------------------------------------------
# Hyperparameters.  Single change vs v2.7: GAMMA 0.99 -> 0.98.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000     # v2.7 baseline
BUFFER_SIZE      = 250_000     # v2.7 baseline
BATCH_SIZE       = 256         # v2.7 baseline

GAMMA            = 0.98        # E4 change: was 0.99 in v2.7.
                               # Fallback if E4 fails: set to 0.97.

TAU              = 0.005       # v2.7 baseline; Haarnoja et al. 2018 default
LR_START         = 3e-4        # v2.7 baseline
LR_END           = 5e-5        # v2.7 baseline; linear decay

ENT_COEF         = 0.05        # FIXED.  Auto-tuning is the v2.5 failure
                               # mode (entropy collapse, see HANDOFF_v4
                               # Section 4.3).
MAX_GRAD_NORM    = 1.0         # v2.7 baseline
LEARNING_STARTS  = 1_000       # v2.7 baseline
GRADIENT_STEPS   = 1           # v2.7 baseline
TRAIN_FREQ       = 1           # one gradient step per env step

EVAL_FREQ        = 25_000      # v2.7 baseline
N_EVAL_EPISODES  = 9           # v2.7 baseline
CHECKPOINT_FREQ  = 25_000      # match v2.10 cadence (was 50_000 in v2.7;
                               # finer cadence helps locate cascade onset)

ACTOR_HIDDEN  = [128, 128]     # v2.7 baseline
CRITIC_HIDDEN = [256, 256]     # v2.7 baseline

# Diagnostics frequencies
BIAS_RATIO_FREQ          = 25_000
BIAS_RATIO_N_EPISODES    = 3
ACTION_STATS_FREQ        = 1_000
LR_LOG_FREQ              = 1_000


def train_sac_e4(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
) -> SAC:
    """Train a SAC v2.10 E4 agent (v2.7 architecture + gamma reduction).

    Parameters
    ----------
    seed : int
        Random seed.  Use seed 0 first for paired comparison against v2.7
        seed 0 published numbers; expand to seeds 1, 2 only if seed 0 results
        meet the acceptance criterion.
    output_dir : str
        Root directory for checkpoints and best-model artefacts.  Results go
        to <output_dir>/sac_v210_e4_seed{seed}/.
    wandb_project : str or None
        WandB project name.  None disables WandB logging entirely.
    total_timesteps : int
        Default 250_000 (matches v2.7 / v2.9).
    gamma : float
        Discount factor.  Defaults to GAMMA constant above (0.98).  Override
        to 0.97 for the fallback experiment without code edits.

    Returns
    -------
    SAC
        The trained model.  Also saved to disk as
        <output_dir>/sac_v210_e4_seed{seed}/sac_v210_e4_seed{seed}_final.zip.

    Notes
    -----
    The returned model is loadable via the existing v2.7 SAC eval script:
        python -m scripts.experiments.exp_rl --mode eval \
            --model <best_model.zip> --scenario all --budget all --forecast perfect

    The runner's architecture auto-detection (runner.py:_detect_critic_arch)
    will recognise this checkpoint as a v2.7-format VDN SAC and load it via
    the V27CTDESACPolicy path.
    """
    run_name = f"sac_v210_e4_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Config dict (logged to WandB for provenance).
    # -------------------------------------------------------------------------
    config = {
        "version": "2.10.0-E4",
        "experiment": "E4_SAC_gamma_reduction",
        "seed": seed,
        "algorithm": "SAC (stable_baselines3)",
        "policy_class": "V27CTDESACPolicy (VDN twin-Q)",
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": gamma,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "ent_coef": ENT_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "actor_hidden": ACTOR_HIDDEN,
        "critic_hidden": CRITIC_HIDDEN,
        "learning_starts": LEARNING_STARTS,
        "gradient_steps": GRADIENT_STEPS,
        "train_freq": TRAIN_FREQ,
        "eval_freq": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "obs_dim": 1097,
        "n_agent_features": 8,
        "curriculum": "NONE (v2.7 baseline)",
        "replay_buffer_class": "stable_baselines3.common.buffers.ReplayBuffer",
        "n_step": 1,
        "changes_vs_v27": [
            f"gamma {0.99} -> {gamma} (E4: bootstrap leverage reduction, "
            "Strategy D in HANDOFF_v5 Section 3.2)",
            f"checkpoint_freq {CHECKPOINT_FREQ} (was 50_000 in v2.7)",
        ],
    }

    # -------------------------------------------------------------------------
    # WandB
    # -------------------------------------------------------------------------
    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # -------------------------------------------------------------------------
    # Environments - v2.7 layout exactly (no curriculum, no overshoot feat).
    # -------------------------------------------------------------------------
    train_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    bias_eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
        use_overshoot_feature=False,
    )])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    # -------------------------------------------------------------------------
    # Policy kwargs - v2.7 VDN factorised critic.
    # -------------------------------------------------------------------------
    policy_kwargs = make_sac_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # -------------------------------------------------------------------------
    # SAC model.
    #
    # CRITICAL: ent_coef=ENT_COEF (0.05 fixed) - NOT 'auto'.  SAC's default
    # is 'auto' which is the v2.5 failure mode (entropy collapse).
    #
    # No replay_buffer_class override: this uses SB3's standard 1-step
    # ReplayBuffer.  The custom NStepReplayBuffer (src/rl/nstep_buffer.py)
    # was the source of the E3 cascade; it is intentionally NOT used here.
    # -------------------------------------------------------------------------
    model = SAC(
        policy=V27CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        tau=TAU,
        ent_coef=ENT_COEF,                                # FIXED (not 'auto')
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(save_dir / "checkpoints"),
        name_prefix=run_name,
        save_replay_buffer=False,
        verbose=1,
    )
    rotating_buffer_callback = RotatingReplayBufferCheckpoint(
        save_freq=CHECKPOINT_FREQ,
        save_path=save_dir,
        verbose=1,
    )
    grad_clip_callback = GradClipCallback(max_grad_norm=MAX_GRAD_NORM)

    # v2.10 diagnostics
    bias_ratio_cb = BiasRatioCallback(
        eval_env=bias_eval_env,
        eval_freq=BIAS_RATIO_FREQ,
        n_eval_episodes=BIAS_RATIO_N_EPISODES,
        save_path=str(save_dir),
        verbose=1,
    )
    action_stats_cb = ActionStatsCallback(log_freq=ACTION_STATS_FREQ)
    optimizer_lr_cb = OptimizerLRCallback(log_freq=LR_LOG_FREQ)

    cb_list = [
        eval_callback,
        checkpoint_callback,
        rotating_buffer_callback,
        grad_clip_callback,
        bias_ratio_cb,
        action_stats_cb,
        optimizer_lr_cb,
    ]

    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            wandb_cb = WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ,
                verbose=0,
            )
            cb_list.append(wandb_cb)
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    callbacks = CallbackList(cb_list)

    # -------------------------------------------------------------------------
    # Banner
    # -------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  SAC training - v2.10.0 E4 (gamma reduction pilot) - seed {seed}")
    print(f"  Architecture: v2.7 obs (1097-dim, 8 features/agent)")
    print(f"  Algorithm:    SAC (stable_baselines3) + standard 1-step ReplayBuffer")
    print(f"  Entropy:      ent_coef={ENT_COEF} FIXED (auto-tuning disabled)")
    print(f"  GAMMA:        {gamma}  (was 0.99 in v2.7; E4 single change)")
    print(f"  LR schedule:  {LR_START:.0e} -> {LR_END:.0e} (linear)")
    print(f"  Total steps:  {total_timesteps:,}")
    print(f"  Checkpoint:   every {CHECKPOINT_FREQ:,} steps")
    print(f"  Output:       {save_dir}")
    print(f"{'='*72}\n")

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    finally:
        if wandb_active:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    final_path = save_dir / f"{run_name}_final"
    model.save(str(final_path))
    print(f"\n[train] Final model saved to {final_path}.zip")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Train SAC v2.10 E4 (v2.7 architecture with gamma reduced from "
            "0.99 to 0.98 - Strategy D in HANDOFF_v5)."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA,
                        help=(
                            "Discount factor.  Default 0.98 (E4 pilot).  Set "
                            "to 0.97 for the fallback experiment if 0.98 "
                            "still cascades."
                        ))
    args = parser.parse_args()

    train_sac_e4(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
    )
