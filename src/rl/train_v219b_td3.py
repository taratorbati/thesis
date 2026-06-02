# src/rl/train_v219b_td3.py  v2.19b-TD3  (corrected v2.19 run)
# -----------------------------------------------------------------------------
# Re-run of the TD3 controller after diagnosing why v2.19 collapsed.
#
# WHAT WENT WRONG IN v2.19 (diagnosis, re-derived from committed eval files):
#   v2.19 set out to remove the SAC entropy "action-pin" so the deterministic
#   actor could reach the tanh=-1 boundary (0 mm) on wet days.  It worked too
#   well: the actor collapsed to ~0 mm in EVERY scenario (dry/moderate/wet),
#   not just on wet days.
#     * eval mean-reward was flat at ~-6 for all 250k steps (never improved;
#       SAC v2.18 sat near 0);
#     * the twin-Q critic diverged: q_pred_mean -21 -> -57 -> -105 -> -145 while
#       the realised return was only ~-4 (monotone negative miscalibration);
#     * the trained policy applied <0.5 mm on 69% (dry) / 93% (mod) / 75% (wet)
#       of days, leaving x1 near wilting point and roughly halving yield
#       (dry 1836 vs SAC 4205 kg/ha; water 85 vs 484 mm).
#   ROOT CAUSE: the SAC entropy term was load-bearing for EXPLORATION and
#   replay-buffer coverage, not just for the action-pin.  v2.19 removed it but
#   replaced exploration with only a weak, fast-decaying NormalActionNoise
#   (0.20 -> 0.05 over 100k) plus target-policy smoothing.  Target smoothing
#   only perturbs the Bellman-target action; it provides NO state-action
#   coverage.  So the deterministic actor drifted to the 0 mm corner, the buffer
#   filled with low-water/drought transitions, twin-min pessimism kept the
#   now-unexplored higher-water actions looking bad, and there was no entropy
#   floor or sufficient noise to recover -- a self-reinforcing collapse (exactly
#   the buffer-starvation loop of THESIS_HANDOFF_v6 sec 4.4b, mirrored to the
#   low-water corner).  Contributing factors: learning_starts=1000 (only ~10
#   random episodes before the deterministic actor exploited a near-random
#   critic) and the asymmetric 5x actor LR (made the actor sprint to the
#   boundary faster than the critic could self-correct -- a useful trick under
#   SAC's entropy regulariser, dangerous for a bare deterministic policy).
#
# WHAT v2.19b CHANGES vs v2.19 (only the four anti-collapse fixes; ARCHITECTURE
# AND ENV ARE BYTE-IDENTICAL -- same _TD3SharedActor, same V211 LayerNorm VDN
# critic, same 1097-dim obs, same reward, marker still 2.19):
#   1. learning_starts: 1_000 -> 25_000.  Fill the buffer with random data and
#      give the critic a real warm start before the deterministic actor begins
#      exploiting it (standard TD3 practice; 10% of training).
#   2. Exploration noise: 0.20 -> 0.05 over 100k  becomes  0.40 -> 0.15 over
#      150k, FLOOR HELD AT 0.15.  Higher start and -- crucially -- a sustained
#      floor 3x the old one, so coverage of the moderate/high-water region never
#      dries up (this is the single most important fix; TD3 has no entropy, so
#      collection noise is the ONLY thing keeping the buffer diverse).
#   3. actor_lr_mult: 5.0 -> 1.0 (symmetric LR).  Stop the actor from racing to
#      the boundary ahead of the critic.
#   4. Telemetry + fail-fast guard: add LowActionCoverageCallback (the v2.19 run
#      logged NO coverage at all) and CollapseGuardCallback, which aborts the
#      run if the rolling low-action fraction exceeds 60% after warmup -- so a
#      recurrence is caught around ~30-40k instead of wasting a full 250k run.
#
# TD3-internal knobs are unchanged from v2.19 / Fujimoto et al. 2018:
#   target_policy_noise=0.2, target_noise_clip=0.5, policy_delay=2, gamma=0.99,
#   tau=0.005, LR 3e-4 -> 5e-5, buffer 250k, 250k steps, RAIN_REF=30, linear r6.
#
# ACCEPTANCE CRITERIA (same spirit as v2.19, plus an explicit no-collapse gate):
#   NO-COLLAPSE (the v2.19 regression): eval mean-reward climbs above 0 and the
#     trained policy's dry-year u_mean stays >~4 mm (vs v2.19's 0.92).  The
#     CollapseGuard must NOT trip.
#   PRIMARY: wet x1 median < 134 mm (v2.18: 136; MPC: 130); wet waterlog < 32.
#   STABILITY: q_pred_mean never negative; |q_inflation_pct| < 80%; the critic
#     should now be calibrated because the buffer is no longer degenerate.
#   ATTRIBUTION: if v2.19b reaches MPC x1/waterlog cleanly, it confirms the
#     v2.19 failure was exploration starvation, not the entropy-removal idea.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import IrrigationEnv, RAIN_REF_V216
from src.rl.networks_td3 import TD3VDNPolicy, make_td3_policy_kwargs
from src.rl.callbacks_v210 import (
    BiasRatioCallback,
    ActionStatsCallback,
    OptimizerLRCallback,
)
from src.rl.callbacks_exploration import (
    ExplorationNoiseDecayCallback,
    LowActionCoverageCallback,
    CollapseGuardCallback,
)
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)


# -----------------------------------------------------------------------------
# AsymmetricLRTD3 -- TD3 mirror of AsymmetricLRSAC.  v2.19b defaults the
# multiplier to 1.0 (symmetric): the 5x actor LR that helped the entropy-
# regularised SAC actor made the BARE deterministic TD3 actor sprint to the 0 mm
# boundary ahead of the critic.  Kept tunable (>1.0 still works) but off by
# default.  With mult=1.0 the override is a harmless no-op (super() re-sets the
# LR from the schedule each call, so there is no compounding).
# -----------------------------------------------------------------------------
ACTOR_LR_MULT = 1.0


class AsymmetricLRTD3(TD3):
    actor_lr_mult: float = ACTOR_LR_MULT

    def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
        super()._update_learning_rate(optimizers)
        if self.actor_lr_mult == 1.0:
            return
        for pg in self.actor.optimizer.param_groups:
            pg["lr"] = pg["lr"] * self.actor_lr_mult


# -----------------------------------------------------------------------------
# Hyperparameters.
# -----------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256

GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5

MAX_GRAD_NORM    = 1.0
LEARNING_STARTS  = 25_000     # *** v2.19b CHANGE: 1_000 -> 25_000 ***
GRADIENT_STEPS   = 1
TRAIN_FREQ       = 1
POLICY_DELAY     = 2          # TD3 delayed actor updates

# TD3 target-policy smoothing (Fujimoto et al. 2018 defaults; unchanged).
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP   = 0.5

# Exploration (collection-time) noise -- TD3 has no entropy, so this is the SOLE
# exploration / buffer-coverage source.  *** v2.19b CHANGE: stronger start and a
# SUSTAINED floor (0.40 -> 0.15 over 150k, then 0.15 held) vs v2.19's anaemic
# 0.20 -> 0.05 over 100k. ***
EXPLORE_SIGMA_START = 0.40
EXPLORE_SIGMA_END   = 0.15
EXPLORE_DECAY_STEPS = 150_000
EXPLORE_LOG_FREQ    = 1_000

# Collapse guard (fail-fast on the v2.19 failure mode).
GUARD_COLLAPSE_FRAC = 0.60
GUARD_WARMUP_STEPS  = 30_000   # just past learning_starts
GUARD_CHECK_FREQ    = 2_000
GUARD_WINDOW        = 8
GUARD_ABORT         = True
COVERAGE_LOG_FREQ   = 1_000

EVAL_FREQ        = 25_000
N_EVAL_EPISODES  = 9
CHECKPOINT_FREQ  = 25_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]

BIAS_RATIO_FREQ       = 25_000
BIAS_RATIO_N_EPISODES = 3
ACTION_STATS_FREQ     = 1_000
LR_LOG_FREQ           = 1_000

N_AGENTS = 130

REWARD_OVERSHOOT_MODE = 'linear'
RAIN_NORMALISER       = RAIN_REF_V216   # 30.0


def train_td3_v219b(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    gamma: float = GAMMA,
    actor_lr_mult: float = ACTOR_LR_MULT,
    reward_overshoot_mode: str = REWARD_OVERSHOOT_MODE,
    rain_normaliser: float = RAIN_NORMALISER,
    target_policy_noise: float = TARGET_POLICY_NOISE,
    target_noise_clip: float = TARGET_NOISE_CLIP,
    policy_delay: int = POLICY_DELAY,
    learning_starts: int = LEARNING_STARTS,
    explore_sigma_start: float = EXPLORE_SIGMA_START,
    explore_sigma_end: float = EXPLORE_SIGMA_END,
    explore_decay_steps: int = EXPLORE_DECAY_STEPS,
    guard_collapse_frac: float = GUARD_COLLAPSE_FRAC,
    guard_warmup_steps: int = GUARD_WARMUP_STEPS,
    guard_abort: bool = GUARD_ABORT,
) -> TD3:
    """Train TD3 v2.19b: the v2.19 architecture with the four anti-collapse fixes.

    Identical actor/critic/env/obs/reward to v2.19; only the exploration
    schedule, learning_starts, actor LR multiplier, and telemetry/guard change.
    """
    run_name = f"td3_v219b_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.19.1-TD3b",
        "experiment": "v219b_td3_corrected_exploration_PLUS_v211_LN_critic",
        "seed": seed,
        "algorithm": "TD3 (stable_baselines3), symmetric LR, sustained noise floor",
        "policy_class": "TD3VDNPolicy (deterministic _TD3SharedActor, marker=2.19)",
        "total_timesteps": total_timesteps,
        "gamma": gamma,
        "tau": TAU,
        "actor_lr_mult": actor_lr_mult,
        "learning_starts": learning_starts,
        "target_policy_noise": target_policy_noise,
        "target_noise_clip": target_noise_clip,
        "policy_delay": policy_delay,
        "explore_sigma_start": explore_sigma_start,
        "explore_sigma_end": explore_sigma_end,
        "explore_decay_steps": explore_decay_steps,
        "guard_collapse_frac": guard_collapse_frac,
        "guard_warmup_steps": guard_warmup_steps,
        "guard_abort": guard_abort,
        "rain_normaliser": rain_normaliser,
        "reward_overshoot_mode": reward_overshoot_mode,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "fixes_vs_v219": (
            "learning_starts 1k->25k; explore noise 0.20->0.05/100k => "
            "0.40->0.15/150k (sustained floor); actor_lr_mult 5x->1x; add "
            "LowActionCoverage + CollapseGuard telemetry."
        ),
        "hypothesis": (
            "v2.19 collapsed because removing entropy removed exploration, not "
            "because the entropy-removal idea was wrong. Restoring sustained "
            "collection noise + a critic warm-start (and removing the actor LR "
            "race) should let TD3 reach low wet-day water WITHOUT the global "
            "collapse to 0 mm, giving a calibrated critic and MPC-level x1."
        ),
    }

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
        )

    train_env     = DummyVecEnv([_make_env])
    eval_env      = DummyVecEnv([_make_env])
    bias_eval_env = DummyVecEnv([_make_env])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_td3_policy_kwargs(
        N=N_AGENTS, actor_hidden=ACTOR_HIDDEN, critic_hidden=CRITIC_HIDDEN,
    )

    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    action_noise = NormalActionNoise(
        mean=np.zeros(N_AGENTS, dtype=np.float64),
        sigma=explore_sigma_start * np.ones(N_AGENTS, dtype=np.float64),
    )

    AsymmetricLRTD3.actor_lr_mult = float(actor_lr_mult)
    model = AsymmetricLRTD3(
        policy=TD3VDNPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=gamma,
        tau=TAU,
        action_noise=action_noise,
        policy_delay=policy_delay,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        learning_starts=learning_starts,
        gradient_steps=GRADIENT_STEPS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

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
        save_freq=CHECKPOINT_FREQ, save_path=save_dir, verbose=1,
    )
    grad_clip_callback = GradClipCallback(max_grad_norm=MAX_GRAD_NORM)

    # BiasRatioCallback works for any algorithm exposing .critic and a
    # deterministic policy via .predict (it compares the critic's Q-estimate to
    # the realised return).  TD3 satisfies both; the c2699e1 fix makes it
    # NaN-safe for entropy-free policies.
    bias_ratio_cb = BiasRatioCallback(
        eval_env=bias_eval_env,
        eval_freq=BIAS_RATIO_FREQ,
        n_eval_episodes=BIAS_RATIO_N_EPISODES,
        save_path=str(save_dir),
        verbose=1,
    )
    action_stats_cb = ActionStatsCallback(log_freq=ACTION_STATS_FREQ)
    optimizer_lr_cb = OptimizerLRCallback(log_freq=LR_LOG_FREQ)

    # *** v2.19b: sustained-floor exploration (replaces v2.19's anaemic decay). ***
    noise_decay_cb = ExplorationNoiseDecayCallback(
        sigma_start=explore_sigma_start,
        sigma_end=explore_sigma_end,
        decay_steps=explore_decay_steps,
        log_freq=EXPLORE_LOG_FREQ,
        csv_path=str(save_dir / "exploration_sigma_log.csv"),
        verbose=1,
    )

    # *** v2.19b: coverage telemetry the v2.19 run lacked entirely. ***
    coverage_cb = LowActionCoverageCallback(
        log_freq=COVERAGE_LOG_FREQ,
        csv_path=str(save_dir / "low_action_coverage_log.csv"),
        verbose=0,
    )

    # *** v2.19b: fail-fast guard against the v2.19 collapse. ***
    collapse_guard_cb = CollapseGuardCallback(
        collapse_frac=guard_collapse_frac,
        warmup_steps=guard_warmup_steps,
        check_freq=GUARD_CHECK_FREQ,
        window=GUARD_WINDOW,
        abort_on_collapse=guard_abort,
        csv_path=str(save_dir / "collapse_guard_log.csv"),
        verbose=1,
    )

    cb_list = [
        eval_callback,
        checkpoint_callback,
        rotating_buffer_callback,
        grad_clip_callback,
        bias_ratio_cb,
        action_stats_cb,
        optimizer_lr_cb,
        noise_decay_cb,
        coverage_cb,
        collapse_guard_cb,
    ]

    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            cb_list.append(WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ, verbose=0,
            ))
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    callbacks = CallbackList(cb_list)

    print(f"\n{'='*72}")
    print(f"  TD3 training - v2.19b (corrected) - seed {seed}")
    print(f"  Actor:  deterministic _TD3SharedActor (LeakyReLU + 2x-1 recenter, marker=2.19)")
    print(f"  Critic: _V211FactorizedContinuousCritic (VDN twin-Q + LayerNorm, UNCHANGED)")
    print(f"  FIXES vs v2.19: learning_starts={learning_starts:,} (was 1,000); "
          f"actorLRx{actor_lr_mult} (was 5)")
    print(f"  explore noise: {explore_sigma_start:.2f} -> {explore_sigma_end:.2f} over "
          f"{explore_decay_steps:,} (FLOOR HELD; was 0.20->0.05/100k)")
    print(f"  target_policy_noise={target_policy_noise}  clip={target_noise_clip}  "
          f"policy_delay={policy_delay}")
    print(f"  collapse guard: abort={guard_abort} if rolling low-action >= "
          f"{guard_collapse_frac:.0%} after {guard_warmup_steps:,} steps")
    print(f"  rain_norm={rain_normaliser:.1f}  r6={reward_overshoot_mode}  gamma={gamma}  tau={TAU}")
    print(f"  Total steps: {total_timesteps:,}  | Output: {save_dir}")
    print(f"{'='*72}\n")

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
            "Train TD3 v2.19b: the v2.19 deterministic VDN actor + v2.11 LayerNorm "
            "VDN critic, with the four anti-collapse fixes (learning_starts 25k, "
            "sustained exploration noise 0.40->0.15, symmetric actor LR, collapse "
            "guard).  Evaluated by runner.py via the same 2.19 TD3 marker."
        )
    )
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--output-dir",      type=str,   default="results/rl")
    parser.add_argument("--wandb-project",   type=str,   default=None)
    parser.add_argument("--total-timesteps", type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--gamma",           type=float, default=GAMMA)
    parser.add_argument("--actor-lr-mult",   type=float, default=ACTOR_LR_MULT)
    parser.add_argument("--reward-overshoot-mode", type=str, default=REWARD_OVERSHOOT_MODE,
                        choices=['quadratic', 'linear', 'sqrt'])
    parser.add_argument("--rain-normaliser", type=float, default=RAIN_NORMALISER)
    parser.add_argument("--target-policy-noise", type=float, default=TARGET_POLICY_NOISE)
    parser.add_argument("--target-noise-clip",   type=float, default=TARGET_NOISE_CLIP)
    parser.add_argument("--policy-delay",        type=int,   default=POLICY_DELAY)
    parser.add_argument("--learning-starts",     type=int,   default=LEARNING_STARTS)
    parser.add_argument("--explore-sigma-start", type=float, default=EXPLORE_SIGMA_START)
    parser.add_argument("--explore-sigma-end",   type=float, default=EXPLORE_SIGMA_END)
    parser.add_argument("--explore-decay-steps", type=int,   default=EXPLORE_DECAY_STEPS)
    parser.add_argument("--guard-collapse-frac", type=float, default=GUARD_COLLAPSE_FRAC)
    parser.add_argument("--guard-warmup-steps",  type=int,   default=GUARD_WARMUP_STEPS)
    parser.add_argument("--no-guard-abort", action="store_true",
                        help="warn on collapse but do not stop the run")
    args = parser.parse_args()

    train_td3_v219b(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        gamma=args.gamma,
        actor_lr_mult=args.actor_lr_mult,
        reward_overshoot_mode=args.reward_overshoot_mode,
        rain_normaliser=args.rain_normaliser,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        policy_delay=args.policy_delay,
        learning_starts=args.learning_starts,
        explore_sigma_start=args.explore_sigma_start,
        explore_sigma_end=args.explore_sigma_end,
        explore_decay_steps=args.explore_decay_steps,
        guard_collapse_frac=args.guard_collapse_frac,
        guard_warmup_steps=args.guard_warmup_steps,
        guard_abort=not args.no_guard_abort,
    )
