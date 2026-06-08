# src/rl/train_v221_td3.py  v2.21.0  (gamma-correct biomass shaping; builds on v2.20 n-step)
# -----------------------------------------------------------------------------
# v2.21c = v2.20 Run A's exact-n-step trainer + an ADDITIVE terminal-yield term
# (reward_terminal_yield * x4_final/X4_REF, paid once at episode end ON TOP of the
# dense increment r1). biomass_shaping stays False here -- we KEEP the increment
# biomass return telescopes to a pure terminal-yield objective == MPC's. The
# n-step/warmup machinery below is identical to v2.20; only this and the
# run-name/version are changed. (v2.20 base text retained below.)
#
# TD3 trainer (v2.20 base text retained for the record).  Place in src/rl/ alongside
# train_v219b_td3.py.  This is train_v219b_td3 with exactly three additions,
# everything else (actor, critic, obs, reward, exploration schedule, the 11
# telemetry/guard callbacks, the eval protocol) reused VERBATIM from v2.19b so
# results are directly comparable:
#
#   1. EXACT n-step returns (NStepReplayBufferExact) with a gamma^n bootstrap,
#      wired via the model-gamma trick:  model.gamma = gamma_base ** n_steps,
#      buffer accumulates R_n with gamma_base.  SB3's stock TD3 target then
#      computes  R_n + (1-done) * gamma_base^n * Q  exactly -- no train()
#      override, and the critic still learns the gamma_base(=0.99) return so the
#      bias_ratio q_pred diagnostic stays on the same scale.  (See
#      nstep_buffer_exact.py for the full derivation.)
#   2. WarmupAsymmetricLRTD3 in place of AsymmetricLRTD3, enabling the optional
#      actor-LR "critic-leads" warm-up (Run B).  With mult=1.0/warmup=0 it is a
#      verified no-op (Run A).
#   3. The damping knobs (policy_delay, target_policy_noise) and learning_starts
#      are read from configs_v220.CONFIGS[config_name] instead of the module
#      constants, so one --config switch selects the whole pre-registered run.
#
# A manifest.json (git SHA, full config, the exact-gamma^n note, dev/training
# years) is written to the run dir BEFORE training, so a crashed run is still
# self-describing -- this is the Stage-0 fix for the "everything is v2.19b"
# naming ambiguity.
#
# RUNS NOTHING ON IMPORT.  Launch from the CLI (see __main__) or call
# train_td3_v220(...).
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from climate_data import DEV_YEARS, TRAINING_YEARS
from src.rl.gym_env import IrrigationEnv
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
    NonFiniteGuardCallback,
)
from src.rl.callbacks_eval import FixedScheduleEvalCallback
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)

# Reuse v2.19b's tuned constants and DETERMINISTIC eval schedules verbatim.
from src.rl import train_v219b_td3 as base

# Stage-1 additions.
from src.rl.nstep_buffer_exact import NStepReplayBufferExact
from src.rl.td3_warmup import WarmupAsymmetricLRTD3
from src.rl.configs_v221c import CONFIGS, GAMMA_BASE


def _git_sha() -> str:
    """Best-effort short git SHA of the working tree (for the manifest)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def train_td3_v221c(
    config_name: str = "A",
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: Optional[str] = None,
    total_timesteps: Optional[int] = None,
):
    """Train a Stage-1 v2.20 TD3 run selected by ``config_name`` (see configs_v220).

    Run A = exact n-step alone; Run B = n-step + the damping package.  All other
    machinery is identical to v2.19b.
    """
    if config_name not in CONFIGS:
        raise KeyError(f"unknown config {config_name!r}; choices: {sorted(CONFIGS)}")
    cfg = CONFIGS[config_name]

    if total_timesteps is None:
        total_timesteps = base.TOTAL_TIMESTEPS

    n_steps     = int(cfg["n_steps"])
    gamma_base  = float(cfg["gamma_base"])
    model_gamma = gamma_base ** n_steps          # <-- EXACT n-step bootstrap discount

    reward_du_alpha       = float(cfg["reward_du_alpha"])
    learning_starts       = int(cfg["learning_starts"])
    policy_delay          = int(cfg["policy_delay"])
    target_policy_noise   = float(cfg["target_policy_noise"])
    target_noise_clip     = float(cfg["target_noise_clip"])
    actor_lr_mult         = float(cfg["actor_lr_mult"])
    actor_warmup_updates  = int(cfg["actor_warmup_updates"])
    expose_prev_u         = bool(cfg.get("expose_prev_u", False))
    # v2.21: gamma-correct biomass shaping. The shaping gamma MUST equal the
    # per-step return discount (gamma_base); tying it here prevents drift.
    biomass_shaping       = bool(cfg.get("biomass_shaping", False))
    biomass_shaping_gamma = gamma_base if biomass_shaping else 1.0
    reward_terminal_yield = float(cfg.get("reward_terminal_yield", 0.0))

    # prev_u needs a matching 9-feature actor+critic (Stage 2); fail loudly
    # rather than silently feed a 1227-dim obs to the 8-feature network.
    EnvCls = IrrigationEnv
    if expose_prev_u:
        raise NotImplementedError(
            "expose_prev_u=True requires a matching 9-feature actor+critic. "
            "networks_td3.py hard-codes TD3_N_AGENT_FEATURES=8 and asserts "
            "features_dim==1097; the prev_u env emits 1227-dim obs. Provide a "
            "9-feature network variant first (see gym_env_prev_u.py header). "
            "Keep expose_prev_u=False for Stage 1."
        )

    run_name = f"td3_v221c_{cfg['label']}_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    reward_overshoot_mode = base.REWARD_OVERSHOOT_MODE
    rain_normaliser       = base.RAIN_NORMALISER

    config = {
        "version": "2.21c-TD3",
        "stage": 2,
        "config_name": config_name,
        "label": cfg["label"],
        "git_sha": _git_sha(),
        "seed": seed,
        "algorithm": "WarmupAsymmetricLRTD3 (SB3 TD3) + exact n-step VDN",
        "policy_class": "TD3VDNPolicy (deterministic _TD3SharedActor, marker=2.19)",
        "total_timesteps": total_timesteps,
        # --- the n-step wiring (the headline change) ---
        "n_steps": n_steps,
        "gamma_base": gamma_base,
        "model_gamma": model_gamma,
        "gamma_note": (
            "model.gamma = gamma_base ** n_steps so SB3's stock target gives "
            "R_n + (1-done)*gamma_base^n*Q exactly; buffer accumulates R_n with "
            "gamma_base. Critic learns the gamma_base(=0.99) return."
        ),
        "replay_buffer": "NStepReplayBufferExact",
        # --- damping package (Run B; stock in Run A) ---
        "policy_delay": policy_delay,
        "target_policy_noise": target_policy_noise,
        "target_noise_clip": target_noise_clip,
        "actor_lr_mult": actor_lr_mult,
        "actor_warmup_updates": actor_warmup_updates,
        # --- carried from the diverging v2.20 r5 run for attribution ---
        "learning_starts": learning_starts,
        "reward_du_alpha": reward_du_alpha,
        "biomass_shaping": biomass_shaping,
        "biomass_shaping_gamma": biomass_shaping_gamma,
        "reward_terminal_yield": reward_terminal_yield,
        "expose_prev_u": expose_prev_u,
        # --- inherited v2.19b machinery (unchanged) ---
        "tau": base.TAU,
        "buffer_size": base.BUFFER_SIZE,
        "batch_size": base.BATCH_SIZE,
        "lr_start": base.LR_START,
        "lr_end": base.LR_END,
        "max_grad_norm": base.MAX_GRAD_NORM,
        "gradient_steps": base.GRADIENT_STEPS,
        "train_freq": base.TRAIN_FREQ,
        "explore_sigma_start": base.EXPLORE_SIGMA_START,
        "explore_sigma_end": base.EXPLORE_SIGMA_END,
        "explore_decay_steps": base.EXPLORE_DECAY_STEPS,
        "guard_collapse_frac": base.GUARD_COLLAPSE_FRAC,
        "guard_warmup_steps": base.GUARD_WARMUP_STEPS,
        "rain_normaliser": rain_normaliser,
        "reward_overshoot_mode": reward_overshoot_mode,
        "eval_protocol": (
            "v2.19c DETERMINISTIC held-out: DEV_YEARS x {0.70,0.85,1.00} = 9 "
            "episodes; bias-eval = DEV_YEARS @ 1.00 = 3."
        ),
        "dev_years": list(DEV_YEARS),
        "training_years": list(TRAINING_YEARS),
        "eval_budget_fracs": list(base.EVAL_BUDGET_FRACS),
        "hypothesis": (
            "Run A: bounding the bootstrap horizon with exact n-step (n=5) stops "
            "the q_pred divergence that tracked learning_starts. Run B: adding "
            "TD3's structural damping (policy_delay 3, target noise 0.3, "
            "critic-leads actor warm-up) removes any residual limit cycle. "
            "Success = q_pred bounded, eval improves, final ~= best, guard never "
            "trips."
        ),
    }

    # Write the manifest BEFORE training so a crashed run is self-describing.
    (save_dir / "manifest.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    def _make_env():
        return EnvCls(
            randomize=True,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
            reward_du_alpha=reward_du_alpha,
            biomass_shaping_gamma=biomass_shaping_gamma,
            reward_terminal_yield=reward_terminal_yield,
        )

    def _make_eval_env():
        return EnvCls(
            randomize=False,
            eval_schedule=base.EVAL_SCHEDULE,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
            reward_du_alpha=reward_du_alpha,
            biomass_shaping_gamma=biomass_shaping_gamma,
            reward_terminal_yield=reward_terminal_yield,
        )

    def _make_bias_eval_env():
        return EnvCls(
            randomize=False,
            eval_schedule=base.BIAS_EVAL_SCHEDULE,
            curriculum_warmup_steps=0,
            use_overshoot_feature=False,
            normalize_globals=True,
            reward_overshoot_mode=reward_overshoot_mode,
            rain_normaliser=rain_normaliser,
            reward_du_alpha=reward_du_alpha,
            biomass_shaping_gamma=biomass_shaping_gamma,
            reward_terminal_yield=reward_terminal_yield,
        )

    train_env     = DummyVecEnv([_make_env])
    eval_env      = DummyVecEnv([_make_eval_env])
    bias_eval_env = DummyVecEnv([_make_bias_eval_env])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)
    bias_eval_env.seed(seed + 2000)

    policy_kwargs = make_td3_policy_kwargs(
        N=base.N_AGENTS, actor_hidden=base.ACTOR_HIDDEN, critic_hidden=base.CRITIC_HIDDEN,
    )
    lr_schedule = _make_lr_schedule(base.LR_START, base.LR_END)
    action_noise = NormalActionNoise(
        mean=np.zeros(base.N_AGENTS, dtype=np.float64),
        sigma=base.EXPLORE_SIGMA_START * np.ones(base.N_AGENTS, dtype=np.float64),
    )

    # Configure the warm-up subclass via class attributes (mirrors v2.19b).
    WarmupAsymmetricLRTD3.actor_lr_mult        = actor_lr_mult
    WarmupAsymmetricLRTD3.actor_warmup_updates = actor_warmup_updates

    model = WarmupAsymmetricLRTD3(
        policy=TD3VDNPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=base.BUFFER_SIZE,
        batch_size=base.BATCH_SIZE,
        gamma=model_gamma,                       # gamma_base ** n_steps  (EXACT n-step)
        tau=base.TAU,
        action_noise=action_noise,
        policy_delay=policy_delay,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        learning_starts=learning_starts,
        gradient_steps=base.GRADIENT_STEPS,
        train_freq=base.TRAIN_FREQ,
        replay_buffer_class=NStepReplayBufferExact,
        replay_buffer_kwargs=dict(n_steps=n_steps, gamma=gamma_base),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # --- callbacks: identical set/order to v2.19b -------------------------
    eval_callback = FixedScheduleEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=base.EVAL_FREQ,
        n_eval_episodes=base.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=base.CHECKPOINT_FREQ,
        save_path=str(save_dir / "checkpoints"),
        name_prefix=run_name,
        save_replay_buffer=False,
        verbose=1,
    )
    rotating_buffer_callback = RotatingReplayBufferCheckpoint(
        save_freq=base.CHECKPOINT_FREQ, save_path=save_dir, verbose=1,
    )
    grad_clip_callback = GradClipCallback(max_grad_norm=base.MAX_GRAD_NORM)
    bias_ratio_cb = BiasRatioCallback(
        eval_env=bias_eval_env,
        eval_freq=base.BIAS_RATIO_FREQ,
        n_eval_episodes=base.BIAS_RATIO_N_EPISODES,
        save_path=str(save_dir),
        verbose=1,
    )
    action_stats_cb = ActionStatsCallback(log_freq=base.ACTION_STATS_FREQ)
    optimizer_lr_cb = OptimizerLRCallback(log_freq=base.LR_LOG_FREQ)
    noise_decay_cb = ExplorationNoiseDecayCallback(
        sigma_start=base.EXPLORE_SIGMA_START,
        sigma_end=base.EXPLORE_SIGMA_END,
        decay_steps=base.EXPLORE_DECAY_STEPS,
        log_freq=base.EXPLORE_LOG_FREQ,
        csv_path=str(save_dir / "exploration_sigma_log.csv"),
        verbose=1,
    )
    coverage_cb = LowActionCoverageCallback(
        log_freq=base.COVERAGE_LOG_FREQ,
        csv_path=str(save_dir / "low_action_coverage_log.csv"),
        verbose=0,
    )
    collapse_guard_cb = CollapseGuardCallback(
        collapse_frac=base.GUARD_COLLAPSE_FRAC,
        warmup_steps=base.GUARD_WARMUP_STEPS,
        check_freq=base.GUARD_CHECK_FREQ,
        window=base.GUARD_WINDOW,
        abort_on_collapse=base.GUARD_ABORT,
        csv_path=str(save_dir / "collapse_guard_log.csv"),
        verbose=1,
    )
    nonfinite_guard_cb = NonFiniteGuardCallback(
        stop_on_nonfinite=True,
        csv_path=str(save_dir / "nonfinite_guard_log.csv"),
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
        nonfinite_guard_cb,
    ]
    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            cb_list.append(WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=base.CHECKPOINT_FREQ, verbose=0,
            ))
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    callbacks = CallbackList(cb_list)

    print(f"\n{'='*72}")
    print(f"  TD3 training - v2.21c (additive terminal-yield) - config {config_name} ({cfg['label']}) - seed {seed}")
    print(f"  n-step: n={n_steps}  gamma_base={gamma_base}  model_gamma=gamma_base^n={model_gamma:.6f}")
    print(f"  buffer: NStepReplayBufferExact (exact gamma^n bootstrap)")
    print(f"  policy_delay={policy_delay}  target_policy_noise={target_policy_noise}  clip={target_noise_clip}")
    print(f"  actor_lr_mult={actor_lr_mult}  actor_warmup_updates={actor_warmup_updates:,}")
    print(f"  learning_starts={learning_starts:,}  reward_du_alpha(r5)={reward_du_alpha}  expose_prev_u={expose_prev_u}")
    print(f"  explore noise: {base.EXPLORE_SIGMA_START:.2f} -> {base.EXPLORE_SIGMA_END:.2f} over {base.EXPLORE_DECAY_STEPS:,} (floor held)")
    print(f"  collapse guard: abort={base.GUARD_ABORT} if rolling low-action >= "
          f"{base.GUARD_COLLAPSE_FRAC:.0%} after {base.GUARD_WARMUP_STEPS:,} steps")
    print(f"  dev/eval years: {list(DEV_YEARS)}  ->  training years ({len(TRAINING_YEARS)}): {list(TRAINING_YEARS)}")
    print(f"  git={config['git_sha']}  total steps: {total_timesteps:,}  | Output: {save_dir}")
    print(f"{'='*72}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except BaseException:
        # Mirror the traceback to the REAL stdout (bypassing the rich/tqdm
        # progress-bar proxy), exactly as v2.19b does: (1) if the exception is
        # the rich RecursionError, a normal print() re-enters the broken flush;
        # (2) SB3/Colab otherwise send tracebacks only to stderr.  Best-effort;
        # never masks the original exception.
        import sys, traceback
        _err = sys.__stdout__ or sys.__stderr__
        try:
            if _err is not None:
                _err.write("\n" + "=" * 72 + "\n")
                _err.write("[train] model.learn() raised -- full traceback below "
                           "(mirrored to the real stdout, bypassing the "
                           "progress-bar proxy):\n")
                traceback.print_exc(file=_err)
                _err.write("=" * 72 + "\n")
                _err.flush()
        except Exception:
            pass
        raise
    finally:
        if wandb_active:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    final_path = save_dir / f"{run_name}_final"
    model.save(str(final_path))

    # Stamp completion into the manifest (so a finished run is marked as such).
    try:
        config["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (save_dir / "manifest.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8",
        )
    except Exception:
        pass

    print(f"\n[train] Final model saved to {final_path}.zip")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Train TD3 v2.21c: v2.20 Run A increment reward + an ADDITIVE terminal-yield "
            "bootstrap) and an optional critic-leads actor LR warm-up. --config A "
            "= n-step alone; --config B = n-step + damping package."
        )
    )
    parser.add_argument("--config",          type=str, default="A", choices=sorted(CONFIGS))
    parser.add_argument("--seed",            type=int, default=0)
    parser.add_argument("--output-dir",      type=str, default="results/rl")
    parser.add_argument("--wandb-project",   type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()

    train_td3_v221c(
        config_name=args.config,
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
    )
