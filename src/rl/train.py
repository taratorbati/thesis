# src/rl/train.py  v2.8.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.7.0  (see change_spec_v28.md for full rationale)
#
#  1. Version string bumped to "2.8.0".
#
#  2. Curriculum kwargs exposed via train_sac() signature:
#       curriculum_warmup_steps (default 50_000)
#       curriculum_short_len    (default 60)
#     These are passed to IrrigationEnv at construction so each parallel
#     env knows when to switch from short to full episodes.
#
#  3. Default TOTAL_TIMESTEPS reduced from 500_000 to 250_000 based on the
#     v2.7 seed-0 and seed-1 evaluations: both peaked at step 200k and
#     degraded thereafter; the EvalCallback captures the peak regardless,
#     so the 250k → 500k window was wasted compute.  The 250k cap saves
#     ~50% time per seed and matches what was actually used for seed-1.
#
#  4. Config dict updated to reflect v2.8 obs (1227-dim), feature count
#     (9), reward (unchanged 4-term), and curriculum settings.
#
#  All other training logic, hyperparameters, callbacks, and WandB
#  integration are UNCHANGED from v2.7.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.gym_env import (
    IrrigationEnv,
    CURRICULUM_WARMUP_STEPS_DEFAULT,
    CURRICULUM_SHORT_LEN_DEFAULT,
)
from src.rl.networks import CTDESACPolicy, make_sac_policy_kwargs

# ── training constants ────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 250_000     # v2.8: reduced from 500k (see header)
BUFFER_SIZE      = 250_000     # match — no point oversizing buffer
BATCH_SIZE       = 256
GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5

ENT_COEF         = 0.05    # fixed (auto-tuning disabled since v2.5)

MAX_GRAD_NORM    = 1.0     # gradient clipping (v2.5)
LEARNING_STARTS  = 1_000
GRADIENT_STEPS   = 1

EVAL_FREQ        = 25_000
N_EVAL_EPISODES  = 9
CHECKPOINT_FREQ  = 50_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]


def _make_lr_schedule(lr_start: float, lr_end: float):
    """Linear decay from lr_start (progress=1.0) to lr_end (progress=0.0)."""
    def schedule(progress_remaining: float) -> float:
        return lr_end + (lr_start - lr_end) * progress_remaining
    return schedule


def _resolve_wandb_api_key() -> str | None:
    """Try env var → Kaggle secrets → Colab userdata."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        if key:
            return key
    except Exception:
        pass
    try:
        from google.colab import userdata
        key = userdata.get("WANDB_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return None


def _init_wandb(project: str, run_name: str, config: dict) -> bool:
    try:
        import wandb
        api_key = _resolve_wandb_api_key()
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        wandb.init(
            project=project,
            entity="taratorbati-itmo-university",
            name=run_name,
            config=config,
            reinit=True,
        )
        print(f"[WandB] run initialised: {wandb.run.url}")
        return True
    except Exception as e:
        print(f"[WandB] unavailable ({e}); continuing with local-only logging.")
        return False


class RotatingReplayBufferCheckpoint(BaseCallback):
    """Save replay buffer to a single overwriting file at each checkpoint."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            buf_path = str(self.save_path / "replay_buffer_latest")
            self.model.save_replay_buffer(buf_path)
            if self.verbose > 0:
                print(f"[RotatingBuffer] saved to {buf_path}.pkl  "
                      f"(step {self.num_timesteps})")
        return True


class GradClipCallback(BaseCallback):
    """Clip gradient norms after every SAC update step."""

    def __init__(self, max_grad_norm: float = 1.0):
        super().__init__(verbose=0)
        self.max_grad_norm = max_grad_norm

    def _on_step(self) -> bool:
        if self.model is not None and hasattr(self.model, "policy"):
            torch.nn.utils.clip_grad_norm_(
                self.model.policy.parameters(), self.max_grad_norm
            )
        return True


def train_sac(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: str | None = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    curriculum_warmup_steps: int = CURRICULUM_WARMUP_STEPS_DEFAULT,
    curriculum_short_len:    int = CURRICULUM_SHORT_LEN_DEFAULT,
) -> SAC:
    """Train a SAC agent with the v2.8 environment and hyperparameters.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.  Run seeds 2-6 for the v2.8 campaign.
    output_dir : str
        Directory for model checkpoints and best-model artefacts.
    wandb_project : str | None
        WandB project name.  Pass None to disable WandB logging.
    total_timesteps : int
        Total environment steps.  Default 250 000 (v2.8 — was 500 000 in v2.7).
    curriculum_warmup_steps : int
        Number of env transitions during which episodes are truncated at the
        short length.  Default 50 000 (= 20% of 250k training budget).  Set to
        0 to disable the curriculum entirely (full episodes throughout — v2.7
        baseline behaviour).
    curriculum_short_len : int
        Episode length in days during the warmup window.  Default 60.

    Notes
    -----
    v2.8 environment changes (gym_env.py):
        - OBS_DIM 1097 → 1227: per-agent block now has 9 features (was 8).
          The new feature is x1_overshoot_norm = max(x1-FC, 0)/FC, addressing
          the v2.7 wet-year x1-conditioning weakness.
        - Episode-length curriculum: episodes truncate at 60 days for first
          50 000 env steps, 93 days thereafter.  Reduces critic-target
          variance during the initial value-function learning phase.

    SAC hyperparameters (unchanged from v2.7):
        ent_coef=0.05 (fixed), max_grad_norm=1.0, LR 3e-4→5e-5,
        gradient clipping via GradClipCallback.
    """
    run_name = f"sac_v28_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── config dict for WandB logging ─────────────────────────────────────────
    config = {
        "version": "2.8.0",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "ent_coef": ENT_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "actor_hidden": ACTOR_HIDDEN,
        "critic_hidden": CRITIC_HIDDEN,
        # v2.8 environment
        "obs_dim": 1227,
        "n_agent_features": 9,
        "curriculum_warmup_steps": curriculum_warmup_steps,
        "curriculum_short_len":    curriculum_short_len,
        "episode_lifecycle": "always run full season after warmup",
        "reward_terms": "r1+r2+r3+r6 (unchanged from v2.7)",
        # legacy values still in effect
        "c_term": 0.0,
        "alpha5_rl": 0.0,
        "changes_v28": [
            "obs_dim 1097→1227: per-agent block 8→9 features",
            "added x1_overshoot_norm feature: max(x1-FC,0)/FC, in [0,1]",
            "episode-length curriculum: 60d for first {0}k steps, 93d after".format(
                curriculum_warmup_steps // 1000),
            "total_timesteps default reduced 500k→250k (v2.7 peaked at 200k)",
        ],
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # ── environments ──────────────────────────────────────────────────────────
    def _make_env():
        return IrrigationEnv(
            randomize=True,
            curriculum_warmup_steps=curriculum_warmup_steps,
            curriculum_short_len=curriculum_short_len,
        )
    train_env = DummyVecEnv([_make_env])

    # Eval env: NO curriculum (full episodes always, so eval is on the same
    # distribution we ultimately care about).
    eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,   # disable curriculum at eval time
    )])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)

    # ── policy kwargs ─────────────────────────────────────────────────────────
    policy_kwargs = make_sac_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    # ── LR schedule ──────────────────────────────────────────────────────────
    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # ── model ─────────────────────────────────────────────────────────────────
    model = SAC(
        policy=CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ENT_COEF,
        learning_starts=LEARNING_STARTS,
        gradient_steps=GRADIENT_STEPS,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_dir / "tensorboard"),
    )

    # ── callbacks ─────────────────────────────────────────────────────────────
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
    callbacks = CallbackList([eval_callback, checkpoint_callback,
                              rotating_buffer_callback, grad_clip_callback])

    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            wandb_cb = WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ,
                verbose=0,
            )
            callbacks = CallbackList([eval_callback, checkpoint_callback,
                                      rotating_buffer_callback,
                                      grad_clip_callback, wandb_cb])
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SAC training — v2.8.0 — seed {seed}")
    print(f"  Env: obs_dim=1227 (was 1097), 9 features/agent (was 8)")
    print(f"       new feature: x1_overshoot_norm = max(x1-FC,0)/FC")
    print(f"  Curriculum: short episodes (length {curriculum_short_len}) for")
    print(f"              first {curriculum_warmup_steps:,} env steps, then full 93-day episodes")
    print(f"  Reward: r1+r2+r3+r6 (unchanged from v2.7)")
    print(f"  SAC: ent_coef=0.05 fixed, grad_clip=1.0, LR decay 3e-4→5e-5")
    print(f"  Total steps: {total_timesteps:,}")
    print(f"  Output: {save_dir}")
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

    model.save(str(save_dir / f"{run_name}_final"))
    print(f"\n[train] Final model saved to {save_dir}/{run_name}_final.zip")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SAC v2.8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/rl")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--curriculum-warmup-steps", type=int,
                        default=CURRICULUM_WARMUP_STEPS_DEFAULT)
    parser.add_argument("--curriculum-short-len", type=int,
                        default=CURRICULUM_SHORT_LEN_DEFAULT)
    args = parser.parse_args()

    train_sac(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
        curriculum_warmup_steps=args.curriculum_warmup_steps,
        curriculum_short_len=args.curriculum_short_len,
    )
