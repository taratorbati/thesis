# src/rl/train.py  v2.5.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.4.2  (all changes marked with  # [v2.5])
#
#  1. ent_coef = 0.05 (hardcoded, auto-tuning DISABLED)
#       Previously: ent_coef='auto', target_entropy=-13
#       The dual-gradient entropy auto-tuner caused the Phase-3 explosion:
#       as the actor settled into a low-variance ~2mm/day schedule, entropy
#       fell below the target.  The auto-tuner spiked ent_coef to 1.28 to
#       compensate, injecting sudden exploration noise that shattered the
#       Critic's learned Q-landscape.
#       Fix: hardcode ent_coef=0.05, a standard SAC value for complex
#       continuous-action environments.  Provides steady, unchanging
#       exploration noise; the Critic is never subjected to sudden
#       distribution shocks.  target_entropy argument is removed.
#       Ref: Gemini analysis Part 3, Point 3.
#
#  2. max_grad_norm = 1.0  (gradient clipping added)
#       Breaks the positive feedback loop: large critic loss → large
#       parameter updates → larger Q errors → even larger loss.
#       Ref: Both Claude session record and Gemini analysis.
#
#  3. Linear learning-rate decay: 3e-4 → 5e-5 over total_timesteps
#       In late training, a constant lr=3e-4 allows noise to push network
#       weights away from near-converged optima.  Linear decay stabilises
#       late-training updates without sacrificing early learning speed.
#       Ref: Claude session record Section 12.
#
#  4. total_timesteps raised to 500_000 (was 200_000)
#       The productive Phase-2 window now has more room.  With divergence
#       fixed, the full 500k should yield continued improvement.
#
#  All other training logic (RotatingReplayBufferCheckpoint, WandB
#  integration, year/budget randomisation, EvalCallback) is unchanged.
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

from src.rl.gym_env import IrrigationEnv
from src.rl.networks import CTDESACPolicy, make_sac_policy_kwargs

# ── training constants ────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 500_000
BUFFER_SIZE      = 500_000
BATCH_SIZE       = 256
GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5

ENT_COEF         = 0.05    # [v2.5] hardcoded — auto-tuning DISABLED
# TARGET_ENTROPY removed — not used when ent_coef is a fixed float

MAX_GRAD_NORM    = 1.0     # [v2.5] gradient clipping
LEARNING_STARTS  = 1_000
GRADIENT_STEPS   = 1

EVAL_FREQ        = 25_000
N_EVAL_EPISODES  = 9       # 3 dev years × 3 budget samples
CHECKPOINT_FREQ  = 50_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]


def _make_lr_schedule(lr_start: float, lr_end: float):
    """Linear decay from lr_start (progress=1.0) to lr_end (progress=0.0)."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1.0 at start, 0.0 at end
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
    """Initialise WandB; return True on success, False if unavailable."""
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
    """Save replay buffer to a single overwriting file at each checkpoint.

    Avoids the SB3 default behaviour of writing a new file per checkpoint,
    which at buffer_size=500k fills the 20 GB Kaggle disk limit within ~40
    checkpoints (~3.1 GB × 40 = 124 GB).
    """

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
    """Clip gradient norms after every SAC update step.

    SB3 does not expose a max_grad_norm parameter on SAC.  Passing it via
    optimizer_kwargs crashes because it is not an Adam argument.  This
    callback clips in-place after each gradient step, which is equivalent
    to the standard PyTorch pattern and has no effect on the loss landscape.
    """

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
) -> SAC:
    """Train a SAC agent with the v2.5 hyperparameters.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.  Run seeds 0–4 for the 5-seed campaign.
    output_dir : str
        Directory for model checkpoints and best-model artefacts.
    wandb_project : str | None
        WandB project name.  Pass None to disable WandB logging.
    total_timesteps : int
        Total environment steps.  Default 500 000.
    """
    run_name = f"sac_general_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── config dict for logging ───────────────────────────────────────────────
    config = {
        "version": "2.5.0",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "tau": TAU,
        "lr_start": LR_START,
        "lr_end": LR_END,
        "ent_coef": ENT_COEF,        # [v2.5] hardcoded
        "max_grad_norm": MAX_GRAD_NORM,
        "actor_hidden": ACTOR_HIDDEN,
        "critic_hidden": CRITIC_HIDDEN,
        "c_term": 0.0,               # [v2.5] terminal bonus removed
        "alpha5_rl": 0.0,            # [v2.5] actuator smoothing removed
        "changes_v25": [
            "c_term=0 (terminal bonus removed)",
            "ent_coef=0.05 hardcoded (auto-tuner disabled)",
            "max_grad_norm=1.0 (gradient clipping added)",
            "lr linear decay 3e-4->5e-5",
            "alpha5_rl=0 (actuator smoothing removed from RL reward)",
        ],
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # ── environments ──────────────────────────────────────────────────────────
    train_env = DummyVecEnv([lambda: IrrigationEnv(randomize=True)])
    eval_env  = DummyVecEnv([lambda: IrrigationEnv(randomize=True)])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)

    # ── policy kwargs ─────────────────────────────────────────────────────────
    policy_kwargs = make_sac_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
        # optimizer_kwargs intentionally omitted — max_grad_norm is NOT an Adam
        # argument and crashes if passed via optimizer_kwargs.  Gradient clipping
        # is applied by GradClipCallback after each update step instead.
    )

    # ── LR schedule ──────────────────────────────────────────────────────────
    lr_schedule = _make_lr_schedule(LR_START, LR_END)   # [v2.5]

    # ── model ─────────────────────────────────────────────────────────────────
    model = SAC(
        policy=CTDESACPolicy,
        env=train_env,
        learning_rate=lr_schedule,          # [v2.5] decay schedule
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        ent_coef=ENT_COEF,                  # [v2.5] hardcoded float, not 'auto'
        # target_entropy intentionally omitted — only used with ent_coef='auto'
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
        save_replay_buffer=False,   # handled by RotatingReplayBufferCheckpoint
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

    # ── WandB callback (optional) ─────────────────────────────────────────────
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
    print(f"\n{'='*60}")
    print(f"  SAC training — v2.5.0 — seed {seed}")
    print(f"  Key changes: c_term=0, ent_coef=0.05 (fixed),")
    print(f"               grad_clip=1.0, LR decay, alpha5_rl=0")
    print(f"  Output: {save_dir}")
    print(f"{'='*60}\n")

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
    parser = argparse.ArgumentParser(description="Train SAC v2.5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/rl")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    train_sac(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
    )
