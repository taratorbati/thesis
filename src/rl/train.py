# src/rl/train.py  v2.7.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.5.0  (see change_spec_v27.md for full rationale)
#
#  1. Version string bumped to "2.7.0" for WandB provenance tracking.
#
#  2. Config dict updated to reflect v2.7 changes:
#       - OBS_DIM: 707 → 1097  (8 per-agent features instead of 5)
#       - N_AGENT_FEATURES: 5 → 8  (elev_norm + Nr_norm + Nr_internal_norm
#                                   + n_upstream_norm added)
#       - episode_lifecycle: "always 93 days — no early termination on
#         budget exhaustion"
#       - reward_terms: "r1+r2+r3+r6 only (rb and r5 removed)"
#
#  3. Print banner updated to reflect v2.7 changes.
#
#  All training logic, hyperparameters, callbacks, and WandB integration
#  are UNCHANGED from v2.5.0.  This is intentional: the only things that
#  changed are the environment (gym_env.py) and the network dimensions
#  (networks.py).  The SAC algorithm itself does not need to know about
#  those changes — it discovers the obs dimension from the env at
#  construction time.
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
    """Train a SAC agent with the v2.7 environment and hyperparameters.

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

    Notes
    -----
    v2.7 environment changes (gym_env.py):
        - OBS_DIM 707 → 1097: per-agent block now has 8 features (was 5).
          The three new features are static topographic scalars: Nr_norm,
          Nr_internal_norm, n_upstream_norm.  The gamma slot (previously the
          v2.6 obs-layout bug: x2/theta18) is restored to elev_norm.
        - Episodes always run 93 days: terminated=False always; budget
          exhaustion no longer ends the episode early.
        - Reward simplified: r = r1 + r2 + r3 + r6 only (rb and r5 removed).

    SAC hyperparameters (unchanged from v2.5):
        ent_coef=0.05 (fixed), max_grad_norm=1.0, LR 3e-4→5e-5, 500k steps.
    """
    run_name = f"sac_general_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── config dict for WandB logging ─────────────────────────────────────────
    config = {
        "version": "2.7.0",                     # [v2.7] bumped from 2.5.0
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
        # [v2.7] environment changes
        "obs_dim": 1097,
        "n_agent_features": 8,
        "episode_lifecycle": "always 93 days — no early termination on budget exhaustion",
        "reward_terms": "r1+r2+r3+r6 only (rb and r5 removed)",
        # [v2.5] values still in effect
        "c_term": 0.0,
        "alpha5_rl": 0.0,
        "changes_v27": [
            "obs_dim 707→1097: per-agent block 5→8 features",
            "gamma obs slot restored to elev_norm (was x2/theta18 — bug fix)",
            "3 new static topo features: Nr_norm, Nr_internal_norm, n_upstream_norm",
            "episodes always run 93 days (no early termination on budget exhaustion)",
            "reward simplified: rb (burn-rate) and r5 (delta-u) removed",
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
    print(f"  SAC training — v2.7.0 — seed {seed}")
    print(f"  Env changes:  obs_dim=1097 (was 707), 8 features/agent,")
    print(f"                gamma slot restored (elev_norm, was x2/theta18),")
    print(f"                3 new topo features, episodes always 93 days")
    print(f"  Reward:       r1+r2+r3+r6 only (rb and r5 removed)")
    print(f"  SAC:          ent_coef=0.05 (fixed), grad_clip=1.0, LR decay")
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
    parser = argparse.ArgumentParser(description="Train SAC v2.7")
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
