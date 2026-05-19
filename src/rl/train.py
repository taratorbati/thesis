# src/rl/train.py  v2.9.0
# ─────────────────────────────────────────────────────────────────────────────
# v2.9 = v2.7 baseline + Proposal B (AdaptiveLRCallback).
#
# Exactly ONE change from v2.7:
#   AdaptiveLRCallback added to the callback stack.
#   All other hyperparameters, architecture, env, reward, and obs layout
#   are IDENTICAL to v2.7 (1097-dim obs, 8 features/agent, V27CTDESACPolicy).
#
# What AdaptiveLRCallback does (Proposal B):
#   Monitors a rolling window of critic_loss values.
#   If rolling mean > CRITIC_LOSS_SPIKE_THRESHOLD (default 50):
#     → multiply current LR by LR_REDUCTION_FACTOR (default 0.3)
#     → clamp to LR_FLOOR (default 1e-5)
#     → log the intervention to console and TensorBoard
#   If rolling mean < CRITIC_LOSS_RECOVERY_THRESHOLD (default 5)
#     AND LR was previously reduced:
#     → restore LR to the scheduled value at the current progress step
#     → log the recovery
#   Intervention is rate-limited: minimum INTERVENTION_COOLDOWN_STEPS (default
#   5000) between successive reductions to prevent oscillation.
#
# Motivation (see THESIS_HANDOFF_v3.md §13 for full analysis):
#   v2.7 and v2.8 both exhibit a critic-loss explosion beginning at step
#   ~165k (v2.7) / ~195k (v2.8).  The explosion follows the standard
#   deadly-triad mechanism: overestimated Q values → actor exploits inflated Q
#   → critic receives larger-variance targets → divergence.  The fixed
#   ent_coef=0.05 means the entropy brake does not scale with growing |Q|,
#   so the feedback loop ignites once |actor_loss| crosses ~500.
#
#   Proposal B breaks the loop at the first sign of the spike (critic_loss >
#   50) by reducing LR by 0.7×.  This slows the critic weight update
#   response to the bad bootstrap target, allowing the replay buffer to
#   dilute the pathological transitions before they amplify further.
#   The recovery logic restores LR once critic_loss settles, so the actor
#   is not permanently handicapped.
#
# Why v2.7 architecture (not v2.8):
#   v2.8 (curriculum + x1_overshoot) failed because the 50k-step curriculum
#   biased the policy toward a 60-day spending strategy that the post-warmup
#   phase never unlearned.  The best_model was captured inside the warmup
#   window (step 25k) and performs poorly on the full 93-day problem.
#   v2.9 makes only the stability change to isolate its effect and uses
#   V27CTDESACPolicy (1097-dim obs) to keep the comparison clean.
#
# Obs layout: v2.7 (1097-dim, 8 features/agent) — matches V27CTDESACPolicy.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import shutil
from collections import deque
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
from src.rl.networks import V27CTDESACPolicy, make_sac_policy_kwargs as make_v27_sac_policy_kwargs

# ── training constants (identical to v2.7 except version string) ──────────────
TOTAL_TIMESTEPS  = 250_000
BUFFER_SIZE      = 250_000
BATCH_SIZE       = 256
GAMMA            = 0.99
TAU              = 0.005
LR_START         = 3e-4
LR_END           = 5e-5

ENT_COEF         = 0.05    # fixed (auto-tuning disabled since v2.5)
MAX_GRAD_NORM    = 1.0
LEARNING_STARTS  = 1_000
GRADIENT_STEPS   = 1

EVAL_FREQ        = 25_000
N_EVAL_EPISODES  = 9
CHECKPOINT_FREQ  = 50_000

ACTOR_HIDDEN  = [128, 128]
CRITIC_HIDDEN = [256, 256]

# ── Proposal B thresholds ─────────────────────────────────────────────────────
# These are deliberately conservative so the callback only fires when the
# explosion is clearly starting, not on routine spikes.
CRITIC_LOSS_SPIKE_THRESHOLD    = 50.0    # rolling-mean above this → reduce LR
CRITIC_LOSS_RECOVERY_THRESHOLD = 5.0     # rolling-mean below this → restore LR
LR_REDUCTION_FACTOR            = 0.3     # multiply LR by this on spike
LR_FLOOR                       = 1e-5    # never go below this
INTERVENTION_COOLDOWN_STEPS    = 5_000   # min steps between reductions
ROLLING_WINDOW                 = 1_000   # steps in rolling-mean window


def _make_lr_schedule(lr_start: float, lr_end: float):
    """Linear decay from lr_start (progress=1.0) to lr_end (progress=0.0)."""
    def schedule(progress_remaining: float) -> float:
        return lr_end + (lr_start - lr_end) * progress_remaining
    return schedule


def _resolve_wandb_api_key() -> str | None:
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


class AdaptiveLRCallback(BaseCallback):
    """Proposal B: Reduce LR when critic_loss spikes; restore when it recovers.

    Monitors the rolling mean of critic_loss over the last `window` optimizer
    steps.  When the rolling mean exceeds `spike_threshold`, the learning rate
    is multiplied by `reduction_factor` (floored at `lr_floor`).  When it falls
    back below `recovery_threshold`, the LR is restored to the linear-decay
    schedule value at the current training progress.

    Parameters
    ----------
    lr_schedule : callable
        The base LR schedule (progress_remaining → lr).  Used for restoration.
    spike_threshold : float
        Rolling-mean critic_loss above which LR is reduced.  Default 50.
    recovery_threshold : float
        Rolling-mean critic_loss below which LR is restored.  Default 5.
    reduction_factor : float
        Multiply current LR by this on spike.  Default 0.3.
    lr_floor : float
        Hard lower bound on LR.  Default 1e-5.
    window : int
        Rolling-mean window size in optimizer steps.  Default 1000.
    cooldown_steps : int
        Minimum env steps between successive LR reductions.  Default 5000.
    verbose : int
        0 = silent except on interventions.  1 = log every window.
    """

    def __init__(
        self,
        lr_schedule,
        spike_threshold: float    = CRITIC_LOSS_SPIKE_THRESHOLD,
        recovery_threshold: float = CRITIC_LOSS_RECOVERY_THRESHOLD,
        reduction_factor: float   = LR_REDUCTION_FACTOR,
        lr_floor: float           = LR_FLOOR,
        window: int               = ROLLING_WINDOW,
        cooldown_steps: int       = INTERVENTION_COOLDOWN_STEPS,
        verbose: int              = 0,
    ):
        super().__init__(verbose)
        self.lr_schedule         = lr_schedule
        self.spike_threshold     = spike_threshold
        self.recovery_threshold  = recovery_threshold
        self.reduction_factor    = reduction_factor
        self.lr_floor            = lr_floor
        self.window              = window
        self.cooldown_steps      = cooldown_steps

        self._critic_loss_buf    = deque(maxlen=window)
        self._lr_reduced         = False          # currently in a reduced state?
        self._n_reductions       = 0              # total reduction events
        self._last_reduction_step = -cooldown_steps  # step of last reduction

    def _current_scheduled_lr(self) -> float:
        """LR value the base schedule would give at current training progress."""
        progress = 1.0 - self.num_timesteps / max(self.model._total_timesteps, 1)
        progress = max(0.0, min(1.0, progress))
        return float(self.lr_schedule(progress))

    def _set_lr(self, lr: float) -> None:
        """Write `lr` to all optimizer parameter groups."""
        for opt in [self.model.policy.actor.optimizer,
                    self.model.policy.critic.optimizer]:
            for pg in opt.param_groups:
                pg['lr'] = lr

    def _get_lr(self) -> float:
        return float(
            self.model.policy.actor.optimizer.param_groups[0]['lr']
        )

    def _on_step(self) -> bool:
        # critic_loss is logged by SB3 every gradient step into
        # self.model.logger.  Read it from the locals dict that SB3 provides
        # to callbacks via self.locals.  Only available after learning_starts.
        critic_loss = self.locals.get('critic_loss', None)
        if critic_loss is None:
            # Try the SB3 internal logger record
            try:
                critic_loss = self.model.logger.name_to_value.get(
                    'train/critic_loss', None
                )
            except Exception:
                pass
        if critic_loss is None:
            return True

        self._critic_loss_buf.append(float(critic_loss))
        if len(self._critic_loss_buf) < self.window:
            return True

        rolling_mean = float(np.mean(self._critic_loss_buf))

        # ── spike detection → reduce LR ──────────────────────────────────────
        steps_since_last = self.num_timesteps - self._last_reduction_step
        if (rolling_mean > self.spike_threshold
                and steps_since_last >= self.cooldown_steps):
            current_lr = self._get_lr()
            new_lr = max(current_lr * self.reduction_factor, self.lr_floor)
            self._set_lr(new_lr)
            self._lr_reduced = True
            self._n_reductions += 1
            self._last_reduction_step = self.num_timesteps

            # Log to TensorBoard / WandB via SB3 logger
            self.model.logger.record(
                "adaptive_lr/event",           float(self._n_reductions))
            self.model.logger.record(
                "adaptive_lr/lr_after_reduce", new_lr)
            self.model.logger.record(
                "adaptive_lr/rolling_critic",  rolling_mean)

            print(
                f"[AdaptiveLR] step {self.num_timesteps:>7}: "
                f"critic_loss rolling={rolling_mean:.1f} > {self.spike_threshold} → "
                f"LR {current_lr:.2e} → {new_lr:.2e}  "
                f"(reduction #{self._n_reductions})"
            )

        # ── recovery detection → restore scheduled LR ─────────────────────────
        elif self._lr_reduced and rolling_mean < self.recovery_threshold:
            scheduled_lr = self._current_scheduled_lr()
            self._set_lr(scheduled_lr)
            self._lr_reduced = False

            self.model.logger.record(
                "adaptive_lr/lr_after_restore", scheduled_lr)
            self.model.logger.record(
                "adaptive_lr/rolling_critic",   rolling_mean)

            print(
                f"[AdaptiveLR] step {self.num_timesteps:>7}: "
                f"critic_loss rolling={rolling_mean:.2f} < {self.recovery_threshold} → "
                f"LR restored to scheduled {scheduled_lr:.2e}"
            )

        if self.verbose > 0:
            self.model.logger.record(
                "adaptive_lr/rolling_critic_verbose", rolling_mean)

        return True


def train_sac(
    seed: int = 0,
    output_dir: str = "results/rl",
    wandb_project: str | None = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> SAC:
    """Train a SAC v2.9 agent (v2.7 + Proposal B adaptive LR).

    Parameters
    ----------
    seed : int
        Random seed.  Use same seeds as v2.7 baseline for paired comparison.
    output_dir : str
        Root directory for checkpoints and best-model artefacts.
    wandb_project : str | None
        WandB project name.  None disables WandB logging.
    total_timesteps : int
        Default 250 000 (same as v2.7/v2.8).

    Notes
    -----
    Architecture: V27CTDESACPolicy (1097-dim obs, 8 features/agent).
    This is intentionally identical to v2.7 — only the callback stack differs.

    The single change vs v2.7:
        AdaptiveLRCallback fires when rolling-1000-step mean critic_loss > 50,
        reducing LR by 0.7× (floor 1e-5).  Restores when mean drops below 5.
        Cooldown of 5000 steps between reductions prevents oscillation.

    Falsifiable predictions:
        - If Proposal B works: critic_loss stays below 200 throughout training;
          best_model step shifts from step 200k toward step 225-250k.
        - If Proposal B doesn't work: same explosion pattern as v2.7 but
          slightly later; best_model step unchanged.
    """
    run_name = f"sac_v29_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── config dict ───────────────────────────────────────────────────────────
    config = {
        "version": "2.9.0",
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
        # v2.9-specific
        "obs_dim": 1097,
        "n_agent_features": 8,
        "curriculum": "NONE (v2.7 baseline)",
        "proposal_b": True,
        "proposal_b_spike_threshold":    CRITIC_LOSS_SPIKE_THRESHOLD,
        "proposal_b_recovery_threshold": CRITIC_LOSS_RECOVERY_THRESHOLD,
        "proposal_b_reduction_factor":   LR_REDUCTION_FACTOR,
        "proposal_b_lr_floor":           LR_FLOOR,
        "proposal_b_window":             ROLLING_WINDOW,
        "proposal_b_cooldown_steps":     INTERVENTION_COOLDOWN_STEPS,
        "changes_vs_v27": [
            "AdaptiveLRCallback added (Proposal B)",
            "No curriculum (warmup_steps=0, identical to v2.7)",
            "No x1_overshoot feature (8 features/agent, identical to v2.7)",
        ],
    }

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_active = False
    if wandb_project:
        wandb_active = _init_wandb(wandb_project, run_name, config)

    # ── environments (no curriculum — same as v2.7) ───────────────────────────
    train_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,   # v2.7 behaviour: full episodes throughout
    )])
    eval_env = DummyVecEnv([lambda: IrrigationEnv(
        randomize=True,
        curriculum_warmup_steps=0,
    )])
    train_env.seed(seed)
    eval_env.seed(seed + 1000)

    # ── policy (v2.7 architecture — 1097-dim, 8 features/agent) ───────────────
    policy_kwargs = make_v27_sac_policy_kwargs(
        N=130,
        actor_hidden=ACTOR_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
    )

    # ── LR schedule ──────────────────────────────────────────────────────────
    lr_schedule = _make_lr_schedule(LR_START, LR_END)

    # ── model ─────────────────────────────────────────────────────────────────
    model = SAC(
        policy=V27CTDESACPolicy,
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
    grad_clip_callback   = GradClipCallback(max_grad_norm=MAX_GRAD_NORM)
    adaptive_lr_callback = AdaptiveLRCallback(
        lr_schedule=lr_schedule,
        spike_threshold=CRITIC_LOSS_SPIKE_THRESHOLD,
        recovery_threshold=CRITIC_LOSS_RECOVERY_THRESHOLD,
        reduction_factor=LR_REDUCTION_FACTOR,
        lr_floor=LR_FLOOR,
        window=ROLLING_WINDOW,
        cooldown_steps=INTERVENTION_COOLDOWN_STEPS,
        verbose=0,
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        rotating_buffer_callback,
        grad_clip_callback,
        adaptive_lr_callback,      # ← the only addition vs v2.7
    ])

    if wandb_active:
        try:
            from wandb.integration.sb3 import WandbCallback
            wandb_cb = WandbCallback(
                model_save_path=str(save_dir / "wandb_models"),
                model_save_freq=CHECKPOINT_FREQ,
                verbose=0,
            )
            callbacks = CallbackList([
                eval_callback,
                checkpoint_callback,
                rotating_buffer_callback,
                grad_clip_callback,
                adaptive_lr_callback,
                wandb_cb,
            ])
        except Exception as e:
            print(f"[WandB] WandbCallback unavailable ({e}); continuing without it.")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SAC training — v2.9.0 — seed {seed}")
    print(f"  Architecture: v2.7 (obs_dim=1097, 8 features/agent)")
    print(f"  Change vs v2.7: AdaptiveLRCallback (Proposal B)")
    print(f"    spike threshold:    critic_loss rolling-{ROLLING_WINDOW} > {CRITIC_LOSS_SPIKE_THRESHOLD}")
    print(f"    LR reduction:       ×{LR_REDUCTION_FACTOR} (floor {LR_FLOOR:.0e})")
    print(f"    recovery threshold: critic_loss rolling-{ROLLING_WINDOW} < {CRITIC_LOSS_RECOVERY_THRESHOLD}")
    print(f"    cooldown:           {INTERVENTION_COOLDOWN_STEPS} steps between reductions")
    print(f"  No curriculum (full 93-day episodes throughout)")
    print(f"  SAC: ent_coef=0.05 fixed, LR decay {LR_START:.0e}→{LR_END:.0e}")
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
    parser = argparse.ArgumentParser(description="Train SAC v2.9 (v2.7 + Proposal B)")
    parser.add_argument("--seed",             type=int, default=0)
    parser.add_argument("--output-dir",       type=str, default="results/rl")
    parser.add_argument("--wandb-project",    type=str, default=None)
    parser.add_argument("--total-timesteps",  type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    train_sac(
        seed=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        total_timesteps=args.total_timesteps,
    )
