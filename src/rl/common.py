# src/rl/common.py
# -----------------------------------------------------------------------------
# Shared training utilities for the SAC and TD3 irrigation controllers:
#   * learning-rate schedule + Weights & Biases bootstrap,
#   * lightweight SB3 callbacks (replay-buffer snapshot, gradient clipping),
#   * the two algorithm subclasses that give the actor its own learning rate.
#
# Collecting them here lets both train_sac.py and train_td3.py stay
# self-contained without importing each other.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback


# ── learning-rate schedule ────────────────────────────────────────────────────
def make_lr_schedule(lr_start: float, lr_end: float) -> Callable[[float], float]:
    """Linear decay from ``lr_start`` (progress=1.0) to ``lr_end`` (progress=0.0)."""
    def schedule(progress_remaining: float) -> float:
        return lr_end + (lr_start - lr_end) * progress_remaining
    return schedule


# ── Weights & Biases (optional; training runs fine without it) ────────────────
def _resolve_wandb_api_key() -> Optional[str]:
    """Find a W&B API key from the environment or a Kaggle/Colab secret store."""
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


def init_wandb(project: str, run_name: str, config: dict) -> bool:
    """Initialise a W&B run. Returns True on success, False (and continues) otherwise."""
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


# ── callbacks ─────────────────────────────────────────────────────────────────
class RotatingReplayBufferCheckpoint(BaseCallback):
    """Save the replay buffer to a single overwriting file at each checkpoint.

    A single rolling snapshot (rather than one file per checkpoint) keeps the
    on-disk footprint bounded — full per-checkpoint buffers can reach hundreds
    of gigabytes over a long run.
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
    """Clip the policy's gradient norm after every update step."""

    def __init__(self, max_grad_norm: float = 1.0):
        super().__init__(verbose=0)
        self.max_grad_norm = max_grad_norm

    def _on_step(self) -> bool:
        if self.model is not None and hasattr(self.model, "policy"):
            torch.nn.utils.clip_grad_norm_(
                self.model.policy.parameters(), self.max_grad_norm
            )
        return True


# ── algorithm subclasses with an asymmetric actor learning rate ───────────────
# SB3 re-sets every optimiser to the scheduled LR at the top of each train()
# call, so an LR change only sticks if it is applied inside
# ``_update_learning_rate``. Both subclasses below let the base class set the
# scheduled LR first (keeping critic/entropy LR and SB3 logging correct), then
# adjust only the actor optimiser.

class AsymmetricLRSAC(SAC):
    """SAC whose actor optimiser runs at ``actor_lr_mult`` times the critic LR.

    Counters the LayerNorm-bounded critic gradient: with the critic's effective
    step size held down by LayerNorm, a larger actor LR keeps the policy moving.
    Configure via the class attribute before construction::

        AsymmetricLRSAC.actor_lr_mult = 5.0
    """

    actor_lr_mult: float = 1.0

    def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
        super()._update_learning_rate(optimizers)
        for pg in self.actor.optimizer.param_groups:
            pg["lr"] = pg["lr"] * self.actor_lr_mult


class WarmupAsymmetricLRTD3(TD3):
    """TD3 with an asymmetric actor LR multiplier and an optional warm-up ramp.

    The ramp holds the actor LR low and raises it linearly to full over the
    first ``actor_warmup_updates`` gradient updates ("let the critic lead"),
    while the critic LR is full from step one. With ``actor_lr_mult == 1.0`` and
    ``actor_warmup_updates == 0`` this is an exact no-op (stock TD3). Configure
    via class attributes before construction::

        WarmupAsymmetricLRTD3.actor_lr_mult        = 1.0
        WarmupAsymmetricLRTD3.actor_warmup_updates = 0
    """

    actor_lr_mult: float = 1.0
    actor_warmup_updates: int = 0

    def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
        super()._update_learning_rate(optimizers)

        mult = float(self.actor_lr_mult)
        warm = int(self.actor_warmup_updates)
        if mult == 1.0 and warm <= 0:
            return

        factor = mult
        if warm > 0:
            # self._n_updates counts updates completed in prior train() calls,
            # so the ramp is monotone: 0 at the first post-warmup update,
            # reaching `mult` after `warm` updates.
            factor *= min(1.0, float(self._n_updates) / float(warm))
        if factor == 1.0:
            return

        for pg in self.actor.optimizer.param_groups:
            pg["lr"] = pg["lr"] * factor
