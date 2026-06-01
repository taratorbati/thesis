# src/rl/train_v219_td3.py  v2.19-TD3
# -----------------------------------------------------------------------------
# Train a TD3 controller: deterministic VDN-shared actor + v2.11 LayerNorm VDN
# critic.  See src/rl/networks_td3.py for the architecture rationale.
#
# WHY TD3 (carried diagnosis):
#   v2.18 (SAC, alpha=0.002 + late noise reinjection) matched MPC on the key
#   metrics (mean yield 3785 = 99.3% of MPC-perfect 3810; mean waterlog 18.2;
#   wet x1 136 vs MPC 130) and is a legitimate result.  But at alpha=0.002 the
#   critic calibration wobbled (q_pred_mean briefly negative at 100-125k, then
#   recovered).  TD3 is the principled way to push entropy to zero: it drops the
#   entropy objective (which pinned the actor mean at the 6 mm action-centre) and
#   replaces the implicit smoothing entropy provided with explicit TARGET-POLICY
#   SMOOTHING -- the stabiliser whose absence let v2.7 cascade.  Goal: close the
#   last ~6 mm of wet x1 (136 -> ~130) and get a cleaner critic, while keeping
#   the VDN+LayerNorm machinery that has prevented divergence since v2.11.
#
# WHAT CHANGES vs the SAC family:
#   * Algorithm: SAC -> TD3 (stable_baselines3).
#   * Actor: deterministic _TD3SharedActor (no log_std, no entropy).  IDENTICAL
#     feature pipeline (shared LeakyReLU MLP + 2x-1 recenter + agent-major
#     reshape) as the v2.16 actor; only the head + output are deterministic.
#   * Exploration: explicit NormalActionNoise (SAC got exploration from policy
#     stochasticity; TD3 is deterministic so it needs collection noise).
#     Decays 0.20 -> 0.05 over 100k steps and holds 0.05 (TD3 keeps a small
#     floor; it does not anneal to 0 like the SAC Path-3 experiments).
#   * Target-policy smoothing (TD3-internal, set via constructor):
#     target_policy_noise=0.2, target_noise_clip=0.5 (Fujimoto et al. 2018).
#   * Delayed actor updates: policy_delay=2 (TD3 default).
#
# WHAT STAYS IDENTICAL:
#   VDN LayerNorm twin-Q critic, gamma=0.99, tau=0.005, asymmetric actor LR (5x),
#   LR schedule 3e-4 -> 5e-5, buffer 250k, 250k steps, RAIN_REF=30, linear r6,
#   1097-dim obs.  Evaluation uses the SAME 9-cell grid; runner.py is extended
#   to dispatch the TD3 checkpoint (marker 2.19 + no log_std -> TD3.load).
#
# ACCEPTANCE CRITERIA (evaluate on the 9-cell grid; decide on x1/waterlog):
#   PRIMARY:   wet x1 median < 134 mm (v2.18: 136; MPC: 130); wet waterlog < 32
#              (v2.18: 37; MPC: 18).  I.e. close the gap v2.18 left to MPC.
#   SECONDARY: wet water < 330 mm (v2.18: 336); mean yield >= 3780 (no dry-year
#              regression vs v2.18's 3999 dry).
#   STABILITY: critic_loss < 100 throughout; q_pred_mean never negative;
#              |q_inflation_pct| < 80% (TARGET smoothing should make this
#              CLEANER than v2.18's alpha=0.002 run, which is the whole point).
#   ATTRIBUTION: if TD3 reaches MPC x1/waterlog where SAC alpha=0.002 stalled at
#              136, that isolates the entropy mu-pin as the binding constraint.
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
from src.rl.callbacks_exploration import ExplorationNoiseDecayCallback
from src.rl.train import (
    RotatingReplayBufferCheckpoint,
    GradClipCallback,
    _make_lr_schedule,
    _init_wandb,
)


# -----------------------------------------------------------------------------
# AsymmetricLRTD3 -- TD3 mirror of AsymmetricLRSAC.
# Bumps the actor optimiser LR to actor_lr_mult x the scheduled (critic) LR,
# the same trick that helped the SAC actor escape the LayerNorm-bounded critic
# gradient.  SB3 TD3 exposes self.actor.optimizer (the actor's Adam).
# -----------------------------------------------------------------------------
ACTOR_LR_MULT = 5.0


class AsymmetricLRTD3(TD3):
    actor_lr_mult: float = ACTOR_LR_MULT

    def _update_learning_rate(self, optimizers) -> None:  # type: ignore[override]
        super()._update_learning_rate(optimizers)
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
LEARNING_STARTS  = 1_000
GRADIENT_STEPS   = 1
TRAIN_FREQ       = 1
POLICY_DELAY     = 2          # TD3 delayed actor updates

# TD3 target-policy smoothing (Fujimoto et al. 2018 defaults).
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP   = 0.5

# Exploration (collection-time) noise -- TD3 has no entropy, so this is the
# sole exploration source.  Decays to a small floor (not 0).
EXPLORE_SIGMA_START = 0.20
EXPLORE_SIGMA_END   = 0.05
EXPLORE_DECAY_STEPS = 100_000
EXPLORE_LOG_FREQ    = 1_000

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


def train_td3_v219(
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
    explore_sigma_start: float = EXPLORE_SIGMA_START,
    explore_sigma_end: float = EXPLORE_SIGMA_END,
    explore_decay_steps: int = EXPLORE_DECAY_STEPS,
) -> TD3:
    """Train TD3 v2.19: deterministic VDN actor + v2.11 LayerNorm VDN critic."""
    run_name = f"td3_v219_seed{seed}"
    save_dir = Path(output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": "2.19.0-TD3",
        "experiment": "v219_td3_deterministic_vdn_actor_PLUS_v211_LN_critic",
        "seed": seed,
        "algorithm": "TD3 (stable_baselines3) + asymmetric actor LR",
        "policy_class": "TD3VDNPolicy (deterministic _TD3SharedActor, marker=2.19)",
        "total_timesteps": total_timesteps,
        "gamma": gamma,
        "tau": TAU,
        "actor_lr_mult": actor_lr_mult,
        "target_policy_noise": target_policy_noise,
        "target_noise_clip": target_noise_clip,
        "policy_delay": policy_delay,
        "explore_sigma_start": explore_sigma_start,
        "explore_sigma_end": explore_sigma_end,
        "explore_decay_steps": explore_decay_steps,
        "rain_normaliser": rain_normaliser,
        "reward_overshoot_mode": reward_overshoot_mode,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "hypothesis": (
            "Removing the entropy objective (deterministic TD3) lets the actor "
            "reach the tanh=-1 action boundary (0 mm) that SAC's entropy "
            "Jacobian pinned away from, closing the residual wet x1 gap (136 -> "
            "~130) that alpha=0.002 left; target-policy smoothing replaces the "
            "smoothing entropy provided, keeping the critic at least as stable."
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
        learning_starts=LEARNING_STARTS,
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
    # the realised return).  TD3 satisfies both.
    bias_ratio_cb = BiasRatioCallback(
        eval_env=bias_eval_env,
        eval_freq=BIAS_RATIO_FREQ,
        n_eval_episodes=BIAS_RATIO_N_EPISODES,
        save_path=str(save_dir),
        verbose=1,
    )
    action_stats_cb = ActionStatsCallback(log_freq=ACTION_STATS_FREQ)
    optimizer_lr_cb = OptimizerLRCallback(log_freq=LR_LOG_FREQ)

    noise_decay_cb = ExplorationNoiseDecayCallback(
        sigma_start=explore_sigma_start,
        sigma_end=explore_sigma_end,
        decay_steps=explore_decay_steps,
        log_freq=EXPLORE_LOG_FREQ,
        csv_path=str(save_dir / "exploration_sigma_log.csv"),
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
    print(f"  TD3 training - v2.19 - seed {seed}")
    print(f"  Actor:  deterministic _TD3SharedActor (LeakyReLU + 2x-1 recenter, marker=2.19)")
    print(f"  Critic: _V211FactorizedContinuousCritic (VDN twin-Q + LayerNorm, UNCHANGED)")
    print(f"  target_policy_noise={target_policy_noise}  clip={target_noise_clip}  policy_delay={policy_delay}")
    print(f"  explore noise: {explore_sigma_start:.2f} -> {explore_sigma_end:.2f} over {explore_decay_steps:,} (floor held)")
    print(f"  rain_norm={rain_normaliser:.1f}  r6={reward_overshoot_mode}  gamma={gamma}  tau={TAU}  actorLRx{actor_lr_mult}")
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
            "Train TD3 v2.19: deterministic VDN-shared actor + v2.11 LayerNorm "
            "VDN critic, to close the residual wet-year x1 gap (136->~130) that "
            "SAC alpha=0.002 left, by removing the entropy action-pin and adding "
            "target-policy smoothing.  Evaluated by runner.py via the 2.19 marker."
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
    parser.add_argument("--explore-sigma-start", type=float, default=EXPLORE_SIGMA_START)
    parser.add_argument("--explore-sigma-end",   type=float, default=EXPLORE_SIGMA_END)
    parser.add_argument("--explore-decay-steps", type=int,   default=EXPLORE_DECAY_STEPS)
    args = parser.parse_args()

    train_td3_v219(
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
        explore_sigma_start=args.explore_sigma_start,
        explore_sigma_end=args.explore_sigma_end,
        explore_decay_steps=args.explore_decay_steps,
    )
