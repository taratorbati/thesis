# Change Specification — `gym_env.py` v2.8 (Proposal C + x1-overshoot)

**Project:** ITMO MSc thesis — *Modern Control Methods for Agricultural Irrigation*
**Author of record:** Tara Torbati
**Document purpose:** Specify the v2.7 → v2.8 changes. Two interventions are bundled: a *curriculum* (shorter early-season episodes during training warmup) and an *explicit overshoot feature* in the per-agent observation. Together these address the structural weaknesses observed in the v2.7 seed-0 and seed-1 evaluations.

---

## 1 — Motivation

The v2.7 evaluation across two random seeds (seed 0 and seed 1) confirmed two structural weaknesses that no v2.7 design choice could close:

### 1.1 The x1-conditioning weakness

In wet/100% under perfect forecast, both v2.7 seeds produce per-day irrigation that correlates strongly with rainfall (corr = -0.49 for seed 0, -0.38 for seed 1) but only weakly or zero with current soil moisture (corr(u, x1) = +0.05 seed 0, +0.48 seed 1). MPC at the same cell shows corr(u, x1) = -0.58 — the controller backs off aggressively when soil is already wet, regardless of rainfall forecast.

The empirical consequence is the residual wet-year yield gap of 6–9% across both seeds and the persistent 67–79 waterlog days (vs MPC's 18–20). The policy responds to *expected future* rain but not to *currently observed* high soil moisture.

The cause is that the reward function's overshoot penalty fires only when `max(x1 − FC, 0) > 0`, but the obs encodes x1 normalised against `(WP, FC)` so `x1_norm = 1.0` is the threshold rather than zero. The agent has to learn the threshold from samples, and forecast-based control was sufficient to avoid 95% of overshoots — so the threshold was never properly encoded in the policy. An explicit feature `max(x1 − FC, 0) / FC` makes the overshoot regime additively visible in the observation and creates a direct gradient signal whenever the regime is entered.

### 1.2 The training-instability weakness

Both v2.7 seeds peaked at step 200k and degraded thereafter:
- Seed 0: eval mean reward at step 200k = −0.26 (best), at step 250k = −22.6, at step 500k = −8.5.
- Seed 1: eval mean reward at step 200k = −1.03 (best), at step 250k = −10.9.

The critic-loss exploded in seed 0 starting at step ~165k (training run was capped at 250k for seed 1, but the peak-then-degrade pattern is identical). The root cause is the classic deadly triad in cooperative MARL: function approximation + bootstrapping + off-policy data, amplified by 93-day episode length producing high-variance return targets that grow during training.

A *curriculum* that truncates episodes at day 60 during the first 50k training steps gives the critic a smaller-magnitude target distribution to fit during the initial value-function learning phase. Once the critic is calibrated, the full 93-day episode lengthens the return horizon and the critic continues to learn — but starting from a numerically stable baseline. This is a standard curriculum-learning pattern for long-horizon RL problems.

The hope is that the v2.7 peak at step 200k extends to step 350k+ with the curriculum, producing a better-converged policy by the time EvalCallback saves the best checkpoint.

---

## 2 — Goals and non-goals of v2.8

### Goals

- Add `x1_overshoot_norm` as the 9th per-agent observation feature, making the FC overshoot regime explicitly visible to the actor.
- Introduce an episode-length curriculum: 60-day episodes during the first 50 000 training steps, 93-day episodes thereafter.
- Preserve full v2.7 backwards compatibility for evaluation: the runner must continue to load v2.7 checkpoints with 8-feature obs and produce correctly-shaped outputs.

### Non-goals

- No change to the reward function. The four-term reward `r = r1 + r2 + r3 + r6` is unchanged.
- No change to the SAC hyperparameters (`ent_coef=0.05`, `max_grad_norm=1.0`, LR decay 3e-4 → 5e-5).
- No change to the VDN factorized critic architecture.
- No change to `abm.py`, `soil_data.py`, `climate_data.py`, `src/precompute.py`, `src/terrain.py`.
- No change to the MPC implementation, baselines, or evaluation harness.

The v2.8 → v2.7 comparison will therefore be a clean ablation: one new feature, one curriculum schedule.

---

## 3 — Change 1: Add `x1_overshoot_norm` as 9th per-agent feature

### 3.1 Definition

```python
x1_overshoot_norm = np.clip(np.maximum(self._abm.x1 - _FC_MM, 0.0) / _FC_MM, 0.0, 1.0)
```

- Equals 0 whenever `x1 ≤ FC` (the "healthy" regime).
- Grows linearly with overshoot once `x1 > FC`.
- Bounded above at 1.0 (corresponds to `x1 = 2·FC = 280 mm`, which exceeds the physical saturation cap of `θ_sat × θ_5 = 220 mm`, so the clip never actually fires in practice — but it guards against numerical edge cases).
- Computed per agent, so it varies across the 130 agents.

### 3.2 Why this feature specifically

The reward term `r6 = -ALPHA6 × mean(max(x1 − FC, 0)²) / FC²` is essentially the per-agent overshoot squared, averaged. Putting `max(x1 − FC, 0) / FC` as a feature means the agent observes the *same quantity* that gets squared in its reward. This makes the gradient signal from r6 maximally informative about which feature should change.

### 3.3 Effect on dimensions

- Per-agent block: 8 → 9 features.
- OBS_DIM: 1097 → 1227 (= 9 × 130 + 9 + 48).
- PER_AGENT_INPUT_DIM (actor): 65 → 66.
- PER_AGENT_CRITIC_INPUT_DIM: 66 → 67.

---

## 4 — Change 2: Episode-length curriculum

### 4.1 Schedule

- For env step counts `0` through `CURRICULUM_WARMUP_STEPS - 1` (default `50 000`): episodes truncate at `CURRICULUM_SHORT_LEN` (default `60`) days.
- For env step counts `≥ CURRICULUM_WARMUP_STEPS`: episodes truncate at the full season length `_K = 93` days.

The transition happens at the boundary of an episode reset, not mid-episode. If the env is in step 49 998 of an episode that's already running at the short length, that episode completes at day 60. The next episode starts under the full-length schedule.

### 4.2 Implementation

Each env instance tracks its own `_global_step_count` counter, incremented on every `step()`. At reset, the episode's truncation day is set to:
```python
truncation_day = (
    CURRICULUM_SHORT_LEN
    if self._global_step_count < self._curriculum_warmup_steps
    else _K
)
```
and stored as `self._truncation_day`. The `step()` method uses `truncated = (self._day >= self._truncation_day)`.

### 4.3 Effect on the v2.6 → v2.7 → v2.8 chain

- v2.6: episodes terminated at budget exhaustion (effective length ~83 days mean).
- v2.7: episodes always 93 days.
- v2.8: episodes 60 days for first ~50k steps, 93 days thereafter.

Approximately 5% of v2.8 training transitions (60 days × 833 episodes ≈ 50 000 of 250 000 total transitions) come from short episodes. The remaining 95% come from full-season episodes, which is plenty for the policy to learn the late-season dynamics.

### 4.4 Why 50k steps and 60 days

- **50 000 warmup steps** is roughly 100 short episodes worth of data, sufficient for the critic to develop a stable initial value function before being exposed to the high-variance long-horizon targets.
- **60 days** covers the entire vegetative phase and early reproductive phase, which contains 80% of the cumulative growth-stage variety. Days 60–93 are the late grain-fill phase, which has high agronomic value but doesn't introduce qualitatively new control challenges — the agent has already seen all the soil-moisture and budget-management decision types by day 60. Truncating there reduces episode return variance by ~35% while preserving the diversity of state trajectories needed for learning.

### 4.5 The mid-curriculum boundary handling

When the curriculum transitions from short to full episodes (somewhere around step 50 000), the agent will experience a one-time distributional shift in the return targets. The deadly-triad concern is whether this shift triggers the same explosion that v2.7 saw at step 165k. Empirically, the shift is much smaller than v2.7's full-season-from-day-zero return distribution, so the critic should adapt without explosion. If it doesn't, that's an interesting result worth reporting.

---

## 5 — What does not change

- Reward weights, action space, budget mechanism, episode-termination logic (still no early termination on budget exhaustion).
- SAC training hyperparameters.
- VDN architecture, hidden widths, gradient clipping.
- Test infrastructure other than dimension constants and curriculum-aware tests.
- v2.7 checkpoint loading continues to work via the same legacy-class mechanism that already handled v2.6.

---

## 6 — Files modified

| File | Change |
|---|---|
| `src/rl/gym_env.py` | v2.7.0 → v2.8.0. Add `x1_overshoot_norm` feature, add curriculum tracking, update OBS_DIM 1097 → 1227. |
| `src/rl/networks.py` | v2.7.0 → v2.8.0. Update v2.8 constants (`N_AGENT_FEATURES=9`, `OBS_DIM_DEFAULT=1227`, etc.). Preserve V27_* and V26_* legacy constants and classes. |
| `src/rl/runner.py` | Detection table adds `dim=67, flat → CTDESACPolicy` (v2.8 default). v2.7 (dim=66) and earlier classes remain. `_build_obs` branches on checkpoint version: v2.8 → 9-feature, v2.7 → 8-feature, v2.6 → 5-feature. |
| `src/rl/train.py` | Version string 2.7.0 → 2.8.0. Config dict reflects v2.8 changes. Curriculum kwargs exposed. |
| `tests/test_rl_smoke.py` | OBS_DIM 1097 → 1227 in all assertions. Add `test_x1_overshoot_feature_*` and `test_curriculum_*` tests. |
| `tests/test_factorized_critic.py` | Comment update only (OBS_DIM flows from constant). Add `test_v27_legacy_load_shape` test (similar to v2.6 legacy test, defends v2.7 backwards compat). |

---

## 7 — Validation checklist before retraining

1. All existing v2.7 smoke tests + the new v2.8 tests pass with `OBS_DIM = 1227`.
2. `tests/test_factorized_critic.py` passes; new test confirms v2.7 legacy load still works.
3. Manual probe: `env.reset(); obs[8::9]` should produce a 130-vector that's all zeros at reset (because x1 = FC initially, so overshoot = 0).
4. Manual probe: after stepping with max irrigation for 5 days at wet/100, `obs[8::9]` should be non-zero on at least some agents (those where x1 has been pushed above FC).
5. Manual probe: env with `curriculum_warmup_steps=10` truncates at day 60 for first ~3 episodes, then at day 93 thereafter.
6. 10k-step pilot training run with v2.8 environment completes without critic explosion. Reward magnitudes per step remain in the same scale as v2.7 (the obs grew but reward shape is unchanged).

If all six pass, the 250k training run can proceed under the same Colab Pro / Kaggle setup as v2.7.

---

## 8 — Open questions for after retraining

- If v2.8 still doesn't close the wet-year gap, the next investigation is whether SAC and MPC are optimizing different objectives (MPC's "leave water unused" strategy may be fundamentally unavailable to a randomized-budget-trained SAC).
- The curriculum boundary at step 50 000 should be inspected on the loss curve. If a visible critic-loss spike appears at that step, the curriculum window may need lengthening for subsequent seeds.
- Run v2.8 on seeds 0 and 1 first for direct paired comparison with the v2.7 baseline. If both v2.8 seed-0 and v2.8 seed-1 outperform their v2.7 counterparts, the protocol is validated and the campaign can extend to seeds 2, 3, 4 for a full N=5 sample.

---

*End of specification.*
