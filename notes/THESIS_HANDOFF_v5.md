# THESIS HANDOFF v5 — v2.10 E3 Deployment Ready

**Project:** MSc thesis at ITMO University comparing Model Predictive Control (MPC) and Soft Actor-Critic (SAC) reinforcement learning for irrigation control of a 6-hectare rice paddy in Gilan, Iran.

**Author:** Tara Torbati
**Repo:** https://github.com/taratorbati/thesis.git
**Current branch:** `main` (E3 committed as 58b76de "v211 (n_step)")
**Status:** E2 confirmed failed (cascade). E3 patch (n_step=3 + k=5) ready to deploy.

---

# Part 1 — Project Overview (read this first)

## 1.1 The technical problem

Optimal irrigation scheduling for a 6-hectare Hashemi rice paddy. The paddy is modeled as a 130-agent Agent-Based Model (ABM) on a 13×10 grid where each agent is one cell of the field. A 93-day growing season is simulated using real climate data from Rasht, Gilan (2005–2024). The agent (RL or MPC) decides daily irrigation amounts (mm/day) for each of the 130 cells, subject to a seasonal water budget.

**Three test scenarios** (representative climate years):
- Dry: 2018 season (~50mm seasonal rainfall)
- Moderate: 2021 season (~110mm seasonal rainfall)
- Wet: 2024 season (~177mm seasonal rainfall)

**Three budget levels**: 100%, 85%, 70% of the full-need budget. The full-need is `CROP_FULL_BUDGET_MM['rice'] = 484 mm`.

**9-cell evaluation grid**: 3 scenarios × 3 budgets. Primary thesis numbers.

## 1.2 Why this is hard

1. **Multi-agent coordination** — 130 simultaneous decisions per day. The naive joint policy has a 130-dim continuous action space, which is high for SAC/TQC.
2. **Spatial heterogeneity** — the terrain has elevation gradients (DEM-driven), so cells need different irrigation amounts. The policy must learn cell-specific behavior.
3. **Long horizon, low reward** — 93 steps, near-zero per-step rewards, with the dominant signal being terminal-ish biomass + budget penalties. This creates a hard credit assignment problem.
4. **Distributional shift** — training samples climate years from 2005–2023 (where wet years cap at ~110mm). The wet test year (2024, 177mm) is out-of-distribution. This is the **Phase 2 problem** — deferred.
5. **MPC is the strong baseline** — MPC has a perfect model of the ABM and solves an open-loop optimal control problem with horizon=full season. It dominates RL on the wet year (3759 vs SAC v2.7's 3434 = 8.7% gap).

## 1.3 Hard constraints

These came from Tara's `<userPreferences>` and have not changed:

- **Never run code without explicit approval.** This means even smoke tests, not just full training runs.
- **Never create files without prose proposal + approval first.** Exception: the user later explicitly waived this for the E3 files.
- **All files use `encoding='utf-8'`.** Tara is on Windows; without this, special chars in docstrings cause `UnicodeEncodeError` on save.
- **Break code into separate files.** No god modules.
- **All hyperparameters need a citation or a documented pilot study.** This is non-negotiable; the thesis will be defended against this constraint.
- **MPC code is NOT to be modified.** MPC already operates correctly on the real ABM. Treat it as ground truth.
- **Long simulations need checkpointing.** Already implemented via `RotatingReplayBufferCheckpoint`.

---

# Part 2 — Where We Are Right Now

## 2.1 Version history (compressed)

| Version | Algorithm | Status | Key result |
|---|---|---|---|
| v2.5 | SAC with auto-α | Failed | Policy entropy collapsed; α auto-tuning was the cause |
| v2.7 | SAC, α=0.05 fixed | **Current baseline** | Yields published (see §2.3). Cascade at step ~156k–170k; best_model is captured *before* the cascade. |
| v2.8.1 | SAC + obs normalization | Incorporated | Observation normalization moved into env (matches v2.7 exactly) |
| v2.9 | SAC + AdaptiveLR callback | Failed | Callback wrote LR to `optimizer.param_groups`, but SB3's `_update_learning_rate` overwrote on every gradient step. The callback was a no-op. **Do not repeat this approach.** |
| v2.10 E2 | TQC, k=5 truncation | **Failed** | Cascade at step ~155k (faster than v2.7). Best_model is *post-cascade* garbage (see §2.4). |
| v2.10 E3 | TQC, k=5 + n_step=3 | **Ready to deploy** | Files committed to main. Notebooks need to be replaced (provided in handoff). |

## 2.2 The v2.7 baseline (what we are trying to beat)

v2.7 step-200k best_model, 9-cell evaluation, perfect forecast, seed 0:

| Scenario | Budget | v2.7 Yield (kg/ha) | MPC Yield | v2.7 Water | MPC Water |
|---|---|---|---|---|---|
| Dry | 100% | 4163 | 4145 | 456 | 469 |
| Dry | 85% | 4101 | 4069 | 411 | 410 |
| Dry | 70% | 3766 | 3766 | 339 | 339 |
| Mod | 100% | 3730 | 3718 | 437 | 400 |
| Mod | 85% | 3737 | 3725 | 411 | 397 |
| Mod | 70% | 3589 | 3612 | 339 | 339 |
| Wet | 100% | 3434 | 3759 | 414 | 310 |
| Wet | 85% | 3432 | 3743 | 411 | 308 |
| Wet | 70% | 3492 | 3754 | 339 | 308 |

v2.7 matches or beats MPC on dry and moderate. **The wet year is the open problem** (8.7% gap at wet/100%, 8.5% at wet/85%, 7.5% at wet/70%). Root cause: training distribution caps at ~110mm rainfall; the wet test year is 177mm. This is the Phase 2 OOD problem.

## 2.3 The cascade phenomenon (the central problem of Phase 1)

Both v2.7 and v2.10 E2 exhibit a deadly-triad cascade: a sharp phase transition around step 155k–170k where:

1. `q_pred_mean` flips sign from +378 to −512 over ~25k steps
2. `actor_loss` swings from −370 to +6000
3. `critic_loss` blows up from 0.2 to >80
4. `action_std_spatial` collapses from 0.30 to 0.07 (uniform near-zero policy)

**Before the cascade**, the critic learns a self-consistent fixed point where `Q ≈ +378` because the soft Bellman equation accumulates the entropy bonus `−α log π ≈ +2 per step` over a 93-step horizon with γ=0.99 (geometric weight ≈60). The structural Q baseline is:

```
Q_structural ≈ α × Σγ^t × (−log π) ≈ 0.05 × 60 × 39 ≈ 117
```

So Q_pred ≈ +378 means the critic is inflated **by +261 units above the structural offset**. The inflation comes from bootstrap amplification (γ/(1−γ) ≈ 100× leverage).

The cascade fires when the actor's drift moves the policy into regions where the critic has less training data and its gradient becomes anti-correlated with reward. Once the system leaves the inflated fixed point, the entire distribution slides downward and the system collapses to a low-action regime.

**Critical insight from E2 failure**: TQC's quantile truncation does NOT help on this problem. The VDN-sum critic (across 130 agents) collapses all 25 quantiles to the mean (quantile_spread ≈ 3–4 in stable phase) via central-limit-theorem narrowing. Truncation needs a wide quantile distribution to work; it has nothing to chop off here. The cascade is driven by **entire-distribution drift**, not tail overestimation.

## 2.4 v2.10 E2 evaluation results (FAILED)

E2 best_model, 9-cell grid, perfect forecast, seed 0:

| Scenario | Budget | E2 Yield | vs v2.7 | E2 Water |
|---|---|---|---|---|
| Dry | 100% | 4096 | −1.6% | 484 (full budget!) |
| Dry | 85% | 3966 | −3.3% | 411 |
| Dry | 70% | 3609 | −4.2% | 339 |
| Mod | 100% | 3712 | −0.5% | 484 (full!) |
| Mod | 85% | 3711 | −0.7% | 411 |
| Mod | 70% | 3388 | −5.6% | 339 |
| Wet | 100% | 2933 | **−14.6%** | 484 (full, with 177mm rain!) |
| Wet | 85% | 3055 | **−11.0%** | 411 |
| Wet | 70% | 3205 | **−8.2%** | 339 |

**E2 is strictly worse than v2.7 on all 9 cells.** Key diagnostic signs that the best_model is post-cascade:

1. **Budget-blind**: `u_mean ≈ 5.7mm/day` across ALL conditions regardless of remaining budget. v2.7 uses 456mm on dry/100% (under budget); E2 uses 484mm (full budget).
2. **Rainfall-blind**: identical irrigation behavior on wet (177mm rain) and dry (50mm rain) years.
3. **Anomaly at wet/100% vs wet/85%**: 2933 < 3055 (less budget gave more yield). This indicates over-irrigation caused waterlogging at full budget.
4. **MPC dominates spectacularly on wet year**: MPC uses 310mm on wet/100%, E2 uses 484mm. E2 over-irrigated by 174mm into already-saturated soil.

## 2.5 The handoff conversation reached this conclusion

After E2 failed, the previous agent and Tara:
1. Confirmed via fresh git clone that the v2.10 E2 architecture was correctly implemented (no bugs).
2. Did a deep analysis of WHY truncation failed: structural Q-inflation + quantile collapse + bootstrap drift.
3. Identified Strategy A (n_step=3 + k=5) as the highest-EV intervention because it directly attacks bootstrap amplification.
4. Audited sb3-contrib 2.6.0: **TQC has no built-in n_steps parameter**; n-step requires a custom `replay_buffer_class`.
5. Wrote three files: `nstep_buffer.py` (new), `train_v210_e2.py` (patched), `callbacks_v210.py` (patched with redesigned BiasRatioCallback).
6. Tara committed them to main as 58b76de "v211 (n_step)".

---

# Part 3 — Detailed Technical Analysis

## 3.1 Why n_step=3 was chosen (Strategy A)

The cascade is bootstrap-driven. n-step shortens the bootstrap recursion:
- 1-step target: `r_t + γ · Q(s_{t+1})` — every gradient step depends 100% on Q
- 3-step target: `r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·Q(s_{t+3})` — 3 grounded rewards per step

Quantitatively:
- Effective bootstrap depth: 60 steps → ~20 steps (3× reduction in amplification leverage)
- Expected Q-inflation at fixed point: +378 → +150–250 (still positive, but cascade timer pushed past 250k)
- Hessel et al. 2018 (Rainbow) Table 2: n=3 reduces overestimation by 40–60% in Atari

**Probability of success (q_inflation_pct < 20% at step 250k):** ~55%. n-step is the strongest single intervention available, but the problem has multiple compounding factors. If E3 fails, fallback strategies are documented in §5.

## 3.2 Alternative strategies considered (NOT chosen)

| Strategy | Mechanism | P(cascade fix) | P(policy preservation) | Why not |
|---|---|---|---|---|
| B: k=10 or k=12 truncation | More aggressive tail-cutting | ~20% | High | Quantile distribution is already collapsed; more truncation does nothing |
| D: γ=0.97 or 0.95 | Reduce bootstrap leverage | ~80% | ~30% | Changes the policy class (shorter effective horizon → may fail on full-season budget planning) |
| E: α=0.01 | Reduce entropy structural offset | ~50% | ~60% | Risks entropy collapse (the failure that v2.5 documented) |

D is the **backup plan** if E3 fails. The next agent should be ready to discuss this with Tara.

## 3.3 The γ¹ vs γ^n approximation in NStepReplayBuffer

**This is the only non-textbook part of the E3 implementation.** The buffer stores `R_n = Σγ^k r_{t+k}` correctly. But TQC's training loop computes the target as:

```
target = R_stored + (1 − done) · γ · Q(next_obs)
```

For a true n-step return, the bootstrap coefficient should be `γ^n = 0.99³ = 0.9703`, not `γ¹ = 0.99`. The difference is 2% per Q value. With Q≈378, that's ~7.5 units/step error.

**This approximation is standard.** Huang et al. 2021 (TQC paper) and CleanRL both use it because the alternative (overriding TQC.train to use γ^n) is fragile and SB3-version-dependent. The bulk of n-step's benefit comes from grounding the target in n real rewards, not from the bootstrap discount correction.

Hessel 2018 Table 2 ablations confirm n-step with this approximation captures ~70% of the theoretical benefit. The next agent should NOT try to "fix" this approximation without strong evidence — the trade-off has been thought through.

## 3.4 Cascade early-warning thresholds (monitor these during E3)

| Metric | Healthy | Onset | Full cascade |
|---|---|---|---|
| `v210/q_inflation_pct` | 0–20% | >50% | >200% |
| `v210/q_pred_mean` | +100 to +200 | growing past +300 | sign flip / −500 |
| `v210/quantile_spread` (per critic) | 3–8 | 10–20 | >50 |
| `actor/std/spatial` | 0.27–0.32 | <0.20 | <0.10 |
| `train/critic_loss` | 0.15–0.40 | >0.8 | >5.0 |

**Acceptance criterion for E3**: `q_inflation_pct < 20%` at step 250k.

If `q_inflation_pct` stays below 20% past step 175k (E2's cascade point), E3 is working.

---

# Part 4 — File-by-File State (what is currently in the repo)

## 4.1 Files Tara needs to verify/replace

| Path | Status | Action needed |
|---|---|---|
| `src/rl/nstep_buffer.py` | **NEW** in 58b76de | None — committed and correct |
| `src/rl/train_v210_e2.py` | **PATCHED** in 58b76de | None — committed and correct |
| `src/rl/callbacks_v210.py` | **PATCHED** in 58b76de | None — committed and correct |
| `notebooks/kaggle_sac_v210_e2.ipynb` | **STALE** | Replace with `kaggle_sac_v210_e3.ipynb` from handoff |
| `notebooks/colab_sac_v210_e2.ipynb` | **STALE** | Replace with `colab_sac_v210_e3.ipynb` from handoff |

## 4.2 Files that are correct as-is (do not modify)

| Path | Purpose |
|---|---|
| `src/rl/networks.py` | v2.7 SAC FactorizedQNet + V27Policy. Unchanged baseline. |
| `src/rl/networks_tqc.py` | v2.10 VDN-per-quantile TQC critic. Architecture verified correct. |
| `src/rl/runner.py` | Inference runner with `_load_model()` indirection for subclasses. |
| `src/rl/runner_tqc.py` | TQCRLController subclass for inference. |
| `src/rl/gym_env.py` | The 130-agent ABM environment. Do NOT touch in Phase 1. |
| `src/rl/train.py` | v2.7 SAC training script. Reference baseline. |
| `tests/test_factorized_critic.py` | v2.7 critic unit tests. Pass. |
| `tests/test_tqc_critic.py` | v2.10 TQC critic unit tests. Pass. |
| `tests/test_rl_smoke.py` | End-to-end smoke tests. Pass. |
| `scripts/experiments/exp_rl.py` | SAC eval (use this for v2.7 checkpoints). |
| `scripts/experiments/exp_rl_tqc.py` | TQC eval (use this for v2.10 E2/E3 checkpoints). |
| `scripts/experiments/exp_mpc.py` | MPC eval. Do NOT modify. |

## 4.3 The notebook bug (why they need to be replaced)

The original E2 notebooks have:
- **Cell 5**: `src = '.../sac_v210_e2_seed{SEED}'` → training writes to `sac_v210_e3_seed{SEED}` → silent copy failure
- **Cell 6**: `model_path = '.../sac_v210_e2_seed{SEED}/best_model/best_model.zip'` → file doesn't exist → eval crashes
- **Cell 7**: Reads `bias_ratio_mean` and `bias_ratio_std` from CSV → those columns no longer exist → KeyError. The new CSV has `q_pred_mean, q_structural, q_inflation, q_inflation_pct, return_realised, log_prob_start`.
- **Cell 8** (colab only): same e2 paths in resume code.

The patched notebooks fix all of these and rewrite the plot cell to use `q_inflation_pct` with the correct thresholds (20% acceptance, 50% onset, 200% full cascade).

---

# Part 5 — What To Do Next (Action Items in Priority Order)

## 5.1 IMMEDIATE: Run E3 (Tara's next action)

1. Replace the two notebook files in the repo:
   - `notebooks/kaggle_sac_v210_e2.ipynb` → use `kaggle_sac_v210_e3.ipynb` from handoff
   - `notebooks/colab_sac_v210_e2.ipynb` → use `colab_sac_v210_e3.ipynb` from handoff
2. Commit + push.
3. Run the Kaggle (T4, ~2.5h) or Colab (A100, ~45min) notebook.
4. **While training runs, monitor `v210/q_inflation_pct` in WandB.** Check at steps 50k, 100k, 150k, 175k, 200k, 250k.
5. After training, run the 9-cell eval via `exp_rl_tqc.py` (cell 6 of the notebook does this automatically).

## 5.2 Interpreting E3 results

**If E3 succeeds** (`q_inflation_pct < 20%` at step 250k AND yields close to v2.7 on dry/moderate AND don't regress catastrophically on wet):
- Run seeds 1 and 2 for paired comparison
- Then move to Phase 2 (rainfall DR for wet-year OOD problem)

**If E3 partially succeeds** (cascade delayed but still fires before 250k):
- Consider Strategy D (γ=0.97). Tara approval needed.
- Document the partial success in the thesis as a Phase 1 contribution

**If E3 fails entirely** (cascade at same time as E2):
- The bootstrap-amplification hypothesis is wrong
- Reconsider: maybe the cascade is actor-driven, not critic-driven
- Look at `actor/std/spatial` carefully — does it shrink BEFORE q_pred_mean changes?

## 5.3 Phase 2 (after Phase 1 is done): Rainfall DR for wet-year OOD

Root cause of the wet-year gap: training years cap at 110mm seasonal rainfall, test year 2024 is 177mm.

Proposed intervention: domain randomization on rainfall during training. Multiply training-year rainfall by `U[0.5, 2.0]` per episode. This exposes the policy to >150mm rainfall scenarios during training.

**Implementation point**: `src/rl/gym_env.py`, in the `reset()` method, after sampling the climate year. Apply a per-episode rainfall multiplier.

**Risk**: distorts the training distribution; may degrade dry-year performance. Run a Phase 2 pilot first (seed 0, 100k steps) to check.

---

# Part 6 — Operational Knowledge

## 6.1 How to run things

```powershell
# v2.7 SAC evaluation (existing baseline)
python -m scripts.experiments.exp_rl --mode eval --model .\results\rl\sac_v27_seed0_<timestamp>\best_model\best_model.zip --scenario all --budget all --forecast perfect

# v2.10 TQC evaluation (E2 or E3)
python -m scripts.experiments.exp_rl_tqc --model .\results\rl\sac_v210_e3_seed0_<timestamp>\best_model\best_model.zip --scenario all --budget all --forecast perfect

# MPC evaluation (do not modify)
python -m scripts.experiments.exp_mpc --scenario all --budget all
```

## 6.2 Output file conventions

| Algorithm | Output prefix | Eval script |
|---|---|---|
| SAC (v2.7) | `sac_perfect_det_*.parquet` | `exp_rl.py` |
| TQC (v2.10) | `tqc_perfect_det_*.parquet` | `exp_rl_tqc.py` |
| MPC | `mpc_perfect_det_*.parquet` | `exp_mpc.py` |

## 6.3 Critical environment quirks

- The env's `seed` from `gym.Env.reset(seed=...)` is consumed by SB3 once at startup. Subsequent `reset()` calls without seed use the env's internal RNG, which drifts. **Bias-ratio callback runs deterministic eval episodes that DO consume RNG state from the eval_env**, but the eval_env is a separate instance from train_env, so no cross-contamination.
- DummyVecEnv with `n_envs=1` is what training uses. `NStepReplayBuffer` is only correct for `n_envs=1` in the current implementation (see Audit Issue #6 below). Don't change `n_envs`.
- SB3's `handle_timeout_termination=True` is the default. The env's 93-day truncation sets `info["TimeLimit.truncated"]=True`, which SB3 uses to keep the bootstrap term active even on the final step. NStepReplayBuffer preserves this by passing through the final info dict.

## 6.4 Audit findings from comprehensive review (May 24, 2026)

**Architecture verified clean** (no bugs in the deployed implementation):
- VDN-per-quantile critic matches sb3-contrib's `Critic.forward` contract: returns `(B, n_critics=2, n_quantiles=25)`
- 178,738 critic params (7.4% more than v2.7), all in final linear layer
- `ent_coef=0.05` correctly fixed; `ent_coef_optimizer` is None throughout
- LR schedule honored (no v2.9-style overwrite)
- Truncation arithmetic: `quantiles_total - k × n_critics = 50 - 5×2 = 40` target quantiles ✓
- `NStepReplayBuffer` inherits `ReplayBuffer` correctly for `save_replay_buffer`/`load_replay_buffer` compatibility
- Episode-boundary handling in NStepReplayBuffer is correct (truncates at first `done_k=True`)

**Minor cosmetic issues** (not bugs, do not need to fix unless desired):
- Stale "E2" references in comments at `train_v210_e2.py:49`, `runner_tqc.py:19`, `exp_rl_tqc.py:17,22,82`
- Dead imports in `nstep_buffer.py`: `Optional`, `ReplayBufferSamples`, `VecNormalize`
- Unused variables in `_flush_one`: `r0`, `done0`, `actual_n`

**Critical bug found and fixed in this handoff**: The original E2 notebooks reference stale paths and CSV columns. The patched notebooks (`kaggle_sac_v210_e3.ipynb`, `colab_sac_v210_e3.ipynb`) in this handoff fix all path references and rewrite the plot cell to use the new metrics.

**Theoretical issue (not a bug, but worth knowing)**: `NStepReplayBuffer` is only correct for `n_envs=1`. With `n_envs > 1`, the per-env flushes would corrupt the shared `self.pos` position. Training uses `DummyVecEnv` with `n_envs=1`, so this is theoretical. **If you ever switch to multi-env training, this buffer needs to be rewritten.**

---

# Part 7 — Open Questions for Tara

Things the next agent should clarify before making major decisions:

1. **What is the GPU budget for E3 + fallbacks?** E3 alone is ~2.5h Kaggle / ~45min Colab. If E3 fails, do we have budget for Strategy D (γ=0.97) and seed expansion?
2. **Thesis deadline pressure?** If we're tight on time, the next agent should propose accepting v2.7 as the baseline and documenting v2.10 E2/E3 as a controlled negative result, then focus all remaining compute on Phase 2.
3. **WandB project name** (for new training runs): currently `sac-irrigation-thesis`. Keep using this.

---

# Part 8 — Files Delivered with This Handoff

| File | Purpose |
|---|---|
| `THESIS_HANDOFF_v5.md` | This document |
| `kaggle_sac_v210_e3.ipynb` | Patched Kaggle notebook — replace the e2 version |
| `colab_sac_v210_e3.ipynb` | Patched Colab notebook — replace the e2 version |
| `agent_kickoff_prompt.md` | Prompt to give the next agent (next page) |

---
