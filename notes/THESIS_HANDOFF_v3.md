# THESIS PROJECT HANDOFF DOCUMENT — v3
**Student:** Tara Torbati  
**Institution:** ITMO University, MSc  
**Topic:** Modern Control Methods for Agricultural Irrigation  
**Repository:** https://github.com/taratorbati/thesis.git  
**Document date:** May 19, 2026  
**Purpose:** Complete technical handoff for continuation in a new chat session. Contains full project history, all design decisions with rationale, all results, all problems identified in the v2→v3 chat session, and the exact state of the v2.9 work-in-progress.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Controller 1: MPC (CasADi+IPOPT)](#3-controller-1-mpc)
4. [Controller 2: SAC-RL Evolution History](#4-controller-2-sac-rl-evolution-history)
5. [v2.6 Results — Starting Baseline](#5-v26-results--starting-baseline)
6. [v2.6 Wet-Year Pathology Analysis](#6-v26-wet-year-pathology-analysis)
7. [v2.7 Design Decisions and Rationale](#7-v27-design-decisions-and-rationale)
8. [v2.7 Results — Corrected (v2.8.1 runner fix)](#8-v27-results--corrected-v281-runner-fix)
9. [v2.7 Noisy-Forecast Results](#9-v27-noisy-forecast-results)
10. [Critical Analysis of Stability Proposals](#10-critical-analysis-of-stability-proposals)
11. [v2.8 Design, Execution and Failure Analysis](#11-v28-design-execution-and-failure-analysis)
12. [Code Fixes Applied (v2.8.1)](#12-code-fixes-applied-v281)
13. [Full Comparative Analysis (all controllers, all cells)](#13-full-comparative-analysis-all-controllers-all-cells)
14. [New Findings from v3 Chat Session](#14-new-findings-from-v3-chat-session)
15. [v2.9 Design — Current Work-in-Progress](#15-v29-design--current-work-in-progress)
16. [Files Produced — v2.9](#16-files-produced--v29)
17. [Key Technical Constants](#17-key-technical-constants)
18. [Repository Structure](#18-repository-structure)
19. [Operational Instructions](#19-operational-instructions)
20. [Thesis Narrative Status](#20-thesis-narrative-status)

---

## 1. Project Overview

### What the thesis is
An MSc thesis comparing two water-budgeted rice irrigation controllers on a 130-agent agent-based model (ABM) of a real 6-hectare Hashemi paddy field in Gilan, Iran:
- **MPC** (Model Predictive Control): CasADi + IPOPT optimizer, horizon Hp, receding-horizon replanning each day.
- **SAC-RL** (Soft Actor-Critic): VDN factorized critic architecture, centralised training / decentralised execution (CTDE), 130-dimensional continuous action space.

The ABM simulates physically realistic soil-water dynamics calibrated against NASA MERRA-2 climate data (r=0.74 validation).

### Evaluation structure
Three climate scenarios × three budget levels = 9 evaluation cells per controller:
- **Scenarios:** dry (2022, 40 mm), moderate (2018, 109 mm), wet (2024, 177 mm)
- **Budgets:** 100% / 85% / 70% of reference seasonal need (484 mm)
- **Forecast modes:** perfect (ground truth), noisy (AR(1) σ=0.15, ρ=0.6, seed=42)

### Constraints
- The assistant must NEVER run code unless explicitly asked
- The assistant must get direct approval on file structure before creating files
- All parameters must be backed by scientific research
- Code is broken into separate understandable files
- All results are saved (nothing needs to be rerun)
- Long simulations checkpoint during the run for crash recovery

---

## 2. System Architecture

### ABM
- **130 agents**, each a paddy plot (~46 m²)
- **State per agent:** x1=soil moisture (mm), x2=accumulated GDD, x3=maturation stress, x4=biomass (g/m²), x5=surface ponding (mm)
- **Key thresholds:** WP=80 mm, FC=140 mm, ST=112 mm, saturation≈220 mm

### Observation space
- v2.7 per-agent block (8 features): x1_norm, x5_norm, x4_norm, x3, elev_norm, Nr_norm, Nr_internal_norm, n_upstream_norm → OBS_DIM=1097
- v2.8 per-agent block (9 features): same + x1_overshoot_norm → OBS_DIM=1227
- v2.9 per-agent block: **same as v2.7 (8 features, OBS_DIM=1097)** — no new features
- Global scalar block (9): day_frac, budget_frac, budget_total_norm, burn_rate, rain_today, ETc_today, h2, h7, g_base
- Forecast block (48): 8-day forecast × 6 variables

### Normalisation formulas (v2.8.1 fix — CRITICAL)
All dynamic features in `runner.py` must match `gym_env.py` exactly:
```python
x1_norm  = clip((x1 - WP) / (FC - WP), 0.0, 1.5)   # NOT x1/FC
x5_norm  = clip(x5 / X5_REF,            0.0, 2.0)
x4_norm  = clip(x4 / X4_REF,            0.0, 1.5)
x3       = clip(x3,                      0.0, 2.0)
```
WP and FC in runner are derived from `crop['theta2']*crop['theta5']` and `crop['theta6']*crop['theta5']`. This was wrong in all versions before v2.8.1.

### SAC architecture (VDN factorized)
- **SharedActor:** MLP [per_agent_input_dim → 128 → 128 → 2] applied in parallel to all 130 agents
- **FactorizedContinuousCritic:** Q_total = Σₙ Q_local. Each Q_local: MLP [per_agent_critic_input_dim → 256 → 256 → 1]. Twin-Q, clipped double-Q.
- **v2.7 dims:** actor input=65, critic input=66. **v2.8 dims:** actor=66, critic=67.
- **v2.9 dims:** same as v2.7 (65/66). Uses `V27CTDESACPolicy`.

---

## 3. Controller 1: MPC

### Key results (perfect forecast, Hp=3)

| Cell | Hp=3 yield | Water used | Waterlog days |
|---|---:|---:|---:|
| dry/100% | 4122 | 463 mm | 0.0 |
| dry/85% | 4068 | 411 mm | 0.0 |
| dry/70% | 3771 | 339 mm | 0.0 |
| mod/100% | 3698 | 398 mm | 6.5 |
| mod/85% | 3694 | 398 mm | 6.7 |
| mod/70% | 3598 | 339 mm | 0.3 |
| wet/100% | 3717 | **299 mm** | 20.2 |
| wet/85% | 3722 | **300 mm** | 20.1 |
| wet/70% | 3727 | **300 mm** | 20.0 |

**Key MPC insight:** In wet/100%, MPC uses only 299 mm (62% of the 484 mm budget) — it stops irrigating because x1 is already high from rainfall. This is enabled by MPC's explicit look-ahead optimiser, not by a trained policy. SAC cannot replicate this without either a specific training signal or an OOD-robust wet-year dev signal.

**Noisy forecast:** Mean degradation 0.17%, worst case 0.46%. Extremely robust.

---

## 4. Controller 2: SAC-RL Evolution History

### v2.6: Bug-fix and tuning release
- γ-slot bug: 5th per-agent feature was `x2/theta18` (uniform GDD) instead of `elev_norm`. Caused spatial blindness.
- 500k steps, seed 0 only. Best model at step 350k.
- v2.6 results in: `results/legacy/runs_legacy_sac_vdn_2.6/`

### v2.7: Core architecture improvement
Four changes vs v2.6:
1. Fixed γ bug → `elev_norm` restored
2. Added 3 topographic features (Nr_norm, Nr_internal_norm, n_upstream_norm) → per-agent block 5→8
3. No early termination on budget exhaustion (full 93-day episodes always)
4. Removed rb and r5 from reward → r = r1 + r2 + r3 + r6

SAC hyperparameters unchanged: ent_coef=0.05 fixed, LR 3e-4→5e-5, γ=0.99, τ=0.005, batch=256.

### v2.8: Bundled experiment — FAILED
Two changes vs v2.7:
1. x1_overshoot_norm added as 9th per-agent feature
2. 50k-step episode-length curriculum (60 days → 93 days)

**Failed.** Seed 0 best_model was captured at step 25k (inside curriculum warmup). The 60-day-optimised policy never recovered after the curriculum lifted. All metrics regressed vs v2.7. See §11 for full analysis.

### v2.9: Proposal B — CURRENT
One change vs v2.7: `AdaptiveLRCallback` added.
Architecture: v2.7 (1097-dim obs, 8 features/agent, V27CTDESACPolicy).
No curriculum, no x1_overshoot feature. See §15.

---

## 5. v2.6 Results — Starting Baseline

| Cell | SAC_v26 | MPC_Hp3 | vs MPC |
|---|---:|---:|---:|
| dry/100% | 4090 | 4122 | −0.8% |
| dry/70% | 3691 | 3771 | −2.1% |
| wet/100% | **3169** | 3717 | **−14.8%** |
| wet/70% | 3360 | 3727 | **−9.8%** |

### v2.6 Pathology Diagnostics
- `corr(u, rain)` wet/100% = +0.378 (irrigated MORE when raining)
- `corr(u, x1)` wet/100% = +0.03 (no soil-moisture response)
- CV of daily irrigation = 0.06 (near-constant trickle)
- Waterlog days/agent = 90.4 (vs MPC's 20.2)

---

## 6. v2.6 Wet-Year Pathology Analysis

Three structural root causes:
1. **γ bug:** Spatial blindness — all 130 agents received identical context
2. **Budget illiteracy:** Early termination hid late-season drought consequences
3. **Easy-escape policy:** Constant 5 mm/day trickle satisfied drought term; forecast-based control handled 95% of overshoot

---

## 7. v2.7 Design Decisions and Rationale

| Change | Rationale |
|---|---|
| Fix γ bug → elev_norm | Chapter 4 compliance; spatial context required |
| Add Nr_norm, Nr_internal_norm, n_upstream_norm | Close information asymmetry: ABM uses these for routing; actor needs them too |
| Remove early termination | Agent must experience late-season drought from overspending |
| Remove rb, r5 | rb never bound on converged policy; r5 was already 0 |

---

## 8. v2.7 Results — Corrected (v2.8.1 runner fix)

### The runner bug (v2.8.1)
`runner.py` used `x1_norm = x1/FC` since its first commit. `gym_env.py` uses `x1_norm = clip((x1-WP)/(FC-WP), 0, 1.5)`. At x1=WP (80mm): runner produced 0.57, env produced 0.0. They only agreed at x1=FC.

**Impact:** ≤10 kg/ha (≤0.3%) per cell. The policy was robust to this mismatch — it was not conditioning strongly on x1_norm in the first place, which is why the mismatch had such small numerical impact. The bug was real but did not drive the wet-year gap.

**Fix:** `runner.py` v2.8.1 uses correct formula; `reset()` now stores `self._wp_total`. Regression test added: `test_runner_obs_matches_env_obs` in `tests/test_rl_smoke.py`.

### Corrected v2.7 yields (perfect forecast)

| Cell | SAC v2.7 s0 | SAC v2.7 s1 | 2-seed mean | MPC Hp=3 | vs MPC |
|---|---:|---:|---:|---:|---:|
| dry/100% | 4163 | 3816 | 3990 | 4122 | −3.2% |
| dry/85% | 4101 | 3847 | 3974 | 4068 | −2.3% |
| dry/70% | 3766 | 3712 | 3739 | 3771 | −0.9% |
| mod/100% | 3730 | 3527 | 3629 | 3698 | −1.9% |
| mod/85% | 3737 | 3579 | 3658 | 3694 | −1.0% |
| mod/70% | 3589 | 3565 | 3577 | 3598 | −0.6% |
| **wet/100%** | **3434** | **3376** | **3405** | **3717** | **−8.4%** |
| wet/85% | 3432 | 3382 | 3407 | 3722 | −8.5% |
| wet/70% | 3492 | 3485 | 3489 | 3727 | −6.4% |

**Seed-to-seed variance:** mean |Δ%| = 3.4%, max = 8.7% (dry/100%). With N=2 the 95% CI on the mean is roughly ±4.5× the half-difference — dry/moderate cell comparisons vs MPC are not individually significant at the level of individual cells, but the pattern is consistent across all 6 cells.

### Closed-loop diagnostics (corrected, pooled N=12090)

| Metric | SAC v2.6 | SAC v2.7 s0 | SAC v2.7 s1 | MPC Hp=3 |
|---|---:|---:|---:|---:|
| corr(u, rain) wet/100% | +0.378 | **−0.489** | −0.376 | −0.309 |
| corr(u, x1) pooled wet/100% | +0.03 | +0.048 | +0.470 | **−0.540** |
| waterlog days wet/100% | 90.4 | 76.4 | 66.8 | 20.2 |

Rain-response is solved. x1-conditioning is NOT solved. Spatial differentiation (elev corr) is solved in seed 0 but not well-captured by corr(u,elev) because absolute spatial std is tiny for all controllers (~0.002-0.014 mm/day range).

---

## 9. v2.7 Noisy-Forecast Results

AR(1) noise: σ=0.15, ρ=0.6, noise_seed=42. Same seed used for both MPC and SAC.

| Controller | Mean yield degradation | Worst cell |
|---|---:|---:|
| MPC Hp=3 | +0.17% | +0.46% (wet/70%) |
| SAC v2.7 seed 0 | +0.66% | +1.50% (mod/85%) |
| SAC v2.7 seed 1 | −0.39% (noise) | +0.60% (wet/100%) |

Both controllers are robust. Max SAC degradation (1.5%) is within inter-seed variance (3.4%). The "improvement" for seed 1 under noise is sampling noise, not a real effect.

---

## 10. Critical Analysis of Stability Proposals

### Root cause of the explosion
The v2.7/v2.8 critic explosion (begins step ~165–195k) follows the SAC deadly triad:
1. **Overestimated Q** → actor exploits inflated values
2. **Actor commits to those actions** → critic receives large-variance bootstrapped targets
3. **Fixed ent_coef=0.05** means entropy brake does not scale with growing |Q| → feedback loop ignites

Observable signature: `|actor_loss|` grows monotonically (−185 at step 1k → −499 at step 185k → −8925 at step 250k). Once |actor_loss| ≈ 500, the cascade becomes exponential. critic_loss goes from 10 at step 185k to 6×10⁵ by step 250k.

### Rejected proposals
| Proposal | Rejection reason |
|---|---|
| Lower max_grad_norm (1.0→0.5) | Delay tactic; doesn't address overestimation |
| Lower LR_START (3e-4→1.5e-4) | Shifts peak later, burns 2× compute |
| Lower tau (0.005→0.001) | 5× slower convergence, requires thesis justification |
| Huber loss | Implementation risk in SB3; masks real value function problem |

### Proposal B (AdaptiveLRCallback) — CHOSEN for v2.9
Monitors rolling-1000-step critic_loss mean. If > 50: multiply LR by 0.3 (floor 1e-5). If < 5 after reduction: restore to scheduled value. Cooldown: 5000 steps.

**Why this addresses the cause:** At step ~185k (rolling critic_loss first exceeds 50), slowing LR by 0.7× reduces the critic's update velocity in response to the bad bootstrap target. The replay buffer then dilutes the pathological transitions before they can compound.

**Why it wasn't chosen before:** The handoff v2 deprioritised it because "EvalCallback captures the best policy at step 200k already." v2.8's failure changed this: the best_model was captured at step 25k (inside the curriculum warmup), not at 200k. With the curriculum removed (v2.9 = v2.7 base), the best_model should again peak near step 200k, and Proposal B now offers a real chance to extend that window.

### Proposal C (curriculum) — TRIED in v2.8, FAILED
The 50k warmup on 60-day episodes biased the policy toward an unconditional spending strategy. Best_model was selected at step 25k (mid-warmup). Post-warmup recovery never happened. All metrics regressed vs v2.7. See §11.

---

## 11. v2.8 Design, Execution and Failure Analysis

### What was implemented
1. `x1_overshoot_norm = clip(max(x1-FC,0)/FC, 0, 1)` added as 9th per-agent feature
2. Episode-length curriculum: 60-day episodes for first 50k steps, then full 93 days

### Training results (seed 0, 250k steps)

| Step | critic_loss | |actor_loss| | Eval reward |
|---:|---:|---:|---:|
| 25k | 0.56 | 319 | **−2.23 (PEAK / best_model)** |
| 75k | 0.20 | 373 | — |
| 125k | 0.25 | 452 | −2.36 |
| 175k | 6.01 | 485 | −3.28 |
| 195k | **196** | 532 | — |
| 250k | **636,000** | 8925 | −30.57 |

Best_model captured at **step 25k** — inside the 50k warmup window. v2.7's best_model was at step 200k.

### Failure diagnosis
The curriculum truncated episodes at 60 days for steps 0–50k. During this window, the policy converged to a 60-day-optimal strategy: spend 484÷93≈5.2 mm/day uniformly (optimal for the truncated problem). After step 50k the curriculum lifted, but the actor was stuck in this basin. It never unlearned the unconditional-spending strategy.

**Evidence:**
- Eval reward peaks at step 25k (inside warmup), then degrades monotonically
- v2.8 best_model uses full 484 mm budget in wet/100% (vs MPC's 299 mm)
- corr(u, rain) = +0.05 in wet/100% (was −0.49 in v2.7; v2.8 **lost rain-response**)
- corr(u, x1) pooled = +0.23 in wet/100% (worse than v2.7's +0.05)
- Waterlog days = 92.5 in wet/100% (vs v2.7's 76.4; worst ever)
- Yield in wet/100% = 2909 kg/ha (−21.7% vs MPC, vs v2.7's −7.6%)

The curriculum delayed the explosion by ~30k steps (v2.8 explodes at step ~195k vs v2.7's ~165k) but did not prevent it and actively damaged policy quality.

### v2.8 yields (seed 0 best_model) vs v2.7 and MPC

| Cell | v2.8 s0 | v2.7 s0 | MPC | v2.8 vs MPC | v2.8 vs v2.7 |
|---|---:|---:|---:|---:|---:|
| dry/100% | 4083 | 4163 | 4122 | −0.9% | −80 |
| dry/85% | 3942 | 4101 | 4068 | −3.1% | −159 |
| dry/70% | 3572 | 3766 | 3771 | −5.3% | −194 |
| mod/100% | 3707 | 3730 | 3698 | +0.2% | −23 |
| mod/85% | 3694 | 3737 | 3694 | 0.0% | −43 |
| mod/70% | 3365 | 3589 | 3598 | −6.5% | −224 |
| **wet/100%** | **2909** | **3434** | **3717** | **−21.7%** | **−525** |
| wet/85% | 3043 | 3432 | 3722 | −18.2% | −389 |
| wet/70% | 3197 | 3492 | 3727 | −14.2% | −295 |

v2.8 is worse than v2.7 in every cell. In wet cells the deficit is catastrophic.

---

## 12. Code Fixes Applied (v2.8.1)

### runner.py v2.8.1
**Bug:** `_build_obs()` used `x1_norm = state['x1'] / fc` (old formula); no clip on x5/x4/x3.  
**Fix:** All dynamic feature formulas now match `gym_env.py` exactly.  
**Impact on results:** ≤10 kg/ha (≤0.3%) per cell. Bug was real but policy was robust to it.  
**New in reset():** `self._wp_total = crop['theta2'] * crop['theta5']`, `self._x1_range = FC - WP`.

### tests/test_rl_smoke.py v2.8.1
Added `test_runner_obs_matches_env_obs`: runs 30 env steps, recomputes runner dynamic features from ABM state, asserts bit-identity with env obs. Regression guard against any future re-introduction of the mismatch.

### Notebook Cell 6 (both v2.7 and v2.8 notebooks)
Fixed two problems:
1. **Scenario:** forced wet/100% (year=2024, budget=484mm) instead of default dry/100%
2. **Correlation:** changed from field-mean time-series (N=93) to **pooled (day×agent)** with N=12,090. The N=93 version had insufficient statistical power.

### scripts/analysis/compute_correlations.py (new)
Standalone CLI that reads saved parquet files, computes pooled correlations for all 9 cells, and writes a CSV. No model re-run required.

---

## 13. Full Comparative Analysis (all controllers, all cells)

### Yields — perfect forecast

|  | dry/100 | dry/85 | dry/70 | mod/100 | mod/85 | mod/70 | wet/100 | wet/85 | wet/70 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPC Hp=3 | 4122 | 4068 | 3771 | 3698 | 3694 | 3598 | 3717 | 3722 | 3727 |
| SAC v2.7 s0 | 4163 | 4101 | 3766 | 3730 | 3737 | 3589 | 3434 | 3432 | 3492 |
| SAC v2.7 s1 | 3816 | 3847 | 3712 | 3527 | 3579 | 3565 | 3376 | 3382 | 3485 |
| SAC v2.8 s0 | 4083 | 3942 | 3572 | 3707 | 3694 | 3365 | 2909 | 3043 | 3197 |
| Fixed schedule | 3607 | 3619 | 3439 | 3302 | 3309 | 3184 | 2790 | 3144 | 3428 |

### Δ vs MPC Hp=3 (perfect forecast)

| Controller | dry mean | mod mean | wet mean |
|---|---:|---:|---:|
| SAC v2.7 s0 | +0.6% | +0.6% | −7.2% |
| SAC v2.7 s1 | −5.5% | −3.5% | −7.4% |
| SAC v2.7 2-seed mean | −2.3% | −1.2% | −7.3% |
| SAC v2.8 s0 | −3.1% | −2.1% | **−18.0%** |

### Pooled corr(u, x1) — perfect forecast

|  | dry/100 | mod/100 | wet/100 |
|---|---:|---:|---:|
| MPC Hp=3 | −0.12 | −0.44 | **−0.54** |
| SAC v2.7 s0 | +0.18 | −0.02 | +0.05 |
| SAC v2.7 s1 | +0.27 | +0.02 | +0.47 |
| SAC v2.8 s0 | +0.68 | −0.37 | +0.23 |

### Waterlog days/agent — perfect forecast

|  | dry/100 | mod/100 | wet/100 |
|---|---:|---:|---:|
| MPC Hp=3 | 0.0 | 6.5 | **20.2** |
| SAC v2.7 s0 | 14.5 | 42.4 | 76.4 |
| SAC v2.7 s1 | 31.9 | 48.9 | 66.8 |
| SAC v2.8 s0 | 70.9 | 53.3 | **92.5** |

### Noisy forecast degradation (mean % yield loss)

| Controller | Mean | Worst |
|---|---:|---:|
| MPC Hp=3 | +0.17% | +0.46% |
| SAC v2.7 s0 | +0.66% | +1.50% |
| SAC v2.7 s1 | −0.39% | +0.60% |
| SAC v2.8 s0 | **+0.18%** | +0.30% |

v2.8's near-zero noise sensitivity is a symptom of its unconditional spending policy (not conditioning on dynamic state → also not perturbed by noisy dynamic state). It is not a positive finding.

---

## 14. New Findings from v3 Chat Session

### Finding 1: The wet test year (2024) is severely out-of-distribution

The training set contains 20 years. Their rainfall distribution:
- Min: 14.5 mm, P25: 30.5 mm, Median: 44.6 mm, P90: 68.0 mm, **Max: 82.1 mm**
- Dev set (used for best_model selection): max 88.4 mm (2023)
- Test "moderate" (2018): **109 mm — already 33% above training max**
- Test "wet" (2024): **177 mm — 115% above training max**

The wet test year is more than twice as rainy as the wettest year the policy was ever trained on. The dev set (which drives best_model selection) has a max of 88 mm — less than half of 2024's 177 mm.

**Implication for interpreting results:**
- The wet-year gap (−7% for v2.7, −22% for v2.8) is primarily an **out-of-distribution generalisation gap**, not a policy architecture failure.
- MPC closes this gap because it plans against the actual weather at evaluation time with no learning component.
- A policy trained or evaluated on more wet years would likely perform better. This is a real research limitation, not a fixable code problem.
- In the thesis: frame the wet-year gap as "generalisation to an OOD climate extreme (P100+115% year)" rather than "policy weakness."

### Finding 2: The runner bug had near-zero numerical impact

Despite being a real formula mismatch (training vs inference normalisation), correcting the runner changed yields by ≤10 kg/ha (≤0.3%) per cell. The policy was not conditioning strongly on x1_norm to begin with (as confirmed by corr(u, x1) ≈ +0.05), so feeding a differently-scaled value of that feature at inference made little difference.

The bug should still be fixed and kept fixed (see v2.8.1 changes in §12), and the corrected numbers should be reported. But the wet-year gap cannot be explained by the bug.

### Finding 3: v2.7 spatial differentiation is negligible in absolute terms

The spatial std of season-mean irrigation across agents:
- MPC Hp=3: 0.014–0.081 mm/day depending on cell
- SAC v2.7 s0: 0.002–0.004 mm/day
- SAC v2.7 s1: 0.007–0.011 mm/day

Everyone has very small absolute spatial differentiation. The handoff v2's claim that "corr(u, elev) fixed from +0.05 to +0.96" is true but misleading — the absolute variation is so small that the correlation of a near-constant vector with elevation is not agronomically meaningful. In the thesis, report corr(u, elev) for completeness but note that absolute spatial std is 1–2 orders of magnitude smaller than MPC's.

### Finding 4: The v2.8 curriculum failure mode is well-understood

The 50k warmup caused the policy to peak mid-warmup (step 25k) and never recover. The correct analogy: a student who studied only the first half of the course material and is then suddenly given the full exam. They "specialise" in the short problem and their partial-problem knowledge doesn't transfer.

This does not mean curriculum learning is wrong in principle — a shorter warmup (10–15k steps, covering ~20 full short episodes) would leave far more of the 250k budget for the full-problem phase. But since v2.8 bundled the curriculum with the x1_overshoot feature, we can't isolate whether the feature was useful without the curriculum damage. v2.9 tests only Proposal B to keep variables separated.

### Finding 5: Proposal B is the highest-value remaining intervention

Given:
- The explosion mechanism is understood (|actor_loss| growth → bootstrap cascade at step ~185k)
- v2.7's best_model is correctly identified at step ~200k (just before explosion)
- The explosion degrades the policy past step 200k but the best_model is already captured

**Proposal B (AdaptiveLRCallback)** is the only remaining intervention that could:
1. Keep critic_loss bounded after step 185k
2. Allow the policy to continue improving past step 200k
3. Potentially improve the best_model step and the best_model yield
4. Confirm or deny that a stable post-200k phase yields a better policy

If Proposal B works (critic stays bounded), it answers: "is the step-200k peak a true plateau or an explosion-truncated optimum?" If it doesn't work (explosion shifts slightly later), the conclusion is "v2.7 at step 200k is the true policy optimum."

---

## 15. v2.9 Design — Current Work-in-Progress

### Single change vs v2.7
**Added:** `AdaptiveLRCallback` in `train.py`.  
**Nothing else changed:** same obs layout (1097-dim, 8 features), same architecture (V27CTDESACPolicy), no curriculum, no x1_overshoot.

### AdaptiveLRCallback mechanics

```python
# Fires when rolling-1000-step mean critic_loss > SPIKE_THRESHOLD (50)
new_lr = max(current_lr * LR_REDUCTION_FACTOR (0.3), LR_FLOOR (1e-5))

# Recovers when rolling mean < RECOVERY_THRESHOLD (5) and LR was reduced
restored_lr = lr_schedule(current_progress)

# Rate-limited: minimum COOLDOWN_STEPS (5000) between reductions
```

The callback logs every event to TensorBoard/WandB as `adaptive_lr/event`, `adaptive_lr/lr_after_reduce`, `adaptive_lr/lr_after_restore`, `adaptive_lr/rolling_critic_verbose`.

### Falsifiable predictions

| Prediction | Success looks like | Failure looks like |
|---|---|---|
| Explosion suppressed | critic_loss stays < 200 throughout 250k; 0–3 reduction events visible in `adaptive_lr/event` | Same explosion at ~165–195k, just slightly later |
| Best_model step shifts later | `eval/mean_reward` peaks at step ≥ 225k | Still peaks at ~200k |
| Policy quality improves | Yield ≥ v2.7 s0 in ≥5 of 9 cells | No improvement vs v2.7 |

If all three succeed: run seed 1 for paired comparison, then decide on more seeds.  
If explosion suppressed but quality unchanged: confirms explosion was after the true policy peak.  
If Proposal B doesn't suppress explosion: consider auto-tuned ent_coef as v2.10.

### Compute estimate
- 250k steps on A100: ~28–35 min
- 12 Colab compute units available
- One v2.9 seed uses ~1 compute unit; budget for 10–12 more seeds total
- **Priority:** v2.9 seed 0, then v2.9 seed 1, then v2.7 seeds 2–4 if Proposal B shows improvement

---

## 16. Files Produced — v2.9

All files are in `/mnt/user-data/outputs/` (v2.8.1 fixes folder + v2.9 files).

### Files to commit (v2.9 training)

| File | Destination | What changed |
|---|---|---|
| `src/rl/train.py` | `src/rl/train.py` | Bumped to v2.9.0; added `AdaptiveLRCallback`; uses `V27CTDESACPolicy`; removed curriculum kwargs |
| `notebooks/colab_sac_v29.ipynb` | `notebooks/colab_sac_v29.ipynb` | New notebook for v2.9 training |

### Files already committed (v2.8.1 fixes — must be in repo before v2.9 training)

| File | What changed |
|---|---|
| `src/rl/runner.py` | v2.8.1: corrected x1_norm and other dynamic feature formulas |
| `tests/test_rl_smoke.py` | v2.8.1: added `test_runner_obs_matches_env_obs` |
| `notebooks/colab_sac_v27.ipynb` | v2.8.1: Cell 6 fixed (scenario + pooled corr) |
| `notebooks/colab_sac_v28.ipynb` | v2.8.1: Cell 6 fixed |
| `scripts/analysis/compute_correlations.py` | New: CLI for pooled correlation analysis on saved parquets |

### Pre-flight checklist for v2.9
```bash
# Must pass before training
python -m pytest tests/test_rl_smoke.py -v           # 18 tests expected
python -m pytest tests/test_factorized_critic.py -v   # 9 tests expected

# Sanity: obs_dim should be 1097 (v2.7 arch)
python -c "
from src.rl.gym_env import IrrigationEnv
e = IrrigationEnv(randomize=False)
o, _ = e.reset()
assert o.shape[0] == 1097, f'Got {o.shape[0]}'
print(f'obs_dim = {o.shape[0]} OK')
"
```

### WandB run name format: `sac_v29_seed{N}`

---

## 17. Key Technical Constants

### Field and crop
- N_AGENTS = 130, Season = 93 days
- WP = 80 mm (theta2 × theta5), FC = 140 mm (theta6 × theta5), ST = 112 mm
- FULL_SEASON_NEED_MM = 484 mm, UB_MM = 12 mm/day

### Reward weights (unchanged v2.7–v2.9)
- ALPHA1 = 1.0 (biomass increment), ALPHA2 = 0.016 (water cost)
- ALPHA3 = 0.1 (drought stress), ALPHA6 = 8.0 (FC-overshoot penalty)

### SAC hyperparameters (unchanged v2.7–v2.9)
- ent_coef = 0.05 (fixed), max_grad_norm = 1.0
- LR_START = 3e-4, LR_END = 5e-5 (linear decay)
- gamma = 0.99, tau = 0.005, batch_size = 256
- learning_starts = 1000, gradient_steps = 1
- Actor hidden: [128, 128], Critic hidden: [256, 256]
- TOTAL_TIMESTEPS = 250,000, EVAL_FREQ = 25,000, CHECKPOINT_FREQ = 50,000

### v2.9-specific
- AdaptiveLR spike threshold: 50 (rolling-1000-step mean)
- AdaptiveLR reduction factor: 0.3, floor: 1e-5
- AdaptiveLR recovery threshold: 5
- AdaptiveLR cooldown: 5000 steps

### Climate data split
- **Training years (20):** 2000–2025 excluding 2018, 2022, 2024 (test) and 2002, 2016, 2023 (dev)
- **Dev years (3):** 2002 (27 mm), 2016 (77 mm), 2023 (88 mm) — max 88 mm
- **Test years (3):** 2022 dry (40 mm, P40 of train), 2018 moderate (109 mm, P100+33%), 2024 wet (177 mm, P100+115%)
- **Training max rainfall:** 82 mm (2013)
- **2024 vs training max:** 115% above → severe OOD

---

## 18. Repository Structure

```
thesis/
├── src/
│   ├── rl/
│   │   ├── gym_env.py       ← v2.8 (x1_overshoot_norm + curriculum; unchanged for v2.9)
│   │   ├── networks.py      ← v2.8 (V27CTDESACPolicy + CTDESACPolicy + legacy)
│   │   ├── runner.py        ← v2.8.1 (FIXED: x1_norm formula, all dynamic features)
│   │   └── train.py         ← v2.9 (AdaptiveLRCallback; V27CTDESACPolicy; no curriculum)
│   ├── mpc/                 ← unchanged
│   ├── terrain.py, precompute.py, forecast.py ← unchanged
├── tests/
│   ├── test_rl_smoke.py     ← v2.8.1 (18 tests incl. runner obs parity test)
│   └── test_factorized_critic.py ← v2.8 (9 tests)
├── notebooks/
│   ├── colab_sac_v29.ipynb  ← NEW (v2.9 training)
│   ├── colab_sac_v28.ipynb  ← v2.8.1 (Cell 6 fixed)
│   └── colab_sac_v27.ipynb  ← v2.8.1 (Cell 6 fixed)
├── scripts/analysis/
│   └── compute_correlations.py ← NEW (pooled correlation CLI)
├── results/
│   ├── runs/
│   │   ├── sac_perfect_det_*_seed0.{json,parquet}  ← v2.8 best_model (9 cells)
│   │   ├── sac_noisy_ns42_det_*_seed0.{json,parquet} ← v2.8 noisy (9 cells)
│   │   ├── sac v27 run results/                    ← corrected v2.7 seeds 0+1 (72 files)
│   │   ├── mpc_perfect_*_Hp{3,8,14}.{json,parquet} ← MPC (27 cells)
│   │   ├── mpc_noisy_*_Hp3_seed42.{json,parquet}   ← MPC noisy (9 cells)
│   │   ├── fixed_schedule_*.{json,parquet}          ← 9 cells
│   │   └── no_irrigation_*.{json,parquet}           ← 3 cells
│   └── legacy/runs_legacy_sac_vdn_2.6/             ← v2.6 results
```

### Result naming
- SAC perfect: `sac_perfect_det_{scenario}_rice_{budget}pct_seed{N}.{json|parquet}`
- SAC noisy: `sac_noisy_ns42_det_{scenario}_rice_{budget}pct_seed{N}.{json|parquet}`
- v2.9 will write to `results/rl/sac_v29_seed{N}/` (separate from v2.7/v2.8 runs)

---

## 19. Operational Instructions

### Running v2.9 training (Colab)
1. Pull latest repo: includes v2.8.1 runner fix and v2.9 train.py
2. Upload `notebooks/colab_sac_v29.ipynb`
3. Runtime → A100 GPU
4. Add WANDB_API_KEY as Colab Secret
5. Run Cell 1 (mount Drive, clone, install)
6. Run Cell 2 (WandB + GPU check)
7. Run Cell 3 (tests + self-test for AdaptiveLRCallback — must all pass)
8. Run Cell 4 with `SEED = 0`
9. Run Cell 5 immediately after training to copy to Drive

**Critical WandB metrics to monitor:**
- `adaptive_lr/event`: expect 1–3 events around step 160–200k; 0 means callback isn't firing; >5 means oscillation
- `train/critic_loss`: should stay < 200 if Proposal B works; check for any post-intervention recovery
- `eval/mean_reward`: should peak at step ≥ 225k if Proposal B succeeds

### Running post-training evaluation (v2.9)
```bash
# Perfect forecast — all 9 cells
python -m scripts.experiments.exp_rl \
    --mode eval \
    --model results/rl/sac_v29_seed0/best_model/best_model.zip \
    --scenario all --budget all \
    --forecast perfect --force

# Noisy forecast
python -m scripts.experiments.exp_rl \
    --mode eval \
    --model results/rl/sac_v29_seed0/best_model/best_model.zip \
    --scenario all --budget all \
    --forecast noisy --noise-seed 42 --force

# Correlation analysis
python -m scripts.analysis.compute_correlations \
    --model results/rl/sac_v29_seed0/best_model/best_model.zip \
    --scenario all --budget all \
    --out results/correlations_v29_seed0.csv
```

### Interpreting v2.9 results
Compare to v2.7 seed 0 corrected numbers (stored in `results/runs/sac v27 run results/`):
- **Proposal B success:** critic_loss bounded AND wet/100% yield ≥ 3434 AND corr(u,x1) wet/100% < 0
- **Partial success:** critic_loss bounded but yields similar → confirms explosion was occurring after the true policy peak
- **Failure:** same explosion pattern; consider auto-tuned ent_coef for v2.10

### Decision tree after v2.9 seed 0
```
Did Proposal B suppress the explosion (critic_loss < 200 throughout)?
├── YES: Did yields improve meaningfully (≥3500 wet/100%)?
│   ├── YES → Run v2.9 seed 1; if consistent, consider adding x1_overshoot for v2.10
│   └── NO  → Explosion was after policy peak; run 3 more v2.7 seeds to improve N
└── NO: Run v2.10 = v2.7 + auto-tuned ent_coef (ent_coef='auto', target_entropy=-65)
```

---

## 20. Thesis Narrative Status

### Established results (defensible now)
- v2.7 corrected (N=2 seeds) matches MPC within 0.6–3.2% in dry/moderate cells
- Wet-year gap is 6–9% (v2.7) vs MPC — primarily OOD generalisation (test year 115% above training max)
- Rain-response fixed (v2.7 corr(u,rain) = −0.49 vs v2.6's +0.38)
- Spatial differentiation sign fixed (corr(u,elev) = +0.96 seed 0) though absolute std is small
- Both controllers robust to AR(1) forecast noise (≤1.5% degradation)
- v2.8 failure documented: curriculum biased toward short-episode strategy; all metrics regressed
- Obs normalisation bug identified and fixed (≤0.3% numerical impact)

### Suggested Chapter 5 structure (current results)
1. **Baseline controllers** — Fixed schedule vs NoIrr
2. **MPC performance** — Hp=3 as operating point; Hp comparison
3. **SAC architectural evolution** — Monolithic → VDN → v2.6 → v2.7
4. **v2.7 results (N=2 corrected)** — yields, correlations, seed variance
5. **SAC vs MPC comparison** — 2-seed mean ± range vs MPC; wet-year gap and OOD framing
6. **Noise robustness** — perfect vs noisy both controllers
7. **Training stability analysis** — EvalCallback curve, critic explosion mechanism, Proposal B
8. **v2.8 ablation (negative result)** — curriculum failure mode, what it teaches
9. **Discussion / future work** — OOD wet years, Proposal B (v2.9), auto-tuned entropy

### Things still to do
- [ ] Run v2.9 seed 0; evaluate all 9 cells perfect + noisy
- [ ] Decide on more seeds based on v2.9 seed 0 results
- [ ] Update Chapter 4 obs layout section (note v2.7=8 feat, v2.8=9 feat, v2.9=8 feat)
- [ ] Fill Chapter 5 tables from analysis_full.csv (analysis outputs in /mnt/user-data/outputs/)
- [ ] Frame wet-year gap as OOD generalisation in thesis text (not policy failure)
- [ ] Update defense slides (currently use v2.6 numbers)
- [ ] Note v2.8 as negative-result ablation (it is publishable as such)

---

## Appendix A: Mistakes and Corrections

### Mistake 1: Early termination on budget exhaustion (v2.6)
Corrected in v2.7.

### Mistake 2: γ observation bug (v2.5/v2.6)
5th per-agent feature was uniform GDD scalar. Corrected in v2.7.

### Mistake 3: Overestimating rb penalty
Initial analysis suggested rb was "too weak." Correct analysis showed ~30:1 overpowered vs r1. Removed.

### Mistake 4: Delay-tactic stability proposals
Lowering max_grad_norm/LR/tau were critiqued as delay tactics. Replaced with Proposal C/B.

### Mistake 5: "Beats MPC in 4 cells" claim (single-seed artifact)
Corrected after seed 1: two-seed mean trails MPC in all 9 cells.

### Mistake 6: Conflicting seed recommendations
Resolved: use paired-samples design (same seeds for each protocol version).

### Mistake 7: runner.py x1_norm mismatch (v2.8.1)
`x1/FC` used throughout history instead of `(x1-WP)/(FC-WP)`. Numerical impact ≤0.3% but methodology was incorrect. Fixed in v2.8.1.

### Mistake 8: Correlation methodology (v2.8.1)
Notebook Cell 6 computed `corr(field-mean u, field-mean x1)` over 93 days (N=93, insufficient power). Fixed to pooled (day×agent) N=12,090.

### Mistake 9: Wrong scenario in Cell 6 diagnostics
Cell 6 ran dry/100% by default and results were cited as wet/100%. Fixed in v2.8.1.

### Mistake 10: v2.8 curriculum warmup too long (50k)
50k steps = 100 short episodes. Policy converged to 60-day-optimal strategy and never recovered. If curriculum is revisited, warmup should be ≤15k steps (≤30 short episodes).

### Mistake 11: Bundling two changes in v2.8
Curriculum + x1_overshoot feature were tested together. When v2.8 failed, it was impossible to determine which change caused the failure. v2.9 corrects this by testing only Proposal B.

---

## Appendix B: Q&A

**Q: Why does SAC not respond to x1 even though r6 penalty is strong?**  
A: SAC learned forecast-based control (use 8-day rain forecast to predict overshoot) instead of state-based control (use current x1). This handles 95% of overshoot. The remaining 5% (soil already high, no rain expected) was a small gradient signal. x1_overshoot_norm feature was intended to fix this, but the curriculum damage in v2.8 prevented evaluation.

**Q: Is the wet-year gap fixable?**  
A: Partially. The 115%-OOD gap is fundamentally a generalisation problem. Fixing it without retraining on wet years requires the policy to extrapolate from the training distribution. Including a higher-rainfall dev year (110+ mm) in the EvalCallback would give the best_model selection criterion a better signal. Adding an explicit budget-return reward term would close some of MPC's deliberate underspending. These are future work items.

**Q: Why not auto-tune ent_coef?**  
A: v2.5 disabled it because auto-tuning caused oscillation. With fixed ent_coef=0.05, the entropy brake does not scale with growing |Q|, which is part of the explosion mechanism. Auto-tuning is the correct fix for that specific issue. v2.10 should test `ent_coef='auto'` with `target_entropy=-65` (= -N/2 rather than -N, to avoid over-exploration).

**Q: How many compute units does a training run cost?**  
A: ~1 Colab Pro compute unit per 250k steps on A100. 12 units remain = ~10–12 more seeds at 250k steps.

**Q: What is the right N for statistical defensibility?**  
A: N=5 is the community standard. N=2 is what we have for v2.7. With seed variance of 3.4% mean, N=2 gives 95% CI of roughly ±5% on dry/moderate comparisons — enough to show the direction but not significance. N=5 would give ±2%, which is defensible. The 12 remaining compute units can fund N=5 for v2.9 (if seed 0 shows promise).

**Q: Does Proposal B + curriculum make sense as v2.10?**  
A: Only if v2.9 Proposal B alone is not sufficient to fix the explosion AND the curriculum warmup is shortened to ≤15k steps. The lesson from v2.8 is that a 50k warmup is too long; the lesson from v2.9 (pending) is whether adaptive LR is sufficient. If both are needed, use 10k warmup + Proposal B, and accept that you're stacking interventions again.

---

*End of handoff document v3.*  
*Continue in new chat by sharing this file.*
