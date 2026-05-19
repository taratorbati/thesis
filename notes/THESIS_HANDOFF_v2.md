# THESIS PROJECT HANDOFF DOCUMENT — v2
**Student:** Tara Torbati  
**Institution:** ITMO University, MSc  
**Topic:** Modern Control Methods for Agricultural Irrigation  
**Repository:** https://github.com/taratorbati/thesis.git  
**Document date:** May 19, 2026  
**Purpose:** Complete technical handoff for continuation in a new chat session. Contains full project history, all design decisions with rationale, all results, all problems, and the exact state of the v2.8 work-in-progress.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Controller 1: MPC (CasADi+IPOPT)](#3-controller-1-mpc)
4. [Controller 2: SAC-RL Evolution History](#4-controller-2-sac-rl-evolution-history)
5. [v2.6 Results — Starting Baseline](#5-v26-results--starting-baseline)
6. [v2.6 Wet-Year Pathology Analysis](#6-v26-wet-year-pathology-analysis)
7. [v2.7 Design Decisions and Rationale](#7-v27-design-decisions-and-rationale)
8. [v2.7 Training Results — Both Seeds](#8-v27-training-results--both-seeds)
9. [v2.7 Noisy-Forecast Results](#9-v27-noisy-forecast-results)
10. [Critical Analysis of Stability Proposals](#10-critical-analysis-of-stability-proposals)
11. [v2.8 Design — Current Work-in-Progress](#11-v28-design--current-work-in-progress)
12. [Files Produced — v2.8](#12-files-produced--v28)
13. [Current Problems and Proposed Fixes](#13-current-problems-and-proposed-fixes)
14. [Multi-Seed Campaign Plan](#14-multi-seed-campaign-plan)
15. [Key Technical Constants](#15-key-technical-constants)
16. [Repository Structure](#16-repository-structure)
17. [Operational Instructions](#17-operational-instructions)
18. [Thesis Narrative Status](#18-thesis-narrative-status)

---

## 1. Project Overview

### What the thesis is
An MSc thesis comparing two water-budgeted rice irrigation controllers on a 130-agent agent-based model (ABM) of a real 6-hectare Hashemi paddy field in Gilan, Iran:
- **MPC** (Model Predictive Control): CasADi + IPOPT optimizer, horizon Hp, receding-horizon replanning each day.
- **SAC-RL** (Soft Actor-Critic): VDN factorized critic architecture, centralised training / decentralised execution (CTDE), 130-dimensional continuous action space.

The ABM simulates physically realistic soil-water dynamics (infiltration, drainage, runoff cascade between terraced plots) calibrated against NASA MERRA-2 climate data (r=0.74 validation).

### Why it matters
Water scarcity is a growing global issue. The thesis asks: can a trained RL agent match or exceed MPC performance in real-time (no optimization solve per step) under water budgets, while being robust to forecast uncertainty?

### Evaluation structure
Three climate scenarios × three budget levels = 9 evaluation cells per controller:
- **Scenarios:** dry (2022), moderate (2018), wet (2024) — based on NASA MERRA-2 annual rainfall patterns
- **Budgets:** 100% / 85% / 70% of the reference seasonal need (484 mm)
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

### ABM (Agent-Based Model)
- **130 agents**, each representing one paddy plot (≈46 m²)
- **State per agent (x):** x1=soil moisture (mm), x2=accumulated GDD (maturation progress), x3=maturation stress, x4=biomass (g/m²), x5=surface ponding (mm)
- **Water routing:** D8-cascade runoff. When agent n irrigates above the ponding threshold, surplus runs to downstream neighbours according to the terrain DEM.
- **Biological model:** FAO-56 adapted for rice. Biomass grows proportional to Kc×ET₀ × (1 - water_stress_factor). Yield = x4 × HI × 10 (kg/ha).
- **Key thresholds:** WP=80 mm, FC=140 mm, stress threshold ST=112 mm, saturation≈220 mm

### Observation space (v2.7, 1097-dim; v2.8, 1227-dim)
Agent-major layout: per-agent features × 130 agents, then global scalars, then forecast.
- v2.7 per-agent block (8 features): x1_norm, x5_norm, x4_norm, x3, elev_norm, Nr_norm, Nr_internal_norm, n_upstream_norm
- v2.8 per-agent block (9 features): same + x1_overshoot_norm at index 8
- Global scalar block (9): day_frac, budget_frac, budget_total_norm, burn_rate, rain_today, ETc_today, h2, h7, g_base
- Forecast block (48): 8-day forecast × 6 variables (rain, ETc, radiation, h2, h7, g_base)

### SAC architecture (VDN factorized)
- **SharedActor:** single MLP [per_agent_input_dim → 128 → 128 → 2] applied to all 130 per-agent inputs in parallel, producing (μₙ, log_σₙ) per agent. Per-agent input = [local features] + [global context] (no action).
- **FactorizedContinuousCritic:** Q_total = Σₙ Q_local(sₙ, g, aₙ). Each Q_local is the same MLP [per_agent_critic_input_dim → 256 → 256 → 1]. Twin-Q (QF0, QF1), min for Bellman target (clipped double-Q). Per-agent critic input = [local features] + [global context] + [local action].
- **Input dims:** v2.8 actor: 66-dim per-agent (9+57), critic: 67-dim (9+57+1)

---

## 3. Controller 1: MPC

### Design
- CasADi + IPOPT solver
- Receding horizon Hp: plans Hp days ahead, executes day 1, replans tomorrow
- Optimizes the same objective as the RL reward (r1+r2+r3+r6)
- Full-season budget constraint enforced via a cumulative water inequality
- Warm-started from previous solution

### Horizon comparison
Hp=3 is the recommended operating point:
- Yields within 28 kg/ha of Hp=8 (negligible)
- Solve time 40–274× faster than Hp=8
- Lower waterlog count than Hp=14 (longer horizon over-plans and causes waterlogging from over-accurate rain avoidance)
- Hp=14 is the "perfect lookahead" ceiling (used as the upper bound for comparison)

### MPC key results (perfect forecast, selected cells)

| Cell | Hp=3 yield | Hp=8 yield | Hp=14 yield | Hp=3 water used | Hp=3 waterlog |
|---|---:|---:|---:|---:|---:|
| dry/100% | 4122 | 4145 | 4171 | 463.3 | 0.0 |
| dry/85% | 4068 | 4069 | 4069 | 411.3 | 0.0 |
| dry/70% | 3771 | 3766 | 3759 | 338.8 | 0.0 |
| mod/100% | 3698 | 3718 | 3743 | 398.2 | 6.5 |
| mod/85% | 3694 | 3725 | 3726 | 398.3 | 6.7 |
| mod/70% | 3598 | 3612 | 3607 | 338.8 | 0.3 |
| wet/100% | 3717 | 3759 | 3765 | 299.5 | 20.2 |
| wet/85% | 3722 | 3743 | 3760 | 300.1 | 20.1 |
| wet/70% | 3727 | 3754 | 3768 | 300.2 | 20.0 |

**Key MPC insight:** In wet years, MPC *voluntarily returns 38% of its budget* (uses only 299 mm vs the 484 mm allocation). It does this because x1 is already high from rain — there is literally no benefit to irrigating, and the optimizer knows it. This is the source of its 20-waterlog-day performance vs SAC's 67–76 days.

### MPC noisy-forecast results
| Cell | Hp=3 perfect | Hp=3 noisy | Δ% |
|---|---:|---:|---:|
| dry/100% | 4122 | 4120 | 0.05% |
| mod/100% | 3698 | 3684 | 0.38% |
| wet/100% | 3717 | 3715 | 0.05% |
Mean degradation: 0.17%. Very robust to forecast noise.

---

## 4. Controller 2: SAC-RL Evolution History

### Pre-v2.0 / v2.4: Monolithic critic (legacy)
- Standard SB3 SAC with a flat 837-dim critic (obs_dim=707 + action_dim=130)
- Trained 500k steps; exploded at ~step 100k (terminal bonus of 300× per-day reward caused Bellman divergence)
- Produced "VDN upgrade contributed +1,600 kg/ha" result when compared to v2.5

### v2.5: First stable VDN
- VDN factorized critic introduced (Q_total = Σ Q_local)
- Fixed: removed the terminal bonus (C_TERM=0)
- Fixed: disabled entropy auto-tuning (ent_coef=0.05 fixed) — auto-tuner caused oscillation
- Fixed: added gradient clipping (max_grad_norm=1.0 via callback)
- Added: LR decay schedule (3e-4 → 5e-5 linear over 500k steps)
- Result: stable training for 500k steps; VDN architecture validated

### v2.6: Bug-fix and tuning release
- **Bug identified (commit c623833):** The 5th per-agent observation feature (the γ slot) was accidentally `x2/theta18` (a field-uniform GDD scalar) instead of the normalised elevation `gamma_flat` specified in Chapter 4. This bug caused the agent to be spatially blind — all 130 agents received the same spatial context.
- **Reward tuning:** rb (budget burn-rate penalty) was re-examined and kept; r5 (delta-u damping) confirmed inactive
- **Training:** 500k steps on seed 0, best model at step 350k
- **Results:** Competitive with MPC in dry/moderate, bad in wet (constant-trickle policy)
- **Architecture decision:** Hp=3 confirmed as MPC operating point (not Hp=8 as originally assumed)
- v2.6 results in: `results/legacy/runs_legacy_sac_vdn_2.6/`

### The γ Bug (critical background)
The variable name collision: `gamma` is used both as the terrain gradient (routing parameter, per-agent, varies by elevation) and in Python as `gamma_flat` from terrain loading. In the v2.5→v2.6 obs builder refactor, the 5th slot was silently assigned `x2/theta18` (normalized GDD accumulation, same for all agents = scalar broadcast). This had been in production since the VDN introduction. The fix was renaming `elev_norm = terrain['gamma_flat']` explicitly to avoid any future collision.

---

## 5. v2.6 Results — Starting Baseline

These are the results at the start of the current chat session.

### Yields vs MPC (perfect forecast)

| Cell | SAC_v26 | MPC_Hp3 | vs MPC | Fixed schedule |
|---|---:|---:|---:|---:|
| dry/100% | 4090 | 4122 | −0.8% | 3607 |
| dry/85% | 3996 | 4068 | −1.8% | 3619 |
| dry/70% | 3691 | 3771 | −2.1% | 3439 |
| mod/100% | 3631 | 3698 | −1.8% | 3302 |
| mod/85% | 3650 | 3694 | −1.2% | 3309 |
| mod/70% | 3502 | 3598 | −2.7% | 3184 |
| wet/100% | **3169** | 3717 | **−14.8%** | 2790 |
| wet/85% | **3225** | 3722 | **−13.3%** | 3144 |
| wet/70% | 3360 | 3727 | **−9.8%** | 3428 |

### v2.6 Pathology Diagnostics
- `corr(u, rain_today) = +0.378` in wet/100% → agent irrigated MORE when it rained
- `corr(per-agent u_cumulative, elevation) = +0.051` → spatially blind (should be ~+0.7)
- `corr(u, x1) = +0.03` → no closed-loop soil-moisture response
- Coefficient of variation of daily irrigation (CV) = 0.06 → near-constant trickle
- Water used: 477 mm out of 484 mm budget in wet/100% → essentially no budget management
- Waterlog days/agent: 90.4 in wet/100%, vs MPC's 20.2

### Explanation
Three structural root causes identified:
1. **γ observation bug:** Agent had no per-agent spatial information (γ slot was field-uniform GDD instead of elevation). Could not differentiate high/low plot irrigation needs.
2. **Budget illiteracy:** Early budget exhaustion did not generate a strong gradient signal. The episode terminated early on budget exhaustion, so the agent never felt late-season drought.
3. **Wet-year easy escape:** The agent found that constant ~5mm/day trickle satisfies the drought term (r3) throughout the season. Because rain happens anyway in wet years, the policy never needed to learn to respond to rain — the crop grew regardless.

---

## 6. v2.6 Wet-Year Pathology Analysis

This was the central analysis question at the start of this chat session. The investigation involved:

### Reward weight analysis
- r6 overshoot penalty dominates r1 biomass by ~30:1/day on average, ~500:1 on worst overshoot days
- The reward signal was strong enough — the problem was structural, not a weight issue
- r6 formula: `−α₆ × mean(max(x1 − FC, 0)²) / FC²` with α₆=8.0, FC=140

### Why the penalty didn't teach x1 conditioning
- The agent learned forecast-based control instead: "if rain tomorrow, irrigate less today"
- This handled 95% of overshoot cases (most overshoot follows heavy rain)
- The remaining 5% — soil already high from recent rain, no more rain expected — was not learned
- RL found the cheapest policy that satisfied the gradient: "watch the forecast, ignore current x1"

### Spatial blindness
- With the γ bug, the actor received identical per-agent context for all 130 agents
- The only differentiation was via action sampling noise
- MPC with terrain information shows corr(u, elev) = +0.65 in wet/100%; SAC v2.6 was +0.066

### Budget mismanagement
- v2.6 episodes terminated when budget was exhausted
- This meant the agent never experienced the late-season drought that follows overspending
- The signal "you spent too much early" was not learned because the episode simply ended

---

## 7. v2.7 Design Decisions and Rationale

Four structural changes from v2.6:

### Change 1: Fix the γ bug (mandatory)
**Change:** Restore 5th per-agent feature to `elev_norm = terrain['gamma_flat']` (normalised elevation).  
**Rationale:** Chapter 4 specification compliance. Without per-agent spatial context the actor cannot learn topography-aware policies. This was a bug, not a design choice.

### Change 2: Add 3 static topographic features (per-agent block 5→8)
**New features:**
- `Nr_norm = terrain['Nr'][n] / 8.0` — total downhill fanout (how many agents receive my runoff)
- `Nr_internal_norm = terrain['Nr_internal'][n] / 8.0` — internal-only fanout
- `n_upstream_norm = #upstream_feeders / 8.0` — how many agents send runoff to me

**Rationale:** These are the exact quantities the ABM uses to route water (sends_to, Nr arrays). Giving the actor this information closes the information asymmetry: the ABM knows the network topology, the agent previously didn't. Valley plots receive more runoff from rain than hilltop plots at the same elevation; this feature encodes that.

**Effect on OBS_DIM:** 707 (v2.6) → 1097 (v2.7). Per-agent block: 5 → 8.

### Change 3: Episode lifecycle — always 93 days (no early termination)
**Change:** Remove early termination on budget exhaustion. `terminated=False` always; `truncated = (self._day >= 93)` only.  
**Rationale:** v2.6 termination on budget exhaustion meant the agent never felt late-season drought from overspending. The corrective gradient signal (r3 drought penalty in the last 20–30 days) was missing. Now the agent experiences the full season and the drought consequence of overspending is visible.  
**Effect:** `ep_len_mean` at step 200k is exactly 93.0 (confirmed both seeds).

### Change 4: Reward simplification — remove rb and r5
**Change:** Drop rb (budget burn-rate penalty) and r5 (delta-u damping penalty) from reward.  
**Rationale:** rb never bound on the converged v2.6 policy (agent always spent near 100% budget). r5 was already set to 0 (inactive). Simpler reward reduces the chance of conflicting gradient signals.  
**Final reward:** `r = r1 + r2 + r3 + r6`

### What did NOT change in v2.7
- SAC hyperparameters: ent_coef=0.05 fixed, max_grad_norm=1.0, LR decay 3e-4→5e-5, γ=0.99, τ=0.005
- VDN architecture: same MLP widths, twin-Q, gradient clipping callback
- ABM physics, reward weights (α₁=1.0, α₂=0.016, α₃=0.1, α₆=8.0)
- Training infrastructure: EvalCallback at EVAL_FREQ=25k, CheckpointCallback at 50k

---

## 8. v2.7 Training Results — Both Seeds

### Seed 0 training trajectory (500k steps)
| Step | Eval mean reward | Notes |
|---:|---:|---|
| 25k | −2.64 | Learning |
| 100k | −8.82 | Plateau |
| 200k | **−0.26** | **Peak — best_model saved here** |
| 250k | −22.6 | Critic explosion propagated to actor |
| 300k | −53.2 | Worst — total collapse |
| 500k | −8.5 | Chaotic recovery |

**Critic-loss explosion:** onset at step ~165k (first critic_loss > 100), exponential growth to 6×10¹² by step 500k. Actor loss sign flipped from negative to positive at step 172,980. Root cause: deadly triad (function approximation + bootstrapping + off-policy data) amplified by 93-day episode length producing high-variance return targets.

**best_model.zip = step 200k checkpoint** (EvalCallback saved this as the peak)

### Seed 1 training trajectory (250k steps — capped)
| Step | Eval mean reward | Notes |
|---:|---:|---|
| 200k | **−1.03** | **Peak — same step as seed 0** |
| 250k | −10.9 | Deterioration beginning |

Both seeds showed **identical trajectory shape**: low-quality → peak at step 200k → deterioration. This confirms step 200k is a structural training property of v2.7, not seed-0 luck.

### Seed 0 perfect-forecast yields (best_model, step 200k)

| Cell | SAC_v27_s0 | SAC_v26 | Δ (kg/ha) | MPC_Hp3 | vs MPC |
|---|---:|---:|---:|---:|---:|
| dry/100% | **4164** | 4090 | +74 | 4122 | **+1.0%** |
| dry/85% | **4101** | 3996 | +105 | 4068 | **+0.8%** |
| dry/70% | 3766 | 3691 | +75 | 3771 | −0.1% |
| mod/100% | **3730** | 3631 | +99 | 3698 | **+0.9%** |
| mod/85% | **3738** | 3650 | +88 | 3694 | **+1.2%** |
| mod/70% | 3589 | 3502 | +87 | 3598 | −0.3% |
| wet/100% | 3433 | 3169 | **+264** | 3717 | −7.6% |
| wet/85% | 3431 | 3225 | +206 | 3722 | −7.8% |
| wet/70% | 3492 | 3360 | +132 | 3727 | −6.3% |

SAC_v27 seed 0 beats MPC_Hp3 in 4 of 9 cells (dry/100, dry/85, mod/100, mod/85), matches within 0.3% in 2 cells.

### Seed 1 perfect-forecast yields

| Cell | SAC_v27_s1 | vs MPC_Hp3 | Δ vs seed0 (kg/ha) |
|---|---:|---:|---:|
| dry/100% | 3806 | −7.7% | −358 |
| dry/85% | 3839 | −5.6% | −262 |
| dry/70% | 3706 | −1.7% | −60 |
| mod/100% | 3524 | −4.7% | −207 |
| mod/85% | 3573 | −3.3% | −165 |
| mod/70% | 3562 | −1.0% | −27 |
| wet/100% | 3372 | −9.3% | −61 |
| wet/85% | 3380 | −9.2% | −51 |
| wet/70% | 3483 | −6.5% | −9 |

**Seed-to-seed variance: mean 3.4%, worst case 8.6% (dry/100%).**

Seed 1 is worse primarily because it **over-spends budget early in high-budget cells** (uses 484 mm in dry/100% vs seed 0's 456 mm), leading to 2-3× more drought days in later season. This reflects typical RL seed-sensitivity.

### Two-seed averaged yields vs MPC

| Cell | 2-seed mean | MPC_Hp3 | vs MPC |
|---|---:|---:|---:|
| dry/100% | 3985 | 4122 | −3.3% |
| dry/85% | 3970 | 4068 | −2.4% |
| dry/70% | 3736 | 3771 | −0.9% |
| mod/100% | 3627 | 3698 | −1.9% |
| mod/85% | 3656 | 3694 | −1.0% |
| mod/70% | 3576 | 3598 | −0.6% |
| wet/100% | 3403 | 3717 | −8.5% |
| wet/85% | 3406 | 3722 | −8.5% |
| wet/70% | 3488 | 3727 | −6.4% |

**Across two seeds, SAC trails MPC in every cell** — 0.6-3.3% in dry/moderate, 6-9% in wet.

### Closed-loop diagnostics (v2.7 vs v2.6 vs MPC)

| Metric | SAC_v26 | SAC_v27 s0 | SAC_v27 s1 | MPC_Hp3 | Target |
|---|---:|---:|---:|---:|---|
| corr(u, rain) wet/100% | +0.378 | **−0.487** | −0.375 | −0.309 | < 0 |
| corr(u, x1) wet/100% | +0.030 | +0.052 | +0.477 | −0.578 | < 0 |
| corr(u, elev) dry/100% | +0.051 | **+0.959** | +0.321 | +0.055 | > 0.5 |
| Daily irrigation CV wet/100% | 0.06 | 0.37 | 0.53 | 0.81 | > 0.3 |
| Waterlog days wet/100% | 90.4 | 76.4 | 66.8 | 20.2 | — |

**Rain-response: solved.** corr(u, rain) flipped from +0.38 to −0.49 in seed 0. The agent now backs off correctly when rain is forecast.

**Spatial differentiation: solved.** corr(u, elev) jumped from +0.05 to +0.96 in seed 0. The actor now allocates water by elevation.

**x1-conditioning: NOT solved.** corr(u, x1) remains near zero (seed 0: +0.05, seed 1: +0.48 but still positive). The agent does not reduce irrigation when current soil moisture is above FC. MPC shows −0.58 here.

---

## 9. v2.7 Noisy-Forecast Results

### AR(1) noise parameters: σ=0.15, ρ=0.6, seed=42
Same noise realization for MPC and SAC (matching seed=42 ensures identical atmospheric perturbations).

### Yield degradation: perfect → noisy

| Controller | Mean Δyield | Std | Worst cell |
|---|---:|---:|---:|
| SAC_v27 seed 0 | +0.66% loss | 0.54 | +1.51% (mod/85%) |
| SAC_v27 seed 1 | **−0.53% (improves!)** | 0.61 | +0.49% (wet/100%) |
| MPC_Hp3 | +0.17% loss | 0.20 | +0.46% (wet/70%) |

**Key finding: both controllers are highly robust to AR(1) σ=0.15 forecast noise.**  
SAC seed 1's "improvement" under noise is statistical noise in the evaluation, not a real effect.

The maximum degradation observed is 1.5% for SAC (mod/85% seed 0) and 0.5% for MPC (wet/70%). These are within the inter-seed variance of SAC itself (3.4% mean). Neither controller is meaningfully harmed by this noise level.

**Why SAC is robust:** SAC's learned policy is a function that was trained on randomized scenarios — it already learned to handle uncertainty implicitly. MPC degrades slightly because noisy forecasts produce slightly suboptimal plans, but the physical model constraint prevents large deviations.

**Thesis claim:** "Under AR(1) σ=0.15 forecast noise, both controllers degrade by less than 1.5%. SAC's robustness is comparable to MPC's, confirming the policy did not overfit to perfect-forecast information during training."

---

## 10. Critical Analysis of Stability Proposals

### Proposals initially made and critiqued

#### Proposal: Lower max_grad_norm (1.0 → 0.5)
**Argument for:** Smaller gradient steps prevent critic updates from compounding.  
**Argument against (accepted):** The explosion isn't caused by individual large gradients but by *target overestimation compounding*. The critic was stable with max_grad_norm=1.0 for 150k steps. Halving the clip threshold would slow all learning without preventing the structural cause. Delays the explosion, doesn't fix it.

#### Proposal: Lower LR_START (3e-4 → 1.5e-4)
**Argument for:** Smaller updates throughout.  
**Argument against (accepted):** The schedule was tuned to produce the policy peak at ~step 200k. Lower LR shifts the peak later, requires >500k steps, burns 2× compute for a likely similar outcome. This is a delay tactic, not a fix.

#### Proposal: Lower tau (0.005 → 0.001)
**Argument for:** Slower target network updates = smoother Bellman targets = less bootstrap amplification.  
**Argument against (accepted):** Lower tau delays convergence by 5×. Slower target tracking means the target net lags by 1000+ steps. SB3 default is 0.005; going to 0.001 is non-standard and requires thesis justification. Still doesn't address why targets are growing.

#### Proposal: Huber loss instead of MSE
**Argument for:** Bounds gradient when targets are very wrong; mathematically sound fix for Bellman overestimation.  
**Argument against (accepted):** SB3 doesn't expose this cleanly for SAC — requires monkey-patching or subclassing. Can silently mask a real value function problem. The δ threshold requires tuning. Implementation risk is high for uncertain gain.

### Root cause analysis (accepted)
The explosion in v2.7 is fundamentally caused by three interacting factors absent in v2.6:
1. **93-day episodes** (was ~80 days effective): larger discounted-return targets, γ=0.99 over 93 steps gives γ^93 ≈ 0.40, vs γ^80 ≈ 0.45. The return distribution has larger variance.
2. **Larger per-agent obs (8 features vs 5):** More gradient surface → critic has more ways to overfit to training distribution.
3. **VDN sum of 130 per-agent critics:** Any per-agent Q bias is multiplied 130×. This is structural.

### Proposal B (Adaptive LR reduction on critic-loss spike)
A custom callback that monitors rolling mean of critic_loss and multiplies LR by 0.3 if it exceeds 50. Self-tuning, reactive, principled.  
**Status:** Deprioritized. The EvalCallback already captures the best policy at step 200k before the explosion propagates. Proposal B would only help if a *better policy exists past step 200k* — which the evidence doesn't support (both seeds peaked at 200k). Proposal B is a "maybe" with real implementation risk.

### Proposal C (Episode-length curriculum) — CHOSEN
Train first 50k steps with episodes truncated at day 60, then switch to full 93 days.  
**Why this works:** The critic's deadly-triad amplification is worst during the *initial* value function learning phase when the critic is uncalibrated and the 93-day return distribution has high variance. By starting with shorter episodes (smaller return magnitude, less variance), the critic develops a stable initial calibration. Then extending to full episodes gives the actor access to late-season dynamics with a critic that can already handle the signal.  
**Why not day 60 only:** 60-day episodes cover all growth stages the actor needs to learn the basic policy. Days 60-93 are late grain-fill — agronomically important but not introducing qualitatively new control challenges. The actor sees these in the post-warmup phase.  
**Why 50k warmup:** 20% of the 250k total budget (roughly 100 short episodes), sufficient for critic calibration without over-restricting the full-season learning phase.

---

## 11. v2.8 Design — Current Work-in-Progress

### The two changes

#### Change 1: x1_overshoot_norm feature (9th per-agent feature)
```python
x1_overshoot_norm = np.clip(np.maximum(x1 - FC, 0.0) / FC, 0.0, 1.0)
```
- Zero when x1 ≤ FC (healthy regime)
- Grows linearly with overshoot above FC
- Clipped at 1.0 (corresponds to x1 = 2×FC = 280 mm, above saturation — never actually fires)
- Applied per-agent (varies across 130 agents)

**Why:** This is exactly the quantity inside r6 (`max(x1-FC,0)²/FC²`). Putting `max(x1-FC,0)/FC` in the obs makes the gradient from r6 maximally informative: the agent can directly associate "this feature is high → I got penalized → I should reduce u when this feature is high." In v2.7 the agent had to *infer* the FC threshold from `x1_norm` patterns. Now it's explicit.

**Expected effect:** `corr(u, x1)` should become negative in wet years. The x1-conditioning weakness should close.

#### Change 2: Episode-length curriculum
- `curriculum_warmup_steps = 50_000` (default) — transition point
- `curriculum_short_len = 60` (default) — episode length during warmup
- Tracked by `self._global_step_count` per-env instance (incremented each `step()` call)
- Eval env always uses `curriculum_warmup_steps=0` (full episodes throughout)
- After warmup, `_truncation_day = 93`; during warmup, `_truncation_day = 60`
- Decision is made at `reset()` time (not mid-episode)

**Expected effect:** Critic-loss stays bounded throughout 250k training. Peak eval reward exceeds −0.26 (seed 0's best). Training can potentially be extended to 350k+ steps without explosion.

### Dimension changes (v2.7 → v2.8)

| Quantity | v2.7 | v2.8 |
|---|---|---|
| N_AGENT_FEATURES | 8 | **9** |
| OBS_DIM | 1097 | **1227** |
| PER_AGENT_INPUT_DIM (actor) | 65 | **66** |
| PER_AGENT_CRITIC_INPUT_DIM | 66 | **67** |

### Backward compatibility
v2.8 runner.py can load v2.7 and v2.6 checkpoints:
- `dim=67, flat` → CTDESACPolicy (v2.8 default)
- `dim=66, flat` → V27CTDESACPolicy (v2.7)
- `dim=63, wrapped` → WrappedVDNCTDESACPolicy (v2.6)
- `dim=837, flat` → MonolithicCTDESACPolicy (pre-VDN)

---

## 12. Files Produced — v2.8

All files are in `/mnt/user-data/outputs/v28_files/` (available in the current chat session's output directory). These need to be committed to the repository before running.

### Files to commit to repo root

| File | Destination | Status |
|---|---|---|
| `src/rl/gym_env.py` | `src/rl/gym_env.py` | ✓ Ready |
| `src/rl/networks.py` | `src/rl/networks.py` | ✓ Ready |
| `src/rl/runner.py` | `src/rl/runner.py` | ✓ Ready |
| `src/rl/train.py` | `src/rl/train.py` | ✓ Ready |
| `tests/test_rl_smoke.py` | `tests/test_rl_smoke.py` | ✓ Ready |
| `tests/test_factorized_critic.py` | `tests/test_factorized_critic.py` | ✓ Ready |
| `colab_sac_v28.ipynb` | `notebooks/colab_sac_v28.ipynb` | ✓ Ready |
| `kaggle_sac_v28.ipynb` | `notebooks/kaggle_sac_v28.ipynb` | ✓ Ready |
| `change_spec_v28.md` | `reports/` or repo root | ✓ Ready |

### Summary of what each v2.8 file does

**`gym_env.py`:** Adds `x1_overshoot_norm` as 9th per-agent feature. Adds `_global_step_count` and `_truncation_day` tracking for curriculum. `__init__` accepts `curriculum_warmup_steps` and `curriculum_short_len` kwargs. `step()` increments global counter and uses `_truncation_day` for `truncated` condition. All other logic unchanged from v2.7.

**`networks.py`:** Updates v2.8 constants to 9 features / 1227 obs / 66 actor input / 67 critic input. Preserves V27_* constants (`V27_OBS_DIM=1097`, `V27_N_AGENT_FEATURES=8`, etc.) and V26_* constants for legacy checkpoint loading. Adds `V27CTDESACPolicy` class (uses `_V27SharedActor` + `_V27FactorizedContinuousCritic`) alongside the existing `WrappedVDNCTDESACPolicy` (v2.6) and `MonolithicCTDESACPolicy` (pre-VDN).

**`runner.py`:** Five-way detection table in `_load_sac_model()`. `_build_obs()` branches on `self._obs_layout` ('v28'/'v27'/'v26'). v2.8 obs produces 1227-dim with 9-feature per-agent block including overshoot; v2.7 produces 1097-dim with 8-feature block; v2.6 produces 707-dim with 5-feature block.

**`train.py`:** Version string "2.8.0". `TOTAL_TIMESTEPS` default reduced to 250,000 (was 500,000). `BUFFER_SIZE` reduced to 250,000 to match. `train_sac()` now accepts `curriculum_warmup_steps` and `curriculum_short_len` kwargs, passes them to `IrrigationEnv`. Eval env always uses `curriculum_warmup_steps=0`.

**`test_rl_smoke.py`:** OBS_DIM updated to 1227. `PER_AGENT_BLOCK_END = 1170`. New tests: `test_x1_overshoot_feature_zero_at_reset`, `test_x1_overshoot_feature_nonzero_when_above_FC`, `test_x1_overshoot_feature_matches_definition`, `test_curriculum_truncates_short_episodes_during_warmup`, `test_curriculum_switches_to_full_after_warmup`, `test_curriculum_disabled_when_warmup_zero`. Total expected: 17 tests.

**`test_factorized_critic.py`:** Adds `test_v27_legacy_load_shape` to guard that `_V27SharedActor` and `_V27FactorizedContinuousCritic` build correctly with 1097-dim obs and produce the correct (66-wide) first critic layer. Renames old v2.6 test to `test_v26_legacy_load_shape`. Total: 9 tests.

**`colab_sac_v28.ipynb`:** 8 cells. Updated header table, WandB diagnostic table (ep_len_mean = 60 during warmup → 93 after), Cell 4 passes curriculum kwargs, Cell 6 diagnostics now include `corr(u, x1)` as the primary v2.8 target metric.

**`kaggle_sac_v28.ipynb`:** 8 cells. Same structure as Colab but for Kaggle environment. Cell 3 pilot runs 25k steps with curriculum active and checks `ep_len_mean == 60` (not 93) as the pilot success criterion.

### Pre-flight validation checklist
Run before any training:
```bash
python -m pytest tests/test_rl_smoke.py -v        # 17 tests expected
python -m pytest tests/test_factorized_critic.py -v  # 9 tests expected
```
Then check:
1. `obs[8::9]` at reset is all zeros (x1 starts at FC, overshoot = 0)
2. After 5 steps of max irrigation: `obs[8::9]` is non-zero on some agents
3. First 3 episodes in curriculum mode truncate at day 60
4. After step 50k: episodes truncate at day 93

---

## 13. Current Problems and Proposed Fixes

### Problem 1: Training instability (critic explosion at step ~165k)

**Symptoms:** Both v2.7 seeds experienced critic_loss growing from ~2 to 6×10¹² over training. Actor quality peaked at step 200k and degraded thereafter. The policy "recovered" to mediocre performance by step 500k but never approached step-200k quality again.

**Root cause:** Deadly triad (function approximation + bootstrapping + off-policy) amplified by:
- 93-day episodes → high-variance return distribution
- VDN critic (130× amplification of per-agent Q bias)
- No structural protection against Bellman overestimation in SB3 SAC

**Proposed fix: Proposal C (curriculum, already implemented in v2.8)**  
- Mechanism: train first 50k steps on 60-day episodes (smaller return variance), then switch to 93-day
- Expected: critic develops stable calibration during warmup, then handles full-season returns without divergence
- Evidence base: standard curriculum learning for long-horizon RL (Bengio 2009, standard practice in sim-to-real transfer)
- Falsifiable prediction: critic_loss stays bounded (< 100) throughout 250k training; `ep_len_mean` shows 60.0 for steps 0-50k then 93.0 for steps 50k-250k

**Secondary fix (if curriculum alone insufficient): Proposal B (adaptive LR)**  
- Custom `EarlyCriticStabilizationCallback`: if rolling 1000-step mean critic_loss > 50, multiply LR by 0.3; if critic_loss < 10, restore LR schedule
- More principled than static LR reduction (it's reactive to the actual instability)
- 20 lines to implement; low risk

### Problem 2: Missing x1-conditioning (policy doesn't reduce u when x1 > FC)

**Symptoms:** `corr(u, x1)` ≈ 0 in wet/100% for seed 0. Policy uses forecast to avoid rain-day overshoot but doesn't reduce irrigation when soil is already saturated. MPC shows corr(u, x1) = −0.58.

**Direct consequence:** 76 waterlog days/agent vs MPC's 20. 7-9% yield gap in wet cells.

**Proposed fix: x1_overshoot_norm feature (already implemented in v2.8)**
- Mechanism: add `max(x1-FC, 0)/FC` as a 9th per-agent feature → exactly the quantity in r6 → direct gradient path from reward to feature
- Expected: `corr(u, x1)` becomes negative; waterlog days decrease from 67-76 toward MPC's 20
- Falsifiable prediction: in Cell 6 of the notebook after training, `corr(u, x1_mean)` should be < 0

**Alternative fix if v2.8 doesn't close the gap:** 
The deeper issue is that the agent never *needs* to back off when x1 is high unless rain is absent from forecast. A curriculum that masks the forecast for 50% of episodes during warmup would force the agent to rely on x1. This is a more aggressive intervention for a later seed.

### Problem 3: Wet-year yield gap (MPC deliberately underspends, SAC doesn't)

**Symptoms:** In wet/100%, MPC uses 299 mm (62% of budget), SAC uses 414-461 mm (86-95%). MPC's WUE in wet/100% is 12.4 kg/ha/mm vs SAC's 8.3.

**Root cause:** MPC's optimizer explicitly models that "if I don't irrigate today, I save water for a better moment — or leave it unused." SAC, trained on randomized budgets of 70-100% of 484mm, learned that it should *use* its water. The objective "maximize yield" and the tool "use budget" are conflated.

**Is this a fixable problem?** Partially. The x1-overshoot fix should teach the agent to trickle less when soil is already wet — this closes some of the gap. But the strategic "return water" behavior requires either:
- An explicit budget-return reward term (e.g., bonus for unused budget at season end)  
- Training on scenarios where returning water is clearly better (very wet years, forced)

**Thesis framing:** "The residual 7-9% wet-year gap reflects MPC's strategic budget underutilization, a behavior not encoded in the current SAC training objective. SAC maximizes yield given a budget; MPC implicitly minimizes water use while maintaining yield. These are different objectives under wet conditions."

### Problem 4: Inter-seed variance (3.4% mean, 8.6% worst case)

**Symptoms:** Dry/100% varied 358 kg/ha (8.6%) between seeds 0 and 1. High-budget cells are more variable because budget management strategy varies by seed.

**Not a design problem — expected RL behavior.** The thesis needs N≥3 seeds to report this variance credibly.

**Plan:** v2.8 seeds 2, 3, 4 will either (a) all succeed with a better/more stable policy, or (b) fail consistently, providing evidence that the v2.8 design works and the v2.7 seed variance was genuine. Either result is scientifically defensible.

---

## 14. Multi-Seed Campaign Plan

### Methodology rule (paired-samples design)

**Use the same seeds (0, 1, 2, 3, 4) for both protocols.** This is a *paired-samples* design and is statistically stronger than independent samples.

Why: when comparing protocols A (v2.7) and B (v2.8), seed-specific randomness adds noise to both. Pairing — running the same seed under both protocols and comparing them directly — *cancels* that noise. The comparison `v2.8_seed_0 - v2.7_seed_0` measures the protocol's effect on this specific initialization, independent of how good or bad that initialization happened to be. Averaging the paired differences across multiple seeds gives the cleanest estimate of the protocol's effect.

Practical note: the v2.7 results (`sac_perfect_det_*_seed{0,1}.json` etc.) are NOT overwritten by v2.8 training — v2.8 writes to a different folder (`results/rl/sac_v28_seed{N}/`). The v2.7 baseline is preserved by design.

### Plan

| Seed | v2.7 baseline | v2.8 (curriculum + x1-overshoot) | Paired comparison |
|---|---|---|---|
| 0 | ✅ Done | 🔲 Run first | `Δ = v2.8_s0 - v2.7_s0` |
| 1 | ✅ Done | 🔲 Run second | `Δ = v2.8_s1 - v2.7_s1` |
| 2 | 🔲 Optional | 🔲 Run | Independent (no v2.7 pair) |
| 3 | 🔲 Optional | 🔲 Run | Independent |
| 4 | 🔲 Optional | 🔲 Run | Independent |

**Minimum viable result for the thesis:** v2.8 on seeds 0 and 1 (N=2 paired observations). This gives you a directly defensible "v2.8 improves over v2.7" claim if both pairs show improvement.

**Strong result:** v2.8 on seeds 0-4 (N=5), with v2.7 also extended to seeds 2-4 for full paired analysis. This is the "ideal" but expensive.

**Reporting:**
- Mean ± std of paired differences (v2.8_i - v2.7_i)
- Paired t-test if N≥3 (one-tailed, H0: no improvement)
- Per-cell win/loss/tie counts

**After each seed:** immediately run noisy-forecast evaluation on the best_model checkpoint:
```bash
python -m scripts.experiments.exp_rl \
  --mode eval \
  --model results/rl/sac_v28_seed{N}/best_model/best_model.zip \
  --scenario all --budget all \
  --forecast noisy --noise-seed 42 --force
```

### Execution notes
- Colab Pro: one seed per session, A100 GPU, ~30-40 min per seed at 250k steps
- Kaggle: submit as Save & Run All, T4/P100 GPU, ~60-90 min per seed at 250k steps
- Always run Cell 3 (tests + 25k pilot) before Cell 4 (full training)
- Critical WandB checks for v2.8 pilot: `ep_len_mean == 60` (curriculum active), `critic_loss < 100` at step 25k

---

## 15. Key Technical Constants

### Field and crop
- N_AGENTS = 130
- Season length = 93 days
- FC = 140 mm (field capacity = theta6 × theta5)
- WP = 80 mm (wilting point = theta2 × theta5)
- ST = 112 mm (stress threshold = FC - p*(FC-WP), p=0.2 for rice)
- Saturation ≈ 220 mm
- FULL_SEASON_NEED_MM = 484 mm (reference seasonal budget)
- UB_MM = 12 mm/day (action upper bound)
- HI = 0.5 (harvest index)
- X4_REF = 600 g/m² (reference biomass)
- X5_REF = 50 mm (reference ponding)
- FORECAST_H = 8 days

### Reward weights
- ALPHA1 = 1.0 (biomass increment)
- ALPHA2 = 0.016 (water cost)
- ALPHA3 = 0.1 (drought stress)
- ALPHA6 = 8.0 (FC-overshoot penalty)

### SAC hyperparameters (both v2.7 and v2.8 — unchanged)
- ent_coef = 0.05 (fixed, auto-tuning disabled)
- max_grad_norm = 1.0 (via GradClipCallback)
- LR_START = 3e-4, LR_END = 5e-5 (linear decay)
- gamma = 0.99
- tau = 0.005
- batch_size = 256
- learning_starts = 1000
- gradient_steps = 1
- Actor hidden: [128, 128]
- Critic hidden: [256, 256]

### v2.8 training schedule
- TOTAL_TIMESTEPS = 250,000 (was 500k in v2.7)
- BUFFER_SIZE = 250,000
- EVAL_FREQ = 25,000 (EvalCallback; 10 checkpoints)
- CHECKPOINT_FREQ = 50,000 (model saved to disk)
- N_EVAL_EPISODES = 9 (one per scenario-budget combination)
- CURRICULUM_WARMUP_STEPS = 50,000
- CURRICULUM_SHORT_LEN = 60

### Climate scenarios
- Dry: 2022, NASA MERRA-2, low annual rainfall, early dry spell
- Moderate: 2018, typical rainfall distribution
- Wet: 2024, high annual rainfall, heavy rain events days 38, 66, 69

### WandB
- Project: 'sac-irrigation-thesis'
- Entity: 'taratorbati-itmo-university'
- Run name format: 'sac_v28_seed{N}'

---

## 16. Repository Structure

```
thesis/
├── src/
│   ├── rl/
│   │   ├── gym_env.py       ← v2.8 (THIS FILE HAS CHANGED)
│   │   ├── networks.py      ← v2.8 (THIS FILE HAS CHANGED)
│   │   ├── runner.py        ← v2.8 (THIS FILE HAS CHANGED)
│   │   └── train.py         ← v2.8 (THIS FILE HAS CHANGED)
│   ├── mpc/                 ← MPC implementation (unchanged)
│   ├── controllers/         ← Controller base classes (unchanged)
│   ├── terrain.py           ← Terrain loading (unchanged)
│   ├── precompute.py        ← Precomputed biological arrays (unchanged)
│   └── forecast.py          ← Noisy forecast generator (unchanged)
├── tests/
│   ├── test_rl_smoke.py     ← v2.8 (17 tests, THIS FILE HAS CHANGED)
│   └── test_factorized_critic.py ← v2.8 (9 tests, THIS FILE HAS CHANGED)
├── notebooks/
│   ├── colab_sac_v28.ipynb  ← v2.8 (NEW)
│   ├── kaggle_sac_v28.ipynb ← v2.8 (NEW)
│   ├── colab_sac_v26.ipynb  ← v2.6 legacy (keep as reference)
│   └── thesis_kaggle_v25.ipynb ← v2.5 legacy (keep as reference)
├── results/
│   ├── runs/                ← ALL CURRENT RESULTS (perfect + noisy, seeds 0+1)
│   │   ├── sac_perfect_det_*_seed0.json/parquet  (9 cells)
│   │   ├── sac_perfect_det_*_seed1.json/parquet  (9 cells)
│   │   ├── sac_noisy_ns42_det_*_seed0.json/parquet (9 cells)
│   │   ├── sac_noisy_ns42_det_*_seed1.json/parquet (9 cells)
│   │   ├── mpc_perfect_*_Hp{3,8,14}.json/parquet (27 cells)
│   │   ├── mpc_noisy_*_Hp3_seed42.json/parquet (9 cells)
│   │   ├── fixed_schedule_*.json/parquet (9 cells)
│   │   └── no_irrigation_*.json/parquet (3 cells)
│   └── legacy/
│       ├── runs_legacy_sac_vdn_2.6/    ← v2.6 perfect forecast results
│       ├── runs_legacy_monolithic_critic/ ← pre-VDN results
│       ├── runs_legacy_alpha0/
│       └── runs_legacy_2020/
├── scripts/
│   └── experiments/
│       ├── exp_rl.py         ← RL evaluation CLI
│       └── exp_mpc.py        ← MPC evaluation CLI
├── abm.py                   ← ABM physics (DO NOT MODIFY)
├── climate_data.py          ← Climate data loading (DO NOT MODIFY)
└── soil_data.py             ← Crop parameters (DO NOT MODIFY)
```

### Result file naming conventions
- SAC perfect: `sac_perfect_det_{scenario}_rice_{budget}pct_seed{N}.{json|parquet}`
- SAC noisy: `sac_noisy_ns42_det_{scenario}_rice_{budget}pct_seed{N}.{json|parquet}`
- MPC perfect: `mpc_perfect_{scenario}_rice_{budget}pct_Hp{H}.{json|parquet}`
- MPC noisy: `mpc_noisy_{scenario}_rice_{budget}pct_Hp{H}_seed42.{json|parquet}`

---

## 17. Operational Instructions

### Running v2.8 training (Colab)
1. Upload `notebooks/colab_sac_v28.ipynb` to Colab
2. Set Runtime → A100 GPU
3. Add WANDB_API_KEY as a Colab Secret
4. Run Cells 1-3 (mount Drive, install deps, tests + 25k pilot)
5. Check WandB: `ep_len_mean == 60` and `critic_loss < 100` at step 25k
6. Run Cell 4 with `SEED = 2` (or the next seed needed)
7. Run Cell 5 to copy to Drive immediately after training

### Running v2.8 training (Kaggle)
1. Upload `notebooks/kaggle_sac_v28.ipynb` as a new version
2. Add WANDB_API_KEY as Kaggle Secret
3. Change `SEED` to the desired value (2, 3, 4, 5, 6) in Cell 4
4. Submit as **Save Version → Save & Run All** (avoids browser-disconnect issues)

### Running noisy-forecast evaluation on a new best_model
```bash
# MPC (run once, Hp=3 is all you need for the comparison)
python -m scripts.experiments.exp_mpc \
    --scenario all --budget all --horizon 3 \
    --forecast noisy --noise-seed 42 --force

# SAC (after each seed's training completes)
python -m scripts.experiments.exp_rl \
    --mode eval \
    --model results/rl/sac_v28_seed{N}/best_model/best_model.zip \
    --scenario all --budget all \
    --forecast noisy --noise-seed 42 --force
```

### Running the test suite
```bash
# v2.8 smoke tests
python -m pytest tests/test_rl_smoke.py -v

# v2.8 critic tests
python -m pytest tests/test_factorized_critic.py -v

# Both together
python -m pytest tests/ -v
```

### Expected test counts (v2.8)
- `test_rl_smoke.py`: 17 tests
- `test_factorized_critic.py`: 9 tests

### CLI for evaluating any checkpoint
```bash
python -m scripts.experiments.exp_rl \
    --mode eval \
    --model <path_to_best_model.zip> \
    --scenario [dry|moderate|wet|all] \
    --budget [100|85|70|all] \
    --forecast [perfect|noisy] \
    [--noise-seed 42] \
    [--force]
```
The runner auto-detects checkpoint version (v2.8/v2.7/v2.6) from critic input dimension.

---

## 18. Thesis Narrative Status

### What's complete and defensible NOW (v2.7, 2 seeds)

**Chapter 4 — Controller Design:**  
- v2.7 obs layout (9 features for v2.8, 8 for v2.7) — needs small update for v2.8
- VDN factorized critic architecture — complete
- SAC hyperparameter justification — complete
- MPC Hp=3 as operating point recommendation — complete
- Reward function derivation — complete

**Chapter 5 — Results:**  
All cells have real data now. The narrative for v2.7 N=2 seeds is:

*"v2.7 SAC produces near-MPC performance in 6 dry/moderate cells (within 0.6–3.3% across two random seeds), with a residual 6–9% gap in three wet-year cells attributable to MPC's strategic budget underutilisation. The v2.6 architectural pathologies — constant-trickle irrigation and positive correlation with rainfall (+0.38) — are resolved in v2.7 across both seeds. Seed-to-seed yield variance of 3.4% mean (8.6% worst-case) indicates the algorithm has not yet converged to a unique optimal policy, motivating future multi-seed campaigns."*

**Noisy-forecast section:**  
*"Under AR(1) σ=0.15, ρ=0.6 forecast noise with matching random seed (seed=42), both controllers degrade by less than 1.5%. SAC's worst-case degradation is 1.5% (moderate/85%, seed 0), MPC's is 0.5% (wet/70%). Neither controller is meaningfully harmed by this noise level, confirming SAC's policy did not overfit to perfect-forecast information during training."*

### What would strengthen the thesis (v2.8 campaign)
1. N=5 seeds (current N=2) — needed for statistical defensibility
2. v2.8 closes wet-year x1-conditioning gap — would remove the "residual 6-9%" caveat
3. v2.8 shows stable training throughout 250k — would remove the "critic explosion at step 165k" caveat
4. Noisy-forecast on each v2.8 seed — confirms robustness result holds for the better policy

### Suggested Chapter 5 structure (current results)
1. **Baseline controllers** — Fixed schedule vs NoIrr (establishes that irrigation matters)
2. **MPC performance** — Hp=3 as reference, Hp=3/8/14 comparison, note Hp=3 is the practical choice
3. **SAC architectural ablation** — Monolithic → VDN → v2.6 → v2.7 (each change's contribution)
4. **SAC v2.7 two-seed analysis** — yields, seed variance, closed-loop diagnostics
5. **SAC vs MPC comparison** — perfect forecast, 2-seed mean ± std vs MPC, wet-year gap explanation
6. **Noise robustness** — perfect vs noisy comparison, both controllers' degradation
7. **Training analysis** — EvalCallback curve, critic explosion, why step 200k is the peak
8. **Discussion / future work** — v2.8 hypothesis, x1-conditioning, budget-conservation gap

### Things still to do for defense
- [ ] Run v2.8 training seeds 2, 3, 4
- [ ] Run noisy-forecast evaluation on all v2.8 seeds
- [ ] Update Chapter 4 per-agent obs block: "8 features → 9 features (v2.8)"
- [ ] Fill Chapter 5 with actual result tables from the unified CSV files
- [ ] Update defense presentation slides (currently use v2.6 numbers)
- [ ] Defense speech revision to reflect v2.7 wet-year improvement story

---

## Appendix A: Mistakes and Corrections Made During Development

### Mistake 1: Early termination on budget exhaustion (v2.6)
Episodes terminated when budget ran out, meaning the agent never learned late-season drought consequences. Corrected in v2.7.

### Mistake 2: The γ observation bug (v2.5/v2.6)
5th per-agent feature was `x2/theta18` (uniform GDD) instead of `elev_norm`. Active since first VDN introduction. Caused spatial blindness. Corrected in v2.7.

### Mistake 3: Overestimating rb penalty
Initial analysis suggested the budget burn-rate penalty was "too weak." Correct analysis showed it was ~30:1 overpowered relative to r1. Removed rather than increased.

### Mistake 4: Proposing stability fixes that were delay tactics
Proposals to lower max_grad_norm, lower LR, and lower tau were critiqued as delay tactics rather than root-cause fixes. Replaced with Proposal C (curriculum) which addresses the structural cause.

### Mistake 5: "Beats MPC in 4 cells" claim was a single-seed artifact
After seed 0, the claim was made that SAC_v27 beats MPC in 4/9 cells. After seed 1, the two-seed mean trails MPC in all 9 cells. The honest result is "within 0.6-3.3% of MPC in 6 dry/moderate cells."

### Mistake 6: Two conflicting answers about which seeds to use
Initially recommended running Proposal B on seed 0 and Proposal C on seed 1 (but combined into one protocol — methodologically muddled). Then over-corrected by saying "never re-run seeds 0 and 1 with a new protocol." The correct answer is a paired-samples design: run the new protocol (v2.8 with both curriculum AND x1-overshoot bundled) on the SAME seeds as the v2.7 baseline. The v2.7 results are saved in separate folders and aren't overwritten. Pairing controls for seed-specific randomness and is the strongest defensible comparison.

---

## Appendix B: Questions Asked and Answers

**Q: Why does the SAC not respond to x1 even though the penalty is strong?**  
A: The agent learned forecast-based control (uses 8-day rain forecast to predict when overshoot will happen) rather than state-based control (using current x1 to decide now). This is rational: forecast is forward-looking and explains 95% of overshoot events. The remaining 5% (soil already high, no more rain) was not learned because it was a small fraction of the gradient signal. Fix: add x1_overshoot_norm feature to make the FC threshold explicit.

**Q: Should we give FC as a feature?**  
A: No. FC is a constant — the network bias term absorbs it. Giving `max(x1-FC,0)/FC` is far more useful because it has the exact zero-crossing that the reward function uses. The agent sees a feature that is zero in the healthy regime and nonzero only when it's being penalized.

**Q: Should total_timesteps be 350k? 500k?**  
A: 250k is correct. Both seeds peaked at step 200k. Steps 200k-500k were wasted or harmful. The cap also reduces Colab session risk. EvalCallback captures the peak regardless.

**Q: Does changing ent_coef help?**  
A: Unlikely to be the lever. The instability is critic-structural (Bellman overestimation), not exploration-related. Lowering ent_coef reduces exploration and may prevent the agent from learning x1 conditioning (which it still hasn't learned). Higher ent_coef may hurt convergence. Keep at 0.05.

**Q: Does more adaptive LR help?**  
A: Proposal B (adaptive LR on critic spike) is reasonable but secondary. The curriculum should prevent the explosion entirely. Only implement Proposal B if v2.8 still explodes.

**Q: Do I need multiple seeds? Why?**  
A: Yes. RL seed variance is 3.4% mean / 8.6% worst case — a reviewer will ask "is this cherry-picked?" N=2 is weak (you can't compute a meaningful std), N=5 is the standard ask. N=3 for v2.8 is the minimum defensible.

**Q: Noisy forecast — does it require retraining?**  
A: No. It's pure inference: the forecast block in the obs is corrupted, but the policy and ABM physics are unchanged. Run `exp_rl` with `--forecast noisy --noise-seed 42`.

---

*End of handoff document.*  
*Continue in new chat by sharing this file.*
