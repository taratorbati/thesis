# Thesis Implementation Report v2.1
## MPC and RL Controllers for Constrained Irrigation in Topographically Heterogeneous Terrain

**Author:** Tara Torbati, ITMO University, R4237c  
**Supervisor:** Peregudin A.A.  
**Document version:** 2.1 — May 2026  
**Status:** MPC complete; RL implemented and training on Kaggle GPU

---

## 1. Executive Summary

This thesis investigates whether a Soft Actor-Critic (SAC) RL agent can match MPC performance on constrained irrigation for a topographically heterogeneous rice field in Gilan Province, Iran. The plant model is a 130-agent crop-soil ABM with cascade water routing and surface ponding dynamics, validated against NASA satellite data (r=0.74).

The MPC uses CasADi+IPOPT in a multiple-shooting formulation with smooth approximations. First result: 4261 kg/ha on dry/100% (18% above the fixed-schedule baseline of 3607 kg/ha).

## 2. Scope

**In scope:** Rice (Hashemi, 93-day season). Evaluation: 3 held-out years × 3 budgets = 9 cells. Training: 23 years × continuous U(70%,100%) budget.

**Out of scope:** Tobacco (Phase 2), field validation, stochastic MPC, edge profiling, model-based RL.

## 3. System Architecture

Three layers: Environment (ABM, terrain, climate) → Controllers (all implement base.Controller) → Runner (run_season, identical loop for all controllers).

Cascade routing mode. Initial x1=140mm (FC), x5=0. Budget tracked field-averaged.

## 4. MPC Formulation

- **Variables:** u in R^(N×Hp), N=130, Hp in {8,14}
- **Shooting states:** x1, x5 only. x2 precomputed. x3, x4 tracked from true state.
- **Cost:** 5 terms — terminal biomass (Mayer), water (path), drought (path), sink ponding (path), delta-u (path). All O(1) normalized.
- **Weights:** alpha1=1.0, alpha2=0.01 (nominal), alpha3=0.1, alpha4=0.5, alpha5=0.005.
- **Constraints:** Box [0,12mm], budget (linear, inside IPOPT), dynamics (equality).
- **Solver:** CasADi+IPOPT, MUMPS, smooth approx, max_iter=500, tol=1e-4.

## 5. RL Formulation

- **Env:** Gymnasium wrapper. Obs=707 dims. Action=Box(0,1,shape=(130,)).
- **Obs layout (707 dims at H=8):**
  - [0:650]   Per-agent (5×130): x1_norm, x5_norm, x4_norm, x3, elevation
  - [650:659] Global scalars (9): day_frac, budget_frac, burn_rate,
              rain_today, ETc_today, h2_today, h7_today, g_base_today,
              budget_total_norm
  - [659:707] Per-day forecasts (6×8=48): rain, ETc, rad, h2, h7, g_base
- **Reward:** Dense, approximate negation of MPC path cost under gamma→1.
- **Algorithm:** SAC (SB3 2.6.0). Shared-parameter actor (CTDE). Centralized critic.
- **Training design:** One policy across 23 years × U(70%,100%) budget,
  randomized per episode. Eval cells never seen during training.
- **Eval cells:** 2018/2022/2024 × {70%,85%,100%} = 9 cells (held out entirely).
- **Hyperparameters (all from measurement):**
  - target_entropy=-65 (pilot: -130 clearly worse; -65 vs -32 within noise)
  - buffer_size=200k, gradient_steps=1, batch_size=256
  - Kaggle T4 measured: 68 steps/sec; 500k steps ~2 hrs/seed; 5 seeds ~10 GPU-hrs

## 6. Evaluation Framework

Controllers × 9 eval cells. Metrics: yield (kg/ha), WUE (kg/m3), budget compliance, drought days, solve/inference time. Mann-Whitney U across 5 seeds.

## 7. Scenario Split

| Year | Rainfall | Role |
|------|----------|------|
| 2022 | 39.7 mm  | Eval: dry (in-distribution) |
| 2018 | 108.8 mm | Eval: moderate (upper training edge) |
| 2024 | 176.8 mm | Eval: wet (OOD extreme) |
| 23 others | 14–88 mm | Training only |

## 8. Key Files

| File | Role |
|------|------|
| abm.py | Ground-truth crop-soil ABM |
| climate_data.py | TRAINING_YEARS, EVAL_YEARS constants |
| src/rl/gym_env.py | Gymnasium env, obs_dim=707 |
| src/rl/networks.py | CTDESACPolicy, SharedActor (62-dim per-agent input) |
| src/rl/train.py | SAC training loop |
| src/rl/runner.py | Inference (707-dim obs, matches gym_env exactly) |
| src/mpc/dynamics_sym.py | CasADi dynamics (v2.2: drought stress fix) |
| src/forecast.py | PerfectForecast, NoisyForecast (AR(1) rho=0.6) |
| src/mpc/controller.py | MPCController |
| src/controllers/base.py | Abstract Controller interface |
