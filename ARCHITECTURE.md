# Thesis Implementation Report v2.0
## MPC and RL Controllers for Constrained Irrigation in Topographically Heterogeneous Terrain

**Author:** Tara Torbati, ITMO University, R4237c  
**Supervisor:** Peregudin A.A.  
**Document version:** 2.0 — April 2026  
**Status:** MPC implemented and tested; convergence fix pending; RL not yet started

---

## 1. Executive Summary

This thesis investigates whether a Soft Actor-Critic (SAC) RL agent can match MPC performance on constrained irrigation for a topographically heterogeneous rice field in Gilan Province, Iran. The plant model is a 130-agent crop-soil ABM with cascade water routing and surface ponding dynamics, validated against NASA satellite data (r=0.74).

The MPC uses CasADi+IPOPT in a multiple-shooting formulation with smooth approximations. First result: 4261 kg/ha on dry/100% (18% above the fixed-schedule baseline of 3607 kg/ha).

## 2. Scope

**In scope:** Rice (Hashemi, 93-day season). 3 scenarios × 3 budgets × 2 MPC horizons × 2 forecast modes × 2 RL info modes. 6 controllers total.

**Out of scope:** Tobacco (Phase 2), field validation, stochastic MPC, edge profiling, model-based RL.

## 3. System Architecture

Three layers: Environment (ABM, terrain, climate) → Controllers (all implement base.Controller) → Runner (run_season, identical loop for all controllers).

Cascade routing mode. Initial x1=140mm (FC), x5=0. Budget tracked field-averaged. Crop selection via get_crop() function.

## 4. MPC Formulation

- **Variables:** u ∈ R^(N×Hp), N=130, Hp∈{8,14}
- **Shooting states:** x1, x5 only. x2 precomputed. x3, x4 tracked from true state.
- **Cost:** 5 terms — terminal biomass (Mayer), water (path), drought (path), sink ponding (path), Δu (path). All O(1) normalized.
- **Weights:** α1=1.0, α2=0.01 (nominal), α3=0.1, α4=0.5, α5=0.005. Sweep: α2∈{0.0001,0.01,0.03}.
- **Constraints:** Box [0,12mm], budget (linear, inside IPOPT), dynamics (equality).
- **Solver:** CasADi+IPOPT, MUMPS, smooth approx default, max_iter=500, tol=1e-4, acceptable_tol=1e-3.

## 5. RL Formulation (planned)

- **Env:** Gymnasium wrapper. Obs=660 dims. Action=Box(0,1,shape=(130,)).
- **Reward:** Dense, mirrors MPC cost. Terminal bonus.
- **Budget:** Soft penalty + early termination + burn-rate shaping.
- **Algorithm:** SAC (SB3). Shared-params actor (CTDE). Centralized critic.
- **Training:** 500k-2M steps, 5 seeds, Kaggle GPU.

## 6. Evaluation Framework

6 controllers × 3 scenarios × 3 budgets. Metrics: yield, WUE, budget compliance, drought/waterlog days, sink ponding, spatial equity, solve time. Mann-Whitney U tests. Weight sensitivity: 7 runs.

## 7. Current File Structure

See the repository at https://github.com/taratorbati/thesis for the actual file tree.

Key directories: src/ (terrain, persistence, precompute, forecast, runner, controllers/, mpc/), scripts/ (experiments/, preprocess/), results/ (runs/, precomputed/), notes/, history/, reports/.

## 8. Build Status

| Step | Status |
|---|---|
| A: Foundation | ✅ Complete |
| B: Controllers + runner + baselines | ✅ Complete, 12 runs |
| C: Precomputation | ✅ Complete, 6 cached files |
| D: MPC | ⚠️ Working, convergence fix pending |
| D-fix: smooth + x3 + fallback + IPOPT | 📋 Prepared, not yet in repo |
| E: RL | ❌ Not started |
| F: Analysis | ❌ Not started |

## 9. Known Issues

1. **MPC convergence (fix prepared):** Non-smooth ops cause 76s/solve. Fix: smooth approx default + IPOPT tuning. Expected: 2-5s/solve.
2. **x3 drift (fix prepared):** controller.py approximated x3 instead of reading from true state.
3. **Infeasible fallback (fix prepared):** Falls back to previous action (dangerous). Fix: fall back to zero.
4. **Equal-elevation neighbors:** 34 pairs exist (3.3%). Non-issue — no water trapping.

## 10. Baseline Results

No-irrigation yields: dry 1462, moderate 1426, wet 2266 kg/ha. Fixed-schedule best: moderate/85% = 3699 kg/ha. Wet/100% disaster: 2763 kg/ha (over-irrigation drowns sinks). MPC first result: dry/100% = 4261 kg/ha (18% above fixed-schedule).

## 11. Locked Decisions

UB=12mm. α2=0.01 nominal. Hp∈{8,14}. Initial x1=FC=140mm. x5=0. Budgets={100%,85%,70%}. Smooth approx default. Crop=rice Phase 1. SAC with CTDE actor. 5 seeds on Kaggle. Long-format parquet + JSON sidecar.

## 12. Parameter Tables

Soil: θ1=0.096, θ2=0.15, θ3=10mm, θ4=0.05, θ6=0.35, θsat=0.55.
Rice: θ5=400mm, θ7=10°C, θ9=35°C, θ10=42°C, θ11=0.003, θ12=0.003, θ13=0.65, θ14=0.8, θ18=1250, θ19=0.95, θ20=417, Kc=1.15, p=0.20, HI=0.42. Season DOY 141-233.
Derived: FC=140mm, WP=60mm, TAW=80mm, RAW=16mm, ST=124mm, SAT=220mm.
Budgets: 100%=484mm, 85%=411mm, 70%=339mm.

*This document supersedes version 1.0 and all prior analysis documents.*
