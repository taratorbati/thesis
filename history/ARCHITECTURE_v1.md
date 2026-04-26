# Thesis Implementation Report
## MPC and RL Controllers for Constrained Irrigation in Topographically Heterogeneous Terrain

**Author:** Tara Torbati, ITMO University, R4237c
**Supervisor:** Peregudin A.A.
**Document version:** 1.0 — April 2026
**Status:** Architecture finalized; Step A (foundation) implemented

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [Research question and scope](#2-research-question-and-scope)
3. [Existing codebase audit](#3-existing-codebase-audit)
4. [System architecture](#4-system-architecture)
5. [MPC formulation](#5-mpc-formulation)
6. [RL formulation](#6-rl-formulation)
7. [Evaluation framework](#7-evaluation-framework)
8. [File and module structure](#8-file-and-module-structure)
9. [Persistence and caching](#9-persistence-and-caching)
10. [Build plan](#10-build-plan)
11. [Risks and mitigations](#11-risks-and-mitigations)
12. [Open questions](#12-open-questions)
13. [Appendix A — locked decisions](#appendix-a--locked-decisions)
14. [Appendix B — parameter tables](#appendix-b--parameter-tables)

---

## 1. Executive Summary

This thesis investigates whether a Soft Actor-Critic (SAC) reinforcement learning agent can match the performance of a Model Predictive Controller (MPC) on the constrained irrigation problem for a topographically heterogeneous rice field in Gilan Province, Iran. The plant model — a 130-agent crop-soil ABM with cascade water routing and waterlogging dynamics — is already built, validated against NASA satellite observations, and ready to serve as both the MPC's prediction model and the RL training environment.

The contribution is twofold. First, a constraint-aware MPC formulation extending Lopez-Jimenez et al. (2024) with explicit ponding penalties at sink agents, drought-stress regularization, and a global seasonal water budget enforced inside the optimizer rather than via post-hoc clipping. Second, a head-to-head comparison between this MPC and a SAC agent under both information-equal and deployment-realistic forecast conditions, evaluated on three climatic scenarios (dry 2022, moderate 2020, wet 2024) and three water-budget levels (100%, 85%, 70% of full irrigation need).

The MPC is implemented in Python via CasADi with the IPOPT interior-point solver. The RL is implemented in Stable-Baselines3 with a centralized-training-decentralized-execution (CTDE) actor-critic, trained on Kaggle GPU. All experiment results are stored as parquet files with JSON metadata sidecars to support reproducible analysis without re-execution.

---

## 2. Research Question and Scope

### 2.1 Primary research question

Can a Soft Actor-Critic reinforcement learning agent match the constrained irrigation performance of a Model Predictive Controller using the same topographical agent-based model, while offering orders-of-magnitude lower inference latency suitable for edge deployment on agricultural cyber-physical systems?

### 2.2 Secondary questions

- How does forecast uncertainty (a 7-day rainfall noise distribution with σ growing in √horizon) affect MPC performance versus the perfect-forecast upper bound?
- Does providing the RL agent with the same forecast information as the MPC close the performance gap, or does the model-based controller retain a structural advantage from its dynamics knowledge?
- How sensitive are the comparison results to the cost-function weight choices, particularly the water/biomass tradeoff weight α₂?

### 2.3 What is in scope

- Phase 1: Rice (Hashemi cultivar). 93-day field season. DOY 141 to 233 (May 20 to August 20).
- Three climatic scenarios from a 25-year NASA POWER record.
- Three water-budget levels per scenario.
- Two MPC prediction horizons (H_p = 7 and H_p = 14).
- Two forecast modes for MPC (perfect, noisy).
- Two information modes for RL (no forecast, 7-day noisy forecast).
- Six controllers: no-irrigation, fixed-schedule (linear-decay 19-event), MPC-perfect, MPC-noisy, SAC-no-forecast, SAC-forecast.

### 2.4 What is out of scope (Phase 1)

- Tobacco. Deferred to Phase 2 if Phase 1 completes ahead of schedule.
- Field validation. Simulation-only thesis, stated as such.
- Per-agent soil heterogeneity. Uniform silty loam justified by regional soil survey.
- Stochastic MPC (scenario-tree). Nominal MPC with noisy forecast is the standard comparator.
- Edge hardware profiling. Discussed qualitatively in CPS chapter, not benchmarked.
- Differentiable-physics RL or model-based RL extensions.

### 2.5 What was rejected

Three architectural alternatives proposed by prior LLM analyses (Gemini's responses, the blueprint document) were considered and rejected:

- **Decomposed per-agent MPC with Lagrangian budget coordination.** Rejected because the cascade DAG provides natural sparsity that IPOPT exploits directly, making decomposition unnecessary for the 1820-variable problem size.
- **JAX with projected gradient descent.** Rejected because PGD on a non-convex non-smooth problem with global budget coupling converges at O(1/√k) rate requiring ~1000+ iterations, slower in practice than IPOPT's ~50 Newton iterations. Additionally, the loss of classical MPC convergence guarantees weakens the thesis framing of MPC as the gold-standard benchmark.
- **Spatial aggregation into 3-5 management zones.** Rejected because the topographical heterogeneity is the central thesis contribution; aggregating destroys the per-agent control resolution that the ABM is designed to exploit.

---

## 3. Existing Codebase Audit

The student arrives at this implementation phase with substantial existing code. The following components are already built and validated:

### 3.1 Working components

- **`abm.py`** — 130-agent CropSoilABM class with five state variables (x₁ soil water, x₂ thermal time, x₃ maturity stress, x₄ biomass, x₅ surface ponding). Includes three water-routing modes (none, simple, cascade), the cascade boundary condition for sink agents, and the corrected biomass equation where drought stress h₃ multiplies the daily growth increment.
- **`soil_data.py`** — Rice (Hashemi cultivar) and Tobacco parameters with FAO-56 references for field capacity, wilting point, root depth, crop coefficient, depletion fraction, base temperature, and harvest index. Calibrated radiation use efficiency for biomass in g/m² total dry matter.
- **`preprocess.py`** — NASA POWER 25-year record (2000–2025) ingestion. Filters April–October, cleans the 375 mm rainfall anomaly on 2022-04-07, fills missing values with monthly averages, computes both Hargreaves and Penman-Monteith ET₀ (PM is the production reference).
- **`climate_data.py`** — Loads cleaned CSV, extracts crop-season climate dictionaries for 2020/2022/2024.
- **`cross_validate_gwetroot.py`** — Validates the ABM against NASA GWETROOT satellite soil wetness. Pearson r = 0.738 for the dry 2022 scenario, confirming the model captures temporal soil-moisture dynamics correctly.
- **`run_comparison.py`, `run_comparison_2.py`** — Demonstrate the necessity of cascade routing mode by showing that the simple-routing mode fails to predict ~2200 mm of accumulated ponding at sink agents under heavy irrigation in the wet scenario.
- **`validate_physics_fao.py`** — FAO-56 unit test on a flat 10-agent topology, comparing the ABM's water balance to the textbook FAO-56 single-bucket equation.
- **8 `plot_*.py` scripts** — Climate visualizations (temperature, rainfall, ET₀, humidity, wind, pressure, radiation, soil moisture).
- **`gilan_farm.tif`** — 10×13 DEM, elevation range 74–181 m, bowl-shaped topology with 3 sink agents in the southeastern corner.

### 3.2 Issues identified in audit

- **Code duplication.** The `build_directed_graph` function is implemented identically in `cross_validate_gwetroot.py`, `run_comparison.py`, and `run_comparison_2.py`. Resolved in Step A by extracting to `src/terrain.py`.
- **Crop selection is a comment-toggle.** `soil_data.py` ends with `theta = RICE`. Switching crops requires editing the file. To be resolved in Step B by switching to a function-parameter style.
- **No controller infrastructure.** Existing scripts hardcode `u = np.zeros(N)` or `u = np.full(N, 5.0)` directly inside their daily loops. Adding any new controller (fixed schedule, MPC, RL) requires writing a new script from scratch.
- **No persistent caching of run results.** Every comparison script re-simulates from scratch on each invocation. Results are saved as plot images, not as machine-readable trajectories. Cross-run analysis requires re-running everything.
- **No precomputation.** Thermal time x₂(k), heat stress h₂(k), low-temp stress h₇(k), and the growth function g(x₂(k)) are all functions of weather alone (temperature for thermal terms, x₂ alone for growth) and could be computed once per scenario rather than recomputed inside the daily loop. This is fine for the existing simulation-only scripts but matters for MPC, where the symbolic graph inside IPOPT becomes significantly larger when these functions are computed online.
- **No forecast generation infrastructure.** MPC requires perfect and noisy forecast streams; these need to be generated and (optionally) cached for reproducibility.
- **Existing scripts are not amenable to parameter sweeps.** Running across 6 controllers × 3 scenarios × 3 budget levels × 5 seeds requires programmatic invocation, not manual script editing.

### 3.3 Decisions about existing code

- **Refactor sparingly.** The thesis student is busy and the existing code works. The only existing files modified during Step A are the three with the duplicated graph builder. All other existing files are untouched.
- **Build new infrastructure alongside, not replacing.** New code lives under `src/` and `controllers/`; experiment scripts live under `scripts/experiments/`. Existing scripts (`preprocess.py`, the `plot_*.py` files, the validation scripts) remain at the project root and continue to work as before.
- **Adopt parquet for new run output.** Existing scripts continue producing PNG plots and text reports as they do today; the new infrastructure produces parquet files plus JSON metadata for use by the new aggregation and figure-generation scripts.

---

## 4. System Architecture

### 4.1 Three-layer architecture

The system consists of three layers with strict separation of concerns:

**Layer 1: Environment layer.** The crop-soil ABM, terrain graph, climate data, and forecast generators. This layer is pure simulation — given a state, control, and weather realization, it produces the next state. It has no knowledge of who is asking it (MPC, RL, or a baseline controller).

**Layer 2: Controller layer.** A common `Controller` abstract interface implemented by no-irrigation, fixed-schedule, MPC, and RL controllers. Each implementation knows how to convert a state observation and forecast into an irrigation action vector. Controllers do not know how the simulation runs.

**Layer 3: Runner layer.** A `run_season(controller, abm, climate, budget)` function that orchestrates the day-by-day simulation, calling the controller for actions, stepping the ABM, tracking the budget, and writing checkpoints. This layer treats all controllers identically — same loop for everyone.

This design ensures fair comparison: every controller experiences the exact same simulator dynamics, the same climate realizations, the same initial state, and the same budget accounting. The only thing that varies between runs is the controller's policy.

### 4.2 Information flow per day

```
                           ┌────────────────────┐
                           │ Climate (true)     │
                           │ rainfall, ET, T... │
                           └─────────┬──────────┘
                                     │
                                     ▼
┌─────────────────────┐    ┌──────────────────────┐
│ Forecast generator  │───▶│      Runner          │
│ (perfect or noisy)  │    │  - state x(t)        │
└─────────────────────┘    │  - budget remaining  │
                           │  - forecast          │
                           └─────────┬────────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │    Controller      │
                           │  .step() → u(t)    │
                           └─────────┬──────────┘
                                     │
                                     ▼
                           ┌────────────────────┐
                           │       ABM          │
                           │ x(t+1) = f(x,u,w) │
                           └─────────┬──────────┘
                                     │
                                     ▼
                           Trajectory record
                           Checkpoint to disk
```

### 4.3 Why cascade routing matters for the controller

The ABM's three runoff modes were validated as part of the existing work. The key finding from `run_comparison_2.py` is that the cascade mode (top-to-bottom processing of agents within a single day) is the only physically realistic representation of within-day water redistribution on the steep Gilan terrain. Under heavy irrigation in the wet scenario, the simple-routing mode predicts ~200 mm of ponding at sink agents while cascade predicts ~2200 mm. A controller making decisions on the simple model would systematically under-predict its own ponding consequences and over-irrigate.

Therefore, cascade is the routing mode used by both the ABM (when serving as the MPC prediction model) and the ABM (when serving as the RL training environment).

### 4.4 Where the runners diverge

The controller-runner abstraction is identical for MPC and RL. They diverge only in:

- **What goes into the controller's `.step()` method.** The MPC needs a forecast horizon. The RL agent needs a flat observation vector.
- **What happens during training.** RL training has an outer loop over episodes that the MPC does not need.
- **Where computation happens.** MPC computation is online (inside the receding-horizon loop). RL computation is offline (during training); inference is a forward pass.

---

## 5. MPC Formulation

### 5.1 Decision variables

At each daily MPC solve at time t, the decision variable is

$$u \in \mathbb{R}^{N \times H_p}$$

with N = 130 agents and prediction horizon H_p ∈ {7, 14}. Worst case: 1820 variables.

### 5.2 Cost function

The cost combines five terms, each normalized to be O(1) per time step:

$$J(u) = -\alpha_1 \frac{\bar{x}_4(H_p)}{x_{4,\text{ref}}} + \alpha_2 \sum_{k=1}^{H_p} \frac{\sum_n u^n(k)}{W_{\text{daily,ref}}} + \alpha_3 \sum_{k=1}^{H_p} \frac{1}{N} \sum_{n=1}^{N} \frac{\max(ST - x_1^n(k), 0)}{ST - WP} + \alpha_4 \sum_{k=1}^{H_p} \frac{1}{|\mathcal{S}|} \sum_{n \in \mathcal{S}} \frac{x_5^n(k)}{x_{5,\text{ref}}} + \alpha_5 \sum_{k=0}^{H_p-2} \frac{\|u(k+1) - u(k)\|_2^2}{u_{\max}^2 N}$$

| Term | Type | Role | Reference value |
|---|---|---|---|
| α₁ terminal biomass | Mayer | Maximize end-of-horizon yield | x_{4,ref} = 900 g/m² |
| α₂ water cost | Path | Penalize total irrigation | W_{daily,ref} = 5 mm × N |
| α₃ drought stress | Path | Soft-keep x₁ above stress threshold | ST = θ₂θ₅ + (1-p)(θ₆-θ₂)θ₅ = 124 mm |
| α₄ ponding at sinks | Path (sinks only) | Penalize sustained sink flooding | x_{5,ref} = 10 mm |
| α₅ Δu regularization | Path | Numerical regularization, prevents end-of-horizon dump | u_max = UB = 12 mm |

**Why terminal biomass, not cumulative.** Biomass is monotonically non-decreasing in this model. Σ_k x̄_4(k) double-counts and rewards early growth over late growth, which is opposite of what is wanted for rice grain fill in the final weeks. The original Lopez-Jimenez 2024 paper correctly uses terminal biomass; this thesis follows.

**Why the stress and ponding terms are cumulative path costs.** Drought and waterlogging that persist through the horizon are physically damaging proportional to their duration. A path cost captures this; a terminal-only cost would not.

**Why the Δu term is included.** It regularizes the finite-horizon problem against end-of-horizon "dump all remaining budget on day H_p" artifacts. In simulation it has no physical interpretation as valve wear, but in the optimization it prevents pathological control trajectories.

**Why ponding only at sinks.** Non-sink agents drain naturally. Transient ponding on slopes is a minor cost and the cascade routing already redistributes it. Sink agents are where ponding accumulates and damages crops; restricting α₄ to 𝒮 (the set of sink agents, |𝒮| = 3) keeps the term focused on the physically meaningful subset.

### 5.3 Nominal weights

| Weight | Nominal value | Source of choice |
|---|---|---|
| α₁ | 1.0 | Reference (anchor) |
| α₂ | 0.01 | Economic estimate (rice $2.5/kg, water $0.1/m³, HI = 0.42); to be refined when student provides exact prices |
| α₃ | 0.1 | Light drought regularizer; will be checked at ±3× in sensitivity sweep |
| α₄ | 0.5 | Heavier weight because rice waterlogging is directly damaging; will be checked at ±3× in sensitivity sweep |
| α₅ | 0.005 | Small numerical regularizer; not swept |

These values are placeholders configured via YAML and will be revised when economic prices are available.

### 5.4 Constraints

- **Actuator box:** 0 ≤ u^n(k) ≤ UB = 12 mm/day, for all n, k. Enforced as IPOPT box constraint.
- **Global water budget:** Σ_{k=0}^{H_p-1} Σ_{n=1}^{N} u^n(k) ≤ W_{remaining}(t). Single linear inequality. **Enforced inside IPOPT, not via post-hoc clipping.**
- **Dynamics:** x(k+1) = f(x(k), u(k), ŵ(k)) for each agent and step, with cascade routing applied inside f. Multiple-shooting formulation: dynamics enter as equality constraints in IPOPT.
- **State bounds (x₁ ≥ 0, x₅ ≥ 0):** Enforced implicitly inside the ABM step function via `np.clip(..., 0, None)`. Not exposed as separate IPOPT constraints.

No hard x₁ ≤ FC constraint. Soft handling via the α₃ stress term and the natural ABM saturation at θ_sat θ₅. Imposing a hard upper bound risks infeasibility under wet weather where rainfall alone exceeds FC.

### 5.5 Solver: CasADi + IPOPT

CasADi provides symbolic algorithmic differentiation and a sparse NLP interface; IPOPT is the primal-dual interior-point solver from Wächter & Biegler. Together they handle medium-scale non-convex NLPs robustly with classical convergence guarantees.

**Why this choice.** Three reasons in priority order:

1. **Thesis framing.** The research question positions MPC as the model-based gold-standard benchmark. IPOPT's classical KKT convergence preserves this framing. A first-order projected-gradient alternative would make the comparison "RL without guarantees vs MPC without guarantees" — weaker.
2. **Wall-clock performance.** Despite higher per-iteration cost, IPOPT's quadratic local convergence (typically 30–80 iterations) beats first-order methods on this problem size. Expected wall-clock per solve: 1–5 seconds.
3. **Debuggability.** IPOPT has 25 years of documentation, examples, and forum support. CasADi is the standard Python interface for nonlinear MPC across chemical and process engineering.

**Specific configuration:**
- Linear solver backend: MUMPS (cross-platform, bundled with IPOPT). MA27 as optional upgrade if MUMPS proves slow.
- Tolerance: tol = 1e-6, max_iter = 200.
- Multiple-shooting formulation with state continuity as equality constraints (more robust than single-shooting for stiff systems).
- Warm start: shift previous solution by one step at each receding-horizon iteration.
- Non-smooth operators (max, min, SCS runoff kink): start with CasADi's native fmax/fmin; if IPOPT shows convergence issues, switch to explicit smooth approximations: max(x, 0) ≈ 0.5(x + √(x² + ε²)) with ε = 0.01.

### 5.6 Receding-horizon loop

Pseudocode for the daily MPC loop:

```
budget_remaining = W_total                        # at season start
for t in range(season_length):
    state_t = abm.get_state()
    forecast = generate_forecast(climate_true, t, H_p, mode)   # perfect or noisy
    u_optimal = mpc.solve(state_t, forecast, budget_remaining, H_p)
    u_today = u_optimal[:, 0]                                  # first column only
    abm.step(u_today, climate_true[t])
    budget_remaining -= u_today.sum()
    record_trajectory(t, state_t, u_today, budget_remaining)
    if t % checkpoint_interval == 0:
        save_partial_run()
```

Note: the optimizer plans H_p days ahead but only the first day's actions are applied. Tomorrow the entire problem is resolved from the new state — this is the receding-horizon principle.

### 5.7 Forecast modes

Two modes:

- **Perfect:** ŵ(t+j|t) = w(t+j) for all j ∈ {0, ..., H_p-1}. The MPC sees the true future weather. Upper bound on MPC performance.
- **Noisy:** ŵ(t+j|t) = w(t+j) × (1 + ε_j), where ε_j ~ 𝒩(0, σ_j²) and σ_j = 0.15 × √j. Same multiplicative noise applied to rainfall and ET₀; temperature unperturbed (operational temperature forecasts are accurate). Per-day fresh noise realization. To support reproducibility, all noise realizations for a given (scenario, seed) tuple are pre-generated and cached.

### 5.8 Pre-computable quantities

The following are functions of weather alone (independent of u and x₁) and can be pre-computed once per scenario, then read by the MPC as time-varying parameters:

- **Thermal time trajectory x₂(k)** — depends only on temperature.
- **Heat stress h₂(k)** — depends only on T_max.
- **Low-temperature stress h₇(k)** — depends only on T_mean.
- **Mean-temperature accumulation h₁(k)** — depends only on T_mean.
- **Growth function g(x₂(k))** — depends only on x₂(k).

Pre-computing these as scenario-specific arrays turns the MPC's symbolic graph from "compute g(x₂) inside every iteration" into "look up g[k] from a table." This removes the two-branch sigmoid evaluation from each IPOPT iteration, with empirical experience suggesting ~30% solve-time reduction.

This is implemented in Step C as `precompute.py`.

---

## 6. RL Formulation

### 6.1 Gymnasium environment

The RL environment wraps the existing `CropSoilABM` with the standard Gymnasium API:

```python
env = IrrigationEnv(crop='rice', scenario_pool=['dry', 'moderate', 'wet'],
                    forecast_mode='no_forecast' or '7_day_noisy',
                    budget_levels=[1.0, 0.85, 0.70])
obs, info = env.reset()                  # randomizes scenario, budget, seed
obs, reward, terminated, truncated, info = env.step(action)
```

### 6.2 Observation space

A flat real-valued vector. Total dimension 660:

- **Global features (10):**
  - budget_remaining / W_total
  - day / season_length
  - rainfall_today / 10 mm
  - ET₀_today / 10 mm
  - mean(x₁) / FC
  - std(x₁) / FC
  - mean(x₄) / x_{4,ref}
  - fraction of agents in drought (x₁ < ST)
  - fraction of agents in waterlog (x₁ > FC)
  - cumulative 7-day rainfall forecast / 70 mm — **zeroed in no-forecast mode**

- **Per-agent features (130 × 5 = 650):**
  - x₁ / FC
  - x₂ / θ₁₈
  - x₄ / x_{4,ref}
  - x₅ / x_{5,ref}
  - γ^n (normalized elevation)

### 6.3 Action space

Box(0, 1, shape=(N,)). Each component is a fraction of UB applied to the corresponding agent. This matches the MPC's per-agent decision exactly, making the comparison clean.

### 6.4 Reward function

Mirrors the MPC cost in dense per-step form:

```
r(k) = α₁ · Δx̄₄(k) / Δx_{4,ref}
     - α₂ · (Σ u^n(k)) / W_{daily,ref}
     - α₃ · drought_term(k)
     - α₄ · ponding_at_sinks(k)
```

Plus a terminal bonus at harvest: +5 × x̄₄(T) / x_{4,ref}.

Same α weights as the MPC. This is intentional: identical objective, identical normalization. Differences in performance reflect differences in policy (MPC's planned-optimal vs RL's learned-policy), not differences in objective definition.

### 6.5 Budget constraint handling (soft penalty + early termination)

Three-tier handling, all soft:

1. **Hard cliff:** If Σu > W_total at any point, reward -= 100 and the episode terminates with `truncated = True`.
2. **Burn-rate shaping:** If (budget_used / W_total) > (day / season_length) + 0.2, reward -= 0.5 for that step. Penalizes spending faster than seasonal-uniform pace.
3. **Action clipping:** Inside `step()`, action is clipped to ensure Σu ≤ budget_remaining is never violated. This is a safety net, not a primary mechanism.

### 6.6 Reset randomization

- `scenario` ~ Uniform({'dry', 'moderate', 'wet'})
- `initial_x_1` = FC × Uniform(0.9, 1.1)
- `initial_x_5` = 0
- `forecast_noise_seed` = random
- `budget_total` ~ Uniform({100%, 85%, 70%}) of full irrigation need

The agent thus learns one policy that conditions on scenario, initial state, budget, and forecast — generalizing across configurations. This is more sample-efficient than training one policy per (scenario, budget) pair.

### 6.7 Episode length

93 days (rice season).

### 6.8 Algorithm: SAC

Soft Actor-Critic from Stable-Baselines3. Chosen because:

- Off-policy and sample-efficient compared to PPO/A2C.
- Entropy regularization prevents premature convergence under stochastic weather.
- Continuous action space, native fit.
- Well-studied for resource-allocation and constrained-control problems.

### 6.9 Network architecture (CTDE)

**Actor (shared parameters across agents):**

For each of the 130 agents:
- Input per agent: [5 per-agent features ‖ 10 global features] = 15 dims
- Hidden: [256, 256]
- Output: per-agent (mean, log_std), squashed via tanh and rescaled to [0, 1]

The actor network is applied 130 times with shared weights, treating each agent as a "copy of the same policy" conditioned on local features and shared global state. This is the centralized-training-decentralized-execution (CTDE) pattern from multi-agent RL.

**Critic (centralized):**
- Input: [flat state 660 ‖ flat action 130] = 790 dims
- Hidden: [512, 512, 256]
- Output: scalar Q

The critic sees the full global state and full joint action, enabling it to learn the budget-constraint shadow price.

### 6.10 Why parameter sharing matters

A naive 130-dim independent action head would require the actor to learn ~260 independent (mean, log_std) parameters per layer, ignoring spatial structure. Parameter sharing exploits the symmetry that "an agent at position γ with soil water x₁ should be irrigated according to a function of (γ, x₁) — not according to which absolute index it has." This is essential to make 130-agent SAC tractable on Kaggle compute budget.

### 6.11 Training

- 500k–2M environment steps per seed, with checkpointing every 10k steps to survive Kaggle session interruptions.
- 5 seeds per (scenario_pool, forecast_mode) configuration.
- Total compute target: ~30 training runs × ~3 GPU-hours = ~90 GPU-hours, fits within 3 weeks of Kaggle's 30 GPU-hours/week free tier.
- Learning rate 3e-4, batch_size 256, gamma 0.99, tau 0.005, ent_coef "auto".

---

## 7. Evaluation Framework

### 7.1 Six controllers

| ID | Name | Information access | Role |
|---|---|---|---|
| C1 | No-irrigation | None | Rainfed lower bound |
| C2 | Fixed-schedule | Budget total only | Linear-decay 19-event; farmer-style heuristic |
| C3 | MPC-perfect | Perfect 14-day forecast + full state | Information-rich upper bound |
| C4 | MPC-noisy | Noisy 7-day forecast + full state | Realistic deployment benchmark |
| C5 | SAC-no-forecast | Same-day observation + budget remaining | Fair-deployment RL |
| C6 | SAC-forecast | + 7-day rainfall forecast in observation | Equal-information RL |

### 7.2 Three scenarios × three budget levels

Selected from the 25-year NASA POWER record (full justification in `thesis (9).pdf`):

| Scenario | Year | Rice rainfall (mm) | Character |
|---|---|---|---|
| S1 Dry | 2022 | 39.7 | Budget-critical |
| S2 Moderate | 2020 | 42.1 | Standard test |
| S3 Wet | 2024 | 176.8 | Waterlogging risk |

| Budget level | Multiplier | Rice budget (mm) |
|---|---|---|
| 100% | 1.00 | 484 |
| 85% | 0.85 | 411 |
| 70% | 0.70 | 339 |

### 7.3 Fixed-schedule controller specification

19 irrigation events spaced every 5 days through the 93-day rice season. Linear-decay weights:

$$w_j = \frac{2(K - j + 1)}{K(K+1)}, \quad j = 1, ..., K, \quad K = 19$$

These weights sum to 1. Per-event amount: u_j = W_budget × w_j. Each event amount is clipped at UB × interval = 12 × 5 = 60 mm per agent for the 5-day interval (i.e., averaged over the 5 days, this is u_j / 5 mm/day). Distribution is uniform across all 130 agents.

This produces a front-loaded schedule: the first event is approximately 50 mm at the 100% budget, the last event approximately 2 mm. Reflects standard Gilan farmer practice.

### 7.4 Run grid

- 6 controllers
- 3 scenarios
- 3 budget levels
- For MPC: 2 horizons (H_p = 7, 14)
- For C3, C4, C5, C6: stochastic replicates per configuration

Stochastic-controller seeds:
- C4 (MPC-noisy): 5 seeds (forecast noise realizations)
- C5, C6 (SAC): 5 seeds (training seeds; evaluation per training seed uses 10 random episodes)

Approximate total:
- C1, C2 (deterministic): 9 runs each = 18 runs
- C3 (MPC-perfect, deterministic): 9 × 2 horizons = 18 runs
- C4 (MPC-noisy): 9 × 2 horizons × 5 seeds = 90 runs
- C5, C6 (SAC): 9 × 5 seeds × 10 eval episodes = 450 episodes (5 trained models per scenario_pool, evaluated 10 times each)

Estimated wall-clock:
- C1, C2: minutes
- C3, C4: ~3-5 seconds per MPC step × 93 days × 108 runs = ~5-15 hours
- C5, C6 training: ~90 GPU-hours on Kaggle
- C5, C6 evaluation: ~10 minutes total

### 7.5 Evaluation metrics

| Metric | Definition | Unit |
|---|---|---|
| Terminal biomass | x̄₄(T) | g/m² |
| Yield | x̄₄(T) × HI × 10 | kg/ha |
| Water use | (Σ_{k,n} u^n(k)) / N | mm per agent |
| WUE | yield / water_use | kg/(ha·mm) |
| Budget compliance | 1 if Σu ≤ W_total else 0 | binary |
| Drought days | (1/N) Σ_n Σ_k 𝟙[x₁^n(k) < ST] | days |
| Waterlog days | (1/N) Σ_n Σ_k 𝟙[x₁^n(k) > FC] | days |
| Sink ponding-days | Σ_{n∈𝒮} Σ_k 𝟙[x₅^n(k) > 5 mm] / |𝒮| | days |
| Spatial equity | std(x₄(T)) / mean(x₄(T)) | dimensionless |
| Mean solve time | (1/T) Σ_k τ_solve(k) | ms |
| Max solve time | max_k τ_solve(k) | ms |

### 7.6 Statistical tests

For each metric and each pair of controllers on identical scenario × budget configurations:
- Mann-Whitney U test (non-parametric, robust to non-normal residuals).
- Multiple-comparison correction: Bonferroni-Holm.
- Significance threshold p < 0.05 after correction.
- Reported in supplementary table; flagged in main text where significant.

### 7.7 Weight sensitivity (7 runs total)

All on dry scenario, 70% budget, rice, H_p = 14:

- α₂ ∈ {0.001, 0.01, 0.1} → 3 runs (α₃, α₄ at nominal)
- α₃ ∈ {0.03, 0.3} → 2 runs (α₂, α₄ at nominal)
- α₄ ∈ {0.17, 1.5} → 2 runs (α₂, α₃ at nominal)

Presented as one summary table in the thesis.

---

## 8. File and Module Structure

The project follows clean-code principles: files split when they have meaningfully different responsibilities, combined when they are aspects of one cohesive thing. Existing flat scripts are left at the project root; new infrastructure lives under `src/` and `controllers/`.

```
thesis-irrigation/
├── README.md
├── ARCHITECTURE.md                          # this document
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── (existing files — untouched)
│   ├── preprocess.py
│   ├── soil_data.py
│   ├── climate_data.py
│   ├── abm.py
│   ├── validate_physics_fao.py
│   ├── plot_*.py (8 files)
│   ├── run_plots.py
│   ├── gilan_farm.tif
│   └── POWER_..._LST.csv
│
├── (existing files — refactored to use src/terrain.py)
│   ├── cross_validate_gwetroot.py
│   ├── run_comparison.py
│   └── run_comparison_2.py
│
├── src/                                     # shared utilities
│   ├── __init__.py
│   ├── terrain.py                           # ✓ Step A — DEM, graph, sinks, hilltops
│   ├── persistence.py                       # ✓ Step A — parquet I/O, checkpointing
│   ├── runner.py                            # Step B — generic run_season()
│   └── precompute.py                        # Step C — cached thermal time, forecasts
│
├── controllers/
│   ├── __init__.py
│   ├── base.py                              # Step B — Controller abstract interface
│   ├── no_irrigation.py                     # Step B
│   ├── fixed_schedule.py                    # Step B — linear-decay 19-event
│   └── mpc/                                 # Step D
│       ├── __init__.py
│       ├── mpc.py                           # solver setup, runner
│       ├── mpc_dynamics_sym.py              # CasADi symbolic ABM
│       ├── mpc_cost.py                      # 5 cost terms + combine
│       └── mpc_smoothing.py                 # soft_max, soft_min, soft_clip
│
├── rl/                                      # Step E
│   ├── __init__.py
│   ├── gym_env.py                           # Gymnasium wrapper
│   ├── networks.py                          # shared actor + centralized critic
│   ├── train.py                             # SB3 SAC training with checkpointing
│   └── runner.py                            # inference using trained model
│
├── scripts/
│   ├── experiments/
│   │   ├── exp_no_irrigation.py             # Step B
│   │   ├── exp_fixed_schedule.py            # Step B
│   │   ├── exp_mpc.py                       # Step D
│   │   ├── exp_train_sac.py                 # Step E (also Kaggle notebook)
│   │   └── exp_eval_sac.py                  # Step E
│   ├── grids/
│   │   ├── run_main_grid.py                 # Step F — full 6×3×3 sweep
│   │   └── run_weight_sensitivity.py        # Step F — 7 sensitivity runs
│   └── analysis/
│       ├── aggregate_results.py             # Step F — load all parquets → tables
│       └── generate_figures.py              # Step F — produce all thesis figures
│
├── notebooks/
│   ├── kaggle_train_sac.ipynb               # Step E — for Kaggle GPU
│   └── exploration.ipynb                    # ad-hoc analysis
│
├── results/                                 # outputs (gitignored)
│   ├── preprocessing/                       # already exists
│   ├── crossval/                            # already exists
│   ├── comparison/                          # already exists
│   ├── runs/                                # NEW — parquet outputs
│   │   └── partial/                         # crash-recovery checkpoints
│   ├── precomputed/                         # NEW — cached thermal time, forecasts
│   ├── checkpoints/                         # NEW — SAC model checkpoints
│   ├── figures/                             # NEW — generated thesis figures
│   └── tables/                              # NEW — CSV tables for thesis
│
└── tests/
    ├── unit/
    │   ├── test_terrain.py
    │   ├── test_persistence.py
    │   ├── test_runner.py
    │   ├── test_fixed_schedule.py
    │   └── test_mpc_cost.py
    └── integration/
        ├── test_no_irrigation_short_run.py
        ├── test_mpc_short_run.py
        ├── test_sac_one_episode.py
        └── test_crash_recovery.py
```

Roughly 35–40 source files when complete. Each file is small enough to read in one screen but substantial enough to feel cohesive.

### 8.1 Why this layout

- **`src/`** holds infrastructure shared by all controllers and experiments — terrain, persistence, the runner, precomputation. Things you want to import from many places.
- **`controllers/`** holds the controller implementations. Each controller is one file (or one folder if it has helpers, like MPC). All implement `controllers.base.Controller`.
- **`rl/`** is sibling to controllers because the RL agent has training infrastructure (networks, train loop) that is fundamentally different in shape from a controller's `step()` method. The trained SAC policy itself implements `Controller`, but the training and evaluation code is RL-specific.
- **`scripts/`** holds the things you actually run. They are thin: load configs, call into `src/`, save results.
- **`tests/`** mirrors `src/` structure. `unit/` for in-isolation tests; `integration/` for end-to-end short runs.

### 8.2 Existing files left at the project root

Existing scripts (`preprocess.py`, `plot_*.py`, etc.) remain at the project root because:
1. They work today and are run individually.
2. Moving them under `src/` would change their import paths and break the user's muscle memory.
3. There is no benefit to relocating them — they are not imported by other code.

The new infrastructure imports `abm`, `soil_data`, `climate_data` from the project root in the standard Python style.

---

## 9. Persistence and Caching

### 9.1 Result file format: parquet (long format) + JSON sidecar

Each completed run produces two files:

- `results/runs/{controller}_{scenario}_{budget}_{seed}.parquet` — long-format trajectory.
- `results/runs/{controller}_{scenario}_{budget}_{seed}.json` — metadata sidecar.

**Long-format schema:** one row per (day, agent) pair. For a 93-day rice run with 130 agents: 12,090 rows. Columns:

| Column | Type | Source |
|---|---|---|
| day | int | 0..92 |
| agent | int | 0..129 |
| x1 | float | soil water (mm) |
| x2 | float | thermal time (°C·day) |
| x3 | float | maturity stress (dimensionless) |
| x4 | float | biomass (g/m²) |
| x5 | float | surface ponding (mm) |
| u | float | irrigation applied (mm) |
| rainfall | float | climate rainfall that day (broadcast) |
| et0 | float | climate ET₀ that day (broadcast) |
| budget_remaining | float | budget after this day (broadcast) |

**Why long format:**
- Standard pandas idiom: `df.groupby('day')['x4'].mean()` for time series, `df[df.day == 92].pivot()` for spatial maps.
- Trivially concatenable across runs: `pd.concat([load_run(f) for f in files])` for cross-run analysis.
- Native input to seaborn and plotnine.
- Parquet's columnar compression handles the broadcast duplication; file size is ~50 KB per run.

**Metadata schema:** JSON with at minimum:
- `scenario` (string)
- `year` (int)
- `crop` (string)
- `controller` (string)
- `budget_total` (float, mm)
- `seed` (int)
- `solve_times` (list[float], optional, for MPC)
- `final_metrics` (dict)
- `config_snapshot` (dict, the YAML that produced this run)
- `wallclock_seconds` (float)
- `completed_at` (ISO timestamp, added automatically)

### 9.2 Skip-if-exists logic

`should_skip(filepath, force=False)` returns True if the parquet file exists and we are not forcing recomputation. Used by all experiment scripts:

```python
output_path = f"results/runs/mpc_perfect_dry_70_seed0.parquet"
if should_skip(output_path):
    print("Already done, skipping")
    continue
# ... run experiment ...
save_run(output_path, trajectory, metadata)
```

This means re-running `scripts/grids/run_main_grid.py` is idempotent: it picks up where it left off, computing only the missing runs. Critical for crash recovery and incremental development.

### 9.3 Mid-run checkpointing for MPC

For long MPC runs (93 daily solves, total ~5 minutes per run), a partial checkpoint is saved every 10 days as `{controller}_{scenario}_..._partial.parquet`. The runner detects an existing partial file at startup and resumes from `last_completed_day + 1` with the same RNG state. On successful completion of the full season, the partial file is discarded and the final file written.

### 9.4 SAC training checkpointing

Stable-Baselines3 has built-in `CheckpointCallback` infrastructure. Configured to save model + replay buffer every 10k environment steps, surviving Kaggle's 9-hour session limit. Resume by loading the latest checkpoint from `results/checkpoints/`.

### 9.5 Precomputation cache

Functions of weather alone (thermal time, growth function evaluations, noise realizations) are computed once and cached under `results/precomputed/`. The cache key is `{quantity}_{scenario}_{crop}.npz`. Generated by `scripts/preprocess/03_precompute_thermal.py` and `04_precompute_forecasts.py`. Subsequent reads are O(1) from disk.

### 9.6 No automatic cache invalidation

By design, the system does not detect parameter changes that would invalidate cached results. If you change `α₂` in the YAML, the existing parquet files do not become stale — they were correct under the *old* α₂. To re-run with new parameters, either delete the relevant files or pass `--force` to the script.

This keeps the caching logic simple. The student is responsible for knowing what they changed and re-running accordingly.

---

## 10. Build Plan

The plan is organized as a sequence of vertical slices. Each slice produces a working end-to-end capability before the next slice begins.

### Step A — Foundation utilities (✓ COMPLETE)

**Files:** `src/terrain.py`, `src/persistence.py`, `src/__init__.py`. Refactored: `cross_validate_gwetroot.py`, `run_comparison.py`, `run_comparison_2.py` (deduplicated graph builder).

**Outcome:** Single source of truth for terrain and graph; parquet I/O ready; existing scripts unchanged in behavior but no longer duplicate code.

### Step B — Controller interface and runner (~1 day)

**Files:**
- `controllers/base.py` — abstract `Controller` with `reset()` and `step(state, day, budget_remaining, climate_today, forecast) -> action`
- `controllers/no_irrigation.py` — trivial implementation returning zeros
- `controllers/fixed_schedule.py` — linear-decay 19-event scheduler
- `src/runner.py` — `run_season(controller, abm, climate, budget_total, output_path, ...)` returning a trajectory dict and saving parquet+JSON
- `scripts/experiments/exp_no_irrigation.py` — CLI: `--scenario dry --crop rice`
- `scripts/experiments/exp_fixed_schedule.py` — CLI: `--scenario dry --crop rice --budget 70`

**At this point:** Both baselines work end-to-end. Output is queryable parquet. Resuming after crash works. Comparison plots can be generated from cached data.

**Also at this point:** Switch to crop-as-parameter. This means small changes to `climate_data.py` (extract_scenario takes a crop dict) and to `cross_validate_gwetroot.py` (already imports RICE explicitly). No edit to `soil_data.py` itself except removing the `theta = RICE` line at the bottom.

### Step C — Precomputation (~half day)

**Files:**
- `src/precompute.py` — functions to compute thermal time x₂(k), heat stress h₂(k), low-temp stress h₇(k), growth g(x₂(k)) per (scenario, crop) and forecast noise realizations per (scenario, seed)
- `scripts/preprocess/03_precompute_thermal.py` — runs precomputation, saves to `results/precomputed/`
- `scripts/preprocess/04_precompute_forecasts.py` — generates noise realizations

**At this point:** The MPC dynamics layer can read these as time-varying parameters rather than recomputing. Speedup of ~30% expected on solve time.

### Step D — MPC controller (~1 week)

**Files:**
- `controllers/mpc/mpc_dynamics_sym.py` — CasADi symbolic version of the ABM step
- `controllers/mpc/mpc_cost.py` — five cost terms and combiner
- `controllers/mpc/mpc_smoothing.py` — soft_max, soft_min, soft_clip
- `controllers/mpc/mpc.py` — NLP builder, IPOPT options, warm starting, runner
- `scripts/experiments/exp_mpc.py` — CLI: `--scenario dry --crop rice --budget 70 --horizon 14 --forecast perfect`

**Validation:** First run is a 30-day mini-MPC on dry scenario with H_p = 7. Compare its first-day actions to a hand-calculated "pour-water-on-driest-agent" heuristic to sanity-check. Then run full season at H_p = 14 and verify budget compliance.

**At this point:** All deterministic baselines (C1–C3) and stochastic MPC (C4) are working.

### Step E — RL infrastructure (~1 week + compute)

**Files:**
- `rl/gym_env.py` — Gymnasium wrapper around `CropSoilABM`. Uses `src/runner.py` underneath.
- `rl/networks.py` — shared actor + centralized critic
- `rl/train.py` — SAC training with SB3, checkpointing, scenario randomization
- `rl/runner.py` — inference: load checkpoint, instantiate `Controller`-compatible policy, plug into `run_season`
- `notebooks/kaggle_train_sac.ipynb` — Kaggle-deployable training notebook
- `scripts/experiments/exp_train_sac.py` — local training (laptop, slower)
- `scripts/experiments/exp_eval_sac.py` — load trained model, run evaluation

**Validation:** Random-policy baseline on the gym env to verify the wrapper correctness (no NaN in observations, episode ends cleanly, reward signs are right). Then short SAC training (50k steps) on dry scenario to confirm training pipeline. Then full training on Kaggle.

**At this point:** All six controllers work. The infrastructure for the main experimental grid is complete.

### Step F — Aggregation and figures (~1 week)

**Files:**
- `scripts/grids/run_main_grid.py` — sweep all configurations, skip-if-exists, generate all run files
- `scripts/grids/run_weight_sensitivity.py` — 7 sensitivity runs
- `scripts/analysis/aggregate_results.py` — load all run files, compute metrics, statistical tests, build summary tables → `results/tables/*.csv`
- `scripts/analysis/generate_figures.py` — produce all thesis figures from cached results → `results/figures/*.pdf`

**At this point:** The thesis Chapter 5 (Results) can be written from cached data. Re-generating any figure is fast.

### Estimated total wall-clock

- Steps A–C: ~3 days active development
- Step D: ~1 week active development + ~10 hours of MPC runs
- Step E: ~1 week active development + ~3 weeks of Kaggle GPU
- Step F: ~1 week of analysis and writing

**Total: ~5–6 weeks active engineering, with RL training running in background.**

---

## 11. Risks and Mitigations

| ID | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | IPOPT fails to converge on non-smooth kinks (SCS runoff, stress max/min) | Medium | High | Smooth approximations (max(x,0) ≈ 0.5(x+√(x²+ε²))); start with native fmax/fmin |
| R2 | MPC solve time exceeds 30 s per step | Low | Medium | Move-blocking (H_c < H_p) as fallback; reduce H_p to 7 only |
| R3 | SAC fails to learn (plateaus near fixed-schedule level) | Medium | High | Lagrangian SAC upgrade; reduce action dim via elevation-zone aggregation |
| R4 | Kaggle GPU quota exhausted | Medium | Medium | Reduce to 3 seeds; consider Colab Pro |
| R5 | MPC and RL both fail under wet scenario (waterlogging) | Medium | Medium | Increase α₄ specifically for wet scenario; document sensitivity |
| R6 | Thesis deadline pressure | Medium | High | Prioritize comparison A (no-forecast RL); defer comparison B if time-constrained |
| R7 | Weight choices distort comparison | Medium | Medium | Sensitivity sweep documents robustness; sweep at three α₂ values minimum |
| R8 | Long-run laptop crash mid-simulation | Medium | Low | Mid-run checkpointing every 10 days; skip-if-exists makes resumption automatic |
| R9 | Reviewer questions lack of field data | High | Low | Cross-validation against NASA satellite (already done, r=0.74) and FAO-56 unit test (already done) suffice |

---

## 12. Open Questions

These are deferred and will be resolved as the implementation proceeds.

- **Final α weights.** Placeholder values are used until the student provides exact rice and water prices for Gilan. Will be revised before main grid runs.
- **MPC mid-run failure handling.** If IPOPT returns INFEASIBLE on a particular day, what should the controller do? Default proposal: fall back to previous day's action shifted forward. This requires testing.
- **SAC hyperparameter tuning depth.** SB3 defaults are used initially. If learning is unstable, ent_coef and learning_rate may need targeted tuning. Budget: 1–2 days of tuning if needed.
- **Tobacco inclusion (Phase 2).** Decision deferred until rice MPC and rice SAC are both demonstrably working.
- **Initial state randomization width.** Currently FC × Uniform(0.9, 1.1). May tighten to Uniform(0.95, 1.05) if it adds too much variance to evaluation.

---

## Appendix A — Locked Decisions

This is a single-source-of-truth summary table of every architectural decision made during the design phase.

| Item | Locked value | Rationale |
|---|---|---|
| Solver | CasADi + IPOPT (MUMPS backend) | Classical guarantees; faster wall-clock than PGD on this problem size |
| MPC formulation | Centralized sparse NLP, multiple-shooting | Cascade DAG provides natural sparsity |
| Decision variables | u ∈ ℝ^{N×H_p} = ℝ^{1820 max} | All 130 agents, full horizon |
| Cost function | 5 terms (terminal biomass, water, drought path, sink ponding path, Δu reg) | Each O(1) normalized; matches RL reward |
| Constraints | Box on u, global budget linear, dynamics equality | Budget inside solver, not post-hoc |
| Smoothing | CasADi native fmax/fmin first; smooth approximations as fallback | ε = 0.01 if needed |
| Horizons | H_p ∈ {7, 14}, H_c = H_p (no move-blocking initially) | Sensitivity analysis on horizon |
| Forecast modes | Perfect, noisy (σ_j = 0.15√j on rainfall + ET₀) | Two studies: information-equal and deployment-realistic |
| RL algorithm | SAC via Stable-Baselines3 | Off-policy, entropy-regularized, sample-efficient |
| RL action space | Box(0, 1, shape=(N,)) — fraction of UB per agent | Matches MPC decision exactly |
| RL actor | Shared MLP across 130 agents, 5 + 10 input dims, [256, 256] | CTDE pattern; parameter sharing exploits spatial symmetry |
| RL critic | Centralized MLP, 660 + 130 input dims, [512, 512, 256] | Sees full state and joint action |
| RL budget handling | Soft penalty + early termination + burn-rate shaping + safety clip | Three-tier handling |
| RL training | 500k–2M steps, 5 seeds per config, on Kaggle GPU | ~90 GPU-hours total |
| Crops | Rice (Hashemi) only Phase 1; tobacco deferred | Single-crop story is defensible; reduces experimental load |
| UB (irrigation cap) | 12 mm/day per agent | FAO-56 compatible; consistent with deficit-irrigation framing |
| Initial x₁ | FC × θ₅ = 140 mm for all rice runs | Post-puddling drainage equilibrium; agronomically realistic |
| Initial x₅ | 0 mm | Mountainous terrain; no puddling pulse on slopes |
| Budget levels | {100%, 85%, 70%} of full irrigation need | 50% rejected as too harsh for rice |
| Scenarios | Dry 2022, moderate 2020, wet 2024 | From 25-year NASA POWER record |
| Fixed schedule | 19 events every 5 days, linear-decay weights, sums to W_budget | Front-loaded, farmer-realistic |
| Result format | Long-format parquet + JSON metadata sidecar | Standard pandas idiom; enables cross-run analysis |
| Skip logic | File-existence check, no hashing | Simple and predictable |
| Crop selection | Function parameter (not module toggle) | CLI flag works cleanly; no muscle-memory anxiety |
| Crop runner-time scope | Phase 1 = rice only | Tobacco decision deferred until MPC + SAC demonstrably work |

---

## Appendix B — Parameter Tables

### B.1 Soil parameters (silty loam, FAO-56 Table 19)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Water uptake coefficient | θ₁ | 0.096 | Lopez-Jimenez et al. 2024 |
| Wilting point (volumetric) | θ₂ | 0.15 | FAO-56 Table 19 |
| SCS initial abstraction | θ₃ | 10.0 mm | Lopez-Jimenez et al. 2024 |
| Drainage coefficient | θ₄ | 0.05 | Lopez-Jimenez et al. 2024 |
| Field capacity (volumetric) | θ₆ | 0.35 | FAO-56 Table 19 |
| Saturation (volumetric) | θ_sat | 0.55 | Saxton & Rawls 2006 |

### B.2 Rice parameters (Hashemi cultivar, Gilan)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Root depth | θ₅ | 400 mm | FAO-56 Table 22 |
| Base temperature | θ₇ | 10°C | Standard GDD convention |
| Heat stress onset | θ₉ | 35°C | — |
| Extreme heat threshold | θ₁₀ | 42°C | — |
| Heat stress maturity rate | θ₁₁ | 0.0030 | Lopez-Jimenez et al. 2024 |
| Drought stress maturity rate | θ₁₂ | 0.0030 | Lopez-Jimenez et al. 2024 |
| RUE (g DM / MJ incident solar) | θ₁₃ | 0.65 | Calibrated; Patel et al. 2013 (1.3 g/MJ on intercepted PAR) |
| Drought sensitivity | θ₁₄ | 0.80 | Lopez-Jimenez et al. 2024 |
| Maturity GDD | θ₁₈ | 1250 °C·day | Sadidi Shal et al. 2021 |
| Max interception | θ₁₉ | 0.95 | Lopez-Jimenez et al. 2024 |
| 50%-interception GDD | θ₂₀ | 417 °C·day | Calibrated |
| Crop coefficient | K_c | 1.15 | FAO-56 Table 12 |
| Depletion fraction | p | 0.20 | FAO-56 Table 22 |
| Harvest index | HI | 0.42 | Yoshida 1981 |
| Nursery GDD | x₂_init | 210 °C·day | 20-day nursery + 5°C greenhouse |
| Initial biomass | x₄_init | 60 g/m² | 25 hills/m² × 2.5 g/seedling |
| Season start DOY | — | 141 (May 20) | Gilan agricultural office |
| Season end DOY | — | 233 (Aug 20) | + 93 days |

### B.3 Derived rice quantities

| Quantity | Formula | Value |
|---|---|---|
| FC × θ₅ | Field capacity (root zone, mm) | 140 mm |
| WP × θ₅ | Wilting point (root zone, mm) | 60 mm |
| TAW | FC − WP | 80 mm |
| RAW | p × TAW | 16 mm |
| Stress threshold ST | FC − RAW | 124 mm |
| Saturation × θ₅ | Saturation (root zone, mm) | 220 mm |

### B.4 MPC weights (placeholder — to be revised after price data)

| Weight | Value | Description |
|---|---|---|
| α₁ | 1.0 | Terminal biomass (anchor) |
| α₂ | 0.01 | Water cost (economic estimate) |
| α₃ | 0.1 | Drought regularization |
| α₄ | 0.5 | Sink ponding penalty |
| α₅ | 0.005 | Δu numerical regularization |

### B.5 Budget calculations

Based on 25-year mean ET₀ = 5.02 mm/day for Gilan:
- Full crop demand: ETc = 1.15 × 5.02 × 93 = 537 mm
- Mean season rainfall: 53 mm
- Full irrigation need: max(537 − 53, 0) = 484 mm

| Budget level | Multiplier | Budget (mm) |
|---|---|---|
| 100% | 1.00 | 484 |
| 85% | 0.85 | 411 |
| 70% | 0.70 | 339 |

---

## Document Status

This is the canonical architecture document for the thesis implementation. It supersedes:
- The blueprint document (`thesis_blueprint__1_.pdf`)
- Gemini's prior analyses (`Gemini_s_responses.txt`, `mpc__1_.tex`)
- Previous TODO list (`todo.txt`)

When implementation decisions are made that diverge from this document, this document is updated. The locked-decisions table in Appendix A is the single source of truth for the canonical configuration.
