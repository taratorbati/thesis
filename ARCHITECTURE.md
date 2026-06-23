# Thesis Implementation Architecture
## MPC and RL Controllers for Constrained Irrigation in Topographically Heterogeneous Terrain

**Author:** Tara Torbati, ITMO University, R4237c
**Supervisor:** Peregudin A. A.

---

## 1. Summary

This thesis asks whether reinforcement-learning agents (SAC and TD3) can match a
Model Predictive Controller on constrained irrigation of a topographically
heterogeneous rice field. The plant is a 130-cell crop-soil ABM with cascade
water routing and surface-ponding dynamics, cross-validated against NASA GWETROOT
(Pearson r = 0.74 on the dry year). All controllers are evaluated on the same
held-out cells through one shared closed-loop runner.

## 2. System layers

1. **Environment** (`src/model/`) — the ABM (`src/model/abm.py`), terrain graph
   (`src/model/terrain.py`), climate (`src/model/climate_data.py`), and
   precomputed biology/forecasts (`src/sim/precompute.py`, `src/model/forecast.py`).
2. **Controllers** — all implement `src/controllers/base.Controller`:
   no-irrigation, fixed-schedule, reactive-schedule, MPC (`src/mpc/`), and RL
   (`src/rl/`).
3. **Runner** — `src/sim/runner.run_season` drives every controller through an
   identical loop (cascade routing, initial x1 = FC, field-averaged budget).

## 3. Year split (`src/model/climate_data.py`)

| Set | Years | Use |
|---|---|---|
| Train | 20 years (2000-2025 minus dev/test) | sampled uniformly per training episode |
| Dev | 2002, 2016, 2023 | deterministic best-model selection during training |
| Test | 2022 (dry), 2018 (moderate), 2024 (wet) | final evaluation only |

Evaluation grid: 3 test years x 3 budgets {70%, 85%, 100%} = 9 cells.

## 4. MPC formulation (`src/mpc/`)

- **Variables:** u in R^(N x Hp), N = 130, Hp in {8, 14}.
- **Shooting states:** x1 (root-zone water), x5 (surface ponding); x2 precomputed;
  x3, x4 tracked from the true state.
- **Cost (5 terms, all O(1) normalised):** terminal biomass (Mayer), water,
  drought, sink ponding, and delta-u, with weights alpha1=1.0, alpha2=0.016,
  alpha3=0.1, alpha4=0.5, alpha5=0.005.
- **Constraints:** box [0, 12 mm], seasonal budget (linear), dynamics (equality).
- **Solver:** CasADi + IPOPT (MUMPS), smooth approximations, tol 1e-4.

## 5. RL formulation (`src/rl/`)

### Observation (agent-major, flat)
- **Per-cell block, 8 x 130 = 1040:** x1_norm, x5_norm, x4_norm, x3 (dynamic);
  elev_norm, Nr_norm, Nr_internal_norm, n_upstream_norm (static topography).
- **Global block:** 4 scalars (day_frac, budget_frac, budget_total_norm,
  burn_rate) + 48 forecast values (rain, ETc, radiation, h2, h7, g_base over an
  8-day horizon).
- **Today's weather is fed once**, via forecast day 0. `dedupe_today_weather`
  (default True) drops the redundant today-scalars -> 52-dim global / 1092-dim
  obs. The legacy layout (today repeated; 57-dim global / 1097-dim) is still
  loadable: the networks infer dims from the obs width and the runner
  auto-detects the layout.

### Action
`Box(0, 1)^130`, scaled to [0, 12] mm/day and clipped to the remaining budget.

### Networks (`networks.py`, `networks_td3.py`)
- **Actor:** one parameter-shared MLP over all cells (each cell sees its local
  features + the broadcast global block, re-centred to ~[-1, 1], LeakyReLU).
  SAC uses (mu, log_std) squashed-Gaussian heads; TD3 uses a deterministic
  mu + tanh head.
- **Critic (shared by both):** twin VDN — Q_total = sum_n Q_local(s_n, g, a_n),
  with LayerNorm after each hidden layer to suppress value divergence.

### Reward
r1 biomass increment, r2 water cost, r3 drought-stress, r6 field-capacity
overshoot (linear, aligned with the ABM's waterlog term). TD3 adds r5
(control-rate smoothing, mirroring MPC term 5) and an additive terminal-yield
bonus.

### The two chosen controllers
- **SAC** (`train_sac.py`): ent_coef = 0.002, asymmetric actor LR (x5),
  two-phase exploration noise (anneal then late re-injection), 250k steps.
- **TD3** (`train_td3.py`): exact n-step returns (n = 5) via the model-gamma
  trick, policy_delay = 2, target-policy smoothing, learning_starts = 50k,
  r5 = 0.005, terminal-yield = 1.0, 250k steps.

Both select the best model on the dev set via `FixedScheduleEvalCallback`, so the
SAC-vs-TD3 comparison is apples-to-apples.

## 6. Evaluation

Each controller is run over the 9 cells with perfect and AR(1)-noisy forecasts.
Metrics: yield (kg/ha), water-use efficiency, budget compliance, drought days,
wet-year waterlog days, control smoothness (mean |delta u|), and solve/inference
time. Significance via Mann-Whitney U across seeds.

## 7. Key files

| File | Role |
|---|---|
| `src/model/abm.py` | Ground-truth crop-soil ABM |
| `src/model/climate_data.py` | Train/dev/test year split and scenario loader |
| `src/rl/gym_env.py` | Gymnasium env, observation/reward (dedup flag) |
| `src/rl/networks.py` | `SharedActor`, `VdnCritic`, `SacVdnPolicy` |
| `src/rl/networks_td3.py` | `DeterministicSharedActor`, `Td3VdnPolicy` |
| `src/rl/train_sac.py`, `src/rl/train_td3.py` | The two chosen trainers |
| `src/rl/runner.py` | Inference runner (auto-detects SAC/TD3 + obs layout) |
| `src/rl/common.py` | Shared SB3 helpers (LR schedule, callbacks, LR-asymmetric algorithms) |
| `src/rl/nstep_buffer.py` | Exact n-step replay buffer (TD3) |
| `src/mpc/controller.py`, `src/mpc/dynamics_sym.py` | MPC controller and CasADi dynamics |
| `src/model/forecast.py` | Perfect and AR(1)-noisy forecasts |
| `src/controllers/base.py` | Abstract Controller interface |
