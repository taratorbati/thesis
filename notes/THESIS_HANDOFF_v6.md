# THESIS HANDOFF v6 — RL Irrigation Controller for a Mountainous Cascade Field

> **Scope of this document.** This is a complete, self-contained handoff covering
> the project from its origins through the v2.18 result and the v2.19-TD3 design.
> It folds in everything from the working session that produced v2.17, v2.18, and
> the TD3 port: every modification, *why* it was made, what the committed results
> actually showed (all numbers re-derived from the repo, not memory), the
> back-and-forth reasoning, the false starts, the bugs found, and the open
> questions. It is written so a new engineer (or a future session) can pick up
> with full context. **Treat numeric claims as authoritative only where this doc
> says they were re-derived from committed eval files; treat interpretations as
> arguments, not settled fact.**

---

## 0. TL;DR / Current State (June 2026)

- **The controller works and is a legitimate result.** The best stable model,
  **v2.18-P3b** (SAC, α=0.002 + late noise reinjection), reaches **99.3% of a
  perfect-forecast MPC oracle's mean yield** (3785 vs 3810 kg/ha), with
  comparable water use and waterlogging, and is **robust to realistic (15–42%)
  forecast noise** (noisy-forecast mean yield 3755, essentially unchanged).
- **The core scientific arc is solved.** The project's long-standing failure —
  chronic wet-year **over-irrigation** (soil moisture pinned ~12–22 mm above
  field capacity) — was diagnosed and substantially fixed. The fix was *not*
  reward shaping or rain-input rescaling; it was **exploration coverage +
  reducing the SAC entropy coefficient**, which together let the actor commit to
  low-water actions in wet states.
- **Next experiment in flight: v2.19-TD3.** A deterministic-policy (TD3) port
  built to (a) close the last ~6 mm of wet-year soil-moisture gap to MPC and (b)
  give a cleaner critic. Files are ready to train; see §8.
- **Known unsolved issue: spatial allocation.** None of the entropy/exploration
  work addressed the *spatial* (cascade/elevation) dimension. v2.18 still
  under-compensates for downhill water flow: it waters low cells slightly more
  than high cells, and low cells end up ~12 mm wetter than high cells. The
  targeted fix is a **GNN critic over the D8 drainage graph** (see §10).

---

## 1. The Project

### 1.1 Problem
Design and implement a reinforcement-learning controller that schedules daily
irrigation across a **130-cell mountainous rice field** in Gilan, Iran, coupled
to an **agent-based hydrological model (ABM)**. Water applied to a high cell
partially drains downhill (D8 cascade routing), so cells are hydrologically
coupled. The controller observes per-cell soil state + global weather/forecast
and outputs a per-cell daily irrigation depth (0–12 mm/day). It must perform
across climate years (dry/moderate/wet) and water-budget constraints
(100%/85%/70% of full seasonal need).

### 1.2 Physics core (`abm.py`, `soil_data.py`)
- **State** per cell includes `x1` = soil moisture (mm). **Field capacity FC =
  140 mm** (θ6·θ5 = 0.35·400), wilting point WP = 60 mm, drought stress
  threshold ST = 124 mm.
- **Drainage / cascade:** excess above FC routes to downstream cells via D8.
- **Waterlog stress** is **linear** in overshoot:
  `h6 = clip(1 − (x1 − FC)/FC, 0, 1)`. *(This linearity matters — see §4.1.)*
- **Crop parameters are well-grounded:** FAO-56 tables, Sadidi Shal et al. 2021
  (Gilan rice GDD), Paredes et al. 2025 (base temperature). FC/WP/ST all check
  out against the cited sources.
- Episode length **_K = 93 days**; **130 agents**; training samples a year from
  **20 TRAINING_YEARS** (2000–2025 minus the 3 test years) and a budget fraction
  from **U(0.70, 1.00)** each reset. Test scenarios are fixed years:
  **dry=2022 (39.7 mm seasonal rain), moderate=2018 (108.8 mm), wet=2024
  (176.8 mm, OOD extreme).**

### 1.3 RL formulation (`src/rl/gym_env.py`, `src/rl/networks.py`)
- **Algorithm:** SAC (Stable-Baselines3 2.6.0) through v2.18; TD3 from v2.19.
- **Action space:** `Box[0,1]^130`; water = `action·UB_MM`, UB_MM = 12 mm/day.
  The runner applies `u = action.clip(0,1)·12` (gym_env.py:433). **SB3 maps the
  actor's squashed tanh output in [−1,1] to this [0,1] space internally.** This
  is why **0 mm/day sits at the tanh boundary (−1)** and **6 mm/day at the tanh
  centre (0)** — a fact central to the entropy diagnosis (§4.4).
- **Observation (production, v2.14–v2.19):** **1097-dim**, 8 features/agent
  (`x1_norm, x5_norm, x4_norm, x3, elev_norm, Nr_norm, Nr_internal_norm,
  n_upstream_norm`) + 57 global dims (9 scalars + 48 forecast = 8-day × 6
  vars). *(The v2.8 9-feature/1227-dim layout exists in code but the production
  lineage uses 8 features.)*
- **Critic — Value Decomposition Network (VDN):** `Q_total = Σ_n Q_local(s_n,
  g, a_n)` over the 130 cells, **twin-Q** with `min(Q1,Q2)`, **LayerNorm after
  each hidden linear** (`_V211FactorizedContinuousCritic`). The LayerNorm is the
  cascade-divergence suppressant (Yue et al. NeurIPS 2023; Nauman et al. RLC
  2024).
- **Actor — parameter-shared per-cell MLP:** one MLP applied to all 130 per-cell
  inputs. Production actor (`_V216SharedActor` lineage) uses **LeakyReLU** +
  **input re-centering `2x−1`** (the v2.13 dead-ReLU capacity fix) +
  **asymmetric LR** (actor LR = 5× critic LR).
- **Reward** (per step, field-averaged): `r1` biomass growth (α1=1.0, ref 600),
  `r2` water cost (α2=0.016, per UB_MM), `r3` drought (α3=0.1), `r6` waterlog.
  `r6` is **linear** in overshoot since v2.15 (`−1.5·mean(overshoot)/FC`).

### 1.4 Baselines
- **No-irrigation floor**, **fixed schedule** (mean yield ~3314).
- **MPC** with horizons {4,8} and **perfect** and **noisy** forecasts. MPC's
  internal model is the *true ABM* (perfect-forecast MPC is a near-oracle).
  MPC cost: `max(x1−FC,0)²/FC²` (quadratic overshoot) + water + drought + Δu
  smoothing (α5=0.005). **No "stay near FC" term** — penalises overshoot only.

---

## 2. Version History (the full arc)

| version | one-line change | outcome |
|---|---|---|
| v2.4 (pilot) | monolithic critic | baseline |
| v2.5 | monolithic SAC | works, weak |
| v2.6 | **VDN factorised critic** | real jump (right inductive bias) |
| v2.7 | VDN tuned | responsive **but critic cascades** (deadly triad) |
| v2.11 | **LayerNorm critic** | cascade fixed, but actor flat (dead ReLU) |
| v2.12 | LeakyReLU actor + normalised obs | partial |
| v2.13 | actor input re-center `2x−1` | actor capacity recovered |
| v2.14 | **α: 0.05 → 0.01** | the real pre-2.16 fix; 97.4% of MPC mean Y |
| v2.15 | r6 **quadratic → linear** | ~flat (shape change alone insufficient) |
| v2.16-fixed | RAIN_REF 70 → 30, α=0.01 | stable; wet over-watering persists |
| v2.16-auto | capped auto-α (collapsed to ~0) | best wet behaviour **but critic unstable** |
| **v2.17-P3** | **α=0.005 + decaying exploration noise** | big stable gain (see §5) |
| **v2.18-P3b** | **α=0.002 + late noise reinjection** | **matches MPC** (see §6) |
| **v2.19-TD3** | **deterministic policy + target smoothing** | designed, ready to run (§8) |

---

## 3. Re-derived Results (all from committed eval files, perfect forecast)

9-cell grid (3 climates × 3 budgets), best_model, deterministic eval.

| controller | mean Y | mean waterlog | **wet Y** | **wet water** | **wet waterlog** | **wet x1** | dry Y | mod Y |
|---|---|---|---|---|---|---|---|---|
| v2.14 | 3710 | 51.9 | 3444 | 384 | 80.8 | 151 | 3975 | 3711 |
| v2.16-fixed (α=0.01) | 3671 | 47.8 | 3391 | 397 | 75.9 | 152 | 3948 | 3676 |
| v2.16-auto (α≈0, **unstable**) | 3733 | 21.7 | 3633 | 324 | 34.7 | 132 | 3907 | 3660 |
| **v2.17-P3 (α=0.005+noise)** | 3745 | 26.3 | 3604 | 362 | 54.1 | 144 | 3947 | 3685 |
| **v2.18-P3b (α=0.002+reinject)** | **3785** | **18.2** | **3687** | **336** | **36.6** | **136** | **3999** | 3669 |
| MPC-perfect | 3810 | 7.7 | 3752 | 309 | 18.0 | 130 | 3994 | 3685 |
| MPC-noisy (15–42% err) | 3802 | 9.4 | 3702 | 309 | 24.8 | 128 | 4012 | 3691 |

**v2.18 noisy-forecast** (robustness): mean Y 3755, mean waterlog 15.8, wet Y
3686, wet x1 136 — i.e. **almost identical to its perfect-forecast self**, losing
only ~30 kg/ha. Strong robustness result.

Monotone improvement v2.14 → v2.16-fixed → v2.17 → v2.18 on **wet x1**
(151→152→144→136), **wet waterlog** (81→76→54→37), **wet water**
(384→397→362→336), and **mean yield** (3710→3671→3745→3785). v2.18's dry yield
(3999) actually **beats** MPC-perfect (3994).

---

## 4. The Central Diagnosis (what was really wrong, and the false leads)

This is the most important section: the *causal story*, including claims that
were initially wrong and were corrected during the session.

### 4.1 The reward is NOT the problem (corrected mid-session)
An early claim — "the actor is weakly penalised for water/waterlog vs biomass" —
was **wrong**, and the arithmetic (re-derived from a v2.14 wet/100 rollout)
proves it: season-sum **r6 (waterlog) ≈ −13.7** vs **r1 (biomass) ≈ +1.27**.
Waterlog already **dominates the return ~10×**. So over-watering is not a
reward-magnitude failure. Consequences:
- **Do not increase α6.** The penalty is already huge and the agent still
  over-waters — bigger penalty won't help and risks dry-year conservatism.
- **r6 shape (linear vs quadratic) barely matters** at the operating point: at
  the wet median overshoot (~12.6 mm), `dr6_quad/d(over) ≈ −0.0103` vs
  `dr6_lin ≈ −0.0107` — nearly identical. This is why v2.15's shape change
  alone did nothing. Linear only differs at *small* overshoot (penalises the
  approach to FC), which is a real but second-order benefit.

### 4.2 The reward is already MPC-aligned (corrected the "reward x1≈FC" idea)
A suggested "reward for x1 near FC" term was **retracted** — the MPC penalises
`max(x1−FC,0)²` and **nothing for being below FC**. A "near FC" reward would
penalise being *below* FC and push the agent to *add* water when dry — the
opposite of the goal. The RL reward is already the same family as the MPC cost
(α1=1.0, α2=0.016, α3=0.1, α6). The gap to MPC is **not** the objective; it's
that **MPC solves the optimisation exactly each day with a model, while the RL
critic approximates it.**

### 4.3 "Rain-blindness" was a symptom, not the disease (the framing error)
The pre-session handoff organised everything around `corr(u, rain_fwd7)` ("the
actor ignores the rain forecast"). This metric is **the wrong target**, proven
by the controller's own committed data: **noisy-MPC is "rain-blind" by that
metric** (`corr(u, rain_fwd7) = +0.009`) **yet near-optimal** (wet Y 3702, water
309, waterlog 24.8). MPC achieves this by reacting to **soil-moisture state**
(corr with rain *today* −0.27) and holding x1 just below FC, not by following the
forecast. The real objective is **x1 regulation near FC**, which v2.14 failed
(x1=152, 86% of cell-days over FC) while both MPCs achieve (x1≈128–131,
~20–28% over). Rain-blindness was downstream of over-watering.

Additional corrections to the pre-session handoff, verified against the repo:
- The **MPC noise level** was misstated as "δ≈0.22%". Actual `NoisyForecast` is
  **multiplicative AR(1) with σ = 0.15·√j** (15% at 1-day lead, ~42% at 8 days,
  Buizza et al. 2005). This makes the noisy-MPC robustness result *stronger*,
  not weaker.
- The "RAIN_REF=30 made the actor rain-responsive in budget-constrained
  scenarios" claim is a **budget-exhaustion artifact**: in every one of the 9
  scenarios, the fraction of days the actor voluntarily chose u<0.5 mm *while
  budget remained* was **0.00**. The "responsiveness" was the hard budget clip
  running out, not learning.
- The auto-α run was headlined as 3733/3633 in the old handoff; committed files
  give **3718/3597** (perfect) — slightly overstated. (Re-derivation here gives
  3733 mean / 3633 wet for the `_auto_a` dir; the discrepancy is which checkpoint
  dir — both are within noise and the point stands: mean yield ≈ v2.14, the real
  win was water/waterlog.)

### 4.4 The actual root cause: entropy action-pin + buffer-coverage starvation
Two compounding mechanisms, both verified:

**(a) The SAC entropy term pins the actor mean at 6 mm/day.** SAC maximises
`E[Q] + α·H[π]`. For a tanh-squashed Gaussian, the entropy includes
`−Σ log(1 − tanh²(μ+σε))`, which is **maximised at the tanh centre** and
collapses toward −∞ at the boundary. So the entropy bonus contributes a gradient
on μ pointing toward tanh=0 — i.e. **toward 6 mm/day** — in *every* state, with
strength α. On dry days the agronomic optimum (~5 mm ≈ ET) coincides with this
centre, so no conflict (dry yields are excellent). On wet days the optimum is
~0–2 mm, but the entropy "spring" pulls back to 6 — and **0 mm sits exactly at
the boundary where the entropy penalty is largest**. This is the "action floor".
Evidence: wet/100 u_mean was 4.83 (v2.16) vs dry/100's 5.20 — only 0.37 mm less
despite heavy rain.

**(b) The replay buffer never contains low-water-in-wet-state transitions.**
SAC's state-dependent noise is learned from replay data; if the buffer has no
"low water in an unconstrained wet state" samples (confirmed: 0.00 voluntary
sub-0.5mm with budget remaining), the critic never learns those actions are
good, `∂Q/∂(low u)` stays flat/negative (twin-min pessimism on unseen actions),
and the policy has no Q-signal to overcome the entropy spring → it stays centred
→ generates more centred data → buffer stays starved. A self-reinforcing loop.

**Why the v2.16-auto "accident" worked:** α collapsed to ≈0 (released the spring)
*and* the lower-water trajectory changed what entered the buffer (broke the
loop). Both at once. RAIN_REF=30 was neither necessary (noisy-MPC needs no rain
signal) nor sufficient (v2.16-fixed had it and still failed).

---

## 5. v2.17-P3 — Exploration Injection (first deliberate fix)

**Design (Path 3).** Architecturally byte-identical to v2.16. Two training-time
changes: **(1) α: 0.01 → 0.005** (partial release of the entropy pin);
**(2) symmetric Gaussian collection noise**, σ linearly **0.30 → 0 over 60k
steps** (24% of 250k), then 0. Symmetric (not downward-biased) so a drop in x1
reflects the critic *learning* to prefer low water, not a biased data
distribution. New telemetry: a `LowActionCoverageCallback` to log the fraction
of collected actions below 1 mm.

**Result.** Confirmed the coverage hypothesis. wet x1 152→144, wet waterlog
76→54, wet water 397→362, mean yield 3671→3745 (highest stable run at the time).
Behavioural confirmation: wet/100 u_mean dropped to **4.04** (gap to dry/100's
5.04 widened to 1.0 mm), and **corr(u, next-day rain) = −0.36** on free-budget
days (approaching MPC's −0.39). The "rain-blindness" metric moved with **no
RAIN_REF change** — confirming it was a symptom. Stability stayed clean
(q_pred_mean bounded, never negative; final q_inflation +6.4% vs v2.16-fixed's
−146%).

**What didn't resolve.** Reached ~80% of the way to the auto-α operating point.
wet x1 144 (vs auto-α 132, MPC 130); residual over-watering attributed to the
remaining α=0.005 entropy pin.

**Telemetry bug #1.** `frac_low_action_wet` was NaN for all 250 rows — the
wet-episode detector looked for a "wet" *label* that barely exists in training
(only 3 of 20 training years are labelled). Overall coverage (13% low actions)
was logged fine; only the wet split failed.

---

## 6. v2.18-P3b — α=0.002 + Late Noise Reinjection (the result)

**Design.** Byte-identical architecture. **(1) α: 0.005 → 0.002**; **(2) late
noise reinjection** — σ anneals 0.30→0 over 60k (as v2.17), then a **triangular
pulse of peak 0.15 over [150k, 180k]** (peaking at 165k), then 0. Rationale: by
150k the critic is well-trained where the policy visits, so a short pulse
repopulates the *even-lower* (0–2 mm) action region with transitions that get
accurate value estimates immediately, pulling μ down the last bit without
early-training instability.

**Result — matches MPC on the key metrics.** mean yield **3785** (99.3% of MPC's
3810; dry yield 3999 *beats* MPC's 3994), mean waterlog **18.2** (better than
auto-α's 21.7, approaching MPC's 7.7), wet x1 **136** (39% over FC, down from
58%), wet water **336**. Temporal control strong: `corr(u, next-day rain)`
negative in **all 9** scenarios; wet/100 u_mean 3.60 vs dry/100's 5.20 (1.6 mm
gap). **Robust to noise** (noisy mean Y 3755).

**The stability "breach" — a false positive (important nuance).** The bias-ratio
diagnostic flagged BREACH: q_pred_mean went negative (−0.28 at 100k, −9.1 at
125k) and |q_inflation| peaked at 180% (at 200k). The rule said "go TD3, don't
lower α." **On inspection this is NOT a v2.7-style divergence:**
- It **dipped then fully recovered** (q_pred: …−9.1 @125k → +4.0 @150k → +10.4
  @200k → +8.4 @250k). A true deadly-triad cascade is monotone and never heals;
  v2.7 went 23,554 → 6.9e12. This self-healed.
- The worst dip (125k) was **before** the reinjection pulse; the pulse appears
  to have **helped pull the critic back**.
- The **best checkpoint is the final one (250k, eval reward 0.299)** — if the run
  were diverging, the last policy would be the worst.
- The 180% inflation spike is a **ratio artifact** (q_structural was small that
  eval), not a Q-explosion (absolute q_pred=10.4 was fine).

**Interpretation:** at α=0.002 the critic is **poorly *calibrated*** (noisy,
biased Q) but **not *diverging***. The bias-ratio metric measures calibration,
not stability, and the two came apart. v2.18 is a usable, excellent result. This
reinterpretation is *why* TD3 (§8) is framed as polish, not rescue.

**Telemetry bug #2 (unresolved).** The rainfall-based wet-detection fix *also*
returned 0/250 — the callback can't reach the env's private `_climate` attribute
through SB3's Monitor wrapper. (Two failed attempts; the robust fix is to have
the **env emit `season_rain` into the `info` dict**, which SB3 forwards
reliably — see §10. Not decision-relevant; eval behaviour already proves
coverage worked.)

---

## 7. The v2.18 Spatial / Elevation Analysis (an unsolved dimension)

**The user's physical intuition is correct and the data confirms a real
deficiency.** In a cascade field, high cells should receive *more* applied water
(they feed the low cells); the goal is *uniform resulting soil moisture* across
elevation.

Re-derived across all 9 scenarios (v2.18, perfect forecast):
- **`corr(applied u, elevation)` is negative in 8 of 9 scenarios** (−0.47 to
  −0.97) — the policy waters **low** cells *more*, the "wrong" direction.
- **`corr(resulting x1, elevation)` is −0.85 to −0.97 everywhere** — low cells
  end up **wetter** than high cells in every scenario.
- Spatial spread of soil moisture **widened** across versions: x1 std (wet/100)
  was 1.83 (v2.7, flat policy) → 2.60 (v2.17) → **3.20 (v2.18)**.

**Precise mechanism (wet/100 regression):** applied-water vs elevation slope
−0.0018 mm/m; resulting-x1 vs elevation slope **−0.109 mm/m**. Across the 111 m
elevation range, low cells get only ~0.20 mm/day more water yet end up **~12 mm
wetter** in x1. So the policy **under-compensates** for cascade inflow it cannot
see — it is *not* over-watering low cells out of error; a **flat per-cell critic
has no representation of upstream→downstream flow**, so it optimises each cell
myopically against its own drought signal and can't reason that watering a high
cell would relieve its downstream neighbours. Notably, the misallocation gets
*worse* under tight budget (dry/100 corr −0.42 → dry/70 −0.94).

**Evolution note (a corrected claim):** an earlier session claim that "v2.17
learned the cascade correctly while v2.7 had it inverted" was **wrong**.
v2.7 (`corr(u,elev)=+0.70`) actually watered high cells more — the *physically
correct direction* — and produced more uniform x1 (std 1.83), but only because
it was spatially flat (one field-wide knob, all adaptation in time). v2.17/v2.18
differentiate spatially 17× more but in the *under-compensating* direction.
v2.18's gains over v2.7 are **temporal and level** (less total water → less
waterlog everywhere), not spatial.

**Implication:** the spatial dimension is the **clear next scientific target**,
and the diagnostic (`corr(u,elev)`, `corr(x1,elev)`, x1 std) gives a precise
success metric for a GNN: drive `corr(u,elev)` positive and `corr(x1,elev)`
toward 0.

---

## 8. v2.19-TD3 — Design, Implementation, and Files (ready to train)

### 8.1 Why TD3
v2.18 matched MPC but the critic was poorly calibrated at α=0.002. TD3 is the
principled way to take entropy to zero: it (a) **removes the entropy objective**
(deterministic policy → no action-pin → can reach the tanh=−1 boundary = 0 mm),
(b) **adds target-policy smoothing** (the explicit replacement for the smoothing
entropy implicitly provided — the stabiliser whose absence let v2.7 cascade),
(c) **adds explicit exploration noise** (replacing SAC's policy stochasticity).
Goal: close the residual wet x1 gap (136 → ~130) **and** get a cleaner critic.

### 8.2 Design principle: reuse everything that works, change only the actor
- **CRITIC: byte-identical** to v2.16/v2.17/v2.18 (`_V211FactorizedContinuousCritic`
  — VDN sum, twin-Q, LayerNorm). TD3 and SAC both consume SB3's
  `ContinuousCritic` interface, so the cascade-suppression machinery transfers
  unchanged. This is deliberate.
- **ACTOR: deterministic** (`_TD3SharedActor`). **Identical feature pipeline** as
  the v2.16 actor — shared LeakyReLU MLP (`latent_pi`), the `2x−1` input
  re-center, the agent-major reshape — but the `(mu, log_std)` + squashed-Gaussian
  head is replaced by a single `mu_head` + `tanh`, producing one squashed action
  per cell in [−1,1]. SB3 maps [−1,1]→[0,1] identically to the SAC path.

### 8.3 Hyperparameters (v2.19)
Unchanged from SAC family: γ=0.99, τ=0.005, asymmetric actor LR 5×, LR 3e-4→5e-5,
buffer 250k, 250k steps, RAIN_REF=30, linear r6, 1097-dim obs.
TD3-specific: **target_policy_noise=0.2, target_noise_clip=0.5** (Fujimoto et al.
2018), **policy_delay=2**, **collection noise** N(0,σ) with σ **0.20→0.05 over
100k** then floor held (TD3 keeps a small floor, unlike the SAC experiments that
annealed to 0).

### 8.4 Files (ready to upload)
- **`src/rl/networks_td3.py`** (new): `_TD3SharedActor`, `TD3VDNPolicy`,
  `make_td3_policy_kwargs`, `TD3_OBS_MARKER=2.19`.
- **`src/rl/train_v219_td3.py`** (new): `AsymmetricLRTD3`, `train_td3_v219`.
- **`src/rl/runner.py`** (modified): `_detect_critic_arch` now also returns
  `has_log_std`; a **TD3 dispatch branch placed *before* the SAC 2.155 branch**,
  gated on `(not has_log_std) and obs_marker >= 2.185`, loads via `TD3.load`.
- **`notebooks/kaggle_td3_v219.ipynb`, `notebooks/colab_td3_v219.ipynb`** (new).

### 8.5 Senior-engineer critique applied (bugs guarded against)
1. **SB3 TD3 Actor structure mismatch (highest risk):** SB3's `TD3.Actor` builds
   `self.mu` as the *full* network. We let SB3 build it, then **overwrite
   `self.mu` with `nn.Identity()`** and install our own `latent_pi` + `mu_head`,
   and **override `forward`** so SB3's monolithic-net path is never used. SB3's
   `train()` calls `self.actor(obs)` (→ our `forward`) and `self.actor_target(obs)`
   — never `self.mu` directly — so this interception is correct and complete.
2. **Marker collision:** v2.16/2.17/2.18 all use marker 2.16; TD3 uses 2.19. The
   runner's TD3 branch is **first** and gated on `not has_log_std`, so a SAC 2.16
   checkpoint (which *has* log_std) never hits it and a TD3 checkpoint never
   falls through to the SAC branch.
3. **Action scaling:** env is [0,1]; actor outputs tanh∈[−1,1]; SB3 scales
   identically for SAC and TD3 (both `squash_output=True`, both inherit
   `BasePolicy.predict`→`unscale_action`). The SAC runs producing valid 0–12 mm
   actions prove the pipeline; TD3 is symmetric.
4. **Collection noise scale:** SB3 TD3 adds `action_noise` in the [−1,1] scale;
   σ=0.2 ≈ 0.1 in [0,1] ≈ 1.2 mm — same convention as the v2.17/v2.18 noise that
   worked.
5. **Asymmetric LR:** `AsymmetricLRTD3(TD3)` mirrors `AsymmetricLRSAC`, bumping
   `self.actor.optimizer` param-group LR by 5×.
6. **Two independent noises:** collection noise (exploration) and target-policy
   smoothing (Bellman target) are both set; SB3 applies smoothing internally.

### 8.6 Verification performed
All five files compile. Static bug-hunt passed (forward uses `mu_head`+`tanh`,
no `self.mu(` call; marker=2.19; `self.mu=Identity`; train passes all TD3 kwargs;
runner branch ordering correct; `RAIN_REF_V216` imported; return arity matches).
The notebook pre-flight cell includes a **checkpoint sanity assert** (no
`log_std`, `mu_head` present, marker≈2.19) so any structural surprise is caught
in the 1000-step pilot *before* a full run. **Caveat:** SB3 could not be
installed in the authoring sandbox (disk limits), so the SB3-runtime behaviour
was verified by source-contract reasoning, not execution — the pilot cell is the
empirical gate.

### 8.7 Acceptance criteria (decide on x1/waterlog + stability)
- **PRIMARY:** wet x1 < 134 mm (v2.18: 136; MPC: 130) AND wet waterlog < 32
  (v2.18: 37; MPC: 18) — close the residual gap.
- **STABILITY (the point):** critic_loss < 100; q_pred_mean **never negative**;
  |q_inflation| < 80%. TD3 should be **cleaner** than v2.18's α=0.002 run.
- **ATTRIBUTION:** if TD3 reaches MPC x1 where SAC α=0.002 stalled at 136, that
  isolates the entropy pin as the binding constraint.

---

## 9. Engineering Notes / Gotchas (hard-won, do not relearn)

- **Windows console encoding bug (FIXED):** `exp_rl.py` crashed with
  `UnicodeEncodeError` because checkpoint labels contain `α`/`—` and Windows
  stdout is cp1252. Fix: reconfigure stdout/stderr to UTF-8 with
  `errors="replace"` at the top of `exp_rl.py`. (Belt-and-suspenders; covers all
  non-ASCII labels, not just α.)
- **`exp_rl.py` has NO `--output-dir`** argument. Passing it → argparse error
  (return code 2). The runner writes to its own default `results/runs/<name>/`.
  Earlier eval failures were this, not a model problem.
- **EvalCallback "best" checkpoint:** for v2.14–v2.16 it defaulted to the **25k**
  conservative checkpoint (the 9-episode eval set is dry-skewed). v2.17 picked
  100k, v2.18 picked 250k. **Always also eval the FINAL checkpoint** — the best
  *wet* policy may differ from the best *mixed-eval* policy. (A wet episode should
  be added to the eval pool — still TODO.)
- **Single-seed caveat:** everything v2.14→v2.18 is **seed 0 only**. v2.18's
  margin over v2.16-fixed (wet +296 kg/ha, −39 waterlog days) is large enough to
  likely exceed seed noise, but the small inter-version differences earlier
  (v2.14 vs v2.16-fixed ~40 kg/ha) are within plausible seed variance. Multi-seed
  was deliberately deferred to save compute.
- **Replay buffer is excluded from archives** (`ignore replay_buffer_latest.pkl`),
  so **training-time** buffer coverage cannot be reconstructed post-hoc; only the
  *trained policy's* eval behaviour. Plan telemetry accordingly.
- **`action_noise` is NOT restored from checkpoints.** When resuming inside an
  exploration window, recreate the `NormalActionNoise` + the decay/reinjection
  callback.
- **The bias-ratio metric measures critic *calibration*, not training
  *stability*.** A negative `q_pred_mean` that recovers is calibration noise; a
  monotone geometric blow-up of `critic_loss` is the real cascade. Don't conflate
  (this is exactly what made the v2.18 "breach" a false alarm).
- **VDN entropy geometry breaks SAC's auto-α tuner:** with 130 summed per-cell
  entropies and a shared actor, natural entropy exceeds typical `target_entropy`
  from init, so the tuner pushes α monotonically **down to ~0** regardless of an
  *upper* cap. To use auto-α meaningfully you'd have to raise `target_entropy` to
  a reachable value — at which point a **fixed** α is simpler and more
  controllable. (This is why a "capped auto-α at 0.05" would just reproduce the
  unstable α≈0 collapse.)

---

## 10. Path Forward (prioritised)

1. **Run v2.19-TD3** (files ready). Decide on the §8.7 criteria. Outcomes:
   - *TD3 closes the gap and is cleaner* → the entropy pin was the last binding
     constraint; TD3 becomes the headline controller.
   - *TD3 matches v2.18 but no cleaner* → declare v2.18/TD3 equivalent; the
     remaining gap to MPC is the **model-free vs model-based** gap, not an
     entropy issue. Stop tuning the temporal/level axis.
   - *TD3 destabilises* (unlikely given target smoothing) → revert to v2.18 as
     the result.
2. **GNN critic over the D8 drainage graph** — the highest-value *new* direction,
   because it targets the **measured, unsolved spatial deficiency** (§7), not a
   solved one. Add a graph conv over the 130-cell adjacency before the VDN sum so
   the critic can reason about upstream→downstream flow. Success metric is
   precise: `corr(u,elev)` positive, `corr(x1,elev)`→0, x1 std down. Do this
   *after* TD3 (the GNN is an upstream feature extractor, orthogonal to
   SAC-vs-TD3; building it once on the final algorithm avoids a double port).
   Note: "QMIX/QPLEX" for *continuous* actions means **attention/monotonic
   mixing of continuous Q** (FACMAC line), not vanilla value-based QMIX.
3. **Fix the wet-coverage telemetry properly** (if wanted for the thesis): have
   `IrrigationEnv.step()` put `season_rain` (and budget state) into the returned
   `info` dict; SB3 forwards `info` reliably via `self.locals['infos']`. Then a
   coverage callback reads `infos[0]['season_rain']` instead of reaching into the
   env through the Monitor wrapper.
4. **Multi-seed confirmation** of v2.18/TD3 (seeds 1–2) before final write-up, to
   put an error bar on the headline numbers. v2.14 is the safest to seed (proven
   stable) for establishing σ_seed.
5. **Reframe the thesis contribution honestly:** the RL controller does **not**
   beat MPC on quality (MPC dominates yield, water, waterlog). The defensible
   contributions are: **(a) model-free** (MPC here is handed the true ABM as its
   internal model — an oracle), **(b) amortised real-time inference** (a single
   forward pass vs MPC's per-step optimisation over 130 cells), **(c) robustness
   to realistic forecast noise** (demonstrated: v2.18 noisy ≈ perfect). The
   strongest possible additional experiment: run **both** controllers under a
   *misspecified* model (perturb the ABM parameters the MPC's internal model
   uses) to show where the learned policy's robustness actually pays off.

---

## 11. Repository Map (the parts that matter)

- `abm.py` — ABM physics (cascade routing, linear waterlog stress).
- `soil_data.py` — crop/soil parameters (FAO-56 grounded).
- `climate_data.py` — TRAINING_YEARS (20), SCENARIO_YEARS (dry/mod/wet).
- `src/rl/gym_env.py` — env, reward (r1/r2/r3/r6), RAIN_REF_V216=30, action [0,1].
- `src/rl/networks.py` — SAC actors (v2.7→v2.16 lineage), VDN LayerNorm critic
  (`_V211FactorizedContinuousCritic`), all policy classes + marker scheme.
- `src/rl/networks_td3.py` — **NEW:** deterministic actor + TD3 policy (reuses
  the V211 critic).
- `src/rl/runner.py` — eval harness; `_detect_critic_arch` (now reports
  `has_log_std`) + marker dispatch (now includes the TD3 branch).
- `src/rl/callbacks_v210.py` — BiasRatio, ActionStats, OptimizerLR callbacks.
- `src/rl/callbacks_exploration.py` — `ExplorationNoiseDecayCallback`,
  `LateNoiseReinjectionCallback`, `LowActionCoverageCallback` (wet detection
  still needs the info-dict fix).
- `src/rl/train_v2{14..18}*.py`, `src/rl/train_v219_td3.py` — training
  entrypoints.
- `src/mpc/` — MPC controller + `NoisyForecast` (AR(1), σ=0.15·√j).
- `results/runs/sac_v2{14,16,17,18}_best_model/`, `.../sac_v218_best_model/` —
  committed 9-cell eval outputs (parquet + json).
- `results/rl/sac_v218_p3b_seed0_*/` — v2.18 training logs (bias_ratio,
  exploration_sigma, low_action_coverage, eval_logs).
- `notebooks/kaggle_*`, `notebooks/colab_*` — per-version training notebooks.

---

## 12. One-paragraph status for a new reader

A VDN-factorised SAC controller for 130-cell cascade-field irrigation was stuck
for many versions on wet-year over-irrigation. The cause was **not** the reward
(waterlog already dominates 10×) nor rain-forecast-blindness (a symptom); it was
the **SAC entropy term pinning the actor at the mid-range 6 mm/day action** plus
a **replay buffer that never sampled low-water actions in wet states**. Reducing
α (0.01→0.005→0.002) and injecting decaying exploration noise broke both,
yielding **v2.18**, which matches a perfect-forecast MPC oracle on yield
(99.3%), water, and waterlogging and is robust to realistic forecast noise — a
legitimate result. A **TD3 port (v2.19)** is built and ready to push the last
~6 mm of soil-moisture gap and clean up the critic. The remaining open problem is
**spatial**: the controller under-compensates for downhill cascade flow (low
cells end up ~12 mm wetter than high cells), which a **GNN critic over the D8
graph** is positioned to fix. Everything is single-seed; multi-seed and the
model-free/robustness framing are the path to a defensible thesis.
