# Change Specification — `gym_env.py` v2.7

**Project:** ITMO MSc thesis — *Modern Control Methods for Agricultural Irrigation*
**Author of record:** Tara Torbati
**Document purpose:** Specify the changes from v2.6 → v2.7 of the SAC training environment and supporting code, with enough rationale that a future co-author or reviewer can reconstruct *why* each change was made.

---

## 1 — Context: What we learned from the v2.6 evaluation

The v2.6 SAC checkpoint (step 350 k, VDN factorized critic) was the first end-to-end-trained policy that achieved competitive yields against MPC in dry and moderate scenarios — within 97–99 % across all six dry/moderate cells. The wet-year cells, however, exposed a structural failure mode that is worth explaining in detail because it directly motivates every change in this document.

### 1.1 The wet-year failure mode

In the wet/100 % evaluation cell (2024 climate, 484 mm budget):

- SAC_best delivered **3 168 kg/ha**, against MPC_Hp3's **3 717 kg/ha** — an 85 % efficiency.
- SAC_best used **99 %** of the available budget; MPC_Hp3 voluntarily returned **38 %** of it.
- SAC_best recorded **90 waterlog-days per agent** (soil moisture above field capacity), the highest of any controller in the entire study.

The day-by-day trajectory analysis revealed what the policy actually does: it outputs a **near-constant 5.13 mm/day field-mean irrigation** with a coefficient of variation of 0.055 — essentially flat for the entire 93-day season. The standard deviation across the season is 0.28 mm. On day 38, which received 35.81 mm of rainfall, the SAC policy delivered 5.79 mm of irrigation, pushing soil moisture to 186 mm (46 mm above field capacity). On the same day, MPC_Hp3 delivered 0.28 mm. On day 69, with 33.53 mm of rainfall, SAC delivered 5.71 mm pushing soil moisture to 205.71 mm — within 14 mm of full saturation. MPC delivered 0.09 mm.

The correlation between daily irrigation and same-day rainfall, across the wet-year trajectory, is **+0.38** for SAC_best (it irrigates *more* when it rains) and **−0.31** for MPC (it irrigates *less* when it rains). The correlation between daily irrigation and current soil moisture is **+0.03** for SAC (no closed-loop response) and **−0.58** for MPC.

### 1.2 What this is not

For a long time we believed the wet-year failure was caused by the FC-overshoot penalty α₆ being too weak. Cumulatively over the season, however, r₆ totals −35.6 reward-units while r₁ (growth) totals +1.16. On individual rainy days r₆ exceeds |r₁| by a factor of 30–500. The penalty is *not* too weak. The reward function, evaluated honestly, would assign MPC's wet-year trajectory a net step-reward of −0.008 and SAC's a net step-reward of −0.355 — a difference of 4 000 ×. The signal is clearly present. The agent simply cannot extract it.

### 1.3 What it actually is

Three reinforcing causes produce the observed constant-trickle behaviour, and they map cleanly to three classes of fix:

1. **The actor is functionally blind to spatial structure.** Each of the 130 agents shares a single parameter-shared MLP, and each agent's per-agent observation block (5 features) is constructed inside `_build_obs` so that 4 of the 5 features vary little across agents (`x4`, `x3`, `x5` are determined by climate which is field-uniform; `x5` is zero almost everywhere outside rain events). The only feature carrying meaningful per-agent variation is `x1`, the soil moisture. Worse, the 5th feature — which Chapter 4 specifies as **normalised elevation** `γ⁽ⁿ⁾` — was replaced during the v2.5/v2.6 rewrite by `x2 / θ₁₈`, the normalised growing-degree-day accumulation, which is identical across all 130 agents on any given day. The actor literally has no static per-agent feature to differentiate agents on. The empirical consequence: the correlation between per-agent total irrigation and elevation is **+0.066** for SAC vs **+0.65** for MPC. SAC delivers the same volume to the hilltop and to the valley.

2. **The agent does not feel any reward consequence for over-requesting water.** The environment contains a per-step truncation `irr_mm = minimum(irr_mm, remaining)` (gym_env.py lines 201–202). When the agent requests more than the remaining budget, the environment silently delivers less, the reward is computed on what was delivered, and the agent observes the next state — but nowhere in the loop does it ever pay a price for the *request*. The cheapest robust strategy across the randomised-budget training distribution is therefore to output a constant rate and let the environment do the budgeting. The cumulative water used in every evaluation cell is exactly the budget, or one millimetre below it.

3. **The episode ends the moment the budget is exhausted** (`terminated = budget_done`, line 238). If the agent overspends early, the simulation never continues to the late-season grain-fill phase where the consequences of overspending would manifest. The agent never sees, during training, what running out of water on day 70 looks like on day 90.

Items 2 and 3 together prevent the agent from learning to *condition* on the budget signal even though that signal is in the observation (scalar slot 1, `budget_frac`). Item 1 prevents the agent from learning per-agent specialised behaviour even though the VDN critic gives it the per-agent credit-assignment signal.

These three causes are the targets of the v2.7 changes. None of them is a hyperparameter that needs tuning; all three are structural and have determinate fixes.

---

## 2 — Goals and non-goals of v2.7

### Goals

- Restore Chapter 4 spec compliance for the per-agent observation block (Item 1, partial fix).
- Enrich the per-agent observation with static topographic features so the actor can specialise spatially (Item 1, full fix).
- Restructure the episode lifecycle so the agent feels the late-season consequences of early overspending (Items 2 and 3).
- Simplify the reward function: remove components that demonstrably do not bind, keep only those that the v2.6 sensitivity sweep validated.

### Non-goals

- No change to the action space, the actuator cap, or the seasonal budget definition.
- No change to the operating reward weights α* (those were chosen by the calibration sweep documented in Chapter 4).
- No change to the VDN factorised critic architecture (only its input width changes, by 3 features per agent).
- No change to the training algorithm, hyperparameters, or schedule (still SAC + GradClipCallback + LR decay + ent_coef = 0.05).
- No change to `abm.py`, `soil_data.py`, `climate_data.py`, or `src/precompute.py`.
- No change to the MPC implementation, the baselines, or the evaluation protocol.

The v2.7 → v2.6 comparison will therefore be a clean ablation of (a) the observation-spec bug fix, (b) the topographic-feature enrichment, and (c) the episode-lifecycle redesign, taken together.

---

## 3 — Change 1: Restore Chapter 4 spec compliance for the per-agent block

### 3.1 The bug

Chapter 4 of the dissertation, in the SAC observation specification (around line 808 of `ch4_controller_design.tex`), states:

> *"for each agent n, the normalized root-zone moisture x₁⁽ⁿ⁾/FC, [x5, x4], raw maturation index x₃⁽ⁿ⁾, and normalized elevation γ⁽ⁿ⁾."*

The original implementation, in commit `eaace04` (introduced May 10 2026), followed this specification correctly:

```python
self.elev_norm = self.terrain['gamma_flat']      # normalized elevation, per-agent
# ...
elev = self.elev_norm
return np.concatenate([x1_norm, x5_norm, x4_norm, x3, elev, ...])
```

Commit `c623833` ("VDN critic structure update") rewrote `gym_env.py` against the actual ABM and `soil_data` interfaces and, in doing so, replaced the 5th per-agent feature:

```python
gamma   = np.clip(self._abm.x2 / _GDD_MATURITY, 0.0, 1.0)
agent_block = np.stack([x1_norm, x5_norm, x4_norm, x3, gamma], axis=1)
```

The replacement was driven by a variable-name collision. The terrain module had long stored the normalised-elevation array under the variable name `gamma_flat`, following the naming convention of the Lopez-Jimenez et al. (2024) topographical MPC paper. In the agronomic literature, however, `γ` is sometimes used for a maturity index or development stage. The rewrite's author saw `gamma` in the docstring, recalled the agronomic meaning, and chose `x2 / θ₁₈` (the GDD fraction) as the source — which is not what Chapter 4 specifies and not what the original code provided.

The replacement is doubly damaging because `x2` (accumulated thermal time) is computed by the ABM from a single field-wide temperature input, so it takes **identical values across all 130 agents** on any given day. The 5th per-agent feature therefore contributes zero variance across agents and provides no signal that the shared actor could use to differentiate them.

### 3.2 The fix

The 5th per-agent feature is restored to `_TERRAIN['gamma_flat']` (normalised elevation in [0, 1]). To prevent the variable-name collision from recurring, the slot is renamed `elev_norm` in the `_build_obs()` body:

```python
elev_norm = _TERRAIN['gamma_flat']     # static, set once at env construction
# inside _build_obs:
agent_block = np.stack(
    [x1_norm, x5_norm, x4_norm, x3, elev_norm], axis=1
)
```

The Chapter 4 notation `γ⁽ⁿ⁾` remains unchanged — both because no thesis text needs to be edited for this restoration and because the symbol `γ` for elevation cascade weights matches the underlying paper. The variable in `terrain.py` also retains its existing name `gamma_flat`; only the local variable inside the obs builder is renamed.

### 3.3 What this fix does not do on its own

Restoring elevation to the obs is necessary but not sufficient. Elevation alone cannot fully determine an agent's hydrological role, because the field's elevation surface contains valleys and ridges: two cells at the same height can have completely different downstream connectivity. A cell at the top of a small ridge has 8 lower neighbours and is a runoff source; a cell in a valley has 0 lower neighbours and is a runoff sink. Their elevation values may be identical but their cascade roles are opposite. Section 4 addresses this.

---

## 4 — Change 2: Enrich the per-agent block with static topographic features

### 4.1 Motivation

The cascade routing in `abm.py` (line 84) uses three pieces of per-agent information:

- the agent's elevation (`gamma_flat[n]`),
- the count of downhill neighbours (`Nr[n]`),
- the set of downhill neighbours within the field (`sends_to[n]`).

These quantities are static — fixed by the digital elevation model, computed once at terrain load, and never updated during a season. The ABM uses all three to decide where runoff goes; the actor currently sees none. This is the asymmetry that has to close.

### 4.2 The three new features

We add three static topographic features per agent, all already computable from `_TERRAIN`:

| Feature | Formula | Source | Range |
|---|---|---|---|
| `Nr_norm` | `_TERRAIN['Nr'][n] / 8.0` | already in terrain | [0.125, 0.75] obs |
| `Nr_internal_norm` | `_TERRAIN['Nr_internal'][n] / 8.0` | already in terrain | [0, 0.75] obs |
| `n_upstream_norm` | `(Σ_m 𝟙[n ∈ sends_to[m]]) / 8.0` | derived once at env construction | [0, 0.75] obs |

Their interpretation:

- `Nr_norm` measures **total fanout** including water that leaves the field at boundary cells. High `Nr_norm` means "I am at the top of a slope; water I receive disperses widely; I keep little of what arrives."
- `Nr_internal_norm` measures **fanout to interior cells**. Comparing this with `Nr_norm` tells the actor whether it is on the field boundary (`Nr > Nr_internal`) or in the interior (`Nr = Nr_internal`).
- `n_upstream_norm` measures **upstream feed**. High value means "I receive substantial runoff from above; my x1 will rise from rainfall faster than a hilltop cell would." This is what makes valley cells different from hilltop cells even at the same elevation.

Together these three plus `elev_norm` (from Change 1) give the actor a complete static signature of each agent's hydrological position. They are computed once at terrain load, stored as numpy arrays on the env instance, and indexed in `_build_obs()` with no per-step cost. The empirical maximum value across the 130 agents in the Gilan terrain is 6, so dividing by 8 yields a clean [0, 0.75] range with headroom for terrains with denser connectivity.

### 4.3 Effect on observation dimension

The per-agent block grows from 5 features to 8. The total observation dimension becomes:

```
OBS_DIM = 8 × 130 + 9 + 48 = 1097
```

up from 707. The per-agent actor input grows from 62 dims to 65; the per-agent critic input grows from 63 dims to 66. The actor and critic hidden widths (128, 256) are unchanged. The first-layer parameter count increases by roughly 5 %, which is negligible.

### 4.4 Why not also include n_upstream_internal?

We considered a fifth feature, `n_upstream_internal_norm`, measuring upstream feed from interior cells only. It is highly correlated with `n_upstream_norm` (boundary cells generally have zero upstream-internal anyway) and adds little independent signal. We omit it for parsimony but it remains a candidate if v2.7 retraining shows residual spatial blindness.

### 4.5 Defensibility in Chapter 4

Chapter 4 will need a small revision (deferred until v2.7 retraining is complete and committed):

- Per-agent block description: 5 → 8 features.
- Observation total: 707 → 1097.
- New paragraph explaining the three topographic features, with one sentence linking each to its role in the ABM cascade. This grounds the design choice in the simulation physics rather than presenting it as ad-hoc feature engineering.

---

## 5 — Change 3: Simplify the reward function

### 5.1 Drop the burn-rate penalty `r_b`

The v2.6 reward contained a soft penalty on excessive cumulative water-use pace:

```python
daily_pace = FULL_SEASON_NEED_MM / _K
burn_rate  = self._water_used / max(self._day * daily_pace, 1e-6)
rb         = -LAMBDA_BUDGET * max(burn_rate - 1.0, 0.0) ** 2
```

This term was designed to nudge the agent away from front-loading water consumption. In practice, the v2.6 policy stabilised at exactly the linear-pace rate (~5.2 mm/day = 484/93) and so `burn_rate ≤ 1` on every step of every cell, making `rb = 0` everywhere. The term was pure dead weight, contributing no gradient signal in the training distribution it was meant to shape. It is removed in v2.7 along with the `LAMBDA_BUDGET` constant.

### 5.2 Remove the dead delta-u code path

`ALPHA5_RL` has been zero since v2.6. The associated code (`_prev_actions` tracking, delta_u computation, `r5` term) is dead. We remove the dead branch entirely. The `_prev_actions` instance attribute is also removed (it was only used by the disabled r5 term).

### 5.3 Do not add a ponding penalty

A surface-ponding penalty `r_pond = −α_pond × mean(x5)/X5_REF` would in principle deliver a same-step signal at the moment of over-irrigation. We have ruled it out on prior evidence: the Chapter 4 sensitivity sweep over α₄ established that

- α₄ alone caused the MPC to underwater and store water in saturated soil (the ponding penalty pushed the controller to keep x5 = 0 by absorbing water into x1 well past field capacity);
- α₄ + α₆ produced no measurable improvement over α₆ alone.

Adding `r_pond` to the RL reward would introduce a degree of freedom not present in the MPC formulation, breaking the policy-equivalence comparison that Chapter 4 §4.6 builds on. We leave it out.

### 5.4 Do not add an overspend penalty

A linear or quadratic penalty proportional to `max(requested − remaining, 0)` would tell the agent at the moment of decision that it tried to overspend. We have rejected this approach because the agronomic value of saved water varies dramatically across the season:

- During the early vegetative phase (days 0–14) the per-day growth reward `r₁` averages +0.003 reward-units.
- During the peak grain-fill phase (days 45–59) it averages +0.025 — eight times higher.
- During late ripening (days 90–92) it returns to +0.007.

A flat overspend penalty constant `λ_over` cannot reflect this. A value scaled to make `λ_over × 1 mm` meaningful relative to peak-grain-fill growth would be devastatingly large during the vegetative phase, where it would dominate growth by 17 ×. A value scaled to early-season would be effectively zero during grain-fill. A time-varying `λ_over(t)` would introduce a tunable schedule whose justification we cannot ground in the irrigation control literature. We do not add it.

Instead, the temporal weighting of overspend pain is delivered through state transitions: see Change 4.

### 5.5 Final reward formula

```
r(t) = r1 + r2 + r3 + r6

  r1 = ALPHA1  × (mean(x4_new) − mean(x4_old)) / X4_REF
  r2 = −ALPHA2 × mean(irr_delivered) / UB_MM
  r3 = −ALPHA3 × mean(max(ST − x1, 0)) / (ST − WP)
  r6 = −ALPHA6 × mean(max(x1 − FC, 0)²) / FC²
```

with `(ALPHA1, ALPHA2, ALPHA3, ALPHA6) = (1.0, 0.016, 0.1, 8.0)` from the v2.6 sensitivity sweep. The reward is the negation of the MPC path cost at α*, omitting the α₄ and α₅ terms (both zero) and the burn-rate term (which was always zero in practice). This is a four-term reward, down from seven, with no loss of binding signal.

---

## 6 — Change 4: Restructure the episode lifecycle

### 6.1 The change

Two related changes to `step()`:

```python
# v2.6 (current):
season_done = (self._day >= _K)
budget_done = (self._water_used >= self._budget_mm - 1e-6)
terminated  = budget_done
truncated   = season_done and not budget_done

# v2.7:
terminated  = False
truncated   = (self._day >= _K)
```

The episode now always runs for the full 93 days. The per-step clip `irr_mm = np.minimum(irr_mm, remaining)` is preserved: when the budget is exhausted, the env continues to deliver zero irrigation (since `remaining = 0`), and the ABM continues to advance through climate. Soil moisture, biomass growth, and drought stress all continue to be computed; the agent receives non-trivial observations every step; rewards `r₁` (growth slows or stops) and `r₃` (drought stress) continue to be earned.

### 6.2 Why this delivers correctly time-weighted overspend pain

The seasonal growth profile is non-uniform by a factor of 8 (Section 5.4). An episode where the agent overspends early and runs out by day 70 produces:

- 70 days of normal growth, with `r₁` accumulating roughly as on any healthy trajectory;
- 23 days of zero irrigation, during which soil moisture decays (via drainage `phi3`, transpiration `phi1`, and any rainfall that does not infiltrate), and biomass growth `h6 × h3 × g × rad` falls as the drought stress factor `h3` approaches `1 − θ₁₄ = 0.20` (rice is highly drought-sensitive).

If the budget-exhaustion happens during grain-fill (days 45–59), the lost `r₁` from those 23 days is approximately `15 × 0.025 + 8 × 0.018 ≈ 0.52` reward-units forgone — a substantial fraction of the seasonal total. If exhaustion happens late (after day 80), the lost `r₁` is `13 × 0.007 ≈ 0.09` — small.

The agent therefore experiences a *higher cost for early overspending than for late overspending* without any reward-weight schedule. The temporal weighting is delivered automatically by the underlying biological dynamics. This is the cleanest available mechanism: it requires no new hyperparameters, no new α terms, no decisions about linear-vs-quadratic shape, and its calibration is the ABM physics itself rather than a discretionary choice by the researcher.

### 6.3 Why we keep the per-step clip and do not add an overspend penalty

The clip is the cheapest possible way to guarantee physical budget compliance for every reported run. With it, every v2.7 evaluation result will satisfy `water_used ≤ budget_total` by construction. The alternative — letting the agent physically exceed the budget and relying on a reward penalty to push it back into compliance — would either fail (if the penalty is too small) or destabilise training (if too large), and would in either case produce evaluation rows that the thesis would have to flag as non-compliant. Defensibility is improved by keeping the clip.

The "delivered amount" is what the ABM sees and what the reward `r₂` (water cost) is computed on. The agent therefore pays for what is actually irrigated, not for what it requested. From the agent's point of view: requesting 5 mm when 5 mm is available behaves identically to requesting 5 mm when 0 mm is available, in terms of immediate reward — the difference shows up only in the next-step state (soil moisture and biomass eventually diverge) and accumulates over the rest of the episode. With the episode running to 93 days, this divergence has time to manifest in the reward stream.

### 6.4 Budget randomisation during training is retained

`reset()` continues to sample `budget_frac ~ U(0.70, 1.00)`. We had considered fixing the budget at 100 % during training and producing separate evaluations by re-training, but on reflection the original randomised design is the right one for v2.7. With Change 4 in place, the agent now sees, across the training distribution, episodes in which the same opening behaviour leads to different terminal outcomes depending on the available budget. This is precisely the signal that teaches the policy to *condition* on `budget_frac` (already present in scalar slot 1 of the observation) — a signal that v2.6 could not deliver because the early-termination + invisible-clip combination collapsed all budget levels into "trickle until env stops you."

We hypothesise — and Chapter 5 should claim — that in v2.7 the per-day irrigation profile in low-budget cells will diverge from the per-day profile in high-budget cells, and that this divergence is what closes the wet-year gap.

---

## 7 — What does not change

- Action space: `Box(0, 1, (130,))` scaled by `UB_MM = 12.0`. Per-agent physical actuator cap of 12 mm/day is unchanged.
- Field-mean water accounting in `step()` (introduced in v2.6 to fix the 130 × inflation bug) is unchanged.
- Seasonal budget reference `FULL_SEASON_NEED_MM = 484.0` is unchanged.
- Forecast horizon `FORECAST_H = 8` is unchanged; the 48-dim forecast block in the observation is unchanged.
- The 9-dim scalar block of the observation is unchanged: `[day_frac, budget_frac, budget_total_norm, burn_rate, rain_today, ETc_today, h2, h7, g_base]`. Note: `burn_rate` is retained in the *observation* (it's an informative signal for the actor) even though the *reward* penalty `r_b` based on it is removed. The actor may still find the burn-rate scalar useful for decision-making.
- The ABM (`abm.py`), the crop parameters (`soil_data.py`), the climate loader (`climate_data.py`), and the precompute module (`src/precompute.py`) are unchanged. v2.7 is purely a wrapper and observation/reward redesign.
- The MPC controller, the baselines, and the evaluation/runner pipeline are unchanged.
- The VDN factorised critic architecture in `networks.py` is structurally unchanged. Only the input widths are updated to match the new `OBS_DIM = 1097`.
- The training script (`train.py`) is structurally unchanged. The `GradClipCallback`, the LR decay schedule (3e-4 → 5e-5), the fixed `ent_coef = 0.05`, and the 500 k-step budget are all retained.
- Both unit-test files (`tests/test_factorized_critic.py` and `tests/test_rl_smoke.py`) need at most a one-line change each to reflect the new `OBS_DIM`.

---

## 8 — Files to be modified

| File | Change scope |
|---|---|
| `src/rl/gym_env.py` | Major rewrite of `_build_obs`, `step`, `_compute_reward`, and `__init__`. Constants updated. |
| `src/rl/networks.py` | Three constants updated: `N_AGENT_FEATURES` 5 → 8; `OBS_DIM_DEFAULT` 707 → 1097; `PER_AGENT_INPUT_DIM` 62 → 65; `PER_AGENT_CRITIC_INPUT_DIM` 63 → 66. No structural changes. |
| `tests/test_rl_smoke.py` | One-line update: `assert obs.shape == (1097,)` instead of `(707,)`. Other tests should pass unchanged. |
| `tests/test_factorized_critic.py` | One-line update if it hardcodes `OBS_DIM` anywhere; otherwise unchanged. |

Files explicitly **not** modified by this spec:

- `src/rl/train.py` (training loop is unchanged)
- `src/rl/runner.py` (evaluation loader is unchanged; auto-detection of checkpoint architecture is preserved)
- `abm.py`, `soil_data.py`, `climate_data.py`, `src/precompute.py`, `src/terrain.py`
- All scripts in `scripts/` (the experiment harness)
- LaTeX chapters (deferred until v2.7 retraining results are in hand)

---

## 9 — Validation checklist before retraining

Before launching the 500 k-step training run with v2.7 code, the following must be verified locally:

1. `tests/test_rl_smoke.py` passes with the updated `OBS_DIM` assertion. In particular, the first reset produces a 1097-dim observation with finite values everywhere.
2. `tests/test_factorized_critic.py` passes, confirming the gradient flow and shape invariants of the VDN critic with the new input width.
3. Three sanity probes on a fresh env instance (not part of the test suite — just informal checks):
   - The 5th per-agent feature varies across the 130 agents on day 0 (it is `elev_norm`, not a field-uniform value).
   - The three new topographic features are all static across a full season (the env never updates them).
   - The episode runs to day 93 even when the budget is set to a small value (e.g. 50 mm) that would have triggered early termination under v2.6.
4. A 10 k-step pilot training run completes without critic-loss explosion. Reward magnitudes per step stay within roughly the same scale as v2.6 (the reward shape has not changed; only the obs space and the episode lifecycle have).

If all four pass, the 500 k-step run can proceed under the same Colab Pro setup as v2.6.

---

## 10 — Open questions to revisit after retraining

- If v2.7 still exhibits constant-rate behaviour in any cell, the next candidate fix is to introduce `n_upstream_internal_norm` as a 9th per-agent feature (4.4 above), or to abandon shared-actor parameterisation in favour of per-agent embeddings.
- If v2.7 overshoots the budget on any cell (it should not — the clip guarantees compliance — but if a bug introduces this it must be caught), the run is invalid for thesis reporting.
- If v2.7 underspends substantially (e.g. uses < 80 % of budget in dry/100 %), this indicates the agent has overlearned the late-season-drought signal and is now over-conservative. Mitigation in that case would be to reduce the early-termination protection or to introduce a small terminal yield bonus — to be decided based on the observed behaviour, not predicted in advance.
- Chapter 5 narrative for the v2.6 → v2.7 transition needs to be drafted *after* the v2.7 results are in, to avoid pre-committing to a story that the data may not support.

---

*End of specification.*
