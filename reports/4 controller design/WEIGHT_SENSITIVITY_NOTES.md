# Weight Sensitivity Analysis — Methodology Notes

This document accompanies `scripts/experiments/exp_weight_sensitivity.py`
(v2.3) and `src/mpc/cost.py` (v2.3). It documents the economic and
agronomic rationale behind every value in the weight sweep, so that the
thesis methodology section can cite precise sources for each parameter.

---

## 1. Cost-function structure

The MPC minimizes the six-term normalized cost:

```
J = -α₁ · x4_terminal / x4_ref                                    (terminal biomass)
  + α₂ · Σₖ (Σₙ uⁿₖ) / W_daily_ref                                 (water cost)
  + α₃ · Σₖ (1/N) Σₙ max(ST - x1ⁿₖ, 0) / (ST - WP)                (drought)
  + α₄ · Σₖ (1/N) Σₙ x5ⁿₖ / x5_ref                                (ponding)
  + α₅ · Σₖ ‖uₖ - uₖ₋₁‖² / (u_max² · N)                          (Δu)
  + α₆ · Σₖ (1/N) Σₙ [max(x1ⁿₖ - FC, 0) / FC]²                   (x1 > FC, NEW)
```

Each term is normalized to be O(1) per time step, so the weights αᵢ are
dimensionless multipliers expressing the relative importance of each
objective.

The nominal operating point is:
```
α = (1.0, 0.01, 0.1, 0.5, 0.005, 0.0)
```
which matches the configuration of the 27 already-completed MPC runs
in `results/runs/mpc_perfect_*.parquet`. Setting α₆ = 0.0 by default
preserves backward compatibility — existing runs are reproducible.

---

## 2. Why α₂ (water cost) needs a price-tier sweep

The α₂ value is not a numerical hyperparameter — it encodes the
**economic environment** in which the controller operates. Iran has a
multi-tier water-pricing system in which the same volume of water can
cost orders of magnitude more depending on who is using it.

### 2.1 Conversion from real prices to α₂

Take a target seasonal yield of 4198 kg/ha (the MPC dry/100%/Hp=8 result)
and Hashemi rice at 500,000 toman/kg market price. Then:

```
revenue_per_ha = 4198 kg/ha × 500,000 toman/kg ≈ 2.099 × 10⁹ toman/ha
```

A full irrigation budget of 484 mm corresponds to:

```
volume_per_ha = 484 mm × 10 m³/(mm·ha) = 4840 m³/ha
```

The α₂ value is calibrated so that the cost of a full seasonal water
budget, in fractional units of revenue, matches:

```
α₂ ≈ (volume × price_per_m³) / revenue
```

### 2.2 The four price tiers

| Tier | Price (toman/m³) | Full-budget cost | % of revenue | α₂ |
|---|---|---|---|---|
| **Subsidized agricultural** (real Iranian) | 175 | 847,000 | 0.04% | **0.0004** |
| **Domestic base** (≤5 m³/month per family member) | 7,000 | 33,880,000 | 1.61% | **0.016** |
| **Domestic tier d (×2.8)** (high-consumption household) | 19,000 | 91,960,000 | 4.38% | **0.044** |
| **Industrial** (industrial parks) | 115,000 | 556,600,000 | 26.5% | **0.265** |

Sources:
- Iran Water Resources Management Co. tariff schedules (1402-1403 / 2023-2024)
- Mesgaran & Azadi (2018), "A national adaptation plan for water scarcity in Iran"
- Nouri et al. (2023), "Water management dilemma in the agricultural sector of Iran"

### 2.3 Thesis narrative arc

The α₂ sweep tells a policy story:

- **α₂ = 0.0004** (subsidized agri): water is essentially free relative
  to revenue. Even an *optimal* MPC has weak economic incentive to
  conserve. This is the ground-truth Iranian farmer's economic situation.
- **α₂ = 0.016 / 0.044**: scarcity-adjusted pricing (urban-tier costs).
  Tests how the controller would respond if subsidies were removed.
- **α₂ = 0.265**: industrial pricing — equivalent to a counterfactual
  in which agricultural water is reallocated to industry.

The current 27 runs at α₂ = 0.01 sit between "subsidized" and "domestic
base" — call this the "moderate-realism baseline" in the thesis text.

---

## 3. Why α₆ (x1 > FC penalty) is added

### 3.1 The motivation

Two observations from the 27 completed MPC runs prompted this term:

1. **Issue 1 (Dry/Hp=14 over-FC excursion)**: at dry/100%/Hp=14, the
   controller pushes x1 above field capacity (140 mm) for **21.6
   agent-days per agent** despite the dry climatology offering no
   mass-balance reason to do so. The water sits above FC and slowly
   drains via phi₃ = θ₄ · max(x1 − FC, 0) at 5%/day (Issue 1).

2. **Issue 3 (Wet/Hp=14 chronic waterlogging)**: at wet/100%/Hp=14, the
   controller keeps x1 at 175–205 mm (near saturation) for >25
   consecutive days, producing 61 waterlog-days/agent and 40-mm peak
   ponding (Issue 3).

The implicit waterlog stress (h6) embedded in the ABM's biomass
increment x4_inc = θ₁₃ · h3 · h6 · h7 · g · rad provides only a
*multiplicative* and *delayed* disincentive: at x1 = 200 mm, h6 = 0.57,
costing only ~43% of that day's increment, easily recovered later.

Under the cheap-water price tiers being introduced (α₂ = 0.0004), the
problem becomes worse: the optimizer has even less reason to stop
pumping water in. **Without a direct quadratic penalty on x1 > FC, the
α₂ sweep would generate pathological "drown the field" policies in any
scenario where the α₄ ponding penalty is inactive (i.e., dry scenarios
where x5 = 0).**

### 3.2 The functional form

```
J_overFC = α₆ · Σₖ (1/N) Σₙ [max(x1ⁿₖ - FC, 0) / FC]²
```

Quadratic (not linear) so that:
- Small overshoots within ABM transient dynamics (e.g. x1 = 145 mm
  for one day after a rain event) produce a near-zero penalty
  (~0.001 per agent per step).
- Large persistent overshoots (e.g. x1 = 200 mm) produce a steep
  penalty (~0.184 per agent per step).

This shape preserves the ABM's natural transient buffering above FC
(which is physically real — the soil holds water above FC, draining at
5%/day) while strongly discouraging the optimizer from *deliberately
parking* water at high x1.

### 3.3 Reference

Setter, T.L. et al. (1997). *Review of prospects for germplasm
improvement for waterlogging tolerance in wheat, barley and oats*.
Field Crops Research 51(1-2): 85-104. — Establishes that even
flood-tolerant species incur rapid yield loss when root-zone hypoxia
persists beyond 1-2 days.

---

## 4. Sweep groups and what each one tests

| Group | Scenario | Hp | Configs | Tests |
|---|---|---|---|---|
| **A: α₂ price tiers** | dry/100% | 8 | 4 | How does the optimal water policy change as water moves from subsidized to industrial pricing? |
| **B: α₃ drought** | dry/100% | 8 | 2 | Standard ±1-decade regularization sweep (Saltelli 2004) |
| **C: α₄ ponding (wet)** | wet/100% | 14 | 2 | Does increasing α₄ resolve Issue 3 (61-day waterlogging at Hp=14)? Run at Hp=14 deliberately because the anomaly only manifests there. |
| **D: α₆ overshoot** | dry/100% | 8 | 3 | Does α₆ resolve the FC overshoot at modest computational cost? |
| **E: α₆ validation Hp=14** | dry/100% | 14 | 1 | Final proof: does α₆ = 2.0 eliminate the 21.6-waterlog-day anomaly at the horizon length where it was most severe? |

Total: **12 runs**, ~18 hours of compute.

---

## 5. Output file naming and how to interpret results

Each run produces a parquet + JSON pair:
```
mpc_perfect_{scenario}_rice_{budget_pct}pct_Hp{N}_wsens_{name}.parquet
mpc_perfect_{scenario}_rice_{budget_pct}pct_Hp{N}_wsens_{name}.json
```

The JSON `final_metrics` block carries the same keys as the existing
27 runs. Direct comparison across the sweep is straightforward via:

```python
import pandas as pd, json, glob

records = []
for jf in sorted(glob.glob("results/runs/mpc_perfect_*_wsens_*.json")):
    with open(jf) as f:
        d = json.load(f)
    records.append({
        "config":    jf.split("_wsens_")[-1].replace(".json", ""),
        "scenario":  d["scenario"],
        "Hp":        int(d["controller"].split("Hp")[-1]),
        **d["final_metrics"],
    })
df = pd.DataFrame(records)
```

For each sweep group, the headline plots in the thesis are expected to be:

- **Group A (α₂)**: yield vs water_used, parameterized by α₂ → traces the
  Pareto frontier between biomass and water as water price rises.
- **Group C (α₄ wet)**: waterlog_days_per_agent vs α₄ → demonstrates
  that α₄ tuning resolves Issue 3.
- **Groups D+E (α₆)**: x1 trajectory plots showing the FC-overshoot
  shrinking as α₆ increases; final demonstration at Hp=14 closes Issue 1.

---

## 6. Items deliberately left unchanged

The following design decisions were considered and explicitly rejected
during the v2.3 cost-function update:

1. **IPOPT solver tolerances** (`acceptable_tol`, `max_iter`) — left at
   their existing values. Tightening them would slow solves significantly
   without addressing the root cause (the cost function shape).

2. **ABM physics** — the ABM allows x1 > FC by design (soil holds
   gravitational water above FC, drains at 5%/day). This is correct
   physics and is not modified.

3. **The smooth approximation parameter ε = 0.01** — already validated
   as preserving mass-balance to <0.05 mm over the 93-day season. Not
   modified.

4. **The 27 existing nominal-weight runs** — fully valid and unchanged.
   The α₆ default of 0.0 ensures bit-identical reproduction of those
   results when the nominal weights are used.
