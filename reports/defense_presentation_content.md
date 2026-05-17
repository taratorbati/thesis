# Defense Presentation — Slide Content
## Tara Torbati — ITMO University, R4237c
## Modern Control Methods for Agricultural Irrigation

> Slide-by-slide content for the deck that accompanies the 10-minute defense speech.
> Each slide is sized for high-information density on a 16:9 layout.
> "Visual" describes what should be on the slide; "Speaker notes" mirror the speech.
> 6 slides total, matching the speech section markers.

---

## SLIDE 1 — Title

### Visual

**Main title (large, top-left):**
Modern Control Methods for Agricultural Irrigation

**Subtitle (one line under title, accent color):**
MPC vs. Reinforcement Learning in Water-Constrained Crop Systems

**Author block (mid-page):**
Tara Torbati  |  ITMO University, R4237c
Supervisor: Peregudin A. A.
Preliminary Defense — 2026

**Site identifier (bottom):**
Study site: Gilan Province, Iran  •  6 ha Hashemi rice field  •  38.298°N, 48.847°E

### Speaker notes
Open with name, thesis title, supervisor. Establish that this is a controlled comparison on a real site.

---

## SLIDE 2 — Motivation

### Visual

**Three metric cards (top row):**

| 70 % | 40 % | 238 000 |
|---|---|---|
| of global freshwater | of world population | hectares of paddy |
| consumed by agriculture | under water stress by 2030 | in Gilan Province alone |

**Section title (below cards):**
The Control Engineering Gap

**Four bullets:**
- Traditional irrigation in Iran is reactive and open-loop — no forecast, no constraint awareness
- MPC can enforce strict seasonal water budgets but requires expensive online optimisation
- RL offers millisecond-latency decisions after training — but constraint satisfaction is not guaranteed
- This thesis rigorously benchmarks both on a high-fidelity ABM of a real Iranian rice field

### Speaker notes
Anchor the global problem first, then narrow to Iran/Gilan, then state the engineering tradeoff that motivates the comparison.

---

## SLIDE 3 — Simulation Environment

### Visual

**Left column — bullets:**
- 130 crop-soil agents on a 10 × 13 DEM grid
- Elevation range 74–181 m (Talish Mountains)
- 25-year NASA POWER climate dataset (2000–2025)
- Cascade water routing — eliminates "bathtub" mass-balance errors
- Decoupled surface & subsurface hydrology (corrects published baseline)
- Drought stress integrated into biomass eq. (FAO AquaCrop approach)
- Validated against NASA MERRA-2 GWETROOT: **r = 0.74**

**Right column — per-agent state vector (5 colored cards):**

| State | Description |
|---|---|
| x₁ | Root moisture (mm) |
| x₂ | Thermal time (GDD) |
| x₃ | Maturation stress |
| x₄ | Biomass (g/m²) |
| x₅ | Surface ponding (mm) |

**Bottom strip — three scenario cards:**

| Dry 2022 | Moderate 2018 | Wet 2024 |
|---|---|---|
| 39.7 mm | 108.8 mm | 176.8 mm |
| seasonal rainfall | seasonal rainfall | seasonal rainfall |

### Speaker notes
Three points must land:
1. The ABM is faithful to the real site (climate, topography, soil data sources)
2. Three mathematical corrections were necessary — establishes that the work is *not* a black-box reuse of a published framework
3. The model is validated independently against satellite observations

---

## SLIDE 4 — MPC Design & Results

### Visual

**Top half — Cost function header:**
Five-term cost J  (recommended operating point α*)

**Five colored bars (left-to-right):**

| Symbol | Term | Note |
|---|---|---|
| α₁ = 1.0 | Terminal biomass reward | anchor term |
| α₂ = 0.016 | Water cost (Iranian domestic-base tariff) | 7 000 toman/m³ |
| α₃ = 0.1 | Drought stress regulariser | <1% yield sensitivity |
| α₅ = 0.005 | Actuator smoothing ΔU² | tie-breaking |
| α₆ = 8.0 | FC-overshoot soft penalty (x₁ > FC) | eliminates waterlogging |

**Status strip:**
**Hp\* = 8 days**  |  3 120 decision vars, 2 081 constraints (per step)  |  warm-start: 2.5× solver speedup  |  solver: CasADi + IPOPT

**Bottom half — Results bar chart:**
Grouped bar chart — Yield (kg/ha) at 100% budget across {Dry, Moderate, Wet} for:
- No Irrigation (gray)
- Fixed Schedule (teal)
- MPC Hp=8 (red accent)

Data points:
- Dry: 1462 / 3607 / 4145
- Moderate: 1478 / 3302 / (TBD)
- Wet: 2243 / 2790 / 3759

**Three callouts (right side):**

| +14.9 % | +34.7 % | 64 % |
|---|---|---|
| vs fixed-schedule | vs fixed-schedule | budget used |
| dry / 100 % | wet / 100 % | wet year |

### Speaker notes
Top-half is design (cost function calibration), bottom-half is empirical validation. Most important data point: the **wet-year 64% budget utilization** — proves the MPC reasons about forecast.

---

## SLIDE 5 — SAC Design & The VDN Upgrade

### Visual

**Top half — Architecture diagram (two boxes):**

**Left box — Actor (deployed online):**
- Input: 62-dim per-agent obs (5 local + 57 global)
- Arch: 62 → 128 → 128 → 1
- Shared weights across all 130 agents
- Output: 130 actions in ~1 ms (CPU)

**Right box — Critic (training only):**
- VDN factorisation: **Q_total = Σ_n Q_loc(s_n, g, a_n)**
- Local MLP: 63 → 256 → 256 → 1, shared across agents
- Twin Q (clipped double-Q for stability)
- Never used at deployment

Between them: **CTDE** label with arrow

**Bottom half — Two-column comparison:**

**Left column — "v2.4 Pilot Diagnosis":**
- Monolithic critic (837 → 256 → 256 → 1)
- Critic loss exploded at step 100k: 0.67 → 84 693
- Policy collapsed to flat 2 mm/day across all 130 agents
- Used MORE water in wet year (231 mm) than dry (190 mm) — opposite of correct
- Root cause: **spatial credit assignment failure**

**Right column — "v2.6 Architectural Fix":**
- VDN factorised critic — per-agent gradient signal restored
- c_term = 0 (removes terminal bootstrapping trap)
- ent_coef = 0.05 hardcoded (no auto-tuner panic)
- α₅ = 0 in RL reward (allows weather-responsive switching)
- max_grad_norm = 1.0, LR decay
- 5-seed campaign currently training

### Speaker notes
This is the most important slide. The narrative must be:
1. Initial design was rigorous but exhibited two diagnostic failures
2. The failures were properly *diagnosed*, not papered over
3. The VDN fix is principled (matches the reward's additive structure) and minimal
4. Be transparent — the committee will respect honesty about what didn't work

---

## SLIDE 6 — Comparison & Conclusions

### Visual

**Top — Head-to-head bar chart:**
Yield delta vs. Fixed Schedule (kg/ha) — positive = better. Two bar series:
- MPC vs Fixed (teal): +538, +675, +648, +969, +842, +730
- SAC v2.4 pilot vs Fixed (amber): −1130, −910, −617, +355, +247, +131

(Categories: Dry/100%, Dry/85%, Dry/70%, Wet/100%, Wet/85%, Wet/70%)

**Two summary lines below chart:**
- **MPC Hp=8:** +492 to +969 kg/ha over fixed-schedule across all completed cells
- **SAC v2.4 pilot:** below fixed-schedule in dry, above in wet (coincidental — not scenario-aware)

**Middle — Computational comparison (two large numbers):**

| MPC mean per-decision | SAC mean per-decision |
|---|---|
| **25.9 min** | **~1 ms** |
| wall-clock per day | CPU inference |

Center callout: **25 000× faster**

**Bottom — Three conclusions with status icons:**

- ✓ **H1 confirmed** — MPC outperforms fixed schedule by 492–969 kg/ha; waterlogging virtually eliminated
- ◎ **H2 in progress** — Pilot SAC pathologies diagnosed (Q-divergence + spatial homogenisation); v2.6 with VDN critic currently training
- ⏳ **H3 pending** — Forecast-noise evaluation framework operational; results due before final defense

**Final line (bottom, accent color):**
Remaining for final defense: complete v2.6 5-seed training, noisy-forecast robustness analysis, moderate-year MPC Hp=8 cells.

### Speaker notes
Close with confidence. The framing is "the thesis already proves what it set out to prove on the MPC side, has diagnosed (not glossed over) the issues on the RL side, and has a clearly scoped path to completion." This is what a committee wants to hear at a preliminary defense.

---

## Design notes for the deck

**Color palette:**
- Deep navy (#0D1B3E) — dominant background
- Teal (#0B9B8A) — primary accent
- Light teal (#14B8A6) — secondary accent
- Off-white (#F0F4F8) — body text on dark
- Red (#E05252) — MPC / negative deltas
- Amber (#F59E0B) — SAC / warnings
- Green (#22C55E) — positive outcomes

**Typography:**
- Sans-serif (Calibri or Inter)
- Titles 24pt bold, body 13–15pt
- Numbers in metric cards: 30pt+ bold accent color

**Layout principles:**
- Title bar with thin teal rule under each slide
- Max 4 bullets per column
- Bullets are short headlines, not full sentences
- Charts use consistent palette across all slides
- Avoid icons unless they reinforce a metric (✓, ◎, ⏳)

**Animations:** None. Static slides. The committee should be able to absorb each slide in 90 seconds.

**Backup slides (if asked) — not in main deck but prepared:**
- B1: Detailed ABM mass-balance corrections
- B2: MPC sensitivity sweep — 33 configurations
- B3: WandB training curves (v2.4 pilot, three phases)
- B4: VDN gradient localization unit test results
- B5: 25-year climatology with scenario years highlighted
- B6: Iranian water tariff context & economic interpretation
