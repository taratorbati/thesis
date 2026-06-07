# THESIS HANDOFF v7 — Staged Refinement Plan (TD3 → MPC parity, stably)

Entry point for the next session. Part 1 is the verified state and evidence;
Part 2 is the staged plan with decision gates; Part 3 is the Stage-1 code that
implements Stage 1 and how to run it. Nothing here has been run — these are
plans and source files only.

---

## PART 1 — VERIFIED STATE (recomputed from committed telemetry/eval files)

### 1.1 The two fixed reference points
- **v2.18 SAC — the standing baseline.** 9-cell mean yield 3785 vs MPC-perfect-Hp8
  3810 (**99.3%**); stable, smooth, reproducible on one seed. Its blemish: it
  fails the wet-year gate — perfect-forecast wet x1 ≈ 135.7 (target < 134) and
  wet waterlog ≈ 36.6 (MPC 18). It also has the **lowest drought of any RL run**
  (its entropy floor keeps it watering).
- **MPC — the benchmark.** A *local* NLP optimum (CasADi + IPOPT, non-convex,
  relaxed tol 1e-4, infeasibility fallback), so RL beating it is possible in
  principle, not a contradiction.

### 1.2 The TD3 line and the problem to fix
v2.19 (drop entropy) collapsed to 0 mm (exploration starvation). v2.19b/c fixed
that (learning_starts 25k, sustained noise floor 0.40→0.15, symmetric LR,
collapse guard, deterministic dev-set eval). v2.20 added the MPC delta-u reward
r5 (α5=0.005). The open problems:

1. **Critic divergence.** v2.20 r5 (ls=50k, dev {2002,2004,2013}) q_pred drifts
   monotonically to **−222** and never recovers; guard trips once at 184k.
2. **Pulsing.** Perfect-forecast wet mean|Δu| = **2.51** (MPC 0.98); r5 at
   α5=0.005 did **not** smooth (pre-r5 TD3 was 2.12).
3. **Drought seesaw.** Every TD3 variant is worse than SAC on drought-days
   (perfect dry/mod/wet): SAC 32.2/26.7/17.9; v2.20 r5 44.4/48.9/32.2;
   MPC 18.8/18.3/10.8 — TD3 buys waterlog compliance by under-watering and pays
   in drought with **no net yield gain** (RL spends ~the same water as MPC).
4. **Multistability.** Same ls/reward, different dev-year set → opposite outcome
   (a v2.19c-style dev set was healthy; {2002,2004,2013} diverged).

### 1.3 Mechanisms (what's actually going on)
- **Critic shape.** The VDN critic is a **scalar sum per twin**
  (`q_total = local_q.sum(dim=1)`); twin-min is over two scalars, not 130
  per-cell values. Divergence is **bootstrap-horizon amplification** (γ=0.99 over
  93 steps ≈ 100× effective horizon) feeding an actor↔critic **limit cycle**
  (it recovers in 2 of 4 runs), **not** the v2.7 deadly-triad blow-up (which hit
  1e12). VDN-sum is adequate — it is *not* the bug.
- **Dive onset tracks `learning_starts`** in all four runs (~27k / ~51k / ~26k /
  ~51k for ls 25/50/25/50k): the instability is seeded the moment gradient
  updates begin and the deterministic actor starts moving on an untrained critic.
  → motivates both n-step (bound the bootstrap) and the actor-LR warm-up
  (let the critic lead).
- **Multistability source.** `climate_data.py`: `TRAINING_YEARS = 2000–2025 −
  EVAL_YEARS{2018,2022,2024} − DEV_YEARS`. A fixed-seed `np_random.choice` draws
  the same *indices* but different *years* when the dev set changes the pool, so
  changing the eval set silently changes what the agent trains on.
- **Horizon is the divergence driver (proof).** v2.6 (early termination ≈ 50-step
  horizon): critic loss < 75 for 500k steps. v2.7 (full 93-step horizon):
  |Q| → 400 then cascade. Identical otherwise.
- **n-step history (corrected).** The earlier "missing entropy-bonus bug" claim
  was wrong. The real reasons the v2.10 E3 n-step buffer didn't help: (1) it
  applied γ¹ not γⁿ in the target (a documented approximation); (2) a stochastic
  entropy critic needs importance sampling for multi-step off-policy returns (the
  SAC(λ)/Truncated-TD(λ) paper in-repo, never implemented); (3) it was bolted to
  TQC, whose quantiles collapse on a VDN-sum scalar. **TD3 removes all three** —
  deterministic target (uncorrected small-n is standard: D4PG, Ape-X), and the
  right critic.

### 1.4 Standing decisions
- **Dev set {2002, 2004, 2013} stays.** Train/eval must be mutually exclusive;
  sample-order sensitivity is a *training-stability* problem to fix in the
  optimiser, not by picking a lucky split. (No dev-set handpicking.)
- **Seeds are the acceptance test, not a victory lap.** "Mistake 5": v2.7 "beat
  MPC in 4 cells" vanished on seed 1. A per-cell "beats MPC" claim is valid only
  on the seed **mean**.
- Budget ≈ 10–12 runs (~1 compute unit / 250k steps on A100; N=5 is field
  standard). Refine first, then seed.

---

## PART 2 — THE STAGED PLAN

Philosophy: change **one major thing per run**, pre-register success **before**
the run, judge every run against the same scorecard, and let seeds be the
acceptance test.

### Stage 0 — Lock the harness (0 runs, do first)
- **0.1 Acceptance scorecard** (§2 below) — built into the eval aggregator so
  it's computed for every run, including the season-long FC-band tracking metric
  (fraction of cell-days with x1 ∈ [ST, FC] = [112, 140]).
- **0.2 Naming + manifest.** `td3_<version>_<key-change>_seed<k>` and a
  per-run `manifest.json` (git SHA, full hyperparameters, dev/eval split, seed,
  one-line hypothesis). Kills the "everything is v2.19b" ambiguity. *(Done in the
  v2.20 trainer.)*
- **0.3 Checkpoint/save policy.** Model checkpoint every 25k (also the cascade
  sweep). Telemetry CSVs flush every 2k. **Do not** snapshot the full replay
  buffer every 25k — the v2.4 "zombie" died from 124 GB of buffer snapshots; use
  one rolling snapshot or skip. Resume from latest model + manifest.
- **0.4 Re-score v2.18 SAC and v2.19b TD3** through the fixed scorecard (re-eval
  only) so every later delta is apples-to-apples.

### Stage 1 — Stabilise the critic (2 runs, +1 if needed) — CODE PROVIDED
Two angles on the learning_starts-onset dive.
- **Run A — n-step alone.** Exact-γⁿ n-step (n=5), everything else = the
  diverging v2.20 r5 run (ls=50k, r5=0.005). Isolates n-step's effect on the
  divergence.
- **Run B — + damping package.** policy_delay 2→3, target_policy_noise 0.2→0.3
  (clip 0.5), actor-LR warm-up 0→full over the first 25k updates (critic leads).
  Isolates incremental damping. These are TD3's **own structural** stabilisers,
  not the "Mistake 4" delay tactics (global LR/τ/grad-norm throttling).
- **1.3 Multistability checkpoint.** If the winner's q_pred still shows a deep
  excursion (even one that recovers), run a 2nd seed before Stage 2; if clean
  (|q_pred| within ~2–3× realised-return scale and guard never trips), fold the
  multistability test into Stage 4.
- **Gate:** q_pred bounded, eval improves ~monotonically, final ≈ best → proceed.
  Reduced-but-not-bounded → Run B. No effect → fall back (n=3, or lead with
  damping).

### Stage 2 — Pulsing + drought seesaw (1–2 runs)
- **2.1 Calibrate offline (0 runs).** Use the season-sum magnitude method that
  set ALPHA6_LIN: compute season-sums of r1/r3/r6 and a terminal-yield term over
  the 26-year distribution, then set α3 and the yield-term weight to a deliberate
  balance against r6 (drought caps −0.1 vs waterlog −0.86 — that asymmetry is why
  the optimiser rations).
- **Run C — drought + smoothing.** Stage-1 winner + {α3↑, terminal-yield term,
  **Markov r5** (prev_u in obs — see Part 3 / Stage-2 note), α5 active}. Gate:
  dry drought ↓ toward MPC, mean|Δu| ↓ toward ~1.0, no yield loss, stability
  preserved.
- **Run D (conditional).** If smoothing still doesn't bite: raise α5 ×3–5, or
  drop r5 entirely rather than carry a dead term.

### Stage 3 — Spatial competence (2–3 runs)
- **3.1 Spatial reward term (0 runs design).** Reward *allocation* not just
  aggregate compliance — penalise cross-cell stress dispersion, or reward
  matching the cascade-implied need (D8 routing). Calibrate magnitude.
- **Run E — GNN actor** over the D8 adjacency, replacing the param-shared MLP;
  keep the VDN-sum critic (isolate the actor change). Gate: corr(u, elevation)
  and per-cell allocation improve, stability preserved.
- **Run F (optional) — QMIX/QPLEX mixer** for non-additive spatial credit
  (expressivity upgrade, not a stability fix). **TQC stays excluded** (quantiles
  collapse on VDN-sum; adds pessimism to an over-negative critic).

### Stage 4 — Seed reproduction = the acceptance test (2–4 runs) — CODE PROVIDED
Freeze one config; run 2–4 seeds (N=5 ideal). Decide on three things: (a) the
**mean** meets the scorecard; (b) every seed converges (final ≈ best, q_pred
bounded) — the real multistability verdict; (c) forecast sensitivity holds under
noise.

### Stage 5 — Thesis integration (both outcomes publishable)
Solid contributions regardless of the seed verdict: the stability diagnosis
(bootstrap-horizon divergence, v2.6 vs v2.7), the n-step-as-horizon-control
result, and the reward-asymmetry / drought-seesaw finding. If seeds confirm
stable MPC parity-or-better → headline result. If not → v2.18 SAC remains the
published baseline and the TD3 + n-step line is a characterised partial/negative
result (as v2.8/v2.9 already are). The contribution is the diagnosis + method,
not a single yield number.

---

## 2. ACCEPTANCE SCORECARD (pre-registered; MPC perfect-Hp8 reference)

| Metric | MPC ref | Target | Stretch |
|---|---|---|---|
| 9-cell mean yield | 3810 | ≥ 3790 (≥99.5%) | > 3810 |
| Dry-year drought days/agent | 18.8 | ≤ 22 | ≤ 19 |
| Wet-year waterlog days/agent | 18.0 | ≤ 20 | ≤ 18 |
| Wet x1 (FC=140) | 130 | 128–134 pooled; low mean\|x1−FC\| | ≈130 |
| FC-band tracking (x1 ∈ [112,140]) | (MPC profile) | match MPC fraction | — |
| Smoothness mean\|Δu\| | 0.98 | ≤ 1.2 | ≤ 1.0 |
| Forecast sensitivity (noisy/perfect yield) | 99.8% | ≥ 98.5% | ≥ 99.5% |
| Spatial corr(u, elevation) | +0.96 | positive, \|·\| ≥ 0.3 | match MPC allocation |
| Stability | final≈best, \|Q\| bounded | final within 3% of best; guard never trips | identical across seeds |

---

## PART 3 — STAGE-1 CODE (this handoff)

Drop all six files into the repo, mirroring the paths below (they import via
`src.rl....` and `from climate_data import ...`, same as `train_v219b_td3.py`).

| File | Role |
|---|---|
| `src/rl/nstep_buffer_exact.py` | Exact-γⁿ n-step replay buffer (clean reimpl.) |
| `src/rl/td3_warmup.py` | `WarmupAsymmetricLRTD3`: actor-LR warm-up (critic-leads) |
| `src/rl/gym_env_prev_u.py` | `IrrigationEnvPrevU`: prev_u in obs (Stage 2; off now) |
| `src/rl/configs_v220.py` | `CONFIGS["A"]` (n-step), `CONFIGS["B"]` (n-step + damping) |
| `src/rl/train_v220_td3.py` | Trainer: v2.19b machinery + n-step + warm-up + manifest |
| `src/rl/run_seeds.py` | Stage-4 resumable seed driver |

### 3.1 The exact-γⁿ trick (no `train()` override)
The buffer accumulates `R_n = Σ_{k<n} γ_base^k r_{t+k}` (truncated at any done)
with **its own** `γ_base = 0.99`. The trainer sets the **model's** gamma to
`γ_base ** n`. SB3's stock TD3 target is then, for every sampled transition,
`R_n + (1−done)·γ_base^n·Q(s_{t+n})` — exactly the n-step target (terminal
transitions zero the bootstrap via `(1−done)`, so the discount value there is
irrelevant). The critic still learns the `γ_base`-discounted return, so the
`bias_ratio` q_pred-vs-realised-return diagnostic stays on the same scale. This
is precisely the decoupling the v2.10 E3 buffer lacked.

### 3.2 What Run A vs Run B change (one variable each)
- **Run A** = the diverging v2.20 r5 config **+ exact n-step (n=5)**. Stock TD3
  knobs (policy_delay 2, target noise 0.2), no warm-up.
- **Run B** = Run A **+ damping** (policy_delay 3, target noise 0.3, actor-LR
  warm-up 25k updates). Bundles three knobs for budget; a follow-up could ablate
  which mattered.

### 3.3 Run commands
```bash
# Stage 1
python -m src.rl.train_v220_td3 --config A --seed 0      # n-step alone
python -m src.rl.train_v220_td3 --config B --seed 0      # n-step + damping
# (optional smoke / quick look)
python -m src.rl.train_v220_td3 --config A --seed 0 --total-timesteps 60000

# Stage 4 (after a config is frozen)
python -m src.rl.run_seeds --config A --seeds 0 1 2
```
Each run writes to `results/rl/td3_v220_<label>_seed<k>/`: `manifest.json`,
`best_model/`, `<run>_final.zip`, `checkpoints/` (every 25k), `eval_logs/`, and
the telemetry CSVs (`bias_ratio_log`, `collapse_guard_log`,
`low_action_coverage_log`, `exploration_sigma_log`, `nonfinite_guard_log`). The
seed driver writes a resumable `seed_campaign_<config>_<label>.json`.

### 3.4 Stage-1 decision gate (what to read off the run)
Look at `bias_ratio` q_pred over training and the eval-reward curve:
- **bounded q_pred + eval improving + final ≈ best + guard never trips** →
  n-step worked; go to 1.3 (multistability check) then Stage 2.
- **dip reduced but still deep** → run Run B (damping).
- **no change** → fall back to n=3 or lead with damping; re-open the diagnosis.

### Stage-2 NOTE — `prev_u` also needs a network change (not just the env)
`gym_env_prev_u.py` is complete on the env side (obs 1097→1227, prev_u as a 9th
agent-major feature). But `networks_td3.py` hard-codes the per-agent feature
count for **both** actor and critic (`TD3_N_AGENT_FEATURES = V27_N_AGENT_FEATURES
= 8`, per-agent input 65, `features_dim` assert 1097). To switch prev_u on you
must also provide a 9-feature actor+critic variant (`_N_AGENT_FEATURES=9`,
per-agent input 66, assert 1227) — **do not** edit the shared `V27_*` constants
in place (the v2.16–v2.18 SAC family reuses them). Until then keep
`expose_prev_u=False`; the trainer raises a clear error if it is True.

---

## OPEN QUESTIONS / TO DECIDE
- n=5 vs n=3 (revisit if Run A is noisy but bounded).
- Whether damping is needed at all (skip Run B if Run A is clean).
- Stage-2 yield-term form (terminal vs per-step) and α3 magnitude (calibrate
  offline first).
- GNN actor architecture over D8 (Stage 3) and whether a mixer earns its keep.
- Seed count for Stage 4 (3 vs 5) given final budget.
