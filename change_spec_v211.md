# Change Specification — SAC v2.11 (LayerNorm critic — cascade prevention)

**Project:** ITMO MSc thesis — *Modern Control Methods for Agricultural Irrigation*
**Author of record:** Tara Torbati
**Document purpose:** Specify the v2.7 → v2.11 architecture change. A single intervention — `LayerNorm` inserted after each hidden Linear layer in the VDN critic — is added to test the hypothesis that the v2.7 deadly-triad cascade is driven by neural-network optimization dynamics rather than by an unavoidable trade-off between bootstrap leverage and credit-assignment horizon. All other v2.7 design choices are preserved exactly.

---

## 1 — Motivation

### 1.1 What v2.7 demonstrated, and what it cost

The v2.7 SAC controller matched MPC at 6 of 9 grid cells under perfect forecast (dry/all, moderate/all, wet/70%) and trailed MPC by 6–8% on wet/100% and wet/85%. Cross-seed evaluation showed meaningful variance: v2.7 seed-1 produced yields 6–8% below seed-0 on dry/100% and dry/85%, while wet cells were within 2%. The published v2.7 best_model achieves these numbers at training step 200k.

This result, however, was extracted in a 25k-step window immediately preceding the catastrophic divergence of the critic. The v2.7 seed-0 training log shows `critic_loss` growing from 1.07 at step 100k to 11.8 at step 150k, 98.5 at step 160k, 803 at step 200k, 23,600 at step 250k, and 1.2 × 10¹⁰ at step 300k. The cascade is reproducible across seeds: seed-1 cascades at step 165k ± 5k. Past step 250k the policy yields collapse by 25–30% across all 9 cells.

The Phase 1 study (v2.10 experiments E2, E3, E4) interpreted the cascade as a deadly-triad failure mode (Van Hasselt et al. 2018) driven by the high bootstrap leverage `1/(1−γ) ≈ 100` over the 93-day rice season. Three remediations were tested at the algorithm or hyperparameter level:

- **E2 — TQC k=5 truncation (Kuznetsov et al. 2020).** Per-agent quantile distributions were summed across the 130 agents before truncating the top k. The central-limit-theorem narrowing of the summed distribution made the truncation structurally inert: the top quantiles to be dropped were arbitrarily close to the median. The cascade fired at step 175k with the same signature as v2.7.
- **E3 — TQC k=5 + custom n-step replay buffer.** The buffer was intended to accumulate `n` consecutive rewards and bootstrap from `s_{t+n}`. Two bugs deflated the soft-Bellman target on intermediate steps: the entropy bonus `−α log π` was dropped for `t+1, …, t+n−1`, and the bootstrap used `γ^1` instead of `γ^n`. The net effect was a ~8-unit per-transition target under-bias. The result was a negative Q cascade (q_inflation_pct = −44% at step 25k, −339% at step 100k) — the same divergence mechanism with the sign reversed.
- **E4 — SAC + γ = 0.98.** This succeeded at preventing the cascade (q_inflation_pct stayed in [−16%, −5%] through 250k) but produced a policy that irrigated near 6 mm/day uniformly regardless of cell, year, or rainfall. Across the 9 cells the yields fell by a mean of 256 kg/ha vs v2.7, with the worst regressions on wet/100% (−16%). The effective horizon `1/(1−γ) = 50` is shorter than the 93-day season, so credit assignment from late-season biomass back to early-season irrigation decisions could not propagate.

Phase 1 thus characterised a **trade-off surface** between cascade prevention and credit-assignment horizon, parameterised by γ. The Phase 1 conclusion was that no γ within `[0.97, 0.99]` jointly satisfies both constraints with the existing architecture.

### 1.2 Why v2.11 exists — a different cascade hypothesis

Two recent peer-reviewed analyses of Q-value divergence in deep RL identify a mechanism distinct from bootstrap leverage:

- **Yue, Kang, Shi, Ma, Liu, Zhao (NeurIPS 2023, arXiv:2310.04411).** Using Neural Tangent Kernel (NTK) theory, the authors derive a sufficient condition for Q-function divergence in deep RL: the *Self-Excite Eigenvalue Measure* (SEEM) of the critic's normalised kernel matrix exceeding a critical threshold. When SEEM > threshold, the critic's own gradient updates amplify its prediction errors *polynomially* in training step — a property of the network's optimization dynamics, not of the Bellman operator's fixed-point structure. The authors empirically validate this prediction on offline RL benchmarks and show that **`LayerNorm` inserted after each hidden Linear in the critic reliably keeps SEEM below the threshold, eliminating Q-divergence with no detrimental bias on the learned policy.**
- **Nauman, Bortkiewicz, Miłoś, Trzciński, Ostaszewski, Cygan (RLC 2024, arXiv:2403.05996).** Confirms Yue et al.'s prediction in *online* RL on the dm_control suite. Demonstrates that LayerNorm or a related unit-ball normalisation at the critic's penultimate features stabilises training at high update-to-data ratios (UTD), where standard SAC consistently diverges.

The v2.7 cascade signature is consistent with the SEEM mechanism. `critic_loss` grows roughly factor-of-10 per 10k steps across twelve orders of magnitude (1.07 → 1.2 × 10¹⁰). Clean exponential growth is the optimization-dynamics signature; bifurcation in the Bellman operator would more typically saturate or oscillate. The reproducibility of cascade onset across seeds (within ±5k steps) further indicates a deterministic property of the training dynamics rather than a stochastic basin escape.

**The hypothesis v2.11 tests:** the v2.7 cascade can be prevented by suppressing SEEM via `LayerNorm` in the critic, *without* reducing γ and *without* shortening the credit-assignment horizon. If true, the Phase 1 trade-off surface is not load-bearing — the cascade is an artefact of an under-regularised critic, and v2.7-quality policies should be reachable with stable late-training dynamics. If false, the cascade is bifurcation-driven and a different intervention is needed (the open question is then between an actor-side entropy renormalisation, a reinstated burn-rate penalty, or a narrow-band γ reduction).

---

## 2 — Goals and non-goals of v2.11

### Goals

- Insert `nn.LayerNorm(hidden_dim)` after each hidden `nn.Linear` layer in the VDN critic, before the ReLU activation. This matches the placement Yue et al. 2023 validate.
- Preserve every other component of the v2.7 baseline exactly: actor architecture (`_V27SharedActor`, 8 features per agent, 128–128 MLP), observation layout (1097-dim total, 8 features × 130 agents + 57 global), reward function (`r = r1 + r2 + r3 + r6` with ALPHA1=1.0, ALPHA2=0.016, ALPHA3=0.1, ALPHA6=8.0), γ=0.99, τ=0.005, fixed `ent_coef=0.05`, LR 3e-4 → 5e-5 linear, batch 256, 250 000 total steps, learning_starts=1000, gradient_steps=1, train_freq=1, MAX_GRAD_NORM=1.0.
- Preserve full backwards compatibility of evaluation tooling. The runner's checkpoint-detection logic must continue to dispatch v2.7, v2.6, v2.8 and pre-VDN monolithic checkpoints to their respective policy classes, and additionally dispatch v2.11 checkpoints to a new `V211CTDESACPolicy`.

### Non-goals

- No change to the reward function or environment dynamics.
- No change to the actor architecture, the observation layout, or the SAC algorithm class.
- No change to γ, ent_coef, learning rates, batch size, total timesteps, or any other v2.7 hyperparameter.
- No change to the MPC controller, baselines, or evaluation harness (`scripts/experiments/exp_rl.py`, `scripts/analysis/*`).
- No change to `abm.py`, `soil_data.py`, `climate_data.py`, `src/precompute.py`, `src/terrain.py`, `src/rl/gym_env.py`, `src/rl/callbacks_v210.py`, `src/rl/nstep_buffer.py`, `src/rl/networks_tqc.py`, `src/rl/train.py`, `src/rl/train_v210_e2.py`, `src/rl/train_v210_e4.py`.

---

## 3 — Specification

### 3.1 New critic architecture

The per-agent Q-network in the v2.7 critic has the structure

```
Linear(66, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1)
```

(input 66 = 8 per-agent features + 57 global features + 1 action scalar). The v2.11 per-agent Q-network is

```
Linear(66, 256) → LayerNorm(256) → ReLU → Linear(256, 256) → LayerNorm(256) → ReLU → Linear(256, 1)
```

The `LayerNorm` placement (after `Linear`, before `ReLU`) follows Yue et al. 2023 Section 4.3. The output `Linear` layer has no `LayerNorm` (per the same reference). The twin-Q structure, target network, sum-decomposition across the 130 agents, parameter sharing across agents within a critic head, and observation-feature ordering are all unchanged from v2.7.

LayerNorm normalises along the feature dimension of the `(B·N, hidden_dim)` tensor flowing through the per-agent MLP — i.e. each (batch element × agent) row is normalised to zero mean and unit variance across the 256 hidden units, then scaled and shifted by learned per-feature affine parameters γ and β.

Parameter overhead: 2 critics × 2 hidden layers × 2 × 256 = **2048 extra parameters**, against the ~166 000 parameters of the v2.7 twin critic. Negligible compute overhead per forward/backward pass.

### 3.2 New checkpoint-detection signal

A v2.11 critic's `state_dict` contains a 1-D LayerNorm gamma at index 1 of each per-agent Q-net: `critic.qf{0,1}.1.weight` with shape `(hidden_dim,)`. The v2.7 critic has no key at index 1 (index 1 is `ReLU`, which has no parameters). The runner uses this signal:

```python
has_layernorm = (
    'critic.qf0.1.weight' in state_dict
    and state_dict['critic.qf0.1.weight'].ndim == 1
)
```

For `dim=66, key_fmt='flat'`, the new dispatch table is:

- `(66, 'flat', has_layernorm=True)` → `V211CTDESACPolicy` (v2.11), obs_layout `'v27'`.
- `(66, 'flat', has_layernorm=False)` → `V27CTDESACPolicy` (v2.7), obs_layout `'v27'`.

All other detection branches (`dim=63 wrapped`, `dim=63 flat`, `dim=67 flat`, `dim=837 flat`) are unchanged.

### 3.3 Acceptance criteria

**Primary — cascade suppression.** All three must hold throughout the 250k training run:

1. `|q_inflation_pct| < 30%` at every BiasRatioCallback measurement (computed at every 25k-step boundary). v2.7 hits +209% at step 200k.
2. The 1k-step rolling mean of `critic_loss` never exceeds 50 past step 100k. v2.7 hits 23 600 at step 250k.
3. `actor/std/spatial` (the mean per-agent action standard deviation produced by ActionStatsCallback) stays in [0.20, 0.40]. v2.7 drops to <0.10 post-cascade.

**Secondary — policy quality preserved.** On the 9-cell perfect-forecast grid with the deterministic actor:

4. Yield within ±3% of v2.7 best_model in each cell. Concretely: dry/100% ≥ 4040, dry/85% ≥ 3979, dry/70% ≥ 3653, moderate/100% ≥ 3618, moderate/85% ≥ 3625, moderate/70% ≥ 3481, wet/100% ≥ 3331, wet/85% ≥ 3329, wet/70% ≥ 3387.

**Stretch — improvement on v2.7.** If any of:

5. The 9-cell mean perfect-forecast yield exceeds v2.7's by ≥ 50 kg/ha across cells.
6. Late-training (post-step-200k) checkpoints exceed step-200k yields by ≥ 50 kg/ha on a majority of cells (i.e. the architecture genuinely benefits from training beyond the v2.7 cascade window).
7. Wet/100% yield exceeds 3500 (closing >30% of the v2.7 wet-year gap to MPC).

Then v2.11 supersedes v2.7 as the Phase 1 baseline and a follow-up Phase 2 (domain randomisation + wet-year oversampling) launches from v2.11.

### 3.4 Early-kill rule

If at any BiasRatioCallback measurement past step 150k **both** `q_inflation_pct > 100%` **and** `actor/std/spatial < 0.15` hold, the run is stopped. LayerNorm has failed to suppress the cascade on this problem. The pre-cascade best_model checkpoint is retained as the v2.11 artefact, and the conclusion is "LayerNorm insufficient — cascade hypothesis is bifurcation-driven, not NTK-driven."

The follow-up experiment in that case combines LayerNorm with a narrow-band γ reduction (`γ = 0.985`, effective horizon 67 — between v2.7's 100 and E4's 50). This is an intermediate point on the Phase 1 trade-off surface that v2.10 did not test.

---

## 4 — Hypotheses and pre-registered predictions

Following Yue et al. 2023 and Nauman et al. 2024, the v2.11 run distinguishes three outcomes:

**Outcome A — LayerNorm fully suppresses the cascade.** `q_inflation_pct` stays in [−30%, +30%] throughout 250k. Eval reward continues to climb past step 200k (where v2.7 peaked). 9-cell yields meet or exceed v2.7's. This is consistent with the NTK-driven divergence hypothesis. *Posterior weight on this outcome: 50%.*

**Outcome B — LayerNorm partially suppresses the cascade.** `q_inflation_pct` stays below v2.7's +209% but exceeds the 30% acceptance threshold; the cascade is delayed by 50k–100k steps but eventually fires; eval reward improves marginally. This is consistent with the cascade being driven by *both* NTK dynamics and Bellman bifurcation, with LayerNorm fixing one but not the other. *Posterior weight: 25%.*

**Outcome C — LayerNorm has no detectable effect on the cascade.** `q_inflation_pct` and `critic_loss` follow v2.7's trajectory within seed variance. The cascade is bifurcation-driven, not NTK-driven. The Phase 1 trade-off surface is real and the next intervention must be on the algorithm side (entropy-normalised actor loss, narrow-band γ, or reinstated burn-rate penalty). *Posterior weight: 20%.*

**Outcome D — LayerNorm introduces a new failure mode.** The critic under-fits because LayerNorm is over-regularising the function class; eval reward stays below v2.7's plateau throughout. Yue et al. 2023 explicitly tested for this on offline RL and found no detrimental bias, but the v2.11 problem is multi-agent and the empirical track record on multi-agent factorised critics is thinner. *Posterior weight: 5%.*

Outcomes A and B reproduce the published literature in a new setting (multi-agent VDN factorised critic, long-horizon agricultural control). Outcome C is a novel negative result that motivates further investigation. Outcome D is recoverable by reverting to v2.7.

---

## 5 — Files added or modified

| File | Status | Change |
|------|--------|--------|
| `src/rl/networks.py` | modified | Adds `V211_*` dimension constants, `_V211FactorizedQNet`, `_V211FactorizedContinuousCritic`, `V211CTDESACPolicy`. ~80 lines added. All v2.6, v2.7, v2.8 classes byte-identical. Header comment updated. |
| `src/rl/runner.py` | modified | `_detect_critic_arch` returns a 3-tuple including a `has_layernorm: bool` flag. `_load_sac_model` adds the `(66, flat, True)` → `V211CTDESACPolicy` branch. `_detect_critic_input_dim` updated to unpack the 3-tuple. Backwards-compatible: all five prior dispatch branches continue to fire unchanged. |
| `tests/test_factorized_critic.py` | modified | Adds `test_v211_layernorm_critic_shape`, `test_v211_param_count_vs_v27`, `test_v211_policy_actor_is_v27_compatible`, `test_v211_layernorm_actually_normalises`. All v2.6, v2.7, v2.8 tests pass unchanged. |
| `src/rl/train_v211.py` | new | Training script. Mirrors `train_v210_e4.py` (the most recent template) with `V211CTDESACPolicy`, `GAMMA = 0.99`, output dir `sac_v211_seed{N}`. Imports `train_sac_v211` as the entry point. |
| `notebooks/colab_sac_v211.ipynb` | new | Colab notebook. 9 cells: title/overview, repo clone + deps, secrets + GPU check, pre-flight (smoke + factorised-critic tests + 1000-step pilot), full 250k training, Drive copy, 9-cell perfect-forecast + noisy-forecast eval, Q-inflation diagnostic plot, resume-from-checkpoint stub. |
| `notebooks/kaggle_sac_v211.ipynb` | new | Kaggle notebook. Same 9-cell structure with `/kaggle/working` paths, `kaggle_secrets.UserSecretsClient` for the WandB key, and a zip-archive cell instead of the Drive copy. |

No deletions. No file renames. No changes to evaluation code paths.

---

## 6 — Validation checklist before retraining

1. `pytest tests/test_factorized_critic.py -v` — all 13 tests pass (9 existing + 4 new v2.11 tests).
2. `pytest tests/test_rl_smoke.py -v` — all v2.8 smoke tests pass (no regression).
3. Manual probe: instantiate `_V211FactorizedContinuousCritic`, check that `critic.qf0[1]` is an `nn.LayerNorm` and `critic.qf0[2]` is `nn.ReLU`. Confirms layer ordering matches Yue 2023.
4. Manual probe: instantiate `V211CTDESACPolicy`, confirm `policy.actor` is an instance of `_V27SharedActor` (the v2.11 actor is byte-identical to v2.7's) and `policy.critic` is `_V211FactorizedContinuousCritic`.
5. Manual probe: save a freshly-instantiated v2.11 SAC model, reload via `_load_sac_model`, confirm the auto-detected `arch_label` contains `'v2.11'` and the prediction on a fixed observation is byte-identical between the original and reloaded models.
6. 1000-step pilot training (`train_sac_v211(seed=999, total_timesteps=1000)`) completes without error on Colab T4 and produces a `best_model/best_model.zip` in the output directory.
7. The v2.7 best_model.zip in the existing repo still loads correctly via `_load_sac_model` after the runner edits, dispatches to `V27CTDESACPolicy` (NOT to `V211CTDESACPolicy`), and produces predictions consistent with prior evaluation results.

If all seven pass, the 250k training run can proceed under the same Colab Pro / Kaggle setup as v2.7 and v2.10 E4. Expected wall-clock cost: ~30–55 min on A100, ~2–2.5 h on Kaggle T4.

---

## 7 — Open questions for after retraining

- If Outcome A (full cascade suppression) holds, does the eval reward continue to climb past step 250k? If so, a follow-up run with `TOTAL_TIMESTEPS = 400_000` may produce a meaningfully stronger policy. This costs one additional Colab session.
- If Outcome A holds and 9-cell yields exceed v2.7's by a margin larger than seed-1-vs-seed-0 variance (>250 kg/ha mean), v2.11 supersedes v2.7 as the Phase 1 baseline. Phase 2 (wet-year oversampling and rainfall domain randomisation `U[0.7, 1.5]` with coupled ET) should launch from v2.11, not v2.7.
- If Outcome B (partial suppression) holds, the next experiment combines v2.11's LayerNorm critic with `γ = 0.985` (effective horizon 67). This tests a previously-untested intermediate point on the Phase 1 trade-off surface.
- If Outcome C (no effect) holds, the cascade is bifurcation-driven. The next experiment is an actor-side intervention: scale the entropy coefficient in the actor loss only by `1/|mean(Q)|`, so the entropy term doesn't get drowned out by the inflating Q. This is one additional Colab session.
- Seed variance: v2.7 showed 6–8% dry-cell yield gap between seed-0 and seed-1. If v2.11 seed-0 passes the acceptance criterion, seed-1 and seed-2 are run to establish that the result is reproducible. If seed variance on v2.11 is materially smaller than on v2.7 (e.g. <3%), this is itself a publishable finding — LayerNorm not only prevents the cascade but tightens the seed-to-seed variance.

---

*End of specification.*
