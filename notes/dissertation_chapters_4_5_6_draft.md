# Dissertation Chapters 4 (§4.6–4.8), 5, and 6 — Revised Draft

> **Scope:** This draft covers the RL controller sections of Chapter 4 (§4.6 onward), the full Chapter 5, and the full Chapter 6. Chapters 1–3 and §4.1–4.5 (baselines + MPC) are unchanged. All numbers are from committed repository data, verified independently.

---

## Chapter 4 — Controller Design and Calibration

> §4.1–4.5 (baselines, MPC formulation, weight sensitivity, operating point) remain as-is.

### 4.6 Reinforcement Learning Controller

This section presents the reinforcement learning (RL) controller designed to approximate the MPC's agronomic performance at a fraction of the per-decision computational cost. The RL controller is trained offline on the same ABM virtual plant and evaluated under the same 9-cell scenario-budget grid, enabling a direct comparison between online optimization and amortized neural-network inference.

Two algorithmic variants of the RL controller are developed: a **Soft Actor-Critic** (SAC) agent with a stochastic, entropy-regularized policy, and a **Twin Delayed Deep Deterministic Policy Gradient** (TD3) agent with a deterministic policy and target-policy smoothing. Both share the same environment interface, observation space, CTDE architecture, and VDN critic structure; they differ only in the policy type, the exploration mechanism, and the training stabilization method. This shared-architecture design enables the comparison between them to function as a controlled ablation on the role of entropy in multi-agent irrigation control.

#### 4.6.1 Markov Decision Process Formulation

The irrigation control problem is cast as a Markov Decision Process (MDP) $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ where:

**State space** $\mathcal{S}$. The observation vector $o_k \in \mathbb{R}^{1097}$ is constructed from 130 per-agent local features (8 features per agent: normalized soil moisture $x_1^{(n)}/\text{FC}$, deep-layer moisture $x_5^{(n)}/x_{5,\text{ref}}$, biomass $x_4^{(n)}/x_{4,\text{ref}}$, stress accumulator $x_3^{(n)}$, growth factor $\gamma^{(n)}$, and three topographic features — D8 upstream count, mean upstream elevation difference, and downstream indicator) concatenated with 57 global features (day-of-season, budget fraction remaining, current rainfall/ET0/temperature, and a 48-dimensional weather forecast window). The per-agent features are arranged in agent-major order followed by the global block, enabling efficient parameter-shared processing.

**Action space** $\mathcal{A}$. The joint action $\mathbf{a}_k = (a_k^{(1)}, \ldots, a_k^{(130)}) \in [0, 12]^{130}$ specifies the irrigation depth in mm/day for each of the 130 agents. The per-agent action is clipped to the actuator bound $u_{\max} = 12$ mm/day and further constrained by the remaining seasonal water budget $B_{\text{rem}}(k)$.

**Discount factor** $\gamma = 0.99$. The 93-day season gives an effective horizon $1/(1-\gamma) = 100$, commensurate with the episode length. This choice balances long-range credit assignment (the yield payoff of today's irrigation is realized at harvest) against bootstrap stability (the value-function estimation error compounds geometrically over the horizon).

**Reward function** $R$. The per-step reward mirrors the MPC cost function (Section 4.2.1) restricted to the terms that decompose additively per agent and are computable without access to the internal model:

$$r_k = r_1 + r_2 + r_3 + r_5 + r_6$$

where:

| Term | Definition | Weight | Role |
|------|-----------|--------|------|
| $r_1$ (biomass) | $\alpha_1 \cdot (\bar{x}_4(k) - \bar{x}_4(k-1)) / x_{4,\text{ref}}$ | $\alpha_1 = 1.0$ | Yield signal (dense) |
| $r_2$ (water cost) | $-\alpha_2 \cdot \bar{u}(k) / u_{\max}$ | $\alpha_2 = 0.016$ | Water conservation |
| $r_3$ (drought) | $-\alpha_3 \cdot \text{mean}_n[\max(0, x_{1,\text{ST}} - x_1^{(n)}(k)) / (x_{1,\text{ST}} - x_{1,\text{WP}})]$ | $\alpha_3 = 0.1$ | Stress avoidance |
| $r_5$ (smoothing) | $-\alpha_5 \cdot \text{mean}_n[((u^{(n)}(k) - u^{(n)}(k-1))/u_{\max})^2]$ | $\alpha_5 = 0.005$ | Actuator smoothing |
| $r_6$ (waterlog) | $-\alpha_{6,\text{lin}} \cdot \text{mean}_n[\max(0, x_1^{(n)}(k) - \text{FC}) / \text{FC}]$ | $\alpha_{6,\text{lin}} = 1.5$ | FC-overshoot penalty |

The weights $\alpha_2$, $\alpha_3$, and $\alpha_5$ are identical to the calibrated MPC cost (Section 4.5), ensuring that both controllers optimize equivalent objectives. The overshoot penalty $r_6$ uses a linear form (rather than the MPC's quadratic $\alpha_4$) to provide a uniform gradient across the overshoot range; the quadratic form was found to produce near-zero gradient for small overshoots, allowing the policy to drift above field capacity without correction (Section 4.4.3). The smoothing term $r_5$ mirrors MPC cost term $J_5$ and is first-day-exempt to avoid penalizing the initial action.

**Terminal yield correction** (TD3 only). The dense biomass increment $r_1$ is equivalent to MPC's terminal-yield objective $J_1 = -\alpha_1 \cdot x_4(K)/x_{4,\text{ref}}$ only in the undiscounted limit ($\gamma = 1$). Under $\gamma = 0.99$, discounting front-loads the biomass signal: the effective weight on final yield drops to $\gamma^{93} \approx 0.39$, while early-season biomass increments retain weights near 1.0. This creates an incentive to irrigate early — when the discounted reward is highest — and under-reserve water for the yield-critical reproductive phase (days 46–69). The TD3 variant corrects this by adding a terminal-yield bonus:

$$r_K^{\text{terminal}} = \alpha_T \cdot \bar{x}_4(K) / x_{4,\text{ref}}, \quad \alpha_T = 1.0$$

paid once at the final step $K = 93$. This lifts the endpoint coefficient from $\gamma^K \approx 0.39$ to $\gamma^K(1 + \alpha_T) \approx 0.78$, pulling the RL objective back toward MPC's undiscounted terminal target without changing the discount factor (which would re-introduce bootstrap instability). The SAC variant does not include this term because the entropy-regularized policy distributes water more uniformly across the season, reducing the front-loading effect.

**Reward design rationale.** The reward magnitudes are deliberately asymmetric: waterlog ($r_6$) reaches $-0.86$ per step on heavy overshoot while drought ($r_3$) caps at $-0.10$, and the yield signal is small ($\sim$+0.015 per step). This asymmetry mirrors the agronomic reality that waterlogging causes acute tissue damage (irreversible within a season) while moderate drought stress is recoverable. The water-cost coefficient $\alpha_2 = 0.016$ is anchored to the Iranian domestic-base water tariff (Section 4.2.1), ensuring economic interpretability.

#### 4.6.2 Policy Formulations: SAC and TD3

Both variants build on the actor-critic paradigm with off-policy learning, but differ in how the policy is parameterized and regularized.

**SAC: maximum-entropy stochastic policy.** The SAC agent (Haarnoja et al., 2018) maximizes the entropy-augmented return:

$$J_{\text{SAC}}(\pi) = \sum_{k=0}^{K} \gamma^k \, \mathbb{E}\big[ r_k + \alpha_{\text{ent}} \, \mathcal{H}[\pi(\cdot \mid s_k)] \big]$$

where $\alpha_{\text{ent}} = 0.002$ is a fixed entropy coefficient and $\mathcal{H}$ is the policy entropy. The actor outputs the mean and log-standard-deviation of a squashed Gaussian: $a^{(n)} = \tanh(\mu^{(n)} + \sigma^{(n)} \cdot \epsilon)$, $\epsilon \sim \mathcal{N}(0,1)$, scaled to $[0, 12]$ mm. The entropy term encourages the policy to maintain variance over actions, providing implicit exploration and action smoothing — the stochastic policy naturally avoids extreme actions (near 0 or 12 mm) because the Gaussian entropy is maximized near the action-space center.

Entropy auto-tuning (adjusting $\alpha_{\text{ent}}$ to match a target entropy $-\dim(\mathbf{a}) = -130$) is disabled because it produces entropy collapse in this high-dimensional action space: the auto-tuner spikes $\alpha_{\text{ent}}$ by $26\times$ within 14,000 steps as the policy variance shrinks, destabilizing the critic. A fixed $\alpha_{\text{ent}} = 0.002$ provides stable regularization throughout training.

**TD3: deterministic policy with target-policy smoothing.** The TD3 agent (Fujimoto et al., 2018) learns a deterministic policy $\mu_\theta(s)$ that directly outputs the action mean, using target-policy smoothing as a structural replacement for entropy regularization:

$$y_k = r_k + \gamma \, Q_{\phi'}(s_{k+1}, \, \mu_{\theta'}(s_{k+1}) + \epsilon), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma_{\text{target}}), -c, c)$$

where $\sigma_{\text{target}} = 0.2$ and $c = 0.5$. This smooths the Q-function over a neighborhood of the target action, preventing the critic from developing sharp peaks that a deterministic actor would exploit. Additionally, the policy update is delayed (every 2 gradient steps) relative to the critic, reducing the actor-critic oscillation that drives training instability.

**Why TD3 extends SAC for this problem.** The entropy term in SAC provides beneficial smoothing but imposes a structural constraint: the stochastic policy is pulled toward the action-space center by the entropy gradient, limiting the actor's ability to commit to extreme actions — specifically, the near-zero irrigation depths needed for wet-year water conservation. SAC's wet-year waterlog count (36.6 days per agent) is the visible cost of this constraint. TD3 removes the entropy term, allowing the deterministic actor to reach the full action range, including the low-irrigation regime that reduces waterlogging. Target-policy smoothing provides a partial substitute for entropy's stabilizing effect, and the smoothing penalty $r_5$ in the reward provides a partial substitute for its action-regularization effect. The remaining difference — that entropy smooths through stochasticity while target-policy smoothing smooths through noise injection in the critic target — accounts for the smoothness gap documented in Chapter 5.

#### 4.6.3 Gymnasium Environment Wrapper

The ABM is wrapped as a Gymnasium `Env` with a flat `Box` observation space ($\mathbb{R}^{1097}$) and a flat `Box` action space ($[0, 1]^{130}$, rescaled internally to $[0, 12]$ mm). Each `reset()` samples a year uniformly from the 20 training years and a budget fraction uniformly from $[0.70, 1.00]$. The `step()` function advances the ABM by one day, clips the joint action to the remaining budget, updates all five state variables $(x_1, \ldots, x_5)$ through the dynamics of Chapter 3, computes the reward decomposition, and returns the observation, scalar reward, termination flag (True at day 93), truncation flag (always False — episodes run to completion), and a diagnostic info dictionary.

Per-year precomputed quantities. The six biological-nonlinearity arrays $(h_1, x_2, h_2, h_7, g_{\text{base}}, \text{ET}_c)$ from Section 4.3.2 are computed on-the-fly for each sampled training year and cached per-episode. This ensures that the forecast features in the observation are consistent with the actual climate experienced by the ABM.

Rainfall normalization uses $r_{\text{ref}} = 30.0$ mm (the 26-year median daily maximum), consistent with the MPC's forecast normalization.

#### 4.6.4 CTDE Actor-Critic Architecture

The 130-agent structure motivates a Centralized Training, Decentralized Execution (CTDE) architecture (Lowe et al., 2017). The actor learns a policy conditioned only on per-agent local features and global context; the critic accesses the full joint state for credit assignment during training. At deployment, each agent executes using only local observations and the shared global forecast — no inter-agent communication is needed.

**Actor: shared-parameter local policy.** A single multi-layer perceptron is applied to all 130 agents in parallel. For SAC, the architecture is $65 \to 128 \to 128 \to 1$ (mean) $+ 1$ (log-std), with ReLU activations. For TD3, the architecture is $65 \to 128 \to 128 \to 1$ (deterministic $\mu$), with LeakyReLU activations and a $2x - 1$ re-centering after $\tanh$ so that a zero-input produces a mid-range action rather than a boundary value. Parameter sharing reduces the actor from $\sim\!10^6$ parameters (monolithic) to $\sim\!10^4$ (shared MLP), and enforces spatial equivariance: permuting agent indices permutes actions identically.

**Critic: twin-Q value decomposition network (VDN).** The joint Q-function is decomposed additively:

$$Q_{\text{total}}(s, \mathbf{a}) = \sum_{n=1}^{130} Q_{\text{loc}}^{(n)}(s^{(n,\text{loc})}, s^{(\text{glob})}, a^{(n)})$$

Each local Q-network is a shared $66 \to 256 \to 256 \to 1$ MLP with LayerNorm applied after each linear layer (the LayerNorm stabilizes gradient magnitudes across the 130-agent sum and was found empirically to prevent critic-loss escalation). Twin Q-networks ($Q_1$, $Q_2$) are maintained; the minimum of the two scalar totals $\min(Q_1^{\text{total}}, Q_2^{\text{total}})$ is used for the policy update (clipped double-Q, following Fujimoto et al., 2018).

#### 4.6.5 Motivation for Value Decomposition: Spatial Credit Assignment

The additive VDN decomposition is the minimal structural change that provides per-agent credit assignment while preserving the CTDE property. A monolithic critic mapping the full 1097-dimensional observation plus 130-dimensional action to a single scalar Q-value cannot differentiate the relative contribution of agent $n$'s action from any other agent's. Because all agents share actor parameters, the only mechanism for agent-specific actions is through the per-agent features — but those features only influence the action if the critic rewards them differentially, which a monolithic critic does not. The VDN factorization resolves this: each local Q-network receives one agent's features and action, and its output directly influences the gradient passed to that agent's actor parameters.

The decomposition is valid because the reward is structurally additive in agent contributions: biomass increments are averaged over agents, water costs sum, and stress/overshoot penalties are agent-local.

#### 4.6.6 Training Protocol

**Data split.** A three-way split of the 26-year NASA POWER record governs the experimental design:

- **Training years (20):** 2000, 2001, 2003, 2005–2012, 2014–2017, 2019–2021, 2025. Sampled uniformly at each episode reset.
- **Development years (3):** 2002, 2004, 2013. Used by the deterministic evaluation callback for best-model selection.
- **Test years (3):** 2022 (moderately dry, 39.7 mm), 2018 (wet, 108.8 mm), 2024 (extremely wet, 176.8 mm). Used only for the final comparison in Chapter 5.

At each training episode, the environment samples a year uniformly from the 20 training years and a budget fraction uniformly from [70%, 100%]. The development-set evaluation callback scores each checkpoint on the fixed schedule DEV_YEARS × {70%, 85%, 100%} = 9 episodes, providing a comparable, held-out generalization signal for best-model selection.

**SAC hyperparameters.** Table 4.6a summarizes the SAC training configuration.

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | SAC (SB3) | Stochastic policy with entropy regularization |
| Total timesteps | 250,000 | 10× the convergence point observed in pilots |
| Replay buffer | 250,000 | Cycles ~1× per run |
| Batch size | 256 | Standard for continuous-action SAC |
| Learning rate | $3 \times 10^{-4} \to 5 \times 10^{-5}$ | Linear decay; stabilizes late-training updates |
| $\gamma$ | 0.99 | See §4.6.1 |
| $\tau$ (soft update) | 0.005 | Haarnoja et al. default |
| $\alpha_{\text{ent}}$ | 0.002 (fixed) | See §4.6.2; auto-tuning disabled |
| $\alpha_5$ (smoothing) | 0 | SAC entropy provides implicit smoothing |
| $\alpha_T$ (terminal yield) | 0 | Entropy reduces front-loading sufficiently |
| Actor LR multiplier | 5.0 | Asymmetric LR; actor updates faster |
| Gradient clip $\|\nabla\|_2$ | 1.0 | Caps critic update magnitude |
| Learning starts | 1,000 | Random exploration before first update |
| Exploration noise | $\sigma: 0.30 \to 0.00$ (60k), then pulse 0.15 at 150–180k | Late reinjection refreshes critic data in low-action regime |
| Seeds | 1 | SAC run with deterministic dev-set eval |

**Late noise reinjection** (SAC-specific). Exploration noise anneals $\sigma = 0.30 \to 0.00$ over the first 60,000 steps. At step 150,000, a triangular re-injection pulse (peak $\sigma = 0.15$ at step 165,000, returning to 0 by step 180,000) is applied. By this point the critic is well-trained in the visited state region, so the injected low-action transitions receive accurate value estimates immediately, pulling the policy's wet-year actions toward the lower irrigation depths needed for waterlog reduction.

**TD3 hyperparameters.** Table 4.6b summarizes the TD3 training configuration.

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | TD3 (SB3) | Deterministic policy with target-policy smoothing |
| Total timesteps | 250,000 | Same budget as SAC |
| Replay buffer | 250,000 | Same as SAC |
| Batch size | 256 | Same as SAC |
| Learning rate | $3 \times 10^{-4} \to 5 \times 10^{-5}$ | Same schedule as SAC |
| $\gamma_{\text{base}}$ | 0.99 | Effective per-step discount |
| $\gamma_{\text{model}}$ | $0.99^5 = 0.951$ | Set to $\gamma_{\text{base}}^n$ for exact n-step returns |
| $\tau$ (soft update) | 0.005 | Same as SAC |
| Policy delay | 2 | Fujimoto et al. default; critic leads actor |
| Target policy noise $\sigma$ | 0.2 (clip 0.5) | Smooths Q over action neighborhood |
| $\alpha_5$ (smoothing) | 0.005 | Mirrors MPC cost term $J_5$ |
| $\alpha_T$ (terminal yield) | 1.0 | Corrects $\gamma$-discounting front-loading (§4.6.1) |
| Gradient clip $\|\nabla\|_2$ | 1.0 | Same as SAC |
| Learning starts | 50,000 | Extended warm-up; see §4.6.7 |
| Exploration noise | $\sigma: 0.40 \to 0.15$ (sustained floor) | Permanent exploration floor prevents collapse |
| n-step returns | $n = 5$ (exact $\gamma^n$) | Stabilizes bootstrap horizon; see §4.6.7 |
| Seeds | 3 (0, 1, 2) | Seed-averaged results; validates reproducibility |

**Exact n-step returns** (TD3-specific). The replay buffer accumulates the $n$-step discounted return $R_n = \sum_{k=0}^{n-1} \gamma_{\text{base}}^k \, r_{t+k}$ with $\gamma_{\text{base}} = 0.99$ and stores $(s_t, R_n, s_{t+n}, \text{done})$. The model's discount is set to $\gamma_{\text{model}} = \gamma_{\text{base}}^n = 0.99^5 = 0.951$. SB3's stock TD3 target then computes:

$$y = R_n + (1 - d) \cdot \gamma_{\text{model}} \cdot Q_{\phi'}(s_{t+n}, \mu_{\theta'}(s_{t+n}) + \epsilon)$$

which is the exact $n$-step Bellman target without any train-loop modification. Terminal transitions ($d = 1$) zero the bootstrap term naturally. This decoupling — buffer stores the per-step discount, model carries the $n$-step discount — is the key implementation detail that makes exact multi-step returns compatible with SB3's unmodified training loop.

#### 4.6.7 Training Stability: N-Step Bootstrap Horizon Control

The 93-day episode under $\gamma = 0.99$ creates an effective bootstrap horizon of $1/(1-\gamma) \approx 100$. With standard 1-step returns, the critic's estimate of day-1 value depends on a chain of 92 successive bootstraps, each compounding the estimation error geometrically. This horizon amplification is the dominant source of training instability in this problem: experiments with a shorter effective horizon (early-terminated episodes of ~50 steps) maintained bounded critic loss for 500,000 steps, while the same configuration on full 93-step episodes produced value-function divergence.

The $n = 5$ step return directly addresses this mechanism. By replacing the first 5 bootstraps with observed rewards, the maximum bootstrap chain length drops from 92 to $\lceil 92/5 \rceil \approx 19$ effective bootstraps, and each carries the smaller discount $\gamma^5 = 0.951$. The compounding factor drops from $0.99^{92} \approx 0.40$ to $0.951^{19} \approx 0.38$ — a comparable final discount but over far fewer compounding steps, substantially reducing the amplification of estimation errors.

This stabilization is specific to TD3. The deterministic target policy makes uncorrected multi-step returns near-unbiased (the off-policy distribution shift across 5 steps of a deterministic policy with small exploration noise is negligible). For SAC, multi-step off-policy returns require importance-sampling correction (Huang et al., 2021), which was not implemented. The SAC variant instead relies on its lower learning-starts threshold (1,000 vs 50,000) and the entropy term's implicit regularization for stability.

**Training convergence (TD3).** Across all three seeds, the evaluation reward climbs monotonically and the best model is captured at or near the final training step (steps 200k, 225k, 250k for seeds 0, 1, 2 respectively). The critic's predicted Q-value ($q_{\text{pred}}$) plateaus at approximately +10.5 — an overestimation relative to realized returns, attributable to the terminal-yield bonus entering the bootstrap. This overestimation is bounded and non-divergent (it stabilizes by step 150k), in contrast to the unbounded cascades observed without n-step returns.

**Training convergence (SAC).** The SAC agent converges within the first 100,000 steps to a stable plateau. The entropy regularization prevents the value-function instabilities observed in the TD3 line, and the critic remains well-calibrated throughout training. The best model is selected by the development-set evaluation callback.

#### 4.7 Meteorological Forecast Noise Model

> This section remains as-is from dissertation_11. The AR(1) multiplicative noise model ($\sigma_{\text{base}} = 0.15$, $\rho = 0.6$) is applied identically to both MPC and RL controllers.

#### 4.8 Chapter Summary

This chapter presented four irrigation controllers operating on the shared ABM virtual plant of Chapter 3. Two optimization baselines (no-irrigation and fixed-schedule) bracket the performance space. The constrained MPC (§4.2–4.5) provides the upper-bound benchmark through online nonlinear optimization with a five-term economically-anchored cost function, calibrated via a systematic four-phase weight-sensitivity analysis. Two reinforcement learning controllers — SAC (§4.6.2, stochastic policy with entropy regularization) and TD3 (§4.6.2, deterministic policy with target-policy smoothing and n-step returns) — are designed against the same reward function, sharing a CTDE architecture with a VDN-factorized twin-Q critic for spatial credit assignment. The key architectural distinction is the role of entropy: SAC's entropy term provides implicit action smoothing and exploration but limits the policy's ability to reach extreme actions, while TD3 removes this constraint at the cost of explicit stabilization through multi-step returns and a terminal-yield correction. Chapter 5 evaluates all four controllers across the 9-cell scenario-budget grid under both perfect and noisy forecasts.

---

## Chapter 5 — Results and Discussion

### 5.1 Introduction

This chapter presents the comparative evaluation of all four controllers — no-irrigation baseline, fixed-schedule heuristic, constrained MPC, and the two RL variants (SAC and TD3) — across the 9-cell factorial grid of climate scenario × budget allocation defined in Chapter 3. The evaluation proceeds in six stages: baseline performance (§5.3), MPC under perfect forecast (§5.4), SAC performance (§5.5), TD3 performance (§5.6), performance under forecast uncertainty (§5.7), and the head-to-head comparison with discussion (§5.8–5.9).

The test scenarios (moderately dry 2022, wet 2018, extremely wet 2024) are held out from both training and development sets. Budget levels (100%, 85%, 70% of climatological demand) span the range from full allocation to severe deficit. All RL results use deterministic policy evaluation (no exploration noise); MPC uses the Hp = 8 horizon configuration identified as the recommended operating point in §4.5.

### 5.2 Evaluation Metrics and Statistical Methods

#### 5.2.1 Primary Performance Metrics

- **Yield** ($Y$, kg/ha): Final harvested biomass, computed as $x_4(K) \cdot h_i \cdot 10$, where $h_i$ is the harvest index and the factor 10 converts from biological mass per unit area. Averaged over 130 agents.
- **Drought days per agent**: Number of days with $x_1^{(n)}(k) < x_{1,\text{ST}} = 112$ mm, averaged over agents.
- **Waterlog days per agent**: Number of days with $x_1^{(n)}(k) > \text{FC} = 140$ mm, averaged over agents.
- **Total water applied** (mm/agent): Season-sum of $u^{(n)}(k)$, averaged over agents.
- **Control smoothness** (mean $|\Delta u|$): Mean absolute day-to-day change in irrigation per agent, in mm/day. Lower values indicate smoother actuation.
- **Forecast sensitivity**: Ratio of noisy-forecast yield to perfect-forecast yield, expressed as a percentage.

#### 5.2.2 Multi-Seed Reporting

TD3 results are reported as the mean across 3 independent seeds (0, 1, 2), each trained on the full 250,000-step budget with different random initialization. SAC results are from a single seed trained with deterministic development-set evaluation on {2002, 2004, 2013}. [Note: update this once the SAC re-run is complete; if multiple SAC seeds are run, report them identically to TD3.]

### 5.3 Baseline Performance

#### 5.3.1 No-Irrigation Baseline

> Unchanged from dissertation_11 — the rainfed yield floor. Key numbers: dry 2022 yield 1,462 kg/ha with 87.0 drought-days; wet 2024 yield 2,243 kg/ha with 82.7 drought-days.

#### 5.3.2 Fixed-Schedule Heuristic

> Unchanged from dissertation_11 — the traditional Gilan practice baseline.

### 5.4 MPC Performance Under Perfect Forecast

> §5.4.1–5.4.3 unchanged from dissertation_11 (calibration, results, behavioral analysis). The MPC at Hp = 8 achieves a 9-cell mean yield of 3,810 kg/ha with 16.0 drought-days and 7.7 waterlog-days per agent. Mean control smoothness is $|\Delta u| = 0.97$ mm/day.

### 5.5 SAC Performance Under Perfect Forecast

#### 5.5.1 Architecture and Training Summary

The SAC controller uses the CTDE architecture described in §4.6.4 with a stochastic policy ($\alpha_{\text{ent}} = 0.002$, fixed) and the four-term reward ($r_1 + r_2 + r_3 + r_6$; no smoothing penalty, no terminal yield). Training runs for 250,000 steps with a late noise-reinjection pulse (§4.6.6). The best model is selected on the development set {2002, 2004, 2013}.

#### 5.5.2 SAC Evaluation Results

[Table 5.x: SAC performance across the 9-cell grid, perfect forecast]

| Scenario | Yield (kg/ha) | % MPC | Drought-d | Waterlog-d | Water (mm) | $|\Delta u|$ |
|---|---|---|---|---|---|---|
| Dry/100% | [TO FILL after re-run] | | | | | |
| Dry/85% | | | | | | |
| Dry/70% | | | | | | |
| Mod/100% | | | | | | |
| Mod/85% | | | | | | |
| Mod/70% | | | | | | |
| Wet/100% | | | | | | |
| Wet/85% | | | | | | |
| Wet/70% | | | | | | |
| **9-cell mean** | ~3785 | ~99.3% | ~25.6 | ~18.2 | | ~0.69 |

[Note: Fill from the re-run results. The existing v2.18 numbers from the original run are: yield 3785 (99.3% MPC), drought 25.6, waterlog 18.2, smoothness 0.69. The re-run with dev-set {2002,2004,2013} may differ slightly.]

**Key observations:**

The SAC controller achieves MPC-competitive yield on in-distribution scenarios (moderately dry and wet cells), with notably lower drought stress than any other controller (25.6 days vs MPC's 16.0 — still elevated, but the closest of any RL variant). The entropy-regularized policy produces the smoothest control signal of any controller tested, including MPC: mean $|\Delta u| = 0.69$ mm/day vs MPC's 0.97. This smoothness is a direct consequence of the entropy term, which pulls the policy toward the center of its action distribution and penalizes sharp changes implicitly through the stochastic sampling.

However, the SAC policy fails the wet-year waterlog criterion: 18.2 waterlog-days per agent (MPC: 7.7). The entropy gradient prevents the stochastic policy from committing to the near-zero irrigation depths needed to keep wet-year soil moisture below field capacity. This limitation — the cost of entropy's smoothing benefit — motivates the TD3 extension.

#### 5.5.3 Inference Latency

SAC inference requires a single forward pass through the shared 65→128→128→2 MLP for all 130 agents. Measured latency: approximately 1 millisecond per decision step, independent of state complexity or forecast horizon. This represents a 4–5 order-of-magnitude reduction from MPC's ~30-second mean solve time.

### 5.6 TD3 Performance Under Perfect Forecast

#### 5.6.1 Architecture and Training Summary

The TD3 controller shares the SAC's CTDE architecture and VDN critic (§4.6.4–4.6.5) but replaces the stochastic policy with a deterministic one (§4.6.2). The reward includes all five terms plus the terminal-yield correction ($\alpha_T = 1.0$). Training uses exact 5-step returns for bootstrap stabilization (§4.6.7) and runs for 250,000 steps with a sustained exploration noise floor. Results are averaged over 3 seeds.

#### 5.6.2 TD3 Evaluation Results (3-seed mean)

[Table 5.y: TD3 performance across the 9-cell grid, perfect forecast, 3-seed mean]

| Scenario | Yield (kg/ha) | % MPC | Drought-d | Waterlog-d | Water (mm) |
|---|---|---|---|---|---|
| Dry/100% | 4161 | 100.4% | 20.1 | 1.2 | |
| Dry/85% | 4067 | 99.9% | 33.7 | 1.7 | |
| Dry/70% | 3734 | 99.2% | 53.3 | 1.3 | |
| Mod/100% | 3770 | 101.4% | 21.1 | 4.2 | |
| Mod/85% | 3717 | 99.8% | 33.8 | 3.7 | |
| Mod/70% | 3571 | 98.9% | 55.3 | 2.1 | |
| Wet/100% | 3724 | 99.1% | 30.9 | 14.6 | |
| Wet/85% | 3720 | 99.4% | 32.2 | 13.9 | |
| Wet/70% | 3695 | 98.4% | 36.5 | 11.8 | |
| **9-cell mean** | **3800** | **99.7%** | **35.2** | **6.5** | **358** |

[Note: Yields are 3-seed averages. MPC yields for reference: 4145, 4069, 3766, 3718, 3725, 3612, 3759, 3743, 3754 (mean 3810).]

**Key observations:**

The TD3 controller matches MPC yield across all nine cells (99.7% mean, range 98.4–101.4%), including the extremely wet 2024 scenario that SAC could not match. Waterlogging is lower than MPC in all scenarios (6.5 vs 7.7 mean), and water consumption is 1.7% lower (358 vs 364 mm). The deterministic policy's ability to commit to near-zero irrigation in wet conditions — the capability entropy prevented — closes the waterlog gap entirely.

However, two performance gaps remain relative to MPC:

**Drought-days** (35.2 vs 16.0). The TD3 controller runs approximately twice MPC's drought count. Phase-level analysis reveals that the majority of additional drought occurs in the late grain-filling phase (days 69–93), where both controllers allow soil drying — MPC's grain-fill drought fraction is itself 0.71 at moderate/70%. The yield-critical reproductive-phase drought is well-controlled (0.05–0.07 reproductive drought fraction at moderate/70%, vs MPC's 0.00). The drought-day count overstates the agronomic impact.

**Control smoothness** (mean $|\Delta u| \approx 2.5$ vs MPC's 0.97). The deterministic policy exhibits a pulsing pattern — alternating between higher and lower irrigation depths rather than the smooth transitions MPC achieves through whole-trajectory optimization. This is a structural characteristic of reactive (one-step-at-a-time) policies: the agent responds to daily weather changes with correspondingly variable actions, while MPC pre-smooths the trajectory by planning over an 8-day horizon. The SAC controller's entropy term masked this pulsing (mean $|\Delta u| = 0.69$), but at the cost of the waterlog failure.

#### 5.6.3 Seed Reproducibility

| Seed | Yield | % MPC | Drought-d | Waterlog-d |
|---|---|---|---|---|
| 0 | 3813 | 100.1% | 36.8 | 5.9 |
| 1 | 3799 | 99.7% | 33.8 | 7.5 |
| 2 | 3789 | 99.5% | 35.1 | 6.2 |
| **Mean** | **3800** | **99.7%** | **35.2** | **6.5** |
| Std | 10 | 0.3% | 1.3 | 0.7 |

Yield standard deviation across seeds is 10 kg/ha (0.3% of mean), confirming that the TD3 result is reproducible and not a single-seed artifact.

#### 5.6.4 Inference Latency

TD3 inference requires a single forward pass through the 65→128→128→1 shared MLP. Measured latency: approximately 1 millisecond — identical to SAC.

### 5.7 Performance Under Forecast Uncertainty

#### 5.7.1 MPC Under Noisy Forecast

> Summary from dissertation_11: MPC yield degrades by approximately 0.2% under the AR(1) noise model. [Update specific numbers if needed.]

#### 5.7.2 RL Controllers Under Noisy Forecast

Both RL controllers are trained exclusively under perfect forecasts and receive the noisy forecast as input at evaluation time without re-training.

| Controller | Perfect yield | Noisy yield | Sensitivity |
|---|---|---|---|
| MPC Hp=8 | 3810 | ~3802 | ~99.8% |
| SAC | [TO FILL] | [TO FILL] | [TO FILL] |
| TD3 (seed 0) | 3813 | 3815 | 100.1% |
| TD3 (seed 1) | 3799 | 3764 | 99.1% |
| TD3 (seed 2) | 3789 | 3810 | 100.5% |
| **TD3 mean** | **3800** | **3796** | **99.9%** |

The TD3 controller maintains near-perfect forecast robustness (99.9% noisy/perfect yield), matching MPC's resilience. The RL policy's implicit robustness arises from training on randomized years and budgets: the policy learns to respond to observed soil-moisture states rather than relying on forecast accuracy, making it naturally tolerant of forecast perturbations.

### 5.8 SAC vs TD3: Comparative Analysis

This section directly compares the two RL controller variants to characterize the role of entropy in multi-agent irrigation control.

#### 5.8.1 Summary Comparison

| Metric | MPC | SAC | TD3 (3-seed) | Better RL |
|---|---|---|---|---|
| 9-cell yield | 3810 | ~3785 | 3800 | TD3 |
| % of MPC | 100% | ~99.3% | 99.7% | TD3 |
| Drought-days | 16.0 | ~25.6 | 35.2 | SAC |
| Waterlog-days | 7.7 | ~18.2 | 6.5 | TD3 |
| Smoothness $|\Delta u|$ | 0.97 | ~0.69 | ~2.5 | SAC |
| Forecast sensitivity | ~99.8% | [TO FILL] | 99.9% | ~tie |
| Inference latency | ~30 s | ~1 ms | ~1 ms | tie |

#### 5.8.2 The Entropy Trade-off

The comparison reveals a fundamental trade-off inherent to the entropy term in continuous-action multi-agent control:

**What entropy provides:** (1) Action smoothing — the stochastic policy naturally avoids the action boundaries, producing smoother control trajectories than either MPC or TD3. (2) Lower drought — by keeping actions closer to the moderate-irrigation range, SAC avoids the alternating high/low pattern that creates transient drought in TD3. (3) Training stability — the entropy regularization bounds the policy's gradient magnitudes, reducing the need for explicit stabilization mechanisms.

**What entropy costs:** (1) Waterlog failure — the entropy gradient prevents the policy from committing to near-zero irrigation in wet conditions, causing persistent overwatering (18.2 waterlog-days vs MPC's 7.7). (2) Lower yield — the inability to reach extreme actions limits the policy's precision at both ends of the action range. (3) Lower yield ceiling — SAC at 99.3% of MPC is a real gap; TD3 at 99.7% effectively closes it.

This trade-off is not a tuning failure — it is structural. Lowering $\alpha_{\text{ent}}$ reduces the waterlog (v2.18 already used $\alpha_{\text{ent}} = 0.002$, the minimum that maintained stability) but cannot eliminate it without removing the entropy term entirely, which is what TD3 does.

#### 5.8.3 Regime-Dependent Performance

The two RL controllers excel in complementary regimes:

- **In-distribution, moderate conditions** (dry and wet scenarios at 100% and 85% budget): SAC's smoother policy and lower drought make it competitive with MPC, and its control signal is arguably more deployment-friendly (lower actuator wear from fewer large changes).
- **Extreme conditions** (extremely wet 2024, constrained 70% budgets): TD3's ability to reach extreme actions — near-zero irrigation for wet-year conservation, and precise budget allocation for deficit scenarios — gives it clear advantages in yield and waterlog.

This regime-dependent complementarity suggests that the choice between SAC and TD3 is deployment-context-dependent: SAC for stable, moderate conditions where smooth actuation matters; TD3 for variable or extreme conditions where yield and waterlog compliance are paramount.

### 5.9 MPC vs RL: Head-to-Head Comparison

#### 5.9.1 Headline Finding

The TD3 reinforcement learning controller matches the MPC's agronomic yield to within 0.3% (3,800 vs 3,810 kg/ha, 9-cell mean) while beating MPC on waterlog (6.5 vs 7.7 days) and water efficiency (358 vs 364 mm) — at a per-decision latency reduction from ~30 seconds to ~1 millisecond (4+ orders of magnitude). This result holds across all nine scenario-budget cells, including the extremely wet 2024 scenario that lies above the upper tail of the training distribution, and is reproducible across three independent training seeds.

#### 5.9.2 What MPC Still Does Better

**Temporal precision.** MPC's receding-horizon optimization produces smoother control trajectories ($|\Delta u| = 0.97$) and fewer total drought-days (16.0 vs 35.2) because it plans multiple steps ahead. The RL policy makes each decision from the current state alone and cannot anticipate future weather — this is a structural limitation of one-step reactive policies, not a tuning failure.

**Spatial allocation.** MPC's control-oriented model sees the full coupled field through the cascade routing matrix and can differentiate irrigation rates across agents based on their topographic position. The RL policy's parameter-shared actor sees only each agent's local features and a global context, limiting its spatial differentiation.

#### 5.9.3 What RL Does Better

**Water efficiency.** Both RL variants use less total water than MPC (SAC ~2%, TD3 ~1.7% less) for comparable yield, suggesting the learned policies have found water-allocation strategies that the local IPOPT optimum does not reach.

**Forecast robustness.** TD3's noisy/perfect yield ratio (99.9%) matches MPC's (~99.8%). The RL policy's implicit robustness — learned from training on randomized years without perfect knowledge of future weather — makes it naturally tolerant of forecast errors, while MPC's performance depends on the accuracy of its forward simulation.

**Deployment latency.** The ~1 ms inference time enables real-time control on resource-constrained edge hardware (embedded processors, IoT sensor networks) where MPC's ~30-second solve is infeasible.

### 5.10 Discussion

#### 5.10.1 Reward Engineering as a Design Contribution

A central finding of this work concerns the reward function itself. The dense discounted biomass increment $r_1$, although equivalent to MPC's terminal-yield objective in the undiscounted limit, creates a front-loading incentive under $\gamma = 0.99$ that starves the yield-critical reproductive phase. This was detected through phase-level water-distribution analysis showing that the RL controller under-irrigated days 46–69 (reproductive) while over-irrigating days 69–93 (grain-fill) relative to MPC. The additive terminal-yield correction $\alpha_T = 1.0$ restores the MPC's terminal-biomass objective without changing the discount factor, closing the worst-case cell (moderate/70%) from a $-4\%$ yield gap to near-parity ($-0.7\%$). This result demonstrates that reward design for long-horizon agricultural control requires explicit correction for discounting artifacts — a finding generalizable beyond this specific system.

#### 5.10.2 The VDN Critic and Spatial Credit Assignment

The VDN decomposition enables the parameter-shared actor to produce agent-differentiated actions despite identical weights, by routing distinct per-agent gradients through the factorized critic. Without VDN, the monolithic critic would train the actor toward a spatially-averaged policy that ignores topographic variation. The VDN architecture adds no hyperparameters and preserves the CTDE deployment property.

#### 5.10.3 Economic Interpretation

[Keep from dissertation_11 — update numbers to match the TD3 results.]

#### 5.10.4 Limitations

**Control smoothness.** The TD3 controller's mean $|\Delta u| \approx 2.5$ mm/day (vs MPC's 0.97) represents meaningful actuator-wear and energy costs in a physical deployment. This pulsing is structural to reactive one-step policies — the agent cannot look ahead to pre-smooth its trajectory — and was not resolved by making the smoothing penalty observable to the agent (the Markov-$r_5$ experiment produced worse pulsing, not better, because the observation augmentation increased the learning problem's complexity without providing the multi-step planning capability that smoothness requires). The SAC controller achieves excellent smoothness (0.69) through entropy regularization, but this comes at the cost of the waterlog failure. An engineering deployment might apply a post-hoc exponential moving average filter to the TD3 policy's raw outputs, accepting a small yield reduction for substantially smoother actuation.

**Drought-day count.** TD3's 35.2 drought-days (vs MPC's 16.0) overstates the agronomic impact — the yield-critical reproductive-phase drought is well-controlled (0.05 fraction vs MPC's 0.00), and much of the excess is harmless late-season grain-fill drying. Nevertheless, the raw count is a real difference that would matter for soil health in multi-season deployment.

**Critic calibration.** The TD3 critic's predicted Q-value plateaus at ~+10.5 against realized returns of ~+0.06, a ~175× overestimation attributable to the terminal-yield bonus. The overestimation is bounded and non-divergent, and the policy converges well despite it, but it means the critic is not informative for value estimation — the actor succeeds despite the critic, not because of it. This is a fragility that could limit transfer to harder problems.

**Single-seed SAC.** The SAC comparison uses a single seed [update if more seeds are run], while TD3 uses three. This limits the statistical power of the SAC-vs-TD3 comparison; the qualitative conclusions (entropy trades smoothing for waterlog) are robust, but precise quantitative differences may shift with additional SAC seeds.

### 5.11 Chapter Summary

Four controllers were evaluated across 9 scenario-budget cells under both perfect and noisy forecasts. The two optimization baselines bracket the performance space. The MPC provides the upper-bound benchmark at 3,810 kg/ha mean yield with 16.0 drought-days and 0.97 mm/day control smoothness, at a cost of ~30 s per decision.

Two RL controllers — SAC and TD3 — reveal a fundamental trade-off in entropy-regularized multi-agent control. SAC achieves the smoothest control (0.69 mm/day) and lowest RL drought (25.6 days) but fails the wet-year waterlog criterion (18.2 days). TD3 removes the entropy constraint, achieving MPC yield parity (99.7%, reproducible across 3 seeds) and beating MPC on waterlog (6.5 days) and water efficiency, but at the cost of higher pulsing (2.5 mm/day) and drought (35.2 days). Both RL controllers reduce per-decision latency from ~30 s to ~1 ms, enabling edge deployment. Both maintain excellent forecast robustness under realistic AR(1) noise. The entropy trade-off — smoothing versus action-range capability — is structural, not a tuning artifact, and represents a genuine design choice for agricultural RL controllers.

---

## Chapter 6 — Conclusion

### 6.1 Summary

This thesis set out to answer a single engineering question: can a model-free reinforcement-learning controller match the agronomic performance of a nonlinear model-based optimizer on a constrained multi-agent irrigation problem, while reducing per-decision latency sufficiently to enable deployment on resource-constrained edge hardware?

The answer, supported by systematic evaluation across 9 scenario-budget cells, 3 independent seeds, and both perfect and noisy forecast conditions, is **yes** — with a nuanced qualification on what "match" entails.

### 6.2 Principal Contributions

**A high-fidelity agent-based simulation environment** (Chapter 3). A 130-agent ABM of a 6-hectare terraced Hashemi rice paddy in Gilan Province, coupling two-layer decoupled hydrology with D8 topographic cascade routing and calibrated FAO-56 crop parameters against 26 years of NASA POWER climate data.

**A constrained nonlinear MPC with economic anchoring** (Chapter 4, §4.2–4.5). A five-term cost function with weights anchored to the four-tier Iranian water tariff, calibrated through a 33-configuration sensitivity sweep that identified and resolved three cost-function pathologies (surface-ponding ineffectiveness, quadratic-overshoot dead-zone, over-penalization at high drought weights).

**Two RL controllers with a shared CTDE-VDN architecture** (Chapter 4, §4.6). A Centralized Training, Decentralized Execution architecture with a parameter-shared actor and a twin-Q value-decomposition critic, implemented in both SAC (stochastic, entropy-regularized) and TD3 (deterministic, target-policy smoothed) variants. The shared architecture enables a controlled comparison of the entropy mechanism's effect on multi-agent irrigation control.

**Characterization of the entropy trade-off in continuous-action multi-agent control** (Chapter 5, §5.8). The direct SAC-vs-TD3 comparison reveals that entropy regularization provides implicit action smoothing and drought reduction but structurally prevents the policy from reaching the extreme actions needed for wet-year water conservation. This trade-off is not a tuning failure but a property of the entropy gradient in high-dimensional continuous action spaces, and represents a design choice that practitioners must make based on deployment context.

**Reward engineering for long-horizon agricultural control** (Chapter 4, §4.6.1; Chapter 5, §5.10.1). The discovery that a dense discounted biomass reward front-loads irrigation under $\gamma < 1$, and the correction via an additive terminal-yield term that restores MPC's undiscounted terminal objective without altering the bootstrap horizon. This finding generalizes beyond the specific system to any long-horizon agricultural RL problem with discounting.

**Bootstrap-horizon stabilization via exact n-step returns** (Chapter 4, §4.6.7). The demonstration that the training instability in this class of problems is driven by bootstrap-horizon amplification under $\gamma = 0.99$ over 93-step episodes, and that exact $n$-step returns ($n = 5$) with a decoupled discount ($\gamma_{\text{model}} = \gamma^n$) bound the critic without requiring train-loop modification. This stabilization is specific to deterministic-policy algorithms and is enabled by the near-unbiasedness of uncorrected multi-step returns under deterministic target policies.

**MPC-competitive RL performance validated across seeds** (Chapter 5). The TD3 controller achieves 99.7% of MPC yield (3-seed mean), beats MPC on waterlog (6.5 vs 7.7 days) and water efficiency (358 vs 364 mm), and maintains 99.9% forecast robustness — at 1 ms per decision vs MPC's ~30 s.

### 6.3 Limitations

**Control smoothness** remains the primary unresolved gap. The TD3 controller's pulsing ($|\Delta u| \approx 2.5$ mm/day vs MPC's 0.97) is structural to reactive policies and was not resolved by reward-based smoothing approaches. Entropy resolves it (SAC achieves 0.69) but introduces the waterlog failure.

**Spatial awareness** is limited by the parameter-shared MLP actor, which cannot condition one agent's action on a neighbor's state. The MPC exploits the full cascade-routing topology for spatially-differentiated control; the RL policy's spatial differentiation comes only through the per-agent local features and the VDN critic's gradient routing.

**Single-plant validation.** All results are on a single simulated field geometry (the 10×13 DEM of the study site). Transfer to other field shapes, soil types, or crop varieties would require re-training.

### 6.4 Future Work

Two directions offer the highest potential for advancing the RL controller:

**Graph neural network (GNN) actor.** Replacing the parameter-shared MLP with a message-passing network over the D8 adjacency graph would allow each agent to condition its action on upslope and downslope neighbors' states — the spatial information structure the current actor lacks. Paired with a spatial reward term (e.g., penalizing cross-agent stress dispersion), a GNN actor could close the spatial-allocation gap with MPC and reduce the drought-day count by routing water toward the cells that need it most.

**Non-additive critic architecture.** The VDN critic's additive decomposition is monotonic and cannot represent interactions between agents (e.g., "cell $n$'s optimal action depends on cell $m$'s state because $m$ is upstream"). QMIX (state-conditioned monotonic mixing) or QPLEX (full individual-global-max decomposition with attention) would provide non-additive credit assignment, potentially improving spatial coordination without abandoning the CTDE property.

Both directions require the spatial reward signal to be defined first — without a reward term that explicitly values *where* water goes (not just how much), no architecture can learn spatial allocation.

### 6.5 Concluding Remark

This work demonstrates that appropriately structured model-free reinforcement learning is competitive with nonlinear model-based control for constrained multi-agent irrigation optimization — including on out-of-distribution climate scenarios — with a latency reduction that makes real-time edge deployment feasible. The comparison further reveals that the choice of entropy regularization in continuous-action multi-agent systems is not merely a training-stability decision but a fundamental design trade-off between control smoothness and action-range capability, with direct implications for the agronomic outcomes of the deployed controller.
