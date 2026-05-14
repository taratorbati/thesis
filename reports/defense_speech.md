# Preliminary Defense Speech — 10 Minutes
## Tara Torbati — ITMO University, R4237c
## Thesis: Modern Control Methods for Agricultural Irrigation

> **Delivery notes**
> - Target ~1,400 words at conversational pace.
> - Pauses marked with `[pause]`. Slide cues marked with `[Slide N]`.
> - Speak slowly through the technical sections (sections 3 and 4).
> - Look at committee members during the conclusions, not the slides.

---

## [Slide 1 — Title]

Good morning, members of the committee. My name is Tara Torbati, and over the next ten minutes I will present the work I have completed for my master's thesis: **Modern Control Methods for Agricultural Irrigation**, supervised by Professor Peregudin. [pause]

This work is a head-to-head comparison of Model Predictive Control and Reinforcement Learning on a high-fidelity simulation of a real Iranian rice field. The objective is to determine whether learning-based control can match the precision of optimization-based control in a constrained agricultural setting — and at what computational cost.

---

## [Slide 2 — Motivation]

Let me start with the problem. Globally, **agriculture consumes about seventy percent of all freshwater**, and by 2030 nearly half the world's population will live under water stress. Iran is one of the most acutely affected countries — its anthropogenic drought is now a security-level issue. My study site is in **Gilan Province**, on the slopes of the Talish Mountains. Gilan produces over two hundred thousand hectares of rice, which is by far the most water-intensive crop in the region. [pause]

Traditional irrigation in Gilan is reactive and open-loop. There is no forecast awareness, no constraint enforcement, and no spatial precision. The control engineering community has two main answers to this: **Model Predictive Control**, which can enforce strict water budgets but is computationally expensive, and **Reinforcement Learning**, which offers millisecond-latency decisions after training, but has no inherent constraint guarantees. The central question of this thesis is which of these is actually viable for a smallholder Iranian farmer operating under a binding seasonal water budget.

---

## [Slide 3 — Simulation Environment]

To answer that question, I built a high-fidelity simulation of a six-hectare rice field, parameterized for **130 individual crop-soil agents** arranged on a ten-by-thirteen grid over real satellite-derived elevation data. The model runs at a daily timestep over a 93-day Hashemi rice season and is driven by **25 years of NASA POWER climate data** for the actual coordinates of the site. [pause]

Each agent has five state variables: root-zone soil moisture, accumulated thermal time, a maturation stress index, accumulated biomass, and surface ponding. The agents are coupled through topographical surface-water routing — when an upslope agent's surface ponding exceeds its storage, the excess cascades downhill to its specific downstream neighbours based on the actual digital elevation model. [pause]

This is important. Several **mathematical corrections** to the baseline framework were needed to make this physically valid — most importantly, decoupling surface hydrology from subsurface hydrology to eliminate a mass-balance violation in the published baseline, and integrating drought stress directly into the biomass equation in the FAO AquaCrop style so that water deficit actually reduces yield rather than just accelerating senescence. Without these corrections, no controller comparison would be meaningful, because all controllers would produce indistinguishable yields. The corrected ABM was then **cross-validated against NASA MERRA-2 satellite soil-moisture observations** with a Pearson correlation of 0.74 on the no-irrigation baseline. [pause]

---

## [Slide 4 — MPC Design]

With the validated environment in place, I designed the Model Predictive Controller. The MPC solves a constrained nonlinear program at every daily timestep, with a receding 8-day horizon, **3,120 decision variables, and 2,081 constraints**. [pause]

The cost function has five active terms: a terminal biomass reward, a water cost term anchored to the Iranian domestic-base water tariff of 7,000 toman per cubic meter, a drought stress regularizer, an actuator smoothing penalty, and a soft penalty on root-zone soil moisture exceeding field capacity to prevent waterlogging. The weights for these five terms were not chosen heuristically — they were calibrated through a **four-phase, 33-configuration sensitivity sweep**, which is documented in detail in Chapter 4 of the thesis. That sweep resolved three pathologies in the initial formulation, including an irrigation-on-rainy-days behaviour caused by a runoff-pumping incentive in the cost. [pause]

The MPC results, under perfect forecast, are strong. In the **dry test year** at 100% budget, the MPC achieves 4,145 kilograms per hectare — a fifteen percent improvement over the fixed-schedule heuristic. In the **wet year**, the improvement is even larger, almost 35%, and the MPC voluntarily uses only 64% of the available budget — it recognizes incoming rainfall and withholds water preemptively. The fixed schedule, by contrast, blindly delivers 484 millimeters regardless of rain, producing 83.7 waterlog-days per agent in the wet year. [pause]

The downside is computational cost. Mean solve time per day is **about 26 seconds, with a worst case of 274 seconds for a single decision**. Over a 93-day season, that's 22 to 51 wall-minutes per scenario. Acceptable in simulation; impossible on a smallholder farmer's edge hardware.

---

## [Slide 5 — SAC Design]

That computational gap is what motivates the Reinforcement Learning controller. I implemented a Soft Actor-Critic agent with a **Centralized Training, Decentralized Execution** architecture. [pause]

The actor uses **parameter sharing across all 130 agents** — the same 62-input-dimension neural network is applied to each agent in parallel, producing all 130 actions in a single forward pass. This makes inference roughly **25,000 times faster than the MPC** — about one millisecond per decision on a standard CPU. [pause]

The critical design decision concerns the **critic architecture**. My initial implementation used a standard monolithic critic — an 837-dimensional input mapped to a scalar Q-value. After 200,000 training steps, that pilot run revealed two distinct failure modes which I want to be transparent about.

[pause] First, around step 100,000, the critic loss diverged catastrophically from about 0.7 to over 84,000. I diagnosed this to a terminal reward roughly 300 times larger than per-step rewards, combined with an entropy auto-tuner that spiked the temperature coefficient from 0.03 to 1.28 when the policy became too deterministic.

[pause] Second — and this is more fundamental — the resulting policy **collapsed to a spatially uniform 2 millimeters per day** applied to all 130 agents regardless of their location, soil state, or the weather forecast. **The policy was actually using more water in the wet year than in the dry year**, which is the opposite of correct behaviour. [pause]

I diagnosed the root cause as **spatial credit assignment failure**: a single scalar Q-value cannot tell the shared actor which of the 130 agents deserves credit for a given reward, so the actor cannot learn agent-specific policies even when the per-agent input features are differentiated. [pause]

The fix is an architectural upgrade to a **Value Decomposition Network critic**. Instead of one global Q-value, the critic decomposes Q-total as the sum of 130 local Q-values, each computed by a shared MLP that sees one agent's local state, the global context, and one agent's action. Because the derivative of a sum is linear, gradient flows cleanly through the summation node to the responsible agent. This change preserves the CTDE structure, requires no new hyperparameters, and is justified specifically because four of the five reward terms decompose exactly as additive per-agent contributions. [pause]

This corrected version 2.6 of the controller — with the VDN critic, the terminal bonus removed, the entropy coefficient hardcoded, and the actuator-smoothing penalty removed from the RL reward only — is currently in its five-seed training campaign on Kaggle. Preliminary diagnostic tests confirm that gradient signal is now reaching individual agents.

---

## [Slide 6 — Comparison & Conclusions]

So where does this leave the central question? [pause]

**On agronomic precision**, the MPC clearly leads. Across all completed evaluation cells, the MPC delivers between 492 and 969 kilograms per hectare more yield than the fixed-schedule baseline, eliminates waterlogging, and voluntarily returns water to the budget when rainfall is sufficient. **On computational efficiency**, the SAC leads by a factor of 25,000 — one millisecond versus 26 seconds per decision.

[pause] The diagnostic finding that motivated the VDN critic upgrade is itself, I would argue, a contribution: it shows precisely why naive monolithic SAC fails on spatial agricultural control, and it provides a minimal architectural change that addresses the failure without abandoning the CTDE framework. [pause]

The work that remains for the final defense is to **complete the 5-seed v2.6 training campaign**, run the **noisy-forecast robustness evaluation** for both controllers using the AR(1) forecast model that has already been implemented, and complete the three pending moderate-year MPC cells. [pause]

Thank you. I welcome your questions.

---

## Estimated timing

| Section | Words | Pace | Time |
|---|---|---|---|
| Title (Slide 1) | 85 | slow | 0:40 |
| Motivation (Slide 2) | 165 | normal | 1:10 |
| Environment (Slide 3) | 245 | careful | 1:50 |
| MPC (Slide 4) | 280 | careful | 2:00 |
| SAC (Slide 5) | 380 | careful, but firm | 2:50 |
| Conclusions (Slide 6) | 210 | confident, eye contact | 1:30 |
| **Total** | **~1365** | | **~10:00** |
