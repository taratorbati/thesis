# Controller Design, Parameter Calibration, and Sensitivity Analysis

**Ministry of Science and Higher Education of the Russian Federation**
**Federal State Autonomous Educational Institution of Higher Education**
**"National Research University ITMO"**
**«Faculty of Control Systems and Robotics»**

Field of study (specialty) 15.04.06 – Robotics and artificial intelligence

**REPORT** on Research Internship Assignment
Topic: Controller Design, Parameter Calibration, and Sensitivity Analysis for Constrained Irrigation Optimization

Student: Tara Torbati, group R4237c
Supervisor: Peregudin A.A.

Saint Petersburg, 2026

---

## Abstract

This report presents the complete mathematical formulation, architectural design, parameter calibration methodology, and sensitivity analysis for the irrigation controllers deployed on a 130-agent topographical agricultural field in Gilan Province, Iran. The controllers span a hierarchy of increasing sophistication: a rainfed no-irrigation baseline, a fixed-schedule heuristic reflecting traditional Gilan practices, a Model Predictive Controller (MPC) formulated as a large-scale Non-Linear Program (NLP) solved via the IPOPT interior-point method within the CasADi symbolic framework, and a Soft Actor-Critic (SAC) reinforcement learning agent trained under a Centralized Training with Decentralized Execution (CTDE) paradigm.

Each controller addresses the same constrained finite-horizon optimal control problem: maximizing terminal Hashemi rice biomass over a 93-day cultivation season while strictly adhering to a seasonal water budget of 484 mm at full allocation, with deficit scenarios at 85% and 70%. The MPC cost function comprises six normalized penalty terms — terminal biomass, water consumption, drought stress, surface ponding, control-rate regularization, and a soft penalty on root-zone moisture exceeding field capacity — with weights anchored to real Iranian water-pricing tiers ranging from subsidized agricultural rates (175 toman/m³) to industrial rates (115,000 toman/m³). A weight-sensitivity sweep covering twelve parameter configurations is described, designed to characterize the controller's response across the full economic and agronomic envelope. The Soft Actor-Critic architecture, the unified evaluation framework, the meteorological forecast uncertainty model, and the experimental grid are also formally documented. This chapter serves as the methodological bridge between the virtual plant described in the preceding system description and the comparative experimental results that follow.

---

## Contents

1. Introduction
2. Problem Formulation and System Constraints
   2.1 State Space Representation
   2.2 Action Space and Actuator Constraints
   2.3 Global Water Budget Constraint
   2.4 System Dynamics and Optimal Control Objective
3. Baseline and Heuristic Controllers
   3.1 The No-Irrigation Baseline ($C_1$)
   3.2 The Fixed-Schedule Heuristic ($C_2$)
4. Model Predictive Control (MPC) Architecture
   4.1 The Multi-Objective Cost Function
       4.1.1 Terminal Biomass (Mayer Cost)
       4.1.2 Water Consumption Penalty
       4.1.3 Drought Stress Regularization
       4.1.4 Ponding Penalty
       4.1.5 Actuator Smoothing (Control-Rate Regularization)
       4.1.6 Field-Capacity Overshoot Penalty
   4.2 Weight Calibration and Economic Anchoring
   4.3 Control-Oriented Model (COM) and Symbolic Formulation
       4.3.1 Dimensionality Reduction of the State Space
       4.3.2 Deterministic Caching of Biological Nonlinearities
       4.3.3 Static Embedding of Topographical Cascade Routing
       4.3.4 Smooth Approximations for Solver Convergence
   4.4 NLP Construction and Solver Configuration
       4.4.1 Decision Variable and Constraint Structure
       4.4.2 IPOPT Solver Configuration and Termination Criteria
       4.4.3 IPOPT Termination Status and Convergence Statistics
   4.5 Prediction Horizon Selection
   4.6 Warm-Starting and Computational Acceleration
5. Weight Sensitivity Analysis
   5.1 Motivation: Empirical Observations Driving the Sweep
   5.2 Sweep Strategy: One-at-a-Time Over Justified Ranges
   5.3 Sweep Group A — Water-Price Tier Sweep ($\alpha_2$)
   5.4 Sweep Group B — Drought Regularizer Sweep ($\alpha_3$)
   5.5 Sweep Group C — Ponding Weight Sweep at Wet/Hp=14 ($\alpha_4$)
   5.6 Sweep Group D — FC-Overshoot Penalty Sweep ($\alpha_6$)
   5.7 Sweep Group E — $\alpha_6$ Validation at Hp=14
   5.8 Sweep Results
   5.9 Recommended Operating Point
6. Soft Actor-Critic (SAC) Architecture
   6.1 Markov Decision Process Formulation
   6.2 Gymnasium Environment Wrapper
   6.3 Centralized Training with Decentralized Execution
   6.4 Budget Constraint Handling
   6.5 Planned Training Protocol and Hyperparameters
7. Uncertainty Modeling and Unified Evaluation Framework
   7.1 Meteorological Forecast Uncertainty
   7.2 Controller Summary Table
   7.3 Unified Software Evaluation Architecture
   7.4 Experimental Grid
8. Conclusion

---

# 1. Introduction

The preceding chapter established the high-fidelity Agent-Based Model (ABM) that serves as the virtual plant for this research: a 130-agent crop-soil simulation environment on a topographically heterogeneous field in Gilan Province, Iran, with five coupled state variables per agent, cascade surface water routing governed by a directed elevation graph, and validated hydrological dynamics (Pearson $r = 0.74$ against NASA GWETROOT satellite observations in the dry-year scenario). With the physical environment rigorously defined and parameterized for Hashemi rice cultivation, the present chapter addresses the central engineering question:

*how should the irrigation control policy be designed to maximize crop yield under strict water resource constraints, and how do model-based and learning-based approaches compare in this setting?*

The agricultural irrigation problem possesses several characteristics that make it particularly challenging from a control-theoretic perspective. First, the system dynamics are nonlinear and spatially coupled: surface runoff cascades through the directed elevation graph, creating complex interdependencies between the 130 agents that a controller must anticipate. Second, the system is subject to significant stochastic disturbances in the form of unpredictable precipitation events, which can range from zero rainfall over extended dry periods to concentrated storm events exceeding 20 mm/day. Third, the problem is subject to hard constraints at multiple scales: a per-agent actuator limit of 12.0 mm/day imposed by the physical irrigation infrastructure, and a global seasonal water budget that reflects regional water scarcity. Finally, the objective is inherently multi-criteria, requiring the controller to balance biomass maximization against water conservation, drought prevention, and waterlogging mitigation.

Beyond the design of each controller, this chapter also documents how the cost-function weights and the prediction horizon were selected, as these parameters carry substantial scientific and policy weight in their own right. The water-cost weight $\alpha_2$, for instance, encodes the economic environment in which the controller operates — agricultural water in Iran is priced approximately 660× lower than industrial water, and the same controller architecture operating under different price tiers can produce qualitatively different policies. Similarly, certain pathological behaviors observed during preliminary evaluation (notably the optimizer parking water above field capacity at long prediction horizons) motivated the introduction of an additional cost term, $\alpha_6$, whose calibration is the subject of a dedicated sensitivity sweep.

This chapter is organized as follows. Section 2 formalizes the constrained optimal control problem, defining the state space, action space, system dynamics, and the general cost structure. Section 3 presents the two open-loop baseline controllers that serve as lower-bound performance references. Section 4 details the Model Predictive Control architecture, including the six-term cost function, the Control-Oriented Model with its smooth approximation techniques, the NLP construction, the IPOPT solver configuration, the prediction horizon selection, and the weight calibration methodology. Section 5 introduces the weight sensitivity analysis, motivating it through specific empirical observations and laying out the five sweep groups that probe the controller's sensitivity along each calibration axis. Section 6 describes the Soft Actor-Critic reinforcement learning architecture, including the Markov Decision Process formulation, the CTDE network structure, the budget constraint handling mechanism, and the planned training protocol. Section 7 establishes the unified evaluation framework and the experimental grid. Section 8 concludes with a summary of the design decisions and their theoretical trade-offs.

---

# 2. Problem Formulation and System Constraints

The intelligent irrigation management of the 130-agent topographical grid is formally defined as a constrained, finite-horizon optimal control problem. Over the $K = 93$ day cultivation season for Hashemi rice, the control architecture must dynamically balance the conflicting objectives of maximizing terminal crop yield against the cost of water consumption, while strictly adhering to the physical limitations of the irrigation hardware and the regional water allocation. To ensure a standardized comparative evaluation across both traditional heuristics and advanced learning-based algorithms, the system state, action space, constraint structure, and transition dynamics must be rigorously defined.

## 2.1 State Space Representation

The discrete-time system state at day $k$, denoted as $\mathbf{X}_k$, aggregates the physical and biological conditions of all $N = 130$ spatial agents. The state space $\mathcal{X}$ is a multi-dimensional continuous space where each agent $n \in \{1, 2, \ldots, N\}$ is described by a state vector $x_k^n$ encompassing both hydrological and phenological variables:

$$\mathbf{X}_k = \{x_k^1, x_k^2, \ldots, x_k^N\} \in \mathcal{X} \tag{1}$$

where each $x_k^n$ consists of five state variables:

- $x_1^n$: Root-zone soil moisture (mm), bounded by $[WP \cdot \theta_5, \theta_{sat} \cdot \theta_5] = [60, 220]$ mm.
- $x_2^n$: Accumulated thermal time (°C·days), monotonically increasing from $x_{2,init} = 210$ to maturity threshold $\theta_{18} = 1250$.
- $x_3^n$: Crop maturity index (dimensionless), tracking cumulative heat and drought stress penalties.
- $x_4^n$: Accumulated aboveground dry biomass (g/m²), initialized at $x_{4,init} = 60$ g/m² reflecting the 20-day nursery period.
- $x_5^n$: Surface ponding depth (mm), representing un-infiltrated water remaining on the soil surface.

The full system state therefore lies in $\mathbb{R}^{5N} = \mathbb{R}^{650}$. To accurately simulate the initial flooded condition required for freshly puddled rice fields in Gilan, the system enforces a strict state initialization constraint across the entire spatial grid. At $k = 0$, the root-zone soil moisture is uniformly initialized to 100% field capacity:

$$x_1^n(0) = \theta_6 \cdot \theta_5 = 0.35 \times 400 = 140 \text{ mm} \quad \forall n \in \{1, \ldots, N\} \tag{2}$$

with $x_5^n(0) = 0$ mm (no initial ponding). The thermal time and biomass are initialized at their nursery-calibrated values $x_2^n(0) = 210$ and $x_4^n(0) = 60$ g/m² respectively, while $x_3^n(0) = 0$.

## 2.2 Action Space and Actuator Constraints

The control action at day $k$, denoted as $\mathbf{U}_k$, dictates the daily irrigation depth applied to each agent. The action space $\mathcal{U}$ is bounded by the physical realities of the pumping and distribution infrastructure. Let $u_k^n$ represent the irrigation command dispatched to agent $n$ on day $k$. The controller is strictly constrained by a maximum daily actuator limit, $u_{\max}$, establishing a bounded continuous action space:

$$\mathcal{U} = \left\{ \mathbf{U}_k \in \mathbb{R}^N \mid 0 \leq u_k^n \leq u_{\max} \;\; \forall n \in \{1, \ldots, N\} \right\} \tag{3}$$

Based on the physical capacities of precision drip infrastructure and the hydraulic constraints of the Gilan field, the actuator cap is defined as $u_{\max} = 12.0$ mm/day. This value represents the maximum volumetric delivery rate achievable by a single irrigation zone without exceeding the soil infiltration capacity or generating excessive surface runoff. Any policy $\pi$ or optimization solver must ensure $\mathbf{U}_k \in \mathcal{U}$ at every discrete time step.

## 2.3 Global Water Budget Constraint

A defining feature of the constrained irrigation problem, and the primary motivation for this research in the context of Iran's anthropogenic drought crisis, is the presence of a hard global resource constraint. The total cumulative irrigation volume applied across all agents over the entire growing season must not exceed a finite seasonal water budget $W_{total}$:

$$\sum_{k=0}^{K-1} \sum_{n=1}^{N} u_k^n \leq W_{total} \cdot N \tag{4}$$

where $W_{total}$ is expressed in mm (equivalent depth per agent). The full irrigation requirement for Hashemi rice, derived from the 25-year climatological water budget analysis, is $W_{full} = 484$ mm. To evaluate controller robustness under increasing water scarcity, three budget tiers are defined:

**Table 2.1: Seasonal water budget tiers for Hashemi rice.**

| Budget Tier | Fraction of $W_{full}$ | $W_{total}$ (mm) | Scarcity Context |
|---|---|---|---|
| Full allocation | 100% | 484.0 | Nominal water availability |
| Moderate deficit | 85% | 411.4 | Regional quota reduction |
| Severe deficit | 70% | 338.8 | Drought-year rationing |

The budget values are anchored to the agronomic water demand calculation $\text{ET}_c = \text{ET}_0^{PM} \cdot K_c \cdot N_{days} = 5.02 \times 1.15 \times 93 = 537.0$ mm, minus the 25-year average seasonal rainfall of 53.3 mm, yielding a full irrigation need of $I_{full} \approx 484$ mm. The 85% and 70% tiers reflect realistic policy scenarios: the moderate deficit corresponds to a 15% volumetric quota reduction that water authorities in Iranian provinces have historically imposed during low-flow years, while the severe deficit represents the extreme rationing required during consecutive drought years.

This constraint is fundamentally different in character for MPC versus RL. The MPC enforces it as a hard linear inequality constraint within the NLP formulation (Section 4.4), guaranteeing mathematical satisfaction. The RL agent, lacking native constrained-optimization mechanisms, must learn to respect the budget through reward shaping, action clipping, and early termination penalties (Section 6.4).

## 2.4 System Dynamics and Optimal Control Objective

The system transitions from state $\mathbf{X}_k$ to $\mathbf{X}_{k+1}$ via the nonlinear system dynamics $f_{system}$, which encompass both the topographical cascade routing and the biological growth models defined in the preceding chapter:

$$\mathbf{X}_{k+1} = f_{system}(\mathbf{X}_k, \mathbf{U}_k, \mathbf{W}_k) \tag{5}$$

where $\mathbf{W}_k$ represents the stochastic meteorological disturbance vector at day $k$, comprising rainfall $P(k)$, reference evapotranspiration $\text{ET}_0(k)$, mean/max/min temperatures $T_{mean}(k)$, $T_{max}(k)$, $T_{min}(k)$, and solar radiation $R_s(k)$. The overarching objective for any deployed controller is to find a policy $\pi(\mathbf{X}_k) \to \mathbf{U}_k$ that minimizes an expected cumulative cost function $J$ over the finite horizon $K$:

$$\min_\pi \mathbb{E}_{\mathbf{W}} \left[ \Phi(\mathbf{X}_K) + \sum_{k=0}^{K-1} L(\mathbf{X}_k, \mathbf{U}_k) \right] \tag{6}$$

subject to $\mathbf{U}_k \in \mathcal{U}$ and the global budget constraint (Eq. 4), where $\Phi(\mathbf{X}_K)$ represents the terminal Mayer cost (maximizing the terminal biomass $x_4$ at harvest), and $L(\mathbf{X}_k, \mathbf{U}_k)$ represents the intermediate Lagrange path costs penalizing cumulative water expenditure, localized ponding ($x_5$), drought stress, actuator oscillation, and excursion of soil moisture above field capacity. This general optimal control formulation serves as the architectural foundation. The subsequent sections detail how this objective is uniquely approximated and solved by the baseline heuristics, the predictive optimizer (MPC), and the reinforcement learning agent (SAC).

---

# 3. Baseline and Heuristic Controllers

To rigorously evaluate the performance of the advanced closed-loop predictive (MPC) and learning-based (SAC) algorithms, two open-loop baseline controllers are implemented. These baselines represent the lower bound of crop survival and the standard agricultural heuristic currently deployed in Gilan Province, respectively. Since these are open-loop heuristics, they calculate the irrigation trajectory strictly as a function of time and total seasonal water budget, remaining entirely blind to both the daily topographical field state ($\mathbf{X}_k$) and the meteorological forecast.

## 3.1 The No-Irrigation Baseline ($C_1$)

The No-Irrigation controller ($C_1$) serves as the absolute lower-bound performance metric (the rainfed yield ceiling) and verifies that the simulation environment produces meaningful biomass when water is freely available from natural rainfall alone. The control action is identically zero across all agents and all days:

$$u_k^n = 0 \quad \forall n \in \{1, \ldots, N\}, \;\; \forall k \in \{0, \ldots, K-1\} \tag{7}$$

This baseline establishes the survival floor: any active controller that fails to outperform $C_1$ provides no value, while the gap between $C_1$ and an irrigated controller quantifies the marginal yield contribution of irrigation under each scenario.

## 3.2 The Fixed-Schedule Heuristic ($C_2$)

The Fixed-Schedule controller ($C_2$) reflects the traditional irrigation practice in the Gilan rice-growing region: water is delivered on a pre-planned calendar across 19 events per season with linearly decaying daily volume, allocated such that the cumulative seasonal volume exactly equals the assigned budget tier. The 19-event schedule is anchored to the rice phenological calendar, with greater frequency during the vegetative and reproductive phases (DOY 141–200) and tapering during ripening (DOY 200–233). The daily volume on each scheduled irrigation day $k_j$ is computed as:

$$u_{k_j}^n = \frac{2 W_{total}}{19} \cdot \frac{20 - j}{19} \quad \text{for } j \in \{1, \ldots, 19\} \tag{8}$$

with $u_k^n = 0$ on all non-scheduled days. The same schedule is applied uniformly across all 130 agents, regardless of topographical position or local soil moisture state. This heuristic matches the volumetric expectation of the budget while encoding the seasonal weighting that experienced Iranian rice farmers apply intuitively.

---

# 4. Model Predictive Control (MPC) Architecture

The Model Predictive Controller ($C_3$ in the perfect-forecast configuration, $C_4$ in the noisy-forecast configuration) is implemented as a receding-horizon optimization problem solved at every daily timestep. At each step $k$, the controller observes the true plant state $\mathbf{X}_k$, queries the meteorological forecast over the prediction horizon $H_p$, formulates and solves a Non-Linear Program (NLP), applies the first-step optimal action $\mathbf{U}_k^*$ to the plant, and discards the remainder of the predicted control trajectory.

## 4.1 The Multi-Objective Cost Function

The MPC cost function comprises six normalized penalty terms, balancing terminal crop performance against intermediate operational costs and physical risk factors. Each term is constructed to remain $\mathcal{O}(1)$ in magnitude per timestep, ensuring that the cost-weight calibration is interpretable as a direct economic priority ratio. The total cost is:

$$J = J_{biomass} + J_{water} + J_{drought} + J_{ponding} + J_{\Delta u} + J_{overFC} \tag{9}$$

The six terms are detailed below.

### 4.1.1 Terminal Biomass (Mayer Cost)

The primary objective of the controller is to maximize the field-mean terminal biomass at harvest, achieved by penalizing its negative value normalized by a reference biomass $x_{4,ref}$:

$$J_{biomass} = -\alpha_1 \cdot \frac{1}{N} \sum_{n=1}^N \frac{x_4^n(K)}{x_{4,ref}} \tag{10}$$

The reference value $x_{4,ref} = 900$ g/m² corresponds to a target yield of $\sim$3,800 kg/ha for Hashemi rice (using harvest index $HI = 0.42$), which represents the mean of the operational yield range observed across nominal Gilan irrigation practices. The weight $\alpha_1 = 1.0$ serves as the anchor against which all other weights are calibrated.

### 4.1.2 Water Consumption Penalty

To incorporate the economic cost of irrigation water, a path penalty proportional to the daily field-total irrigation volume is included:

$$J_{water} = \alpha_2 \sum_{j=0}^{H_p-1} \frac{1}{W_{daily,ref}} \sum_{n=1}^N u^n(k+j) \tag{11}$$

where $W_{daily,ref} = 5.0 \cdot N = 650$ is a normalization reference based on the mean seasonal $\text{ET}_c \approx 5.8$ mm/day rounded to 5.0 mm. The weight $\alpha_2$ is calibrated against real Iranian water-pricing tiers, as detailed in Section 4.2.

### 4.1.3 Drought Stress Regularization

While the biological model penalizes growth during dry periods through the drought stress function $h_3$, an explicit penalty is added to the controller objective to aggressively prevent soil moisture ($x_1$) from dropping below the stress threshold ($ST$):

$$J_{drought} = \alpha_3 \sum_{j=0}^{H_p-1} \frac{1}{N} \sum_{n=1}^N \frac{\max(0, ST - x_1^n(k+j))}{ST - WP} \tag{12}$$

where $ST = (\theta_6 - p \cdot (\theta_6 - \theta_2)) \cdot \theta_5 = (0.35 - 0.20 \times 0.20) \times 400 = 124$ mm is the management allowable depletion threshold for rice ($p = 0.20$ per FAO-56), and $WP = \theta_2 \cdot \theta_5 = 0.15 \times 400 = 60$ mm is the permanent wilting point. The weight $\alpha_3 = 0.1$ ensures the controller proactively irrigates before severe deficit occurs, without dominating the biomass incentive.

### 4.1.4 Ponding Penalty

Due to the topographical cascade routing, excessive rainfall and over-irrigation can cause temporary localized flooding. Following the fractional routing fix (terrain.py v2.0), which eliminated true sink agents by allowing off-farm drainage through the padded DEM, the ponding penalty is applied to the field-mean surface ponding across all 130 agents:

$$J_{ponding} = \alpha_4 \sum_{j=0}^{H_p-1} \frac{1}{N} \sum_{n=1}^N \frac{x_5^n(k+j)}{x_{5,ref}} \tag{13}$$

where $x_{5,ref} = 50.0$ mm serves as the normalization reference representing acute storm ponding depth, and $\alpha_4 = 0.5$ places a high priority on preventing persistent waterlogging across the field. The choice of $x_{5,ref} = 50$ mm reflects the realistic ponding depths observed during monsoon-season storm events in Gilan, ensuring that the penalty term remains $\mathcal{O}(1)$ during typical wet-weather conditions rather than saturating the cost function.

### 4.1.5 Actuator Smoothing (Control-Rate Regularization)

To prevent erratic, high-frequency oscillatory irrigation commands across consecutive days — which would be physically damaging to valve actuators and operationally impractical — a standard control-rate regularization term is included:

$$J_{\Delta u} = \alpha_5 \sum_{j=1}^{H_p-1} \frac{\| \mathbf{u}(k+j) - \mathbf{u}(k+j-1) \|^2}{u_{\max}^2 \cdot N} \tag{14}$$

where the normalization by $u_{\max}^2 \cdot N$ ensures this term remains $\mathcal{O}(1)$ even at maximum actuator swing. The weight $\alpha_5 = 0.005$ provides gentle smoothing without materially constraining the optimizer's freedom to respond to sudden rainfall events.

### 4.1.6 Field-Capacity Overshoot Penalty

A sixth cost term penalizes excursion of root-zone soil moisture above field capacity ($FC = \theta_6 \cdot \theta_5 = 140$ mm). The penalty is quadratic in the normalized excess and acts only when $x_1 > FC$:

$$J_{overFC} = \alpha_6 \sum_{j=0}^{H_p-1} \frac{1}{N} \sum_{n=1}^N \left[ \frac{\max(0, x_1^n(k+j) - FC)}{FC} \right]^2 \tag{15}$$

The motivation for this term is the following empirical observation. The ABM permits $x_1 > FC$ as a transient physical state — gravitational water held in the saturation zone drains at a rate of $\theta_4 = 5\%$ per day, so excess water above field capacity persists for approximately two weeks before fully draining. While this is correct soil physics, it allowed the optimizer at long horizons to deliberately park soil moisture well above FC, treating the soil as an unbounded reservoir. The waterlogging stress $h_6$ embedded in the ABM's biomass increment $x_4^{inc} = \theta_{13} \cdot h_3 \cdot h_6 \cdot h_7 \cdot g \cdot R_s$ provides only an indirect, multiplicative disincentive that is easily offset within a finite prediction horizon by improved transpiration on subsequent days. Figure 4.1 illustrates the resulting pathology in a representative dry-year scenario at full budget allocation.

**Figure 4.1**: *Field-mean soil-moisture trajectories across three prediction horizons under the dry/100% scenario, plotted with $\alpha_6 = 0$. Field capacity ($FC = 140$ mm) is marked as a horizontal dashed line. The Hp=14 controller (red) keeps soil moisture above FC for 21.6 agent-days on average across the 93-day season, while Hp=8 (orange) does so for only 1.9 days and Hp=3 (blue) effectively never crosses FC. The phenomenon is horizon-specific: it does not manifest at short horizons, where the optimizer cannot foresee enough future evapotranspiration demand to justify pre-filling the soil. Without the $\alpha_6$ penalty, longer horizons paradoxically produce lower yield in saturated conditions despite their theoretically richer information set.*

The quadratic shape of $J_{overFC}$ is chosen so that small transient overshoots within normal ABM dynamics (e.g., $x_1 = 145$ mm for a single day after a rain event, contributing a penalty of order $0.001$ per agent) are barely penalized, while persistent excursions deep into the saturation zone (e.g., $x_1 = 200$ mm, contributing 0.184 per agent per step) produce a steep monotonic disincentive. The term is fully $C^2$-smooth via the same `smooth_max_zero` operator used elsewhere in the COM (Section 4.3.4).

In the default operating point, $\alpha_6 = 0$ and the term is inactive — this preserves direct comparability with the original five-term formulation. The active value of $\alpha_6$ is determined through the sensitivity sweep in Section 5.6.

## 4.2 Weight Calibration and Economic Anchoring

The six cost weights $\{\alpha_1, \alpha_2, \alpha_3, \alpha_4, \alpha_5, \alpha_6\}$ are not arbitrary tuning parameters but are calibrated to reflect real economic and agronomic trade-offs. The primary anchor is $\alpha_1 = 1.0$, representing the revenue from a hectare of Hashemi rice at typical Gilan market prices (500,000 toman/kg $\times$ 4,200 kg/ha $\approx 2.1$ billion toman/ha). The water cost weight $\alpha_2$ is calibrated against this anchor using the volumetric price of irrigation water in Iran. Because Iran operates a multi-tier water-pricing system in which the same volume of water can cost orders of magnitude more depending on the user category, four economically grounded values of $\alpha_2$ are evaluated:

**Table 4.1: $\alpha_2$ calibration against Iranian water-pricing tiers.**

| Tier | Price (toman/m³) | Full-budget cost (toman/ha) | % of revenue | $\alpha_2$ |
|---|---:|---:|---:|---:|
| Subsidized agricultural (real) | 175 | 847,000 | 0.04% | 0.0004 |
| Domestic base (≤5 m³/month per family member) | 7,000 | 33,880,000 | 1.61% | 0.016 |
| Domestic tier d (×2.8) | 19,000 | 91,960,000 | 4.38% | 0.044 |
| Industrial | 115,000 | 556,600,000 | 26.5% | 0.265 |

The conversion uses a target yield of 4,198 kg/ha at 500,000 toman/kg, full-budget volume of 4,840 m³/ha, and equates the cost-fraction-of-revenue with the dimensionless cost-function ratio. The pricing data are drawn from the Iran Water Resources Management Company tariff schedule (1402-1403 / 2023-2024), the Stanford Iran 2040 Project water-scarcity analysis [10], and the agricultural water governance review of Nouri et al. [9]. The four tiers are deliberately spread across nearly three orders of magnitude to characterize how the controller's policy adapts as the economic environment shifts from heavily subsidized agricultural pricing to industrial-scarcity pricing.

The weights $\alpha_3 = 0.1$ (drought regularization), $\alpha_4 = 0.5$ (ponding), and $\alpha_5 = 0.005$ (control-rate) follow standard regularization-magnitude conventions and are confirmed via the sensitivity sweep in Section 5. The weight $\alpha_6$ for the field-capacity overshoot penalty is initially set to $0$ in the default operating point; its active calibration is determined by Sweep Group D (Section 5.6) and validated at the longest horizon by Sweep Group E (Section 5.7).

The complete weight vector at the default operating point is therefore:
$$\boldsymbol{\alpha}_{default} = \{1.0, 0.01, 0.1, 0.5, 0.005, 0.0\}$$

with $\alpha_2 = 0.01$ chosen as a moderate intermediate value between subsidized and domestic-base pricing for the initial parameter exploration. The recommended operating point following the sensitivity sweep is reported in Section 5.9.

## 4.3 Control-Oriented Model (COM) and Symbolic Formulation

To enable real-time numerical optimization within the receding horizon framework, the high-fidelity Agent-Based Model (ABM) described in the preceding chapter must be translated into a Control-Oriented Model (COM). The COM is implemented as a CasADi symbolic computational graph using the SX (Scalar eXpression) framework, defining the explicit mapping $f_{COM}: (\mathcal{X}_k, \mathcal{U}_k, \mathcal{P}_k) \to \mathcal{X}_{k+1}$, where $\mathcal{P}_k$ represents exogenous parameters (weather forecasts and precomputed biological quantities). To ensure tractability for the IPOPT interior-point solver, several rigorous dimensionality reduction and computational strategies are applied.

### 4.3.1 Dimensionality Reduction of the State Space

While the complete biological system tracks five state variables per agent, maintaining all variables as shooting states in the Non-Linear Programming (NLP) formulation would result in $5 \times 130 \times H_p = 5{,}200$ shooting variables at $H_p = 8$ alone, producing prohibitively large and dense Jacobian matrices. Because thermal time ($x_2$) is purely driven by exogenous climate data and is therefore entirely decoupled from the control action, it is precomputed offline for the full season and injected as a time-varying parameter. Maturity ($x_3$) acts as a slow-moving, monotonic biological clock that is tracked from the plant's true state at each receding-horizon step to prevent open-loop drift. Biomass ($x_4$) is accumulated inline within the COM strictly to evaluate the terminal Mayer cost but is not used as a shooting state. The COM therefore restricts the formal shooting states exclusively to the fast-moving hydrological variables:

$$\mathcal{X}_{shoot} = \{x_1, x_5\} \tag{16}$$

This reduces the shooting variables from $5N$ to $2N = 260$ per horizon step, yielding a total of $2N \cdot H_p + N \cdot H_p = 3N \cdot H_p$ NLP variables (two states plus one control per agent per step) plus the initial state parameters.

### 4.3.2 Deterministic Caching of Biological Nonlinearities

The biological penalty functions for heat stress ($h_2$), cold stress ($h_7$), thermal time accumulation ($h_1$), and the base growth function ($g_{base}$) rely on highly nonlinear sigmoid formulations. Because these functions depend strictly on deterministic meteorological variables (temperature and radiation) and are entirely decoupled from the control action ($u$), they are precomputed offline for the entire 93-day season. By injecting these precomputed scalar arrays into the CasADi graph as time-varying parameters rather than calculating them inline, the NLP solver bypasses the expensive evaluation of sigmoidal gradients and their second derivatives, significantly accelerating the per-step solve time and improving Hessian sparsity.

### 4.3.3 Static Embedding of Topographical Cascade Routing

A primary challenge in agricultural MPC is simulating spatial water routing without relying on computationally prohibitive coupled Partial Differential Equations (PDEs). The COM resolves this by statically embedding the field's directed topographical graph into the symbolic formulation. At NLP construction time, the 130 agents are sorted into a strict topological order (from maximum elevation to terminal boundary agents). During the symbolic forward pass of each horizon step, surface hydrology is unrolled sequentially: for a given agent $n$ processed at position $p$ in the topological order, all uphill contributions from agents at positions $1, \ldots, p-1$ have already been computed and are available as symbolic expressions.

The surface water availability for agent $n$ is computed as:

$$W_{surf}^{(n)}(k) = x_5^{(n)}(k) + P(k) + u^{(n)}(k) + \sum_{m \in U(n)} \frac{\phi_2^{(m)}(k)}{N_r^{(m)}} \tag{17}$$

where $U(n)$ is the set of uphill neighbors of agent $n$, and $N_r^{(m)}$ is the *total* number of lower-elevation neighbors of agent $m$, computed using the padded DEM (terrain.py v2.0). This fractional routing boundary condition is critical: the DEM is padded by one cell on each edge by extrapolating the slope. $N_r$ counts all lower neighbors including external (off-farm) pad cells, while the routing loop only sends water to internal agents. When the COM divides $\phi_2$ by $N_r$, the fraction directed toward pad cells is implicitly removed from the mass balance, accurately simulating off-farm drainage without introducing dummy variables to the solver. This fixed the "bathtub effect" observed in the initial implementation, where three sink agents ($N_r = 0$) trapped all cascade runoff and experienced catastrophic waterlogging.

### 4.3.4 Smooth Approximations for Solver Convergence

The original ABM equations contain several non-smooth operations — $\max(0, x)$, $\min(a, b)$, and conditional branching via `ca.if_else` — that create discontinuities in the first and second derivatives of the NLP objective and constraints. While CasADi can represent these operations symbolically, the IPOPT interior-point solver requires $C^2$-continuous (twice continuously differentiable) functions to reliably compute the Hessian of the Lagrangian and perform Newton steps. Using exact non-smooth operators caused severe convergence failures: IPOPT would stall at "Restoration Failed" or "Search Direction Becomes Too Small" after encountering kinks in the derivative landscape.

To resolve this, smooth $C^2$-continuous approximations are employed with a smoothing parameter $\varepsilon = 0.01$:

**Smooth maximum:** The half-rectifier $\max(0, x)$ is replaced by:

$$\widetilde{\max}(0, x) = \frac{x + \sqrt{x^2 + \varepsilon^2}}{2} \tag{18}$$

This function satisfies $\widetilde{\max}(0, x) \to \max(0, x)$ as $\varepsilon \to 0$, is everywhere differentiable, and introduces an error bounded by $\varepsilon/2 = 0.005$ mm — negligible relative to the soil moisture range of $[60, 220]$ mm.

**Smooth minimum:** The $\min(a, b)$ operation is derived from the smooth maximum via:

$$\widetilde{\min}(a, b) = a + b - \widetilde{\max}(a, b) = \frac{a + b - \sqrt{(a-b)^2 + \varepsilon^2}}{2} \tag{19}$$

**Guarded denominator for drought stress:** The drought stress function $h_3 = 1 - \theta_{14} \cdot \max(1 - \phi_1 / \text{ET}_c, 0)$ contains a division by $\text{ET}_c$ that approaches zero on cool, humid days when the atmospheric evaporative demand vanishes. To prevent division by zero in the symbolic graph, the denominator is guarded:

$$h_3^{(n)}(k) = 1 - \theta_{14} \cdot \widetilde{\max}\left(1 - \frac{\phi_1^{(n)}(k)}{\text{ET}_c(k) + \varepsilon},\, 0\right) \tag{20}$$

**SCS runoff smoothing:** The conditional SCS curve number computation originally used `ca.if_else`, which creates a derivative discontinuity at $W_{surf} = \theta_3$. This is replaced by the smooth formulation:

$$\phi_2^{(n)} = \frac{\left[\widetilde{\max}(0, W_{surf}^{(n)} - \theta_3)\right]^2}{W_{surf}^{(n)} + 4\theta_3} \tag{21}$$

The smoothing acts purely through the numerator's $\widetilde{\max}$ operator, so when $W_{surf} < \theta_3$, the squared term goes smoothly to zero; when $W_{surf} > \theta_3$, the original SCS formula is recovered up to $\mathcal{O}(\varepsilon^2)$ smoothing error.

The value $\varepsilon = 0.01$ was selected through systematic experimentation: values below $0.001$ reintroduced convergence failures due to near-discontinuous derivatives, while values above $0.1$ introduced unacceptable mass-balance artifacts (artificial water creation exceeding 0.5 mm/season). At $\varepsilon = 0.01$, the total mass-balance error is bounded below 0.05 mm over the 93-day season, which is four orders of magnitude smaller than the typical seasonal water budget.

The same `smooth_max_zero` operator is reused in the new $J_{overFC}$ term (Section 4.1.6), so the entire NLP — dynamics, constraints, and objective — is uniformly $C^2$-smooth.

## 4.4 NLP Construction and Solver Configuration

### 4.4.1 Decision Variable and Constraint Structure

The NLP is constructed using a direct multiple-shooting formulation. At each prediction horizon step $j \in \{0, \ldots, H_p - 1\}$, the decision variables consist of the shooting states $\{x_1^n(k+j), x_5^n(k+j)\}$ for all $N = 130$ agents and the control inputs $\{u^n(k+j)\}$ for all agents. The complete variable vector $\mathbf{z}$ is organized as:

$$\mathbf{z} = \underbrace{[x_1^{1:N}(k), x_5^{1:N}(k)]}_{\text{initial states (fixed)}} ,\, \underbrace{u^{1:N}(k), x_1^{1:N}(k+1), x_5^{1:N}(k+1), u^{1:N}(k+1), \ldots}_{\text{free variables}} \tag{22}$$

For prediction horizon $H_p = 8$, this yields the following NLP dimensions:

**Table 4.2: NLP dimensions for $H_p = 8$ and $H_p = 14$.**

| Quantity | $H_p = 8$ | $H_p = 14$ |
|---|---:|---:|
| Total symbolic variables | 3,380 | 5,720 |
| Free decision variables | 3,120 | 5,460 |
| Equality constraints (shooting gaps) | 2,080 | 3,640 |
| Inequality constraint (budget) | 1 | 1 |
| Total constraints | 2,081 | 3,641 |
| Box bounds on $u$ | $2 \times 130 \times H_p$ | $2 \times 130 \times H_p$ |
| NLP build time (typical) | 20–50 s | 37–44 s |

The 260 initial-state variables ($x_1^{1:N}(k)$ and $x_5^{1:N}(k)$) are fixed to the current true plant state at each receding-horizon step (they appear as parameters, not free variables). The 2,080 equality constraints at $H_p = 8$ enforce the shooting gap closure: each constraint requires that the symbolic COM-predicted next state matches the shooting variable at the subsequent step. The single inequality constraint enforces the remaining seasonal water budget.

### 4.4.2 IPOPT Solver Configuration and Termination Criteria

The NLP is solved using IPOPT (Interior Point OPTimizer), an open-source primal-dual interior-point algorithm for large-scale nonlinear optimization [4]. IPOPT implements a line-search filter method based on the barrier reformulation of the KKT (Karush–Kuhn–Tucker) optimality conditions. At each iteration, IPOPT solves a symmetric indefinite linear system to obtain the Newton search direction, using the MUMPS (MUltifrontal Massively Parallel Sparse direct Solver) as the default linear algebra backend.

The solver is configured with parameters tuned for the irrigation-scale NLP. Because the cost function and constraints involve smooth approximations of physically non-smooth operations (Section 4.3.4), the convergence tolerances are relaxed slightly relative to IPOPT defaults to avoid excessive iteration counts at high-fidelity precision that the underlying physics does not require:

**Table 4.3: IPOPT solver configuration.**

| Option | Value | Description |
|---|---|---|
| `tol` | $10^{-4}$ | Desired convergence tolerance. The agricultural irrigation problem does not require machine-precision optimality; relaxing this from the IPOPT default $10^{-8}$ to $10^{-4}$ provides $\sim$2× speedup with negligible practical effect on the irrigation profile. |
| `acceptable_tol` | $10^{-3}$ | Acceptable convergence tolerance. If the algorithm encounters `acceptable_iter` consecutive iterates satisfying this relaxed tolerance, it terminates with status code 1. |
| `acceptable_iter` | 10 | Number of consecutive acceptable iterates required for early termination. |
| `max_iter` | 500 | Maximum number of interior-point iterations per NLP solve. |
| `mu_init` | $10^{-2}$ | Initial barrier parameter. Larger than the IPOPT default to provide more robust handling of the smooth-approximation transitions. |
| `nlp_scaling_method` | `gradient-based` | Auto-scale the NLP based on gradient magnitudes at the initial point. |
| `linear_solver` | `MUMPS` | Sparse direct solver for the KKT system. |
| `warm_start_init_point` | `yes` | Use warm-start initial guess from the previous solve (Section 4.6). |
| `print_level` | 0 | Suppress IPOPT console output during batch runs. |

The relaxed tolerance choices are appropriate given the underlying problem characteristics: soil moisture is meaningful at the 1 mm scale, irrigation actions at the 0.1 mm scale, and the smooth-approximation error itself is already $\mathcal{O}(0.005)$ mm. Tightening `tol` below $10^{-4}$ produces solutions whose differences are physically indistinguishable while substantially increasing iteration counts.

### 4.4.3 IPOPT Termination Status and Convergence Statistics

IPOPT reports one of several exit conditions, of which two are relevant to the completed simulations:

**Solve_Succeeded (Return code 0):** This status indicates that IPOPT found a locally optimal point satisfying all desired convergence criteria simultaneously: the scaled NLP error is below `tol = 1e-4`, the constraint violation is below `constr_viol_tol = 1e-4`, and the complementarity is below `compl_inf_tol = 1e-4`. When all conditions hold, the returned solution represents a KKT point of the NLP to the desired numerical precision.

**Solved_To_Acceptable_Level (Return code 1):** This status indicates that the algorithm could not converge to the desired tolerance specified by `tol` but did find a point satisfying the relaxed acceptable tolerance level. Specifically, the IPOPT iteration encountered 10 consecutive iterations during which the scaled NLP error remained below `acceptable_tol = 1e-3`. This outcome typically occurs when the NLP is poorly conditioned at a particular timestep — for example, when a sudden large rainfall event creates a near-discontinuity in the optimal irrigation profile despite the smooth approximations, or when the cost gradient becomes near-flat in the saturation regime.

The practical impact of `Solved_To_Acceptable_Level` on irrigation quality is minimal in this application. The constraint violation tolerance is relaxed from $10^{-4}$ mm to $10^{-2}$ mm, which for a soil moisture range of $[60, 220]$ mm represents less than 0.01% relative error.

The convergence statistics collected across the 27 completed MPC runs (3 scenarios × 3 budgets × 3 horizons, 837 daily solves per horizon group) are summarized in Table 4.4.

**Table 4.4: IPOPT solve-time statistics across 837 daily solves per horizon group (27 nominal MPC runs).**

| Metric | $H_p = 3$ | $H_p = 8$ | $H_p = 14$ |
|---|---:|---:|---:|
| Mean solve time (s) | 3.2 | 23.5 | 105.0 |
| Median solve time (s) | $\sim$2.5 | 9.9 | $\sim$45 |
| Maximum solve time (s) | 62.6 | 508.1 | 15,556.3 |
| Solves > 60 s | 0 | 72 | 334 |
| Solves > 300 s | 0 | 5 | 48 |
| Total wallclock per scenario × budget (range, hours) | 0.05–0.1 | 0.7–1.4 | 1.0–7.6 |

The increasing prevalence of long solves at $H_p = 14$ (40% of solves take longer than 60 seconds) reflects the non-convexity of the NLP: at certain receding-horizon steps, particularly under wet-scenario saturated-soil conditions, the cost gradient becomes near-flat and IPOPT requires many iterations to reach the acceptable tolerance level.

**Other possible status codes** (rare or not observed in the completed simulations) include `Maximum_Iterations_Exceeded` (code -1), `Restoration_Failed` (code -2), `Infeasible_Problem_Detected` (code 2), `Search_Direction_Becomes_Too_Small` (code 3), and `Error_In_Step_Computation` (code -3). In any of these failure modes, the controller defaults to a zero-irrigation fail-safe ($u_k^n = 0$ for all $n$), allowing the physical crop to survive on natural rainfall until the receding horizon re-establishes a feasible trajectory on the subsequent day.

## 4.5 Prediction Horizon Selection

The prediction horizon $H_p$ governs the trade-off between the MPC's look-ahead depth and its computational cost. Three values are evaluated in this research, spanning a logarithmic range of look-ahead depths:

**$H_p = 3$ days:** A short tactical horizon that approximates the most reliable portion of operational meteorological forecasts (the 1-3 day window where forecast skill is highest). At this horizon, the NLP contains 1,170 free decision variables and 781 constraints. Per-step solve times are on the order of 3 seconds, making this horizon well-suited to edge-device deployment.

**$H_p = 8$ days:** A medium horizon spanning approximately one week of future weather, corresponding to the practical accuracy limit of operational meteorological forecasts. The NLP contains 3,120 free decision variables and 2,081 constraints, with mean per-step solve times of 23.5 seconds.

**$H_p = 14$ days:** An extended horizon doubling the look-ahead, allowing the MPC to plan across a full two-week window. The NLP grows to 5,460 free variables and 3,641 constraints. Mean per-step solve times rise to 105 seconds, with worst-case solves observed up to 15,556 seconds — these extreme cases occur when the NLP is poorly conditioned near saturation and IPOPT exhausts its iteration budget before terminating at the acceptable tolerance.

A definitive recommendation regarding the optimal $H_p$ value is deferred to the sensitivity analysis (Section 5), as the horizon-dependent performance interacts with several of the cost-weight calibration choices. Notably, the FC-overshoot phenomenon documented in Section 4.1.6 is horizon-specific, and resolving it through the $\alpha_6$ penalty may alter the relative ranking of horizons.

## 4.6 Warm-Starting and Computational Acceleration

To reduce solve times across consecutive daily solves, the NLP is warm-started by shifting the previous day's optimal trajectory forward by one step. The control sequence $\{u^*(k), u^*(k+1), \ldots, u^*(k + H_p - 1)\}$ from the previous solve is shifted to provide an initial guess $\{u^*(k+1), u^*(k+2), \ldots, u^*(k + H_p - 1), u^*(k + H_p - 1)\}$ for the current solve, where the final element is duplicated. The corresponding state trajectory is similarly shifted. This provides IPOPT with a feasible starting point close to the optimal solution, typically reducing the iteration count from 200+ (cold start) to 30–80 (warm start).

---

# 5. Weight Sensitivity Analysis

## 5.1 Motivation: Empirical Observations Driving the Sweep

The default operating point $\boldsymbol{\alpha}_{default} = \{1.0, 0.01, 0.1, 0.5, 0.005, 0.0\}$ established in Section 4.2 uses approximate values derived from physical reasoning and order-of-magnitude economic anchoring. To verify and refine these choices, three preliminary observations from the 27 nominal MPC runs (3 scenarios × 3 budgets × 3 horizons at the default operating point) were made.

**Observation 1 — Field-capacity overshoot at long horizons.** Under the dry-year scenario at full budget, the $H_p = 14$ controller pushes soil moisture above field capacity for 21.6 agent-days per agent on average over the 93-day season. The same scenario at $H_p = 8$ exhibits the phenomenon for only 1.9 days, and at $H_p = 3$ for effectively zero days. The phenomenon (illustrated in Figure 4.1) is not a forecast-uncertainty artifact — these are perfect-forecast experiments — but rather a structural feature of the cost function: with $\alpha_6 = 0$, no term directly penalizes $x_1 > FC$, and the implicit waterlog stress in the biomass increment is too weak to override the optimizer's incentive to maximize transpiration over the look-ahead window. This motivates Sweep Group D ($\alpha_6$ tuning) and Sweep Group E ($\alpha_6$ validation at $H_p = 14$).

**Observation 2 — Horizon inversion in the wet-year scenario.** At wet/100%, the $H_p = 3$ controller produces 879.9 g/m² terminal biomass while $H_p = 14$ produces only 807.7 g/m². This is theoretically impossible for a correctly converged optimizer: a longer horizon can always replicate a shorter-horizon policy, so its objective value cannot be worse. To diagnose this, the actual MPC cost function $J$ was evaluated *ex post* on each completed run trajectory using the default weight vector. The result is shown in Table 5.1.

**Table 5.1: Ex-post evaluation of the MPC cost function $J$ on completed nominal trajectories at full budget. Lower $J$ is better. The five-term decomposition is: $J = J_{biomass} + J_{water} + J_{drought} + J_{ponding} + J_{\Delta u}$ (the last contribution is included in the totals but not shown separately).**

| Scenario | $H_p$ | $J_{biomass}$ | $J_{water}$ | $J_{drought}$ | $J_{ponding}$ | $J_{total}$ |
|---|---:|---:|---:|---:|---:|---:|
| Dry | 3 | $-1.092$ | $+0.929$ | $+0.085$ | $0.000$ | $-0.079$ |
| Dry | 8 | $-1.111$ | $+0.952$ | $+0.008$ | $0.000$ | $-0.151$ |
| Dry | 14 | $-1.124$ | $+0.957$ | $+0.011$ | $0.000$ | **$-0.156$** |
| Moderate | 3 | $-1.052$ | $+0.923$ | $+0.097$ | $0.000$ | $-0.032$ |
| Moderate | 8 | $-1.066$ | $+0.952$ | $+0.019$ | $0.000$ | $-0.095$ |
| Moderate | 14 | $-1.085$ | $+0.953$ | $+0.019$ | $0.000$ | **$-0.114$** |
| Wet | 3 | $-0.978$ | $+0.624$ | $+0.043$ | $0.000$ | **$-0.311$** |
| Wet | 8 | $-0.953$ | $+0.733$ | $+0.003$ | $+0.022$ | $-0.196$ |
| Wet | 14 | $-0.898$ | $+0.892$ | $+0.009$ | $+0.271$ | $+0.275$ |

In the dry and moderate scenarios, $J$ improves monotonically with $H_p$, as expected. In the wet scenario, however, $J$ is *worst* at $H_p = 14$ ($+0.275$) and *best* at $H_p = 3$ ($-0.311$). Since a 14-day horizon controller can always reproduce the 3-day controller's trajectory within its own search space, $J(H_p=14) > J(H_p=3)$ implies that IPOPT is converging to a suboptimal local minimum at the longer horizon. This is consistent with the FC-overshoot phenomenon: under wet conditions, the saturated-soil regime produces a near-flat cost gradient, allowing IPOPT to terminate at the acceptable tolerance before escaping the local basin. Resolving this through the $\alpha_6$ penalty changes the cost surface topology, which is expected to mitigate the local-minimum trap. This motivates the entire sweep program.

**Observation 3 — Chronic waterlogging at wet/$H_p=14$.** The $H_p = 14$ controller in the wet/100% scenario maintains $x_1 \in [175, 205]$ mm (near saturation) for over 25 consecutive days during the latter half of the season, producing 61.1 waterlog-days per agent and a peak ponding depth of 40.8 mm. This is a direct manifestation of the local minimum identified in Observation 2: the optimizer parks soil moisture in the saturation zone because the marginal disincentive (via $h_6$ in the biomass increment) is too weak to push back against the water-application incentive. This motivates Sweep Group C ($\alpha_4$ tuning at wet/$H_p=14$).

**Synthesis.** The three observations have a common root cause: the default cost function provides insufficient direct disincentive against parking soil moisture above field capacity, which manifests as different surface phenomena depending on the climatic context (silent over-irrigation in dry scenarios; chronic waterlogging in wet scenarios). The sensitivity sweep characterizes the controller's response to corrective tuning along three axes: the economic axis ($\alpha_2$ across the four Iranian price tiers), the agronomic axis ($\alpha_3$ for drought regularization, $\alpha_4$ for ponding), and the new physical-bound axis ($\alpha_6$ for the FC-overshoot penalty).

## 5.2 Sweep Strategy: One-at-a-Time Over Justified Ranges

Given the computational cost of MPC runs (1–8 hours per simulation depending on horizon and scenario), an exhaustive multi-dimensional sweep over the full $\boldsymbol{\alpha}$ space is infeasible. Instead, a one-at-a-time (OAT) sensitivity analysis is performed, following the methodology of Saltelli et al. [14]: each weight is varied independently over a justified physical/economic range while the other weights are held at their default values. This identifies the Pareto frontier in each dimension and reveals the relative importance of each weight, while keeping the total compute under 20 hours.

The five sweep groups are summarized in Table 5.2 and detailed in the subsequent subsections.

**Table 5.2: Weight sensitivity sweep groups.**

| Group | Weight swept | Scenario | Budget | $H_p$ | # Configs | Approx. compute |
|---|---|---|---|---:|---:|---:|
| A | $\alpha_2$ (water price tiers) | Dry | 100% | 8 | 4 | 4 h |
| B | $\alpha_3$ (drought regularizer) | Dry | 100% | 8 | 2 | 2 h |
| C | $\alpha_4$ (ponding penalty) | Wet | 100% | 14 | 2 | 2 h |
| D | $\alpha_6$ (FC overshoot) | Dry | 100% | 8 | 3 | 3 h |
| E | $\alpha_6$ validation at long horizon | Dry | 100% | 14 | 1 | 7.6 h |

## 5.3 Sweep Group A — Water-Price Tier Sweep ($\alpha_2$)

Sweep Group A explores the four economically-grounded $\alpha_2$ values from Table 4.1, anchored to real Iranian water-pricing tiers. This sweep characterizes how the optimal irrigation policy changes as the economic environment shifts from heavily subsidized agricultural pricing to industrial-scarcity pricing. The expected qualitative behavior is that increasing $\alpha_2$ should reduce total water use and shift biomass downward along the yield/water Pareto frontier, with the steepness of the frontier indicating the controller's price elasticity.

The sweep configurations are:
- $\alpha_2 = 0.0004$ — Subsidized agricultural (real Iranian price)
- $\alpha_2 = 0.016$ — Domestic base tariff
- $\alpha_2 = 0.044$ — Domestic high-consumption tariff (coefficient $d \times 2.8$)
- $\alpha_2 = 0.265$ — Industrial pricing

All other weights remain at the default values, including $\alpha_6 = 0$.

## 5.4 Sweep Group B — Drought Regularizer Sweep ($\alpha_3$)

Sweep Group B varies $\alpha_3$ by $\pm 1$ order of magnitude around the default value of 0.1, following standard regularization sensitivity practice [14]. The two configurations are:
- $\alpha_3 = 0.03$ — Reduced drought aversion
- $\alpha_3 = 0.3$ — Heightened drought aversion

The expected behavior is that increasing $\alpha_3$ shifts the policy toward earlier and more aggressive irrigation, reducing drought-day count at the cost of higher water use.

## 5.5 Sweep Group C — Ponding Weight Sweep at Wet/Hp=14 ($\alpha_4$)

Sweep Group C addresses Observation 3 directly: the chronic 61-day waterlogging at wet/100%/$H_p = 14$. The sweep is conducted at the specific scenario where the pathology manifests, rather than at the default sweep scenario, to test whether $\alpha_4$ tuning can resolve the local-minimum trap. The configurations are:
- $\alpha_4 = 2.0$ — 4× the default
- $\alpha_4 = 5.0$ — 10× the default

The default $\alpha_4 = 0.5$ run already exists from the nominal grid and serves as the baseline. The expected behavior, if the local minimum is breakable through cost-surface reshaping, is a sharp reduction in waterlog-day count and ponding peak as $\alpha_4$ increases. The references for the choice of upper bound are [15], which establishes that even flood-tolerant species incur rapid yield loss when root-zone hypoxia persists beyond 1–2 days.

## 5.6 Sweep Group D — FC-Overshoot Penalty Sweep ($\alpha_6$)

Sweep Group D introduces the $\alpha_6$ term and tests three values spanning $\sim$1.5 orders of magnitude:
- $\alpha_6 = 0.1$ — Mild FC-overshoot disincentive
- $\alpha_6 = 0.5$ — Same magnitude as the ponding weight
- $\alpha_6 = 2.0$ — 4× the ponding weight

The sweep is conducted at dry/100%/$H_p = 8$ to economize compute. The expected behavior is a reduction in waterlog-day count from the nominal 1.9 days toward zero, with a possible secondary effect on water-use efficiency if the controller becomes more conservative in its irrigation timing.

## 5.7 Sweep Group E — $\alpha_6$ Validation at Hp=14

Sweep Group E is a single-configuration validation run: $\alpha_6 = 2.0$ applied at dry/100%/$H_p = 14$, the scenario where the FC-overshoot pathology is most severe (21.6 waterlog-days at $\alpha_6 = 0$). This run definitively tests whether the new term resolves the pathology at the horizon length where it manifests. If successful, it also indirectly addresses Observation 2 (the wet-scenario local minimum), since the cost-surface flattening produced by FC-overshoot is the same mechanism that traps the wet-scenario optimizer.

## 5.8 Sweep Results

*[The complete sweep results will be populated upon completion of `exp_weight_sensitivity.py`. The expected outputs are summarized below as placeholder structures.]*

**Table 5.3: Sweep results summary across all twelve sweep configurations.** [TO BE FILLED AFTER SWEEP COMPLETION]

| Group | Config | Yield (kg/ha) | Water used (mm) | WUE (kg/ha/mm) | Drought-days | Waterlog-days | $J_{total}$ |
|---|---|---:|---:|---:|---:|---:|---:|
| A | $\alpha_2 = 0.0004$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| A | $\alpha_2 = 0.016$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| A | $\alpha_2 = 0.044$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| A | $\alpha_2 = 0.265$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| B | $\alpha_3 = 0.03$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| B | $\alpha_3 = 0.3$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| C | $\alpha_4 = 2.0$ (wet/Hp=14) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| C | $\alpha_4 = 5.0$ (wet/Hp=14) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| D | $\alpha_6 = 0.1$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| D | $\alpha_6 = 0.5$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| D | $\alpha_6 = 2.0$ | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| E | $\alpha_6 = 2.0$ (dry/Hp=14) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Figure 5.1**: *[PLACEHOLDER] Pareto frontier of yield versus water consumption across the four $\alpha_2$ price tiers (Sweep Group A). The frontier is expected to shift downward and leftward as water becomes economically more expensive, with the curvature indicating the controller's price elasticity.*

**Figure 5.2**: *[PLACEHOLDER] Drought-day and waterlog-day counts across the $\alpha_3$ sweep (Sweep Group B), showing the regularizer's effect on the dry-side stress profile.*

**Figure 5.3**: *[PLACEHOLDER] Time-series comparison of $x_5$ (surface ponding) and $x_1$ (root-zone moisture) at wet/100%/$H_p = 14$ across the three $\alpha_4$ values (0.5 default, 2.0, 5.0). If the local minimum is breakable, the trajectories at higher $\alpha_4$ should show the chronic 25-day saturation zone collapsing toward shorter, sharper ponding events.*

**Figure 5.4**: *[PLACEHOLDER] Field-mean $x_1$ trajectories at dry/100%/$H_p = 8$ across the four $\alpha_6$ values (0 default, 0.1, 0.5, 2.0). The expected effect is a progressive flattening of the trajectory toward $x_1 \leq FC$ as $\alpha_6$ increases.*

**Figure 5.5**: *[PLACEHOLDER] Validation comparison at dry/100%/$H_p = 14$: $x_1$ trajectory at $\alpha_6 = 0$ (the original 21.6-waterlog-day case from Figure 4.1) versus $\alpha_6 = 2.0$. If $\alpha_6 = 2.0$ resolves the FC-overshoot pathology at this horizon, the latter trajectory should remain at or below FC throughout the season.*

## 5.9 Recommended Operating Point

*[To be determined after sweep completion. The recommended $\boldsymbol{\alpha}^*$ vector will be the configuration that simultaneously satisfies:*
1. *Resolves the FC-overshoot pathology at $H_p = 14$ (waterlog-days drops to or near zero)*
2. *Reduces the wet-scenario local-minimum trap (ex-post $J(H_p = 14)$ no longer exceeds $J(H_p = 3)$)*
3. *Preserves a high yield-per-water ratio across all three climatic scenarios*
4. *Reflects a defensible economic interpretation of the water-pricing environment*

*The choice of $H_p$ for the main results chapter will also be made at this stage, after the sweep clarifies the horizon-cost-function interaction.]*

**Recommended operating point:** [TBD after sweep]

---

# 6. Soft Actor-Critic (SAC) Architecture

While Model Predictive Control provides a mathematically rigorous, constraint-aware optimization framework, the requirement to solve a massive Non-Linear Program at each timestep poses significant latency challenges for low-power edge computing devices deployed in agricultural Cyber-Physical Systems. To investigate a low-latency alternative, a Reinforcement Learning agent will be trained to approximate the optimal control policy. The problem is formulated as a Markov Decision Process (MDP) and solved using the Soft Actor-Critic (SAC) algorithm, an off-policy actor-critic method based on the maximum entropy reinforcement learning framework [6, 7].

## 6.1 Markov Decision Process Formulation

The irrigation control problem is defined by the MDP tuple $\langle S, A, P, R, \gamma \rangle$.

### 6.1.1 Observation Space ($S$)

To provide the RL agent with sufficient information to manage both spatial topographical heterogeneity and the global seasonal water budget, the state observation is defined as a flat, continuous vector of 660 dimensions. The vector concatenates 10 global field features with 5 local features for each of the $N = 130$ agents:

**Global Features (10 dimensions):** Normalized remaining budget ($W_{remaining}/W_{total}$), normalized current day ($k/K$), daily rainfall $P(k)$, reference evapotranspiration $\text{ET}_0(k)$, field-mean soil moisture ($\bar{x}_1/FC$), field-moisture standard deviation, field-mean biomass ($\bar{x}_4/x_{4,ref}$), fraction of agents in drought ($x_1 < ST$), fraction of agents in waterlogging ($x_1 > FC$), and the cumulative 7-day rainfall forecast ($\sum_{j=1}^7 P(k+j)$).

**Local Features (650 dimensions):** For each agent $n$, the network observes its normalized soil moisture ($x_1^n/FC$), normalized thermal time ($x_2^n/\theta_{18}$), normalized biomass ($x_4^n/x_{4,ref}$), normalized surface ponding ($x_5^n/x_{5,ref}$), and its static normalized topographical elevation ($\gamma^n$). The elevation feature enables the policy to learn spatially differentiated irrigation strategies — for example, reducing irrigation for high-elevation agents whose runoff will cascade to lower neighbors.

### 6.1.2 Action Space ($A$)

The continuous action space mirrors the MPC actuator constraints. The SAC actor outputs a bounded vector $\mathbf{a}_k \in [0, 1]^N$ via a tanh squashing function. During the environment step, this output is linearly scaled by the physical actuator cap ($u_{\max} = 12.0$ mm/day) such that $u_k^n = a_k^n \cdot u_{\max}$, matching the exact decision space of the MPC baseline.

### 6.1.3 Reward Formulation ($R$)

To ensure a fair and direct comparison between the learned policy and the computed MPC policy, the dense per-step reward $r_k$ is engineered as the exact mathematical negation of the MPC path cost function (Section 4.1):

$$r_k = -(J_{water}(k) + J_{drought}(k) + J_{ponding}(k) + J_{overFC}(k)) \tag{23}$$

The $\alpha_6$-weighted FC-overshoot penalty is included in the SAC reward to maintain the cost-function equivalence after the recommended operating point from the sensitivity sweep is adopted.

At the terminal timestep $K = 93$, a sparse positive bonus is awarded based on the normalized terminal biomass:

$$r_K = +5 \cdot \frac{\bar{x}_4(K)}{x_{4,ref}} \tag{24}$$

The terminal bonus weight of $+5$ was selected to ensure that the cumulative terminal reward is of the same order of magnitude as the sum of dense path rewards over 93 steps, preventing the agent from myopically minimizing path costs at the expense of long-term biomass accumulation. This reward structure forces the agent to learn the exact same agronomic and economic trade-offs as the predictive optimizer.

## 6.2 Gymnasium Environment Wrapper

The ABM is wrapped as a standard Gymnasium (formerly OpenAI Gym) environment implementing the `reset()`, `step()`, and `render()` interface. The `reset()` method initializes the 130-agent field to the standard initial conditions ($x_1 = 140$ mm, $x_5 = 0$, $x_2 = 210$, $x_4 = 60$, $x_3 = 0$), samples a climate scenario, and resets the budget counter. The `step(action)` method: (i) clips the raw action to satisfy the actuator bounds and remaining budget, (ii) applies the irrigation to the ABM, (iii) advances the ABM by one day including cascade routing, (iv) constructs the 660-dimensional observation vector, (v) computes the dense reward, and (vi) returns the standard Gymnasium tuple (obs, reward, terminated, truncated, info). The episode terminates naturally at $k = 93$ or is truncated early if the budget violation penalty is triggered.

## 6.3 Centralized Training with Decentralized Execution

A naive multi-agent implementation — where a single monolithic policy network directly outputs 130 independent actions from the 660-dimensional state — would fail to exploit the spatial symmetry of the field and require an intractable number of parameters. To overcome this, the architecture employs Centralized Training with Decentralized Execution (CTDE):

**The Decentralized Actor:** The actor network operates on the principle of parameter sharing. A single Multi-Layer Perceptron (MLP) with hidden layers of [256, 256] and ReLU activations is instantiated. During the forward pass, this network is applied identically across all 130 agents. For agent $n$, the actor concatenates the 10 global features with the 5 local features of agent $n$ (totaling 15 inputs) to output a parameterized Gaussian distribution (mean $\mu^n$ and log standard deviation $\log \sigma^n$) for that specific agent's irrigation action. This forces the network to learn a generalized policy: *how to irrigate any agent given its specific elevation and moisture state, conditioned on the global budget status*.

**The Centralized Critic:** Unlike the actor, the twin critic networks evaluate the joint action value $Q(\mathbf{S}_k, \mathbf{A}_k)$. Each critic takes the full 660-dimensional state and the 130-dimensional joint action vector as input (790 dimensions total), passing through hidden layers of [512, 512, 256] with ReLU activations. By observing the entire state-action space during offline training, the centralized critic learns the shadow price of the global water budget and the cascading runoff effects across the topographical graph, guiding the decentralized actor toward a globally optimal cooperative policy. The twin-critic architecture (two independent $Q$-networks with minimum taken for the target) follows the standard SAC formulation to mitigate overestimation bias.

## 6.4 Budget Constraint Handling

While IPOPT natively handles the global water budget $\sum u \leq W_{total}$ via a hard linear inequality constraint in the NLP, standard RL algorithms lack native constrained-optimization mechanisms. The SAC environment enforces the budget via a three-tier soft penalty system:

**Tier 1 — Action Clipping (Hard Safety Net):** As an absolute guarantee, the environment clips the daily action array so that the total requested volume across all agents never mathematically exceeds the remaining budget. Specifically, if $\sum_n u_k^n > W_{remaining}(k)$, all actions are proportionally scaled down: $u_k^n \gets u_k^n \cdot W_{remaining}(k) / \sum_n u_k^n$. This ensures that the budget constraint is never violated, regardless of the agent's policy output.

**Tier 2 — Burn-Rate Shaping:** To prevent the agent from dumping its entire budget in the first few weeks (a common failure mode during early training), a negative reward shaped by the normalized season progression is applied if the daily expenditure rate significantly outpaces the seasonal timeline. Let $f_{spent} = 1 - W_{remaining}/W_{total}$ be the fraction of budget spent and $f_{time} = k/K$ be the fraction of the season elapsed. If $f_{spent} > f_{time} + 0.15$ (the agent is spending more than 15 percentage points ahead of schedule), a penalty of $r_{burn} = -2.0 \cdot (f_{spent} - f_{time})$ is applied.

**Tier 3 — Early Termination:** If the agent triggers the hard clip limit by attempting to exceed the total budget, a severe penalty ($r = -100$) is applied, and the episode is immediately truncated. This catastrophic penalty signal provides a strong gradient during early training to discourage budget-violating policies.

## 6.5 Planned Training Protocol and Hyperparameters

The SAC agent will be trained using Stable-Baselines3 (SB3) [8], a well-maintained PyTorch implementation of standard RL algorithms. Training is planned on Kaggle's free GPU tier, which provides NVIDIA Tesla P100 GPUs with a 9-hour session limit and 30 GPU-hours per week. The complete hyperparameter configuration is:

**Table 6.1: Planned SAC training hyperparameters.**

| Hyperparameter | Value | Source/Justification |
|---|---|---|
| Learning rate (actor & critic) | $3 \times 10^{-4}$ | Default SAC |
| Replay buffer size | $10^6$ transitions | Standard for continuous control |
| Batch size | 256 | Standard for continuous control |
| Discount factor ($\gamma$) | 0.99 | Near-horizon: 93-step episodes |
| Soft update coefficient ($\tau$) | 0.005 | Default SAC |
| Entropy coefficient ($\alpha$) | Auto-tuned | Constrained dual gradient descent |
| Target entropy | $-\dim(A) = -130$ | Default: $-|A|$ |
| Total training steps | $5 \times 10^6$ | Kaggle budget permitting |
| Random seeds | 5 | Statistical significance |

The entropy coefficient $\alpha$ is automatically tuned via the constrained formulation introduced by Haarnoja et al. [7], where $\alpha$ is treated as a Lagrange multiplier on the minimum entropy constraint $\mathcal{H}[\pi] \geq \mathcal{H}_{target}$. The target entropy is set to $-130$ (one per action dimension), following the standard heuristic of $-\dim(A)$.

Training will be conducted for 5 independent random seeds on the dry/100% scenario at the recommended operating point determined by the sensitivity sweep (Section 5.9). Each seed will run for up to $5 \times 10^6$ environment steps (approximately 53,763 full episodes of 93 steps each). Once trained, the best-performing seed (selected by validation reward) will be evaluated across all 9 scenario × budget combinations to assess generalization.

---

# 7. Uncertainty Modeling and Unified Evaluation Framework

To guarantee the validity of the comparative analysis, the experimental framework must ensure that all controllers are subjected to identical environmental conditions, constraints, and standardized evaluation metrics. Furthermore, to evaluate the real-world robustness of the predictive models (MPC and SAC), a formal mathematical model for meteorological forecast uncertainty must be established.

## 7.1 Meteorological Forecast Uncertainty

While the baseline controllers ($C_1$, $C_2$) and the "No-Forecast" SAC ($C_5$) operate without future weather information, the predictive controllers — the Perfect MPC ($C_3$), Noisy MPC ($C_4$), and Forecast SAC ($C_6$) — rely on a prediction horizon. To benchmark the theoretical upper bound of the predictive architecture against a realistic deployment scenario, two distinct forecast modes are formalized.

**1. Perfect Forecast Mode.** In the perfect information scenario, the forecast vector provided to the controller exactly matches the true future weather realization generated by the environment simulation. For any meteorological variable $w$ at a future day $k + j$ predicted from the current day $k$:

$$\hat{w}(k+j \mid k) = w(k+j) \quad \forall j \in \{0, \ldots, H_p - 1\} \tag{25}$$

This mode establishes the absolute performance ceiling for the predictive controllers.

**2. Noisy Forecast Mode.** To simulate the decaying accuracy of real-world meteorological forecasts over time, a stochastically degrading noise model is applied to the rainfall and reference evapotranspiration ($\text{ET}_0$) vectors. Temperature forecasts are left unperturbed, reflecting the high operational accuracy of thermal forecasting (RMSE $< 1\,°$C at 7-day lead time). The multiplicative noise injection is defined as:

$$\hat{w}(k+j \mid k) = w(k+j) \cdot (1 + \epsilon_j) \tag{26}$$

where the error term $\epsilon_j$ is drawn from a Gaussian distribution with zero mean and a variance that grows proportionally to the square root of the look-ahead horizon $j$:

$$\epsilon_j \sim \mathcal{N}(0, \sigma_j^2), \quad \sigma_j = 0.15 \cdot \sqrt{j} \tag{27}$$

This formulation ensures that tomorrow's forecast ($j = 1$) retains high fidelity ($\sigma_1 = 0.15$, corresponding to $\pm 15\%$ typical error), while forecasts extending toward the end of a 14-day horizon exhibit severe uncertainty ($\sigma_{14} = 0.56$, corresponding to $\pm 56\%$ typical error), rigorously testing the stability of the receding-horizon control loop. The square-root decay rate is consistent with the empirical skill degradation observed in operational NWP (Numerical Weather Prediction) models.

## 7.2 Controller Summary Table

To provide a complete reference for the experimental evaluation, all six controller variants across the four architectures are formally defined:

**Table 7.1: Complete controller taxonomy.**

| ID | Controller | Type | Forecast | Closed-Loop |
|---|---|---|---|---|
| $C_1$ | No-Irrigation | Baseline (open-loop) | None | No |
| $C_2$ | Fixed-Schedule Heuristic | Open-loop | None | No |
| $C_3$ | MPC (Perfect Forecast) | Model-based | Perfect | Yes |
| $C_4$ | MPC (Noisy Forecast) | Model-based | Noisy | Yes |
| $C_5$ | SAC (No Forecast) | Learning-based | None | Yes |
| $C_6$ | SAC (7-day Forecast) | Learning-based | 7-day sum | Yes |

## 7.3 Unified Software Evaluation Architecture

To eliminate implementation bias during the comparative evaluation, the control loop is strictly abstracted into three separated software layers:

**The Environment Layer** maintains the true physical state of the field ($\mathbf{X}_k$), executes the topographical cascade routing using the high-fidelity ABM, and calculates true biological growth. This layer is identical for all six controllers and is never modified or approximated during evaluation.

**The Controller Layer** provides a standardized interface. All six controllers ($C_1$ through $C_6$) inherit from an identical abstract base class, requiring them to implement a single method: `get_action(state, forecast) -> action_vector`. This ensures that the heavy IPOPT NLP solver and the trivial zero-action baseline interact with the environment through the exact same API.

**The Runner Layer** is a centralized simulation orchestrator that manages the day-by-day transition loop. At each step $k$, it: (i) queries the instantiated controller for an action $\mathbf{U}_k$, (ii) clips the action to satisfy the remaining budget, (iii) applies the action to the environment, (iv) deducts the applied volume from the global seasonal budget counter, (v) advances the climate data to the next step, and (vi) logs the complete state, action, and diagnostic information to a long-format Parquet file with a JSON metadata sidecar.

This architecture guarantees that variations in final agronomic metrics (yield, water consumption, stress days) are strictly attributable to the intrinsic quality of the underlying control policy, not to implementation artifacts.

## 7.4 Experimental Grid

The complete experimental evaluation spans the following factorial design:

**Table 7.2: Experimental grid for MPC evaluation across the nominal operating point.**

| | 100% Budget | 85% Budget | 70% Budget |
|---|:---:|:---:|:---:|
| Dry scenario (2022) | ✓ | ✓ | ✓ |
| Moderate scenario (2020) | ✓ | ✓ | ✓ |
| Wet scenario (2024) | ✓ | ✓ | ✓ |

Each cell is evaluated at $H_p \in \{3, 8, 14\}$, yielding 27 nominal MPC runs (already complete). The sensitivity sweep adds 12 additional runs (Sweep Groups A–E), bringing the total MPC simulations to 39. The final operating point ($\boldsymbol{\alpha}^*, H_p^*$) determined by Section 5.9 will be re-applied across the full 9-cell grid for the noisy-forecast MPC ($C_4$) and the SAC controllers ($C_5$, $C_6$). With the fixed-schedule baseline ($C_2$) already complete for all 9 cells across 3 repetitions, and the no-irrigation baseline ($C_1$) complete for 3 repetitions, the full evaluation comprises approximately 90 simulation runs across all controllers, scenarios, budgets, horizons, and sweep configurations.

---

# 8. Conclusion

This chapter has established the complete mathematical and architectural framework for the irrigation controllers that will be comparatively evaluated in the subsequent results chapter. The principal design decisions and their theoretical implications are summarized as follows.

The constrained optimal control problem was formalized with a six-term multi-objective cost function, where each term is normalized to $\mathcal{O}(1)$ and anchored to real agronomic and economic quantities. The MPC controller was implemented as a large-scale NLP with 3,120 decision variables and 2,081 constraints (at $H_p = 8$), solved by the IPOPT interior-point method using smooth $C^2$-continuous approximations ($\varepsilon = 0.01$) to guarantee convergence. The fractional DEM-padded routing boundary condition resolves the bathtub effect and enables physically realistic off-farm drainage simulation within the symbolic graph.

A novel sixth cost term, $J_{overFC}$, was introduced to penalize root-zone moisture excursions above field capacity. This term was motivated by an empirical observation: at long prediction horizons under conditions of full water availability, the optimizer was found to deliberately park soil moisture in the saturation zone, exploiting a weakness in the original cost function topology. The associated weight $\alpha_6$ is calibrated through a dedicated sensitivity sweep.

The water-cost weight $\alpha_2$ was anchored to the four-tier Iranian water-pricing system, spanning from heavily subsidized agricultural rates (175 toman/m³) to industrial rates (115,000 toman/m³). This economic anchoring transforms the abstract weight calibration into a policy-relevant analysis: the same controller architecture, evaluated under different price tiers, characterizes the agricultural water economy at multiple realistic scenarios.

The SAC reinforcement learning agent was designed with a CTDE architecture that exploits spatial symmetry through parameter sharing (15-dimensional per-agent input to a shared $256 \times 256$ actor), while the centralized twin-critic (790-dimensional input through $512 \times 512 \times 256$ layers) captures the global budget dynamics and inter-agent runoff coupling. The budget constraint is enforced through a three-tier soft penalty system that provides hard safety (action clipping), temporal shaping (burn-rate penalty), and catastrophic deterrence (early termination). The SAC reward is constructed as the exact negation of the MPC path cost (including the $\alpha_6$ term), enabling rigorous policy-equivalence comparison.

The theoretical trade-off between these approaches is clear. MPC provides mathematical guarantees on constraint satisfaction and can exploit exact model knowledge and weather forecasts to compute high-quality irrigation profiles, but at a computational cost ranging from 3 seconds per daily decision at $H_p = 3$ to 105 seconds at $H_p = 14$, with worst-case latencies exceeding 4 hours in poorly-conditioned cases. SAC, once trained, executes a simple feed-forward neural network inference in sub-millisecond time, enabling real-time deployment on edge computing hardware, but lacks native constraint guarantees and may exhibit suboptimal behavior in out-of-distribution weather scenarios. The recommended operating point and the choice of prediction horizon are deferred to the post-sweep analysis (Section 5.9), and the subsequent experimental chapter will quantify the full set of trade-offs through systematic comparison across the 9-cell scenario × budget grid.

---

# References

[1] J. Lopez-Jimenez, N. Quijano, L. Dewasme, and A. Vande Wouwer, "Agent-based model predictive control of soil-crop irrigation with topographical information," *Control Engineering Practice*, vol. 150, p. 106012, 2024.

[2] J. Lopez-Jimenez, L. Dewasme, A. Vande Wouwer, and N. Quijano, "Dynamic modeling of crop-soil systems to design monitoring and automatic irrigation processes: A review with worked examples," *Water*, vol. 14, no. 15, p. 2404, 2022.

[3] R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, "Crop evapotranspiration: Guidelines for computing crop water requirements," FAO Irrigation and Drainage Paper No. 56, 1998.

[4] A. Wächter and L. T. Biegler, "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming," *Mathematical Programming*, vol. 106, no. 1, pp. 25–57, 2006.

[5] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, "CasADi: A software framework for nonlinear optimization and optimal control," *Mathematical Programming Computation*, vol. 11, no. 1, pp. 1–36, 2019.

[6] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in *Proc. ICML*, vol. 80, pp. 1861–1870, 2018.

[7] T. Haarnoja, A. Zhou, K. Hartikainen, et al., "Soft actor-critic algorithms and applications," *arXiv preprint arXiv:1812.05905*, 2019.

[8] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," *JMLR*, vol. 22, no. 268, pp. 1–8, 2021.

[9] H. Nouri et al., "Water management dilemma in the agricultural sector of Iran: A review focusing on water governance," *Agricultural Water Management*, vol. 278, p. 108162, 2023.

[10] M. B. Mesgaran and P. Azadi, "A national adaptation plan for water scarcity in Iran," Stanford Iran 2040 Project, Working Paper 6, 2018.

[11] N. Jalali et al., "Water requirement of Hashemi rice," *Water Res. Agric.*, 2021.

[12] W. J. Rawls, D. Brakensiek, and K. E. Saxton, "Estimation of soil water properties," *Transactions of the ASAE*, vol. 25, no. 5, pp. 1316–1320, 1982.

[13] P. Polcz et al., "Smart epidemic control: A hybrid model blending ODEs and agent-based simulations for optimal, real-world intervention planning," *PLoS Computational Biology*, vol. 21, no. 5, p. e1013028, 2025.

[14] A. Saltelli, S. Tarantola, F. Campolongo, and M. Ratto, *Sensitivity Analysis in Practice: A Guide to Assessing Scientific Models*. Wiley, 2004.

[15] T. L. Setter et al., "Review of prospects for germplasm improvement for waterlogging tolerance in wheat, barley and oats," *Field Crops Research*, vol. 51, no. 1-2, pp. 85–104, 1997.

[16] Iran Water Resources Management Co., "Domestic, agricultural, and industrial water tariff schedules for fiscal year 1402-1403 (2023-2024)," Tehran, Iran, 2023.
