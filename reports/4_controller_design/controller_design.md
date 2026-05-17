## **Ministry of Science and Higher Education of the Russian Federation** **Federal State Autonomous Educational Institution of Higher** **Education** **"National Research University ITMO"** **«Faculty of Control Systems and Robotics»** Field of study (specialty) 15.04.06 – Robotics and artificial intelligence **REPORT** on Research Internship Assignment topic: Controller Design and Multi-Agent Evaluation Framework for Constrained Irrigation Optimization Student: Tara Torbati, group R4237c Supervisor: Peregudin A.A. Saint Petersburg 2026


**Abstract**


This report presents the complete mathematical formulation, architectural design, and

evaluation framework for the four irrigation controllers deployed on a 130-agent topograph
ical agricultural field in Gilan Province, Iran. The controllers span a hierarchy of increas
ing sophistication: a rainfed no-irrigation baseline, a fixed-schedule heuristic reflecting

traditional Gilan practices, a Model Predictive Controller (MPC) formulated as a large
scale Non-Linear Program (NLP) solved via the IPOPT interior-point method within the

CasADi symbolic framework, and a Soft Actor-Critic (SAC) reinforcement learning agent

trained under a Centralized Training with Decentralized Execution (CTDE) paradigm.

Each controller addresses the same constrained finite-horizon optimal control problem:

maximizing terminal Hashemi rice biomass over a 93-day cultivation season while strictly

adhering to a seasonal water budget of 484 mm at full allocation, with deficit scenarios

at 85% and 70%. The report formalizes the multi-objective cost function with five nor
malized penalty terms, details the Control-Oriented Model (COM) including the smooth

approximation techniques required for solver convergence, specifies the NLP structure

(3,120 decision variables, 2,081 constraints at prediction horizon _Hp_ = 8), documents

the IPOPT solver configuration and termination criteria, and presents the Markov Deci
sion Process formulation for the SAC agent. A unified evaluation architecture ensuring

identical environmental conditions across all controllers is established, together with a for
mal meteorological forecast uncertainty model. This chapter serves as the methodological

bridge between the virtual plant described in the preceding system description and the

comparative experimental results that follow.


1


# **Contents**

**1** **Introduction** **4**


**2** **Problem** **Formulation** **and** **System** **Constraints** **4**
2.1 State Space Representation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 Action Space and Actuator Constraints . . . . . . . . . . . . . . . . . . . . . . . 5
2.3 Global Water Budget Constraint . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.4 System Dynamics and Optimal Control Objective . . . . . . . . . . . . . . . . . 7


**3** **Baseline** **and** **Heuristic** **Controllers** **7**
3.1 The No-Irrigation Baseline ( _C_ 1) . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.2 The Fixed-Schedule Heuristic ( _C_ 2) . . . . . . . . . . . . . . . . . . . . . . . . . 8


**4** **Model** **Predictive** **Control** **(MPC)** **Architecture** **8**
4.1 The Multi-Objective Cost Function . . . . . . . . . . . . . . . . . . . . . . . . . 8
4.1.1 Terminal Biomass (Mayer Cost) . . . . . . . . . . . . . . . . . . . . . . . 9
4.1.2 Water Consumption Penalty . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.1.3 Drought Stress Regularization . . . . . . . . . . . . . . . . . . . . . . . . 9
4.1.4 Ponding Penalty . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4.1.5 Actuator Smoothing (Control-Rate Regularization) . . . . . . . . . . . . 10
4.2 Weight Calibration and Economic Anchoring . . . . . . . . . . . . . . . . . . . . 10
4.3 Control-Oriented Model (COM) and Symbolic Formulation . . . . . . . . . . . . 11
4.3.1 Dimensionality Reduction of the State Space . . . . . . . . . . . . . . . . 11
4.3.2 Deterministic Caching of Biological Nonlinearities . . . . . . . . . . . . . 12
4.3.3 Static Embedding of Topographical Cascade Routing . . . . . . . . . . . 12
4.3.4 Smooth Approximations for Solver Convergence . . . . . . . . . . . . . . 12
4.4 NLP Construction and Solver Configuration . . . . . . . . . . . . . . . . . . . . 14
4.4.1 Decision Variable and Constraint Structure . . . . . . . . . . . . . . . . . 14
4.4.2 IPOPT Solver Configuration and Termination Criteria . . . . . . . . . . 14
4.4.3 IPOPT Termination Status Codes . . . . . . . . . . . . . . . . . . . . . . 15
4.5 Prediction Horizon Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
4.6 Warm-Starting and Computational Acceleration . . . . . . . . . . . . . . . . . . 17


**5** **Soft** **Actor-Critic** **(SAC)** **Architecture** **17**
5.1 Markov Decision Process (MDP) Formulation . . . . . . . . . . . . . . . . . . . 17
5.1.1 Observation Space ( _S_ ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
5.1.2 Action Space ( _A_ ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
5.1.3 Reward Formulation ( _R_ ) . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
5.2 Gymnasium Environment Wrapper . . . . . . . . . . . . . . . . . . . . . . . . . 18
5.3 Centralized Training with Decentralized Execution (CTDE) . . . . . . . . . . . 19
5.4 Budget Constraint Handling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
5.5 Training Protocol and Hyperparameters . . . . . . . . . . . . . . . . . . . . . . 20


**6** **Uncertainty** **Modeling** **and** **Unified** **Evaluation** **Framework** **20**


2


6.1 Meteorological Forecast Uncertainty . . . . . . . . . . . . . . . . . . . . . . . . . 21
6.2 Controller Summary Table . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
6.3 Unified Software Evaluation Architecture . . . . . . . . . . . . . . . . . . . . . . 22
6.4 Experimental Grid . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22


**7** **Conclusion** **23**


3


# **1 Introduction**

The preceding chapter established the high-fidelity Agent-Based Model (ABM) that serves as
the virtual plant for this research: a 130-agent crop-soil simulation environment on a topographically heterogeneous field in Gilan Province, Iran, with five coupled state variables per
agent, cascade surface water routing governed by a directed elevation graph, and validated
hydrological dynamics (Pearson _r_ = 0 _._ 738 against NASA GWETROOT satellite observations
in the dry-year scenario). With the physical environment rigorously defined and parameterized
for Hashemi rice cultivation, the present chapter addresses the central engineering question:

_how_ _should_ _the_ _irrigation_ _control_ _policy_ _be_ _designed_ _to_ _maximize_ _crop_ _yield_ _under_ _strict_ _water_

_resource_ _constraints,_ _and_ _how_ _do_ _model-based_ _and_ _learning-based_ _approaches_ _compare_ _in_ _this_

_setting?_


The agricultural irrigation problem possesses several characteristics that make it particularly
challenging from a control-theoretic perspective. First, the system dynamics are nonlinear and
spatially coupled: surface runoff cascades through the directed elevation graph, creating complex interdependencies between the 130 agents that a controller must anticipate. Second, the
system is subject to significant stochastic disturbances in the form of unpredictable precipitation events, which can range from zero rainfall over extended dry periods to concentrated storm
events exceeding 20 mm/day. Third, the problem is subject to hard constraints at multiple
scales: a per-agent actuator limit of 12.0 mm/day imposed by the physical irrigation infrastructure, and a global seasonal water budget that reflects regional water scarcity. Finally, the
objective is inherently multi-criteria, requiring the controller to balance biomass maximization
against water conservation, drought prevention, and waterlogging mitigation.


This chapter is organized as follows. Section 2 formalizes the constrained optimal control problem, defining the state space, action space, system dynamics, and the general cost structure.
Section 3 presents the two open-loop baseline controllers that serve as lower-bound performance
references. Section 4 details the Model Predictive Control architecture, including the multiobjective cost function, the Control-Oriented Model with its smooth approximation techniques,
the NLP construction and variable structure, the IPOPT solver configuration with precise termination criteria, the prediction horizon selection, and the weight calibration methodology.
Section 5 describes the Soft Actor-Critic reinforcement learning architecture, including the
Markov Decision Process formulation, the Gymnasium environment wrapper, the CTDE network structure, the budget constraint handling mechanism, and the training protocol. Section 6
establishes the unified evaluation framework, including the meteorological forecast uncertainty
model, the complete controller taxonomy, and the experimental grid. Section 7 concludes with
a summary of the design decisions and their theoretical trade-offs.

# **2 Problem Formulation and System Constraints**


The intelligent irrigation management of the 130-agent topographical grid is formally defined as
a constrained, finite-horizon optimal control problem. Over the _K_ = 93 day cultivation season
for Hashemi rice, the control architecture must dynamically balance the conflicting objectives of


4


maximizing terminal crop yield against the cost of water consumption, while strictly adhering
to the physical limitations of the irrigation hardware and the regional water allocation. To
ensure a standardized comparative evaluation across both traditional heuristics and advanced
learning-based algorithms, the system state, action space, constraint structure, and transition
dynamics must be rigorously defined.

## **2.1 State Space Representation**


The discrete-time system state at day _k_, denoted as **X** _k_, aggregates the physical and biological
conditions of all _N_ = 130 spatial agents. The state space _X_ is a multi-dimensional continuous
space where each agent _n ∈{_ 1 _,_ 2 _, . . ., N_ _}_ is described by a state vector _x_ _[n]_ _k_ [encompassing] [both]
hydrological and phenological variables:


**X** _k_ = _{x_ [1] _k_ _[, x]_ [2] _k_ _[, . . ., x][N]_ _k_ _[} ∈X]_ (1)


where each _x_ _[n]_ _k_ [consists] [of] [five] [state] [variables:]


  - _x_ _[n]_ 1 [:] [Root-zone] [soil] [moisture] [(mm),] [bounded] [by] [[] _[WP]_ _[·][ θ]_ [5] _[, θ][sat]_ _[·][ θ]_ [5][] = [60] _[,]_ [ 200]] [mm.]

  - _x_ _[n]_ 2 [:] [Accumulated] [thermal] [time] [(] _[◦]_ [C] _[·]_ [days),] [monotonically] [increasing] [from] _[x]_ [2] _[,]_ [init] [=] [210] [to]
maturity threshold _θ_ 18 = 1250.

  - _x_ _[n]_ 3 [:] [Crop] [maturity] [index] [(dimensionless),] [tracking] [cumulative] [heat] [and] [drought] [stress]
penalties.

  - _x_ _[n]_ 4 [:] [Accumulated] [aboveground] [dry] [biomass] [(g/m][2][),] [initialized] [at] _[x]_ [4] _[,]_ [init] [=] [60] [g/m][2] [re-]
flecting the 20-day nursery period.

  - _x_ _[n]_ 5 [:] [Surface] [ponding] [depth] [(mm),] [representing] [un-infiltrated] [water] [remaining] [on] [the] [soil]
surface.


The full system state therefore lies in R [5] _[N]_ = R [650] . To accurately simulate the initial flooded
condition required for freshly puddled rice fields in Gilan, the system enforces a strict state
initialization constraint across the entire spatial grid. At _k_ = 0, the root-zone soil moisture is
uniformly initialized to 100% field capacity:


_x_ _[n]_ 1 [(0) =] _[ θ]_ [6] _[·][ θ]_ [5] [= 0] _[.]_ [35] _[ ×]_ [ 400 = 140] [mm] _∀n ∈{_ 1 _, . . ., N_ _}_ (2)


with _x_ _[n]_ 5 [(0) = 0] [mm] [(no] [initial] [ponding).] [The] [thermal] [time] [and] [biomass] [are] [initialized] [at] [their]
nursery-calibrated values _x_ _[n]_ 2 [(0) = 210] [and] _[x][n]_ 4 [(0) = 60] [g/m][2] [respectively,] [while] _[x][n]_ 3 [(0) = 0][.]

## **2.2 Action Space and Actuator Constraints**


The control action at day _k_, denoted as **U** _k_, dictates the daily irrigation depth applied to each
agent. The action space _U_ is bounded by the physical realities of the pumping and distribution
infrastructure. Let _u_ _[n]_ _k_ [represent] [the] [irrigation] [command] [dispatched] [to] [agent] _[n]_ [on] [day] _[k]_ [.]
The controller is strictly constrained by a maximum daily actuator limit, _u_ max, establishing a
bounded continuous action space:


5


_U_ =        - **U** _k_ _∈_ R _[N]_ _|_ 0 _≤_ _u_ _[n]_ _k_ _[≤]_ _[u]_ [max] _∀n ∈{_ 1 _, . . ., N_ _}_        - (3)


Based on the physical capacities of precision drip infrastructure and the hydraulic constraints
of the Gilan field, the actuator cap is defined as _u_ max = 12 _._ 0 mm/day. This value represents the
maximum volumetric delivery rate achievable by a single irrigation zone without exceeding the
soil infiltration capacity or generating excessive surface runoff. Any policy _π_ or optimization
solver must ensure **U** _k_ _∈U_ at every discrete time step.

## **2.3 Global Water Budget Constraint**


A defining feature of the constrained irrigation problem, and the primary motivation for this
research in the context of Iran’s anthropogenic drought crisis, is the presence of a hard global
resource constraint. The total cumulative irrigation volume applied across all agents over the
entire growing season must not exceed a finite seasonal water budget _W_ total:



_K−_ 1



_k_ =0



_N_


_u_ _[n]_ _k_ _[≤]_ _[W]_ [total] _[·][ N]_ (4)
_n_ =1



where _W_ total is expressed in mm (equivalent depth per agent). The full irrigation requirement for
Hashemi rice, derived from the 25-year climatological water budget analysis, is _W_ full = 484 mm.
To evaluate controller robustness under increasing water scarcity, three budget tiers are defined:


Table 1: Seasonal water budget tiers for Hashemi rice.


**Budget** **Tier** **Fraction** **of** _W_ **full** _W_ **total** **(mm)** **Scarcity** **Context**


Full allocation 100% 484.0 Nominal water availability
Moderate deficit 85% 411.4 Regional quota reduction
Severe deficit 70% 338.8 Drought-year rationing


The budget values are anchored to the agronomic water demand calculation ET _c_ = ET [PM] 0 _·_ _Kc ·_
_N_ days = 5 _._ 02 _×_ 1 _._ 15 _×_ 93 = 537 _._ 0 mm, minus the 25-year average seasonal rainfall of 53.3 mm,
yielding a full irrigation need of _I_ full _≈_ 484 mm. The 85% and 70% tiers reflect realistic policy
scenarios: the moderate deficit corresponds to a 15% volumetric quota reduction that water
authorities in Iranian provinces have historically imposed during low-flow years, while the severe
deficit represents the extreme rationing required during consecutive drought years.


This constraint is fundamentally different in character for MPC versus RL. The MPC enforces
it as a hard linear inequality constraint within the NLP formulation (Section 4.4), guaranteeing
mathematical satisfaction. The RL agent, lacking native constrained-optimization mechanisms,
must learn to respect the budget through reward shaping, action clipping, and early termination
penalties (Section 5.4).


6


## **2.4 System Dynamics and Optimal Control Objective**

The system transitions from state **X** _k_ to **X** _k_ +1 via the nonlinear system dynamics _f_ system, which
encompass both the topographical cascade routing and the biological growth models defined in
the preceding chapter:


**X** _k_ +1 = _f_ system( **X** _k,_ **U** _k,_ **W** _k_ ) (5)


where **W** _k_ represents the stochastic meteorological disturbance vector at day _k_, comprising
rainfall _P_ ( _k_ ), reference evapotranspiration ET0( _k_ ), mean/max/min temperatures _T_ mean( _k_ ),
_T_ max( _k_ ), _T_ min( _k_ ), and solar radiation _Rs_ ( _k_ ). The overarching objective for any deployed controller is to find a policy _π_ ( **X** _k_ ) _→_ **U** _k_ that minimizes an expected cumulative cost function _J_
over the finite horizon _K_ :











min _π_ [E] **[W]**



Φ( **X** _K_ ) +



_K−_ 1




_L_ ( **X** _k,_ **U** _k_ )

_k_ =0



(6)



subject to **U** _k_ _∈U_ and the global budget constraint (Eq. 4), where Φ( **X** _K_ ) represents the terminal Mayer cost (maximizing the terminal biomass _x_ 4 at harvest), and _L_ ( **X** _k,_ **U** _k_ ) represents the
intermediate Lagrange path costs penalizing cumulative water expenditure, localized ponding
( _x_ 5), drought stress, and actuator oscillation. This general optimal control formulation serves
as the architectural foundation. The subsequent sections detail how this objective is uniquely
approximated and solved by the baseline heuristics, the predictive optimizer (MPC), and the
reinforcement learning agent (SAC).

# **3 Baseline and Heuristic Controllers**


To rigorously evaluate the performance of the advanced closed-loop predictive (MPC) and
learning-based (SAC) algorithms, two open-loop baseline controllers are implemented. These
baselines represent the lower bound of crop survival and the standard agricultural heuristic
currently deployed in Gilan Province, respectively. Since these are open-loop heuristics, they
calculate the irrigation trajectory strictly as a function of time and total seasonal water budget,
remaining entirely blind to both the daily topographical field state ( **X** _k_ ) and the meteorological
forecast.

## 3.1 The No-Irrigation Baseline ( C 1 )


The No-Irrigation controller ( _C_ 1) serves as the absolute lower-bound performance metric (the
rainfed benchmark). It strictly enforces a zero-action policy throughout the 93-day season:


_u_ _[n]_ _k_ [= 0] _∀n ∈{_ 1 _, . . ., N_ _},_ _∀k_ _∈{_ 0 _, . . ., K −_ 1 _}_ (7)


7


Evaluating the field dynamics under _C_ 1 isolates the natural hydrological capacity of the Gilan
terrain, establishing a baseline terminal biomass and baseline stress duration against which the
value-add of active irrigation can be quantified. This controller consumes zero water budget by
definition and provides the lower bound for the yield–water trade-off frontier. Any controller
that fails to significantly outperform _C_ 1 would indicate either a flawed cost function or an
ineffective optimization algorithm.

## 3.2 The Fixed-Schedule Heuristic ( C 2 )


Traditional rice cultivation in the Gilan region relies on a front-loaded irrigation strategy, providing heavy water application during the immediate post-transplanting establishment phase
and linearly tapering off as the crop approaches maturity and harvest. To formalize this heuristic into a repeatable mathematical baseline, the Fixed-Schedule controller ( _C_ 2) distributes the
total seasonal water budget ( _W_ total) across _Ke_ = 19 discrete irrigation events. The events are
uniformly spaced at an interval of 5 days. To achieve the front-loaded distribution, a lineardecay weight _wj_ is assigned to each event _j_ _∈{_ 1 _,_ 2 _, . . ., Ke}_ :


_wj_ = [2(] _[K][e][ −]_ _[j]_ [ + 1)] (8)

_Ke_ ( _Ke_ + 1)


By definition, [�] _j_ _[K]_ =1 _[e]_ _[w][j]_ [=] [1][.] The total volumetric allocation for event _j_ is computed as
_wj_ _×_ _W_ total. Rather than applying this volume in a single daily deluge—which would violate physical infrastructure limits and generate massive uncaptured runoff—the volume is distributed evenly across the days within the event interval. Crucially, this traditional heuristic is
spatially uniform; it is entirely unaware of the topographical cascade routing. The calculated
daily rate is dispatched identically to all agents _n_, strictly bounded by the physical actuator
cap _u_ max:




   - _wj_ _· W_ total
_u_ _[n]_ _k_ [= min] interval _[,]_ _[u]_ [max]




_∀n ∈{_ 1 _, . . ., N_ _}_ (9)



Furthermore, a strict accounting mechanism ensures the cumulative applied irrigation does not
exceed _W_ total in instances where a truncated season or floating-point drift alters the total sum.
By comparing the spatially aware MPC and SAC against this spatially blind _C_ 2 baseline, the
exact agronomic value of topographical awareness can be isolated and quantified.

# **4 Model Predictive Control (MPC) Architecture**

## **4.1 The Multi-Objective Cost Function**


The MPC optimization problem is formulated to maximize crop yield while minimizing resource
expenditure and environmental stress over a prediction horizon _Hp_ . To ensure stable convergence within the interior-point solver (IPOPT), the cost function _J_ is heavily normalized. Each
term is scaled to _O_ (1) using agronomically and economically derived reference values. The total


8


objective function is defined as a weighted sum of five normalized terms:


_J_ = _J_ biomass + _J_ water + _J_ drought + _J_ ponding + _J_ ∆ _u_ (10)


**4.1.1** **Terminal** **Biomass** **(Mayer** **Cost)**


To incentivize crop growth without requiring the solver to look ahead to the end of the 93day season, the controller maximizes the field-mean biomass _x_ 4 at the end of the prediction
horizon. This is formulated as a negative cost to convert the maximization into a minimization
compatible with the NLP solver:


_x_ ¯4( _k_ + _Hp_ )
_J_ biomass = _−α_ 1 (11)

_x_ 4 _,_ ref


where _x_ ¯4( _k_ + _Hp_ ) = _N_ 1 - _Nn_ =1 _[x]_ 4 _[n]_ [(] _[k]_ [ +] _[ H][p]_ [)] [is] [the] [field-mean] [biomass] [at] [the] [terminal] [step] [of] [the]
prediction horizon, _x_ 4 _,_ ref = 900 g/m [2] represents the target yield threshold for Hashemi rice
under optimal conditions (corresponding to approximately 3,780 kg/ha via _Y_ = _x_ 4 _×_ HI _×_ 10 =
900 _×_ 0 _._ 42 _×_ 10), and _α_ 1 = 1 _._ 0 serves as the primary economic anchor representing the high
market revenue of the rice crop.


**4.1.2** **Water** **Consumption** **Penalty**


The cost of irrigation is penalized over the prediction horizon to enforce water conservation:



_J_ water = _α_ 2



_Hp−_ 1



_j_ =0




- _N_
_n_ =1 _[u][n]_ [(] _[k]_ [ +] _[ j]_ [)]

(12)
_W_ daily,ref



where _W_ daily,ref = 5 _._ 0 _· N_ = 650 mm represents the theoretical baseline daily demand for the
_N_ = 130 agents (based on the mean seasonal ET _c_ _≈_ 5 _._ 8 mm/day, rounded to 5.0 mm for
normalization). The nominal weight _α_ 2 = 0 _._ 01 is anchored to the domestic water pricing
structure in Iran.


**4.1.3** **Drought** **Stress** **Regularization**


While the biological model penalizes growth during dry periods through the drought stress
function _h_ 3, an explicit penalty is added to the controller objective to aggressively prevent soil
moisture ( _x_ 1) from dropping below the stress threshold ( _ST_ ):



_N_



_n_ =1



max(0 _, ST_ _−_ _x_ _[n]_ 1 [(] _[k]_ [ +] _[ j]_ [))]
(13)
_ST_ _−_ _WP_



_J_ drought = _α_ 3



_Hp−_ 1



_j_ =0



1

_N_



where _ST_ = ( _θ_ 6 _−_ _p ·_ ( _θ_ 6 _−_ _θ_ 2)) _· θ_ 5 = (0 _._ 35 _−_ 0 _._ 20 _×_ 0 _._ 20) _×_ 400 = 124 mm is the management
allowable depletion threshold for rice ( _p_ = 0 _._ 20 per FAO-56), and _WP_ = _θ_ 2 _· θ_ 5 = 0 _._ 15 _×_ 400 =


9


60 mm is the permanent wilting point. The weight _α_ 3 = 0 _._ 1 ensures the controller proactively
irrigates before severe deficit occurs, without dominating the biomass incentive.


**4.1.4** **Ponding** **Penalty**


Due to the topographical cascade routing, excessive rainfall and over-irrigation can cause temporary localized flooding, particularly at low-elevation agents. Following the fractional routing
fix (terrain.py v2.0), which eliminated true sink agents by allowing off-farm drainage through
the padded DEM, the ponding penalty is applied to the field-mean surface ponding across all
130 agents:



_N_



_n_ =1



_x_ _[n]_ 5 [(] _[k]_ [ +] _[ j]_ [)]

(14)
_x_ 5 _,_ ref



_J_ ponding = _α_ 4



_Hp−_ 1



_j_ =0



1

_N_



where _x_ 5 _,_ ref = 50 _._ 0 mm serves as the normalization reference representing acute storm ponding
depth, and _α_ 4 = 0 _._ 5 places a high priority on preventing persistent waterlogging across the
field. The choice of _x_ 5 _,_ ref = 50 mm (increased from an initial value of 10 mm used in an earlier
sink-only formulation) reflects the realistic ponding depths observed during monsoon-season
storm events in Gilan, ensuring that the penalty term remains _O_ (1) during typical wet-weather
conditions rather than saturating the cost function.


**4.1.5** **Actuator** **Smoothing** **(Control-Rate** **Regularization)**


To prevent erratic, high-frequency oscillatory irrigation commands across consecutive days—
which would be physically damaging to valve actuators and operationally impractical—a standard control-rate regularization term is included:



_J_ ∆ _u_ = _α_ 5



_Hp−_ 1



_j_ =1



_∥_ **u** ( _k_ + _j_ ) _−_ **u** ( _k_ + _j −_ 1) _∥_ [2]

(15)
_u_ [2] max _[·][ N]_



where the normalization by _u_ [2] max _[·]_ _[N]_ [ensures this term remains] _[ O]_ [(1)][ even at maximum actuator]
swing. The weight _α_ 5 = 0 _._ 005 provides gentle smoothing without materially constraining the
optimizer’s freedom to respond to sudden rainfall events.

## **4.2 Weight Calibration and Economic Anchoring**


The five cost weights _{α_ 1 _, α_ 2 _, α_ 3 _, α_ 4 _, α_ 5 _}_ are not arbitrary tuning parameters but are calibrated
to reflect real economic and agronomic trade-offs. The primary anchor is _α_ 1 = 1 _._ 0, representing
the revenue from a hectare of Hashemi rice at typical Gilan market prices ( _∼_ 4,000 USD/ha).
The water cost weight _α_ 2 is set relative to this anchor based on the volumetric price of irrigation
water in Iran. To evaluate the sensitivity of the MPC policy to economic assumptions about
water pricing, three values of _α_ 2 are tested:


10


Table 2: Water cost weight sweep for sensitivity analysis.


_α_ 2 **Value** **Water** **Price** **Tier** **Economic** **Interpretation**


0.0001 Subsidized Near-zero marginal cost of water;
mimics the current heavily subsidized
pumping regime in Iran
0.01 Nominal Reflects the domestic water pricing tier;
the default operating point
0.03 Scarcity premium Reflects a future policy scenario with
volumetric water pricing at costrecovery levels


The complete weight vector at the nominal operating point is _**α**_ = _{_ 1 _._ 0 _,_ 0 _._ 01 _,_ 0 _._ 1 _,_ 0 _._ 5 _,_ 0 _._ 005 _}_ .
The sensitivity sweep is conducted on the dry/100% scenario at _Hp_ = 8 across 7 configurations
(varying _α_ 2 while holding other weights fixed) to quantify the Pareto frontier between yield
and water consumption.

## **4.3 Control-Oriented Model (COM) and Symbolic Formulation**


To enable real-time numerical optimization within the receding horizon framework, the highfidelity Agent-Based Model (ABM) described in the preceding chapter must be translated into
a Control-Oriented Model (COM). The COM is implemented as a CasADi symbolic computational graph using the SX (Scalar eXpression) framework, defining the explicit mapping
_f_ COM : ( _Xk, Uk, Pk_ ) _→Xk_ +1, where _Pk_ represents exogenous parameters (weather forecasts and
precomputed biological quantities). To ensure tractability for the IPOPT interior-point solver,
several rigorous dimensionality reduction and computational strategies are applied.


**4.3.1** **Dimensionality** **Reduction** **of** **the** **State** **Space**


While the complete biological system tracks five state variables per agent, maintaining all
variables as shooting states in the Non-Linear Programming (NLP) formulation would result
in 5 _×_ 130 _× Hp_ = 5 _,_ 200 shooting variables at _Hp_ = 8 alone, producing prohibitively large
and dense Jacobian matrices. Because thermal time ( _x_ 2) is purely driven by exogenous climate
data and is therefore entirely decoupled from the control action, it is precomputed offline for
the full season and injected as a time-varying parameter. Maturity ( _x_ 3) acts as a slow-moving,
monotonic biological clock that is tracked from the plant’s true state at each receding-horizon
step to prevent open-loop drift. Biomass ( _x_ 4) is accumulated inline within the COM strictly
to evaluate the terminal Mayer cost but is not used as a shooting state. The COM therefore
restricts the formal shooting states exclusively to the fast-moving hydrological variables:


_X_ shoot = _{x_ 1 _, x_ 5 _}_ (16)


This reduces the shooting variables from 5 _N_ to 2 _N_ = 260 per horizon step, yielding a total of


11


2 _N_ _× Hp_ + _N_ _× Hp_ = 3 _N_ _× Hp_ NLP variables (two states plus one control per agent per step)
plus the initial state parameters.


**4.3.2** **Deterministic** **Caching** **of** **Biological** **Nonlinearities**


The biological penalty functions for heat stress ( _h_ 2), cold stress ( _h_ 7), thermal time accumulation ( _h_ 1), and the base growth function ( _g_ base) rely on highly nonlinear sigmoid formulations.
Because these functions depend strictly on deterministic meteorological variables (temperature
and radiation) and are entirely decoupled from the control action ( _u_ ), they are precomputed
offline for the entire 93-day season. By injecting these precomputed scalar arrays into the
CasADi graph as time-varying parameters rather than calculating them inline, the NLP solver
bypasses the expensive evaluation of sigmoidal gradients and their second derivatives, significantly accelerating the per-step solve time and improving Hessian sparsity.


**4.3.3** **Static** **Embedding** **of** **Topographical** **Cascade** **Routing**


A primary challenge in agricultural MPC is simulating spatial water routing without relying
on computationally prohibitive coupled Partial Differential Equations (PDEs). The COM resolves this by statically embedding the field’s directed topographical graph into the symbolic
formulation. At NLP construction time, the 130 agents are sorted into a strict topological order
(from maximum elevation to terminal boundary agents). During the symbolic forward pass of
each horizon step, surface hydrology is unrolled sequentially: for a given agent _n_ processed at
position _p_ in the topological order, all uphill contributions from agents at positions 1 _, . . ., p −_ 1
have already been computed and are available as symbolic expressions.


The surface water availability for agent _n_ is computed as:



_W_ surf [(] _[n]_ [)][(] _[k]_ [) =] _[ x]_ 5 [(] _[n]_ [)][(] _[k]_ [) +] _[ P]_ [(] _[k]_ [) +] _[ u]_ [(] _[n]_ [)][(] _[k]_ [) +] 
_m∈U_ ( _n_ )



_ϕ_ [(] 2 _[m]_ [)] ( _k_ ) (17)

_Nr_ [(] _[m]_ [)]



where _U_ ( _n_ ) is the set of uphill neighbors of agent _n_, and _Nr_ [(] _[m]_ [)] is the _total_ number of lowerelevation neighbors of agent _m_, computed using the padded DEM (terrain.py v2.0). This
fractional routing boundary condition is critical: the DEM is padded by one cell on each edge
by extrapolating the slope. _Nr_ counts all lower neighbors including external (off-farm) pad cells,
while the routing loop only sends water to internal agents. When the COM divides _ϕ_ 2 by _Nr_,
the fraction directed toward pad cells is implicitly removed from the mass balance, accurately
simulating off-farm drainage without introducing dummy variables to the solver. This fixed
the “bathtub effect” observed in the initial implementation, where three sink agents ( _Nr_ = 0)
trapped all cascade runoff and experienced catastrophic waterlogging.


**4.3.4** **Smooth** **Approximations** **for** **Solver** **Convergence**


The original ABM equations contain several non-smooth operations—max(0 _, x_ ), min( _a, b_ ),
and conditional branching via `ca.if_else` —that create discontinuities in the first and second
derivatives of the NLP objective and constraints. While CasADi can represent these operations


12


symbolically, the IPOPT interior-point solver requires _C_ [2] -continuous (twice continuously differentiable) functions to reliably compute the Hessian of the Lagrangian and perform Newton
steps. Using exact non-smooth operators caused severe convergence failures: IPOPT would
stall at “Restoration Failed” or “Search Direction Becomes Too Small” after encountering kinks
in the derivative landscape.


To resolve this, smooth _C_ [2] -continuous approximations are employed with a smoothing parameter _ε_ = 0 _._ 01:


**Smooth** **maximum:** The half-rectifier max(0 _, x_ ) is replaced by:



_√_
max(0 _, x_ ) = _[x]_ [ +]




_x_ [2] + _ε_ [2]



(18)
2



This function satisfies max(0 _, x_ ) _→_ max(0 _, x_ ) as _ε_ _→_ 0, is everywhere differentiable, and

    introduces an error bounded by _ε/_ 2 = 0 _._ 005 mm—negligible relative to the soil moisture range
of [60, 200] mm.


**Smooth** **minimum:** The min( _a, b_ ) operation is derived from the smooth maximum via:


~~�~~ ( _a −_ _b_ ) [2] + _ε_ [2]
min(� _a, b_ ) = _a_ + _b −_ max(� _a, b_ ) = _[a]_ [ +] _[ b][ −]_ 2 (19)


**Guarded** **denominator** **for** **drought** **stress:** The drought stress function _h_ 3 = 1 _−_ _θ_ 14 _·_
max(1 _−_ _ϕ_ 1 _/_ ET _c,_ 0) contains a division by ET _c_ that approaches zero on cool, humid days when
the atmospheric evaporative demand vanishes. To prevent division by zero in the symbolic
graph, the denominator is guarded:











_h_ [(] 3 _[n]_ [)][(] _[k]_ [) = 1] _[ −]_ _[θ]_ [14] _[·]_ max [�]



_ϕ_ [(] 1 _[n]_ [)][(] _[k]_ [)]
1 _−_ [0]
ET _c_ ( _k_ ) + _ε_ _[,]_



(20)



**SCS** **runoff** **smoothing:** The conditional SCS curve number computation (Eq. 5 in the ABM
chapter) originally used `ca.if_else`, which creates a derivative discontinuity at _W_ surf = _θ_ 3.
This is replaced by the smooth formulation:


max(0 _, W_ surf [(] _[n]_ [)] _[−]_ _[θ]_ [3][)][2]
_ϕ_ [(] 2 _[n]_ [)] = [�] (21)

_W_ surf [(] _[n]_ [)] [+ 4] _[θ]_ [3][ +] _[ ε]_


The value _ε_ = 0 _._ 01 was selected through systematic experimentation: values below
0.001 reintroduced convergence failures due to near-discontinuous derivatives, while values
above 0.1 introduced unacceptable mass-balance artifacts (artificial water creation exceeding
0.5 mm/season). At _ε_ = 0 _._ 01, the total mass-balance error is bounded below 0.05 mm over
the 93-day season, which is four orders of magnitude smaller than the typical seasonal water
budget.


13


## **4.4 NLP Construction and Solver Configuration**

**4.4.1** **Decision** **Variable** **and** **Constraint** **Structure**


The NLP is constructed using a direct multiple-shooting formulation. At each prediction
horizon step _j_ _∈{_ 0 _, . . ., Hp_ _−_ 1 _}_, the decision variables consist of the shooting states
_{x_ _[n]_ 1 [(] _[k]_ [ +] _[ j]_ [)] _[, x][n]_ 5 [(] _[k]_ [ +] _[ j]_ [)] _[}]_ [for] [all] _[N]_ [= 130] [agents] [and] [the] [control] [inputs] _[{][u][n]_ [(] _[k]_ [ +] _[ j]_ [)] _[}]_ [for] [all] [agents.]
The complete variable vector **z** is organized as:











**z** =



_x_ 1:1 _N_ ( _k_ ) _, x_ [1:] 5 _[N]_ ( _k_ )


 - �� initial states (fixed)



_, u_ [1:] _[N]_ ( _k_ ) _, x_ [1:] 1 _[N]_ ( _k_ + 1) _, x_ [1:] 5 _[N]_ ( _k_ + 1) _, u_ [1:] _[N]_ ( _k_ + 1) _, . . ._

 - �� free variables



 (22)



For prediction horizon _Hp_ = 8, this yields the following NLP dimensions:


Table 3: NLP dimensions for _Hp_ = 8 and _Hp_ = 14.


**Quantity** _Hp_ = 8 _Hp_ = 14


Total symbolic variables 3,380 5,720
Free decision variables 3,120 5,460
Equality constraints (shooting gaps) 2,080 3,640
Inequality constraint (budget) 1 1
Total constraints 2,081 3,641
Box bounds on _u_ 2 _×_ 130 _× Hp_ 2 _×_ 130 _× Hp_
NLP build time (typical) 20–50 s 37–44 s


The 260 initial-state variables ( _x_ [1:] 1 _[N]_ ( _k_ ) and _x_ [1:] 5 _[N]_ ( _k_ )) are fixed to the current true plant state at
each receding-horizon step (they appear as parameters, not free variables). The 2,080 equality
constraints at _Hp_ = 8 enforce the shooting gap closure: each constraint requires that the
symbolic COM-predicted next state matches the shooting variable at the subsequent step, i.e.,
_x_ _[n]_ _s_ [(] _[k]_ [ +] _[ j]_ [+ 1)] [=] _[f]_ COM _[ n]_ _,s_ [(] _[k]_ [ +] _[ j]_ [)] [for] _[s]_ _[∈{]_ [1] _[,]_ [ 5] _[}]_ [,] _[n]_ _[∈{]_ [1] _[, . . ., N]_ _[}]_ [,] [and] _[j]_ _[∈{]_ [0] _[, . . ., H][p]_ _[−]_ [2] _[}]_ [.] [The]
single inequality constraint enforces the remaining seasonal water budget.


**4.4.2** **IPOPT** **Solver** **Configuration** **and** **Termination** **Criteria**


The NLP is solved using IPOPT (Interior Point OPTimizer) version 3.14, an open-source
primal-dual interior-point algorithm for large-scale nonlinear optimization. IPOPT implements
a line-search filter method based on the barrier reformulation of the KKT (Karush–Kuhn–
Tucker) optimality conditions. At each iteration, IPOPT solves a symmetric indefinite linear
system to obtain the Newton search direction, using the MUMPS (MUltifrontal Massively
Parallel Sparse direct Solver) as the default linear algebra backend.


The solver is configured with the following key parameters:


14


Table 4: IPOPT solver confguration.


**Option** **Value** **Description**


`tol` 10 _[−]_ [8] Desired convergence tolerance (relative). The algorithm terminates successfully when the scaled NLP error
falls below this threshold.
`acceptable_tol` 10 _[−]_ [6] Acceptable convergence tolerance. If the algorithm encounters
`acceptable_iter` consecutive iterates
satisfying this relaxed tolerance, it
terminates.
`acceptable_iter` 15 Number of consecutive acceptable iterates required to trigger early termination at the acceptable tolerance level.
`max_iter` 3,000 Maximum number of interior-point iterations per NLP solve.
`constr_viol_tol` 10 _[−]_ [4] Maximum allowable constraint violation (absolute) for successful termination.
`dual_inf_tol` 1.0 Maximum allowable dual infeasibility
(absolute).
`compl_inf_tol` 10 _[−]_ [4] Maximum allowable complementarity
violation.
`print_level` 0 Suppress IPOPT console output during
batch runs.
`linear_solver` MUMPS Sparse direct solver for the KKT system.


**4.4.3** **IPOPT** **Termination** **Status** **Codes**


Understanding the termination status is critical for interpreting MPC results. IPOPT reports
one of several exit conditions, of which two are relevant to the completed simulations:


**Solve_Succeeded** **(Return** **code** **0):** This status indicates that IPOPT found a locally optimal point satisfying _all_ desired convergence criteria simultaneously. Specifically, the following
conditions are met: (i) the scaled NLP error (a composite measure of optimality, feasibility, and
complementarity, denoted _Eµ_ in the IPOPT formulation) is smaller than `tol` = 10 _[−]_ [8] ; (ii) the
max-norm of the unscaled dual infeasibility is below `dual_inf_tol` = 1 _._ 0; (iii) the max-norm of
the unscaled constraint violation is below `constr_viol_tol` = 10 _[−]_ [4] ; and (iv) the max-norm of
the unscaled complementarity is below `compl_inf_tol` = 10 _[−]_ [4] . When all four conditions hold,
the returned solution represents a KKT point of the NLP to the desired numerical precision.


**Solved_To_Acceptable_Level (Return code 1):** This status indicates that the algorithm


15


could not converge to the “desired” tolerances specified by `tol`, but it did find a point satisfying the relaxed “acceptable” tolerance level. Specifically, the IPOPT iteration encountered
`acceptable_iter` = 15 consecutive iterations during which the scaled NLP error remained
below `acceptable_tol` = 10 _[−]_ [6] (two orders of magnitude more relaxed than the desired 10 _[−]_ [8] ),
the constraint violation remained below `acceptable_constr_viol_tol` = 0 _._ 01 (two orders
of magnitude more relaxed than the desired 10 _[−]_ [4] ), and the complementarity remained below
`acceptable_compl_inf_tol` = 0 _._ 01. This outcome typically occurs when the NLP is poorly
conditioned at a particular time step—for example, when a sudden large rainfall event creates
a near-discontinuity in the optimal irrigation profile despite the smooth approximations.


The practical impact of `Solved_To_Acceptable_Level` on irrigation quality is minimal in this
application. The constraint violation tolerance is relaxed from 10 _[−]_ [4] mm to 10 _[−]_ [2] mm, which
for a soil moisture range of [60, 200] mm represents less than 0.01% relative error. In the
completed simulation grid, this status was observed on only 3 out of the 90 daily solves across
9 scenario _×_ budget combinations at _Hp_ = 8: specifically at day 30 of wet/100%, day 30 of
moderate/100%, and day 60 of moderate/70%. Inspection of the corresponding _u_ mean values
shows no anomalous behavior, confirming that the acceptable-level solutions are operationally
indistinguishable from fully converged solutions.


**Other** **possible** **status** **codes** (not observed in the completed simulations but handled by
the fail-safe logic) include: `Maximum_Iterations_Exceeded` (code _−_ 1), `Restoration_Failed`
(code _−_ 2), `Infeasible_Problem_Detected` (code 2), `Search_Direction_Becomes_Too_Small`
(code 3), and `Error_In_Step_Computation` (code _−_ 3). In any of these failure modes, the
controller defaults to a zero-irrigation fail-safe ( _u_ _[n]_ _k_ [= 0] [for] [all] _[n]_ [),] [allowing] [the] [physical] [crop] [to]
survive on natural rainfall until the receding horizon re-establishes a feasible trajectory on the
subsequent day.

## **4.5 Prediction Horizon Selection**


The prediction horizon _Hp_ governs the trade-off between the MPC’s look-ahead depth and its
computational cost. Two values are evaluated in this research:


_Hp_ = 8 **days:** This horizon spans approximately one week of future weather, corresponding
to the practical accuracy limit of operational meteorological forecasts. At this horizon, the
NLP contains 3,120 free decision variables and 2,081 constraints, with typical build times of
20–50 seconds and per-step solve times ranging from 1.5 s to 508 s (mean: 10–54 s depending
on scenario and budget).


_Hp_ = 14 **days:** This extended horizon doubles the look-ahead, allowing the MPC to plan across
a full two-week window. The NLP grows to 5,460 free variables and 3,641 constraints. Perstep solve times increase substantially, with means of 45–50 s and worst-case solves exceeding
360 s. The critical question is whether this additional computational investment translates into
improved agronomic outcomes.


The preliminary results at 70% budget indicate that extending the horizon from 8 to 14 days
provides _no_ _measurable_ _yield_ _improvement_ (e.g., dry/70%: 3,779 kg/ha at _Hp_ = 8 versus


16


3,766 kg/ha at _Hp_ = 14), while total runtime increases by 4–5 _×_ (e.g., dry/70%: 957 s at
_Hp_ = 8 versus 4,251 s at _Hp_ = 14). Under severe budget constraints, the optimizer exhausts
the water budget by approximately day 65, after which the remaining solve steps are trivially
_u_ = 0 regardless of horizon length. This finding has important practical implications: for realtime edge deployment scenarios, the shorter _Hp_ = 8 horizon is strongly preferred, as it achieves
equivalent agronomic performance at a fraction of the computational cost.

## **4.6 Warm-Starting and Computational Acceleration**


To reduce solve times across consecutive daily solves, the NLP is warm-started by shifting the
previous day’s optimal trajectory forward by one step. The control sequence _{u_ _[∗]_ ( _k_ ) _, u_ _[∗]_ ( _k_ +
1) _, . . ., u_ _[∗]_ ( _k_ + _Hp −_ 1) _}_ from the previous solve is shifted to provide an initial guess _{u_ _[∗]_ ( _k_ +
1) _, u_ _[∗]_ ( _k_ +2) _, . . ., u_ _[∗]_ ( _k_ + _Hp −_ 1) _, u_ _[∗]_ ( _k_ + _Hp −_ 1) _}_ for the current solve, where the final element is
duplicated. The corresponding state trajectory is similarly shifted. This provides IPOPT with
a feasible starting point close to the optimal solution, typically reducing the iteration count
from 200+ (cold start) to 30–80 (warm start).

# **5 Soft Actor-Critic (SAC) Architecture**


While Model Predictive Control provides a mathematically rigorous, constraint-aware optimization framework, the requirement to solve a massive Non-Linear Program at each time
step poses significant latency challenges for low-power edge computing devices deployed in
agricultural Cyber-Physical Systems. To investigate a low-latency alternative, a Reinforcement
Learning agent is trained to approximate the optimal control policy. The problem is formulated
as a Markov Decision Process (MDP) and solved using the Soft Actor-Critic (SAC) algorithm,
an off-policy actor-critic method based on the maximum entropy reinforcement learning framework.

## **5.1 Markov Decision Process (MDP) Formulation**


The irrigation control problem is defined by the MDP tuple _⟨S, A, P, R, γ⟩_ .


**5.1.1** **Observation** **Space** **(** _S_ **)**


To provide the RL agent with sufficient information to manage both spatial topographical
heterogeneity and the global seasonal water budget, the state observation is defined as a flat,
continuous vector of 660 dimensions. The vector concatenates 10 global field features with 5
local features for each of the _N_ = 130 agents:


**Global** **Features** **(10** **dimensions):** Normalized remaining budget ( _W_ remaining _/W_ total), normalized current day ( _k/K_ ), daily rainfall _P_ ( _k_ ), reference evapotranspiration ET0( _k_ ), field-mean
soil moisture ( _x_ ¯1 _/FC_ ), field-moisture standard deviation, field-mean biomass ( _x_ ¯4 _/x_ 4 _,_ ref), fraction of agents in drought ( _x_ 1 _<_ _ST_ ), fraction of agents in waterlogging ( _x_ 1 _>_ _FC_ ), and the
cumulative 7-day rainfall forecast ( [�][7] _j_ =1 _[P]_ [(] _[k]_ [ +] _[ j]_ [)][).]


17


**Local** **Features** **(650** **dimensions):** For each agent _n_, the network observes its normalized soil moisture ( _x_ _[n]_ 1 _[/FC]_ [),] [normalized] [thermal] [time] [(] _[x][n]_ 2 _[/θ]_ [18][),] [normalized] [biomass] [(] _[x][n]_ 4 _[/x]_ [4] _[,]_ [ref][),]
normalized surface ponding ( _x_ _[n]_ 5 _[/x]_ [5] _[,]_ [ref][),] [and] [its] [static] [normalized] [topographical] [elevation] [(] _[γ][n]_ [).]
The elevation feature enables the policy to learn spatially differentiated irrigation strategies—
for example, reducing irrigation for high-elevation agents whose runoff will cascade to lower
neighbors.


**5.1.2** **Action** **Space** **(** _A_ **)**


The continuous action space mirrors the MPC actuator constraints. The SAC actor outputs
a bounded vector **a** _k_ _∈_ [0 _,_ 1] _[N]_ via a tanh squashing function. During the environment step,
this output is linearly scaled by the physical actuator cap ( _u_ max = 12 _._ 0 mm/day) such that
_u_ _[n]_ _k_ [=] _[ a]_ _k_ _[n]_ _[·][ u]_ [max][,] [matching] [the] [exact] [decision] [space] [of] [the] [MPC] [baseline.]


**5.1.3** **Reward** **Formulation** **(** _R_ **)**


To ensure a fair and direct comparison between the learned policy and the computed MPC
policy, the dense per-step reward _rk_ is engineered as the exact mathematical negation of the
MPC path cost function (Section 4.1):


_rk_ = _−_ ( _J_ water( _k_ ) + _J_ drought( _k_ ) + _J_ ponding( _k_ )) (23)



At the terminal time step _K_ = 93, a sparse positive bonus is awarded based on the normalized
terminal biomass:



_rK_ = +5 _·_ _[x]_ [¯][4][(] _[K]_ [)]



(24)
_x_ 4 _,_ ref



The terminal bonus weight of +5 was selected to ensure that the cumulative terminal reward is
of the same order of magnitude as the sum of dense path rewards over 93 steps, preventing the
agent from myopically minimizing path costs at the expense of long-term biomass accumulation.
This reward structure forces the agent to learn the exact same agronomical and economic tradeoffs as the predictive optimizer.

## **5.2 Gymnasium Environment Wrapper**


The ABM is wrapped as a standard Gymnasium (formerly OpenAI Gym) environment implementing the `reset()`, `step()`, and `render()` interface. The `reset()` method initializes the
130-agent field to the standard initial conditions ( _x_ 1 = 140 mm, _x_ 5 = 0, _x_ 2 = 210, _x_ 4 = 60,
_x_ 3 = 0), samples a climate scenario, and resets the budget counter. The `step(action)` method:
(i) clips the raw action to satisfy the actuator bounds and remaining budget, (ii) applies the
irrigation to the ABM, (iii) advances the ABM by one day including cascade routing, (iv)
constructs the 660-dimensional observation vector, (v) computes the dense reward, and (vi)
returns the standard Gymnasium tuple (obs _,_ reward _,_ terminated _,_ truncated _,_ info). The episode
terminates naturally at _k_ = 93 or is truncated early if the budget violation penalty is triggered.


18


## **5.3 Centralized Training with Decentralized Execution (CTDE)**

A naive multi-agent implementation—where a single monolithic policy network directly outputs 130 independent actions from the 660-dimensional state—would fail to exploit the spatial
symmetry of the field and require an intractable number of parameters. To overcome this, the
architecture employs Centralized Training with Decentralized Execution (CTDE):


**The Decentralized Actor:** The actor network operates on the principle of parameter sharing.
A single Multi-Layer Perceptron (MLP) with hidden layers of [256, 256] and ReLU activations
is instantiated. During the forward pass, this network is applied identically across all 130
agents. For agent _n_, the actor concatenates the 10 global features with the 5 local features of
agent _n_ (totaling 15 inputs) to output a parameterized Gaussian distribution (mean _µ_ _[n]_ and log
standard deviation log _σ_ _[n]_ ) for that specific agent’s irrigation action. This forces the network to
learn a generalized policy: _how_ _to_ _irrigate_ _any_ _agent_ _given_ _its_ _specific_ _elevation_ _and_ _moisture_

_state,_ _conditioned_ _on_ _the_ _global_ _budget_ _status._


**The** **Centralized** **Critic:** Unlike the actor, the twin critic networks evaluate the joint action
value _Q_ ( **S** _k,_ **A** _k_ ). Each critic takes the full 660-dimensional state and the 130-dimensional joint
action vector as input (790 dimensions total), passing through hidden layers of [512, 512, 256]
with ReLU activations. By observing the entire state-action space during offline training, the
centralized critic learns the shadow price of the global water budget and the cascading runoff
effects across the topographical graph, guiding the decentralized actor toward a globally optimal
cooperative policy. The twin-critic architecture (two independent _Q_ -networks with minimum
taken for the target) follows the standard SAC formulation to mitigate overestimation bias.

## **5.4 Budget Constraint Handling**


While IPOPT natively handles the global water budget _u_ _≤_ _W_ total via a hard linear in
[�]
equality constraint in the NLP, standard RL algorithms lack native constrained-optimization
mechanisms. The SAC environment enforces the budget via a three-tier soft penalty system:



**Tier 1 — Action Clipping (Hard Safety Net):** As an absolute guarantee, the environment
clips the daily action array so that the total requested volume across all agents never mathematically exceeds the remaining budget. Specifically, if [�] _n_ _[u]_ _k_ _[n]_ _[>]_ _[W]_ [remaining][(] _[k]_ [)][,] [all] [actions]



ematically exceeds the remaining budget. Specifically, if [�] _n_ _[u]_ _k_ _[n]_ _[>]_ _[W]_ [remaining][(] _[k]_ [)][,] [all] [actions]

are proportionally scaled down: _u_ _[n]_ _k_ _[←]_ _[u]_ _k_ _[n]_ _[·][ W]_ [remaining][(] _[k]_ [)] _[/]_ [ �] _n_ _[u]_ _k_ _[n]_ [.] [This] [ensures] [that] [the] [budget]



are proportionally scaled down: _u_ _[n]_ _k_ _[←]_ _[u]_ _k_ _[n]_ _[·][ W]_ [remaining][(] _[k]_ [)] _[/]_ [ �] _n_ _[u]_ _k_ _[n]_ [.] [This] [ensures] [that] [the] [budget]

constraint is never violated, regardless of the agent’s policy output.



**Tier** **2** **—** **Burn-Rate** **Shaping:** To prevent the agent from dumping its entire budget in the
first few weeks (a common failure mode during early training), a negative reward shaped by
the normalized season progression is applied if the daily expenditure rate significantly outpaces
the seasonal timeline. Let _f_ spent = 1 _−_ _W_ remaining _/W_ total be the fraction of budget spent and
_f_ time = _k/K_ be the fraction of the season elapsed. If _f_ spent _> f_ time +0 _._ 15 (the agent is spending
more than 15 percentage points ahead of schedule), a penalty of _r_ burn = _−_ 2 _._ 0 _·_ ( _f_ spent _−_ _f_ time)
is applied.


**Tier** **3** **—** **Early** **Termination:** If the agent triggers the hard clip limit by attempting to
exceed the total budget, a severe penalty ( _r_ = _−_ 100) is applied, and the episode is immediately


19


truncated. This catastrophic penalty signal provides a strong gradient during early training to
discourage budget-violating policies.

## **5.5 Training Protocol and Hyperparameters**


The SAC agent is trained using Stable-Baselines3 (SB3), a well-maintained PyTorch implementation of standard RL algorithms. Training is conducted on Kaggle’s free GPU tier, which
provides NVIDIA Tesla P100 GPUs with a 9-hour session limit and 30 GPU-hours per week.
The complete hyperparameter configuration is:


Table 5: SAC training hyperparameters.


**Hyperparameter** **Value** **Source/Justification**


Learning rate (actor & critic) 3 _×_ 10 _[−]_ [4] Default SAC
Replay buffer size 10 [6] transitions Standard for continuous control
Batch size 256 Standard for continuous control
Discount factor ( _γ_ ) 0.99 Near-horizon: 93-step episodes
Soft update coefficient ( _τ_ ) 0.005 Default SAC
Entropy coefficient ( _α_ ) Auto-tuned Constrained dual gradient descent
Target entropy _−_ dim( _A_ ) = _−_ 130 Default: _−|A|_
Total training steps 5 _×_ 10 [6] Kaggle budget permitting
Random seeds 5 Statistical significance


The entropy coefficient _α_ is automatically tuned via the constrained formulation introduced by
Haarnoja et al. (2019), where _α_ is treated as a Lagrange multiplier on the minimum entropy
constraint _H_ [ _π_ ] _≥H_ target. The target entropy is set to _−_ 130 (one per action dimension),
following the standard heuristic of _−_ dim( _A_ ).


Training is conducted for 5 independent random seeds on the dry/100% scenario. Each seed
runs for up to 5 _×_ 10 [6] environment steps (approximately 53,763 full episodes of 93 steps each).
Once trained, the best-performing seed (selected by validation reward) is evaluated across all
9 scenario _×_ budget combinations to assess generalization.

# **6 Uncertainty Modeling and Unified Evaluation Frame-** **work**


To guarantee the validity of the comparative analysis, the experimental framework must ensure that all controllers are subjected to identical environmental conditions, constraints, and
standardized evaluation metrics. Furthermore, to evaluate the real-world robustness of the
predictive models (MPC and SAC), a formal mathematical model for meteorological forecast
uncertainty must be established.


20


## **6.1 Meteorological Forecast Uncertainty**

While the baseline controllers ( _C_ 1, _C_ 2) and the “No-Forecast” SAC ( _C_ 5) operate without future
weather information, the predictive controllers—the Perfect MPC ( _C_ 3), Noisy MPC ( _C_ 4), and
Forecast SAC ( _C_ 6)—rely on a prediction horizon. To benchmark the theoretical upper bound of
the predictive architecture against a realistic deployment scenario, two distinct forecast modes
are formalized.


**1.** **Perfect** **Forecast** **Mode.** In the perfect information scenario, the forecast vector provided
to the controller exactly matches the true future weather realization generated by the environment simulation. For any meteorological variable _w_ at a future day _k_ + _j_ predicted from the
current day _k_ :


_w_ ˆ( _k_ + _j|k_ ) = _w_ ( _k_ + _j_ ) _∀j_ _∈{_ 0 _, . . ., Hp −_ 1 _}_ (25)


This mode establishes the absolute performance ceiling for the predictive controllers.


**2.** **Noisy** **Forecast** **Mode.** To simulate the decaying accuracy of real-world meteorological
forecasts over time, a stochastically degrading noise model is applied to the rainfall and reference
evapotranspiration (ET0) vectors. Temperature forecasts are left unperturbed, reflecting the
high operational accuracy of thermal forecasting (RMSE _<_ 1 _[◦]_ C at 7-day lead time). The
multiplicative noise injection is defined as:


_w_ ˆ( _k_ + _j|k_ ) = _w_ ( _k_ + _j_ ) _·_ (1 + _ϵj_ ) (26)


where the error term _ϵj_ is drawn from a Gaussian distribution with zero mean and a variance
that grows proportionally to the square root of the look-ahead horizon _j_ :


_ϵj_ _∼N_ (0 _, σj_ [2][)] _[,]_ _σj_ = 0 _._ 15 _·_ ~~�~~ _j_ (27)


This formulation ensures that tomorrow’s forecast ( _j_ = 1) retains high fidelity ( _σ_ 1 = 0 _._ 15, corresponding to _±_ 15% typical error), while forecasts extending toward the end of a 14-day horizon
exhibit severe uncertainty ( _σ_ 14 = 0 _._ 56, corresponding to _±_ 56% typical error), rigorously testing
the stability of the receding-horizon control loop. The square-root decay rate is consistent with
the empirical skill degradation observed in operational NWP (Numerical Weather Prediction)
models.

## **6.2 Controller Summary Table**


To provide a complete reference for the experimental evaluation, all six controllers are formally
defined:


21


Table 6: Complete controller taxonomy.


**ID** **Controller** **Type** **Forecast** **Closed-Loop**


_C_ 1 No-Irrigation Baseline Open-loop None No
_C_ 2 Fixed-Schedule Heuristic Open-loop None No
_C_ 3 MPC (Perfect Forecast) Model-based Perfect Yes
_C_ 4 MPC (Noisy Forecast) Model-based Noisy Yes
_C_ 5 SAC (No Forecast) Learning-based None Yes
_C_ 6 SAC (7-day Forecast) Learning-based 7-day sum Yes

## **6.3 Unified Software Evaluation Architecture**


To eliminate implementation bias during the comparative evaluation, the control loop is strictly
abstracted into three separated software layers:


**The** **Environment** **Layer** maintains the true physical state of the field ( **X** _k_ ), executes the topographical cascade routing using the high-fidelity ABM, and calculates true biological growth.
This layer is identical for all six controllers and is never modified or approximated during
evaluation.


**The** **Controller** **Layer** provides a standardized interface. All six controllers ( _C_ 1 through _C_ 6)
inherit from an identical abstract base class, requiring them to implement a single method:
`get_action(state,` `forecast)` _→_ `action_vector` . This ensures that the heavy IPOPT
NLP solver and the trivial zero-action baseline interact with the environment through the
exact same API.


**The** **Runner** **Layer** is a centralized simulation orchestrator that manages the day-by-day
transition loop. At each step _k_, it: (i) queries the instantiated controller for an action **U** _k_, (ii)
clips the action to satisfy the remaining budget, (iii) applies the action to the environment, (iv)
deducts the applied volume from the global seasonal budget counter, (v) advances the climate
data to the next step, and (vi) logs the complete state, action, and diagnostic information to a
long-format Parquet file with a JSON metadata sidecar.


This architecture guarantees that variations in final agronomic metrics (yield, water consumption, stress days) are strictly attributable to the intrinsic quality of the underlying control
policy, not to implementation artifacts.

## **6.4 Experimental Grid**


The complete experimental evaluation spans the following factorial design:


22


Table 7: Experimental grid for MPC evaluation.


**100%** **Budget** **85%** **Budget** **70%** **Budget**


**Dry** **scenario** **(2022)** ✓ ✓ ✓

**Moderate** **scenario** **(2020)** ✓ ✓ ✓

**Wet** **scenario** **(2024)** ✓ ✓ ✓


Each cell is evaluated at _Hp_ = 8 (complete for all 9 cells) and _Hp_ = 14 (in progress for 70%
budget). The noisy forecast and SAC evaluations will follow the same 9-cell grid. With the
fixed-schedule baseline ( _C_ 2) already complete for all 9 cells across 3 repetitions, and the noirrigation baseline ( _C_ 1) complete for 3 repetitions, the full evaluation comprises 6 _×_ 9 _×_ 2 = 108
simulation runs (excluding repetitions and horizon variants).

# **7 Conclusion**


This chapter has established the complete mathematical and architectural framework for the
four irrigation controllers that will be comparatively evaluated in the subsequent results chapter.
The principal design decisions and their theoretical implications are summarized as follows.


The constrained optimal control problem was formalized with a five-term multi-objective cost
function, where each term is normalized to _O_ (1) and anchored to real agronomic and economic
quantities. The MPC controller was implemented as a large-scale NLP with 3,120 decision
variables and 2,081 constraints (at _Hp_ = 8), solved by the IPOPT interior-point method using
smooth _C_ [2] -continuous approximations ( _ε_ = 0 _._ 01) to guarantee convergence. The solver achieves
successful convergence ( `Solve_Succeeded` ) on 97% of daily solves, with the remaining 3%
converging to acceptable tolerance levels ( `Solved_To_Acceptable_Level`, relative tolerance
relaxed from 10 _[−]_ [8] to 10 _[−]_ [6] ) with negligible impact on irrigation quality. The fractional DEMpadded routing boundary condition resolves the bathtub effect and enables physically realistic
off-farm drainage simulation within the symbolic graph.


The SAC reinforcement learning agent was designed with a CTDE architecture that exploits
spatial symmetry through parameter sharing (15-dimensional per-agent input to a shared
256 _×_ 256 actor), while the centralized twin-critic (790-dimensional input through 512 _×_ 512 _×_ 256
layers) captures the global budget dynamics and inter-agent runoff coupling. The budget constraint is enforced through a three-tier soft penalty system that provides hard safety (action
clipping), temporal shaping (burn-rate penalty), and catastrophic deterrence (early termination).


The theoretical trade-off between these approaches is clear. MPC provides mathematical guarantees on constraint satisfaction and can exploit exact model knowledge and weather forecasts
to compute globally optimal irrigation profiles, but at a computational cost of 10–54 seconds
per daily decision (mean) with worst-case latencies exceeding 500 seconds. SAC, once trained,
executes a simple feed-forward neural network inference in sub-millisecond time, enabling realtime deployment on edge computing hardware, but lacks native constraint guarantees and may


23


exhibit suboptimal behavior in out-of-distribution weather scenarios. The subsequent experimental chapter will quantify these trade-offs through systematic comparison across the 9-cell
scenario _×_ budget grid.

# **References**


[1] J. Lopez-Jimenez, N. Quijano, L. Dewasme, and A. Vande Wouwer, “Agent-based model
predictive control of soil-crop irrigation with topographical information,” _Control_ _Engineer-_
_ing_ _Practice_, vol. 150, p. 106012, 2024.

[2] J. Lopez-Jimenez, L. Dewasme, A. Vande Wouwer, and N. Quijano, “Dynamic modeling of
crop-soil systems to design monitoring and automatic irrigation processes: A review with
worked examples,” _Water_, vol. 14, no. 15, p. 2404, 2022.

[3] R. G. Allen, L. S. Pereira, D. Raes, and M. Smith, “Crop evapotranspiration: Guidelines
for computing crop water requirements,” FAO Irrigation and Drainage Paper No. 56, 1998.

[4] A. Wächter and L. T. Biegler, “On the implementation of an interior-point filter line-search
algorithm for large-scale nonlinear programming,” _Mathematical_ _Programming_, vol. 106,
no. 1, pp. 25–57, 2006.

[5] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, “CasADi: A software
framework for nonlinear optimization and optimal control,” _Mathematical_ _Programming_
_Computation_, vol. 11, no. 1, pp. 1–36, 2019.

[6] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum
entropy deep reinforcement learning with a stochastic actor,” in _Proc._ _ICML_, vol. 80, pp.
1861–1870, 2018.

[7] T. Haarnoja, A. Zhou, K. Hartikainen, et al., “Soft actor-critic algorithms and applications,”
_arXiv_ _preprint_ _arXiv:1812.05905_, 2019.

[8] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “StableBaselines3: Reliable reinforcement learning implementations,” _JMLR_, vol. 22, no. 268,
pp. 1–8, 2021.

[9] H. Nouri et al., “Water management dilemma in the agricultural sector of Iran: A review
focusing on water governance,” _Agricultural_ _Water_ _Management_, vol. 278, p. 108162, 2023.

[10] M. B. Mesgaran and P. Azadi, “A national adaptation plan for water scarcity in Iran,”
Stanford Iran 2040 Project, Working Paper 6, 2018.

[11] N. Jalali et al., “Water requirement of Hashemi rice,” _Water_ _Res._ _Agric._, 2021.

[12] W. J. Rawls, D. Brakensiek, and K. E. Saxton, “Estimation of soil water properties,” _Trans-_
_actions_ _of_ _the_ _ASAE_, vol. 25, no. 5, pp. 1316–1320, 1982.

[13] P. Polcz et al., “Smart epidemic control: A hybrid model blending ODEs and agent-based
simulations for optimal, real-world intervention planning,” _PLoS_ _Computational_ _Biology_,
vol. 21, no. 5, p. e1013028, 2025.


24


