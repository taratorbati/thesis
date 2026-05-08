## **Ministry of Science and Higher Education of the Russian Federation** **Federal State Autonomous Educational Institution of Higher** **Education** **"National Research University ITMO"** **«Faculty of Control Systems and Robotics»** Field of study (specialty) 15.04.06 – Robotics and artificial intelligence **REPORT** on Research Internship Assignment topic: Development and Mathematical Modeling of a Topographical Virtual Plant for an Agricultural Field in Gilan, Iran Student: Tara Torbati, group R4237c Supervisor: Peregudin A.A. Saint Petersburg 2026


# **Contents**

**1** **Introduction** **2**


**2** **Study** **Site** **and** **Topological** **Graph** **3**
2.1 Digital Elevation Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.2 Directed Graph Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4


**3** **Climate** **Data** **and** **Scenarios** **6**
3.1 Reference Evapotranspiration . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
3.2 25-Year Climatology and Scenario Selection . . . . . . . . . . . . . . . . . . . . 8


**4** **Agent-Based** **Model** **Equations** **9**
4.1 Surface Hydrology and Runoff . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
4.2 Subsurface Hydrology and Infiltration . . . . . . . . . . . . . . . . . . . . . . . . 9
4.3 Stress Penalties and Crop Growth . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4.4 State Update Equations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11


**5** **Model** **Corrections** **and** **Extensions** **11**
5.1 Hydrological Decoupling and Mass Conservation . . . . . . . . . . . . . . . . . . 11
5.2 Surface Routing, Sinks, and Ponding Dynamics . . . . . . . . . . . . . . . . . . 11
5.3 Scale Justification: Omission of Subsurface Lateral Flow . . . . . . . . . . . . . 13
5.4 Agronomic Penalties and Crop Evapotranspiration . . . . . . . . . . . . . . . . . 13
5.5 Biomass Calibration and Initialization . . . . . . . . . . . . . . . . . . . . . . . 14


**6** **Soil** **and** **Crop** **Parameterization** **14**
6.1 Soil Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
6.2 Crop Parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
6.2.1 Rice (Hashemi Cultivar) . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
6.2.2 Tobacco ( _Nicotiana_ _tabacum_ ) . . . . . . . . . . . . . . . . . . . . . . . . 15
6.2.3 Parameter Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15


**7** **Water** **Budget** **and** **Scenario** **Selection** **16**
7.1 Water Budget Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
7.2 Scenario Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16


**8** **Model** **Validation** **17**
8.1 GWETROOT Cross-Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . 17


**9** **Conclusion** **19**


**A** **Plots** **21**


**B** **Codes** **21**


1


# **1 Introduction**

The integration of Internet of Things (IoT) technologies and artificial intelligence has initiated
a paradigm shift in modern precision agriculture, transforming traditional fields into complex
cyber-physical systems [1, 2]. Central to this transformation is the optimization of irrigation, a
critical requirement for mitigating regional water scarcity while maintaining agricultural productivity [3, 4]. While advanced optimal control algorithms like Model Predictive Control
(MPC) and Reinforcement Learning (RL) are highly capable of managing these constrained
multivariate systems, deploying them directly into physical environments is economically risky
and biologically prohibitive. Consequently, the synthesis and training of these intelligent controllers necessitate a high-fidelity virtual environment that accurately captures the complex,
non-linear dynamics of the soil-plant-atmosphere continuum.


This work presents the complete physical and mathematical description of an agricultural system designed based on a land in northwestern Iran. The objective is to construct a high-fidelity
simulation environment that serves a dual purpose: first, as the virtual plant on which the
Model Predictive Controller operates, and second, as the training environment for the Reinforcement Learning agent later developed.


The modeling framework builds upon the agent-based model (ABM) proposed by LópezJiménez et al. [5], which couples soil water dynamics with crop growth on a topographically
heterogeneous field. However, several critical modifications and extensions are introduced to
adapt the model to the specific agronomic conditions of rice cultivation in Gilan Province, Iran,
and to correct certain mass-balance inconsistencies identified in the original formulation.


The report is organized as follows. Section 2 describes the physical field site and the digital elevation model used to define the directed flow graph and agent grid. Section 3 details the 25-year
climate dataset, the reference evapotranspiration modeling, and the contrasting meteorological
scenarios. Section 4 presents the core discrete-time ABM equations governing the decoupled
surface and subsurface water balances, thermal accumulation, and crop growth. Section 5
formally documents the necessary mass-conservation corrections and agronomic extensions applied to the baseline framework. Section 6 parameterizes the environment for Hashemi rice
and tobacco. Section 7 derives the agronomically grounded seasonal water budgets. Finally,
Section 8 presents the ABM validation results, including cross-validation against satellite soil
wetness observations. Section 9 concludes the work.


2


# **2 Study Site and Topological Graph**

The study site is located on the slopes of the Talish Mountains in Gilan Province in northwest
Iran, at coordinates 38 _._ 298 _[◦]_ N, 48 _._ 847 _[◦]_ E. Situated along the southwestern coast of the Caspian
Sea, Gilan is Iran’s primary rice-producing province, accounting for over 238,000 hectares of
paddy fields [6]. The province experiences a humid subtropical climate with hot summers and
relatively mild winters, although seasonal precipitation can vary dramatically between years.
Despite its comparatively favorable water endowment relative to Iran’s central plateau, Gilan faces mounting pressures from declining aquifer levels and increasing competition between
agricultural, urban, and industrial water users [3, 7]. These conditions make this land a representative case study for evaluating constrained irrigation control strategies.


Figure 1: The satellite image of the study site captured using Google Earth

## **2.1 Digital Elevation Model**


The selected field covers approximately 6 hectares and is discretized into a 10 _×_ 13 rectangular
grid, yielding _N_ = 130 crop-soil agents. This resolution represents a practical balance between
spatial detail and computational tractability for the simulation environment: each agent corresponds to a plot of roughly 460 m [2], which is sufficiently fine to capture the field’s topographical
heterogeneity while remaining computationally feasible for the MPC optimization solved at
every daily time step.


The topographical data were obtained from the USGS Earth Explorer digital elevation database
which provides Shuttle Radar Topography Mission (SRTM) data at 30-meter resolution. The
raw elevation raster was processed in QGIS 3034 to extract the study region, reproject it to
UTM Zone 38N, and resample it to the 10 _×_ 13 agent grid. The resulting digital elevation
model (DEM) is stored as a GeoTIFF file and loaded programmatically by the simulation
environment.


3


Figure 2: Three-dimensional surface plot of the 130-agent grid. Elevation ranges from 70 m
to 181 m above sea level, forming a bowl-shaped topology with water convergence toward the
southeastern corner.


The elevation field exhibits a distinctive bowl-shaped topography, with elevations ranging from
70 m above sea level in the southeastern corner to to 181 m in the northwestern and southwestern
ridges. This 111 m elevation gradient creates significant lateral water redistribution through
surface and subsurface runoff: agents on higher ground shed excess water to their downhill
neighbors, while agents in the valley bottom accumulate inflows from multiple directions. This
topographical heterogeneity is a defining feature of the study site and the principal motivation
for adopting an agent-based rather than a lumped-parameter modeling approach.

## **2.2 Directed Graph Construction**



The topographical information is encoded in a directed graph that governs lateral water exchange between agents. Each agent occupies a cell in the 10 _×_ 13 grid and is assigned a
normalized elevation _γ_ [(] _[n]_ [)] _∈_ [0 _,_ 1], computed by min–max normalization of the raw DEM values:



_γ_ [(] _[n]_ [)] = _[z]_ [(] _[n]_ [)] _[ −]_ _[z]_ [min]



(1)
_z_ max _−_ _z_ min



where _z_ [(] _[n]_ [)] is the raw elevation of agent _n_ . An agent at position ( _r, c_ ) can exchange water
with up to eight neighbors in its Moore neighborhood (the eight surrounding cells). A directed
edge from agent _n_ to agent _m_ is created if and only if _γ_ [(] _[m]_ [)] _< γ_ [(] _[n]_ [)], so that water flows strictly
downhill. The number of lower-elevation neighbors for each agent is denoted _Nr_ [(] _[n]_ [)] .


4


Figure 3: Water flow directions between agents, overlaid on the normalized elevation map.
Arrows indicate the direction of lateral surface runoff, governed by the directed graph.


Figure 3 visualizes the resulting directed graph. The arrow density reflects the terrain gradient:
steep areas near the ridges exhibit dense, nearly parallel flow lines, while the flat valley floor
shows sparser, more diffuse connections. Agents in the southeastern corner with _Nr_ = 0 function
as hydrological sinks; they have no lower neighbors and therefore retain all accumulated water.
These sink agents are particularly susceptible to waterlogging stress, a phenomenon that the
ABM captures through the waterlogging stress function _h_ 6 described in Section 4.3.


Figure 4: Spatial distribution of the neighbor count _Nr_ for each agent. Red-bordered cells
indicate sink agents ( _Nr_ = 0) that accumulate runoff from uphill neighbors.


Three sink agents were identified, two located in the southeastern corner of the field at the
lowest elevations (Figure 4). This spatial concentration of sinks creates a natural drainage


5


basin structure that any irrigation controller must account for: over-irrigating uphill agents will
cause water to pool at these sinks, simultaneously wasting budget and inducing waterlogging
stress in the valley.

# **3 Climate Data and Scenarios**


Daily meteorological data were obtained from the NASA POWER (Prediction of Worldwide Energy Resource) database, which provides satellite-derived and reanalysis-based climate records
at a spatial resolution of 0 _._ 5 _×_ 0 _._ 625 (approximately 50 _×_ 60 km). The dataset was requested
for 25-year period of 2000–2025 at the field coordinates (38 _._ 30 _[◦]_ N, 48 _._ 85 _[◦]_ E). Thirteen variables
were extracted, including


  - mean, maximum, and minimum temperatures at 2 m (T2M, T2M_MAX, T2M_MIN,

_◦_ C),


  - dew/frost point at 2 m (T2MDEW, _◦_ C),


  - corrected daily precipitation (PRECTOTCORR, mm/day),


  - relative humidity (RH2M, %),


  - surface shortwave downward irradiance (ALLSKY_SFC_SW_DWN, MJ/m [2] /day),


  - wind speed at 2 m (WS2M, m/s),


  - surface pressure (PS, kPa), and


  - root and surface soil wetness (GWETROOT, GWETTOP, dimensionless, 0–1).


The last two variables were used for validation purposes rather than as model inputs.


The dataset was filtered to cover the agricultural season from April to October for each year,
yielding 5,564 daily records with no missing values. During preprocessing, one extreme rainfall
anomaly (375 mm recorded on April 7, 2022) was identified and replaced with the calendar-day
average of 1.2 mm to prevent anomalous skewing of the soil water balance. Moreover, based
on the available data, two reference evapotranspiration values were calculated and discussed in
the following subsection.

## **3.1 Reference Evapotranspiration**


Reference evapotranspiration (ET0) is the primary driver of crop water demand. Two estimation
methods were compared to determine the most accurate representation for the study site: the
temperature-based Hargreaves equation and the FAO Penman-Monteith equation [8].


**Hargreaves** **equation** :


ET _[H]_ 0 [= 0] _[.]_ [0023] _[ ·]_ [ (] _[T]_ [mean] [+ 17] _[.]_ [8)] _[ ·]_ [ (] _[T]_ [max] _[−]_ _[T]_ [min][)][0] _[.]_ [5] _[ ·][ R][a]_ (2)


6


where _T_ mean is the mean daily temperature ( _[◦]_ C), _T_ max and _T_ min are the daily maximum and
estimated minimum temperatures ( _[◦]_ C), and _Ra_ is the extraterrestrial radiation (expressed in
equivalent evaporation, mm/day).


**FAO** **Penman-Monteith** :


0 _._ 408 ∆( _Rn_ _−_ _G_ ) + _γ_ _T_ +273900 _[u]_ [2] [(] _[e][s]_ _[−]_ _[e][a]_ [)]
ET _[PM]_ 0 = ∆+ _γ_ (1 + 0 _._ 34 _u_ 2) (3)


where ∆ is the slope of the vapor pressure curve (kPa/ _[◦]_ C), _Rn_ is the net radiation at the crop
surface (MJ/m [2] /day), _G_ is the soil heat flux density (MJ/m [2] /day), _γ_ is the psychrometric
constant (kPa/ _[◦]_ C), _T_ is the mean daily air temperature at 2 m height ( _[◦]_ C), _u_ 2 is the wind
speed at 2 m height (m/s), _es_ is the saturation vapor pressure (kPa), _ea_ is the actual vapor
pressure (kPa), ( _es −_ _ea_ ) is the saturation vapor pressure deficit (kPa).


Over the 25-year rice season, the Hargreaves method yielded an average ET0 of 4.59 mm/day,
compared to 5.02 mm/day for the Penman-Monteith method ( _r_ = 0 _._ 940). Although the models
are strongly correlated, the Hargreaves equation systematically underestimates evaporative demand by approximately 8.6%. This underestimation occurs because the Hargreaves formulation
relies on the diurnal temperature range ( _T_ max _−_ _T_ min) as a proxy for atmospheric demand. In
Gilan’s coastal microclimate, the moderated temperature differential (∆ _T_ _≈_ 9 _._ 4 _[◦]_ C) artificially
attenuates the Hargreaves estimate. Therefore, the Penman-Monteith method, which explicitly
incorporates solar radiation, relative humidity, and aerodynamic conductances,


is selected as the standard to ensure high-fidelity calculation of the evaporative disturbances in
the control environment.


Figure 5: Comparison of Hargreaves and Penman-Monteith ET0. Left: seasonal pattern. Right:
scatter plot with regression.


7


## **3.2 25-Year Climatology and Scenario Selection**

The 25-year climatology reveals mean temperatures ranging from 12 _[◦]_ C in early April to 26 _[◦]_ C in
mid-July. Precipitation exhibits a dry mid-season between June and August (0.5–1.5 mm/day)
flanked by wetter months, while solar radiation peaks at 26 MJ/m [2] /day in June. Relative
humidity remains consistently between 60% and 75% throughout the season, which is a defining
feature of Gilan’s Caspian climate. Additional climate figures are provided in Appendix A.


Figure 6: 25-year daily average temperature (April–October, 2000–2025).


Figure 7: 25-year daily average rainfall (April–October, 2000–2025).


To evaluate the irrigation controllers under varying weather conditions, three representative
years were selected from the 25-year record to serve as distinct climate scenarios:


  - **Moderate** **Scenario** **(2020):** Represents baseline conditions.


  - **Dry** **Scenario** **(2022):** Evaluates controller performance under severe water constraints.


  - **Wet** **Scenario** **(2024):** Tests the controller’s ability to capitalize on natural rainfall and
mitigate waterlogging risks in low-elevation agents.


8


Table 1: Selected climate scenarios.

**Scenario** **Year** **Rice** **(mm)** **Tobacco** **(mm)**


Dry 2022 39.7 54.0
Moderate 2020 42.1 73.3
Wet 2024 176.8 163.2

# **4 Agent-Based Model Equations**


The crop-soil system is modeled as a set of _N_ = 130 coupled discrete-time dynamical systems.
Each agent _n_ _∈{_ 1 _, . . ., N_ _}_ maintains five state variables that evolve at a daily time step _k_ :
volumetric soil water content in the root zone ( _x_ 1, mm), cumulative thermal time ( _x_ 2, _[◦]_ C _·_ day),
cumulative maturation index ( _x_ 3), aboveground dry biomass ( _x_ 4, g/m [2] ), and surface ponding
depth ( _x_ 5, mm).


To accurately simulate lateral water redistribution while maintaining computational efficiency
for the optimal controller, the grid is solved sequentially at each time step _k_ following a topological sort of the directed elevation graph, executing from the highest elevation agents down
to the sinks. Furthermore, horizontal subsurface lateral flow is considered negligible for the
460 m [2] agent resolution; thus, agents are coupled exclusively via surface runoff.

## **4.1 Surface Hydrology and Runoff**



The total available surface water ( _W_ surf) for agent _n_ consists of yesterday’s un-infiltrated ponding ( _x_ 5), daily precipitation ( _P_ ), applied irrigation ( _u_ _[n]_ ), and incoming surface runoff routed
from all uphill neighbors _m_ _∈U_ ( _n_ ), divided equally among the neighbor’s downhill receivers
( _Nr_ _[m]_ [):]



_W_ surf _[n]_ [(] _[k]_ [) =] _[ x][n]_ 5 [(] _[k]_ [) +] _[ P]_ [(] _[k]_ [) +] _[ u][n]_ [(] _[k]_ [) +] 

_m∈U_ ( _n_ )



_φ_ _[m]_ 2 [(] _[k]_ [)]
(4)
_Nr_ _[m]_



Surface runoff ( _φ_ 2) is computed using the SCS curve number method. To enforce strict mass
conservation and simulate topographical valley flooding, a boundary condition is applied where
sink agents ( _Nr_ _[n]_ [= 0][)] [cannot] [generate] [runoff:]



_W_ surfsurf _[n]_ [(] _[k]_ [)+4][3] _[θ]_ [3] _[,]_ _Nr_ _[n]_ _[>]_ [ 0] [and] _[W]_ surf _[ n]_ [(] _[k]_ [)] _[ > θ]_ [3]



( _W_ surf _[n]_ [(] _[k]_ [)] _[−][θ]_ [3][)][2]



_φ_ _[n]_ 2 [(] _[k]_ [) =]








_W_ surf [(] _[k]_ [)+4] _[θ]_ [3] (5)

0 _,_ otherwise






## **4.2 Subsurface Hydrology and Infiltration**

Crop transpiration ( _φ_ 1) is the water extracted by plant roots, bounded by the specific crop
evapotranspiration (ET _c_ ), which is derived from the atmospheric reference demand (ET0) and
the crop coefficient ( _Kc_ ):
ET _c_ ( _k_ ) = _Kc ·_ ET0( _k_ ) (6)


9


             -             _φ_ _[n]_ 1 [(] _[k]_ [) = min] max� _θ_ 1( _x_ _[n]_ 1 [(] _[k]_ [)] _[ −]_ _[θ]_ [2] _[θ]_ [5][)] _[,]_ [0]        - _,_ ET _c_ ( _k_ ) (7)


Water that does not run off the surface infiltrates into the root zone. To prevent non-physical
infinite pooling within the soil matrix, infiltration ( _I_ ) is bounded by the physical saturation
capacity of the soil ( _θ_ sat _θ_ 5):


_I_ max _[n]_ [(] _[k]_ [) =] _[ θ]_ [sat] _[θ]_ [5] _[−]_ _[x][n]_ 1 [(] _[k]_ [) +] _[ φ][n]_ 1 [(] _[k]_ [)] (8)


_I_ _[n]_ ( _k_ ) = min� _W_ surf _[n]_ [(] _[k]_ [)] _[ −]_ _[φ][n]_ 2 [(] _[k]_ [)] _[,]_ _[I]_ max _[n]_ [(] _[k]_ [)]              - (9)


An intermediate soil moisture state ( _x_ 1 _,_ temp) is computed to evaluate the temporary water
holding before daily drainage occurs:


_x_ _[n]_ 1 _,_ temp [(] _[k]_ [) =] _[ x][n]_ 1 [(] _[k]_ [) +] _[ I]_ _[n]_ [(] _[k]_ [)] _[ −]_ _[φ][n]_ 1 [(] _[k]_ [)] (10)


If this temporary moisture exceeds the field capacity ( _θ_ 6 _θ_ 5), the subsurface excess ( _E_ sub) becomes vulnerable to gravity. A fraction of this excess, defined by the daily drainage rate ( _θ_ 4),
is lost to deep vertical drainage ( _φ_ 3):


_E_ sub _[n]_ [(] _[k]_ [) = max]         - _x_ _[n]_ 1 _,_ temp [(] _[k]_ [)] _[ −]_ _[θ]_ [6] _[θ]_ [5] _[,]_ [0]         - (11)


_φ_ _[n]_ 3 [(] _[k]_ [) =] _[ θ]_ [4] _[·][ E]_ sub _[n]_ [(] _[k]_ [)] (12)

## **4.3 Stress Penalties and Crop Growth**


Thermal time accumulation ( _h_ 1) drives phenological development:


_h_ _[n]_ 1 [(] _[k]_ [) = max]             - _T_ mean( _k_ ) _−_ _θ_ 7 _,_ 0� (13)


Heat stress ( _h_ 2) reduces maturation efficiency, and low-temperature stress ( _h_ 7) prevents biomass
accumulation:

            - _θ_ 10 _−_ _T_ max( _k_ )             _h_ 2( _k_ ) = clip _,_ 0 _,_ 1 (14)
_θ_ 10 _−_ _θ_ 9



_h_ 7( _k_ ) =








1 _,_ _T_ mean( _k_ ) _> θ_ 7


0 _,_ _T_ mean( _k_ ) _≤_ _θ_ 7



(15)
0 _,_ _T_ mean( _k_ ) _≤_ _θ_ 7



Hydrological stresses directly impact crop development. Drought stress ( _h_ 3) penalizes the
system when actual transpiration falls below the crop’s atmospheric demand (ET _c_ ), while waterlogging stress ( _h_ 6) penalizes root-zone oxygen deprivation when soil moisture exceeds field
capacity:

              -               1 [(] _[k]_ [)]
_h_ _[n]_ 3 [(] _[k]_ [) = 1] _[ −]_ _[θ]_ [14] _[·]_ [ max] 1 _−_ _[φ][n]_ [0] (16)
ET _c_ ( _k_ ) _[,]_


            -            1 [(] _[k]_ [)] _[ −]_ _[θ]_ [6] _[θ]_ [5]
_h_ _[n]_ 6 [(] _[k]_ [) =][ clip] 1 _−_ _[x][n]_ _,_ 0 _,_ 1 (17)

_θ_ 6 _θ_ 5


10


The growth function ( _g_ ) dictates the fraction of intercepted solar radiation contributing to
biomass, forming a two-branch sigmoid curve scaled by the maximum interception parameter
( _θ_ 19):



_g_ ( _x_ _[n]_ 2 [) =]








 _θ_ 19 _/_ (1 + _e_ _[−]_ [0] _[.]_ [01(] _[x]_ 2 _[n][−][θ]_ [20][)] ) _x_ _[n]_ 2 _[≤]_ _[θ]_ [18] _[/]_ [2]

 _θ_ 19 _/_ (1 + _e_ [0] _[.]_ [01(] _[x]_ 2 _[n]_ [+] _[x]_ 3 _[n][−][θ]_ [18][)] ) _x_ _[n]_ 2 _[> θ]_ [18] _[/]_ [2]



(18)
_θ_ 19 _/_ (1 + _e_ [0] _[.]_ [01(] _[x]_ 2 _[n]_ [+] _[x]_ 3 _[n][−][θ]_ [18][)] ) _x_ _[n]_ 2 _[> θ]_ [18] _[/]_ [2]


## **4.4 State Update Equations**

The coupled dynamics for agent _n_ conclude with the daily state updates. Any surface water
that failed to infiltrate remains explicitly conserved in the surface ponding state ( _x_ 5) for the
subsequent day:


_x_ _[n]_ 1 [(] _[k]_ [ + 1) =] _[ x][n]_ 1 _,_ temp [(] _[k]_ [)] _[ −]_ _[φ][n]_ 3 [(] _[k]_ [)] (19)

_x_ _[n]_ 2 [(] _[k]_ [ + 1) =] _[ x][n]_ 2 [(] _[k]_ [) +] _[ h][n]_ 1 [(] _[k]_ [)] (20)

_x_ _[n]_ 3 [(] _[k]_ [ + 1) =] _[ x][n]_ 3 [(] _[k]_ [) +] _[ θ]_ [11] �1 _−_ _h_ 2( _k_ )� + _θ_ 12�1 _−_ _h_ _[n]_ 3 [(] _[k]_ [)]       - (21)

_x_ _[n]_ 4 [(] _[k]_ [ + 1) =] _[ x][n]_ 4 [(] _[k]_ [) +] _[ θ]_ [13] _[·][ h][n]_ 3 [(] _[k]_ [)] _[ ·][ h][n]_ 6 [(] _[k]_ [)] _[ ·][ h]_ [7][(] _[k]_ [)] _[ ·][ g]_ [(] _[x][n]_ 2 [(] _[k]_ [))] _[ ·][ R][s]_ [(] _[k]_ [)] (22)

_x_ _[n]_ 5 [(] _[k]_ [ + 1) =] _[ W]_ surf _[ n]_ [(] _[k]_ [)] _[ −]_ _[φ][n]_ 2 [(] _[k]_ [)] _[ −]_ _[I]_ _[n]_ [(] _[k]_ [)] (23)

# **5 Model Corrections and Extensions**


While the baseline framework [5] provides a robust topographical architecture, implementation for closed-loop control revealed necessary mathematical corrections to enforce strict massbalance and to ensure controller cost-functions were properly coupled to agronomic penalties.
Five primary modifications were integrated into the environment.

## **5.1 Hydrological Decoupling and Mass Conservation**


The original formulation in [5] computes an outflow term ( _ϕ_ out) that lumps surface runoff and
subsurface excess together. While the runoff portion is properly subtracted from the sender’s
state equation, the subsurface excess portion is not. Consequently, a sending agent passes its
subsurface excess to downhill neighbors but never loses that volume from its own root zone. This
generates an unbounded artificial creation of water ("phantom water"), leading to exaggerated
flooding in lower elevations during sustained rainfall.


The proposed decoupled two-layer architecture resolves these flaws. By explicitly bridging
the surface layer ( _W_ surf) and the root zone ( _x_ 1) via a strictly bounded infiltration term ( _I_ ),
the overlapping fluxes are eliminated. Explicitly tracking the exact volume of water moving
between the decoupled layers guarantees strict mass conservation across the entire field.

## **5.2 Surface Routing, Sinks, and Ponding Dynamics**


In the original configuration, surface runoff ( _φ_ 2) is mathematically aggregated with subsurface
flow and routed directly into the soil matrix of adjacent agents. This simplification bypasses


11


surface pooling dynamics, injecting rapid overland flow directly into the root zone of downhill
neighbors.


To more accurately capture the bowl-shaped topography of the field, surface runoff is explicitly
decoupled from subsurface flow in this work. Runoff generated by an agent is routed to lowerelevation neighbors where it arrives as surface inflow. This inflow is added to the neighbor’s
available surface water ( _W_ surf) and must undergo infiltration before entering the root zone.


Crucially, a topological boundary condition was introduced to handle valley flooding. In the
baseline model, sink agents ( _Nr_ = 0) generated runoff that mathematically vanished. By
enforcing _φ_ 2 = 0 for all sinks, and by capturing un-infiltrated water in a new surface ponding
state variable ( _x_ 5), the model realistically simulates sustained, multi-day waterlogging in lowelevation zones.


To justify the extra computational cost of routing water across the elevation map, the cascade
model was compared against a simpler single-step baseline where each grid cell acts independently. Both models were tested under the wet year scenario (2024) with three different
irrigation levels (0 mm, 5 mm, and 8 mm). While the simpler model produces similar fieldaverage yields, the cascade mode reveals substantially different sink agent behavior (Figure 8),
particularly under irrigation. Because the MPC controller must predict and avoid waterlogging
at sink agents, the cascade mode was adopted to ensure the prediction model accurately reflects
within-day runoff accumulation.


Figure 8: Hydrological comparison of runoff modes across 0, 5, and 8 mm/day irrigation levels
(2024)


12


## **5.3 Scale Justification: Omission of Subsurface Lateral Flow**

The baseline model processes horizontal subsurface lateral flow at the same discrete daily time
step as rapid surface runoff. However, given the 460 m [2] spatial resolution of the agents (approximately 21 _×_ 21 m plots), lateral water movement through dense silty loam over a single
day is physically negligible compared to vertical deep drainage and overland flow.


Therefore, horizontal subsurface flow ( _φ_ 4) was omitted from the proposed framework. The root
zones are modeled as independent vertical columns coupled exclusively by the surface runoff
routing graph. This physical simplification significantly increases the sparsity of the Jacobian matrices, thereby reducing the computational complexity and accelerating the prediction
horizon of the MPC.

## **5.4 Agronomic Penalties and Crop Evapotranspiration**


In the original framework, transpiration and drought stress were bounded by the reference
evapotranspiration (ET0), effectively treating all crops equivalently to a standard 12 cm reference grass. This formulation was corrected by introducing crop coefficients ( _Kc_ ) to calculate
crop-specific evapotranspiration (ET _c_ ), ensuring Hashemi rice and tobacco respect their distinct
biological water demands.


Furthermore, in the original model, the drought stress function _h_ 3 appears only in the maturation equation ( _x_ 3), where it accelerates crop senescence under water deficit. However, _h_ 3 is
absent from the biomass equation ( _x_ 4). This means that a completely unirrigated crop would
accumulate the same daily biomass increment as a fully irrigated one, provided all other conditions were equal. The only effect of drought would be to advance the onset of senescence,
which reduces biomass accumulation only indirectly and with considerable delay.


This omission was identified during the initial ABM validation, when the no-irrigation baseline
produced biomass values that were nearly indistinguishable from irrigated scenarios for wheat.
The corrected formulation multiplies the radiation use efficiency term ( _θ_ 13) in the biomass state
update by _h_ 3, so that drought directly reduces the daily biomass increment. This modification
follows the approach of the FAO AquaCrop model, in which water stress reduces radiation use
efficiency proportionally to the ratio of actual to potential transpiration. With _θ_ 14 = 0 _._ 8 for
rice and 0.6 for tobacco, maximum drought reduces the biomass increment by up to 80% and
60%, making irrigation decisions directly consequential for yield outcomes.


This correction is particularly important for the controller comparison objective of this thesis.
Without it, the ABM would produce only marginal differences between irrigated and unirrigated
outcomes, rendering any comparison between MPC, RL, and baseline controllers meaningless.
The original crop used in the baseline paper was wheat in Colombia, which is less watersensitive (lower _θ_ 14); it is possible that the omission went undetected in that context because
the irrigation response was already small.


13


## **5.5 Biomass Calibration and Initialization**

The radiation use efficiency parameter _θ_ 13 was calibrated so that _x_ 4 reliably represents g/m [2] of
aboveground dry matter. The final agricultural yield is calculated as _Y_ = _x_ 4 _×_ HI _×_ 10 (kg/ha).


For Hashemi rice, the simulated yields were validated against typical Gilan farmer reports
( _∼_ 2000 kg/ha) and optimal experimental conditions (3500–4000 kg/ha) [9]. The initial state
conditions reflecting the 20-day nursery period for rice are set to _x_ 2 _,_ init = 210 and _x_ 4 _,_ init = 60.
Conversely, tobacco seedlings are initialized at _x_ 2 _,_ init = 0 and _x_ 4 _,_ init = 8.

# **6 Soil and Crop Parameterization**

## **6.1 Soil Parameters**


The soil at the study site is classified as silty loam, based on regional soil surveys and FAO soil
classification maps for Gilan Province. The soil hydraulic properties used in this study follow
FAO Irrigation and Drainage Paper No. 56 recommendations for this soil type.


Table 2: Soil hydraulic parameters for silty loam (common to all crops)


**Parameter** **Symbol** **Value** **Source**


Water uptake coefficient _θ_ 1 0.096 [5]
Wilting point (volumetric fraction) _θ_ 2 0.15 [8]
SCS threshold (mm) _θ_ 3 10.0 [5]
Drainage coefficient _θ_ 4 0.05 [5]
Field capacity (volumetric fraction) _θ_ 6 0.35 [8]
Saturation (volumetric fraction) _θsat_ 0.50 [10]

## **6.2 Crop Parameters**


Two crops were selected for the simulation environment: rice (Hashemi cultivar) and tobacco
( _Nicotiana_ _tabacum_ ). Both hold significant commercial value in Gilan Province [6, 11]. Their
contrasting agronomic characteristics (high water demand, shallow roots, and high drought
sensitivity for rice, versus moderate water demand, deep roots, and relative drought tolerance
for tobacco) provide a robust basis for evaluating the control algorithms under differing physical
constraints.


**6.2.1** **Rice** **(Hashemi** **Cultivar)**


The base temperature ( _T_ base or _θ_ 7) is set to 10 _[◦]_ C, following standard Growing Degree Day
(GDD) conventions [12, 13] and the baseline ABM [5]. The maturity threshold ( _θ_ 18) was
calibrated to 1250 _[◦]_ C _·_ day based on the 25-year dataset and validated against local field measurements, which report 1216–1352 GDD for the Hashemi cultivar in Gilan [14].


Transplanting is simulated on May 20 (DOY 141), aligning with local practices when the 7day rolling average _T_ mean _≥_ 18 _[◦]_ C and _T_ min _≥_ 12 _[◦]_ C [15, 16]. The harvest is set for August
20 (DOY 233). Extending the season beyond this date sharply increases the risk of extreme


14


WMO R20mm rainfall events ( _≥_ 20 mm/day), which can cause crop lodging and severe harvest
disruption [17, 18].


This yields a 93-day field season. As established in Section 5, the 20-day prior nursery period
contributes an initial thermal accumulation of _x_ 2 _,_ init = 210 GDD and an initial biomass of
_x_ 4 _,_ init = 60 g/m [2] . Other key parameters include a crop coefficient ( _Kc_ ) of 1.15, a shallow root
depth ( _θ_ 5) of 400 mm, and a high drought sensitivity ( _θ_ 14) of 0.80.


**6.2.2** **Tobacco** **(** _**Nicotiana**_ _**tabacum**_ **)**


The base temperature is similarly set to 10 _[◦]_ C, which falls within the standard 10–13 _[◦]_ C restriction range for tobacco [19]. The maturity threshold ( _θ_ 18) is calibrated to 1200 GDD. This
duration corresponds to a 104-day field season, consistent with standard FAO timelines (90–120
days) [20] and general agronomic data (100–130 days) [21]. As tobacco seedlings are typically
delivered directly from nurseries with unknown prior conditions, the initial states are set to
_x_ 2 _,_ init = 0 and a minimal _x_ 4 _,_ init = 8 g/m [2] .


Transplanting occurs on May 25 (DOY 146), with harvest on September 5 (DOY 249). Tobacco
features a significantly deeper root zone ( _θ_ 5 = 700 mm) compared to rice, allowing it to access
a larger reservoir of deep soil moisture. It also exhibits a lower crop coefficient ( _Kc_ = 0 _._ 90)
and lower drought sensitivity ( _θ_ 14 = 0 _._ 60), making it more resilient to constrained irrigation
schedules.


**6.2.3** **Parameter** **Summary**


Table 3: Soil and crop-specifc parameters used in the ABM simulation


**Parameter** **Symbol** **Rice** **Tobacco** **Source**


Root depth _θ_ 5 400 mm 700 mm [8], [20]
Base temperature _θ_ 7 10 _[◦]_ C 10 _[◦]_ C [13], [19]
Heat stress onset _θ_ 9 35 _[◦]_ C 30 _[◦]_ C [20]
Extreme heat _θ_ 10 42 _[◦]_ C 38 _[◦]_ C      Heat stress maturation rate _θ_ 11 0.0030 0.0025 [5]
Drought stress maturation rate _θ_ 12 0.0030 0.0028 [5]
RUE _θ_ 13 0.65 0.30 [22], [23]
Drought sensitivity _θ_ 14 0.80 0.60 [5]
Maturity GDD _θ_ 18 1250 1200 [14], calib.
Max interception _θ_ 19 0.95 0.90 [5]
50% intercept. GDD _θ_ 20 417 450 [24]
Crop coefficient _Kc_ 1.15 0.90 [8], [20]
Depletion fraction _p_ 0.20 0.50 [8], [20]

Harvest index HI 0.42 0.55 [9], [20]

Nursery GDD _x_ 2 _,_ init 210 0 Calib.
Initial biomass _x_ 4 _,_ init 60 g/m [2] 8 g/m [2] Est.


15


# **7 Water Budget and Scenario Selection**

## **7.1 Water Budget Formulation**

To establish the baseline irrigation requirements for both crops, a seasonal water budget was
formulated using the 25-year climatological averages. The total crop water demand (ET _c_ ) is
calculated as the product of the average reference evapotranspiration, the crop coefficient, and
the season duration:
ET _c_ = ET _[PM]_ 0 _· Kc · N_ days (24)


The full irrigation requirement ( _I_ full) represents the water deficit that must be met by the
controller, computed by subtracting the 25-year average seasonal rainfall ( _P_ ) from the crop
demand:
_I_ full = max�ET _c −_ _P,_ 0� (25)


Additionally, constrained irrigation budgets representing 70% and 50% of the full requirement
were calculated to establish deficit scenarios for evaluating controller robustness under strict
water rationing.


Table 4: Agronomic water budget summary based on 25-year climatology.


**Parameter** **Rice** **Tobacco**


Season duration ( _N_ days) 93 days 104 days
Average ET0 (PM) 5.02 mm/day 4.95 mm/day
Total crop demand (ET _c_ ) 537.0 mm 463.4 mm
Average season rainfall ( _P_ ) 53.3 mm 74.5 mm
Full irrigation need ( _I_ full) 483.7 mm 389.0 mm
Deficit budget (70% of _I_ full) 338.6 mm 272.3 mm
Deficit budget (50% of _I_ full) 241.8 mm 194.5 mm


The calculated rice water demand of 537.0 mm aligns strongly with local Hashemi field measurements, which range from 534 mm under System of Rice Intensification (SRI) management
to 632 mm under traditional continuous flooding [9]. Similarly, the calculated tobacco demand
of 463.4 mm falls perfectly within the expected range of 400–600 mm [20]. Notably, despite
having a longer field season, tobacco requires less total irrigation than rice. This is due to
its lower crop coefficient ( _Kc_ ) and the higher average rainfall that occurs during its extended
late-summer season.

## **7.2 Scenario Selection**


To evaluate controller performance under varying climatic extremes, three specific years were
selected from the 25-year dataset to serve as distinct testing scenarios: Dry, Moderate, and Wet.
These scenarios are defined by their total seasonal rainfall relative to the historical average.


16


Table 5: Selected climate scenarios for controller evaluation.

**Scenario** **Year** **Rice** **Rainfall** **(mm)** **Tobacco** **Rainfall** **(mm)**


Dry 2022 39.7 54.0
Moderate 2020 42.1 73.3
Wet 2024 176.8 163.2

# **8 Model Validation**

## **8.1 GWETROOT Cross-Validation**


To verify the physical fidelity of the Agent-Based Model’s hydrological dynamics, the simulated
soil water state ( _x_ 1) was cross-validated against independent satellite observations. Specifically,
the ABM output was compared to the NASA MERRA-2 Root Zone Soil Wetness (GWETROOT) product for the three selected scenario years.


Because the two systems employ different scales and physical boundary conditions, direct absolute comparison is not feasible. Instead, the data were normalized to evaluate the agreement
in temporal wetting and drying patterns. The ABM soil water was normalized to a fractional
saturation index using the permanent wilting point (WP) and saturation (SAT):


_x_ ˆ1( _k_ ) = _[x]_ SAT [1][(] _[k]_ [)] _−_ _[ −]_ WP [WP] (26)


The GWETROOT data is natively provided as a dimensionless 0–1 fraction and was used
without modification. Pearson correlation coefficients were computed to assess the temporal
alignment between the simulated and observed series.


Table 6: Cross-validation of ABM soil moisture against NASA GWETROOT.


**Scenario** **Year** **Pearson** _r_ _p_ **-value**


Dry 2022 0.738 3 _._ 1 _×_ 10 _[−]_ [17]

Moderate 2020 0.534 3 _._ 4 _×_ 10 _[−]_ [8]

Wet 2024 0.338 9 _._ 2 _×_ 10 _[−]_ [4]


17


Figure 9: Normalized soil moisture time series: ABM (blue) vs. NASA GWETROOT (brown).


As shown in Table 6, the dry-year scenario exhibits a strong positive correlation ( _r_ = 0 _._ 738,
_p <_ 0 _._ 001), confirming that the ABM accurately captures fundamental soil drying and recharge
dynamics driven by evapotranspiration and intermittent rainfall. The correlation predictably
decreases in the moderate and wet years. This attenuation is expected because frequent rainfall
events saturate the soil, reducing the signal contrast and making the correlation metric highly
sensitive to minor timing discrepancies in the meteorological data.


While the temporal patterns align, there is a consistent offset in the absolute levels; the normalized ABM baseline fluctuates near 0.07, whereas the GWETROOT average sits higher at
approximately 0.47. This discrepancy is attributable to three distinct structural differences
between the datasets:


1. **Observation** **Depth:** NASA MERRA-2 models a deep root zone of roughly 1000 mm,
which retains moisture significantly longer than the ABM’s shallow 400 mm rice root
zone.


2. **Irrigation** **Status:** The GWETROOT satellite product observes actual field conditions


18


in Gilan, which includes widespread anthropogenic irrigation. Conversely, this specific
ABM validation run simulates an unirrigated, rain-fed control scenario to establish a
zero-control baseline.


3. **Soil** **Parameterization:** The standardized MERRA-2 global soil grid relies on generalized hydraulic parameters that differ from the highly specific silty loam parameters
calibrated for this local model.

# **9 Conclusion**


This chapter established a high-fidelity, mathematically rigorous virtual environment for simulating agricultural dynamics in Gilan Province. By building upon an existing topographical
agent-based framework, several critical extensions were introduced to ensure the model’s suitability for advanced control engineering applications.


Most notably, the hydrological dynamics were decoupled into distinct surface and subsurface
layers. This architectural correction eliminated mass-balance violations present in the baseline
literature, ensuring strict conservation of water fluxes and physically realistic surface pooling.
Furthermore, the direct integration of drought stress penalties into the biomass accumulation
equation established a necessary, responsive cost function, ensuring that simulated crop yields
dynamically reflect the success or failure of applied irrigation strategies.


The environment was successfully parameterized for two commercially significant local crops,
the highly water, sensitive Hashemi rice and the more resilient tobacco, using a 25-year localized
climatology. Finally, the baseline temporal dynamics of the model were successfully crossvalidated against independent NASA MERRA-2 satellite observations.


With the physical, biological, and mathematical boundaries of the virtual plant now validated,
this environment satisfies its dual objective. It serves both as the dynamic plant for the Model
Predictive Controller and as the episodic training environment for the Reinforcement Learning
agent. The subsequent works will detail the formulation, training, and comparative evaluation
of these control algorithms across the established dry, moderate, and wet climatic scenarios.


19


# **References**


[1] J. Bayar, N. Ali, Z. Cao, Y. Ren, and Y. Dong, “Artificial intelligence of things (aiot) for
precision agriculture: applications in smart irrigation, nutrient and disease management,”
_Smart_ _Agricultural_ _Technology_, vol. 12, p. 101629, 2025.


[2] M. Pourgholam-Amiji, I. Hajirad, K. Ahmadaali, and A. Liaghat, “Smart irrigation based
on the iot,” _Iranian_ _Journal_ _of_ _Soil_ _and_ _Water_ _Research_, vol. 55, no. 9, pp. 1647–1678,
2024.


[3] H. Nouri _et_ _al._, “Water management dilemma in the agricultural sector of iran: A review
focusing on water governance,” _Agricultural_ _Water_ _Management_, vol. 278, p. 108162, 2023.


[4] S. Maniam, Y.-K. Tee, E. Memar, H. Y. Wong, and M. Zaman, “Smart irrigation management: Iot-based rnn-lstm model for soil moisture prediction in precision agriculture,”
_Smart_ _Agricultural_ _Technology_, vol. 13, p. 101866, 2026.


[5] J. Lopez-Jimenez, N. Quijano, L. Dewasme, and A. Vande Wouwer, “Agent-based model
predictive control of soil-crop irrigation with topographical information,” _Control_ _Engi-_
_neering_ _Practice_, vol. 150, p. 106012, 2024.


[6] SurfIran, “Gilan’s rice fields,” 2025. [Online]. Available: https://surfiran.com/mag/ricetransplantation-in-gilan-province/


[7] M. B. Mesgaran and P. Azadi, “A national adaptation plan for water scarcity in iran,”
Stanford Iran 2040 Project, Working Paper 6, 2018.


[8] R. G. Allen _et_ _al._, “Crop evapotranspiration,” FAO, Irrig. Drain. Paper 56, 1998.


[9] N. Jalali _et_ _al._, “Water requirement of hashemi rice,” _Water_ _Res._ _Agric._, 2021.


[10] W. J. Rawls, D. Brakensiek, and K. E. Saxton, “Estimation of soil water properties,”
_Transactions_ _of_ _the_ _ASAE_, vol. 25, no. 5, pp. 1316–1320, 1982.


[11] “Tobacco,” Encyclopaedia Iranica. [Online]. Available:
https://www.iranicaonline.org/articles/tobacco/


[12] G. S. McMaster and W. W. Wilhelm, “Growing degree-days,” _Agric. For. Meteorol._, vol. 87,
pp. 291–300, 1997.


[13] S. Laohasiriwong _et_ _al._, “High temperature alters phenology in rice,” _Plants_, vol. 12, no. 3,
p. 666, 2023.


[14] S. M. T. Sadidi Shal, M. J. Zohd Ghodsi, E. Asadi Oskouei, and A. D. Zahra, “Comparison
of growing degree day of different phenological stages of hashemi rice in guilan province,”
_Journal_ _of_ _Climate_ _Research_, vol. 1400, no. 45, pp. 143–152, 2021.


[15] P. Paredes _et_ _al._, “Base and upper temperature thresholds for gdd,” _Agric._ _Water_ _Manag._,
vol. 319, 2025.


20


[16] R. Faghani _et al._, “Planting date and seedling age effects on rice yield,” _Afr. J. Biotechnol._,
2011.


[17] A. A. Sarma _et_ _al._, “Adaptation of wetland rice to extreme weather,” 2018.


[18] S. Nayak _et_ _al._, “Extreme rainfall indices and rice productivity,” 2011.


[19] Y. Y. Li _et_ _al._, “Cold stress effects on tobacco,” _BMC_ _Plant_ _Biol._, vol. 21, p. 131, 2021.


[20] FAO, “Crop information: Tobacco.” [Online]. Available: https://www.fao.org/landwater/databases-and-software/crop-information/tobacco/en/


[21] “Tobacco,” Encyclopaedia Britannica, 1999.


[22] S. R. Patel _et_ _al._, “Rue in irrigated rice,” _Paddy_ _Water_ _Environ._, vol. 11, pp. 477–486,
2013.


[23] R. Ferrara, “Rue in flue-cured tobacco,” _Field_ _Crops_ _Res._, 2002.


[24] Y. Zhang _et_ _al._, “Lai simulation of tobacco,” _Chin._ _J._ _Eco-Agric._, 2017.

# **Appendix A Plots**


Your content for Appendix A goes here.

# **Appendix B Codes**


Your content for Appendix B goes here.


21


