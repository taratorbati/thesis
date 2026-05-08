### **ITMO University** **Faculty of Control Systems and Robotics** Report on course work « A Review of Modern Irrigation Methods in the Context of Iran’s Water Scarcity Crisis» for the subject «Challenges and Approaches of Modern Robotics» Student: Tara Torbati Group: R4237c Professor: Peregudin A. A. Saint Petersburg 2026


# Table of Contents

### 1. Introduction ............................................................................................................ 4 1.1 The Water Scarcity Challenge .......................................................................... 4 1.2 Limitations of Conventional Approaches ......................................................... 4 1.3 Toward Data-Driven and Controlled Irrigation Systems ................................. 4 1.4 Motivation for Advanced Control and Learning Approaches .......................... 5 1.5 Scope and Purpose of This Work...................................................................... 5 2. The Macro Crisis: Anthropogenic Drought in Iran ................................................ 7 2.1 Climatic Trends and Human Drivers ................................................................ 7 2.2 Infrastructure Expansion and System Imbalance ............................................. 7 2.3 Environmental and Socioeconomic Consequences .......................................... 8 2.4 Implications for Water Management ................................................................ 9 3. Governance Gaps and the Limits of Efficiency-Based Approaches ....................10 3.1 Conflicting Objectives in Water and Agricultural Policy ...............................10 3.2 The Rebound Effect and System-Level Inefficiency .....................................10 3.3 Limitations of Conventional Policy Instruments ........................................... 11 3.4 Implications for Irrigation Management ........................................................12 3.5 Irrigation as a Constrained Control Problem ..................................................12 4. Modern and Smart Irrigation Technologies: A Cyber-Physical Control Perspective ...............................................................................................................14 4.1 The Transition to Precision and Data-Driven Irrigation .................................14 4.2 System Architecture and the Role of Feedback..............................................14


### 4.3 From Monitoring to Decision-Making: The Role of Data .............................15 4.4 Limitations of Conventional Control Strategies .............................................15 4.5 Toward Predictive and Optimal Control .........................................................16 4.6 The Need for Systematic Evaluation ..............................................................17 5. Feasibility and Impact Analysis for Iran ..............................................................19 5.1 Potential Impact of Optimized Irrigation Strategies ......................................19 5.2 Structural and Operational Constraints ..........................................................19 5.3 Uncertainty, Risk, and the Need for Pre-Deployment Evaluation .................20 5.4 Bridging Simulation and Real-World Deployment ........................................20 5.5 Implications for Method Selection .................................................................21 6. References ............................................................................................................22


# 1. Introduction

## 1.1 The Water Scarcity Challenge

Water scarcity has emerged as one of the most critical environmental and socioeconomic
challenges of the 21st century, particularly in arid and semi-arid regions. Iran represents a
prominent case of this crisis, where long-term declines in renewable water resources are coupled
with rising temperatures and increasingly variable precipitation patterns [2].

While climatic factors contribute to this trend, the severity of water scarcity in Iran is primarily
driven by unsustainable patterns of consumption. The agricultural sector accounts for
approximately 90% of total freshwater withdrawals [3], operating within a geographical context
where only a limited portion of land is naturally suited for intensive cultivation. To sustain
production, agricultural practices have become heavily dependent on groundwater extraction,
often exceeding natural recharge rates [1].

The result is a progressive depletion of aquifers, manifesting in declining water tables, land
subsidence, and the degradation of ecosystems. These trends indicate that the current trajectory
of water use is not only inefficient but structurally unsustainable.

## 1.2 Limitations of Conventional Approaches


Efforts to address agricultural water use have traditionally focused on improving irrigation
efficiency, primarily through the adoption of modern irrigation technologies such as sprinkler
and drip systems. While these approaches enhance field-level efficiency, they have not
consistently resulted in reductions in overall water consumption.

This apparent contradiction is largely explained by the rebound effect, where gains in efficiency
lead to behavioral and economic adjustments—such as expansion of cultivated land or shifts to
more water-intensive crops—that offset potential water savings. As a result, improvements in
irrigation technology alone are insufficient to address water scarcity at the system level [4].

More broadly, irrigation practices are often governed by heuristic or experience-based decisionmaking, which lacks the capacity to systematically account for environmental variability, longterm resource constraints, and complex system dynamics. This limits the effectiveness of both
traditional and technologically enhanced irrigation strategies.

## 1.3 Toward Data-Driven and Controlled Irrigation Systems


The increasing availability of environmental sensing, data acquisition, and computational tools
has enabled a transition toward data-driven irrigation systems, where water application is
adjusted dynamically based on observed conditions. These systems form the basis of modern


precision agriculture and introduce the possibility of managing irrigation as a feedback-driven
control process.

However, the effectiveness of such systems depends not only on the availability of data, but on
the decision-making methodologies used to translate this information into control actions.
Determining when and how much water to apply requires balancing competing objectives, such
as maximizing crop yield and minimizing water use, under uncertainty and strict resource
constraints.

This shifts the focus from hardware improvements to the design of control strategies capable of
operating within complex, dynamic environments.

## 1.4 Motivation for Advanced Control and Learning Approaches


Formulating irrigation as a control problem highlights the need for methods that can explicitly
incorporate system dynamics, anticipate future conditions, and operate under constraints such as
limited water availability. Conventional control strategies, including fixed schedules and simple
threshold-based rules, are not sufficient to address these requirements.

Advanced methodologies from control theory and artificial intelligence offer promising
alternatives. In particular:

  - Model-based approaches, such as Model Predictive Control (MPC), provide a framework
for optimizing control actions over a future horizon while enforcing constraints.

  - Data-driven approaches, such as Reinforcement Learning (RL), enable adaptive policy
learning in complex and uncertain environments without requiring explicit system
models.

Both approaches have been explored in the context of irrigation, yet their relative performance
and suitability for water-constrained agricultural systems remain dependent on environmental
conditions, system assumptions, and implementation constraints.

## 1.5 Scope and Purpose of This Work


This document provides the conceptual and contextual foundation for a broader investigation
into irrigation control strategies. It examines the drivers of water scarcity in Iran, the limitations
of existing irrigation practices, and the evolution toward data-driven and automated systems.

Building on this foundation, the work motivates the need for systematic evaluation of advanced
control methodologies under realistic agricultural conditions. In particular, it supports a
comparative analysis of model-based and learning-based approaches to irrigation control, with
the goal of assessing their effectiveness in managing water use under strict constraints.

The subsequent sections develop this perspective by first analyzing the macro-level dynamics of
water scarcity, then examining governance and technological factors influencing irrigation, and


finally establishing the basis for evaluating advanced control strategies in a structured and
controlled setting.


# 2. The Macro Crisis: Anthropogenic Drought in Iran

## 2.1 Climatic Trends and Human Drivers

Iran is located in an arid and semi-arid climatic zone characterized by high evaporation rates and
variable precipitation. Over recent decades, the country has experienced a gradual increase in
average temperatures and modest declines in precipitation. While these trends contribute to
increasing water stress, they do not fully explain the magnitude of observed water depletion.

A more comprehensive explanation emerges when examining total water storage trends.
Satellite-based observations and long-term hydrological data indicate a substantial decline in
groundwater reserves that cannot be attributed solely to meteorological variability [2]. Instead,
the evidence points to a form of anthropogenic drought, where water scarcity is primarily driven
by human activities rather than natural climatic deficits [1], [3].

The dominant factor in this process is the sustained over-extraction of groundwater for
agricultural use. Over multiple decades, extraction rates have consistently exceeded natural
recharge, leading to a cumulative depletion of aquifer systems. This dynamic reframes the water
crisis as a problem of resource management and consumption, rather than purely environmental
change.


Figure 1: Nationwide decline in Total Water Storage (TWS) anomalies in Iran observed via

GRACE satellite data, indicating severe anthropogenic drought. Adapted from [2].


## 2.2 Infrastructure Expansion and System Imbalance

Historically, water management strategies in Iran have emphasized increasing supply through
large-scale infrastructure development. This approach, often described as a supply-oriented
paradigm, has included extensive dam construction and inter-basin water transfer projects

[1], [3].

While such measures temporarily increased water availability, they also introduced significant
system-level imbalances. Reservoir expansion altered natural flow regimes, increased
evaporation losses, and reduced water availability for downstream ecosystems. At the same time,
agricultural expansion continued to drive demand, particularly through increased groundwater
extraction.


The proliferation of wells, many of which operate without effective regulation, has created a
highly decentralized and difficult-to-monitor extraction system. In combination with subsidized
energy for pumping, this has removed key economic constraints on groundwater use [3]. The
result is a system in which water withdrawal is weakly regulated and largely disconnected from
long-term resource availability.

This combination of supply expansion and unregulated demand has led to widespread depletion
of groundwater reserves across multiple regions.


Figure 2: The infrastructure paradox in Iran, illustrating the simultaneous expansion of dam

capacity and aggressive groundwater withdrawal from 1990 to 2015. Adapted from [1].


## 2.3 Environmental and Socioeconomic Consequences

The sustained imbalance between water extraction and natural recharge has produced a range of
environmental and socioeconomic impacts.

Environmentally, reduced surface flows and declining groundwater levels have contributed to the
degradation of wetlands, lakes, and river systems. In some cases, the loss of these water bodies
has led to secondary effects such as soil degradation and the emergence of dust and salinityrelated issues [1], [3].

At the subsurface level, excessive groundwater withdrawal has caused land subsidence in several
regions. This process reflects the irreversible compaction of aquifer systems, resulting in longterm loss of storage capacity and damage to infrastructure.

From a socioeconomic perspective, declining water availability directly affects agricultural
productivity and rural livelihoods. As water resources become increasingly scarce or saline,
farming becomes less viable, contributing to economic instability and migration pressures. In
parallel, competition for limited water resources has intensified across sectors, including
agriculture, industry, and urban use.

## 2.4 Implications for Water Management


The macro-level dynamics of water scarcity in Iran highlight a fundamental limitation of existing
management approaches. Systems driven by decentralized decision-making, short-term
incentives, and weak enforcement mechanisms have proven unable to regulate resource use
within sustainable limits.

This reinforces the need for approaches that can operate under explicit constraints, account for
system dynamics, and respond to environmental variability. In the context of agriculture, this
implies moving beyond purely infrastructural or efficiency-based solutions toward methods that
optimize water use within defined resource boundaries.

Such considerations provide the foundation for viewing irrigation not only as an agricultural
practice, but as a constrained and dynamic system requiring systematic analysis and control.


# 3. Governance Gaps and the Limits of Efficiency-Based Approaches

## 3.1 Conflicting Objectives in Water and Agricultural Policy

A central driver of unsustainable water use in Iran is the misalignment between national policy
objectives related to water management and agricultural production. On one hand, water
authorities are tasked with preserving limited water resources; on the other, agricultural policy
has historically prioritized increasing domestic production under the objective of food selfsufficiency [3].

While this objective aims to reduce reliance on external food sources, it often leads to the
expansion of agriculture into regions that are not naturally suited for water-intensive cultivation.
In practice, this has resulted in the widespread production of crops that require substantial
irrigation, regardless of local water availability.

This policy environment is further reinforced by economic incentives, particularly subsidies on
energy used for groundwater extraction [1], [3]. By lowering the cost of pumping, these subsidies
weaken the feedback between resource scarcity and consumption, allowing water use to remain
high even as aquifer levels decline.

The result is a system in which short-term production goals are prioritized over long-term
resource sustainability, contributing directly to the over-extraction of groundwater.

## 3.2 The Rebound Effect and System-Level Inefficiency


In response to increasing water stress, efforts have frequently focused on improving irrigation
efficiency through the adoption of modern technologies such as sprinkler and drip systems.
These technologies enhance field-level efficiency, ensuring that a greater proportion of applied
water is utilized by crops.

However, improvements at the field level do not necessarily translate into reduced water
consumption at the system level. This is largely due to the rebound effect, where efficiency gains
alter user behavior in ways that offset potential savings [4].


In agricultural settings, water that is not consumed by crops in traditional systems is often not
lost entirely; a portion returns to the broader hydrological system through percolation or runoff

[4]. When more efficient irrigation technologies reduce these return flows, the apparent
“savings” may not represent actual reductions in total water use [4].

More importantly, farmers may respond to increased efficiency by expanding cultivated area,
increasing cropping intensity, or shifting to more water-intensive crops. These adjustments
increase total water consumption, even as per-unit efficiency improves.


This dynamic highlights a key limitation of efficiency-based approaches: without mechanisms to
constrain total water use, technological improvements alone may fail to achieve meaningful
conservation outcomes.


Figure 3: Water basin accounting diagram illustrating how agricultural withdrawals are
partitioned. Efficiency upgrades often eliminate recoverable return flows rather than reducing

total consumption, triggering the rebound effect. Adapted from [4].

## 3.3 Limitations of Conventional Policy Instruments


Addressing system-level water consumption requires the implementation of policies that directly
limit total extraction, such as volumetric quotas, pricing mechanisms, or regulatory restrictions.
However, the effectiveness of such measures in practice is constrained by several structural
factors.

First, water governance in Iran is highly fragmented, with management often organized along
administrative rather than hydrological boundaries. This complicates coordinated resource
management at the basin level [3].

Second, the widespread presence of unregistered or weakly monitored wells makes it difficult to
accurately measure and control groundwater extraction. Without reliable monitoring,
enforcement of usage limits becomes highly challenging.

Finally, policies that impose strict limits on water use often face resistance from agricultural
communities, particularly when livelihoods depend on continued access to water. As observed in
western Iran, social barriers such as traditional irrigation habits, a lack of trust in official water
management, and the prioritization of short-term economic survival create a 'tragedy of the
commons' that severely limits the effectiveness of traditional regulatory approaches [5]. This
creates a tension between environmental sustainability and economic stability, further limiting
the effectiveness of traditional regulatory approaches.


4: Paradigm model mapping the social, institutional, and cognitive barriers to water scarcity

adaptation among stakeholders. Adapted from [5].

## 3.4 Implications for Irrigation Management


The limitations of both efficiency-based technologies and conventional policy instruments
suggest that improving water sustainability in agriculture requires more than incremental
changes. Specifically, it highlights the difficulty of managing water use in systems characterized
by decentralized decision-making, incomplete information, and competing incentives.

In this context, irrigation management must be understood not only as a technological or policy
challenge, but as a problem of decision-making under constraints. Effective solutions require
approaches that can:

  - Account for system dynamics and environmental variability

  - Operate within explicit water limits

  - Reduce reliance on purely manual or incentive-driven behavior

These requirements point toward the need for more systematic and adaptive decision-making
frameworks, reinforcing the relevance of control-based and data-driven approaches in modern
irrigation systems.

## 3.5 Irrigation as a Constrained Control Problem


Agricultural irrigation, particularly in water-scarce regions such as Iran, can be rigorously
formulated as a constrained, dynamic control problem. Rather than treating irrigation as a static
scheduling task, it is more appropriately understood as a sequential decision-making process
operating under uncertainty, physical dynamics, and strict resource limitations.


At its core, the system evolves over time according to interacting environmental and biological
processes. The state of the system includes variables such as soil moisture within the root zone,
crop growth stage, weather conditions, and soil characteristics. These states are only partially
observable and are subject to stochastic disturbances, particularly from unpredictable
meteorological inputs such as precipitation and temperature fluctuations.

The control input is the irrigation decision, specifically, the timing and volume of water applied
to the field. This decision directly influences soil moisture dynamics and indirectly affects plant
health, evapotranspiration rates, and ultimately crop yield.

A defining feature of irrigation in water-constrained environments is the presence of hard
constraints, most notably a limited seasonal or annual water budget. Additional constraints may
include physical limitations of irrigation infrastructure, soil absorption capacity, and operational
considerations. These constraints fundamentally differentiate the problem from unconstrained
optimization, as any control strategy must ensure that water usage remains within strict limits
while maintaining acceptable agricultural productivity.

The objective of the control system is inherently multi-dimensional. It typically involves
maximizing crop yield or biomass production, improving water use efficiency, and ensuring
robustness against environmental variability. In severely water-scarce contexts, this objective
often shifts toward deficit irrigation strategies, where water is deliberately applied below the
crop’s theoretical optimum in order to maximize productivity per unit of water rather than
absolute yield.

Traditional irrigation practices, such as fixed scheduling or threshold-based control, are limited
in their ability to handle these complexities. They are generally reactive, unable to anticipate
future disturbances, and incapable of optimally balancing competing objectives under
constraints. As a result, there is a growing need for advanced control methodologies that can
incorporate system dynamics, forecast future states, and enforce constraints in a principled
manner.

Within this context, two major paradigms emerge as particularly relevant. Model-based control
approaches, such as Model Predictive Control (MPC), leverage explicit system models to
optimize control actions over a finite time horizon while respecting constraints [11]. In contrast,
data-driven approaches, particularly Reinforcement Learning (RL), learn control policies directly
from interaction with the environment, enabling adaptation to complex, nonlinear dynamics
without requiring an explicit model [10].

The increasing availability of environmental sensing, remote data acquisition, and computational
resources enables the practical deployment of such methods within modern irrigation systems.
However, determining which class of control strategy is more suitable for real-world, waterconstrained agriculture remains an open and critical question. This motivates the need for
systematic evaluation of advanced control approaches under realistic agro-hydrological
conditions.


# 4. Modern and Smart Irrigation Technologies: A Cyber- Physical Control Perspective

## 4.1 The Transition to Precision and Data-Driven Irrigation

Traditional irrigation practices in Iran have largely relied on fixed schedules or manual decisionmaking based on farmer experience. While these approaches are simple to implement, they are
inherently reactive and poorly suited to environments characterized by water scarcity, climate
variability, and heterogeneous soil conditions.

The emergence of precision agriculture represents a fundamental shift from intuition-based
management to data-driven, adaptive decision-making [6], [7]. In this paradigm, irrigation
systems are no longer treated as passive infrastructure but as dynamic control systems that
continuously monitor environmental conditions and adjust water application accordingly.

This transformation is enabled by the integration of sensing, computation, and actuation into a
unified framework commonly referred to as a Cyber-Physical System (CPS) [7], [12]. Typically,
this IoT-based architecture is structured across multiple layers: the perception/device layer
(sensors), the network layer (wireless communication), the management layer (cloud/edge
processing), and the application layer (user interfaces) [6], [7]. Within such systems, irrigation
decisions are no longer predefined but are computed in real time based on the evolving state of
the agricultural environment.

## 4.2 System Architecture and the Role of Feedback


At a high level, a smart irrigation system operates as a closed-loop control system.
Environmental sensors provide continuous feedback on key state variables such as soil moisture,
climatic conditions, and plant health. This information is processed by a decision-making unit,
which determines the appropriate irrigation action. The resulting control signal is then executed
through actuators such as valves or pumps, directly influencing the system state.

This feedback loop is critical for moving from static irrigation schedules to state-aware and
responsive control strategies. Unlike traditional systems, which operate independently of realtime conditions, closed-loop systems can dynamically adapt to disturbances such as unexpected
rainfall, heatwaves, or changes in crop water demand.

The effectiveness of such systems depends not only on the availability of data but on the quality
of the decision-making logic that interprets this data and generates control actions.


Figure 5: Architecture of a Cyber-Physical System (CPS) in smart agriculture, demonstrating the

closed-loop feedback between data sources, decision-making controllers, and field actuators.

Adapted from [12].

## 4.3 From Monitoring to Decision-Making: The Role of Data


Modern irrigation systems can access a diverse range of data sources, including:

  - Soil moisture and root-zone conditions

  - Weather variables (temperature, humidity, solar radiation, wind)

  - Evapotranspiration estimates

  - Crop health indicators derived from remote sensing [8].

While the availability of these data streams enables more informed decision-making, it also
introduces significant complexity. The relationship between these variables and optimal
irrigation behavior is nonlinear, time-dependent, and context-specific.

As a result, the primary challenge is not data collection but translating data into optimal control
actions under uncertainty and constraints. This shift, from sensing to decision-making, marks the
central role of control theory and machine learning in modern irrigation systems.

## 4.4 Limitations of Conventional Control Strategies


Early automation in irrigation systems relied on relatively simple control mechanisms, such as:

  - Fixed scheduling (time-based irrigation)


  - Threshold-based control (e.g., activating irrigation below a soil moisture level)

While these approaches improve upon purely manual methods, they remain fundamentally
limited. They are reactive rather than predictive, and they cannot effectively incorporate multiple
interacting variables or anticipate future system states.

More advanced heuristic approaches, such as fuzzy logic controllers, partially address these
limitations by combining multiple inputs into rule-based decisions. However, they still lack the
ability to systematically optimize performance over time or guarantee adherence to strict
constraints such as water budgets.

These limitations become particularly critical in water-scarce environments, where inefficient
decisions can have cumulative and irreversible impacts on water resources.

## 4.5 Toward Predictive and Optimal Control


To overcome the limitations of heuristic and reactive strategies, modern irrigation systems
increasingly rely on predictive and optimization-based control methods. These approaches
explicitly account for system dynamics, future uncertainties, and resource constraints when
determining irrigation actions.

Two major paradigms have emerged in this context:

  - Model-based control, where a mathematical representation of the soil–plant–atmosphere
system is used to predict future states and optimize control decisions. Model Predictive
Control (MPC) is a prominent example, capable of handling multi-variable systems and
enforcing strict constraints such as limited water availability [11].


Figure 6. Schematic of the Model Predictive Control (MPC) architecture for agricultural
irrigation. The controller utilizes a predictive plant-soil model alongside real-time weather
forecasts and sensor feedback to optimize water application over a receding horizon. Adapted

from [11].


  - Data-driven control, where control policies are learned directly from data through
interaction with the environment. Reinforcement Learning (RL) is particularly well

suited for this setting, as it can learn to handle complex, nonlinear dynamics without
requiring an explicit system model [10].

Both paradigms aim to determine the optimal irrigation strategy under uncertainty, but they differ
fundamentally in how they represent system knowledge, handle constraints, and scale
computationally.


Figure 7: Classification of technological trends in smart agriculture, highlighting the data

pipeline from IoT collection to Machine Learning prediction. Adapted from [9].

## 4.6 The Need for Systematic Evaluation


Despite the rapid development of smart irrigation technologies and control strategies, their
practical performance in real-world, water-constrained environments remains insufficiently
understood. Many existing studies focus on isolated techniques or controlled experimental
settings, making it difficult to assess how different approaches compare under consistent
conditions.

In particular, there is a need to evaluate how advanced control methods perform when:

  - Operating under strict water constraints

  - Managing long-term crop growth dynamics

  - Responding to stochastic environmental disturbances


  - Balancing trade-offs between yield and water use efficiency

Addressing these challenges requires a systematic and controlled evaluation framework, where
different control strategies can be compared under identical environmental and operational
conditions. This forms the basis for the methodological approach developed in this thesis.


# 5. Feasibility and Impact Analysis for Iran

## 5.1 Potential Impact of Optimized Irrigation Strategies

The adoption of data-driven and control-based irrigation strategies has the potential to
significantly reduce agricultural water consumption while maintaining acceptable levels of crop
productivity. Empirical studies across various regions indicate that transitioning from traditional
irrigation practices to optimized, feedback-driven systems can yield substantial improvements in
water use efficiency and overall resource management.

In the context of Iran, where approximately 90% of freshwater withdrawals are allocated to
agriculture, even modest improvements in irrigation efficiency could translate into large-scale
water savings at the national level. More importantly, the shift from maximizing absolute yield to
optimizing productivity per unit of water aligns directly with the constraints imposed by severe
water scarcity.

Additionally, improved control over irrigation processes enables the integration of alternative
water sources, such as treated wastewater. When managed properly, these sources can
supplement limited freshwater supplies and provide a more stable input independent of climatic
variability [8]. However, their safe and effective use requires precise control over water
application to avoid soil degradation, salinity buildup, and contamination risks.

## 5.2 Structural and Operational Constraints


Despite the theoretical benefits, the practical deployment of advanced irrigation systems in Iran
faces significant structural and operational challenges.

One major constraint is the fragmented nature of agricultural landownership. A large proportion
of farms are small-scale operations, which limits the economic feasibility of deploying
sophisticated sensing and control infrastructure at the individual farm level [5], [9]. This
fragmentation complicates both the adoption of technology and the enforcement of coordinated
water management strategies.

Infrastructure limitations also present a barrier. Reliable access to communication networks,
consistent power supply, and technical maintenance capabilities are unevenly distributed across
rural areas. These constraints can directly impact the reliability and robustness of any
technology-dependent irrigation system.

Furthermore, irrigation decisions are embedded within broader socio-economic contexts.
Farmers often operate under strong economic pressures, prioritizing short-term stability over
long-term resource sustainability. This creates resistance to approaches that impose strict
constraints on water usage, even when such constraints are necessary from a system-wide
perspective.


## 5.3 Uncertainty, Risk, and the Need for Pre-Deployment Evaluation

A critical challenge in adopting advanced irrigation control strategies is the inherent risk
associated with decision-making under uncertainty. Irrigation directly affects crop survival and
yield; therefore, suboptimal or poorly calibrated control policies can result in significant
economic losses.

This risk is particularly pronounced for data-driven approaches such as Reinforcement Learning,
which typically require extensive exploration to learn effective policies. Testing such methods
directly in physical agricultural environments is impractical, as exploratory actions may lead to
crop stress or failure.

Similarly, even model-based approaches depend on assumptions about system dynamics that
may not fully capture real-world variability in soil properties, weather patterns, and crop
responses.

As a result, there is a strong need for robust pre-deployment evaluation frameworks that allow
different control strategies to be tested safely and systematically. High-fidelity simulation
environments, which integrate crop growth models and environmental dynamics, provide a
practical solution to this problem. They enable controlled experimentation across a wide range of
scenarios without exposing real crops or farmers to risk.

## 5.4 Bridging Simulation and Real-World Deployment


While simulation-based evaluation is essential, it introduces its own challenges, particularly the
gap between simulated and real-world conditions. Differences in environmental variability,
measurement noise, and system uncertainties can lead to discrepancies between simulated
performance and field outcomes [11].

To address this, control strategies must be evaluated not only on their nominal performance but
also on their robustness and adaptability. Methods that rely heavily on precise models may
struggle when faced with unmodeled dynamics, while purely data-driven approaches may require
large amounts of representative training data to generalize effectively.

This highlights the importance of comparing different control paradigms under consistent
conditions, with careful attention to:

  - Sensitivity to environmental uncertainty

  - Ability to operate under strict resource constraints

  - Computational requirements for real-time deployment

  - Stability and reliability over long time horizons

Understanding these trade-offs is essential for determining which approaches are most suitable
for eventual physical implementation in water-constrained agricultural systems.


## 5.5 Implications for Method Selection

The combination of environmental constraints, infrastructural limitations, and deployment risks
implies that the selection of an appropriate irrigation control strategy cannot be based solely on
theoretical performance. Instead, it must consider a broader set of criteria, including robustness,
scalability, and practical feasibility.

In this context, the evaluation of advanced control methods, particularly model-based and
learning-based approaches, becomes a necessary step toward identifying solutions that can be
realistically deployed. Rather than assuming the superiority of any single method, a systematic
comparison under controlled conditions is required to assess how different approaches perform
across relevant scenarios.


This perspective motivates the need for rigorous experimental analysis, forming the foundation
for the methodological framework developed in this thesis.


# 6. References


[1] M. B. Mesgaran and P. Azadi, "A National Adaptation Plan for Water Scarcity in Iran,"
_Stanford Iran 2040 Project_, Working Paper 6, Aug. 2018.


[2] R. Behling, M. Roessner, and S. Foerster, "Interrelations of vegetation growth and water
scarcity in Iran revealed by satellite time series," _Scientific Reports_, vol. 12, 20784 (2022)..


[3] H. Nouri et al., "Water management dilemma in the agricultural sector of Iran: A review
focusing on water governance," _Agricultural Water Management_, vol. 278, p. 108162, 2023.


[4] D. Pérez-Blanco, C. H. Hampson, and C. G. Coggan, "Irrigation Technology and Water
Conservation: A Review of the Theory and Evidence," _Review of Environmental Economics_
_and Policy_, vol. 14, no. 2, pp. 216-239, 2020.


[5] Y. Azadi, J. Yaghoubi, H. Gholizadeh, S. Gholamrezai, and F. Rahimi-Feyzabad, "Social
barriers to water scarcity adaptation: A grounded theory exploration in arid and semi-arid
regions," _Agricultural Water Management_, vol. 309, p. 109338, Jan. 2025.


[6] G. Farrokhi and M. Gapeleh, "Smart Agriculture Based on Internet of Things [ کشاورزی
هوشمند مبتنی بر اینترنت اشیاء ]," _Roshd-e-Fanavari (Technology Incubator Quarterly)_, vol. 15, no.
59, pp. 30-37, Summer 2019.


[7] M. Pourgholam-Amiji, I. Hajirad, K. Ahmadaali, and A. Liaghat, "Smart Irrigation Based
on the IoT," _Iranian Journal of Soil and Water Research_, vol. 55, no. 9, pp. 1647-1678, Nov.
2024.


[8] I. Hajirad, "Identification of Smart Irrigation Management Methods and Tools and Their
Impact on Optimizing Irrigation Water Consumption," _Journal of Water and Sustainable_
_Development_, vol. 12, no. 2, pp. 159-176, Aug. 2025.


[9] F. Hesabi, I. Jafarpanah, A. Karshenas, and M. H. Ahmadzadeh, "Technological Trends in
Smart Agriculture," _Journal of Entrepreneurial Strategies in Agriculture_, vol. 10, no. 2, pp.
100-115, Jun. 2023.


[10] I. Hajirad, P. Pourmohammad, and M. Pourgholam Amiji, "Optimizing Water
Productivity in the Face of Climate Change: The Central Role of Machine Learning
Approaches," _Iranian Journal of Soil and Water Research_, vol. 56, no. 11, pp. 2929-2949,
Jan. 2026.


[11] E. Lopez-Jimenez et al., "Agent-based model predictive control of soil-crop irrigation
with topographical information," _Control Engineering Practice_, vol. 143, p. 105788, 2024.


[12] E. Lopez-Jimenez et al., "Dynamic Modeling of Crop-Soil Systems to Design Monitoring
and Automatic Irrigation Processes: A Review with Worked Examples," _Water_, vol. 14, no.
15, p. 2404, 2022.


