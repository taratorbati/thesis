# src.mpc — Model Predictive Controller for constrained irrigation.
#
# Solver: CasADi + IPOPT (interior-point, sparse NLP).
# Formulation: multiple-shooting with x1,x5 as shooting states,
#              x2 from precomputed cache, x3/x4 derived inline.
# Cost: 5-term normalized objective (terminal biomass, water, drought,
#       sink ponding, control-rate regularization).
