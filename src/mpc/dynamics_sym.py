# =============================================================================
# src/mpc/dynamics_sym.py
# CasADi symbolic version of the ABM crop-soil dynamics.
#
# This module builds a CasADi function that maps:
#   (x1[k], x5[k], u[k], climate[k], precomputed[k]) → (x1[k+1], x5[k+1], x4_increment[k])
#
# Design decisions:
#   - Only x1 (soil water) and x5 (ponding) are shooting states in the NLP.
#   - x2 (thermal time), h2 (heat stress), h7 (cold stress) are precomputed
#     constants passed as parameters — they depend only on climate, not on u.
#   - x3 (maturity stress) is accumulated inline during the horizon rollout
#     but NOT a shooting state (no continuity constraint). This is a small
#     approximation; x3 affects g() only in the senescence branch.
#   - x4 (biomass) is accumulated inline; the terminal value enters the cost.
#   - Cascade routing: agents are processed high-to-low within each step.
#     The topological order is baked into the symbolic graph at build time.
#
# Smoothing: uses CasADi's native fmax/fmin by default. If IPOPT struggles,
# set use_smooth=True to switch to the C2-smooth approximations from
# smoothing.py.
#
# v2.1 fix: when use_smooth=True, the SCS runoff conditional and the drought
# stress conditional now also use smooth formulations instead of ca.if_else,
# which is a hard (non-differentiable) switch that was causing IPOPT to
# struggle at kink points (~960 hard kinks per NLP with N=130, Hp=8).
#
# v2.2 fix: corrected drought stress to mirror abm.py at ETc = 0.
# The previous form h4 = max(1 - phi1/(ETc+eps), 0) returned h4 = 1
# (maximum stress) when ETc = 0 because 0 / eps = 0 and 1 - 0 = 1. This
# created a COM-vs-SOM inconsistency: with ETc = 0, the ABM (ground truth)
# returns h4 = 0 (no stress, since phi1 < ETc is False when both are zero),
# while the MPC's internal model returned the opposite. Although ETc never
# reaches zero during the rice growing season in Gilan (min ≈ 1.65 mm/day),
# the formulation is now corrected for robustness and to remove the model
# inconsistency. See _drought_stress in abm.py for the reference.
# =============================================================================

import casadi as ca
import numpy as np


def build_dynamics_function(terrain, crop, use_smooth=False):
    """Build the CasADi function for one ABM daily step.

    Parameters
    ----------
    terrain : dict
        From src.terrain.load_terrain. Needs: N, sends_to, Nr,
        topological_order, gamma_flat.
    crop : dict
        Crop parameters from soil_data.get_crop.
    use_smooth : bool
        If True, use smooth approximations of max/min. Default False
        (CasADi native fmax/fmin).

    Returns
    -------
    step_fn : casadi.Function
        Signature: step_fn(x1, x5, u, rain, ETc, rad, h2_k, h7_k, g_base_k)
                   → (x1_next, x5_next, x4_inc, h3_field)
        Where:
            x1, x5, u : (N,) vectors
            rain, ETc, rad, h2_k, h7_k, g_base_k : scalars
            x1_next, x5_next : (N,) vectors (next-day state)
            x4_inc : (N,) vector (biomass increment for this day)
            h3_field : (N,) vector (drought stress, for cost function use)
    """
    N = terrain['N']
    topo_order = terrain['topological_order']
    sends_to = terrain['sends_to']
    Nr_dict = terrain['Nr']

    # Soil/crop parameters
    theta1 = crop['theta1']
    theta2 = crop['theta2']
    theta3 = crop['theta3']
    theta4 = crop['theta4']
    theta5 = crop['theta5']
    theta6 = crop['theta6']
    theta13 = crop['theta13']
    theta14 = crop['theta14']
    theta_sat = crop.get('theta_sat', 0.50)

    fc_total = theta6 * theta5
    sat_total = theta_sat * theta5

    # Choose max/min operators
    if use_smooth:
        from src.mpc.smoothing import smooth_max_zero, smooth_min
        _max0 = lambda x: smooth_max_zero(x, eps=0.01)
        _min = lambda a, b: smooth_min(a, b, eps=0.01)
        _clip01 = lambda x: smooth_min(smooth_max_zero(x, eps=0.005), 1.0, eps=0.005)
    else:
        _max0 = lambda x: ca.fmax(x, 0)
        _min = lambda a, b: ca.fmin(a, b)
        _clip01 = lambda x: ca.fmin(ca.fmax(x, 0), 1)

    # ── Symbolic variables ────────────────────────────────────────────────────

    x1 = ca.SX.sym('x1', N)       # root zone soil water (mm)
    x5 = ca.SX.sym('x5', N)       # surface ponding (mm)
    u  = ca.SX.sym('u', N)        # irrigation (mm/day)

    rain    = ca.SX.sym('rain')    # daily rainfall (mm)
    ETc     = ca.SX.sym('ETc')     # crop evapotranspiration (mm/day)
    rad     = ca.SX.sym('rad')     # solar radiation (MJ/m²/day)
    h2_k    = ca.SX.sym('h2_k')   # heat stress (scalar, precomputed)
    h7_k    = ca.SX.sym('h7_k')   # cold stress (scalar, precomputed)
    g_base_k = ca.SX.sym('g_base_k')  # growth function baseline (scalar, precomputed)

    # ── Transpiration ─────────────────────────────────────────────────────────

    demand = theta1 * (x1 - theta2 * theta5)
    phi1 = _min(_max0(demand), ETc)

    # ── Surface hydrology with cascade routing ────────────────────────────────

    # Initialize W_surf for each agent
    W_surf = x5 + rain + u  # shape (N,)

    # We need to build the cascade symbolically. CasADi SX doesn't support
    # item assignment on vectors, so we build element-by-element and track
    # contributions via explicit symbolic accumulation.

    # Strategy: iterate through topological order. For each agent n:
    #   1. Compute phi2[n] (runoff generated, zero for sinks)
    #   2. Add phi2[n] / Nr[n] to each lower neighbor's W_surf
    #   3. Compute infiltration I[n]
    #
    # Since CasADi SX is symbolic, we track W_surf as a list of expressions
    # that get updated as we process agents top-to-bottom.

    W_surf_list = [W_surf[i] for i in range(N)]
    phi2_list = [ca.SX(0)] * N
    I_list = [ca.SX(0)] * N

    for n in topo_order:
        n = int(n)
        w_n = W_surf_list[n]
        nr_n = Nr_dict[n]

        # SCS runoff (sinks: Nr=0, no runoff)
        if nr_n == 0:
            phi2_n = ca.SX(0)
        else:
            # phi2 = (W - theta3)^2 / (W + 4*theta3) when W > theta3, else 0
            #
            # Smooth form: phi2 = max(W - theta3, 0)^2 / (W + 4*theta3)
            # When W < theta3, max(W-theta3, 0) → 0 smoothly, so phi2 → 0.
            # When W > theta3, max(W-theta3, 0) = W-theta3, recovering the
            # original SCS formula exactly (up to O(eps^2) smoothing error).
            # This replaces the non-differentiable ca.if_else that was causing
            # IPOPT to struggle at kink points.
            excess = w_n - theta3
            excess_pos = _max0(excess)
            phi2_n = excess_pos**2 / (w_n + 4 * theta3)

        phi2_list[n] = phi2_n

        # Route runoff to lower neighbors (within same day — cascade)
        if nr_n > 0:
            per_neighbor = phi2_n / nr_n
            for m in sends_to[n]:
                W_surf_list[m] = W_surf_list[m] + per_neighbor

        # Infiltration
        I_max = _max0(sat_total - x1[n] + phi1[n])
        available = w_n - phi2_n
        I_n = _min(_max0(available), I_max)
        I_list[n] = I_n

    # Assemble vectors from lists
    phi2_vec = ca.vertcat(*phi2_list)
    I_vec = ca.vertcat(*I_list)
    W_surf_vec = ca.vertcat(*W_surf_list)

    # ── Subsurface hydrology ──────────────────────────────────────────────────

    x1_temp = x1 + I_vec - phi1
    E_sub = _max0(x1_temp - fc_total)
    phi3 = theta4 * E_sub
    x1_next = _max0(x1_temp - phi3)

    # ── Surface ponding for next day ──────────────────────────────────────────

    x5_next = _max0(W_surf_vec - phi2_vec - I_vec)

    # ── Drought stress ────────────────────────────────────────────────────────

    # Mirrors abm.py exactly:
    #     h4 = (ETc - phi1) / ETc  when phi1 < ETc, else 0
    #     h3 = 1 - theta14 * h4
    #
    # Rewriting via deficit = max(ETc - phi1, 0) makes the numerator vanish
    # when ETc = 0 (since phi1 = min(demand, ETc) = 0 in that case), so
    # h4 = 0 / eps = 0 and h3 = 1 (no stress). This avoids the COM-vs-SOM
    # inconsistency where the previous form h4 = max(1 - phi1/(ETc+eps), 0)
    # returned h4 = 1 (maximum stress) when ETc = 0.
    #
    # When ETc > 0, phi1 ≤ ETc always (transpiration is min(demand, ETc)),
    # so deficit = ETc - phi1 ≥ 0 and h4 = (ETc - phi1) / ETc = 1 - phi1/ETc,
    # identical to the original ABM expression. The (ETc + 1e-6) denominator
    # is retained only to guarantee numerical safety; the smoothing operator
    # _max0 is no longer needed because (ETc - phi1) is already non-negative
    # by construction, but we keep it for symbolic robustness in the smooth
    # formulation (where _max0 is the smoothed C² operator).
    deficit = _max0(ETc - phi1)
    h4 = deficit / (ETc + 1e-6)
    h3 = 1.0 - theta14 * h4

    # ── Waterlogging stress ───────────────────────────────────────────────────

    # h6 = clip(1 - (x1 - fc_total) / fc_total, 0, 1)
    excess_ratio = (x1 - fc_total) / ca.fmax(fc_total, 1e-6)
    h6 = _clip01(1.0 - excess_ratio)

    # ── Biomass increment ─────────────────────────────────────────────────────

    # x4_inc = theta13 * h3 * h6 * h7 * g_base * rad
    # h7_k and g_base_k are scalars (precomputed, broadcast to all agents)
    x4_inc = theta13 * h3 * h6 * h7_k * g_base_k * rad

    # ── Build CasADi Function ─────────────────────────────────────────────────

    step_fn = ca.Function(
        'abm_step',
        [x1, x5, u, rain, ETc, rad, h2_k, h7_k, g_base_k],
        [x1_next, x5_next, x4_inc, h3],
        ['x1', 'x5', 'u', 'rain', 'ETc', 'rad', 'h2_k', 'h7_k', 'g_base_k'],
        ['x1_next', 'x5_next', 'x4_inc', 'h3'],
    )

    return step_fn
