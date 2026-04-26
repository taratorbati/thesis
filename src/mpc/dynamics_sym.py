# =============================================================================
# src/mpc/dynamics_sym.py
# CasADi symbolic version of the ABM crop-soil dynamics.
#
# Builds a ca.Function (SX internals) that maps:
#   (x1[k], x5[k], u[k], climate[k], precomputed[k]) → (x1[k+1], x5[k+1],
#                                                         x4_increment[k], h3[k])
#
# Design:
#   - SX is used internally for the cascade routing loop (efficient for
#     element-wise scalar operations). The function is wrapped in ca.Function
#     which encapsulates the SX graph.
#   - When the solver (v3.0) calls this function with MX arguments, CasADi
#     treats each call as an opaque graph node → no expression inlining.
#   - Only x1 (soil water) and x5 (ponding) are shooting states.
#   - x2 (thermal time), h2, h7 are precomputed (climate-only).
#   - x3 (maturity stress) is tracked from true state, not a shooting state.
#
# Smoothing (v2.1 fix):
#   - SCS runoff: max(W-θ3, 0)^2 / (W+4θ3) instead of ca.if_else
#   - Drought stress h4: guarded denominator instead of ca.if_else
# =============================================================================

import casadi as ca
import numpy as np


def build_dynamics_function(terrain, crop, use_smooth=False):
    """Build the CasADi function for one ABM daily step.

    Parameters
    ----------
    terrain : dict
        From src.terrain.load_terrain.
    crop : dict
        Crop parameters from soil_data.get_crop.
    use_smooth : bool
        If True, use C2-smooth approximations of max/min.

    Returns
    -------
    step_fn : casadi.Function
        Signature: step_fn(x1, x5, u, rain, ETc, rad, h2_k, h7_k, g_base_k)
                   → (x1_next, x5_next, x4_inc, h3_field)
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

    # ── Symbolic variables (SX — efficient for element-wise cascade) ──────

    x1 = ca.SX.sym('x1', N)
    x5 = ca.SX.sym('x5', N)
    u  = ca.SX.sym('u', N)

    rain     = ca.SX.sym('rain')
    ETc      = ca.SX.sym('ETc')
    rad      = ca.SX.sym('rad')
    h2_k     = ca.SX.sym('h2_k')
    h7_k     = ca.SX.sym('h7_k')
    g_base_k = ca.SX.sym('g_base_k')

    # ── Transpiration ─────────────────────────────────────────────────────

    demand = theta1 * (x1 - theta2 * theta5)
    phi1 = _min(_max0(demand), ETc)

    # ── Surface hydrology with cascade routing ────────────────────────────

    W_surf = x5 + rain + u

    W_surf_list = [W_surf[i] for i in range(N)]
    phi2_list = [ca.SX(0)] * N
    I_list = [ca.SX(0)] * N

    for n in topo_order:
        n = int(n)
        w_n = W_surf_list[n]
        nr_n = Nr_dict[n]

        # SCS runoff — smooth form: max(excess,0)^2 / (W + 4*theta3)
        if nr_n == 0:
            phi2_n = ca.SX(0)
        else:
            excess = w_n - theta3
            excess_pos = _max0(excess)
            phi2_n = excess_pos**2 / (w_n + 4 * theta3)

        phi2_list[n] = phi2_n

        # Route runoff to lower neighbors
        if nr_n > 0:
            per_neighbor = phi2_n / nr_n
            for m in sends_to[n]:
                W_surf_list[m] = W_surf_list[m] + per_neighbor

        # Infiltration
        I_max = _max0(sat_total - x1[n] + phi1[n])
        available = w_n - phi2_n
        I_n = _min(_max0(available), I_max)
        I_list[n] = I_n

    phi2_vec = ca.vertcat(*phi2_list)
    I_vec = ca.vertcat(*I_list)
    W_surf_vec = ca.vertcat(*W_surf_list)

    # ── Subsurface hydrology ──────────────────────────────────────────────

    x1_temp = x1 + I_vec - phi1
    E_sub = _max0(x1_temp - fc_total)
    phi3 = theta4 * E_sub
    x1_next = _max0(x1_temp - phi3)

    # ── Surface ponding ───────────────────────────────────────────────────

    x5_next = _max0(W_surf_vec - phi2_vec - I_vec)

    # ── Drought stress — guarded denominator instead of ca.if_else ────────

    h4 = _max0(1.0 - phi1 / (ETc + 1e-6))
    h3 = 1.0 - theta14 * h4

    # ── Waterlogging stress ───────────────────────────────────────────────

    excess_ratio = (x1 - fc_total) / ca.fmax(fc_total, 1e-6)
    h6 = _clip01(1.0 - excess_ratio)

    # ── Biomass increment ─────────────────────────────────────────────────

    x4_inc = theta13 * h3 * h6 * h7_k * g_base_k * rad

    # ── Build CasADi Function ─────────────────────────────────────────────

    step_fn = ca.Function(
        'abm_step',
        [x1, x5, u, rain, ETc, rad, h2_k, h7_k, g_base_k],
        [x1_next, x5_next, x4_inc, h3],
        ['x1', 'x5', 'u', 'rain', 'ETc', 'rad', 'h2_k', 'h7_k', 'g_base_k'],
        ['x1_next', 'x5_next', 'x4_inc', 'h3'],
    )

    return step_fn
