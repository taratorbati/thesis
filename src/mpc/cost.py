# =============================================================================
# src/mpc/cost.py
# Five-term normalized cost function for the MPC at the recommended operating
# point alpha*.
#
# J = -alpha1 * terminal_biomass / x4_ref
#   + alpha2 * Sum_k (Sum_n u^n(k)) / W_daily_ref
#   + alpha3 * Sum_k (1/N) Sum_n max(ST - x1^n(k), 0) / (ST - WP)
#   + alpha5 * Sum_k ||u(k+1) - u(k)||^2 / (u_max^2 * N)
#   + alpha6 * Sum_k (1/N) Sum_n [max(x1^n(k) - FC, 0) / FC]^2
#
# The original sixth term — the surface-ponding penalty alpha4 * J_ponding — is
# retained in the implementation for backward compatibility but inactive at
# alpha4 = 0. The joint factorial sensitivity analysis (Chapter 4 of the
# thesis) showed that a sufficiently strong alpha6 dynamically subsumes the
# role of alpha4 because surface water either infiltrates (raising x1) or
# drains off-farm. The recommended five-term formulation eliminates this
# redundancy.
#
# Default weights (recommended operating point alpha* — Chapter 4):
#   alpha1 = 1.0      terminal biomass anchor
#   alpha2 = 0.016    domestic-base Iranian water tariff (~7,000 toman/m^3)
#   alpha3 = 0.1      drought stress regularizer
#   alpha4 = 0.0      surface ponding (DISABLED — subsumed by alpha6)
#   alpha5 = 0.005    delta_u regularizer
#   alpha6 = 8.0      x1 > FC soft penalty (resolves chronic waterlogging)
#
# References:
#   Setter et al. (1997) — rice waterlogging tolerance is 1-2 days
#   Allen et al. FAO-56 (1998) — crop coefficients and stress thresholds
# =============================================================================

import casadi as ca
import numpy as np


# Recommended operating point alpha* — Chapter 4 of the thesis.
# Calibrated through a 33-configuration sensitivity sweep across:
#   - Group A: alpha2 price-tier sweep
#   - Group B: alpha3 drought regularizer sweep
#   - Group C: alpha4 ponding penalty sweep
#   - Group D-E: alpha6 FC overshoot sweep
#   - Group F: joint (alpha4, alpha6) factorial — falsifies alpha4 independence
#   - Group G: cross-scenario validation
#   - Group H: alpha6 ceiling test (saturation at alpha6 = 8)
DEFAULT_WEIGHTS = {
    'alpha1': 1.0,     # terminal biomass (anchor, do not change)
    'alpha2': 0.016,   # water cost — domestic-base Iranian tariff
    'alpha3': 0.1,     # drought stress regularizer
    'alpha4': 0.0,     # ponding — INACTIVE (subsumed by alpha6 per Group F)
    'alpha5': 0.005,   # delta_u regularizer
    'alpha6': 8.0,     # x1 > FC soft penalty — ACTIVATED at alpha6* = 8
}

# Default reference values for normalization
DEFAULT_REFS = {
    'x4_ref':       900.0,   # g/m^2, target yield ≈ 3800 kg/ha
    'W_daily_ref':  None,    # computed as 5.0 * N at build time
    'ST':           None,    # stress threshold, computed from crop at build time
    'WP':           None,    # wilting point total, computed from crop at build time
    'FC':           None,    # field capacity total (mm), computed from crop at build time
    'x5_ref':       50.0,    # mm, ponding reference (used only if alpha4 != 0)
    'u_max':        12.0,    # mm/day, actuator cap
}


def build_cost_components(N, crop, sink_agents, weights=None, refs=None):
    """Return a dict of numeric values needed by the solver to build the cost.

    Parameters
    ----------
    N : int
    crop : dict
    sink_agents : list[int]
        Kept for backward compatibility. No longer used for ponding penalty.
    weights : dict, optional
        Override DEFAULT_WEIGHTS. Any subset of {alpha1..alpha6}.
    refs : dict, optional

    Returns
    -------
    dict
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    r = {**DEFAULT_REFS, **(refs or {})}

    fc_total = crop['theta6'] * crop['theta5']
    wp_total = crop['theta2'] * crop['theta5']
    p = crop.get('p', 0.20)
    raw = p * (fc_total - wp_total)
    st = fc_total - raw

    if r['W_daily_ref'] is None:
        r['W_daily_ref'] = 5.0 * N
    if r['ST'] is None:
        r['ST'] = st
    if r['WP'] is None:
        r['WP'] = wp_total
    if r['FC'] is None:
        r['FC'] = fc_total

    n_sinks = max(len(sink_agents), 1)

    return {
        'weights': w,
        'refs': r,
        'N': N,
        'sink_agents': sink_agents,
        'n_sinks': n_sinks,
        'fc_total': fc_total,
        'wp_total': wp_total,
        'stress_threshold': st,
    }


def compute_cost(u_trajectory, x1_trajectory, x5_trajectory, x4_terminal_mean,
                 components, Hp, use_smooth=False):
    """Compute the total scalar cost from trajectory variables.

    All inputs are CasADi symbolic expressions.

    Parameters
    ----------
    u_trajectory : list of ca.SX, each shape (N,)
    x1_trajectory : list of ca.SX, each shape (N,)
    x5_trajectory : list of ca.SX, each shape (N,)
    x4_terminal_mean : ca.SX scalar
    components : dict
    Hp : int
    use_smooth : bool

    Returns
    -------
    J : ca.SX scalar
    terms : dict
    """
    w = components['weights']
    r = components['refs']
    N = components['N']

    if use_smooth:
        from src.mpc.smoothing import smooth_max_zero
        _max0 = lambda x: smooth_max_zero(x, eps=0.01)
    else:
        _max0 = lambda x: ca.fmax(x, 0)

    # Term 1: Terminal biomass (Mayer term, negative = maximize)
    J_biomass = -w['alpha1'] * x4_terminal_mean / r['x4_ref']

    # Term 2: Water cost (path)
    J_water = ca.SX(0)
    for k in range(Hp):
        daily_total = ca.sum1(u_trajectory[k])
        J_water += daily_total / r['W_daily_ref']
    J_water *= w['alpha2']

    # Term 3: Drought stress penalty (path)
    J_drought = ca.SX(0)
    st = r['ST']
    denom = max(st - r['WP'], 1e-6)
    for k in range(Hp):
        deficit = _max0(st - x1_trajectory[k])
        J_drought += ca.sum1(deficit) / (N * denom)
    J_drought *= w['alpha3']

    # Term 4: Ponding penalty — INACTIVE at alpha* (alpha4 = 0).
    # Retained for backward compatibility and for ablation sweeps.
    J_ponding = ca.SX(0)
    if w['alpha4'] != 0.0:
        for k in range(Hp):
            J_ponding += ca.sum1(x5_trajectory[k]) / (N * r['x5_ref'])
        J_ponding *= w['alpha4']

    # Term 5: Control rate regularization (path)
    J_delta_u = ca.SX(0)
    u_max_sq_N = r['u_max']**2 * N
    for k in range(1, Hp):
        diff = u_trajectory[k] - u_trajectory[k-1]
        J_delta_u += ca.dot(diff, diff) / u_max_sq_N
    J_delta_u *= w['alpha5']

    # Term 6: x1 > FC soft penalty (path)
    # Quadratic in the normalized excess. Inactive when x1 <= FC.
    # Resolves the FC-overshoot pathology at long prediction horizons
    # and the chronic-waterlogging pathology in wet-year scenarios.
    J_overfc = ca.SX(0)
    fc = r['FC']
    if w['alpha6'] != 0.0:
        for k in range(Hp):
            excess = _max0(x1_trajectory[k] - fc)
            normalized_sq = (excess / fc) ** 2
            J_overfc += ca.sum1(normalized_sq) / N
        J_overfc *= w['alpha6']

    # Total
    J_total = (J_biomass + J_water + J_drought + J_ponding
               + J_delta_u + J_overfc)

    terms = {
        'biomass': J_biomass,
        'water':   J_water,
        'drought': J_drought,
        'ponding': J_ponding,
        'delta_u': J_delta_u,
        'overfc':  J_overfc,
        'total':   J_total,
    }

    return J_total, terms
