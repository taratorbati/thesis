# =============================================================================
# src/mpc/cost.py
# Five-term normalized cost function for the MPC.
#
# J = -α₁ · terminal_biomass / x4_ref
#   + α₂ · Σ_k (Σ_n u^n(k)) / W_daily_ref
#   + α₃ · Σ_k (1/N) Σ_n max(ST - x1^n(k), 0) / (ST - WP)
#   + α₄ · Σ_k (1/|S|) Σ_{n∈S} x5^n(k) / x5_ref
#   + α₅ · Σ_k ||u(k+1) - u(k)||² / (u_max² · N)
#
# All terms are O(1) per time step when properly normalized.
#
# v2.1 fix: compute_cost now accepts use_smooth flag. When True, the drought
# deficit term (Term 3) uses smooth_max_zero instead of ca.fmax, removing the
# last source of non-differentiable kinks in the cost landscape.
# =============================================================================

import casadi as ca
import numpy as np

# Default weights (from ARCHITECTURE.md, economically grounded)
DEFAULT_WEIGHTS = {
    'alpha1': 1.0,     # terminal biomass (anchor)
    'alpha2': 0.01,    # water cost (domestic water pricing)
    'alpha3': 0.1,     # drought stress regularization
    'alpha4': 0.5,     # sink ponding penalty
    'alpha5': 0.005,   # Δu regularization
}

# Default reference values for normalization
DEFAULT_REFS = {
    'x4_ref':       900.0,   # g/m², target yield ≈ 3800 kg/ha
    'W_daily_ref':  None,    # computed as 5.0 * N at build time
    'ST':           None,    # stress threshold, computed from crop at build time
    'WP':           None,    # wilting point total, computed from crop at build time
    'x5_ref':       10.0,    # mm, ponding reference
    'u_max':        12.0,    # mm/day, actuator cap
}


def build_cost_components(N, crop, sink_agents, weights=None, refs=None):
    """Return a dict of callables, each computing one cost term.

    Parameters
    ----------
    N : int
        Number of agents.
    crop : dict
        Crop parameter dict.
    sink_agents : list[int]
        Indices of sink agents (Nr = 0).
    weights : dict, optional
        Override DEFAULT_WEIGHTS.
    refs : dict, optional
        Override DEFAULT_REFS.

    Returns
    -------
    dict
        Keys: 'biomass', 'water', 'drought', 'ponding', 'delta_u', 'total'.
        Values: description strings and the weight/ref values used.
    components : dict
        The actual numeric values needed by the solver to build the cost.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    r = {**DEFAULT_REFS, **(refs or {})}

    # Compute derived references
    fc_total = crop['theta6'] * crop['theta5']
    wp_total = crop['theta2'] * crop['theta5']
    p = crop.get('p', 0.20)
    raw = p * (fc_total - wp_total)
    st = fc_total - raw  # stress threshold

    if r['W_daily_ref'] is None:
        r['W_daily_ref'] = 5.0 * N
    if r['ST'] is None:
        r['ST'] = st
    if r['WP'] is None:
        r['WP'] = wp_total

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
        Irrigation actions for k = 0..Hp-1.
    x1_trajectory : list of ca.SX, each shape (N,)
        Soil water AFTER step k, for k = 0..Hp-1.
    x5_trajectory : list of ca.SX, each shape (N,)
        Surface ponding AFTER step k, for k = 0..Hp-1.
    x4_terminal_mean : ca.SX scalar
        Field-mean biomass at the end of the horizon.
    components : dict
        From build_cost_components.
    Hp : int
        Prediction horizon.
    use_smooth : bool
        If True, use smooth_max_zero for the drought deficit term.
        Default False (CasADi native ca.fmax).

    Returns
    -------
    J : ca.SX scalar
        Total cost (to be minimized by IPOPT).
    terms : dict
        Individual cost terms (for logging/debugging).
    """
    w = components['weights']
    r = components['refs']
    N = components['N']
    sinks = components['sink_agents']
    n_sinks = components['n_sinks']

    # Choose max(x, 0) operator for cost terms
    if use_smooth:
        from src.mpc.smoothing import smooth_max_zero
        _max0 = lambda x: smooth_max_zero(x, eps=0.01)
    else:
        _max0 = lambda x: ca.fmax(x, 0)

    # ── Term 1: Terminal biomass (Mayer term, negative = maximize) ─────────
    J_biomass = -w['alpha1'] * x4_terminal_mean / r['x4_ref']

    # ── Term 2: Water cost (path) ─────────────────────────────────────────
    J_water = ca.SX(0)
    for k in range(Hp):
        daily_total = ca.sum1(u_trajectory[k])
        J_water += daily_total / r['W_daily_ref']
    J_water *= w['alpha2']

    # ── Term 3: Drought stress penalty (path) ─────────────────────────────
    J_drought = ca.SX(0)
    st = r['ST']
    denom = max(st - r['WP'], 1e-6)
    for k in range(Hp):
        deficit = _max0(st - x1_trajectory[k])  # (N,) vector
        J_drought += ca.sum1(deficit) / (N * denom)
    J_drought *= w['alpha3']

    # ── Term 4: Ponding at sinks (path) ───────────────────────────────────
    J_ponding = ca.SX(0)
    if len(sinks) > 0:
        for k in range(Hp):
            sink_x5_sum = ca.SX(0)
            for s in sinks:
                sink_x5_sum += x5_trajectory[k][s]
            J_ponding += sink_x5_sum / (n_sinks * r['x5_ref'])
    J_ponding *= w['alpha4']

    # ── Term 5: Control rate regularization (path) ────────────────────────
    J_delta_u = ca.SX(0)
    u_max_sq_N = r['u_max']**2 * N
    for k in range(1, Hp):
        diff = u_trajectory[k] - u_trajectory[k-1]
        J_delta_u += ca.dot(diff, diff) / u_max_sq_N
    J_delta_u *= w['alpha5']

    # ── Total ─────────────────────────────────────────────────────────────
    J_total = J_biomass + J_water + J_drought + J_ponding + J_delta_u

    terms = {
        'biomass': J_biomass,
        'water': J_water,
        'drought': J_drought,
        'ponding': J_ponding,
        'delta_u': J_delta_u,
        'total': J_total,
    }

    return J_total, terms
