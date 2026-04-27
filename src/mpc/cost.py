# =============================================================================
# src/mpc/cost.py
# Five-term normalized cost function for the MPC.
#
# J = -α₁ · terminal_biomass / x4_ref
#   + α₂ · Σ_k (Σ_n u^n(k)) / W_daily_ref
#   + α₃ · Σ_k (1/N) Σ_n max(ST - x1^n(k), 0) / (ST - WP)
#   + α₄ · Σ_k (1/N) Σ_n x5^n(k) / x5_ref
#   + α₅ · Σ_k ||u(k+1) - u(k)||² / (u_max² · N)
#
# All terms are O(1) per time step when properly normalized.
#
# v2.2: Ponding penalty applies to ALL agents (field-mean), not just sinks.
#   With fractional routing boundary conditions (terrain.py v2.0), former
#   sink agents drain off-farm, so sink-only penalties would be zero.
#   Penalizing all agents is physically correct: waterlogging damages any
#   agent's crop. X5_REF increased from 10→50mm so that transient ponding
#   from a single storm (typically 10-30mm) produces a small penalty, while
#   persistent multi-day ponding (50+mm) ramps up meaningfully through the
#   horizon summation.
#
#   Reference for waterlogging tolerance: Setter et al. (1997) "Review of
#   prospects for germplasm improvement for waterlogging tolerance in wheat,
#   barley and oats" — rice tolerates 1-2 days of shallow ponding.
# =============================================================================

import casadi as ca
import numpy as np

# Default weights (from ARCHITECTURE.md, economically grounded)
DEFAULT_WEIGHTS = {
    'alpha1': 1.0,     # terminal biomass (anchor)
    'alpha2': 0.01,    # water cost (domestic water pricing)
    'alpha3': 0.1,     # drought stress regularization
    'alpha4': 0.5,     # ponding penalty (all agents)
    'alpha5': 0.005,   # Δu regularization
}

# Default reference values for normalization
DEFAULT_REFS = {
    'x4_ref':       900.0,   # g/m², target yield ≈ 3800 kg/ha
    'W_daily_ref':  None,    # computed as 5.0 * N at build time
    'ST':           None,    # stress threshold, computed from crop at build time
    'WP':           None,    # wilting point total, computed from crop at build time
    'x5_ref':       50.0,    # mm, ponding reference (raised from 10 to match
                             # realistic storm ponding; see terrain.py v2.0)
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
        deficit = _max0(st - x1_trajectory[k])
        J_drought += ca.sum1(deficit) / (N * denom)
    J_drought *= w['alpha3']

    # ── Term 4: Ponding penalty — all agents, field-mean (path) ───────────
    # Transient ponding (one day) penalized lightly; persistent ponding
    # accumulates across the horizon summation naturally.
    J_ponding = ca.SX(0)
    for k in range(Hp):
        J_ponding += ca.sum1(x5_trajectory[k]) / (N * r['x5_ref'])
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
