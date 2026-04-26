# =============================================================================
# src/mpc/solver.py
# Builds and solves the IPOPT NLP for one MPC step.
#
# The NLP is constructed once at controller reset (expensive: ~10-30s for
# N=130, Hp=8). Subsequent solves reuse the compiled NLP structure and
# only update the parameter values.
#
# Formulation: multiple-shooting.
#   Decision variables: u[k] (N,) for k=0..Hp-1   → N*Hp variables
#   Shooting states:    x1[k], x5[k] for k=0..Hp  → 2*N*(Hp+1) variables
#   Equality constraints: dynamics x1[k+1]=f(...), x5[k+1]=f(...)
#   Inequality constraints: global budget Σu ≤ W_remaining
#   Box constraints: 0 ≤ u ≤ UB, x1 ≥ 0, x5 ≥ 0
#
# v3.0: MX-based NLP construction.
#   The dynamics step_fn is built with SX (scalar expressions, efficient for
#   element-wise cascade routing). The NLP variables and horizon unrolling use
#   MX (matrix expressions). When an SX-based ca.Function is called with MX
#   arguments, CasADi keeps each call as a single opaque graph node instead
#   of inlining the full expression tree. For Hp=8 this means IPOPT sees 8
#   compact function-call nodes rather than 8 copies of the 130-agent cascade
#   → dramatically smaller NLP graph, faster AD, faster solves.
# =============================================================================

import time

import casadi as ca
import numpy as np

from src.mpc.dynamics_sym import build_dynamics_function
from src.mpc.cost import build_cost_components, compute_cost


def build_nlp(terrain, crop, Hp, sink_agents, weights=None, refs=None,
              ub_mm_per_day=12.0, use_smooth=False, verbose=True):
    """Build the CasADi NLP structure for the MPC.

    Parameters
    ----------
    terrain : dict
    crop : dict
    Hp : int
        Prediction horizon (days).
    sink_agents : list[int]
    weights : dict, optional
    refs : dict, optional
    ub_mm_per_day : float
    use_smooth : bool
    verbose : bool

    Returns
    -------
    nlp_data : dict
        Keys: 'solver', 'lbx', 'ubx', 'lbg', 'ubg', 'Hp', 'N',
              'n_vars', 'var_indices', 'param_sym', 'components'
    """
    t0 = time.time()
    N = terrain['N']

    if verbose:
        print(f"  Building NLP: N={N}, Hp={Hp}, "
              f"vars={N*Hp + 2*N*(Hp+1)}...")

    # Build the dynamics function (SX-based, encapsulated in ca.Function)
    step_fn = build_dynamics_function(terrain, crop, use_smooth=use_smooth)

    # Cost components
    components = build_cost_components(N, crop, sink_agents, weights, refs)

    # ── Parameters (MX — change each solve) ───────────────────────────────

    x1_init = ca.MX.sym('x1_init', N)
    x5_init = ca.MX.sym('x5_init', N)
    x4_init_mean = ca.MX.sym('x4_init_mean')
    x3_init = ca.MX.sym('x3_init', N)

    W_remaining = ca.MX.sym('W_remaining')

    rain_h   = ca.MX.sym('rain_h', Hp)
    ETc_h    = ca.MX.sym('ETc_h', Hp)
    rad_h    = ca.MX.sym('rad_h', Hp)
    h2_h     = ca.MX.sym('h2_h', Hp)
    h7_h     = ca.MX.sym('h7_h', Hp)
    g_base_h = ca.MX.sym('g_base_h', Hp)

    u_prev = ca.MX.sym('u_prev', N)

    # ── Decision variables (MX) ───────────────────────────────────────────

    opt_vars = []
    lbx = []
    ubx = []
    var_index = {}

    # Controls
    u_vars = []
    for k in range(Hp):
        uk = ca.MX.sym(f'u_{k}', N)
        u_vars.append(uk)
        opt_vars.append(uk)
        lbx += [0.0] * N
        ubx += [ub_mm_per_day] * N
    var_index['u'] = (0, N * Hp)

    # Shooting states for x1 (k=1..Hp)
    x1_vars = []
    x1_start = len(lbx)
    for k in range(Hp):
        x1k = ca.MX.sym(f'x1_{k+1}', N)
        x1_vars.append(x1k)
        opt_vars.append(x1k)
        lbx += [0.0] * N
        ubx += [1e6] * N
    var_index['x1'] = (x1_start, x1_start + N * Hp)

    # Shooting states for x5 (k=1..Hp)
    x5_vars = []
    x5_start = len(lbx)
    for k in range(Hp):
        x5k = ca.MX.sym(f'x5_{k+1}', N)
        x5_vars.append(x5k)
        opt_vars.append(x5k)
        lbx += [0.0] * N
        ubx += [1e6] * N
    var_index['x5'] = (x5_start, x5_start + N * Hp)

    w = ca.vertcat(*opt_vars)
    n_vars = w.shape[0]

    # ── Dynamics constraints (equality) ───────────────────────────────────

    g_eq = []

    x1_traj = [x1_init]
    x5_traj = [x5_init]
    x4_inc_list = []
    x3_current = x3_init

    for k in range(Hp):
        x1_k = x1_traj[k]
        x5_k = x5_traj[k]

        # Call SX-based step_fn with MX arguments → opaque node, not inlined
        result = step_fn(
            x1_k, x5_k, u_vars[k],
            rain_h[k], ETc_h[k], rad_h[k],
            h2_h[k], h7_h[k], g_base_h[k],
        )
        x1_next = result[0]
        x5_next = result[1]
        x4_inc  = result[2]
        h3      = result[3]

        x4_inc_list.append(x4_inc)

        g_eq.append(x1_vars[k] - x1_next)
        g_eq.append(x5_vars[k] - x5_next)

        x1_traj.append(x1_vars[k])
        x5_traj.append(x5_vars[k])

    # ── Budget constraint (inequality) ────────────────────────────────────

    total_water = ca.MX(0)
    for k in range(Hp):
        total_water += ca.sum1(u_vars[k]) / N
    budget_gap = total_water - W_remaining

    # ── Assemble constraints ──────────────────────────────────────────────

    g_all = ca.vertcat(*g_eq, budget_gap)

    n_eq = sum(g.shape[0] for g in g_eq)
    lbg = [0.0] * n_eq + [-1e20]
    ubg = [0.0] * n_eq + [0.0]

    # ── Cost function ─────────────────────────────────────────────────────

    x4_terminal_mean = x4_init_mean
    for k in range(Hp):
        x4_terminal_mean = x4_terminal_mean + ca.sum1(x4_inc_list[k]) / N

    u_traj_with_prev = [u_prev] + u_vars

    J, cost_terms = compute_cost(
        u_trajectory=u_traj_with_prev[1:],
        x1_trajectory=[x1_traj[k+1] for k in range(Hp)],
        x5_trajectory=[x5_traj[k+1] for k in range(Hp)],
        x4_terminal_mean=x4_terminal_mean,
        components=components,
        Hp=Hp,
        use_smooth=use_smooth,
    )

    # Δu from u_prev to u[0]
    diff0 = u_vars[0] - u_prev
    J_delta_u_prev = components['weights']['alpha5'] * ca.dot(diff0, diff0) / (
        components['refs']['u_max']**2 * N)
    J = J + J_delta_u_prev

    # ── Collect all parameters ────────────────────────────────────────────

    p = ca.vertcat(
        x1_init, x5_init, x4_init_mean, x3_init,
        W_remaining,
        rain_h, ETc_h, rad_h, h2_h, h7_h, g_base_h,
        u_prev,
    )

    # ── Build IPOPT solver ────────────────────────────────────────────────

    nlp = {'x': w, 'f': J, 'g': g_all, 'p': p}

    opts = {
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-4,
        'ipopt.acceptable_tol': 1e-3,
        'ipopt.acceptable_iter': 10,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.linear_solver': 'mumps',
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_init': 1e-2,
        'ipopt.nlp_scaling_method': 'gradient-based',
    }

    solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)

    build_time = time.time() - t0
    if verbose:
        print(f"  NLP built: {n_vars} variables, {len(lbg)} constraints, "
              f"{build_time:.1f}s")

    return {
        'solver': solver,
        'lbx': lbx,
        'ubx': ubx,
        'lbg': lbg,
        'ubg': ubg,
        'Hp': Hp,
        'N': N,
        'n_vars': n_vars,
        'var_index': var_index,
        'components': components,
    }


def solve_step(nlp_data, x1_current, x5_current, x4_mean_current, x3_current,
               budget_remaining, forecast_climate, precomputed, u_prev,
               warm_x0=None):
    """Solve the NLP for one MPC step.

    Parameters
    ----------
    nlp_data : dict
        From build_nlp.
    x1_current, x5_current : np.ndarray (N,)
    x4_mean_current : float
    x3_current : np.ndarray (N,)
    budget_remaining : float
    forecast_climate : dict
        'rainfall': (Hp,), 'ETc': (Hp,), 'radiation': (Hp,)
    precomputed : Precomputed
        From src.precompute.get_precomputed. Provides h2, h7, g_base arrays.
    u_prev : np.ndarray (N,)
        Previous day's irrigation action (for Δu cost).
    warm_x0 : np.ndarray, optional
        Warm-start decision variable vector.

    Returns
    -------
    u_optimal : np.ndarray (N,)
        First-step optimal irrigation action.
    solve_info : dict
        'solve_time_ms', 'status', 'cost', 'warm_x0_next'
    """
    solver = nlp_data['solver']
    N = nlp_data['N']
    Hp = nlp_data['Hp']

    # Build parameter vector
    rain_h = np.asarray(forecast_climate['rainfall'][:Hp], dtype=float)
    ETc_h = np.asarray(forecast_climate['ETc'][:Hp], dtype=float)
    rad_h = np.asarray(forecast_climate['radiation'][:Hp], dtype=float)
    h2_h = np.asarray(forecast_climate['h2'][:Hp], dtype=float)
    h7_h = np.asarray(forecast_climate['h7'][:Hp], dtype=float)
    g_base_h = np.asarray(forecast_climate['g_base'][:Hp], dtype=float)

    p_val = np.concatenate([
        x1_current, x5_current, [x4_mean_current], x3_current,
        [budget_remaining],
        rain_h, ETc_h, rad_h, h2_h, h7_h, g_base_h,
        u_prev,
    ])

    # Initial guess
    if warm_x0 is None:
        u_init = np.full(N * Hp, 2.0)
        x1_init = np.tile(x1_current, Hp)
        x5_init = np.tile(x5_current, Hp)
        x0 = np.concatenate([u_init, x1_init, x5_init])
    else:
        x0 = warm_x0

    t0 = time.time()
    sol = solver(
        x0=x0,
        lbx=nlp_data['lbx'],
        ubx=nlp_data['ubx'],
        lbg=nlp_data['lbg'],
        ubg=nlp_data['ubg'],
        p=p_val,
    )
    solve_time_ms = (time.time() - t0) * 1000

    x_opt = np.asarray(sol['x']).flatten()
    status = solver.stats()['return_status']
    cost = float(sol['f'])

    u_start, u_end = nlp_data['var_index']['u']
    u_all = x_opt[u_start:u_end].reshape(Hp, N)
    u_optimal = u_all[0]

    warm_x0_next = _shift_warm_start(x_opt, N, Hp, u_optimal)

    return u_optimal, {
        'solve_time_ms': solve_time_ms,
        'status': status,
        'cost': cost,
        'warm_x0_next': warm_x0_next,
    }


def _shift_warm_start(x_opt, N, Hp, u_last_fallback):
    """Shift the solution by one time step for warm-starting the next solve."""
    u_block = x_opt[:N*Hp].reshape(Hp, N)
    x1_block = x_opt[N*Hp:N*Hp + N*Hp].reshape(Hp, N)
    x5_block = x_opt[N*Hp + N*Hp:].reshape(Hp, N)

    u_shifted = np.vstack([u_block[1:], u_block[-1:]])
    x1_shifted = np.vstack([x1_block[1:], x1_block[-1:]])
    x5_shifted = np.vstack([x5_block[1:], x5_block[-1:]])

    return np.concatenate([
        u_shifted.flatten(),
        x1_shifted.flatten(),
        x5_shifted.flatten(),
    ])
