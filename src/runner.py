# =============================================================================
# src/runner.py
# Generic receding-horizon run loop. Orchestrates the controller-ABM
# interaction for a single crop season and writes results to parquet.
#
# Integration point for every controller. By keeping the loop here, fair
# comparison across controllers is guaranteed: same simulator, same climate,
# same initial state, same budget accounting.
#
# The loop:
#   1. Initialize ABM at FC × root_depth (rice-realistic post-puddling state).
#   2. Reset controller, telling it terrain, crop, season length, budget.
#   3. If the controller has a set_climate() method, call it (MPC needs this).
#   4. Each day:
#        a. Build the controller's daily inputs (state, climate, budget).
#        b. Call controller.step() to get u (per-agent action, mm).
#        c. Clip action against UB and budget_remaining (safety).
#        d. Step the ABM forward one day with (u, climate_today).
#        e. Update budget_remaining -= u.mean()  (field-average mm).
#        f. Append to trajectory; checkpoint every CHECKPOINT_INTERVAL days.
#   4. Save final parquet + JSON metadata; discard partial checkpoint.
# =============================================================================

import time
from pathlib import Path

import numpy as np

from abm import CropSoilABM
from src.persistence import (
    save_run, save_partial, load_partial, discard_partial, should_skip
)


# ── Defaults ─────────────────────────────────────────────────────────────────

UB_MM_PER_DAY = 12.0     # actuator cap per agent per day (mm)
CHECKPOINT_INTERVAL = 10  # save partial run every N days


def run_season(
    controller,
    terrain,
    crop,
    climate,
    budget_total,
    output_path,
    *,
    scenario_name='unknown',
    seed=0,
    runoff_mode='cascade',
    ub_mm_per_day=UB_MM_PER_DAY,
    forecast_provider=None,
    forecast_horizon=14,
    initial_x1=None,
    initial_x5=0.0,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    force=False,
    verbose=True,
):
    """Run one full crop season with the given controller, save to parquet.

    Parameters
    ----------
    controller : src.controllers.base.Controller
    terrain : dict
        Output of src.terrain.load_terrain.
    crop : dict
        Crop parameter dict (from soil_data.get_crop).
    climate : dict
        Output of climate_data.extract_scenario. Must contain at least
        'rainfall', 'temp_mean', 'temp_max', 'radiation', 'ET' arrays
        of length >= crop['season_days'].
    budget_total : float
        Total seasonal water budget, mm summed across the season,
        field-averaged. (E.g. 484 for rice at 100% of full need.)
    output_path : str or Path
        Final destination for the parquet file. The .parquet extension
        is added if missing. A matching .json is written alongside.
    scenario_name : str
        For metadata only. 'dry', 'moderate', 'wet', or anything.
    seed : int
        For metadata only (controller-internal randomness should be
        seeded by the caller, e.g. via numpy or torch).
    runoff_mode : str
        ABM runoff mode: 'cascade' (default), 'simple', or 'none'.
    ub_mm_per_day : float
        Per-agent per-day actuator cap, applied as a safety clip on the
        controller's action.
    forecast_provider : callable or None
        Optional callable f(day, climate, horizon) -> dict that returns
        a forecast dict over the next `horizon` days. If None, the
        controller receives None as its forecast argument. Note: the MPC
        controller manages its own forecast internally via set_climate(),
        so this parameter is typically only used by non-MPC controllers
        that want external forecast access.
    forecast_horizon : int
        Forecast horizon length in days. Ignored if forecast_provider is None.
    initial_x1 : float, np.ndarray, or None
        Initial root-zone soil water (mm). If None, defaults to FC × root_depth.
        Per-agent (shape (N,)) or scalar.
    initial_x5 : float
        Initial surface ponding (mm). Default 0 (no puddling pulse on the
        mountainous Gilan terrain).
    checkpoint_interval : int
        Save a partial-run checkpoint every N days. Set to 0 to disable.
    force : bool
        If True, overwrite an existing output file. Default False (skip).
    verbose : bool
        Print progress info.

    Returns
    -------
    str
        'completed', 'skipped', or 'partial' (if early termination).
    """
    output_path = Path(output_path)
    if output_path.suffix != '.parquet':
        output_path = output_path.with_suffix('.parquet')

    # Skip if already done
    if should_skip(output_path, force=force):
        if verbose:
            print(f"  [skip] {output_path.name} (already exists; use force=True to recompute)")
        return 'skipped'

    # ── Initialize ABM ────────────────────────────────────────────────────────

    N = terrain['N']
    season_days = crop['season_days']
    fc_total = crop['theta6'] * crop['theta5']  # mm field capacity in root zone

    abm = CropSoilABM(
        gamma_flat=terrain['gamma_flat'],
        sends_to=terrain['sends_to'],
        Nr=terrain['Nr'],
        theta=crop,
        N=N,
        runoff_mode=runoff_mode,
        elevation=terrain['elevation_flat'],
    )
    abm.reset()  # use the ABM's internal defaults...

    # ...then override the initial state to our explicit values.
    if initial_x1 is None:
        initial_x1 = fc_total
    abm.x1 = np.full(N, initial_x1) if np.isscalar(initial_x1) else np.asarray(initial_x1, dtype=float).copy()
    abm.x2 = np.full(N, crop.get('x2_init', 0.0))
    abm.x3 = np.zeros(N)
    abm.x4 = np.full(N, crop.get('x4_init', 0.0))
    abm.x5 = np.full(N, initial_x5)

    # ── Initialize controller ─────────────────────────────────────────────────

    controller.reset(
        terrain=terrain, crop=crop, season_days=season_days,
        budget_total=budget_total, scenario_name=scenario_name,
    )

    # If the controller needs full-season climate access (e.g. MPC for forecasts)
    if hasattr(controller, 'set_climate'):
        controller.set_climate(climate)

    # ── Allocate trajectory storage ───────────────────────────────────────────

    trajectory = {
        'x1': np.zeros((season_days, N), dtype=float),
        'x2': np.zeros((season_days, N), dtype=float),
        'x3': np.zeros((season_days, N), dtype=float),
        'x4': np.zeros((season_days, N), dtype=float),
        'x5': np.zeros((season_days, N), dtype=float),
        'u':  np.zeros((season_days, N), dtype=float),
        'rainfall':         np.zeros(season_days, dtype=float),
        'et0':              np.zeros(season_days, dtype=float),
        'budget_remaining': np.zeros(season_days, dtype=float),
    }

    # ── Receding-horizon loop ─────────────────────────────────────────────────

    budget_remaining = float(budget_total)
    wallclock_start = time.time()

    for day in range(season_days):
        # Pre-step: snapshot state for the controller and the trajectory
        state = {
            'x1': abm.x1.copy(),
            'x2': abm.x2.copy(),
            'x3': abm.x3.copy(),
            'x4': abm.x4.copy(),
            'x5': abm.x5.copy(),
        }

        climate_today = {
            'rainfall':  float(climate['rainfall'][day]),
            'temp_mean': float(climate['temp_mean'][day]),
            'temp_max':  float(climate['temp_max'][day]),
            'radiation': float(climate['radiation'][day]),
            'ET':        float(climate['ET'][day]),
        }

        forecast = None
        if forecast_provider is not None:
            forecast = forecast_provider(day, climate, forecast_horizon)

        # Get controller's action
        u = controller.step(
            day=day,
            state=state,
            climate_today=climate_today,
            budget_remaining=budget_remaining,
            forecast=forecast,
        )
        u = np.asarray(u, dtype=float)
        if u.shape != (N,):
            raise ValueError(
                f"Controller {controller.name} returned action of shape {u.shape}, "
                f"expected ({N},)"
            )

        # Safety clips: actuator box + global budget
        u = np.clip(u, 0.0, ub_mm_per_day)
        # Field-averaged spend = u.mean(); cap so we don't bust the seasonal budget
        if u.mean() > budget_remaining:
            scale = budget_remaining / max(u.mean(), 1e-12)
            u = u * scale

        # Step ABM
        new_state = abm.step(u, climate_today)

        # Record trajectory (post-step state, action that produced it, budget after spending)
        budget_remaining = max(budget_remaining - float(u.mean()), 0.0)

        trajectory['x1'][day] = new_state['x1']
        trajectory['x2'][day] = new_state['x2']
        trajectory['x3'][day] = new_state['x3']
        trajectory['x4'][day] = new_state['x4']
        trajectory['x5'][day] = new_state['x5']
        trajectory['u'][day]  = u
        trajectory['rainfall'][day]         = climate_today['rainfall']
        trajectory['et0'][day]              = climate_today['ET']
        trajectory['budget_remaining'][day] = budget_remaining

        # Checkpoint
        if checkpoint_interval > 0 and (day + 1) % checkpoint_interval == 0 and (day + 1) < season_days:
            partial = {k: v[:day + 1] for k, v in trajectory.items()}
            save_partial(output_path, partial, day_completed=day, metadata={
                'scenario': scenario_name,
                'crop': crop['name'],
                'controller': controller.name,
                'budget_total': budget_total,
                'seed': seed,
            })

    # ── Finalize ──────────────────────────────────────────────────────────────

    wallclock_seconds = time.time() - wallclock_start

    final_metrics = _compute_final_metrics(trajectory, crop, terrain, budget_total)

    # Collect solve times from the controller if available (MPC)
    solve_times_list = None
    if hasattr(controller, 'solve_times'):
        solve_times_list = controller.solve_times

    metadata = {
        'scenario':           scenario_name,
        'year':               int(climate.get('year', -1)) if 'year' in climate else None,
        'crop':               crop['name'],
        'controller':         controller.name,
        'runoff_mode':        runoff_mode,
        'budget_total':       float(budget_total),
        'ub_mm_per_day':      float(ub_mm_per_day),
        'season_days':        int(season_days),
        'N_agents':           int(N),
        'initial_x1':         float(np.mean(np.asarray(initial_x1) if not np.isscalar(initial_x1) else np.full(1, initial_x1))),
        'initial_x5':         float(initial_x5),
        'seed':               int(seed),
        'forecast_horizon':   int(forecast_horizon) if forecast_provider is not None else 0,
        'wallclock_seconds':  float(wallclock_seconds),
        'final_metrics':      final_metrics,
    }
    if solve_times_list is not None:
        metadata['solve_times'] = solve_times_list
        metadata['solve_time_mean_ms'] = float(np.mean(solve_times_list))
        metadata['solve_time_max_ms'] = float(np.max(solve_times_list))

    save_run(output_path, trajectory, metadata)
    discard_partial(output_path)

    if verbose:
        solve_info = ''
        if solve_times_list is not None:
            solve_info = (f" solve_mean={np.mean(solve_times_list):.0f}ms"
                          f" solve_max={np.max(solve_times_list):.0f}ms")
        print(f"  [done] {output_path.name} "
              f"yield={final_metrics['yield_kg_ha']:.0f} kg/ha "
              f"water={final_metrics['water_used_mm']:.1f} mm "
              f"({wallclock_seconds:.1f}s){solve_info}")

    return 'completed'


# ── Final-metrics helper ──────────────────────────────────────────────────────

def _compute_final_metrics(trajectory, crop, terrain, budget_total):
    """Compute the metrics dict that ends up in the JSON metadata sidecar."""
    HI = crop.get('HI', 0.45)
    fc_total = crop['theta6'] * crop['theta5']
    p = crop.get('p', 0.20)
    wp_total = crop['theta2'] * crop['theta5']
    raw = p * (fc_total - wp_total)
    stress_threshold = fc_total - raw

    sink_agents = [n for n in range(terrain['N']) if terrain['Nr'][n] == 0]
    n_sinks = max(len(sink_agents), 1)

    x1 = trajectory['x1']
    x4 = trajectory['x4']
    x5 = trajectory['x5']
    u  = trajectory['u']

    terminal_biomass_g_m2 = float(x4[-1].mean())
    yield_kg_ha           = terminal_biomass_g_m2 * HI * 10.0  # g/m² × HI × 10 = kg/ha
    water_used_mm         = float(u.sum() / terrain['N'])      # mean over agents of total seasonal mm
    wue_kg_ha_per_mm      = yield_kg_ha / water_used_mm if water_used_mm > 0 else 0.0
    budget_compliance     = 1 if water_used_mm <= budget_total + 1e-6 else 0

    drought_days_per_agent  = float((x1 < stress_threshold).sum() / terrain['N'])
    waterlog_days_per_agent = float((x1 > fc_total).sum() / terrain['N'])

    sink_pond_days = float((x5[:, sink_agents] > 5.0).sum() / n_sinks)
    sink_x5_max    = float(x5[:, sink_agents].max()) if sink_agents else 0.0

    final_x4 = x4[-1]
    spatial_equity_cv = float(final_x4.std() / final_x4.mean()) if final_x4.mean() > 0 else 0.0

    return {
        'terminal_biomass_g_m2':   terminal_biomass_g_m2,
        'yield_kg_ha':             yield_kg_ha,
        'water_used_mm':           water_used_mm,
        'wue_kg_ha_per_mm':        wue_kg_ha_per_mm,
        'budget_compliance':       int(budget_compliance),
        'drought_days_per_agent':  drought_days_per_agent,
        'waterlog_days_per_agent': waterlog_days_per_agent,
        'sink_pond_days':          sink_pond_days,
        'sink_x5_max_mm':          sink_x5_max,
        'spatial_equity_cv':       spatial_equity_cv,
    }
