# =============================================================================
# src/mpc/controller.py
# MPCController — implements the Controller interface (src.controllers.base).
#
# Wraps the NLP solver, forecast provider, and precomputed data into
# a clean step-by-step controller that the runner can call identically
# to no_irrigation or fixed_schedule.
#
# At each daily step:
#   1. Build a forecast for the next Hp days
#   2. Solve the NLP (warm-started from previous solution)
#   3. Return the first-step action u[0]
#   4. Update internal tracking (x3, x4, warm start, solve times)
# =============================================================================

import numpy as np

from src.controllers.base import Controller
from src.mpc.solver import build_nlp, solve_step
from src.terrain import get_sink_agents
from src.precompute import get_precomputed


class MPCController(Controller):
    """Model Predictive Controller using CasADi + IPOPT.

    Parameters
    ----------
    Hp : int
        Prediction horizon (days). Default 8.
    weights : dict, optional
        Cost function weights. Default from ARCHITECTURE.md.
    ub_mm_per_day : float
        Actuator cap. Default 12.0.
    use_smooth : bool
        Use smooth approximations for non-differentiable ops. Default False.
    forecast_mode : str
        'perfect' or 'noisy'. Default 'perfect'.
    noise_sigma : float
        Noise base for noisy forecast. Default 0.15.
    noise_seed : int or None
        Seed for noisy forecast. Default None.
    verbose : bool
        Print build and per-step info. Default True.
    """

    def __init__(self, Hp=8, weights=None, ub_mm_per_day=12.0,
                 use_smooth=True, forecast_mode='perfect',
                 noise_sigma=0.15, noise_seed=None, verbose=True):
        name = f"mpc_{forecast_mode}_Hp{Hp}"
        super().__init__(name=name)

        self.Hp = Hp
        self.weights = weights
        self.ub_mm_per_day = ub_mm_per_day
        self.use_smooth = use_smooth
        self.forecast_mode = forecast_mode
        self.noise_sigma = noise_sigma
        self.noise_seed = noise_seed
        self.verbose = verbose

        # Filled in reset()
        self._nlp_data = None
        self._precomputed = None
        self._climate = None
        self._crop = None
        self._terrain = None
        self._forecast_provider = None
        self._warm_x0 = None
        self._u_prev = None
        self._x3_current = None
        self._x4_mean_current = None
        self._solve_times = []

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._terrain = terrain
        self._crop = crop
        self._solve_times = []

        N = terrain['N']
        sink_agents = get_sink_agents(terrain)

        # Build NLP (expensive, ~10-30s)
        if self.verbose:
            print(f"  [{self.name}] Building NLP for Hp={self.Hp}...")
        self._nlp_data = build_nlp(
            terrain=terrain,
            crop=crop,
            Hp=self.Hp,
            sink_agents=sink_agents,
            weights=self.weights,
            ub_mm_per_day=self.ub_mm_per_day,
            use_smooth=self.use_smooth,
            verbose=self.verbose,
        )

        # Load precomputed data
        if scenario_name:
            self._precomputed = get_precomputed(scenario_name, crop['name'].lower())
        else:
            self._precomputed = get_precomputed('dry', crop['name'].lower())

        # Set up forecast provider
        if self.forecast_mode == 'perfect':
            from src.forecast import PerfectForecast
            self._forecast_provider = PerfectForecast()
        elif self.forecast_mode == 'noisy':
            from src.forecast import NoisyForecast
            self._forecast_provider = NoisyForecast(
                sigma_base=self.noise_sigma, seed=self.noise_seed
            )
        else:
            raise ValueError(f"Unknown forecast_mode: {self.forecast_mode}")

        # Initial tracking
        fc_total = crop['theta6'] * crop['theta5']
        self._u_prev = np.zeros(N)
        self._x3_current = np.zeros(N)
        self._x4_mean_current = float(crop.get('x4_init', 0.0))
        self._warm_x0 = None

    def set_climate(self, climate):
        """Provide the full-season climate dict to the controller.

        Must be called after reset() and before the first step().
        The runner calls this when using forecast providers that need
        access to the true climate (for perfect or noisy forecasts).
        """
        self._climate = climate

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Solve the MPC NLP and return the first-step action.

        The `forecast` argument from the runner is ignored — the MPC builds
        its own forecast from the full climate array via its forecast provider.
        """
        N = self._terrain['N']

        x1_current = np.asarray(state['x1'], dtype=float)
        x5_current = np.asarray(state['x5'], dtype=float)

        # Build forecast using the internal provider
        if self._climate is None:
            raise RuntimeError(
                "MPCController.set_climate() must be called before step(). "
                "The MPC needs the full-season climate for its forecast provider."
            )

        forecast_climate = self._forecast_provider(
            day, self._climate, self._precomputed, self.Hp
        )

        # Solve
        u_optimal, info = solve_step(
            nlp_data=self._nlp_data,
            x1_current=x1_current,
            x5_current=x5_current,
            x4_mean_current=self._x4_mean_current,
            x3_current=self._x3_current,
            budget_remaining=budget_remaining,
            forecast_climate=forecast_climate,
            precomputed=self._precomputed,
            u_prev=self._u_prev,
            warm_x0=self._warm_x0,
        )

        self._solve_times.append(info['solve_time_ms'])

        if self.verbose and (day % 10 == 0 or day == 0):
            print(f"    day {day:3d}: solve {info['solve_time_ms']:6.0f}ms "
                  f"status={info['status']} cost={info['cost']:.4f} "
                  f"u_mean={u_optimal.mean():.2f}mm")

        # Handle solver failures gracefully
        if 'Infeasible' in info['status']:
            if self.verbose:
                print(f"    WARNING: IPOPT infeasible at day {day}, "
                      f"falling back to zero irrigation")
            u_optimal = np.zeros(N)

        # Update tracking for next step
        self._u_prev = u_optimal.copy()
        self._warm_x0 = info['warm_x0_next']

        # Read x3 and x4 from the true ABM state (not approximated).
        # The runner provides the actual state each day, so we track it
        # exactly rather than accumulating drift from approximations.
        self._x4_mean_current = float(state['x4'].mean())
        self._x3_current = np.asarray(state['x3'], dtype=float).copy()

        return u_optimal

    @property
    def solve_times(self):
        """List of per-step solve times in ms."""
        return list(self._solve_times)
