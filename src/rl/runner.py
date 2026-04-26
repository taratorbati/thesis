# =============================================================================
# src/rl/runner.py
# Inference runner for trained SAC models.
#
# Loads a trained SB3 SAC model, runs it through the full ABM season, and
# saves results in the same parquet + JSON sidecar format as MPC and
# baseline controllers. This enables direct comparison in Step F.
#
# The runner wraps the trained model as a Controller subclass so it can
# be used with the standard run_season() loop.
# =============================================================================

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.controllers.base import Controller
from src.rl.gym_env import IrrigationEnv, UB_MM


class RLController(Controller):
    """Controller that wraps a trained SAC model for inference.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved SB3 model (.zip).
    deterministic : bool
        If True, use the mean action (no sampling). Default True.
    verbose : bool
        Print per-step info. Default True.
    """

    def __init__(self, model_path, deterministic=True, verbose=True):
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.verbose = verbose

        # Load the model (CPU for inference — no GPU needed)
        self.model = SAC.load(str(self.model_path), device='cpu')

        # Will be set in reset()
        self._env_helper = None
        self._inference_times = []

        name = f"sac_{'det' if deterministic else 'stoch'}"
        super().__init__(name=name)

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        """Initialize the internal env helper for observation construction."""
        self._inference_times = []

        # We need an IrrigationEnv instance to build observations.
        # We don't actually step this env — we only use _get_obs().
        # The real ABM is managed by run_season().
        self._terrain = terrain
        self._crop = crop
        self._N = terrain['N']

        # Precomputed data and climate are needed for observations
        from src.precompute import get_precomputed
        from climate_data import load_cleaned_data, extract_scenario_by_name

        self._precomputed = get_precomputed(scenario_name or 'dry', crop['name'].lower())
        df = load_cleaned_data()
        self._climate = extract_scenario_by_name(df, scenario_name or 'dry', crop)

        # Normalization constants
        self._fc_total = crop['theta6'] * crop['theta5']
        self._elev_norm = terrain['gamma_flat']
        self._season_days = season_days
        self._budget_total = budget_total
        self._budget_remaining = float(budget_total)
        self._water_used = 0.0
        self._u_prev = np.zeros(self._N)
        self._day_counter = 0

    def set_climate(self, climate):
        """Accept full-season climate (for compatibility with runner)."""
        self._climate = climate

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Compute action from the trained SAC policy.

        Parameters
        ----------
        day : int
        state : dict with x1, x2, x3, x4, x5
        climate_today : dict
        budget_remaining : float
        forecast : ignored

        Returns
        -------
        u : np.ndarray (N,)
            Irrigation action in mm/day.
        """
        t0 = time.time()

        # Build observation (same structure as gym_env.py)
        obs = self._build_obs(day, state, budget_remaining)

        # Get action from trained model
        action, _ = self.model.predict(obs, deterministic=self.deterministic)

        # Scale [0,1] → [0, UB] mm/day
        u = np.asarray(action, dtype=float).clip(0, 1) * UB_MM

        inference_ms = (time.time() - t0) * 1000
        self._inference_times.append(inference_ms)

        if self.verbose and (day % 10 == 0):
            print(f"    day {day:3d}: inference {inference_ms:.1f}ms "
                  f"u_mean={u.mean():.2f}mm")

        # Track water use for observation construction
        self._water_used += u.mean()
        self._u_prev = u.copy()
        self._day_counter = day + 1

        return u

    def _build_obs(self, day, state, budget_remaining):
        """Construct the 660-dim observation vector."""
        N = self._N
        fc = self._fc_total

        x1_norm = state['x1'] / fc
        x5_norm = state['x5'] / 10.0  # X5_REF
        x4_norm = state['x4'] / 900.0  # X4_REF
        x3 = state['x3']
        elev = self._elev_norm

        day_frac = day / self._season_days
        budget_frac = budget_remaining / max(self._budget_total, 1e-6)
        daily_budget = self._budget_total / self._season_days
        burn_rate = (self._water_used / max(day, 1)) / max(daily_budget, 1e-6) if day > 0 else 0.0

        d = min(day, self._season_days - 1)
        rain_today = float(self._climate['rainfall'][d])
        ETc_today = float(self._precomputed.Kc_ET[d])

        end = min(d + 3, self._season_days)
        rain_forecast = float(self._climate['rainfall'][d:end].sum()) if end > d else 0.0
        ETc_forecast = float(self._precomputed.Kc_ET[d:end].sum()) if end > d else 0.0

        h2_today = float(self._precomputed.h2[d])
        h7_today = float(self._precomputed.h7[d])
        g_base_today = float(self._precomputed.g_base[d])

        scalars = np.array([
            day_frac, budget_frac, burn_rate,
            rain_today, rain_forecast, ETc_today, ETc_forecast,
            h2_today, h7_today, g_base_today,
        ], dtype=np.float32)

        obs = np.concatenate([
            x1_norm, x5_norm, x4_norm, x3, elev, scalars
        ]).astype(np.float32)

        return obs

    @property
    def solve_times(self):
        """Inference times in ms (for compatibility with MPC metadata)."""
        return list(self._inference_times)
