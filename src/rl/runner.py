# =============================================================================
# src/rl/runner.py
# Inference runner for trained SAC models.
#
# Loads a trained SB3 SAC model (CTDESACPolicy), runs it through the full
# ABM season, and saves results in the same parquet + JSON sidecar format
# as the MPC and baseline controllers. This enables direct comparison.
#
# IMPORTANT: All normalization constants (X4_REF, X5_REF, UB_MM) and the
# default forecast horizon are imported from src.rl.gym_env to guarantee
# consistency between training-time observations and inference-time
# observations. A previous version of this file hardcoded X5_REF=10 here
# while the gym_env used X5_REF=50, producing a silent training/inference
# mismatch on the surface-ponding feature. This is now eliminated by
# single-sourcing the constants.
# =============================================================================

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.controllers.base import Controller
# Single-source the normalization constants and forecast horizon from the
# training environment so that train-time and inference-time observations
# are guaranteed to match.
from src.rl.gym_env import (
    IrrigationEnv,
    UB_MM,
    X4_REF,
    X5_REF,
)
# Default forecast horizon for SAC inference. Must match the value used
# during training (see IrrigationEnv.__init__ default).
DEFAULT_FORECAST_HORIZON = 8

# Make sure the CTDESACPolicy class is importable so SAC.load() can
# deserialize the policy object that was pickled during training.
from src.rl.networks import CTDESACPolicy  # noqa: F401  (registration side-effect)


class RLController(Controller):
    """Controller wrapping a trained SAC model for inference.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved SB3 model (.zip).
    deterministic : bool
        If True, use the policy mean (no sampling). Default True.
    forecast_horizon : int
        Days of forecast included in the observation. Must match the
        value used during training. Default 8 (= Hp* of the MPC).
    verbose : bool
        Print per-step info every 10 days. Default True.
    """

    def __init__(
        self,
        model_path,
        deterministic=True,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        verbose=True,
    ):
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.forecast_horizon = forecast_horizon
        self.verbose = verbose

        # Load the model (CPU for inference — no GPU needed).
        # The CTDESACPolicy import above registers the class for
        # SB3's deserialization machinery.
        self.model = SAC.load(str(self.model_path), device='cpu')

        # Will be set in reset()
        self._inference_times = []

        name = f"sac_{'det' if deterministic else 'stoch'}"
        super().__init__(name=name)

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        """Initialize internal state for observation construction."""
        self._inference_times = []

        self._terrain = terrain
        self._crop = crop
        self._N = terrain['N']

        # Precomputed climate-only quantities (same as gym_env)
        from src.precompute import get_precomputed
        from climate_data import load_cleaned_data, extract_scenario_by_name

        scenario = scenario_name or 'dry'
        self._precomputed = get_precomputed(scenario, crop['name'].lower())
        df = load_cleaned_data()
        self._climate = extract_scenario_by_name(df, scenario, crop)

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
        """Accept full-season climate (for compatibility with the runner)."""
        self._climate = climate

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        """Compute action from the trained SAC policy.

        Parameters
        ----------
        day : int
        state : dict with x1, x2, x3, x4, x5
        climate_today : dict
        budget_remaining : float
        forecast : ignored (the SAC policy reads its forecast from
            self._climate via the precomputed window in _build_obs)

        Returns
        -------
        u : np.ndarray (N,)
            Irrigation action in mm/day.
        """
        t0 = time.time()

        obs = self._build_obs(day, state, budget_remaining)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)

        # Scale [0, 1] → [0, UB_MM] mm/day
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
        """Construct the 660-dim observation vector.

        This MUST produce the same layout and normalization as
        IrrigationEnv._get_obs() in src/rl/gym_env.py. All normalization
        constants are imported from gym_env to guarantee consistency.
        """
        N = self._N
        fc = self._fc_total

        x1_norm = state['x1'] / fc
        x5_norm = state['x5'] / X5_REF
        x4_norm = state['x4'] / X4_REF
        x3 = state['x3']
        elev = self._elev_norm

        day_frac = day / self._season_days
        budget_frac = budget_remaining / max(self._budget_total, 1e-6)
        daily_budget = self._budget_total / self._season_days
        burn_rate = (
            (self._water_used / max(day, 1)) / max(daily_budget, 1e-6)
            if day > 0 else 0.0
        )

        d = min(day, self._season_days - 1)
        rain_today = float(self._climate['rainfall'][d])
        ETc_today = float(self._precomputed.Kc_ET[d])

        end = min(d + self.forecast_horizon, self._season_days)
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
