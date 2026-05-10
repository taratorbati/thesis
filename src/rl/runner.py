# =============================================================================
# src/rl/runner.py
# Inference runner for trained SAC models.
#
# Loads a trained SB3 SAC model (CTDESACPolicy), runs it through the full
# ABM season, and saves results in the same format as the MPC runner.
#
# CRITICAL: _build_obs() MUST produce the exact same 707-dim observation
# as IrrigationEnv._get_obs() in src/rl/gym_env.py. The scalar sub-vector
# order is defined in gym_env.py with explicit index comments — any change
# there requires the same change here.
#
# burn_rate is derived from (budget_total - budget_remaining) / budget_total
# rather than from an internal _water_used accumulator. This avoids drift
# between the runner's internal accounting and the outer runner's clipped/
# scaled budget deductions (src/runner.py clips u before deducting from
# budget_remaining, so a self._water_used accumulator would diverge silently).
# =============================================================================

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.controllers.base import Controller
from src.rl.gym_env import (
    UB_MM,
    X4_REF,
    X5_REF,
    FULL_SEASON_NEED_MM,  # float scalar: 484.0 mm
)
from src.rl.networks import CTDESACPolicy  # noqa: F401 (registration side-effect)

DEFAULT_FORECAST_HORIZON = 8


class RLController(Controller):
    """Controller wrapping a trained SAC model for inference.

    Parameters
    ----------
    model_path : str or Path
    deterministic : bool
    forecast_horizon : int
        Must match training. Default 8.
    verbose : bool
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
        self.model = SAC.load(str(self.model_path), device='cpu')
        self._inference_times = []
        super().__init__(name=f"sac_{'det' if deterministic else 'stoch'}")

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._inference_times = []
        self._terrain = terrain
        self._crop = crop
        self._N = terrain['N']
        self._season_days = season_days
        self._budget_total = float(budget_total)
        self._elev_norm = terrain['gamma_flat']
        self._fc_total = crop['theta6'] * crop['theta5']
        self._u_prev = np.zeros(self._N)

        from src.precompute import get_precomputed
        from climate_data import load_cleaned_data, extract_scenario_by_name

        scenario = scenario_name or 'dry'
        self._precomputed = get_precomputed(scenario, crop['name'].lower())
        df = load_cleaned_data()
        self._climate = extract_scenario_by_name(df, scenario, crop)

    def set_climate(self, climate):
        self._climate = climate

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        t0 = time.time()
        obs = self._build_obs(day, state, budget_remaining)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        u = np.asarray(action, dtype=float).clip(0, 1) * UB_MM

        self._inference_times.append((time.time() - t0) * 1000)

        if self.verbose and (day % 10 == 0):
            print(f"    day {day:3d}: inference {self._inference_times[-1]:.1f}ms "
                  f"u_mean={u.mean():.2f}mm")

        self._u_prev = u.copy()
        return u

    def _build_obs(self, day, state, budget_remaining):
        """Construct the 707-dim observation vector.

        Scalar order must exactly match IrrigationEnv._get_obs():
          [0] day_frac
          [1] budget_frac
          [2] budget_total_norm
          [3] burn_rate
          [4] rain_today
          [5] ETc_today
          [6] h2_today
          [7] h7_today
          [8] g_base_today
        """
        N = self._N
        fc = self._fc_total

        x1_norm = state['x1'] / fc
        x5_norm = state['x5'] / X5_REF
        x4_norm = state['x4'] / X4_REF
        x3      = state['x3']
        elev    = self._elev_norm

        day_frac          = day / self._season_days
        budget_frac       = budget_remaining / max(self._budget_total, 1e-6)
        budget_total_norm = self._budget_total / FULL_SEASON_NEED_MM

        # burn_rate: derived from the runner-provided budget_remaining so it
        # stays consistent with the outer runner's clipped budget accounting.
        # water_spent = budget_total - budget_remaining (as tracked by runner)
        water_spent = self._budget_total - float(budget_remaining)
        daily_budget = self._budget_total / self._season_days
        burn_rate = (
            (water_spent / max(day, 1)) / max(daily_budget, 1e-6)
            if day > 0 else 0.0
        )

        d = min(day, self._season_days - 1)
        rain_today   = float(self._climate['rainfall'][d])
        ETc_today    = float(self._precomputed.Kc_ET[d])
        h2_today     = float(self._precomputed.h2[d])
        h7_today     = float(self._precomputed.h7[d])
        g_base_today = float(self._precomputed.g_base[d])

        # Scalar order: MUST match gym_env._get_obs() exactly
        scalars = np.array([
            day_frac,          # [0]
            budget_frac,       # [1]
            budget_total_norm, # [2]
            burn_rate,         # [3]
            rain_today,        # [4]
            ETc_today,         # [5]
            h2_today,          # [6]
            h7_today,          # [7]
            g_base_today,      # [8]
        ], dtype=np.float32)

        H = self.forecast_horizon
        end = min(d + H, self._season_days)

        def _pad(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < H:
                pad_val = arr[-1] if len(arr) > 0 else 0.0
                return np.concatenate([arr,
                    np.full(H - len(arr), pad_val, dtype=np.float32)])
            return arr

        rain_fc = _pad(self._climate['rainfall'][d:end])
        ETc_fc  = _pad(self._precomputed.Kc_ET[d:end])
        rad_fc  = _pad(self._climate['radiation'][d:end])
        h2_fc   = _pad(self._precomputed.h2[d:end])
        h7_fc   = _pad(self._precomputed.h7[d:end])
        g_fc    = _pad(self._precomputed.g_base[d:end])

        return np.concatenate([
            x1_norm, x5_norm, x4_norm, x3, elev,
            scalars,
            rain_fc, ETc_fc, rad_fc, h2_fc, h7_fc, g_fc,
        ]).astype(np.float32)

    @property
    def solve_times(self):
        return list(self._inference_times)
