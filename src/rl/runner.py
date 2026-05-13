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
#
# Noisy forecast support (v2.4.3):
#   forecast_mode='noisy' injects AR(1)-correlated multiplicative noise
#   into the 48 forecast dims of the observation (rain[H], ETc[H],
#   rad[H], h2[H], h7[H], g[H]). The ABM transition always uses the true
#   climate — only the information presented to the policy is corrupted.
#   This evaluates policy robustness to realistic NWP forecast errors
#   without retraining (Chapter 5 disturbance analysis).
#   See src/forecast.py:NoisyForecast for the noise model details.
#
#   Why this is the correct comparison with MPC noisy mode:
#   - MPC noisy: the NLP receives corrupted rainfall/ETc as its forecast
#     input at each receding-horizon step, so it optimizes for the wrong
#     future and produces sub-optimal actions.
#   - SAC noisy: the policy network receives corrupted forecast features
#     (positions 659-706 in the 707-dim obs) and may produce sub-optimal
#     actions because its learned input-to-action mapping was trained on
#     perfect forecasts.
#   - In both cases the ABM advances with true climate, the same noise seed
#     is used, and results are compared to each controller's own perfect-
#     forecast baseline. This isolates the effect of forecast quality on
#     policy performance.
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
    forecast_mode : str
        'perfect' (default) or 'noisy'.
        - 'perfect': true future climate is shown to the policy in the
          forecast block of the observation. This is what the policy was
          trained on and is the primary evaluation mode.
        - 'noisy': AR(1)-correlated multiplicative noise is applied to
          rainfall and ETc in the 8-day forecast block before the obs is
          passed to the policy. The ABM transition still uses true climate.
          Used for the Chapter 5 disturbance robustness analysis.
    noise_sigma : float
        Base noise level (std at 1-day lead). Default 0.15 (15%).
        Ignored when forecast_mode='perfect'.
    noise_rho : float
        AR(1) persistence parameter in [0, 1). Default 0.6.
        Ignored when forecast_mode='perfect'.
    noise_seed : int or None
        RNG seed for NoisyForecast. Set to the same value used for MPC
        noisy evaluation (default 42) so that performance differences
        between controllers are attributable to policy quality, not to
        different noise realizations. Ignored when forecast_mode='perfect'.
    verbose : bool
    """

    def __init__(
        self,
        model_path,
        deterministic=True,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        forecast_mode='perfect',
        noise_sigma=0.15,
        noise_rho=0.6,
        noise_seed=None,
        verbose=True,
    ):
        if forecast_mode not in ('perfect', 'noisy'):
            raise ValueError(
                f"forecast_mode must be 'perfect' or 'noisy', got {forecast_mode!r}"
            )
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.forecast_horizon = forecast_horizon
        self.forecast_mode = forecast_mode
        self.noise_sigma = noise_sigma
        self.noise_rho = noise_rho
        self.noise_seed = noise_seed
        self.verbose = verbose
        self.model = SAC.load(str(self.model_path), device='cpu')
        self._inference_times = []
        self._noisy_forecast = None   # initialized in reset()

        name = f"sac_{'det' if deterministic else 'stoch'}_{forecast_mode}"
        super().__init__(name=name)

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

        # Initialize noisy forecast provider if needed.
        # NoisyForecast is re-seeded at each reset() so every episode gets
        # an independent but reproducible noise trajectory. With a fixed
        # noise_seed, the same trajectory is used for every scenario/budget
        # cell, ensuring that MPC vs SAC performance differences are not
        # due to different noise draws.
        if self.forecast_mode == 'noisy':
            from src.forecast import NoisyForecast
            self._noisy_forecast = NoisyForecast(
                sigma_base=self.noise_sigma,
                rho=self.noise_rho,
                seed=self.noise_seed,
            )
        else:
            self._noisy_forecast = None

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
                  f"u_mean={u.mean():.2f}mm "
                  f"[{self.forecast_mode} forecast]")

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

        Forecast block (48 dims = 6 vars x 8 days, positions 659-706):
          forecast_mode='perfect': true future climate slices (same as
            training). All 6 variables use real data.
          forecast_mode='noisy': AR(1)-correlated noise applied to
            rainfall and ETc. h2, h7, g_base, radiation remain perfect
            because NoisyForecast only corrupts rainfall and ETc (matching
            the MPC's noise model in src/forecast.py), and because these
            temperature/radiation-derived quantities have smaller forecast
            errors than precipitation in operational NWP systems.
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

        H   = self.forecast_horizon
        end = min(d + H, self._season_days)

        def _pad(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < H:
                pad_val = arr[-1] if len(arr) > 0 else 0.0
                return np.concatenate([arr,
                    np.full(H - len(arr), pad_val, dtype=np.float32)])
            return arr

        if self._noisy_forecast is not None:
            # Get the noisy forecast for the current day.
            # Each call to NoisyForecast.__call__ advances the AR(1) state
            # by one step, simulating daily forecast issuance with temporal
            # persistence. NoisyForecast was re-seeded in reset() so the
            # trajectory is deterministic and reproducible.
            fc_dict = self._noisy_forecast(
                day, self._climate, self._precomputed, H
            )
            rain_fc = _pad(fc_dict['rainfall'])
            ETc_fc  = _pad(fc_dict['ETc'])
            rad_fc  = _pad(fc_dict['radiation'])
            # h2, h7, g_base use perfect values: NoisyForecast only corrupts
            # rainfall and ETc to match the MPC noise model.
            h2_fc  = _pad(self._precomputed.h2[d:end])
            h7_fc  = _pad(self._precomputed.h7[d:end])
            g_fc   = _pad(self._precomputed.g_base[d:end])
        else:
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
