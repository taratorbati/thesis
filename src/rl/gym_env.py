# =============================================================================
# src/rl/gym_env.py
# Gymnasium wrapper for the 130-agent crop-soil ABM.
#
# Observation (707 dims at default forecast_horizon=8):
#   Per-agent (5 × 130 = 650):
#     x1_norm, x5_norm, x4_norm, x3, elevation
#   Scalars (9) — order is fixed, must match runner._build_obs exactly:
#     [0] day_frac
#     [1] budget_frac
#     [2] budget_total_norm
#     [3] burn_rate
#     [4] rain_today
#     [5] ETc_today
#     [6] h2_today
#     [7] h7_today
#     [8] g_base_today
#   Per-day forecasts (6 × 8 = 48):
#     rain[d:d+H], ETc[d:d+H], radiation[d:d+H],
#     h2[d:d+H], h7[d:d+H], g_base[d:d+H]
#     Forward-fill padding at season end.
#
# Training design (Chapter 4):
#   Training mode (fixed_scenario=None): each reset() samples a year
#   uniformly from TRAINING_YEARS (23 years, 2000-2025 minus eval years)
#   and a budget uniformly from U(70%, 100%).
#   Eval mode (fixed_scenario set): fixed year and budget.
#
# Note on precomputed quantities during training:
#   h2 (heat stress), h7 (cold stress), and g_base depend primarily on
#   temperature. Since all 23 training years span a narrow range
#   (mean temperature 25-28°C in Gilan rice season), the precomputed
#   'dry' scenario (2022) is used as an approximation for all training
#   years. This is a deliberate simplification documented in the thesis.
# =============================================================================

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from abm import CropSoilABM
from src.terrain import load_terrain
from src.precompute import get_precomputed
from soil_data import get_crop
from climate_data import (
    load_cleaned_data, extract_scenario, extract_scenario_by_name,
    TRAINING_YEARS,
)


# ── Normalization references — must match src/mpc/cost.py DEFAULT_REFS ───────
X4_REF  = 900.0     # g/m², biomass normalization
X5_REF  = 50.0      # mm, ponding normalization
UB_MM   = 12.0      # mm/day, actuator cap
FULL_SEASON_NEED_MM = 484.0   # 100% budget for rice (mm); also used by runner

# ── Cost weights at alpha* — must match src/mpc/cost.py ─────────────────────
ALPHA1 = 1.0
ALPHA2 = 0.016
ALPHA3 = 0.1
ALPHA4 = 0.0      # inactive
ALPHA5 = 0.005
ALPHA6 = 8.0
LAMBDA_BUDGET = 5.0
TERMINAL_BONUS_MULT = 5.0


class IrrigationEnv(gym.Env):
    """Gymnasium environment for multi-agent irrigation control.

    Parameters
    ----------
    fixed_scenario : str or None
        Named eval scenario ('dry', 'moderate', 'wet'). If None, a random
        year from TRAINING_YEARS is used each episode (training mode).
    fixed_budget_pct : float or None
        Budget percentage (70-100). If None, sampled from U(70,100) each
        episode (training mode).
    randomize : bool
        Convenience alias: if True, forces fixed_scenario=None and
        fixed_budget_pct=None regardless of other args. Useful for
        explicitly marking training envs in code that reads them.
    crop_name : str
    dem_path : str
    forecast_horizon : int
        Must match the value used during training. Default 8 (= Hp* MPC).
    seed : int or None
    scenario : str or None
        Legacy alias for fixed_scenario.
    budget_pct : float or None
        Legacy alias for fixed_budget_pct.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        fixed_scenario=None,
        fixed_budget_pct=None,
        randomize=False,
        crop_name='rice',
        dem_path='gilan_farm.tif',
        forecast_horizon=8,
        seed=None,
        # Legacy aliases
        scenario=None,
        budget_pct=None,
    ):
        super().__init__()

        # Handle legacy aliases
        if scenario is not None and fixed_scenario is None:
            fixed_scenario = scenario
        if budget_pct is not None and fixed_budget_pct is None:
            fixed_budget_pct = budget_pct

        # randomize=True overrides any fixed values (explicit training mode)
        if randomize:
            fixed_scenario = None
            fixed_budget_pct = None

        self.fixed_scenario = fixed_scenario
        self.fixed_budget_pct = fixed_budget_pct
        self.crop_name = crop_name
        self.dem_path = dem_path
        self.forecast_horizon = forecast_horizon

        self.crop = get_crop(crop_name)
        self.terrain = load_terrain(dem_path)
        self.N = self.terrain['N']
        self.season_days = self.crop['season_days']

        self.fc_total = self.crop['theta6'] * self.crop['theta5']
        self.wp_total = self.crop['theta2'] * self.crop['theta5']
        p = self.crop.get('p', 0.20)
        self.stress_threshold = self.fc_total - p * (self.fc_total - self.wp_total)
        self.elev_norm = self.terrain['gamma_flat']

        self._df = load_cleaned_data()

        # Pre-load the 'dry' precomputed for use as training-year approximation.
        # See module docstring note on the simplification.
        self._precomputed_dry = get_precomputed('dry', crop_name)

        # obs_dim: 5*N per-agent + 9 scalars + 6*H forecast
        obs_dim = 5 * self.N + 9 + 6 * self.forecast_horizon
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
        )

        self.alpha1 = ALPHA1
        self.alpha2 = ALPHA2
        self.alpha3 = ALPHA3
        self.alpha4 = ALPHA4
        self.alpha5 = ALPHA5
        self.alpha6 = ALPHA6
        self.W_daily_ref = 5.0 * self.N

        self.abm = None
        self.climate = None
        self.precomputed = None
        self.budget_total = 0.0
        self.day = 0
        self.budget_remaining = 0.0
        self.water_used = 0.0
        self.u_prev = None
        self.x4_prev_mean = 0.0

        self._np_random = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # ── Sample year / scenario and budget ─────────────────────────────
        if self.fixed_scenario is not None:
            # Eval mode: fixed named scenario with exact precomputed data
            self.climate = extract_scenario_by_name(
                self._df, self.fixed_scenario, self.crop
            )
            self.precomputed = get_precomputed(self.fixed_scenario, self.crop_name)
        else:
            # Training mode: random year from TRAINING_YEARS.
            # Climate: exact year data (rainfall, temperature, ET).
            # Precomputed (h2, h7, g_base, Kc_ET): use 'dry' approximation
            # since temperature-derived quantities vary little across the 23
            # training years (all in the dry-to-moderate range, 14-89mm).
            year = int(self._np_random.choice(TRAINING_YEARS))
            self.climate = extract_scenario(self._df, year, self.crop)
            self.precomputed = self._precomputed_dry

        if self.fixed_budget_pct is not None:
            budget_pct = float(self.fixed_budget_pct)
        else:
            budget_pct = float(self._np_random.uniform(70.0, 100.0))

        self.budget_total = FULL_SEASON_NEED_MM * (budget_pct / 100.0)

        # ── Reset ABM ─────────────────────────────────────────────────────
        self.abm = CropSoilABM(
            gamma_flat=self.terrain['gamma_flat'],
            sends_to=self.terrain['sends_to'],
            Nr=self.terrain['Nr'],
            theta=self.crop,
            N=self.N,
            runoff_mode='cascade',
            elevation=self.terrain['elevation_flat'],
        )
        self.abm.reset()
        self.abm.x1 = np.full(self.N, self.fc_total)
        self.abm.x2 = np.full(self.N, self.crop.get('x2_init', 0.0))
        self.abm.x3 = np.zeros(self.N)
        self.abm.x4 = np.full(self.N, self.crop.get('x4_init', 0.0))
        self.abm.x5 = np.zeros(self.N)

        self.day = 0
        self.budget_remaining = float(self.budget_total)
        self.water_used = 0.0
        self.u_prev = np.zeros(self.N)
        self.x4_prev_mean = float(self.abm.x4.mean())

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=float).clip(0, 1)
        u = action * UB_MM

        if u.mean() > self.budget_remaining:
            scale = self.budget_remaining / max(u.mean(), 1e-12)
            u = u * scale

        climate_today = {
            'rainfall':  float(self.climate['rainfall'][self.day]),
            'temp_mean': float(self.climate['temp_mean'][self.day]),
            'temp_max':  float(self.climate['temp_max'][self.day]),
            'radiation': float(self.climate['radiation'][self.day]),
            'ET':        float(self.climate['ET'][self.day]),
        }
        self.abm.step(u, climate_today)

        daily_spend = float(u.mean())
        self.budget_remaining = max(self.budget_remaining - daily_spend, 0.0)
        self.water_used += daily_spend

        x4_mean = float(self.abm.x4.mean())

        # ── Reward ────────────────────────────────────────────────────────
        r_biomass = self.alpha1 * (x4_mean - self.x4_prev_mean) / X4_REF
        r_water   = -self.alpha2 * u.sum() / self.W_daily_ref

        deficit = np.maximum(self.stress_threshold - self.abm.x1, 0)
        r_drought = -self.alpha3 * deficit.sum() / (
            self.N * max(self.stress_threshold - self.wp_total, 1e-6))

        du = u - self.u_prev
        r_delta_u = -self.alpha5 * np.dot(du, du) / (UB_MM ** 2 * self.N)

        excess = np.maximum(self.abm.x1 - self.fc_total, 0.0)
        r_overfc = -self.alpha6 * ((excess / self.fc_total) ** 2).mean()

        # burn_rate: field-avg water used per day / daily budget allowance.
        # Uses day+1 because self.day is still the pre-increment value here.
        daily_budget = self.budget_total / self.season_days
        burn_rate = self.water_used / (self.day + 1) / max(daily_budget, 1e-6)
        r_budget = -LAMBDA_BUDGET * max(burn_rate - 1.0, 0.0) ** 2

        reward = r_biomass + r_water + r_drought + r_delta_u + r_overfc + r_budget

        self.u_prev = u.copy()
        self.x4_prev_mean = x4_mean
        self.day += 1

        # ── Termination ───────────────────────────────────────────────────
        terminated = self.budget_remaining <= 0 and self.day < self.season_days
        truncated  = self.day >= self.season_days

        # Terminal bonus on BOTH: prevents the pathological incentive to
        # hoard water that arose when the bonus was truncated-only.
        if terminated or truncated:
            reward += TERMINAL_BONUS_MULT * self.alpha1 * x4_mean / X4_REF

        info = {
            'day': self.day,
            'yield_kg_ha': x4_mean * self.crop.get('HI', 0.42) * 10.0,
            'water_used_mm': self.water_used,
            'budget_remaining': self.budget_remaining,
            'r_biomass': r_biomass, 'r_water': r_water,
            'r_drought': r_drought, 'r_delta_u': r_delta_u,
            'r_overfc': r_overfc, 'r_budget': r_budget,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        x1_norm = self.abm.x1 / self.fc_total
        x5_norm = self.abm.x5 / X5_REF
        x4_norm = self.abm.x4 / X4_REF
        x3      = self.abm.x3
        elev    = self.elev_norm

        day_frac         = self.day / self.season_days
        budget_frac      = self.budget_remaining / max(self.budget_total, 1e-6)
        budget_total_norm = self.budget_total / FULL_SEASON_NEED_MM  # in [0.7, 1.0]

        # burn_rate in obs: water used per day elapsed / daily budget.
        # At day=0 before first step, returns 0.
        # Uses self.day (already incremented after step) as denominator.
        daily_budget = self.budget_total / self.season_days
        burn_rate = (
            self.water_used / max(self.day, 1) / max(daily_budget, 1e-6)
            if self.day > 0 else 0.0
        )

        d = min(self.day, self.season_days - 1)
        rain_today   = float(self.climate['rainfall'][d])
        ETc_today    = float(self.precomputed.Kc_ET[d])
        h2_today     = float(self.precomputed.h2[d])
        h7_today     = float(self.precomputed.h7[d])
        g_base_today = float(self.precomputed.g_base[d])

        # Scalar order — MUST match runner._build_obs exactly.
        # Any change here requires the same change in runner.py.
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
        end = min(d + H, self.season_days)

        def _pad(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < H:
                pad_val = arr[-1] if len(arr) > 0 else 0.0
                return np.concatenate([arr,
                    np.full(H - len(arr), pad_val, dtype=np.float32)])
            return arr

        rain_fc = _pad(self.climate['rainfall'][d:end])
        ETc_fc  = _pad(self.precomputed.Kc_ET[d:end])
        rad_fc  = _pad(self.climate['radiation'][d:end])
        h2_fc   = _pad(self.precomputed.h2[d:end])
        h7_fc   = _pad(self.precomputed.h7[d:end])
        g_fc    = _pad(self.precomputed.g_base[d:end])

        return np.concatenate([
            x1_norm, x5_norm, x4_norm, x3, elev,
            scalars,
            rain_fc, ETc_fc, rad_fc, h2_fc, h7_fc, g_fc,
        ]).astype(np.float32)
