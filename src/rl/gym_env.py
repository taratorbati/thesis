# =============================================================================
# src/rl/gym_env.py
# Gymnasium wrapper for the 130-agent crop-soil ABM, configured at the
# recommended operating point alpha* (Chapter 4 of the thesis).
#
# Observation (707 dims at default forecast_horizon=8):
#   Per-agent (5 × 130 = 650):
#     x1_norm, x5_norm, x4_norm, x3, elevation
#   Scalars (9):
#     day_frac, budget_frac, budget_total_norm, burn_rate,
#     rain_today, ETc_today, h2_today, h7_today, g_base_today
#   Per-day forecasts (6 × forecast_horizon = 48):
#     rain[d:d+H], ETc[d:d+H], radiation[d:d+H],
#     h2[d:d+H], h7[d:d+H], g_base[d:d+H]
#     End-of-season padding uses forward fill (last available value),
#     matching PerfectForecast._slice_pad in src/forecast.py.
#
# budget_total_norm is included so the agent can distinguish "50% of a tight
# budget" from "50% of a generous budget" — essential when budget is
# randomised across episodes during training.
#
# Action: Box(0, 1, shape=(130,)) → scaled to [0, UB_MM] mm/day.
#
# Reward (approximate negation of the MPC path cost at alpha*, under γ→1):
#   r(t) = +alpha1 * Δx4_mean / x4_ref           (biomass progress)
#        - alpha2 * Σu / W_daily_ref              (water cost — domestic tier)
#        - alpha3 * mean(max(ST-x1, 0)) / (ST-WP) (drought)
#        - alpha4 * mean(x5) / x5_ref             (ponding — INACTIVE at alpha4=0)
#        - alpha5 * ||u-u_prev||² / (UB² · N)     (delta-u)
#        - alpha6 * mean([max(x1-FC, 0)/FC]²)     (FC overshoot)
#        - λ_budget * max(burn_rate-1, 0)²        (budget soft penalty)
#   Terminal: + TERMINAL_BONUS_MULT * alpha1 * final_x4_mean / x4_ref
#             (paid on BOTH truncated=True and terminated=True so the agent
#              is not penalised for exhausting the budget after crop is grown)
#
# Note on policy-equivalence: the SAC discounted return (γ=0.99) is an
# approximation to the undiscounted MPC cost. The approximation is accurate
# for the first ~70 days (discount factor 0.99^70 ≈ 0.50) but diverges at
# the terminal step. The word "exact" that appeared in previous versions of
# this docstring was incorrect and has been removed.
#
# Training design:
#   - At each reset(), a year is sampled uniformly from TRAINING_YEARS
#     (23 years: 2000-2025 excluding eval years 2018, 2022, 2024).
#   - Budget is sampled uniformly from U(70%, 100%) of full seasonal need.
#   - This trains a single general policy evaluated on the 9 fixed holdout
#     cells (3 eval years × 3 budgets) shared with the MPC evaluation.
#   - To freeze scenario/budget for evaluation, pass fixed_scenario and
#     fixed_budget_pct at construction time.
# =============================================================================

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from abm import CropSoilABM
from src.terrain import load_terrain, get_sink_agents
from src.precompute import get_precomputed
from soil_data import get_crop
from climate_data import (
    load_cleaned_data, extract_scenario, extract_scenario_by_name,
    TRAINING_YEARS, SCENARIO_YEARS,
)


# ── Normalization references — must match src/mpc/cost.py DEFAULT_REFS ───
X4_REF = 900.0       # g/m², biomass normalization (~3800 kg/ha target)
X5_REF = 50.0        # mm, ponding normalization (only used if alpha4 != 0)
UB_MM = 12.0         # mm/day, actuator cap

FULL_SEASON_NEED_MM = 484.0   # 100% budget for rice (mm, field-averaged)
MAX_BUDGET_MM = FULL_SEASON_NEED_MM            # normalization denominator

# ── Recommended operating point alpha* (Chapter 4) ───────────────────────
ALPHA1 = 1.0
ALPHA2 = 0.016
ALPHA3 = 0.1
ALPHA4 = 0.0
ALPHA5 = 0.005
ALPHA6 = 8.0

LAMBDA_BUDGET = 5.0
TERMINAL_BONUS_MULT = 5.0


class IrrigationEnv(gym.Env):
    """Gymnasium environment for multi-agent irrigation control.

    Parameters
    ----------
    fixed_scenario : str or None
        If set, always use this named scenario (e.g. 'dry'). If None,
        a random year from TRAINING_YEARS is sampled at each reset().
        Use None during training; use a scenario name for evaluation.
    fixed_budget_pct : float or None
        If set, always use this budget percentage (e.g. 100.0). If None,
        budget is sampled uniformly from U(70%, 100%) at each reset().
        Use None during training; use a fixed value for evaluation.
    crop_name : str
        'rice' (only supported in Phase 1).
    dem_path : str
        Path to the GeoTIFF DEM file.
    forecast_horizon : int
        Days of forecast in observation. Default 8 (= Hp* of the MPC).
    seed : int or None
        Random seed for reproducibility.
    """

    metadata = {'render_modes': []}

    def __init__(self, fixed_scenario=None, fixed_budget_pct=None,
                 crop_name='rice', dem_path='gilan_farm.tif',
                 forecast_horizon=8, seed=None,
                 # Legacy positional aliases kept for backward compatibility:
                 scenario=None, budget_pct=None):
        super().__init__()

        # Handle legacy positional args
        if scenario is not None and fixed_scenario is None:
            fixed_scenario = scenario
        if budget_pct is not None and fixed_budget_pct is None:
            fixed_budget_pct = budget_pct

        self.fixed_scenario = fixed_scenario
        self.fixed_budget_pct = fixed_budget_pct
        self.crop_name = crop_name
        self.dem_path = dem_path
        self.forecast_horizon = forecast_horizon

        # Load static data once
        self.crop = get_crop(crop_name)
        self.terrain = load_terrain(dem_path)
        self.N = self.terrain['N']
        self.season_days = self.crop['season_days']

        # Derived agronomic constants
        self.fc_total = self.crop['theta6'] * self.crop['theta5']
        self.wp_total = self.crop['theta2'] * self.crop['theta5']
        p = self.crop.get('p', 0.20)
        self.stress_threshold = self.fc_total - p * (self.fc_total - self.wp_total)

        # Elevation (static, included in obs)
        self.elev_norm = self.terrain['gamma_flat']

        # Load all climate data once; individual years extracted in reset()
        self._df = load_cleaned_data()

        # Observation space:
        # 650 per-agent + 9 scalars + 6*forecast_horizon forecast = 707
        obs_dim = 5 * self.N + 9 + 6 * self.forecast_horizon
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
        )

        # Cost weights
        self.alpha1 = ALPHA1
        self.alpha2 = ALPHA2
        self.alpha3 = ALPHA3
        self.alpha4 = ALPHA4
        self.alpha5 = ALPHA5
        self.alpha6 = ALPHA6
        self.W_daily_ref = 5.0 * self.N

        # Episode state (set in reset)
        self.abm = None
        self.climate = None
        self.precomputed = None
        self.budget_total = None
        self.day = 0
        self.budget_remaining = 0.0
        self.water_used = 0.0
        self.u_prev = None
        self.x4_prev_mean = 0.0

        self._np_random = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # ── Sample year and budget for this episode ───────────────────────
        if self.fixed_scenario is not None:
            # Eval mode: fixed named scenario
            self.climate = extract_scenario_by_name(
                self._df, self.fixed_scenario, self.crop
            )
            self.precomputed = get_precomputed(self.fixed_scenario, self.crop_name)
        else:
            # Training mode: random year from training pool
            year = int(self._np_random.choice(TRAINING_YEARS))
            self.climate = extract_scenario(self._df, year, self.crop)
            # Precomputed quantities depend only on crop+climate; re-derive
            # from the chosen year's data inline.
            # get_precomputed() caches by (scenario_name, crop_name) — since
            # we're sampling arbitrary years we call the year-based variant.
            self.precomputed = get_precomputed(year, self.crop_name)

        if self.fixed_budget_pct is not None:
            budget_pct = float(self.fixed_budget_pct)
        else:
            # Continuous uniform U(70%, 100%) — avoids lookup-table behaviour
            # that discrete {70,85,100} sampling could induce.
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

        # Budget clip
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

        # ── Reward ────────────────────────────────────────────────────────
        x4_mean = float(self.abm.x4.mean())

        r_biomass = self.alpha1 * (x4_mean - self.x4_prev_mean) / X4_REF
        r_water = -self.alpha2 * u.sum() / self.W_daily_ref

        deficit = np.maximum(self.stress_threshold - self.abm.x1, 0)
        r_drought = -self.alpha3 * deficit.sum() / (
            self.N * max(self.stress_threshold - self.wp_total, 1e-6))

        r_ponding = (
            -self.alpha4 * self.abm.x5.sum() / (self.N * X5_REF)
            if self.alpha4 != 0.0 else 0.0
        )

        du = u - self.u_prev
        r_delta_u = -self.alpha5 * np.dot(du, du) / (UB_MM ** 2 * self.N)

        if self.alpha6 != 0.0:
            excess = np.maximum(self.abm.x1 - self.fc_total, 0.0)
            r_overfc = -self.alpha6 * ((excess / self.fc_total) ** 2).sum() / self.N
        else:
            r_overfc = 0.0

        # burn_rate: unified formula used in both reward and obs.
        # Uses self.day + 1 because the day counter is incremented below,
        # so at the moment of the reward self.day still holds the pre-step day.
        daily_budget = self.budget_total / self.season_days
        burn_rate = (
            self.water_used / (self.day + 1) / max(daily_budget, 1e-6)
        )
        r_budget = -LAMBDA_BUDGET * max(burn_rate - 1.0, 0.0) ** 2

        reward = r_biomass + r_water + r_drought + r_ponding + r_delta_u + r_overfc + r_budget

        self.u_prev = u.copy()
        self.x4_prev_mean = x4_mean
        self.day += 1

        # ── Termination ───────────────────────────────────────────────────
        terminated = self.budget_remaining <= 0 and self.day < self.season_days
        truncated = self.day >= self.season_days

        # Terminal bonus on BOTH terminated and truncated so the agent is
        # not penalised for exhausting the budget after the crop has grown.
        # Previously, terminated episodes received no bonus, creating a
        # pathological incentive to hoard water and under-irrigate.
        if terminated or truncated:
            reward += TERMINAL_BONUS_MULT * self.alpha1 * x4_mean / X4_REF

        info = {
            'day': self.day,
            'yield_kg_ha': x4_mean * self.crop.get('HI', 0.42) * 10.0,
            'water_used_mm': self.water_used,
            'budget_remaining': self.budget_remaining,
            'burn_rate': burn_rate,
            'r_biomass': r_biomass, 'r_water': r_water,
            'r_drought': r_drought, 'r_ponding': r_ponding,
            'r_delta_u': r_delta_u, 'r_overfc': r_overfc,
            'r_budget': r_budget,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        x1_norm = self.abm.x1 / self.fc_total
        x5_norm = self.abm.x5 / X5_REF
        x4_norm = self.abm.x4 / X4_REF
        x3 = self.abm.x3
        elev = self.elev_norm

        day_frac = self.day / self.season_days
        budget_frac = self.budget_remaining / max(self.budget_total, 1e-6)
        budget_total_norm = self.budget_total / MAX_BUDGET_MM

        # burn_rate in obs uses same formula as in step() reward.
        # At day=0 (before any step), water_used=0 → burn_rate=0.
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

        scalars = np.array([
            day_frac, budget_frac, budget_total_norm, burn_rate,
            rain_today, ETc_today, h2_today, h7_today, g_base_today,
        ], dtype=np.float32)

        H = self.forecast_horizon
        end = min(d + H, self.season_days)

        def _pad(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < H:
                pad_val = arr[-1] if len(arr) > 0 else 0.0
                return np.concatenate([arr, np.full(H - len(arr), pad_val, dtype=np.float32)])
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
