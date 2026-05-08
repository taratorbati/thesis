# =============================================================================
# src/rl/gym_env.py
# Gymnasium wrapper for the 130-agent crop-soil ABM, configured at the
# recommended operating point alpha* (Chapter 4 of the thesis).
#
# Observation (660 dims):
#   Per-agent (5 × 130 = 650):
#     x1_norm, x5_norm, x4_norm, x3, elevation
#   Scalars (10):
#     day_frac, budget_frac, burn_rate, rain_today, rain_forecast,
#     ETc_today, ETc_forecast, h2_today, h7_today, g_base_today
#
# Action: Box(0, 1, shape=(130,)) → scaled to [0, UB_MM] mm/day.
#
# Reward (dense, exact negation of the MPC path cost at alpha*):
#   r(t) = +alpha1 * Δx4_mean / x4_ref           (biomass progress)
#        - alpha2 * Σu / W_daily_ref              (water cost — domestic tier)
#        - alpha3 * mean(max(ST-x1, 0)) / (ST-WP) (drought)
#        - alpha4 * mean(x5) / x5_ref             (ponding — INACTIVE at alpha4=0)
#        - alpha5 * ||u-u_prev||² / (UB² · N)     (delta-u)
#        - alpha6 * mean([max(x1-FC, 0)/FC]²)     (FC overshoot — ACTIVATED)
#        - λ_budget * max(burn_rate-1, 0)²        (budget soft penalty)
#   Terminal: + TERMINAL_BONUS_MULT * alpha1 * final_x4_mean / x4_ref
#
# CRITICAL: Reward construction matches src/mpc/cost.py at alpha* exactly,
# enabling the policy-equivalence comparison claimed in Chapter 4 §4.6.
#
# Scenarios: 'dry' or 'wet'. The thesis omits the 2020 moderate scenario
# because it is climatologically too close to 2022 (dry) to produce
# distinguishable controller behaviour.
# =============================================================================

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from abm import CropSoilABM
from src.terrain import load_terrain, get_sink_agents
from src.precompute import get_precomputed
from soil_data import get_crop
from climate_data import load_cleaned_data, extract_scenario_by_name


# ── Normalization references — must match src/mpc/cost.py DEFAULT_REFS ───
X4_REF = 900.0       # g/m², biomass normalization (~3800 kg/ha target)
X5_REF = 50.0        # mm, ponding normalization (only used if alpha4 != 0)
UB_MM = 12.0         # mm/day, actuator cap

# ── Recommended operating point alpha* (Chapter 4) ───────────────────────
# These match src/mpc/cost.py DEFAULT_WEIGHTS exactly. Do not modify
# without updating the MPC cost as well, otherwise the SAC reward will
# no longer be the negation of the MPC cost.
ALPHA1 = 1.0      # terminal biomass anchor
ALPHA2 = 0.016    # water cost — domestic-base Iranian tariff
ALPHA3 = 0.1      # drought stress regularizer
ALPHA4 = 0.0      # surface ponding — INACTIVE (subsumed by alpha6 per Group F)
ALPHA5 = 0.005    # delta-u regularizer
ALPHA6 = 8.0      # x1 > FC soft penalty — recommended ceiling-confirmed value

# Budget violation penalty weight (RL-only soft constraint)
LAMBDA_BUDGET = 5.0

# Terminal biomass reward multiplier.
# The path-reward contribution from the per-step Δx4 term accumulates to
# roughly +0.93 over a successful season (full biomass reached), and the
# accumulated path penalties at alpha* sum to roughly -1.7 for a good
# policy. A terminal bonus of 1.0× would leave a successful policy at
# net reward ~+0.2, with weak gradient signal. Using a 5× multiplier
# scales the terminal bonus to +5.0 for full yield, producing a clear
# net-positive return for successful policies and a strong learning
# signal for the sparse end-of-episode reward.
TERMINAL_BONUS_MULT = 5.0


class IrrigationEnv(gym.Env):
    """Gymnasium environment for multi-agent irrigation control.

    Parameters
    ----------
    scenario : str
        'dry' or 'wet'.
    budget_pct : int
        Budget percentage: 100, 85, or 70.
    crop_name : str
        'rice' (only supported in Phase 1).
    dem_path : str
        Path to the GeoTIFF DEM file.
    forecast_horizon : int
        Days of forecast to include in observation. Default 8 to match Hp*=8.
    seed : int or None
        Random seed for reproducibility.
    """

    metadata = {'render_modes': []}

    def __init__(self, scenario='dry', budget_pct=100, crop_name='rice',
                 dem_path='gilan_farm.tif', forecast_horizon=8, seed=None):
        super().__init__()

        self.scenario = scenario
        self.budget_pct = budget_pct
        self.crop_name = crop_name
        self.dem_path = dem_path
        self.forecast_horizon = forecast_horizon

        # Load static data
        self.crop = get_crop(crop_name)
        self.terrain = load_terrain(dem_path)
        self.N = self.terrain['N']
        self.season_days = self.crop['season_days']

        # Derived agronomic constants
        self.fc_total = self.crop['theta6'] * self.crop['theta5']
        self.wp_total = self.crop['theta2'] * self.crop['theta5']
        p = self.crop.get('p', 0.20)
        raw = p * (self.fc_total - self.wp_total)
        self.stress_threshold = self.fc_total - raw

        full_need_mm = {'rice': 484.0, 'tobacco': 389.0}[crop_name]
        self.budget_total = full_need_mm * (budget_pct / 100.0)

        # Elevation (static, included in obs)
        self.elev_norm = self.terrain['gamma_flat']

        # Precomputed climate-only quantities
        self.precomputed = get_precomputed(scenario, crop_name)

        # Load climate
        df = load_cleaned_data()
        self.climate = extract_scenario_by_name(df, scenario, self.crop)

        # Spaces
        obs_dim = 5 * self.N + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N,), dtype=np.float32
        )

        # Cost weights at recommended alpha* (must match src/mpc/cost.py)
        self.alpha1 = ALPHA1
        self.alpha2 = ALPHA2
        self.alpha3 = ALPHA3
        self.alpha4 = ALPHA4
        self.alpha5 = ALPHA5
        self.alpha6 = ALPHA6
        self.W_daily_ref = 5.0 * self.N

        # Will be set in reset()
        self.abm = None
        self.day = 0
        self.budget_remaining = 0.0
        self.water_used = 0.0
        self.u_prev = None
        self.x4_prev_mean = 0.0

        self._np_random = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

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

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=float).clip(0, 1)
        u = action * UB_MM

        # Budget clip (hard constraint enforcement at the RL boundary)
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

        # ── Compute reward (negation of MPC path cost at alpha*) ─────────

        x4_mean = float(self.abm.x4.mean())

        # Term 1: biomass progress (positive contribution)
        r_biomass = self.alpha1 * (x4_mean - self.x4_prev_mean) / X4_REF

        # Term 2: water cost
        r_water = -self.alpha2 * u.sum() / self.W_daily_ref

        # Term 3: drought penalty
        deficit = np.maximum(self.stress_threshold - self.abm.x1, 0)
        r_drought = -self.alpha3 * deficit.sum() / (
            self.N * max(self.stress_threshold - self.wp_total, 1e-6))

        # Term 4: ponding penalty (inactive at alpha4 = 0; computed for logging)
        if self.alpha4 != 0.0:
            r_ponding = -self.alpha4 * self.abm.x5.sum() / (self.N * X5_REF)
        else:
            r_ponding = 0.0

        # Term 5: delta-u penalty
        du = u - self.u_prev
        r_delta_u = -self.alpha5 * np.dot(du, du) / (UB_MM ** 2 * self.N)

        # Term 6: FC overshoot penalty (NEW — matches MPC J_overFC at alpha6=8)
        # Quadratic in normalized excess: max(x1 - FC, 0) / FC, squared.
        # Inactive when x1 ≤ FC (excess = 0 → contribution = 0).
        if self.alpha6 != 0.0:
            excess = np.maximum(self.abm.x1 - self.fc_total, 0.0)
            normalized_sq = (excess / self.fc_total) ** 2
            r_overfc = -self.alpha6 * normalized_sq.sum() / self.N
        else:
            r_overfc = 0.0

        # Budget soft penalty (RL-only — MPC enforces this as hard constraint)
        daily_budget = self.budget_total / self.season_days
        burn_rate = self.water_used / max(self.day + 1, 1) / max(daily_budget, 1e-6)
        r_budget = -LAMBDA_BUDGET * max(burn_rate - 1.0, 0.0) ** 2

        reward = (r_biomass + r_water + r_drought + r_ponding
                  + r_delta_u + r_overfc + r_budget)

        self.u_prev = u.copy()
        self.x4_prev_mean = x4_mean
        self.day += 1

        # ── Termination ───────────────────────────────────────────────────

        terminated = False
        truncated = False

        if self.budget_remaining <= 0 and self.day < self.season_days:
            terminated = True

        if self.day >= self.season_days:
            truncated = True
            # Terminal bonus on harvested biomass.
            # Multiplier > 1 ensures successful policies achieve
            # net-positive return relative to the do-nothing baseline.
            reward += TERMINAL_BONUS_MULT * self.alpha1 * x4_mean / X4_REF

        obs = self._get_obs()

        info = {
            'day': self.day,
            'yield_kg_ha': x4_mean * self.crop.get('HI', 0.42) * 10.0,
            'water_used_mm': self.water_used,
            'budget_remaining': self.budget_remaining,
            'r_biomass':  r_biomass,
            'r_water':    r_water,
            'r_drought':  r_drought,
            'r_ponding':  r_ponding,
            'r_delta_u':  r_delta_u,
            'r_overfc':   r_overfc,
            'r_budget':   r_budget,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        x1_norm = self.abm.x1 / self.fc_total
        x5_norm = self.abm.x5 / X5_REF
        x4_norm = self.abm.x4 / X4_REF
        x3 = self.abm.x3
        elev = self.elev_norm

        day_frac = self.day / self.season_days
        budget_frac = self.budget_remaining / max(self.budget_total, 1e-6)
        daily_budget = self.budget_total / self.season_days
        burn_rate = (self.water_used / max(self.day, 1)) / max(daily_budget, 1e-6) if self.day > 0 else 0.0

        d = min(self.day, self.season_days - 1)
        rain_today = float(self.climate['rainfall'][d])
        ETc_today = float(self.precomputed.Kc_ET[d])

        end = min(d + self.forecast_horizon, self.season_days)
        rain_forecast = float(self.climate['rainfall'][d:end].sum()) if end > d else 0.0
        ETc_forecast = float(self.precomputed.Kc_ET[d:end].sum()) if end > d else 0.0

        h2_today = float(self.precomputed.h2[d])
        h7_today = float(self.precomputed.h7[d])
        g_base_today = float(self.precomputed.g_base[d])

        scalars = np.array([
            day_frac, budget_frac, burn_rate,
            rain_today, rain_forecast, ETc_today, ETc_forecast,
            h2_today, h7_today, g_base_today,
        ], dtype=np.float32)

        obs = np.concatenate([
            x1_norm, x5_norm, x4_norm, x3, elev, scalars
        ]).astype(np.float32)

        return obs
