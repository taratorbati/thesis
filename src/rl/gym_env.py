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
#   uniformly from TRAINING_YEARS (20 years) and a budget uniformly
#   from U(70%, 100%).
#   Eval mode (fixed_scenario set): fixed scenario (named test year or
#   any year integer if passed as fixed_year) and fixed budget.
#
# Precomputed quantities per training year (v2.4):
#   h2, h7, g_base, Kc_ET are computed on-the-fly from each sampled
#   year's temperature/ET data (compute_precomputed_from_climate). Cached
#   in self._precomputed_by_year so each year's computation runs at most
#   once across the lifetime of the env. Eliminates the prior Markov leak
#   where every training year used the dry-year (2022) precomputed.
#
# Reward (approximate negation of MPC path cost):
#   r(t) = +alpha1 * Δx4_mean / x4_ref           (biomass progress)
#        - alpha2 * Σu / W_daily_ref              (water cost)
#        - alpha3 * mean(max(ST-x1, 0)) / (ST-WP) (drought)
#        - alpha5 * ||u-u_prev||² / (UB² · N)     (delta-u)
#        - alpha6 * mean([max(x1-FC, 0)/FC]²)     (FC overshoot)
#        - λ_budget * max(burn_rate-1, 0)²        (budget soft penalty)
#   Terminal: + TERMINAL_BONUS_MULT * alpha1 * final_x4_mean / x4_ref
#             (paid on BOTH truncated=True and terminated=True)
#
# v2.4 reward tuning: LAMBDA_BUDGET reduced from 5.0 to 0.1 based on
#   reward-magnitude analysis. At the previous value, the burn-rate soft
#   penalty was ~100× the sum of all agronomic terms on any step where
#   burn_rate > 1, dominating the gradient and causing training divergence
#   (observed in the 500k-step pilot: eval reward degraded from -3.0 at
#   step 75k to -9.7 at step 500k with multiple instability spikes).
#   The hard budget clip in step() still guarantees constraint feasibility
#   independently of this soft term.
# =============================================================================

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from abm import CropSoilABM
from src.terrain import load_terrain
from src.precompute import get_precomputed, compute_precomputed_from_climate
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

# Budget soft-penalty coefficient.
# Was 5.0 in v2.3 — produced reward terms ~100× larger than agronomic
# components, dominating the gradient and causing training to diverge in
# the 500k-step pilot. Reduced to 0.1 so the soft penalty is on the same
# order of magnitude as the other reward terms while still providing a
# smooth gradient near the budget boundary. The hard u.mean() <=
# budget_remaining clip in step() enforces the constraint absolutely.
LAMBDA_BUDGET = 0.1

TERMINAL_BONUS_MULT = 5.0


class IrrigationEnv(gym.Env):
    """Gymnasium environment for multi-agent irrigation control.

    Parameters
    ----------
    fixed_scenario : str or None
        Named test scenario ('dry', 'moderate', 'wet'). If None, a random
        year from TRAINING_YEARS is used each episode (training mode).
    fixed_budget_pct : float or None
        Budget percentage (70-100). If None, sampled from U(70,100) each
        episode (training mode).
    fixed_year : int or None
        Integer year for episodes that are not a named scenario. Useful
        for evaluating on a specific dev year. Mutually exclusive with
        fixed_scenario (named scenarios already resolve to a year via
        SCENARIO_YEARS).
    year_pool : tuple/list of int or None
        If set (and fixed_scenario/fixed_year are None), sample from this
        pool of years at each reset(). Used by EvalCallback to evaluate
        on a deterministic set of dev years rather than the full training
        pool. Default None (= sample from TRAINING_YEARS).
    randomize : bool
        Convenience alias: if True, forces fixed_scenario=None,
        fixed_budget_pct=None, fixed_year=None regardless of other args.
        Useful for explicitly marking training envs in code that reads them.
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
        fixed_year=None,
        year_pool=None,
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

        # Legacy aliases
        if scenario is not None and fixed_scenario is None:
            fixed_scenario = scenario
        if budget_pct is not None and fixed_budget_pct is None:
            fixed_budget_pct = budget_pct

        # randomize=True overrides any fixed values (explicit training mode)
        if randomize:
            fixed_scenario = None
            fixed_budget_pct = None
            fixed_year = None

        if fixed_scenario is not None and fixed_year is not None:
            raise ValueError(
                "fixed_scenario and fixed_year are mutually exclusive. "
                "Use fixed_scenario for named test scenarios (dry/moderate/wet) "
                "and fixed_year for arbitrary integer years (e.g. dev years)."
            )

        self.fixed_scenario = fixed_scenario
        self.fixed_budget_pct = fixed_budget_pct
        self.fixed_year = fixed_year
        self.year_pool = tuple(year_pool) if year_pool is not None else None
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

        # Per-year precomputed cache. Populated lazily on first reset that
        # needs each year. Eliminates the v2.3 Markov leak (all training
        # years used the dry-year precomputed). Memory cost: ~few KB per
        # year × 20 training years + 3 dev years = negligible.
        self._precomputed_by_year = {}

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
        self._current_year = None   # for debugging / introspection

        self._np_random = np.random.default_rng(seed)

    def _get_precomputed_for_year(self, year):
        """Return cached precomputed for an integer year, computing if needed."""
        if year not in self._precomputed_by_year:
            climate = extract_scenario(self._df, year, self.crop)
            self._precomputed_by_year[year] = compute_precomputed_from_climate(
                climate, self.crop_name, scenario_tag=f"year_{year}"
            )
        return self._precomputed_by_year[year]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # ── Sample year / scenario and budget ─────────────────────────────
        if self.fixed_scenario is not None:
            # Eval mode (named test scenario): use the cached disk precomputed
            # for backward compatibility with the existing MPC eval pipeline.
            self.climate = extract_scenario_by_name(
                self._df, self.fixed_scenario, self.crop
            )
            self.precomputed = get_precomputed(self.fixed_scenario, self.crop_name)
            from climate_data import SCENARIO_YEARS
            self._current_year = SCENARIO_YEARS[self.fixed_scenario]

        elif self.fixed_year is not None:
            # Eval mode (integer year, e.g. dev year): compute on the fly.
            year = int(self.fixed_year)
            self.climate = extract_scenario(self._df, year, self.crop)
            self.precomputed = self._get_precomputed_for_year(year)
            self._current_year = year

        else:
            # Training mode: random year from the active pool.
            # Per-year precomputed (Kc_ET, h2, h7, g_base) computed from
            # this year's temperature/ET, keeping obs and ABM consistent.
            pool = self.year_pool if self.year_pool is not None else TRAINING_YEARS
            year = int(self._np_random.choice(pool))
            self.climate = extract_scenario(self._df, year, self.crop)
            self.precomputed = self._get_precomputed_for_year(year)
            self._current_year = year

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

        # Hard budget clip — enforces feasibility regardless of soft penalty.
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

        # Terminal bonus on BOTH (prevents hoard-water pathology).
        if terminated or truncated:
            reward += TERMINAL_BONUS_MULT * self.alpha1 * x4_mean / X4_REF

        info = {
            'day': self.day,
            'year': self._current_year,
            'budget_total': self.budget_total,
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

        day_frac          = self.day / self.season_days
        budget_frac       = self.budget_remaining / max(self.budget_total, 1e-6)
        budget_total_norm = self.budget_total / FULL_SEASON_NEED_MM  # in [0.7, 1.0]

        # burn_rate in obs: water used per day elapsed / daily budget.
        # At day=0 before first step, returns 0.
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
