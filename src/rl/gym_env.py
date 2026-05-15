# src/rl/gym_env.py  v2.6.0
# ─────────────────────────────────────────────────────────────────────────────
# Full rewrite against the ACTUAL project interfaces.  v2.5 was written
# against a fictional ABM/soil_data API that never existed in the repo.
#
# Interface dependencies (read from repo before writing):
#   abm.py:
#     CropSoilABM(gamma_flat, sends_to, Nr, theta, N, runoff_mode, elevation)
#     .reset()               → initialises x1/x2/x3/x4/x5 arrays
#     .step(u, climate_dict) → returns {'x1':…,'x2':…,'x3':…,'x4':…,'x5':…}
#     .x1 .x2 .x3 .x4 .x5   → (N,) arrays, readable directly
#
#   soil_data.py:
#     get_crop('rice') → dict with keys theta2,theta5,theta6,theta18,HI,p,…
#     NO flat constants (FIELD_CAPACITY_MM etc do not exist)
#
#   src/terrain.py:
#     load_terrain('gilan_farm.tif')
#     → dict: 'gamma_flat'(N,), 'sends_to', 'Nr', 'N', 'elevation_flat'
#
#   climate_data.py:
#     TRAINING_YEARS — tuple of ints
#     load_cleaned_data() → DataFrame
#     extract_scenario(df, year_int, crop) → dict with keys:
#       'rainfall','temp_mean','temp_max','radiation','ET',…
#
#   src/precompute.py:
#     get_precomputed(scenario_or_year, crop_name) → Precomputed
#     Precomputed attrs: .h2 .h7 .g_base .Kc_ET  (all np.ndarray len=season_days)
#
#   src/runner.py (canonical external interface):
#     UB_MM_PER_DAY = 12.0
#     abm.step(u, climate_today)  ← exact call signature
#
# Public names exported (consumed by src/rl/runner.py):
#   UB_MM, X4_REF, X5_REF, FULL_SEASON_NEED_MM
#
# v2.6 reward changes vs v2.4:
#   C_TERM    = 0.0   terminal bonus removed  (Bellman explosion fix)
#   ALPHA5_RL = 0.0   ΔU penalty disabled     (weather-response fix)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from abm import CropSoilABM
from climate_data import TRAINING_YEARS, load_cleaned_data, extract_scenario
from src.precompute import get_precomputed, compute_precomputed_from_climate
from src.terrain import load_terrain
from soil_data import get_crop

# ── public scalar constants (consumed by runner.py) ──────────────────────────
UB_MM               = 12.0    # actuator upper bound mm/day (= runner.UB_MM_PER_DAY)
X4_REF              = 600.0   # reference biomass for normalisation (g/m²)
X5_REF              = 50.0    # reference surface ponding (mm)
FULL_SEASON_NEED_MM = 484.0   # 100% seasonal budget reference (mm)
FORECAST_H          = 8       # forecast horizon (days) — matches MPC Hp*

# ── reward weights ────────────────────────────────────────────────────────────
ALPHA1        = 1.0    # biomass increment
ALPHA2        = 0.016  # water cost
ALPHA3        = 0.1    # drought stress regulariser
ALPHA5_RL     = 0.0    # [v2.6] actuator smoothing DISABLED
ALPHA6        = 8.0    # FC-overshoot penalty
C_TERM        = 0.0    # [v2.6] terminal bonus REMOVED
LAMBDA_BUDGET = 0.1    # budget burn-rate soft penalty

# ── environment dimensions ────────────────────────────────────────────────────
N_AGENTS = 130
OBS_DIM  = 707   # 650 per-agent + 9 scalars + 48 forecast


# ── module-level asset cache (loaded once per process) ───────────────────────
def _load_assets():
    crop    = get_crop('rice')
    terrain = load_terrain('gilan_farm.tif')
    df      = load_cleaned_data()
    return crop, terrain, df


_CROP, _TERRAIN, _CLIMATE_DF = _load_assets()

# per-crop derived thresholds (computed once from the crop dict)
_FC_MM = _CROP['theta6'] * _CROP['theta5']           # field capacity (mm)
_WP_MM = _CROP['theta2'] * _CROP['theta5']           # wilting point  (mm)
_ST_MM = _FC_MM - _CROP['p'] * (_FC_MM - _WP_MM)    # stress threshold (mm)
_HI    = _CROP['HI']                                 # harvest index
_K     = _CROP['season_days']                        # season length (days)
_GDD_MATURITY = _CROP.get('theta18', 1250.0)         # GDD to maturity

# Scenario name → year int mapping (for precompute cache key)
_SCENARIO_YEAR_MAP = {2022: 'dry', 2018: 'moderate', 2024: 'wet'}


class IrrigationEnv(gym.Env):
    """Gymnasium wrapper around the 130-agent crop-soil ABM.

    Observation (707-dim, agent-major layout):
      Per-agent block  (650): [x1_norm, x5_norm, x4_norm, x3, γ] × 130
      Scalar block       (9): day_frac, budget_frac, budget_total_norm,
                               burn_rate, rain_today, ETc_today, h2, h7, g_base
      Forecast block    (48): rain[0:8], ETc[0:8], rad[0:8],
                               h2[0:8], h7[0:8], g_base[0:8]

    Action (130-dim, Box[0,1]):  scaled to [0, UB_MM] mm/day in step().

    Reward: approximate negation of MPC path cost at α*.
    """

    metadata = {"render_modes": []}
    N = N_AGENTS   # public attribute used by smoke tests and networks.py

    def __init__(self, randomize: bool = True):
        super().__init__()
        self.randomize = randomize

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_AGENTS,), dtype=np.float32
        )

        # state — initialised in reset()
        self._abm: CropSoilABM | None = None
        self._precomp = None
        self._climate: dict | None = None
        self._year: int | None = None
        self._budget_mm: float = FULL_SEASON_NEED_MM
        self._water_used: float = 0.0
        self._day: int = 0
        self._prev_actions: np.ndarray = np.zeros(N_AGENTS, dtype=np.float32)
        self._prev_x4_mean: float = 0.0

        # public alias for smoke tests
        self.abm: CropSoilABM | None = None

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            self._year      = int(self.np_random.choice(list(TRAINING_YEARS)))
            budget_frac     = float(self.np_random.uniform(0.70, 1.00))
        else:
            self._year      = 2022   # dry scenario for fixed evaluation
            budget_frac     = 1.0

        self._budget_mm    = FULL_SEASON_NEED_MM * budget_frac
        self._water_used   = 0.0
        self._day          = 0
        self._prev_actions = np.zeros(N_AGENTS, dtype=np.float32)

        # climate for this year
        self._climate = extract_scenario(_CLIMATE_DF, self._year, _CROP)

        # precomputed biological arrays
        # get_precomputed only accepts named scenario strings ('dry','moderate','wet').
        # For the three named test years use the string key and the disk cache.
        # For all other training years, compute on the fly from the already-loaded
        # climate dict — this is the documented path in precompute.py:153.
        scenario = _SCENARIO_YEAR_MAP.get(self._year)
        if scenario is not None:
            self._precomp = get_precomputed(scenario, 'rice')
        else:
            self._precomp = compute_precomputed_from_climate(
                self._climate, 'rice', scenario_tag=str(self._year)
            )

        # construct and reset ABM
        self._abm = CropSoilABM(
            gamma_flat=_TERRAIN['gamma_flat'],
            sends_to=_TERRAIN['sends_to'],
            Nr=_TERRAIN['Nr'],
            theta=_CROP,
            N=_TERRAIN['N'],
            runoff_mode='cascade',
            elevation=_TERRAIN['elevation_flat'],
        )
        self._abm.reset()
        self.abm = self._abm   # public alias

        self._prev_x4_mean = float(np.mean(self._abm.x4))
        return self._build_obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # 1. clip and scale
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        irr_mm = action * UB_MM

        # 2. hard cap so the FIELD-MEAN irrigation this step does not push
        #    the cumulative field-mean water past the budget.
        #    Budget and _water_used are field-mean depths in mm.
        #    Cap each agent at the remaining budget (so even if every agent
        #    were at the cap, mean(irr_mm) <= remaining and the field-mean
        #    accumulator stays within budget).
        remaining = max(self._budget_mm - self._water_used, 0.0)
        irr_mm    = np.minimum(irr_mm, remaining)

        # 3. climate dict for today
        d = min(self._day, _K - 1)
        climate_today = {
            'rainfall':  float(self._climate['rainfall'][d]),
            'temp_mean': float(self._climate['temp_mean'][d]),
            'temp_max':  float(self._climate['temp_max'][d]),
            'radiation': float(self._climate['radiation'][d]),
            'ET':        float(self._climate['ET'][d]),
        }

        # 4. advance ABM, accumulate FIELD-MEAN depth (NOT per-agent times N)
        new_state         = self._abm.step(irr_mm, climate_today)
        water_step_field  = float(np.mean(irr_mm))    # mm of field-mean depth
        self._water_used += water_step_field          # mm of cumulative depth

        # 5. extract state arrays
        x1      = new_state['x1']
        x4_mean = float(np.mean(new_state['x4']))

        # 6. reward
        reward = self._compute_reward(
            x1=x1,
            x4_mean=x4_mean,
            irr_mm=irr_mm,
            action=action,
        )

        self._day         += 1
        self._prev_actions = action.copy()
        self._prev_x4_mean = x4_mean

        # 7. termination
        season_done = (self._day >= _K)
        budget_done = (self._water_used >= self._budget_mm - 1e-6)
        terminated  = budget_done
        truncated   = season_done and not budget_done

        # 8. terminal bonus (disabled in v2.6)
        if (terminated or truncated) and C_TERM > 0.0:
            reward += C_TERM * ALPHA1 * x4_mean / X4_REF

        info = {
            'day':           self._day,
            'water_used_mm': self._water_used,
            'budget_mm':     self._budget_mm,
            'x4_mean':       x4_mean,
            'yield_kg_ha':   x4_mean * _HI * 10.0,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    # ── reward ────────────────────────────────────────────────────────────────
    def _compute_reward(
        self,
        x1: np.ndarray,
        x4_mean: float,
        irr_mm: np.ndarray,
        action: np.ndarray,
    ) -> float:
        r1 = ALPHA1 * (x4_mean - self._prev_x4_mean) / X4_REF

        r2 = -ALPHA2 * float(np.mean(irr_mm)) / UB_MM

        drought = np.maximum(_ST_MM - x1, 0.0)
        r3 = -ALPHA3 * float(np.mean(drought)) / max(_ST_MM - _WP_MM, 1e-6)

        r5 = 0.0
        if ALPHA5_RL > 0.0:
            delta_u = action - self._prev_actions
            r5 = -ALPHA5_RL * float(np.mean(delta_u ** 2))

        overshoot = np.maximum(x1 - _FC_MM, 0.0)
        r6 = -ALPHA6 * float(np.mean(overshoot ** 2)) / max(_FC_MM ** 2, 1e-6)

        rb = 0.0
        if self._day > 0 and self._budget_mm > 0:
            # All quantities are field-mean depths in mm.  burn_rate compares
            # cumulative field-mean water against the linear expected pace.
            daily_pace = FULL_SEASON_NEED_MM / _K
            burn_rate  = self._water_used / max(self._day * daily_pace, 1e-6)
            rb = -LAMBDA_BUDGET * max(burn_rate - 1.0, 0.0) ** 2

        return r1 + r2 + r3 + r5 + r6 + rb

    # ── observation ───────────────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        d = min(self._day, _K - 1)
        p = self._precomp

        # per-agent block (agent-major: 5 contiguous features per agent)
        x1_norm = np.clip((self._abm.x1 - _WP_MM) / max(_FC_MM - _WP_MM, 1e-6),
                          0.0, 1.5)
        x5_norm = np.clip(self._abm.x5 / X5_REF, 0.0, 2.0)
        x4_norm = np.clip(self._abm.x4 / X4_REF, 0.0, 1.5)
        x3      = np.clip(self._abm.x3, 0.0, 2.0)
        gamma   = np.clip(self._abm.x2 / _GDD_MATURITY, 0.0, 1.0)
        agent_block = np.stack(
            [x1_norm, x5_norm, x4_norm, x3, gamma], axis=1
        ).flatten().astype(np.float32)   # (650,)

        # scalar block
        day_frac          = self._day / _K
        budget_remaining  = max(self._budget_mm - self._water_used, 0.0)
        budget_frac       = budget_remaining / max(self._budget_mm, 1e-6)
        budget_total_norm = self._budget_mm / FULL_SEASON_NEED_MM
        if self._day > 0:
            daily_pace = FULL_SEASON_NEED_MM / _K
            burn_rate  = self._water_used / max(self._day * daily_pace, 1e-6)
        else:
            burn_rate = 0.0

        scalar_block = np.array([
            day_frac,
            budget_frac,
            budget_total_norm,
            burn_rate,
            float(self._climate['rainfall'][d]),
            float(p.Kc_ET[d]),
            float(p.h2[d]),
            float(p.h7[d]),
            float(p.g_base[d]),
        ], dtype=np.float32)   # (9,)

        # forecast block
        def _fc_slice(arr, start, length):
            arr = np.asarray(arr, dtype=np.float32)
            end = min(start + length, len(arr))
            chunk = arr[start:end]
            if len(chunk) < length:
                fill = chunk[-1] if len(chunk) > 0 else 0.0
                chunk = np.concatenate([
                    chunk,
                    np.full(length - len(chunk), fill, dtype=np.float32)
                ])
            return chunk

        forecast_block = np.concatenate([
            _fc_slice(self._climate['rainfall'],  d, FORECAST_H),
            _fc_slice(p.Kc_ET,                    d, FORECAST_H),
            _fc_slice(self._climate['radiation'], d, FORECAST_H),
            _fc_slice(p.h2,                       d, FORECAST_H),
            _fc_slice(p.h7,                       d, FORECAST_H),
            _fc_slice(p.g_base,                   d, FORECAST_H),
        ]).astype(np.float32)   # (48,)

        obs = np.concatenate([agent_block, scalar_block, forecast_block])
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape}"
        return obs
