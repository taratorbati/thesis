# src/rl/gym_env.py
# -----------------------------------------------------------------------------
# Gymnasium environment wrapping the 130-cell crop-soil ABM for irrigation RL.
#
# Observation (agent-major, flat float32):
#   Per-cell block (N x 8):
#     dynamic: x1_norm, x5_norm, x4_norm, x3
#     static : elev_norm, Nr_norm, Nr_internal_norm, n_upstream_norm
#   Global block:
#     scalars : day_frac, budget_frac, budget_total_norm, burn_rate
#               [+ rain, ETc, h2, h7, g_base for today, if dedupe is off]
#     forecast: rain[0:8], ETc[0:8], rad[0:8], h2[0:8], h7[0:8], g_base[0:8]
#
#   The forecast slices start at the current day, so forecast[...][0] already IS
#   today's weather. The five "today" scalars therefore duplicate forecast[0].
#   ``dedupe_today_weather`` (default True) drops them, so today's weather is fed
#   exactly once: global block 52-dim (clean) vs 57-dim (legacy, kept so existing
#   checkpoints still load — see networks.py, which infers the width).
#
# Action (130-dim, Box[0, 1]): scaled to [0, UB_MM = 12] mm/day, then clipped to
# the remaining seasonal budget.
#
# Reward = r1 + r2 + r3 + r5 + r6 (+ terminal yield bonus on the final step):
#   r1  biomass increment          +ALPHA1 * d(x4)/X4_REF
#   r2  water cost                 -ALPHA2 * mean(u)/UB_MM
#   r3  drought-stress regulariser -ALPHA3 * mean(max(ST-x1,0)) / (ST-WP)
#   r5  control-rate smoothing     -du_alpha * mean(((u-u_prev)/UB_MM)^2)   (MPC term 5)
#   r6  field-capacity overshoot   -ALPHA6 * mean(max(x1-FC,0)) / FC        (linear; matches ABM h6)
#   terminal: +terminal_yield * x4_final/X4_REF, once at episode end
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.model.abm import CropSoilABM
from src.model.climate_data import TRAINING_YEARS, load_cleaned_data, extract_scenario
from src.sim.precompute import get_precomputed, compute_precomputed_from_climate
from src.model.terrain import load_terrain
from src.model.soil_data import get_crop

# ── public scalar constants (consumed by runner.py) ──────────────────────────
UB_MM = 12.0                 # actuator upper bound (mm/day)
X4_REF = 600.0               # reference biomass for normalisation (g/m^2)
X5_REF = 50.0                # reference surface ponding (mm)
FULL_SEASON_NEED_MM = 484.0  # 100% seasonal water budget reference (mm)
FORECAST_H = 8               # forecast horizon (days)

# Global/forecast feature normalisers (the actor sees ~[0, 1] for every channel).
# Denominators chosen against the 2000-2025 record maxima with headroom.
RAIN_REF = 30.0   # mm/day  (preserves moderate rain events; clips only outliers)
ETC_REF = 8.0     # mm/day  (record max ET0 ~ 7.05)
RAD_REF = 35.0    # MJ m^-2 d^-1  (record max ~ 31.69)

# ── reward weights ────────────────────────────────────────────────────────────
ALPHA1 = 1.0       # biomass increment
ALPHA2 = 0.016     # water cost (mirrors MPC alpha2)
ALPHA3 = 0.1       # drought-stress regulariser
ALPHA6 = 1.5       # FC-overshoot penalty, linear (aligned with the ABM's linear h6)

# ── environment dimensions ────────────────────────────────────────────────────
N_AGENTS = 130
N_AGENT_FEATURES = 8
SCALARS_BASE = 4                 # day_frac, budget_frac, budget_total_norm, burn_rate
SCALARS_TODAY = 5                # rain, ETc, h2, h7, g_base (today) — dropped when deduped
FORECAST_DIMS = 6 * FORECAST_H   # 48


# ── module-level asset cache (loaded once per process) ───────────────────────
def _load_assets():
    return get_crop('rice'), load_terrain('data/gilan_farm.tif'), load_cleaned_data()


_CROP, _TERRAIN, _CLIMATE_DF = _load_assets()

_FC_MM = _CROP['theta6'] * _CROP['theta5']           # field capacity (mm)
_WP_MM = _CROP['theta2'] * _CROP['theta5']           # wilting point  (mm)
_ST_MM = _FC_MM - _CROP['p'] * (_FC_MM - _WP_MM)     # stress threshold (mm)
_HI = _CROP['HI']                                    # harvest index
_K = _CROP['season_days']                            # season length (93)

_SCENARIO_YEAR_MAP = {2022: 'dry', 2018: 'moderate', 2024: 'wet'}

# Static per-cell topographic features (constant across the season).
_ELEV_NORM = _TERRAIN['gamma_flat'].astype(np.float32)
_NR_NORM = np.array([_TERRAIN['Nr'][n] / 8.0 for n in range(_TERRAIN['N'])], dtype=np.float32)
_NR_INTERNAL_NORM = np.array(
    [_TERRAIN['Nr_internal'][n] / 8.0 for n in range(_TERRAIN['N'])], dtype=np.float32)
_n_upstream_counts = np.zeros(_TERRAIN['N'], dtype=np.int32)
for _n_src, _downstream_list in _TERRAIN['sends_to'].items():
    for _m_dst in _downstream_list:
        _n_upstream_counts[_m_dst] += 1
_N_UPSTREAM_NORM = (_n_upstream_counts / 8.0).astype(np.float32)


def global_dim(dedupe_today_weather: bool = True) -> int:
    """Width of the global observation block for a given dedupe setting."""
    scalars = SCALARS_BASE if dedupe_today_weather else SCALARS_BASE + SCALARS_TODAY
    return scalars + FORECAST_DIMS


class IrrigationEnv(gym.Env):
    """Gymnasium wrapper around the 130-cell crop-soil ABM.

    Constructor kwargs
    ------------------
    randomize : bool
        If True, sample a training year and a budget ~ U(0.70, 1.00) per reset.
        If False (and no eval_schedule), use the fixed dry 2022 scenario at 100%.
    eval_schedule : list[(year, budget_frac)] | None
        Deterministic held-out evaluation: reset() walks this fixed list in
        order (bypassing ``randomize``) so every checkpoint is scored on the
        identical set of episodes.
    dedupe_today_weather : bool
        If True (default), today's weather is fed once (via the forecast block);
        the redundant today-scalars are dropped (52-dim global block). If False,
        they are kept (57-dim) to match legacy checkpoints.
    reward_du_alpha : float
        Weight of the control-rate smoothing penalty r5 (0 disables it).
    reward_terminal_yield : float
        Weight of the additive terminal-yield bonus paid once at episode end.
    reward_alpha3 : float
        Weight of the drought-stress penalty r3 (defaults to the module ALPHA3).
    """

    metadata = {"render_modes": []}
    N = N_AGENTS

    def __init__(
        self,
        randomize: bool = True,
        eval_schedule: "list | None" = None,
        dedupe_today_weather: bool = True,
        reward_du_alpha: float = 0.0,
        reward_terminal_yield: float = 0.0,
        reward_alpha3: float = ALPHA3,
    ):
        super().__init__()
        self.randomize = randomize
        self._dedupe_today_weather = bool(dedupe_today_weather)

        if reward_du_alpha < 0.0:
            raise ValueError(f"reward_du_alpha must be >= 0, got {reward_du_alpha!r}")
        self._reward_du_alpha = float(reward_du_alpha)

        if reward_terminal_yield < 0.0:
            raise ValueError(f"reward_terminal_yield must be >= 0, got {reward_terminal_yield!r}")
        self._reward_terminal_yield = float(reward_terminal_yield)

        if reward_alpha3 < 0.0:
            raise ValueError(f"reward_alpha3 must be >= 0, got {reward_alpha3!r}")
        self._alpha3 = float(reward_alpha3)

        if eval_schedule is not None:
            parsed = []
            for item in eval_schedule:
                if len(item) != 2:
                    raise ValueError(
                        f"eval_schedule entries must be (year, budget_frac) pairs; got {item!r}")
                yr, bf = item
                parsed.append((int(yr), float(bf)))
            if len(parsed) == 0:
                raise ValueError("eval_schedule must be non-empty or None")
            eval_schedule = parsed
        self._eval_schedule = eval_schedule
        self._eval_idx = 0

        obs_dim = N_AGENT_FEATURES * N_AGENTS + global_dim(self._dedupe_today_weather)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_AGENTS,), dtype=np.float32)

        # state — initialised in reset()
        self._abm: CropSoilABM | None = None
        self._precomp = None
        self._climate: dict | None = None
        self._year: int | None = None
        self._budget_mm: float = FULL_SEASON_NEED_MM
        self._water_used: float = 0.0
        self._day: int = 0
        self._prev_x4_mean: float = 0.0
        self._prev_irr_mm = None
        self._last_reward_terms: dict = {}
        self.abm: CropSoilABM | None = None   # public alias for smoke tests

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._eval_schedule is not None:
            yr, bf = self._eval_schedule[self._eval_idx % len(self._eval_schedule)]
            self._eval_idx += 1
            self._year, budget_frac = int(yr), float(bf)
        elif self.randomize:
            self._year = int(self.np_random.choice(list(TRAINING_YEARS)))
            budget_frac = float(self.np_random.uniform(0.70, 1.00))
        else:
            self._year, budget_frac = 2022, 1.0   # fixed dry scenario

        self._budget_mm = FULL_SEASON_NEED_MM * budget_frac
        self._water_used = 0.0
        self._day = 0
        self._climate = extract_scenario(_CLIMATE_DF, self._year, _CROP)

        scenario = _SCENARIO_YEAR_MAP.get(self._year)
        if scenario is not None:
            self._precomp = get_precomputed(scenario, 'rice')
        else:
            self._precomp = compute_precomputed_from_climate(
                self._climate, 'rice', scenario_tag=str(self._year))

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
        self.abm = self._abm

        self._prev_x4_mean = float(np.mean(self._abm.x4))
        self._prev_irr_mm = None
        return self._build_obs(), {}

    def reset_eval_schedule(self) -> None:
        """Rewind the deterministic eval schedule to its first episode (no-op if unset)."""
        self._eval_idx = 0

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        irr_mm = action * UB_MM
        remaining = max(self._budget_mm - self._water_used, 0.0)
        irr_mm = np.minimum(irr_mm, remaining)

        d = min(self._day, _K - 1)
        climate_today = {
            'rainfall':  float(self._climate['rainfall'][d]),
            'temp_mean': float(self._climate['temp_mean'][d]),
            'temp_max':  float(self._climate['temp_max'][d]),
            'radiation': float(self._climate['radiation'][d]),
            'ET':        float(self._climate['ET'][d]),
        }

        new_state = self._abm.step(irr_mm, climate_today)
        self._water_used += float(np.mean(irr_mm))

        x1 = new_state['x1']
        x4_mean = float(np.mean(new_state['x4']))
        reward = self._compute_reward(x1=x1, x4_mean=x4_mean, irr_mm=irr_mm)

        self._day += 1
        self._prev_x4_mean = x4_mean
        self._prev_irr_mm = np.asarray(irr_mm, dtype=np.float64).copy()

        terminated = False
        truncated = (self._day >= _K)

        r_term = 0.0
        if truncated and self._reward_terminal_yield > 0.0:
            r_term = self._reward_terminal_yield * x4_mean / X4_REF
            reward = float(reward) + r_term
        self._last_reward_terms['r_term'] = r_term

        info = {
            'day':            self._day,
            'water_used_mm':  self._water_used,
            'budget_mm':      self._budget_mm,
            'x4_mean':        x4_mean,
            'yield_kg_ha':    x4_mean * _HI * 10.0,
            'r1_biomass':     self._last_reward_terms.get('r1', 0.0),
            'r2_water':       self._last_reward_terms.get('r2', 0.0),
            'r3_drought':     self._last_reward_terms.get('r3', 0.0),
            'r5_delta_u':     self._last_reward_terms.get('r5', 0.0),
            'r6_waterlog':    self._last_reward_terms.get('r6', 0.0),
            'r_term_yield':   self._last_reward_terms.get('r_term', 0.0),
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    # ── reward ──────────────────────────────────────────────────────────────
    def _compute_reward(self, x1: np.ndarray, x4_mean: float, irr_mm: np.ndarray) -> float:
        r1 = ALPHA1 * (x4_mean - self._prev_x4_mean) / X4_REF
        r2 = -ALPHA2 * float(np.mean(irr_mm)) / UB_MM
        drought = np.maximum(_ST_MM - x1, 0.0)
        r3 = -self._alpha3 * float(np.mean(drought)) / max(_ST_MM - _WP_MM, 1e-6)
        overshoot = np.maximum(x1 - _FC_MM, 0.0)
        r6 = -ALPHA6 * float(np.mean(overshoot)) / max(_FC_MM, 1e-6)

        r5 = 0.0
        if self._reward_du_alpha > 0.0 and self._prev_irr_mm is not None:
            du_norm = (np.asarray(irr_mm, dtype=np.float64) - self._prev_irr_mm) / UB_MM
            r5 = -self._reward_du_alpha * float(np.mean(du_norm ** 2))

        self._last_reward_terms = {'r1': r1, 'r2': r2, 'r3': r3, 'r5': r5, 'r6': r6}
        return r1 + r2 + r3 + r5 + r6

    # ── observation ───────────────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        d = min(self._day, _K - 1)
        p = self._precomp

        x1_norm = np.clip((self._abm.x1 - _WP_MM) / max(_FC_MM - _WP_MM, 1e-6), 0.0, 1.5)
        x5_norm = np.clip(self._abm.x5 / X5_REF, 0.0, 2.0)
        x4_norm = np.clip(self._abm.x4 / X4_REF, 0.0, 1.5)
        x3 = np.clip(self._abm.x3, 0.0, 2.0)

        agent_block = np.stack([
            x1_norm, x5_norm, x4_norm, x3,
            _ELEV_NORM, _NR_NORM, _NR_INTERNAL_NORM, _N_UPSTREAM_NORM,
        ], axis=1).flatten().astype(np.float32)

        day_frac = self._day / _K
        budget_remaining = max(self._budget_mm - self._water_used, 0.0)
        budget_frac = budget_remaining / max(self._budget_mm, 1e-6)
        budget_total_norm = self._budget_mm / FULL_SEASON_NEED_MM
        if self._day > 0:
            daily_pace = FULL_SEASON_NEED_MM / _K
            burn_rate = self._water_used / max(self._day * daily_pace, 1e-6)
        else:
            burn_rate = 0.0

        scalars = [day_frac, budget_frac, budget_total_norm, burn_rate]
        if not self._dedupe_today_weather:
            scalars += [
                float(self._climate['rainfall'][d]) / RAIN_REF,
                float(p.Kc_ET[d]) / ETC_REF,
                float(p.h2[d]),
                float(p.h7[d]),
                float(p.g_base[d]),
            ]
        scalar_block = np.array(scalars, dtype=np.float32)

        forecast_block = np.concatenate([
            _fc_slice(self._climate['rainfall'],  d, FORECAST_H) / RAIN_REF,
            _fc_slice(p.Kc_ET,                    d, FORECAST_H) / ETC_REF,
            _fc_slice(self._climate['radiation'], d, FORECAST_H) / RAD_REF,
            _fc_slice(p.h2,                       d, FORECAST_H),
            _fc_slice(p.h7,                       d, FORECAST_H),
            _fc_slice(p.g_base,                   d, FORECAST_H),
        ]).astype(np.float32)

        obs = np.concatenate([agent_block, scalar_block, forecast_block])
        expected = self.observation_space.shape[0]
        assert obs.shape == (expected,), f"obs shape {obs.shape}, expected ({expected},)"
        return obs


def _fc_slice(arr, start, length):
    """Forecast slice arr[start:start+length], right-padded with the last value."""
    arr = np.asarray(arr, dtype=np.float32)
    chunk = arr[start:min(start + length, len(arr))]
    if len(chunk) < length:
        fill = chunk[-1] if len(chunk) > 0 else 0.0
        chunk = np.concatenate([chunk, np.full(length - len(chunk), fill, dtype=np.float32)])
    return chunk
