# src/rl/gym_env.py  v2.8.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.7.0  (see change_spec_v28.md for full rationale)
#
#   1. NEW FEATURE — x1_overshoot_norm added as the 9th per-agent feature.
#        Defined as max(x1 − FC, 0) / FC, clipped to [0, 1].  Equals zero
#        whenever the agent is in the healthy regime (x1 ≤ FC), grows
#        linearly above.  This is the SAME quantity that gets squared and
#        averaged in the r6 reward, so the gradient signal from r6 is
#        maximally informative about which feature should change.  Tackles
#        the v2.7 wet-year weakness: corr(u, x1) ≈ 0 across both seeds.
#        Per-agent block:  8 features → 9 features
#        Total OBS_DIM:    1097 → 1227
#
#   2. EPISODE-LENGTH CURRICULUM — short episodes during warmup.
#        For the first CURRICULUM_WARMUP_STEPS env transitions (default
#        50 000), episodes truncate at CURRICULUM_SHORT_LEN days (default
#        60).  After that, episodes return to the full 93-day length.
#        Reduces the high-variance return distribution that drove the
#        v2.7 critic explosion around step 165k.
#
# Backwards compatibility:
#   - The v2.7 8-feature observation layout remains importable via
#     networks.py's V27_* constants.  The runner can load v2.7 checkpoints
#     and produce 8-feature observations for them.
#   - The reward function, action space, ABM interface, and SAC
#     hyperparameters are all unchanged from v2.7.
#
# Interface dependencies (unchanged from v2.7):
#   abm.py:
#     CropSoilABM(gamma_flat, sends_to, Nr, theta, N, runoff_mode, elevation)
#     .reset(), .step(u, climate_dict)
#
#   soil_data.py:
#     get_crop('rice') → dict with theta2, theta5, theta6, theta18, HI, p, …
#
#   src/terrain.py:
#     load_terrain('gilan_farm.tif')
#     → dict: 'gamma_flat', 'sends_to', 'Nr', 'Nr_internal', 'N',
#             'elevation_flat', 'topological_order', …
#
#   climate_data.py:
#     TRAINING_YEARS, load_cleaned_data, extract_scenario
#
#   src/precompute.py:
#     get_precomputed(scenario_or_year, crop_name) → Precomputed
#     compute_precomputed_from_climate(climate_dict, crop_name, scenario_tag)
#
# Public names exported (consumed by src/rl/runner.py):
#   UB_MM, X4_REF, X5_REF, FULL_SEASON_NEED_MM
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
UB_MM               = 12.0    # actuator upper bound mm/day
X4_REF              = 600.0   # reference biomass for normalisation (g/m²)
X5_REF              = 50.0    # reference surface ponding (mm)
FULL_SEASON_NEED_MM = 484.0   # 100% seasonal budget reference (mm)
FORECAST_H          = 8       # forecast horizon (days)

# ── reward weights (unchanged from v2.7) ──────────────────────────────────────
ALPHA1 = 1.0     # biomass increment
ALPHA2 = 0.016   # water cost
ALPHA3 = 0.1     # drought stress regulariser
ALPHA6 = 8.0     # FC-overshoot penalty
C_TERM = 0.0     # terminal bonus (kept as 0)

# ── curriculum defaults (NEW in v2.8) ─────────────────────────────────────────
CURRICULUM_WARMUP_STEPS_DEFAULT = 50_000    # transition point (env steps)
CURRICULUM_SHORT_LEN_DEFAULT    = 60        # short-episode length (days)

# ── environment dimensions (v2.8) ─────────────────────────────────────────────
N_AGENTS         = 130
N_AGENT_FEATURES = 9     # v2.8: 4 dynamic + 4 static topo + 1 overshoot
N_GLOBAL_DIMS    = 57    # 9 scalars + 48 forecast
OBS_DIM          = N_AGENT_FEATURES * N_AGENTS + N_GLOBAL_DIMS    # 1227


# ── module-level asset cache (loaded once per process) ───────────────────────
def _load_assets():
    crop    = get_crop('rice')
    terrain = load_terrain('gilan_farm.tif')
    df      = load_cleaned_data()
    return crop, terrain, df


_CROP, _TERRAIN, _CLIMATE_DF = _load_assets()

# per-crop derived thresholds
_FC_MM = _CROP['theta6'] * _CROP['theta5']           # field capacity (mm)
_WP_MM = _CROP['theta2'] * _CROP['theta5']           # wilting point  (mm)
_ST_MM = _FC_MM - _CROP['p'] * (_FC_MM - _WP_MM)     # stress threshold (mm)
_HI    = _CROP['HI']                                  # harvest index
_K     = _CROP['season_days']                         # season length (93)
_GDD_MATURITY = _CROP.get('theta18', 1250.0)

_SCENARIO_YEAR_MAP = {2022: 'dry', 2018: 'moderate', 2024: 'wet'}


# ── Static per-agent topographic features (unchanged from v2.7) ──────────────
# These are constant across the season; computed once at module load.

_ELEV_NORM = _TERRAIN['gamma_flat'].astype(np.float32)

_NR_NORM = np.array(
    [_TERRAIN['Nr'][n] / 8.0 for n in range(_TERRAIN['N'])],
    dtype=np.float32,
)

_NR_INTERNAL_NORM = np.array(
    [_TERRAIN['Nr_internal'][n] / 8.0 for n in range(_TERRAIN['N'])],
    dtype=np.float32,
)

_n_upstream_counts = np.zeros(_TERRAIN['N'], dtype=np.int32)
for _n_src, _downstream_list in _TERRAIN['sends_to'].items():
    for _m_dst in _downstream_list:
        _n_upstream_counts[_m_dst] += 1
_N_UPSTREAM_NORM = (_n_upstream_counts / 8.0).astype(np.float32)


class IrrigationEnv(gym.Env):
    """Gymnasium wrapper around the 130-agent crop-soil ABM (v2.8).

    Observation (1227-dim, agent-major layout):
      Per-agent block  (1170 = 9 × 130):
        DYNAMIC (updated each step):
          [0] x1_norm             — (x1 − WP)/(FC − WP), in [0, 1.5]
          [1] x5_norm             — surface ponding / X5_REF
          [2] x4_norm             — biomass / X4_REF
          [3] x3                  — accumulated maturation stress
        STATIC (computed once at module load):
          [4] elev_norm           — normalised elevation (Chapter 4 γ⁽ⁿ⁾)
          [5] Nr_norm             — total downhill fanout / 8
          [6] Nr_internal_norm    — internal-only fanout / 8
          [7] n_upstream_norm     — upstream feeders / 8
        DYNAMIC (v2.8 NEW):
          [8] x1_overshoot_norm   — max(x1 − FC, 0) / FC, in [0, 1]
      Scalar block (9, unchanged): day_frac, budget_frac, budget_total_norm,
        burn_rate, rain_today, ETc_today, h2, h7, g_base.
      Forecast block (48, unchanged): rain[0:8], ETc[0:8], rad[0:8],
        h2[0:8], h7[0:8], g_base[0:8].

    Action (130-dim, Box[0,1]): scaled to [0, UB_MM = 12] mm/day.

    Reward (unchanged from v2.7, four terms):
      r(t) = r1 + r2 + r3 + r6.

    Episode termination (v2.8):
      Always terminated=False; truncated triggered when
      `self._day >= self._truncation_day`, where `truncation_day` is set
      at reset to CURRICULUM_SHORT_LEN during the warmup window and to
      _K (93) afterwards.  Budget exhaustion does NOT terminate the
      episode (preserves v2.7 lifecycle).

    Constructor kwargs:
      randomize : bool
          If True, sample year from TRAINING_YEARS and budget from
          U(0.7, 1.0) on each reset.  Set False for fixed-mode evaluation.
      curriculum_warmup_steps : int
          Number of env transitions before switching from short to full
          episodes.  Default 50 000.  Set to 0 to disable the curriculum
          entirely (always full episodes — matches v2.7 behaviour).
      curriculum_short_len : int
          Episode length in days during the warmup window.  Default 60.
    """

    metadata = {"render_modes": []}
    N = N_AGENTS

    def __init__(
        self,
        randomize: bool = True,
        curriculum_warmup_steps: int = CURRICULUM_WARMUP_STEPS_DEFAULT,
        curriculum_short_len:    int = CURRICULUM_SHORT_LEN_DEFAULT,
    ):
        super().__init__()
        self.randomize = randomize
        self._curriculum_warmup_steps = int(curriculum_warmup_steps)
        self._curriculum_short_len    = int(curriculum_short_len)

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
        self._prev_x4_mean: float = 0.0

        # curriculum state (v2.8)
        self._global_step_count: int = 0   # increments on every step() call
        self._truncation_day:    int = _K  # set on each reset

        # public alias for smoke tests
        self.abm: CropSoilABM | None = None

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            self._year  = int(self.np_random.choice(list(TRAINING_YEARS)))
            budget_frac = float(self.np_random.uniform(0.70, 1.00))
        else:
            self._year  = 2022   # dry scenario for fixed evaluation
            budget_frac = 1.0

        self._budget_mm  = FULL_SEASON_NEED_MM * budget_frac
        self._water_used = 0.0
        self._day        = 0

        # Curriculum: decide this episode's truncation day at the start of
        # the episode, so we don't switch mid-episode.
        if (self._curriculum_warmup_steps > 0
                and self._global_step_count < self._curriculum_warmup_steps):
            self._truncation_day = self._curriculum_short_len
        else:
            self._truncation_day = _K

        # climate
        self._climate = extract_scenario(_CLIMATE_DF, self._year, _CROP)

        # precomputed biological arrays
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
        self.abm = self._abm

        self._prev_x4_mean = float(np.mean(self._abm.x4))
        return self._build_obs(), {}

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # 1. clip and scale
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        irr_mm = action * UB_MM

        # 2. per-step budget clip (unchanged from v2.7)
        remaining = max(self._budget_mm - self._water_used, 0.0)
        irr_mm    = np.minimum(irr_mm, remaining)

        # 3. climate for today
        d = min(self._day, _K - 1)
        climate_today = {
            'rainfall':  float(self._climate['rainfall'][d]),
            'temp_mean': float(self._climate['temp_mean'][d]),
            'temp_max':  float(self._climate['temp_max'][d]),
            'radiation': float(self._climate['radiation'][d]),
            'ET':        float(self._climate['ET'][d]),
        }

        # 4. advance ABM, accumulate field-mean water depth
        new_state         = self._abm.step(irr_mm, climate_today)
        water_step_field  = float(np.mean(irr_mm))
        self._water_used += water_step_field

        # 5. extract state arrays
        x1      = new_state['x1']
        x4_mean = float(np.mean(new_state['x4']))

        # 6. reward (unchanged)
        reward = self._compute_reward(x1=x1, x4_mean=x4_mean, irr_mm=irr_mm)

        # 7. advance counters
        self._day += 1
        self._global_step_count += 1
        self._prev_x4_mean = x4_mean

        # 8. termination (v2.8)
        #    terminated=False always (no early termination from budget).
        #    truncated when day reaches the curriculum-dependent truncation day.
        terminated = False
        truncated  = (self._day >= self._truncation_day)

        info = {
            'day':              self._day,
            'water_used_mm':    self._water_used,
            'budget_mm':        self._budget_mm,
            'x4_mean':          x4_mean,
            'yield_kg_ha':      x4_mean * _HI * 10.0,
            'truncation_day':   self._truncation_day,
            'global_step':      self._global_step_count,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    # ── reward (unchanged from v2.7) ──────────────────────────────────────────
    def _compute_reward(
        self,
        x1: np.ndarray,
        x4_mean: float,
        irr_mm: np.ndarray,
    ) -> float:
        r1 = ALPHA1 * (x4_mean - self._prev_x4_mean) / X4_REF
        r2 = -ALPHA2 * float(np.mean(irr_mm)) / UB_MM
        drought   = np.maximum(_ST_MM - x1, 0.0)
        r3 = -ALPHA3 * float(np.mean(drought)) / max(_ST_MM - _WP_MM, 1e-6)
        overshoot = np.maximum(x1 - _FC_MM, 0.0)
        r6 = -ALPHA6 * float(np.mean(overshoot ** 2)) / max(_FC_MM ** 2, 1e-6)
        return r1 + r2 + r3 + r6

    # ── observation (v2.8: 9-feature per-agent block) ─────────────────────────
    def _build_obs(self) -> np.ndarray:
        d = min(self._day, _K - 1)
        p = self._precomp

        # ── dynamic per-agent features ──────────────────────────────────────
        x1_norm = np.clip(
            (self._abm.x1 - _WP_MM) / max(_FC_MM - _WP_MM, 1e-6),
            0.0, 1.5,
        )
        x5_norm = np.clip(self._abm.x5 / X5_REF, 0.0, 2.0)
        x4_norm = np.clip(self._abm.x4 / X4_REF, 0.0, 1.5)
        x3      = np.clip(self._abm.x3, 0.0, 2.0)

        # v2.8 NEW: explicit FC-overshoot feature.  Same quantity that
        # appears in r6 = -α6 × mean(this^2) / FC, giving the gradient
        # from r6 a direct, named feature to flow into.
        x1_overshoot_norm = np.clip(
            np.maximum(self._abm.x1 - _FC_MM, 0.0) / max(_FC_MM, 1e-6),
            0.0, 1.0,
        ).astype(np.float32)

        # Per-agent block: 9 features, agent-major (stack axis=1 + flatten).
        agent_block = np.stack([
            x1_norm,
            x5_norm,
            x4_norm,
            x3,
            _ELEV_NORM,
            _NR_NORM,
            _NR_INTERNAL_NORM,
            _N_UPSTREAM_NORM,
            x1_overshoot_norm,
        ], axis=1).flatten().astype(np.float32)   # (1170,)

        # ── scalar block (unchanged from v2.7) ──────────────────────────────
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
        ], dtype=np.float32)

        # ── forecast block (unchanged from v2.7) ────────────────────────────
        def _fc_slice(arr, start, length):
            arr = np.asarray(arr, dtype=np.float32)
            end = min(start + length, len(arr))
            chunk = arr[start:end]
            if len(chunk) < length:
                fill = chunk[-1] if len(chunk) > 0 else 0.0
                chunk = np.concatenate([
                    chunk,
                    np.full(length - len(chunk), fill, dtype=np.float32),
                ])
            return chunk

        forecast_block = np.concatenate([
            _fc_slice(self._climate['rainfall'],  d, FORECAST_H),
            _fc_slice(p.Kc_ET,                    d, FORECAST_H),
            _fc_slice(self._climate['radiation'], d, FORECAST_H),
            _fc_slice(p.h2,                       d, FORECAST_H),
            _fc_slice(p.h7,                       d, FORECAST_H),
            _fc_slice(p.g_base,                   d, FORECAST_H),
        ]).astype(np.float32)

        obs = np.concatenate([agent_block, scalar_block, forecast_block])
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape}, expected ({OBS_DIM},)"
        return obs
