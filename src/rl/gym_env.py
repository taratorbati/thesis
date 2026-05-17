# src/rl/gym_env.py  v2.7.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.6.x  (see change_spec_v27.md for full rationale)
#
#   1. BUG FIX — restored Chapter 4 spec compliance.
#        The 5th per-agent feature was, since commit c623833, accidentally
#        x2/theta18 (a field-uniform GDD scalar) instead of the normalised
#        elevation gamma_flat that Chapter 4 specifies.  The slot is now
#        renamed ``elev_norm`` in the obs builder to prevent the variable-
#        name collision (terrain "gamma" vs agronomic "gamma") from
#        recurring.
#
#   2. ENRICHMENT — per-agent block now carries three additional static
#        topographic features:
#           Nr_norm          = Nr / 8.0           (total downhill fanout)
#           Nr_internal_norm = Nr_internal / 8.0  (internal-only fanout)
#           n_upstream_norm  = (#feeders) / 8.0   (upstream feed count)
#        These give the shared actor the information the ABM uses
#        internally to route water (sends_to, Nr).
#        Per-agent block:   5 features → 8 features
#        Total OBS_DIM:     707 → 1097
#
#   3. REWARD SIMPLIFIED — burn-rate penalty (rb) and dead delta-u term
#        (r5, already inactive in v2.6) removed.  Final reward is
#                r = r1 + r2 + r3 + r6
#        The four-term form matches what was actually generating gradient
#        in v2.6; rb never bound on the converged policy.
#
#   4. EPISODE LIFECYCLE — episode now ALWAYS runs to the end of the
#        93-day season.  Budget exhaustion no longer terminates early.
#        The per-step clip irr_mm = min(irr_mm, remaining) is preserved
#        for physical compliance; after the budget runs out, effective
#        irrigation is 0 and the agent feels late-season drought through
#        r3 and reduced r1.  This delivers correctly time-weighted
#        overspend pain via the underlying biological dynamics without
#        introducing a tuned penalty hyperparameter.
#
# Interface dependencies (unchanged from v2.6):
#   abm.py:
#     CropSoilABM(gamma_flat, sends_to, Nr, theta, N, runoff_mode, elevation)
#     .reset()               → initialises x1/x2/x3/x4/x5 arrays
#     .step(u, climate_dict) → returns {'x1':…,'x2':…,'x3':…,'x4':…,'x5':…}
#
#   soil_data.py:
#     get_crop('rice') → dict with keys theta2, theta5, theta6, theta18, HI, p, …
#
#   src/terrain.py:
#     load_terrain('gilan_farm.tif')
#     → dict: 'gamma_flat'(N,), 'sends_to', 'Nr', 'Nr_internal', 'N',
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
UB_MM               = 12.0    # actuator upper bound mm/day (= runner.UB_MM_PER_DAY)
X4_REF              = 600.0   # reference biomass for normalisation (g/m²)
X5_REF              = 50.0    # reference surface ponding (mm)
FULL_SEASON_NEED_MM = 484.0   # 100% seasonal budget reference (mm)
FORECAST_H          = 8       # forecast horizon (days) — matches MPC Hp*

# ── reward weights (v2.7: ALPHA5_RL and LAMBDA_BUDGET removed) ───────────────
ALPHA1 = 1.0     # biomass increment
ALPHA2 = 0.016   # water cost
ALPHA3 = 0.1     # drought stress regulariser
ALPHA6 = 8.0     # FC-overshoot penalty
C_TERM = 0.0     # terminal bonus (kept as 0 for completeness; never paid in v2.7)

# ── environment dimensions ────────────────────────────────────────────────────
N_AGENTS         = 130
N_AGENT_FEATURES = 8     # v2.7: 4 dynamic (x1, x5, x4, x3) + 4 static topo
N_GLOBAL_DIMS    = 57    # 9 scalars + 48 forecast (6 vars × 8 days)
OBS_DIM          = N_AGENT_FEATURES * N_AGENTS + N_GLOBAL_DIMS    # 1097


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
_ST_MM = _FC_MM - _CROP['p'] * (_FC_MM - _WP_MM)     # stress threshold (mm)
_HI    = _CROP['HI']                                  # harvest index
_K     = _CROP['season_days']                         # season length (days)
_GDD_MATURITY = _CROP.get('theta18', 1250.0)          # GDD to maturity

# Scenario name → year int mapping (for precompute cache key)
_SCENARIO_YEAR_MAP = {2022: 'dry', 2018: 'moderate', 2024: 'wet'}


# ── Static per-agent topographic features (v2.7) ─────────────────────────────
# These are computed once at module load from _TERRAIN and never change during
# a season.  They are what the ABM uses internally to route water (sends_to,
# Nr); putting them in the actor's observation closes the asymmetry that
# caused SAC_best to be spatially blind in v2.6.
#
#   _ELEV_NORM        — normalised elevation (= terrain['gamma_flat']),
#                       restores the Chapter 4 spec for the 5th per-agent
#                       feature that was broken in commits c623833+.
#   _NR_NORM          — total downhill fanout / 8, in [0, ~0.75].  High value
#                       means "water I receive disperses to many neighbours".
#   _NR_INTERNAL_NORM — internal-only fanout / 8, in [0, ~0.75].  Comparing
#                       this with _NR_NORM tells the actor whether the
#                       agent is at a field boundary.
#   _N_UPSTREAM_NORM  — number of upstream feeders / 8, in [0, ~0.75].
#                       High value means "I receive runoff from many
#                       neighbours" — this is what makes valley cells
#                       hydrologically different from hilltop cells at
#                       similar elevation.
#
# All four are float32 numpy arrays of shape (N_AGENTS,).  Dividing by 8
# (the 8-directional neighbourhood ceiling for D8 routing) keeps values in
# [0, 1] with headroom for terrains with denser connectivity than Gilan.

_ELEV_NORM = _TERRAIN['gamma_flat'].astype(np.float32)

_NR_NORM = np.array(
    [_TERRAIN['Nr'][n] / 8.0 for n in range(_TERRAIN['N'])],
    dtype=np.float32,
)

_NR_INTERNAL_NORM = np.array(
    [_TERRAIN['Nr_internal'][n] / 8.0 for n in range(_TERRAIN['N'])],
    dtype=np.float32,
)

# n_upstream[m] = number of agents n such that m is in sends_to[n]
_n_upstream_counts = np.zeros(_TERRAIN['N'], dtype=np.int32)
for _n_src, _downstream_list in _TERRAIN['sends_to'].items():
    for _m_dst in _downstream_list:
        _n_upstream_counts[_m_dst] += 1
_N_UPSTREAM_NORM = (_n_upstream_counts / 8.0).astype(np.float32)


class IrrigationEnv(gym.Env):
    """Gymnasium wrapper around the 130-agent crop-soil ABM (v2.7).

    Observation (1097-dim, agent-major layout):
      Per-agent block  (1040 = 8 × 130):
        DYNAMIC (updated each step):
          [0] x1_norm  — root-zone moisture mapped via (x1 − WP)/(FC − WP)
          [1] x5_norm  — surface ponding / X5_REF
          [2] x4_norm  — biomass / X4_REF
          [3] x3       — accumulated maturation stress
        STATIC (computed once at module load):
          [4] elev_norm        — normalised elevation (Chapter 4 γ⁽ⁿ⁾)
          [5] Nr_norm          — total downhill fanout / 8
          [6] Nr_internal_norm — internal-only fanout / 8
          [7] n_upstream_norm  — upstream feeders / 8
      Scalar block (9):
          [0] day_frac        — day index / 93
          [1] budget_frac     — remaining / budget_total
          [2] budget_total_norm — budget_total / 484
          [3] burn_rate       — water_used / (day × daily_pace)
                                 (informative signal even though rb is
                                  no longer used in the reward)
          [4] rain_today, [5] ETc_today, [6] h2, [7] h7, [8] g_base
      Forecast block (48): rain[0:8], ETc[0:8], rad[0:8],
                            h2[0:8], h7[0:8], g_base[0:8]

    Action (130-dim, Box[0,1]):
      Scaled to [0, UB_MM = 12] mm/day in step().

    Reward (v2.7, four terms):
      r(t) = r1 + r2 + r3 + r6
        r1 = ALPHA1 × Δ(mean x4) / X4_REF
        r2 = −ALPHA2 × mean(irr_delivered) / UB_MM
        r3 = −ALPHA3 × mean(max(ST − x1, 0)) / (ST − WP)
        r6 = −ALPHA6 × mean(max(x1 − FC, 0)²) / FC²

    Episode: always runs 93 days.  terminated = False on every step;
      truncated = True only when day index reaches 93.  Budget compliance is
      enforced inside step() via the per-step clip and is therefore
      physically guaranteed for every reported run.
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
        self._prev_x4_mean: float = 0.0

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

        # climate for this year
        self._climate = extract_scenario(_CLIMATE_DF, self._year, _CROP)

        # precomputed biological arrays
        # get_precomputed only accepts named scenario strings ('dry','moderate','wet').
        # For training years not in that set, compute on the fly from the loaded
        # climate dict.
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

        # 2. per-step budget clip: physical compliance guarantee.
        #    When budget is exhausted, remaining == 0, effective irrigation
        #    becomes 0, and the ABM advances under climate alone.  This
        #    lets the agent feel late-season drought via reduced r1 and
        #    rising r3, without any tuned overspend penalty.
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

        # 4. advance ABM, accumulate FIELD-MEAN water depth in mm
        new_state         = self._abm.step(irr_mm, climate_today)
        water_step_field  = float(np.mean(irr_mm))
        self._water_used += water_step_field

        # 5. extract state arrays
        x1      = new_state['x1']
        x4_mean = float(np.mean(new_state['x4']))

        # 6. reward
        reward = self._compute_reward(x1=x1, x4_mean=x4_mean, irr_mm=irr_mm)

        # 7. step the day counter (must precede termination logic)
        self._day += 1
        self._prev_x4_mean = x4_mean

        # 8. v2.7 termination logic
        #    Episode ALWAYS runs the full season — no early termination on
        #    budget exhaustion.  truncated fires only at day == K, signalling
        #    the natural end of the season (gymnasium convention: truncated
        #    means "time limit reached", terminated means "absorbing state").
        terminated = False
        truncated  = (self._day >= _K)

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
    ) -> float:
        """Four-term reward (v2.7): r = r1 + r2 + r3 + r6.

        r5 (delta-u) was disabled in v2.6; the entire branch is removed.
        rb (burn-rate) never bound on the v2.6 converged policy; removed.
        The remaining four terms exactly match the MPC path-cost terms
        whose weights were validated by the Chapter 4 α-sensitivity sweep.
        """
        r1 = ALPHA1 * (x4_mean - self._prev_x4_mean) / X4_REF

        r2 = -ALPHA2 * float(np.mean(irr_mm)) / UB_MM

        drought = np.maximum(_ST_MM - x1, 0.0)
        r3 = -ALPHA3 * float(np.mean(drought)) / max(_ST_MM - _WP_MM, 1e-6)

        overshoot = np.maximum(x1 - _FC_MM, 0.0)
        r6 = -ALPHA6 * float(np.mean(overshoot ** 2)) / max(_FC_MM ** 2, 1e-6)

        return r1 + r2 + r3 + r6

    # ── observation ───────────────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        d = min(self._day, _K - 1)
        p = self._precomp

        # ── per-agent block (8 features per agent, agent-major) ─────────────
        # Dynamic features — updated every step.
        x1_norm = np.clip(
            (self._abm.x1 - _WP_MM) / max(_FC_MM - _WP_MM, 1e-6),
            0.0, 1.5,
        )
        x5_norm = np.clip(self._abm.x5 / X5_REF, 0.0, 2.0)
        x4_norm = np.clip(self._abm.x4 / X4_REF, 0.0, 1.5)
        x3      = np.clip(self._abm.x3, 0.0, 2.0)

        # Static topographic features — module-level, broadcast in.
        # Stacking with axis=1 then flattening produces the agent-major
        # layout that SharedActor and FactorizedContinuousCritic expect.
        agent_block = np.stack([
            x1_norm,
            x5_norm,
            x4_norm,
            x3,
            _ELEV_NORM,
            _NR_NORM,
            _NR_INTERNAL_NORM,
            _N_UPSTREAM_NORM,
        ], axis=1).flatten().astype(np.float32)   # (1040,)

        # ── scalar block (9 dims, unchanged from v2.6) ──────────────────────
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
            burn_rate,                              # informative; not in reward
            float(self._climate['rainfall'][d]),
            float(p.Kc_ET[d]),
            float(p.h2[d]),
            float(p.h7[d]),
            float(p.g_base[d]),
        ], dtype=np.float32)   # (9,)

        # ── forecast block (48 dims, unchanged from v2.6) ───────────────────
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
        ]).astype(np.float32)   # (48,)

        obs = np.concatenate([agent_block, scalar_block, forecast_block])
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape}, expected ({OBS_DIM},)"
        return obs
