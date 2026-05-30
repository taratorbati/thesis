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

# ── v2.12 GLOBAL/FORECAST FEATURE NORMALISATION (consumed by runner.py) ───────
# v2.7..v2.11 fed the global scalar block and the forecast block to the actor
# with RAW physical magnitudes (rainfall up to ~11 mm in-season / 64 mm in the
# raw record, Kc_ET up to ~7, radiation up to ~32).  The per-agent dynamic block
# was already normalised to [0, 1.5], but these raw global features dominated the
# actor's first-layer pre-activations and (combined with the LayerNorm-bounded
# critic gradient in v2.11) drove every first-layer ReLU below zero — a verified
# dead-ReLU collapse (≈0/128 units firing on real observations).  v2.12 divides
# each raw feature by a physical reference with headroom so all global features
# land in roughly [0, 1], matching the per-agent block.
#
# Denominators are chosen against the full 2000-2025 climate record maxima
# (rainfall 64.31 mm, radiation 31.69, ET0 7.05) with margin so even unseen
# extreme years stay <= ~1.0:
#     rainfall  : / 70.0   (record max 64.31)
#     Kc_ET     : /  8.0   (record max  7.05)
#     radiation : / 35.0   (record max 31.69)
# h2, h7, g_base already live in [0, ~1], so they are left unscaled.
RAIN_REF = 70.0    # mm/day normaliser for rainfall + rainfall forecast
ETC_REF  = 8.0     # mm/day normaliser for Kc_ET + Kc_ET forecast
RAD_REF  = 35.0    # MJ m^-2 d^-1 normaliser for radiation forecast
NORMALIZE_GLOBALS_DEFAULT = True   # v2.12 default; set False to reproduce v2.7/v2.11 obs

# ── reward weights (unchanged from v2.7) ──────────────────────────────────────
ALPHA1 = 1.0     # biomass increment
ALPHA2 = 0.016   # water cost
ALPHA3 = 0.1     # drought stress regulariser
ALPHA6 = 8.0     # FC-overshoot penalty - QUADRATIC shape (v2.7-v2.14 default)
C_TERM = 0.0     # terminal bonus (kept as 0)

# ── v2.15 NEW: alternative r6 shapes (single-variable test) ──────────────────
# The quadratic r6 (-ALPHA6 * mean(overshoot^2) / FC^2) has been the reward
# overshoot shape from v2.7 through v2.14.  v2.14 closed most of the gap to
# MPC but still over-irrigated in wet years (+37.5 mm vs MPC, +44 waterlog-
# days, WUE 9.34 vs 10.63).  Diagnostic on v2.14 wet/100 rollouts:
#   - actor's 10th-percentile daily action is 2.50 mm (vs MPC's 0.16 mm)
#   - corr(u, 7-day forward rain) ≈ +0.03 (vs MPC's -0.42, v2.7's -0.24)
# The actor cannot push u toward zero on rainy days.  Two compounding causes
# of the weak gradient signal at the operating point:
#   1. d(quadratic r6)/d(overshoot) = -2*ALPHA6*overshoot/FC^2 is small at
#      moderate overshoot (where the actor sits) -> weak dQ/du in those states.
#   2. The ABM's waterlog stress h6 is LINEAR in (x1-FC)/FC, but r6 is
#      QUADRATIC -> the reward proxy is not aligned with the physical yield loss.
# v2.15 changes r6 to a linear shape with the same season-sum magnitude:
#   r6_linear = -ALPHA6_LIN * mean(overshoot) / FC
# Calibration (against 27 v2.14 + MPC rollouts):
#   - quadratic r6 season-sum: 6.79 +- 7.86 (full distribution)
#                              16.99 +- 4.24 (wet-year only)
#   - linear r6 with ALPHA6_LIN=1.0 season-sum: 4.63 +- 4.22
#   - ALPHA6_LIN to match full-distribution season-sum:  1.4657
#   - ALPHA6_LIN to match gradient at RMS overshoot (13.4 mm):  1.5285
# Chosen ALPHA6_LIN = 1.5 bracketed by both calibrations; preserves r6
# dominance over r1+r2+r3 while uniformising the gradient across overshoot.
ALPHA6_LIN  = 1.5    # FC-overshoot penalty - LINEAR shape (v2.15)
ALPHA6_SQRT = 0.5    # FC-overshoot penalty - SQRT shape   (calibrated, unused default)

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
        use_overshoot_feature:   bool = True,
        normalize_globals:       bool = NORMALIZE_GLOBALS_DEFAULT,
        reward_overshoot_mode:   str  = 'quadratic',
    ):
        super().__init__()
        self.randomize = randomize
        self._curriculum_warmup_steps = int(curriculum_warmup_steps)
        self._curriculum_short_len    = int(curriculum_short_len)
        self._use_overshoot_feature   = bool(use_overshoot_feature)
        self._normalize_globals       = bool(normalize_globals)
        # v2.15: validate and store reward_overshoot_mode.  Default 'quadratic'
        # preserves byte-identical behaviour for v2.7-v2.14 training scripts.
        if reward_overshoot_mode not in ('quadratic', 'linear', 'sqrt'):
            raise ValueError(
                f"reward_overshoot_mode must be one of "
                f"'quadratic' (v2.7-v2.14 default), 'linear' (v2.15), or 'sqrt'. "
                f"Got: {reward_overshoot_mode!r}"
            )
        self._reward_overshoot_mode = str(reward_overshoot_mode)

        # obs dim depends on whether x1_overshoot_norm is included:
        #   use_overshoot_feature=True  (v2.8 default): 9 feat/agent → 1227-dim
        #   use_overshoot_feature=False (v2.9 / v2.7):  8 feat/agent → 1097-dim
        _n_feat  = 9 if self._use_overshoot_feature else 8
        _obs_dim = _n_feat * N_AGENTS + N_GLOBAL_DIMS

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32
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

    # ── reward (v2.15: r6 shape selectable; quadratic default = v2.7-v2.14) ──
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
        if self._reward_overshoot_mode == 'linear':
            # v2.15: linear in overshoot.  d(r6)/d(overshoot) = -ALPHA6_LIN/FC
            # is constant across the overshoot range, giving the critic uniform
            # gradient signal at moderate overshoot (where the actor sits).
            # Also aligned with the ABM's linear waterlog stress term h6.
            r6 = -ALPHA6_LIN * float(np.mean(overshoot)) / max(_FC_MM, 1e-6)
        elif self._reward_overshoot_mode == 'sqrt':
            # Sub-quadratic shape (steeper than quadratic at moderate overshoot,
            # gentler than linear at large overshoot).  Provided for ablation.
            r6 = -ALPHA6_SQRT * float(np.mean(np.sqrt(overshoot))) \
                 / max(np.sqrt(_FC_MM), 1e-6)
        else:
            # Default: quadratic (v2.7-v2.14 byte-identical behaviour).
            r6 = -ALPHA6 * float(np.mean(overshoot ** 2)) \
                 / max(_FC_MM ** 2, 1e-6)
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
        # Only included when use_overshoot_feature=True (v2.8 obs layout).
        # When False (v2.7/v2.9 obs layout) this block is skipped.
        if self._use_overshoot_feature:
            x1_overshoot_norm = np.clip(
                np.maximum(self._abm.x1 - _FC_MM, 0.0) / max(_FC_MM, 1e-6),
                0.0, 1.0,
            ).astype(np.float32)
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
            ], axis=1).flatten().astype(np.float32)   # (1170,) v2.8
        else:
            # v2.7 / v2.9: 8 features, no overshoot term
            agent_block = np.stack([
                x1_norm,
                x5_norm,
                x4_norm,
                x3,
                _ELEV_NORM,
                _NR_NORM,
                _NR_INTERNAL_NORM,
                _N_UPSTREAM_NORM,
            ], axis=1).flatten().astype(np.float32)   # (1040,) v2.7

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

        # ── scalar block ────────────────────────────────────────────────────
        # v2.12: rainfall / Kc_ET are divided by physical references so they
        # sit in ~[0, 1] like the per-agent block.  h2, h7, g_base already in
        # [0, ~1].  When normalize_globals=False the raw v2.7/v2.11 values are
        # used (for reproducing / evaluating legacy checkpoints).
        _rain_s = RAIN_REF if self._normalize_globals else 1.0
        _etc_s  = ETC_REF  if self._normalize_globals else 1.0
        _rad_s  = RAD_REF  if self._normalize_globals else 1.0

        scalar_block = np.array([
            day_frac,
            budget_frac,
            budget_total_norm,
            burn_rate,
            float(self._climate['rainfall'][d]) / _rain_s,
            float(p.Kc_ET[d]) / _etc_s,
            float(p.h2[d]),
            float(p.h7[d]),
            float(p.g_base[d]),
        ], dtype=np.float32)

        # ── forecast block (v2.12: same normalisation as the scalar block) ──
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
            _fc_slice(self._climate['rainfall'],  d, FORECAST_H) / _rain_s,
            _fc_slice(p.Kc_ET,                    d, FORECAST_H) / _etc_s,
            _fc_slice(self._climate['radiation'], d, FORECAST_H) / _rad_s,
            _fc_slice(p.h2,                       d, FORECAST_H),
            _fc_slice(p.h7,                       d, FORECAST_H),
            _fc_slice(p.g_base,                   d, FORECAST_H),
        ]).astype(np.float32)

        obs = np.concatenate([agent_block, scalar_block, forecast_block])
        _expected = self.observation_space.shape[0]
        assert obs.shape == (_expected,), (
            f"obs shape {obs.shape}, expected ({_expected},)  "
            f"[use_overshoot_feature={self._use_overshoot_feature}]"
        )
        return obs
