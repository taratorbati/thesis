# src/rl/gym_env.py  v2.5.0
# ─────────────────────────────────────────────────────────────────────────────
# Changes from v2.4.x  (all changes marked with  # [v2.5])
#
#  1. c_term = 0   — Terminal bonus REMOVED.
#       The per-step biomass increment reward integrates to the same total
#       biomass signal over 93 days.  The massive sparse terminal reward
#       (previously 300× a normal step) was the primary driver of Bellman
#       bootstrapping error accumulation and Phase-3 Q-value divergence.
#       Ref: Gemini analysis (May 2026), consistent with Claude session record.
#
#  2. ALPHA5_RL = 0.0  — ΔU actuator-smoothing penalty DISABLED in RL.
#       α₅ penalises action changes, which directly suppresses weather-
#       responsive behaviour.  The MPC retains α₅ = 0.005 (it has access to
#       the full deterministic future so the penalty is safe there).  RL
#       agents are skittish under immediate dense penalties; removing α₅ lets
#       the actor learn to turn water on/off in response to rain events without
#       being punished for the transition.
#       Ref: Gemini analysis Part 3, Point 4.
#
#  All other reward terms (α₁, α₂, α₃, α₆, λ_budget) are unchanged.
#  All other environment logic is unchanged from v2.4.3.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from abm import CropSoilABM
from climate_data import TRAINING_YEARS, SCENARIO_YEARS
from src.precompute import get_precomputed
from soil_data import (
    FIELD_CAPACITY_MM, WILTING_POINT_MM, STRESS_THRESHOLD_MM,
    U_MAX, HARVEST_INDEX,
)

# ── reward weight constants (matching MPC α* operating point) ────────────────
ALPHA1 = 1.0       # biomass reward
ALPHA2 = 0.016     # water cost (Iranian domestic-base tariff)
ALPHA3 = 0.1       # drought stress regulariser
ALPHA5_RL = 0.0    # [v2.5] actuator smoothing DISABLED for RL (was 0.005)
ALPHA6 = 8.0       # FC-overshoot soft penalty

# ── terminal bonus ────────────────────────────────────────────────────────────
C_TERM = 0.0       # [v2.5] terminal bonus REMOVED (was 5.0)

# ── budget over-burn soft penalty ─────────────────────────────────────────────
LAMBDA_BUDGET = 0.1

# ── observation normalisation references ─────────────────────────────────────
FC = FIELD_CAPACITY_MM          # 140 mm
WP = WILTING_POINT_MM           # 68 mm
ST = STRESS_THRESHOLD_MM        # 126 mm
X1_RANGE = FC - WP              # 72 mm — normalises x1 to [0, 1]
X4_REF   = 600.0                # g/m² reference biomass
X5_REF   = 50.0                 # mm — surface ponding normalisation

FULL_SEASON_NEED_MM = 484.0     # 100% budget reference

N_AGENTS   = 130
OBS_DIM    = 707   # 650 per-agent + 9 scalars + 48 forecast
FORECAST_H = 8     # days in forecast window


class IrrigationEnv(gym.Env):
    """Gymnasium wrapper around the 130-agent crop-soil ABM.

    Observation (707-dim):
      • Per-agent block  (650 = 5 × 130): x1_norm, x5_norm, x4_norm, x3, gamma
      • Scalar block     (9): day_frac, budget_frac, budget_total_norm,
                               burn_rate, rain_today, ETc_today, h2_today,
                               h7_today, g_base_today
      • Forecast block   (48 = 6 vars × 8 days): rain[H], ETc[H], rad[H],
                               h2[H], h7[H], g_base[H]

    Action (130-dim, Box [0,1]):  scaled to [0, U_MAX] mm/day in step().

    Reward: approximate negation of MPC path cost at α* (see module header).
    """

    metadata = {"render_modes": []}

    def __init__(self, randomize: bool = True):
        super().__init__()
        self.randomize = randomize

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_AGENTS,), dtype=np.float32
        )

        self._abm: CropSoilABM | None = None
        self._precomp: dict | None = None
        self._year: int | None = None
        self._budget_mm: float | None = None
        self._water_used: float = 0.0
        self._day: int = 0
        self._prev_actions: np.ndarray = np.zeros(N_AGENTS, dtype=np.float32)
        self._prev_x4_mean: float = 0.0

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize:
            self._year = int(self.np_random.choice(list(TRAINING_YEARS)))
            budget_frac = float(self.np_random.uniform(0.70, 1.00))
        else:
            self._year = SCENARIO_YEARS["dry"]
            budget_frac = 1.0

        self._budget_mm = FULL_SEASON_NEED_MM * budget_frac
        self._water_used = 0.0
        self._day = 0
        self._prev_actions = np.zeros(N_AGENTS, dtype=np.float32)

        self._abm = CropSoilABM(year=self._year)
        self._precomp = get_precomputed(self._year, "rice")   # per-year fix (v2.4.3)

        self._prev_x4_mean = float(np.mean(self._abm.get_state()[:, 3]))

        obs = self._build_obs()
        return obs, {}

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        # 1. clip & scale action
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        irr_mm = action * U_MAX   # [0, 12] mm/day per agent

        # 2. hard budget clipping at runner layer
        remaining = max(self._budget_mm - self._water_used, 0.0)
        irr_mm = np.minimum(irr_mm, remaining / N_AGENTS)

        # 3. advance ABM
        self._abm.step(irrigation=irr_mm)
        water_this_step = float(np.mean(irr_mm))
        self._water_used += water_this_step * N_AGENTS

        # 4. get post-step state
        state = self._abm.get_state()   # (130, 5): x1 x2 x3 x4 x5
        x1 = state[:, 0]
        x4 = state[:, 3]
        x4_mean = float(np.mean(x4))

        # 5. compute reward
        reward = self._compute_reward(
            x1=x1,
            x4_mean=x4_mean,
            irr_mm=irr_mm,
            action=action,
        )

        self._day += 1
        self._prev_actions = action.copy()
        self._prev_x4_mean = x4_mean

        # 6. termination logic
        season_done = (self._day >= 93)
        budget_done = (self._water_used >= self._budget_mm - 1e-6)
        terminated = budget_done
        truncated  = season_done and not budget_done

        # 7. terminal bonus — REMOVED in v2.5 (C_TERM = 0)
        if (terminated or truncated) and C_TERM > 0:
            reward += C_TERM * ALPHA1 * x4_mean / X4_REF

        obs = self._build_obs()
        info = {
            "day": self._day,
            "water_used_mm": self._water_used,
            "budget_mm": self._budget_mm,
            "x4_mean": x4_mean,
            "yield_kg_ha": x4_mean * HARVEST_INDEX * 10.0,
        }
        return obs, float(reward), terminated, truncated, info

    # ── reward ────────────────────────────────────────────────────────────────
    def _compute_reward(
        self,
        x1: np.ndarray,
        x4_mean: float,
        irr_mm: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Five-term RL reward (approximate negation of MPC path cost at α*).

        Terms:
          r1  Biomass increment reward         (+)
          r2  Water cost                        (-)
          r3  Drought stress penalty            (-)
          r5  Actuator smoothing  [DISABLED]    (0 in v2.5)
          r6  FC-overshoot penalty              (-)
          rb  Budget burn-rate penalty          (-)
        """
        # r1: biomass increment
        delta_x4 = x4_mean - self._prev_x4_mean
        r1 = ALPHA1 * delta_x4 / X4_REF

        # r2: water cost (mean irrigation across agents, normalised)
        r2 = -ALPHA2 * float(np.mean(irr_mm)) / U_MAX

        # r3: drought stress (mean deficit below ST)
        drought = np.maximum(ST - x1, 0.0)
        r3 = -ALPHA3 * float(np.mean(drought)) / (ST - WP)

        # r5: actuator smoothing — DISABLED in v2.5 (ALPHA5_RL = 0.0)
        if ALPHA5_RL > 0.0:
            delta_u = action - self._prev_actions
            r5 = -ALPHA5_RL * float(np.mean(delta_u ** 2)) / (1.0 ** 2)
        else:
            r5 = 0.0

        # r6: FC-overshoot soft penalty
        overshoot = np.maximum(x1 - FC, 0.0)
        r6 = -ALPHA6 * float(np.mean(overshoot ** 2)) / (FC ** 2)

        # rb: budget burn-rate penalty
        if self._day > 0 and self._budget_mm > 0:
            burn_rate = (self._water_used / N_AGENTS) / (
                self._day * FULL_SEASON_NEED_MM / 93.0
            )
            rb = -LAMBDA_BUDGET * max(burn_rate - 1.0, 0.0) ** 2
        else:
            rb = 0.0

        return r1 + r2 + r3 + r5 + r6 + rb

    # ── observation ───────────────────────────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        """Build the 707-dim observation vector."""
        state = self._abm.get_state()   # (130, 5)
        p = self._precomp
        d = self._day

        # ── per-agent block (650) ──────────────────────────────────────────
        x1_norm = np.clip((state[:, 0] - WP) / X1_RANGE, 0.0, 1.5)
        x5_norm = np.clip(state[:, 4] / X5_REF, 0.0, 2.0)
        x4_norm = np.clip(state[:, 3] / X4_REF, 0.0, 1.5)
        x3      = np.clip(state[:, 2], 0.0, 2.0)
        # gamma: normalised position in maturation cycle
        gamma   = np.clip(state[:, 1] / 1250.0, 0.0, 1.0)
        per_agent = np.stack([x1_norm, x5_norm, x4_norm, x3, gamma], axis=1)  # (130,5)
        agent_block = per_agent.flatten().astype(np.float32)   # 650

        # ── scalar block (9) ───────────────────────────────────────────────
        day_frac          = d / 93.0
        budget_frac       = max(self._budget_mm - self._water_used, 0.0) / self._budget_mm
        budget_total_norm = self._budget_mm / FULL_SEASON_NEED_MM
        burn_rate         = (
            (self._water_used / N_AGENTS) / (d * FULL_SEASON_NEED_MM / 93.0)
            if d > 0 else 0.0
        )
        rain_today  = float(p["rain"][d])   if d < len(p["rain"])   else 0.0
        ETc_today   = float(p["ETc"][d])    if d < len(p["ETc"])    else 0.0
        h2_today    = float(p["h2"][d])     if d < len(p["h2"])     else 1.0
        h7_today    = float(p["h7"][d])     if d < len(p["h7"])     else 1.0
        g_base_today= float(p["g_base"][d]) if d < len(p["g_base"]) else 0.0

        scalar_block = np.array([
            day_frac, budget_frac, budget_total_norm, burn_rate,
            rain_today, ETc_today, h2_today, h7_today, g_base_today,
        ], dtype=np.float32)   # 9

        # ── forecast block (48 = 6 × 8) ───────────────────────────────────
        forecast_vars = ["rain", "ETc", "radiation", "h2", "h7", "g_base"]
        rows = []
        for var in forecast_vars:
            arr = p.get(var, np.zeros(93))
            row = []
            for h in range(FORECAST_H):
                idx = d + h
                row.append(float(arr[idx]) if idx < len(arr) else 0.0)
            rows.append(row)
        forecast_block = np.array(rows, dtype=np.float32).flatten()   # 48

        obs = np.concatenate([agent_block, scalar_block, forecast_block])
        assert obs.shape == (OBS_DIM,), f"obs shape mismatch: {obs.shape}"
        return obs
