# =============================================================================
# src/rl/runner.py
# Inference runner for trained SAC and TD3 irrigation policies.
#
# Loads a trained SB3 checkpoint, auto-detecting which of the two supported
# architectures it is and which observation layout it expects, then drives it
# through a full ABM season. The observation is rebuilt here to match training
# (gym_env.py) exactly.
#
# Auto-detection (from the saved policy.pth state-dict):
#   * SAC vs TD3      : SAC actors have a stochastic log_std head; the
#                       deterministic TD3 actor does not.
#   * weather layout  : the critic's first-layer input width is 8 + G + 1, so
#                       the global block width G (52 deduped, 57 legacy) — and
#                       hence whether today's weather is repeated — is read off
#                       directly. Both networks infer their dims the same way.
# =============================================================================

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC, TD3

from src.controllers.base import Controller
from src.rl.gym_env import (
    UB_MM, X4_REF, X5_REF, FULL_SEASON_NEED_MM,
    RAIN_REF, ETC_REF, RAD_REF, N_AGENT_FEATURES, global_dim,
)
from src.rl.networks import SacVdnPolicy
from src.rl.networks_td3 import Td3VdnPolicy
from src.rl.nstep_buffer import NStepReplayBuffer

DEFAULT_FORECAST_HORIZON = 8

_GLOBAL_DIM_DEDUPED = global_dim(dedupe_today_weather=True)    # 52


def _detect_arch(model_path: Path):
    """Inspect a saved SB3 zip; return (is_td3, dedupe_today_weather, critic_in_dim)."""
    with zipfile.ZipFile(str(model_path)) as zf:
        with zf.open('policy.pth') as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu',
                                    weights_only=False)
    if 'critic.qf0.0.weight' not in state_dict:
        raise KeyError(
            f"Cannot detect critic architecture from {model_path}. "
            f"critic.qf0.* keys: {[k for k in state_dict if k.startswith('critic.qf0')]}")
    critic_in_dim = int(state_dict['critic.qf0.0.weight'].shape[1])
    global_dim_ckpt = critic_in_dim - N_AGENT_FEATURES - 1
    dedupe = (global_dim_ckpt == _GLOBAL_DIM_DEDUPED)
    is_td3 = 'actor.log_std.weight' not in state_dict
    return is_td3, dedupe, critic_in_dim


def load_policy(model_path: Path, device: str = 'cpu'):
    """Load a SAC or TD3 checkpoint with the matching policy class.

    Returns (model, label, dedupe_today_weather).
    """
    is_td3, dedupe, _ = _detect_arch(model_path)
    layout = "deduped (52-dim global)" if dedupe else "legacy (57-dim global)"
    # Inference-only load: neutralise the (training-only) LR schedule, and pin
    # the replay-buffer class so old checkpoints don't import a renamed module.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "replay_buffer_class": NStepReplayBuffer,
    }
    if is_td3:
        custom_objects["policy_class"] = Td3VdnPolicy
        model = TD3.load(str(model_path), device=device, custom_objects=custom_objects)
        return model, f"TD3 VDN, {layout}", dedupe
    custom_objects["policy_class"] = SacVdnPolicy
    model = SAC.load(str(model_path), device=device, custom_objects=custom_objects)
    return model, f"SAC VDN, {layout}", dedupe


class RLController(Controller):
    """Controller wrapping a trained SAC/TD3 policy for full-season inference.

    Rebuilds the observation to match gym_env.py for the detected layout, then
    calls ``model.predict``. Supports perfect or AR(1)-noisy forecasts.
    """

    def __init__(
        self,
        model_path,
        deterministic=True,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        forecast_mode='perfect',
        noise_sigma=0.15,
        noise_rho=0.6,
        noise_seed=None,
        verbose=True,
    ):
        if forecast_mode not in ('perfect', 'noisy'):
            raise ValueError(f"forecast_mode must be 'perfect' or 'noisy', got {forecast_mode!r}")
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.forecast_horizon = forecast_horizon
        self.forecast_mode = forecast_mode
        self.noise_sigma = noise_sigma
        self.noise_rho = noise_rho
        self.noise_seed = noise_seed
        self.verbose = verbose

        self.model, arch_label, self._dedupe = self._load_model()
        if self.verbose:
            print(f"  Loaded checkpoint: {arch_label}")

        self._inference_times = []
        self._noisy_forecast = None
        super().__init__(name=f"rl_{'det' if deterministic else 'stoch'}_{forecast_mode}")

    def _load_model(self):
        """Load the checkpoint. Returns (model, label, dedupe). Subclasses may override."""
        return load_policy(self.model_path, device='cpu')

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._inference_times = []
        self._terrain = terrain
        self._crop = crop
        self._N = terrain['N']
        self._season_days = season_days
        self._budget_total = float(budget_total)

        N = self._N
        self._elev_norm = terrain['gamma_flat'].astype(np.float32)
        self._Nr_norm = np.array([terrain['Nr'][n] / 8.0 for n in range(N)], dtype=np.float32)
        self._Nr_internal_norm = np.array(
            [terrain['Nr_internal'][n] / 8.0 for n in range(N)], dtype=np.float32)
        _ups = np.zeros(N, dtype=np.int32)
        for n_src, downstream in terrain['sends_to'].items():
            for m_dst in downstream:
                _ups[m_dst] += 1
        self._n_upstream_norm = (_ups / 8.0).astype(np.float32)

        # Soil-moisture thresholds (mirror gym_env.py).
        self._fc_total = crop['theta6'] * crop['theta5']
        self._wp_total = crop['theta2'] * crop['theta5']
        self._x1_range = max(self._fc_total - self._wp_total, 1e-6)

        self._u_prev = np.zeros(self._N)

        from src.precompute import get_precomputed
        from climate_data import load_cleaned_data, extract_scenario_by_name
        scenario = scenario_name or 'dry'
        self._precomputed = get_precomputed(scenario, crop['name'].lower())
        df = load_cleaned_data()
        self._climate = extract_scenario_by_name(df, scenario, crop)

        if self.forecast_mode == 'noisy':
            from src.forecast import NoisyForecast
            self._noisy_forecast = NoisyForecast(
                sigma_base=self.noise_sigma, rho=self.noise_rho, seed=self.noise_seed)
        else:
            self._noisy_forecast = None

    def set_climate(self, climate):
        self._climate = climate

    def step(self, day, state, climate_today, budget_remaining, forecast=None):
        t0 = time.time()
        obs = self._build_obs(day, state, budget_remaining)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        u = np.asarray(action, dtype=float).clip(0, 1) * UB_MM
        self._inference_times.append((time.time() - t0) * 1000)
        if self.verbose and (day % 10 == 0):
            print(f"    day {day:3d}: inference {self._inference_times[-1]:.1f}ms "
                  f"u_mean={u.mean():.2f}mm [{self.forecast_mode} forecast]")
        self._u_prev = u.copy()
        return u

    def _build_obs(self, day, state, budget_remaining):
        """Construct the observation, matching gym_env.py for the detected layout."""
        x1_norm = np.clip((state['x1'] - self._wp_total) / self._x1_range, 0.0, 1.5).astype(np.float32)
        x5_norm = np.clip(state['x5'] / X5_REF, 0.0, 2.0).astype(np.float32)
        x4_norm = np.clip(state['x4'] / X4_REF, 0.0, 1.5).astype(np.float32)
        x3 = np.clip(state['x3'], 0.0, 2.0).astype(np.float32)

        agent_block = np.stack([
            x1_norm, x5_norm, x4_norm, x3,
            self._elev_norm, self._Nr_norm, self._Nr_internal_norm, self._n_upstream_norm,
        ], axis=1).flatten().astype(np.float32)

        day_frac = day / self._season_days
        budget_frac = budget_remaining / max(self._budget_total, 1e-6)
        budget_total_norm = self._budget_total / FULL_SEASON_NEED_MM
        water_spent = self._budget_total - float(budget_remaining)
        daily_budget = self._budget_total / self._season_days
        burn_rate = (water_spent / max(day, 1)) / max(daily_budget, 1e-6) if day > 0 else 0.0

        d = min(day, self._season_days - 1)
        scalars = [day_frac, budget_frac, budget_total_norm, burn_rate]
        if not self._dedupe:
            scalars += [
                float(self._climate['rainfall'][d]) / RAIN_REF,
                float(self._precomputed.Kc_ET[d]) / ETC_REF,
                float(self._precomputed.h2[d]),
                float(self._precomputed.h7[d]),
                float(self._precomputed.g_base[d]),
            ]
        scalar_block = np.array(scalars, dtype=np.float32)

        H = self.forecast_horizon
        end = min(d + H, self._season_days)

        def _pad(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < H:
                pad_val = arr[-1] if len(arr) > 0 else 0.0
                return np.concatenate([arr, np.full(H - len(arr), pad_val, dtype=np.float32)])
            return arr

        if self._noisy_forecast is not None:
            fc = self._noisy_forecast(day, self._climate, self._precomputed, H)
            rain_fc = _pad(fc['rainfall']) / RAIN_REF
            ETc_fc = _pad(fc['ETc']) / ETC_REF
            rad_fc = _pad(fc['radiation']) / RAD_REF
        else:
            rain_fc = _pad(self._climate['rainfall'][d:end]) / RAIN_REF
            ETc_fc = _pad(self._precomputed.Kc_ET[d:end]) / ETC_REF
            rad_fc = _pad(self._climate['radiation'][d:end]) / RAD_REF
        h2_fc = _pad(self._precomputed.h2[d:end])
        h7_fc = _pad(self._precomputed.h7[d:end])
        g_fc = _pad(self._precomputed.g_base[d:end])

        return np.concatenate([
            agent_block, scalar_block, rain_fc, ETc_fc, rad_fc, h2_fc, h7_fc, g_fc,
        ]).astype(np.float32)

    @property
    def solve_times(self):
        return list(self._inference_times)
