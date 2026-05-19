# =============================================================================
# src/rl/runner.py  v2.8.1
# Inference runner for trained SAC models.
#
# Loads a trained SB3 SAC model (v2.8 default, v2.7 legacy, v2.6 legacy, or
# pre-VDN monolithic), runs it through the full ABM season, and saves
# results in the same format as the MPC runner.
#
# v2.8.1 fix (obs normalisation — critical):
#   All dynamic per-agent features now match gym_env.py _build_obs() exactly.
#   The previous version had wrong formulas for x1_norm, x5_norm, x4_norm,
#   and x3, causing the inference policy to receive a different input
#   distribution than it was trained on.
#
#   Corrected formulas (mirror gym_env.py line-for-line):
#     x1_norm  = clip((x1 - WP) / (FC - WP), 0.0, 1.5)   [was: x1 / FC]
#     x5_norm  = clip(x5 / X5_REF,            0.0, 2.0)   [was: x5/X5_REF, no clip]
#     x4_norm  = clip(x4 / X4_REF,            0.0, 1.5)   [was: x4/X4_REF, no clip]
#     x3       = clip(x3,                      0.0, 2.0)   [was: x3, no clip]
#
#   The WP offset in x1_norm is the most consequential fix: at x1 = WP
#   (80 mm) the old runner produced 0.57 while the env produced 0.0;
#   they agreed only at x1 = FC (140 mm).
#
#   reset() now stores self._wp_total and self._x1_range alongside
#   self._fc_total so _build_obs() can apply the correct formula.
#
# v2.8 changes (unchanged):
#   - Five-way critic-arch detection for checkpoint auto-loading.
#   - _build_obs() branches on self._obs_layout (v28/v27/v26).
#   - Noisy forecast support via AR(1) NoisyForecast.
# =============================================================================

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.controllers.base import Controller
from src.rl.gym_env import (
    UB_MM,
    X4_REF,
    X5_REF,
    FULL_SEASON_NEED_MM,
)
from src.rl.networks import (
    CTDESACPolicy,              # v2.8 default — dim 67
    V27CTDESACPolicy,           # v2.7 — dim 66
    WrappedVDNCTDESACPolicy,    # v2.6 VDN — dim 63
    MonolithicCTDESACPolicy,    # pre-VDN — dim 837
)


def _detect_critic_arch(model_path: Path):
    """Peek inside a saved SB3 zip and return (input_dim, key_format).

    Returns
    -------
    (int, str)
        input_dim   : first Linear layer's input dimension
                       67  -> v2.8 VDN factorised (9 feat + 57 glob + 1 act)
                       66  -> v2.7 VDN factorised (8 feat + 57 glob + 1 act)
                       63  -> v2.6 VDN factorised (5 feat + 57 glob + 1 act)
                       837 -> pre-VDN monolithic (obs 707 + actions 130)
        key_format  : 'flat'    -> critic.qf0.0.weight
                      'wrapped' -> critic.qf0.local_q_net.0.weight
    """
    with zipfile.ZipFile(str(model_path)) as zf:
        with zf.open('policy.pth') as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu',
                                    weights_only=False)
    if 'critic.qf0.0.weight' in state_dict:
        return state_dict['critic.qf0.0.weight'].shape[1], 'flat'
    if 'critic.qf0.local_q_net.0.weight' in state_dict:
        return state_dict['critic.qf0.local_q_net.0.weight'].shape[1], 'wrapped'
    raise KeyError(
        f"Cannot detect critic architecture from {model_path}. "
        f"Keys starting with 'critic.qf0': "
        f"{[k for k in state_dict if k.startswith('critic.qf0')]}"
    )


def _detect_critic_input_dim(model_path: Path) -> int:
    """Backwards-compat wrapper — returns only the input dim."""
    dim, _ = _detect_critic_arch(model_path)
    return dim


def _load_sac_model(model_path: Path, device: str = 'cpu'):
    """Load a SAC model, auto-selecting the matching policy class.

    Returns
    -------
    (SAC, str, str)
        model, arch_label, obs_layout
        obs_layout: 'v28', 'v27', or 'v26'.
    """
    dim, key_fmt = _detect_critic_arch(model_path)
    if dim == 837:
        policy_class = MonolithicCTDESACPolicy
        label        = 'monolithic (pre-VDN)'
        obs_layout   = 'v26'
    elif dim == 63 and key_fmt == 'wrapped':
        policy_class = WrappedVDNCTDESACPolicy
        label        = 'VDN factorised - v2.6 (local_q_net wrapper)'
        obs_layout   = 'v26'
    elif dim == 63 and key_fmt == 'flat':
        policy_class = WrappedVDNCTDESACPolicy
        label        = 'VDN factorised - v2.6 (flat keys, treated as legacy)'
        obs_layout   = 'v26'
    elif dim == 66 and key_fmt == 'flat':
        policy_class = V27CTDESACPolicy
        label        = 'VDN factorised - v2.7 (8 features/agent)'
        obs_layout   = 'v27'
    elif dim == 67 and key_fmt == 'flat':
        policy_class = CTDESACPolicy
        label        = 'VDN factorised - v2.8 (9 features/agent)'
        obs_layout   = 'v28'
    else:
        raise ValueError(
            f"Unrecognised critic architecture: dim={dim}, key_format={key_fmt!r}. "
            f"Expected (837,flat), (63,wrapped), (63,flat), (66,flat), or (67,flat)."
        )
    model = SAC.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": policy_class},
    )
    return model, label, obs_layout


DEFAULT_FORECAST_HORIZON = 8


class RLController(Controller):
    """Controller wrapping a trained SAC model for inference (v2.8.1).

    Builds the observation matching the checkpoint's training architecture
    using formulas that exactly mirror gym_env.py _build_obs() (v2.8.1 fix).

    Parameters
    ----------
    model_path : str or Path
    deterministic : bool
    forecast_horizon : int  (default 8)
    forecast_mode : str  ('perfect' or 'noisy')
    noise_sigma : float  (AR(1) base std; default 0.15)
    noise_rho   : float  (AR(1) persistence; default 0.6)
    noise_seed  : int or None
    verbose : bool
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
            raise ValueError(
                f"forecast_mode must be 'perfect' or 'noisy', got {forecast_mode!r}"
            )
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.forecast_horizon = forecast_horizon
        self.forecast_mode = forecast_mode
        self.noise_sigma = noise_sigma
        self.noise_rho = noise_rho
        self.noise_seed = noise_seed
        self.verbose = verbose

        self.model, _arch_label, obs_layout = _load_sac_model(
            self.model_path, device='cpu'
        )
        self._obs_layout = obs_layout    # 'v28' | 'v27' | 'v26'

        if self.verbose:
            print(f"  Loaded checkpoint: critic architecture = {_arch_label}")
            layout_desc = {
                'v28': '1227-dim, 9 features/agent',
                'v27': '1097-dim, 8 features/agent',
                'v26': '707-dim, 5 features/agent',
            }
            print(f"  Observation layout = {obs_layout} ({layout_desc[obs_layout]})")
            print(f"  Obs normalisation:  v2.8.1 (matches gym_env.py exactly)")

        self._inference_times = []
        self._noisy_forecast = None

        name = f"sac_{'det' if deterministic else 'stoch'}_{forecast_mode}"
        super().__init__(name=name)

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._inference_times = []
        self._terrain = terrain
        self._crop = crop
        self._N = terrain['N']
        self._season_days = season_days
        self._budget_total = float(budget_total)

        # Static topographic features (all versions need elev_norm; v2.7/v2.8
        # additionally need Nr_norm, Nr_internal_norm, n_upstream_norm).
        N = self._N
        self._elev_norm = terrain['gamma_flat'].astype(np.float32)
        self._Nr_norm = np.array(
            [terrain['Nr'][n] / 8.0 for n in range(N)],
            dtype=np.float32,
        )
        self._Nr_internal_norm = np.array(
            [terrain['Nr_internal'][n] / 8.0 for n in range(N)],
            dtype=np.float32,
        )
        _ups = np.zeros(N, dtype=np.int32)
        for n_src, downstream in terrain['sends_to'].items():
            for m_dst in downstream:
                _ups[m_dst] += 1
        self._n_upstream_norm = (_ups / 8.0).astype(np.float32)

        # ── soil-moisture thresholds (v2.8.1 fix) ────────────────────────────
        # Mirror gym_env.py:
        #   _FC_MM = crop['theta6'] * crop['theta5']
        #   _WP_MM = crop['theta2'] * crop['theta5']
        self._fc_total = crop['theta6'] * crop['theta5']          # FC in mm
        self._wp_total = crop['theta2'] * crop['theta5']          # WP in mm
        self._x1_range = max(self._fc_total - self._wp_total, 1e-6)  # FC - WP

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
                sigma_base=self.noise_sigma,
                rho=self.noise_rho,
                seed=self.noise_seed,
            )
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
                  f"u_mean={u.mean():.2f}mm "
                  f"[{self.forecast_mode} forecast]")

        self._u_prev = u.copy()
        return u

    def _build_obs(self, day, state, budget_remaining):
        """Construct the obs vector matching the checkpoint version.

        Dynamic feature formulas — identical to gym_env.py _build_obs():
          x1_norm  = clip((x1 - WP) / (FC - WP), 0.0, 1.5)
          x5_norm  = clip(x5 / X5_REF,            0.0, 2.0)
          x4_norm  = clip(x4 / X4_REF,            0.0, 1.5)
          x3       = clip(x3,                      0.0, 2.0)

        Per-agent block layout:
          v2.8 (9 feat, 1170 total): x1_norm x5_norm x4_norm x3
                                     elev Nr Nr_int n_up x1_overshoot
          v2.7 (8 feat, 1040 total): x1_norm x5_norm x4_norm x3
                                     elev Nr Nr_int n_up
          v2.6 (5 feat,  650 total): x1_norm x5_norm x4_norm x3 elev

        Scalar block (9): day_frac budget_frac budget_total_norm burn_rate
                          rain_today ETc_today h2 h7 g_base
        Forecast block (48): rain[0:8] ETc[0:8] rad[0:8] h2[0:8] h7[0:8] g[0:8]
        """
        fc       = self._fc_total
        wp       = self._wp_total
        x1_range = self._x1_range    # FC - WP, pre-computed in reset()

        # ── dynamic per-agent features (v2.8.1: all match gym_env.py) ───────
        x1_norm = np.clip(
            (state['x1'] - wp) / x1_range,
            0.0, 1.5,
        ).astype(np.float32)

        x5_norm = np.clip(
            state['x5'] / X5_REF,
            0.0, 2.0,
        ).astype(np.float32)

        x4_norm = np.clip(
            state['x4'] / X4_REF,
            0.0, 1.5,
        ).astype(np.float32)

        x3 = np.clip(state['x3'], 0.0, 2.0).astype(np.float32)

        # ── per-agent block — branch on checkpoint version ───────────────────
        if self._obs_layout == 'v28':
            x1_overshoot_norm = np.clip(
                np.maximum(state['x1'] - fc, 0.0) / max(fc, 1e-6),
                0.0, 1.0,
            ).astype(np.float32)
            agent_block = np.stack([
                x1_norm,
                x5_norm,
                x4_norm,
                x3,
                self._elev_norm,
                self._Nr_norm,
                self._Nr_internal_norm,
                self._n_upstream_norm,
                x1_overshoot_norm,
            ], axis=1).flatten().astype(np.float32)   # (1170,)
        elif self._obs_layout == 'v27':
            agent_block = np.stack([
                x1_norm,
                x5_norm,
                x4_norm,
                x3,
                self._elev_norm,
                self._Nr_norm,
                self._Nr_internal_norm,
                self._n_upstream_norm,
            ], axis=1).flatten().astype(np.float32)   # (1040,)
        else:  # 'v26'
            agent_block = np.stack([
                x1_norm,
                x5_norm,
                x4_norm,
                x3,
                self._elev_norm,
            ], axis=1).flatten().astype(np.float32)   # (650,)

        # ── scalar block ─────────────────────────────────────────────────────
        day_frac          = day / self._season_days
        budget_frac       = budget_remaining / max(self._budget_total, 1e-6)
        budget_total_norm = self._budget_total / FULL_SEASON_NEED_MM

        water_spent  = self._budget_total - float(budget_remaining)
        daily_budget = self._budget_total / self._season_days
        burn_rate = (
            (water_spent / max(day, 1)) / max(daily_budget, 1e-6)
            if day > 0 else 0.0
        )

        d = min(day, self._season_days - 1)
        rain_today   = float(self._climate['rainfall'][d])
        ETc_today    = float(self._precomputed.Kc_ET[d])
        h2_today     = float(self._precomputed.h2[d])
        h7_today     = float(self._precomputed.h7[d])
        g_base_today = float(self._precomputed.g_base[d])

        scalars = np.array([
            day_frac, budget_frac, budget_total_norm, burn_rate,
            rain_today, ETc_today, h2_today, h7_today, g_base_today,
        ], dtype=np.float32)

        # ── forecast block ───────────────────────────────────────────────────
        H   = self.forecast_horizon
        end = min(d + H, self._season_days)

        def _pad(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < H:
                pad_val = arr[-1] if len(arr) > 0 else 0.0
                return np.concatenate([
                    arr,
                    np.full(H - len(arr), pad_val, dtype=np.float32),
                ])
            return arr

        if self._noisy_forecast is not None:
            fc_dict = self._noisy_forecast(
                day, self._climate, self._precomputed, H
            )
            rain_fc = _pad(fc_dict['rainfall'])
            ETc_fc  = _pad(fc_dict['ETc'])
            rad_fc  = _pad(fc_dict['radiation'])
            h2_fc   = _pad(self._precomputed.h2[d:end])
            h7_fc   = _pad(self._precomputed.h7[d:end])
            g_fc    = _pad(self._precomputed.g_base[d:end])
        else:
            rain_fc = _pad(self._climate['rainfall'][d:end])
            ETc_fc  = _pad(self._precomputed.Kc_ET[d:end])
            rad_fc  = _pad(self._climate['radiation'][d:end])
            h2_fc   = _pad(self._precomputed.h2[d:end])
            h7_fc   = _pad(self._precomputed.h7[d:end])
            g_fc    = _pad(self._precomputed.g_base[d:end])

        return np.concatenate([
            agent_block,
            scalars,
            rain_fc, ETc_fc, rad_fc, h2_fc, h7_fc, g_fc,
        ]).astype(np.float32)

    @property
    def solve_times(self):
        return list(self._inference_times)
