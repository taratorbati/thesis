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
    RAIN_REF,
    ETC_REF,
    RAD_REF,
)
from src.rl.networks import (
    CTDESACPolicy,              # v2.8 default — dim 67
    V212CTDESACPolicy,          # v2.12 — LayerNorm critic + LeakyReLU actor + normalised obs
    V211CTDESACPolicy,          # v2.11 — LayerNorm critic, dim 66 + LN params
    V27CTDESACPolicy,           # v2.7 — dim 66
    WrappedVDNCTDESACPolicy,    # v2.6 VDN — dim 63
    MonolithicCTDESACPolicy,    # pre-VDN — dim 837
)


def _detect_critic_arch(model_path: Path):
    """Peek inside a saved SB3 zip and return (input_dim, key_format, has_layernorm).

    Returns
    -------
    (int, str, bool)
        input_dim   : first Linear layer's input dimension
                       67  -> v2.8 VDN factorised (9 feat + 57 glob + 1 act)
                       66  -> v2.7/v2.11 VDN factorised (8 feat + 57 glob + 1 act)
                       63  -> v2.6 VDN factorised (5 feat + 57 glob + 1 act)
                       837 -> pre-VDN monolithic (obs 707 + actions 130)
        key_format  : 'flat'    -> critic.qf0.0.weight
                      'wrapped' -> critic.qf0.local_q_net.0.weight
        has_layernorm : True if 'critic.qf0.1.weight' is present and 1-D
                        (signal that LayerNorm was inserted after the first
                        hidden Linear; v2.11 critic).  v2.7 has no
                        'qf0.1.weight' (index 1 = ReLU, no params).
    """
    with zipfile.ZipFile(str(model_path)) as zf:
        with zf.open('policy.pth') as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu',
                                    weights_only=False)
    # LayerNorm-after-Linear signature: 'critic.qf0.1.weight' with ndim==1.
    has_layernorm = (
        'critic.qf0.1.weight' in state_dict
        and state_dict['critic.qf0.1.weight'].ndim == 1
    )
    # v2.12 signature: the actor registers an 'actor.obs_norm_marker' buffer,
    # signalling that the checkpoint was trained on the normalised global/
    # forecast observation block (and a LeakyReLU actor).  Absent in v2.7/v2.11.
    has_obs_norm = 'actor.obs_norm_marker' in state_dict
    if 'critic.qf0.0.weight' in state_dict:
        return (state_dict['critic.qf0.0.weight'].shape[1], 'flat',
                has_layernorm, has_obs_norm)
    if 'critic.qf0.local_q_net.0.weight' in state_dict:
        return (state_dict['critic.qf0.local_q_net.0.weight'].shape[1], 'wrapped',
                has_layernorm, has_obs_norm)
    raise KeyError(
        f"Cannot detect critic architecture from {model_path}. "
        f"Keys starting with 'critic.qf0': "
        f"{[k for k in state_dict if k.startswith('critic.qf0')]}"
    )


def _detect_critic_input_dim(model_path: Path) -> int:
    """Backwards-compat wrapper — returns only the input dim."""
    dim, _, _, _ = _detect_critic_arch(model_path)
    return dim


def _load_sac_model(model_path: Path, device: str = 'cpu'):
    """Load a SAC model, auto-selecting the matching policy class.

    Returns
    -------
    (SAC, str, str, bool)
        model, arch_label, obs_layout, normalize_globals
        obs_layout: 'v28', 'v27', or 'v26'.
        normalize_globals: True for v2.12 checkpoints (the global/forecast
        block must be normalised at eval time to match training).
    """
    dim, key_fmt, has_ln, has_obs_norm = _detect_critic_arch(model_path)
    normalize_globals = False
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
    elif dim == 66 and key_fmt == 'flat' and has_ln and has_obs_norm:
        # v2.12: v2.11 LayerNorm critic + LeakyReLU actor, trained on the
        # NORMALISED global/forecast observation block.  Eval must apply the
        # same normalisation (normalize_globals=True).
        policy_class = V212CTDESACPolicy
        label        = ('VDN factorised - v2.12 (8 features/agent, LayerNorm '
                        'critic, LeakyReLU actor, normalised globals)')
        obs_layout   = 'v27'
        normalize_globals = True
    elif dim == 66 and key_fmt == 'flat' and has_ln:
        # v2.11: same obs layout as v2.7 (8 features/agent, 1097-dim) but
        # the critic has LayerNorm after each hidden Linear layer.
        policy_class = V211CTDESACPolicy
        label        = 'VDN factorised - v2.11 (8 features/agent, LayerNorm critic)'
        obs_layout   = 'v27'
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
            f"Unrecognised critic architecture: dim={dim}, key_format={key_fmt!r}, "
            f"has_layernorm={has_ln}, has_obs_norm={has_obs_norm}. "
            f"Expected (837,flat), (63,wrapped), (63,flat), "
            f"(66,flat,LN,obsnorm)=v2.12, (66,flat,LN)=v2.11, "
            f"(66,flat)=v2.7, or (67,flat)=v2.8."
        )
    model = SAC.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": policy_class},
    )
    return model, label, obs_layout, normalize_globals


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

        # v2.10 refactor: model loading is delegated to _load_model() so
        # subclasses (TQCRLController) can override only the loader without
        # duplicating the rest of __init__.  Behaviour for SAC checkpoints
        # is unchanged.  v2.12: _load_model may also return a normalize_globals
        # flag (4-tuple); older overrides returning a 3-tuple still work.
        _loaded = self._load_model()
        if len(_loaded) == 4:
            self.model, _arch_label, obs_layout, normalize_globals = _loaded
        else:
            self.model, _arch_label, obs_layout = _loaded
            normalize_globals = False
        self._obs_layout = obs_layout    # 'v28' | 'v27' | 'v26'
        self._normalize_globals = bool(normalize_globals)

        if self.verbose:
            print(f"  Loaded checkpoint: critic architecture = {_arch_label}")
            layout_desc = {
                'v28': '1227-dim, 9 features/agent',
                'v27': '1097-dim, 8 features/agent',
                'v26': '707-dim, 5 features/agent',
            }
            print(f"  Observation layout = {obs_layout} ({layout_desc[obs_layout]})")
            _normdesc = ('v2.12 (per-agent + normalised global/forecast block)'
                         if self._normalize_globals
                         else 'v2.8.1 (per-agent only; raw global/forecast block)')
            print(f"  Obs normalisation:  {_normdesc}")

        self._inference_times = []
        self._noisy_forecast = None

        name = f"sac_{'det' if deterministic else 'stoch'}_{forecast_mode}"
        super().__init__(name=name)

    def _load_model(self):
        """Load the SB3 model checkpoint.

        Default implementation loads a SAC checkpoint via _load_sac_model.
        Subclasses (e.g. TQCRLController) override this to load other
        algorithm classes while inheriting the rest of RLController's
        observation-building and inference logic.

        Returns
        -------
        (model, arch_label, obs_layout)
            model      : the loaded SB3 algorithm instance (must expose
                         model.predict(obs, deterministic=...))
            arch_label : human-readable architecture string for logging
            obs_layout : one of 'v28', 'v27', 'v26'
        """
        return _load_sac_model(self.model_path, device='cpu')

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
        # v2.12: divide raw weather by physical references so the actor sees the
        # same normalised scale it was trained on.  Legacy checkpoints
        # (normalize_globals=False) keep the raw magnitudes.
        _rain_s = RAIN_REF if self._normalize_globals else 1.0
        _etc_s  = ETC_REF  if self._normalize_globals else 1.0
        _rad_s  = RAD_REF  if self._normalize_globals else 1.0

        rain_today   = float(self._climate['rainfall'][d]) / _rain_s
        ETc_today    = float(self._precomputed.Kc_ET[d]) / _etc_s
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
            rain_fc = _pad(fc_dict['rainfall']) / _rain_s
            ETc_fc  = _pad(fc_dict['ETc']) / _etc_s
            rad_fc  = _pad(fc_dict['radiation']) / _rad_s
            h2_fc   = _pad(self._precomputed.h2[d:end])
            h7_fc   = _pad(self._precomputed.h7[d:end])
            g_fc    = _pad(self._precomputed.g_base[d:end])
        else:
            rain_fc = _pad(self._climate['rainfall'][d:end]) / _rain_s
            ETc_fc  = _pad(self._precomputed.Kc_ET[d:end]) / _etc_s
            rad_fc  = _pad(self._climate['radiation'][d:end]) / _rad_s
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
