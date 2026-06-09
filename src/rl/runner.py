# =============================================================================
# src/rl/runner.py  v2.8.1 + TD3 support (v2.19-v2.22)
# Inference runner for trained SAC and TD3 models.
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
    RAIN_REF_V216,
    ETC_REF,
    RAD_REF,
)
from src.rl.networks import (
    CTDESACPolicy,              # v2.8 default — dim 67
    V216CTDESACPolicy,          # v2.16 — v2.15 architecture + RAIN_REF=30 + capped auto-α
    V215CTDESACPolicy,          # v2.15 — v2.14 architecture, linear r6
    V214CTDESACPolicy,          # v2.14 — v2.13 architecture, α=0.01
    V213CTDESACPolicy,          # v2.13 — v2.12 + actor-only input re-centering
    V212CTDESACPolicy,          # v2.12 — LayerNorm critic + LeakyReLU actor + normalised obs
    V211CTDESACPolicy,          # v2.11 — LayerNorm critic, dim 66 + LN params
    V27CTDESACPolicy,           # v2.7 — dim 66
    WrappedVDNCTDESACPolicy,    # v2.6 VDN — dim 63
    MonolithicCTDESACPolicy,    # pre-VDN — dim 837
)


def _detect_critic_arch(model_path: Path):
    """Peek inside a saved SB3 zip and return architecture signature tuple.

    Returns
    -------
    (int, str, bool, float)
        input_dim     : first Linear layer's input dimension (67=v2.8/v2.22, 66=v2.7/11/12/13/19,
                        63=v2.6, 837=pre-VDN monolithic)
        key_format    : 'flat' | 'wrapped'
        has_layernorm : True if critic.qf0.1.weight exists and is 1-D (v2.11+ marker)
        obs_marker    : float value of actor.obs_norm_marker buffer, or 0.0 if absent.
                          0.0  -> v2.7 / v2.11 (raw global obs, ReLU actor)
                          2.12 -> v2.12 (normalised globals + LeakyReLU actor)
                          2.13 -> v2.13 (v2.12 + actor-only input re-centering)
    """
    with zipfile.ZipFile(str(model_path)) as zf:
        with zf.open('policy.pth') as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu',
                                    weights_only=False)
    has_layernorm = (
        'critic.qf0.1.weight' in state_dict
        and state_dict['critic.qf0.1.weight'].ndim == 1
    )
    # Marker buffer: presence + value distinguishes v2.12 (= 2.12) from v2.13 (= 2.13).
    if 'actor.obs_norm_marker' in state_dict:
        obs_marker = float(state_dict['actor.obs_norm_marker'].item())
    else:
        obs_marker = 0.0
    # TD3 vs SAC actor signal: SAC actors have a stochastic log_std head; the
    # deterministic TD3 actor (v2.19) does not.  Absence of actor.log_std.weight
    # is the unambiguous TD3 marker (combined with marker >= 2.185).
    has_log_std = 'actor.log_std.weight' in state_dict
    if 'critic.qf0.0.weight' in state_dict:
        return (state_dict['critic.qf0.0.weight'].shape[1], 'flat',
                has_layernorm, obs_marker, has_log_std)
    if 'critic.qf0.local_q_net.0.weight' in state_dict:
        return (state_dict['critic.qf0.local_q_net.0.weight'].shape[1], 'wrapped',
                has_layernorm, obs_marker, has_log_std)
    raise KeyError(
        f"Cannot detect critic architecture from {model_path}. "
        f"Keys starting with 'critic.qf0': "
        f"{[k for k in state_dict if k.startswith('critic.qf0')]}"
    )


def _detect_critic_input_dim(model_path: Path) -> int:
    """Backwards-compat wrapper — returns only the input dim."""
    dim, _, _, _, _ = _detect_critic_arch(model_path)
    return dim


def _load_sac_model(model_path: Path, device: str = 'cpu'):
    """Load a SAC model, auto-selecting the matching policy class.

    Returns
    -------
    (SAC, str, str, bool, float)
        model, arch_label, obs_layout, normalize_globals, rain_normaliser
        obs_layout: 'v28', 'v27', or 'v26'.
        normalize_globals: True for v2.12+ checkpoints (the global/forecast
        block must be normalised at eval time to match training).
        rain_normaliser: rainfall denominator used at training time
        (RAIN_REF=70.0 for v2.7-v2.15, RAIN_REF_V216=30.0 for v2.16).
        Used by the eval runner to apply the same rainfall scaling.
    """
    dim, key_fmt, has_ln, obs_marker, has_log_std = _detect_critic_arch(
        model_path)
    # obs_marker: 0.0=v2.7/v2.11 (raw obs), 2.12=v2.12 (normalised globals),
    # 2.13=v2.13 (normalised globals + actor-only input re-centering),
    # 2.14=v2.14 (v2.13 architecture trained with α=0.01),
    # 2.15=v2.15 (v2.14 architecture trained with linear r6),
    # 2.16=v2.16 (v2.15 architecture + RAIN_REF=30 + capped auto-α),
    # 2.19=v2.19 (TD3: deterministic actor, no log_std, same VDN LayerNorm critic).
    normalize_globals = False
    rain_normaliser = RAIN_REF   # default for v2.7-v2.15 checkpoints

    # ── v2.19 TD3 branch (MUST be checked before the SAC 2.155 branch) ──────────
    # A TD3 checkpoint has the same VDN LayerNorm critic (dim=66, flat, has_ln)
    # and a marker, but a DETERMINISTIC actor with NO log_std head.  The
    # `not has_log_std` guard is what separates it from the v2.16 SAC family,
    # which shares marker space (>= 2.155).  Loaded via TD3.load, not SAC.load.
    if dim == 66 and key_fmt == 'flat' and has_ln and (not has_log_std) and obs_marker >= 2.185:
        from stable_baselines3 import TD3
        from src.rl.networks_td3 import TD3VDNPolicy
        model = TD3.load(
            str(model_path), device=device,
            custom_objects={'policy_class': TD3VDNPolicy},
        )
        label = ('VDN factorised - v2.19 (TD3: deterministic actor + '
                 'target-policy smoothing; same VDN LayerNorm critic)')
        # v2.19 uses the v2.16 observation contract (normalised globals, RAIN_REF=30).
        return model, label, 'v27', True, RAIN_REF_V216

    # ── v2.22 TD3 branch (9-feature: prev_u in obs → dim 67 critic) ────────
    # v2.22 adds u_{t-1} as a 9th per-agent feature for Markov-r5.  The critic
    # input dim goes from 66 (8-feat) to 67 (9-feat), which would otherwise
    # fall through to the old v2.8 SAC branch (also dim=67 but with log_std).
    # Guard: dim==67 + no log_std (TD3) + has_ln (v2.11 LayerNorm critic).
    if dim == 67 and key_fmt == 'flat' and has_ln and (not has_log_std):
        from stable_baselines3 import TD3
        from src.rl.networks_td3_prevu import TD3VDNPolicyPrevU
        model = TD3.load(
            str(model_path), device=device,
            custom_objects={'policy_class': TD3VDNPolicyPrevU},
        )
        label = ('VDN factorised - v2.22 (TD3 + prev_u: 9 features/agent, '
                 'Markov-r5; same VDN LayerNorm critic)')
        return model, label, 'v27_prevu', True, RAIN_REF_V216

    if dim == 837:
        policy_class = MonolithicCTDESACPolicy
        label = 'monolithic (pre-VDN)'
        obs_layout = 'v26'
    elif dim == 63 and key_fmt == 'wrapped':
        policy_class = WrappedVDNCTDESACPolicy
        label = 'VDN factorised - v2.6 (local_q_net wrapper)'
        obs_layout = 'v26'
    elif dim == 63 and key_fmt == 'flat':
        policy_class = WrappedVDNCTDESACPolicy
        label = 'VDN factorised - v2.6 (flat keys, treated as legacy)'
        obs_layout = 'v26'
    elif dim == 66 and key_fmt == 'flat' and has_ln and obs_marker >= 2.155:
        # v2.16: v2.15 architecture + RAIN_REF=30 + capped auto-α.
        # Architecturally byte-identical to v2.15; the marker change lets the
        # runner identify v2.16 checkpoints AND apply the tighter rain
        # normaliser at eval time.
        # NB: 2.155 = midway between 2.15 and 2.16 — float32 storage of
        # torch.tensor([2.16]) round-trips as 2.1599998... and
        # torch.tensor([2.15]) as 2.1499998..., so strict ">= 2.16" would
        # mis-classify v2.16.
        policy_class = V216CTDESACPolicy
        label = ('VDN factorised - v2.16 (v2.15 architecture + RAIN_REF=30 '
                 '+ capped auto-α for rain-input dynamic range)')
        obs_layout = 'v27'
        normalize_globals = True
        rain_normaliser = RAIN_REF_V216   # tightened rainfall normaliser
    elif dim == 66 and key_fmt == 'flat' and has_ln and obs_marker >= 2.145:
        # v2.15: v2.14 architecture trained with LINEAR r6 (instead of quadratic).
        # Architecturally byte-identical to v2.14; the marker change lets the
        # runner identify r6-linear checkpoints in saved files.
        # NB: thresholds use mid-points (2.145 = midway between 2.14 and 2.15)
        # to tolerate float32 storage precision — torch.tensor([2.15]) round-trips
        # through the saved zip as 2.1499998... and torch.tensor([2.14]) as
        # 2.1399998..., so a strict ">= 2.15" check would mis-classify v2.15.
        policy_class = V215CTDESACPolicy
        label = ('VDN factorised - v2.15 (v2.14 architecture + linear r6 '
                 'for uniform overshoot-gradient signal)')
        obs_layout = 'v27'
        normalize_globals = True
    elif dim == 66 and key_fmt == 'flat' and has_ln and obs_marker >= 2.135:
        # v2.14: v2.13 architecture trained with α=0.01.  Architecturally
        # identical to v2.13; the marker change lets the runner identify
        # α-tuned checkpoints in saved files.
        # NB: thresholds use mid-points (2.135 = midway between 2.13 and 2.14)
        # to tolerate float32 storage precision — torch.tensor([2.14]) round-trips
        # through the saved zip as 2.1399998... and torch.tensor([2.13]) as
        # 2.1299999..., so a strict ">= 2.14" check would mis-classify v2.14.
        policy_class = V214CTDESACPolicy
        label = ('VDN factorised - v2.14 (v2.13 architecture + α=0.01 '
                 'for entropy/reward SNR recovery)')
        obs_layout = 'v27'
        normalize_globals = True
    elif dim == 66 and key_fmt == 'flat' and has_ln and obs_marker >= 2.125:
        # v2.13: v2.12 + actor-only input re-centering (2*x - 1 on the actor's
        # per-agent input).  Same global/forecast normalisation as v2.12 (the
        # runner must still divide rainfall/Kc_ET/radiation by their refs).
        # 2.125 = midway between 2.12 and 2.13.
        policy_class = V213CTDESACPolicy
        label = ('VDN factorised - v2.13 (v2.12 + actor-only input '
                 're-centering for dead-ReLU capacity recovery)')
        obs_layout = 'v27'
        normalize_globals = True
    elif dim == 66 and key_fmt == 'flat' and has_ln and obs_marker >= 2.0:
        # v2.12: v2.11 LayerNorm critic + LeakyReLU actor, trained on the
        # NORMALISED global/forecast observation block.  Eval must apply the
        # same normalisation (normalize_globals=True).
        # Threshold 2.0 catches the v2.12 marker (2.119...) while excluding
        # legacy v2.11/v2.7 checkpoints which have NO marker (treated as 0.0).
        policy_class = V212CTDESACPolicy
        label = ('VDN factorised - v2.12 (8 features/agent, LayerNorm '
                 'critic, LeakyReLU actor, normalised globals)')
        obs_layout = 'v27'
        normalize_globals = True
    elif dim == 66 and key_fmt == 'flat' and has_ln:
        # v2.11: same obs layout as v2.7 (8 features/agent, 1097-dim) but
        # the critic has LayerNorm after each hidden Linear layer.
        policy_class = V211CTDESACPolicy
        label = 'VDN factorised - v2.11 (8 features/agent, LayerNorm critic)'
        obs_layout = 'v27'
    elif dim == 66 and key_fmt == 'flat':
        policy_class = V27CTDESACPolicy
        label = 'VDN factorised - v2.7 (8 features/agent)'
        obs_layout = 'v27'
    elif dim == 67 and key_fmt == 'flat':
        policy_class = CTDESACPolicy
        label = 'VDN factorised - v2.8 (9 features/agent)'
        obs_layout = 'v28'
    else:
        raise ValueError(
            f"Unrecognised critic architecture: dim={dim}, key_format={key_fmt!r}, "
            f"has_layernorm={has_ln}, obs_marker={obs_marker}. "
            f"Expected (837,flat), (63,wrapped), (63,flat), "
            f"(66,flat,LN,marker>=2.16)=v2.16, (66,flat,LN,marker>=2.15)=v2.15, "
            f"(66,flat,LN,marker>=2.14)=v2.14, (66,flat,LN,marker>=2.13)=v2.13, "
            f"(66,flat,LN,marker>=2.12)=v2.12, "
            f"(66,flat,LN)=v2.11, (66,flat)=v2.7, or (67,flat)=v2.8."
        )
    model = SAC.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": policy_class},
    )
    return model, label, obs_layout, normalize_globals, rain_normaliser


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
        # is unchanged.
        # v2.12: _load_model may also return a normalize_globals flag (4-tuple).
        # v2.16: _load_model may additionally return a rain_normaliser (5-tuple).
        # Older overrides returning 3- or 4-tuples still work.
        _loaded = self._load_model()
        if len(_loaded) == 5:
            (self.model, _arch_label, obs_layout,
             normalize_globals, rain_normaliser) = _loaded
        elif len(_loaded) == 4:
            self.model, _arch_label, obs_layout, normalize_globals = _loaded
            rain_normaliser = RAIN_REF
        else:
            self.model, _arch_label, obs_layout = _loaded
            normalize_globals = False
            rain_normaliser = RAIN_REF
        self._obs_layout = obs_layout    # 'v28' | 'v27' | 'v26'
        self._normalize_globals = bool(normalize_globals)
        self._rain_normaliser = float(rain_normaliser)

        if self.verbose:
            print(f"  Loaded checkpoint: critic architecture = {_arch_label}")
            layout_desc = {
                'v28': '1227-dim, 9 features/agent',
                'v27': '1097-dim, 8 features/agent',
                'v27_prevu': '1227-dim, 9 features/agent (8 + prev_u)',
                'v26': '707-dim, 5 features/agent',
            }
            print(
                f"  Observation layout = {obs_layout} ({layout_desc[obs_layout]})")
            _normdesc = ('v2.12 (per-agent + normalised global/forecast block)'
                         if self._normalize_globals
                         else 'v2.8.1 (per-agent only; raw global/forecast block)')
            print(f"  Obs normalisation:  {_normdesc}")
            if self._normalize_globals:
                print(
                    f"  rain_normaliser:    {self._rain_normaliser:.1f} mm/day")

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
        fc = self._fc_total
        wp = self._wp_total
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

        x4_norm = np.clip(state['x4'] / X4_REF, 0.0, 1.5).astype(np.float32)

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
        elif self._obs_layout == 'v27_prevu':
            # v2.22: 8 base features + prev_u_norm as 9th per-agent feature.
            # Mirrors gym_env_prev_u.py: prev_u_norm = clip(u_{t-1}/UB_MM, 0, 1).
            # On the first day (or after reset), self._u_prev is zeros.
            prev_u_norm = np.clip(
                self._u_prev / UB_MM, 0.0, 1.0,
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
                prev_u_norm,
            ], axis=1).flatten().astype(np.float32)   # (1170,)
        else:  # 'v26'
            agent_block = np.stack([
                x1_norm,
                x5_norm,
                x4_norm,
                x3,
                self._elev_norm,
            ], axis=1).flatten().astype(np.float32)   # (650,)

        # ── scalar block ─────────────────────────────────────────────────────
        day_frac = day / self._season_days
        budget_frac = budget_remaining / max(self._budget_total, 1e-6)
        budget_total_norm = self._budget_total / FULL_SEASON_NEED_MM

        water_spent = self._budget_total - float(budget_remaining)
        daily_budget = self._budget_total / self._season_days
        burn_rate = (
            (water_spent / max(day, 1)) / max(daily_budget, 1e-6)
            if day > 0 else 0.0
        )

        d = min(day, self._season_days - 1)
        # v2.12: divide raw weather by physical references so the actor sees the
        # same normalised scale it was trained on.  Legacy checkpoints
        # (normalize_globals=False) keep the raw magnitudes.
        # v2.16: rainfall denominator is per-checkpoint (RAIN_REF=70.0 for
        # v2.7-v2.15, RAIN_REF_V216=30.0 for v2.16).  Set in __init__ from
        # the loaded model's marker.
        _rain_s = self._rain_normaliser if self._normalize_globals else 1.0
        _etc_s = ETC_REF if self._normalize_globals else 1.0
        _rad_s = RAD_REF if self._normalize_globals else 1.0

        rain_today = float(self._climate['rainfall'][d]) / _rain_s
        ETc_today = float(self._precomputed.Kc_ET[d]) / _etc_s
        h2_today = float(self._precomputed.h2[d])
        h7_today = float(self._precomputed.h7[d])
        g_base_today = float(self._precomputed.g_base[d])

        scalars = np.array([
            day_frac, budget_frac, budget_total_norm, burn_rate,
            rain_today, ETc_today, h2_today, h7_today, g_base_today,
        ], dtype=np.float32)

        # ── forecast block ───────────────────────────────────────────────────
        H = self.forecast_horizon
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
            ETc_fc = _pad(fc_dict['ETc']) / _etc_s
            rad_fc = _pad(fc_dict['radiation']) / _rad_s
            h2_fc = _pad(self._precomputed.h2[d:end])
            h7_fc = _pad(self._precomputed.h7[d:end])
            g_fc = _pad(self._precomputed.g_base[d:end])
        else:
            rain_fc = _pad(self._climate['rainfall'][d:end]) / _rain_s
            ETc_fc = _pad(self._precomputed.Kc_ET[d:end]) / _etc_s
            rad_fc = _pad(self._climate['radiation'][d:end]) / _rad_s
            h2_fc = _pad(self._precomputed.h2[d:end])
            h7_fc = _pad(self._precomputed.h7[d:end])
            g_fc = _pad(self._precomputed.g_base[d:end])

        return np.concatenate([
            agent_block,
            scalars,
            rain_fc, ETc_fc, rad_fc, h2_fc, h7_fc, g_fc,
        ]).astype(np.float32)

    @property
    def solve_times(self):
        return list(self._inference_times)
