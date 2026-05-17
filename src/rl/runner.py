# =============================================================================
# src/rl/runner.py  v2.7
# Inference runner for trained SAC models.
#
# Loads a trained SB3 SAC model (v2.7 CTDESACPolicy, or legacy v2.6 variants),
# runs it through the full ABM season, and saves results in the same format
# as the MPC runner.
#
# v2.7 changes:
#   - Critic-arch detection now recognises four input dimensions:
#         dim=66, flat     → CTDESACPolicy             (v2.7, current default)
#         dim=63, wrapped  → WrappedVDNCTDESACPolicy   (v2.6 best_model.zip)
#         dim=63, flat     → WrappedVDNCTDESACPolicy   (no such checkpoint
#                            in current repo; we accept and treat as legacy
#                            VDN to be safe)
#         dim=837, flat    → MonolithicCTDESACPolicy   (pre-VDN pilot)
#   - _build_obs() branches on self._is_v27_obs to produce either the v2.7
#     1097-dim layout (8 features/agent including 3 new static topographic
#     features) or the legacy v2.6 707-dim layout (5 features/agent).  This
#     allows the same runner to evaluate v2.7 checkpoints and v2.6 best_model
#     checkpoints — important for the Chapter 5 architecture-comparison story.
#
# burn_rate is derived from (budget_total - budget_remaining) / budget_total
# rather than from an internal _water_used accumulator. This avoids drift
# between the runner's internal accounting and the outer runner's clipped/
# scaled budget deductions (src/runner.py clips u before deducting from
# budget_remaining, so a self._water_used accumulator would diverge silently).
#
# Noisy forecast support (unchanged from v2.4.3):
#   forecast_mode='noisy' injects AR(1)-correlated multiplicative noise
#   into the 48 forecast dims of the observation.  The ABM transition always
#   uses the true climate — only the information presented to the policy is
#   corrupted.
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
    FULL_SEASON_NEED_MM,   # float scalar: 484.0 mm
)
from src.rl.networks import (
    CTDESACPolicy,              # v2.7 default — dim 66
    MonolithicCTDESACPolicy,    # pre-VDN — dim 837
    WrappedVDNCTDESACPolicy,    # v2.6 VDN — dim 63
)


def _detect_critic_arch(model_path: Path):
    """Peek inside a saved SB3 zip and return (input_dim, key_format).

    Returns
    -------
    (int, str)
        input_dim   : first Linear layer's input dimension
                       66  → v2.7 VDN factorised (8 feat + 57 glob + 1 act)
                       63  → v2.6 VDN factorised (5 feat + 57 glob + 1 act)
                       837 → pre-VDN monolithic (obs 707 + actions 130)
        key_format  : 'flat'    → critic.qf0.0.weight  (nn.Sequential subclass)
                      'wrapped' → critic.qf0.local_q_net.0.weight
                                  (nn.Module + wrapper)
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

    Four known checkpoint variants:
      dim=837, flat     → MonolithicCTDESACPolicy   (pre-VDN pilot)
      dim=63,  wrapped  → WrappedVDNCTDESACPolicy   (v2.6 best_model.zip)
      dim=63,  flat     → WrappedVDNCTDESACPolicy   (v2.6 alt-key checkpoint,
                                                     defensive — none exists
                                                     in the current repo)
      dim=66,  flat     → CTDESACPolicy             (v2.7 DEFAULT)

    Returns
    -------
    (SAC, str, str)
        model, arch_label, obs_layout
        obs_layout is 'v27' (1097-dim, 8 features/agent) or
                     'v26' (707-dim, 5 features/agent).
    """
    dim, key_fmt = _detect_critic_arch(model_path)
    if dim == 837:
        policy_class = MonolithicCTDESACPolicy
        label        = 'monolithic (pre-VDN)'
        obs_layout   = 'v26'
    elif dim == 63 and key_fmt == 'wrapped':
        policy_class = WrappedVDNCTDESACPolicy
        label        = 'VDN factorised – v2.6 (local_q_net wrapper)'
        obs_layout   = 'v26'
    elif dim == 63 and key_fmt == 'flat':
        # Defensive: no such checkpoint exists in the current repo, but if
        # one appears (e.g. an alt v2.6 training run), use the legacy v2.6
        # path since the per-agent feature count is still 5.
        policy_class = WrappedVDNCTDESACPolicy
        label        = 'VDN factorised – v2.6 (flat keys, treated as legacy)'
        obs_layout   = 'v26'
    elif dim == 66 and key_fmt == 'flat':
        policy_class = CTDESACPolicy
        label        = 'VDN factorised – v2.7 (8 features/agent)'
        obs_layout   = 'v27'
    else:
        raise ValueError(
            f"Unrecognised critic architecture: dim={dim}, key_format={key_fmt!r}. "
            f"Expected (837, flat), (63, wrapped), (63, flat), or (66, flat)."
        )
    model = SAC.load(
        str(model_path),
        device=device,
        custom_objects={"policy_class": policy_class},
    )
    return model, label, obs_layout


DEFAULT_FORECAST_HORIZON = 8


class RLController(Controller):
    """Controller wrapping a trained SAC model for inference.

    Builds the observation in the layout matching the checkpoint's training
    architecture (v2.7 = 1097-dim, v2.6 = 707-dim) and queries the SAC policy
    to produce a 130-dim irrigation action.

    Parameters
    ----------
    model_path : str or Path
    deterministic : bool
    forecast_horizon : int
        Must match training. Default 8.
    forecast_mode : str
        'perfect' (default) or 'noisy'.
        - 'perfect': true future climate is shown to the policy in the
          forecast block of the observation.  This is the primary
          evaluation mode.
        - 'noisy': AR(1)-correlated multiplicative noise is applied to
          rainfall and ETc in the 8-day forecast block before the obs
          is passed to the policy.  The ABM transition still uses true
          climate.  Used for the Chapter 5 disturbance robustness analysis.
    noise_sigma : float
        Base noise level (std at 1-day lead). Default 0.15 (15%).
        Ignored when forecast_mode='perfect'.
    noise_rho : float
        AR(1) persistence parameter in [0, 1). Default 0.6.
        Ignored when forecast_mode='perfect'.
    noise_seed : int or None
        RNG seed for NoisyForecast. Set to the same value used for MPC
        noisy evaluation (default 42) so that performance differences
        between controllers are attributable to policy quality, not to
        different noise realizations. Ignored when forecast_mode='perfect'.
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
        self._is_v27_obs = (obs_layout == 'v27')
        if self.verbose:
            print(f"  Loaded checkpoint: critic architecture = {_arch_label}")
            print(f"  Observation layout = {obs_layout} "
                  f"({'1097-dim, 8 features/agent' if self._is_v27_obs else '707-dim, 5 features/agent'})")

        self._inference_times = []
        self._noisy_forecast = None   # initialized in reset()

        name = f"sac_{'det' if deterministic else 'stoch'}_{forecast_mode}"
        super().__init__(name=name)

    def reset(self, terrain, crop, season_days, budget_total, scenario_name=None):
        self._inference_times = []
        self._terrain = terrain
        self._crop = crop
        self._N = terrain['N']
        self._season_days = season_days
        self._budget_total = float(budget_total)

        # Static topographic features — shared by v2.7 (used) and v2.6 (only
        # elev_norm used).  Pre-computed once at reset() so _build_obs() is
        # fast.
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

        self._fc_total = crop['theta6'] * crop['theta5']
        self._u_prev = np.zeros(self._N)

        from src.precompute import get_precomputed
        from climate_data import load_cleaned_data, extract_scenario_by_name

        scenario = scenario_name or 'dry'
        self._precomputed = get_precomputed(scenario, crop['name'].lower())
        df = load_cleaned_data()
        self._climate = extract_scenario_by_name(df, scenario, crop)

        # Noisy forecast provider — same logic as v2.6.
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
        """Construct the observation vector.

        v2.7 layout (1097-dim):
          Per-agent block (8 × 130 = 1040, agent-major):
            [x1_norm, x5_norm, x4_norm, x3, elev_norm,
             Nr_norm, Nr_internal_norm, n_upstream_norm]
          Scalar block (9, positions 1040–1048)
          Forecast block (48, positions 1049–1096)

        v2.6 LEGACY layout (707-dim):
          Per-agent block (5 × 130 = 650, agent-major):
            [x1_norm, x5_norm, x4_norm, x3, elev_norm]
          Scalar block (9, positions 650–658)
          Forecast block (48, positions 659–706)

        Branching happens once on self._is_v27_obs, set at load time.

        Scalar order (both layouts):
          [0] day_frac
          [1] budget_frac
          [2] budget_total_norm
          [3] burn_rate
          [4] rain_today
          [5] ETc_today
          [6] h2_today
          [7] h7_today
          [8] g_base_today
        """
        N = self._N
        fc = self._fc_total

        # Common dynamic features
        x1_norm = state['x1'] / fc
        x5_norm = state['x5'] / X5_REF
        x4_norm = state['x4'] / X4_REF
        x3      = state['x3']

        # Per-agent block — assembled in agent-major order
        if self._is_v27_obs:
            # 8 features per agent
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
        else:
            # Legacy v2.6: 5 features per agent
            agent_block = np.stack([
                x1_norm,
                x5_norm,
                x4_norm,
                x3,
                self._elev_norm,
            ], axis=1).flatten().astype(np.float32)   # (650,)

        # Scalars — unchanged between layouts
        day_frac          = day / self._season_days
        budget_frac       = budget_remaining / max(self._budget_total, 1e-6)
        budget_total_norm = self._budget_total / FULL_SEASON_NEED_MM

        # burn_rate: derived from the runner-provided budget_remaining so it
        # stays consistent with the outer runner's clipped budget accounting.
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
            # h2, h7, g_base use perfect values — NoisyForecast only
            # corrupts rainfall and ETc to match the MPC noise model.
            h2_fc = _pad(self._precomputed.h2[d:end])
            h7_fc = _pad(self._precomputed.h7[d:end])
            g_fc  = _pad(self._precomputed.g_base[d:end])
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
