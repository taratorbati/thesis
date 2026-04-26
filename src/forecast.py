# =============================================================================
# src/forecast.py
# Forecast providers for the MPC controller.
#
# A forecast provider is a callable:
#   f(day, climate, precomputed, horizon) → dict
#
# The returned dict contains arrays of length `horizon` for the MPC's
# prediction window, including both climate variables and precomputed
# quantities (h2, h7, g_base, Kc_ET → ETc).
#
# Two providers:
#   PerfectForecast  — returns the true future weather (upper bound on MPC info)
#   NoisyForecast    — multiplicative Gaussian noise on rainfall and ET₀,
#                      with σ growing as 0.15·√j (realistic forecast skill)
# =============================================================================

import numpy as np


class PerfectForecast:
    """Returns the true future weather over the horizon. No noise."""

    def __call__(self, day, climate, precomputed, horizon):
        """
        Parameters
        ----------
        day : int
            Current day (zero-indexed).
        climate : dict
            Full-season climate arrays (from extract_scenario).
        precomputed : Precomputed
            From src.precompute.get_precomputed.
        horizon : int
            Number of days to forecast.

        Returns
        -------
        dict
            Keys: 'rainfall', 'ETc', 'radiation', 'h2', 'h7', 'g_base'.
            Each is an array of length min(horizon, remaining_days).
        """
        n_days = len(climate['rainfall'])
        end = min(day + horizon, n_days)
        actual_horizon = end - day

        # Pad with last-day values if horizon extends beyond season
        def _slice_pad(arr, start, length):
            available = arr[start:start + length]
            if len(available) < length:
                pad = np.full(length - len(available), available[-1] if len(available) > 0 else 0.0)
                return np.concatenate([available, pad])
            return available

        Kc = precomputed.Kc_ET[day] / max(climate['ET'][day], 1e-6) if climate['ET'][day] > 0 else 1.15

        return {
            'rainfall':  _slice_pad(climate['rainfall'], day, horizon),
            'ETc':       _slice_pad(precomputed.Kc_ET, day, horizon),
            'radiation': _slice_pad(climate['radiation'], day, horizon),
            'h2':        _slice_pad(precomputed.h2, day, horizon),
            'h7':        _slice_pad(precomputed.h7, day, horizon),
            'g_base':    _slice_pad(precomputed.g_base, day, horizon),
        }


class NoisyForecast:
    """Multiplicative Gaussian noise on rainfall and ET₀.

    Noise model:
        ŵ(t+j|t) = w(t+j) × (1 + ε_j)
        ε_j ~ N(0, σ_j²)
        σ_j = sigma_base × √j

    Applied to rainfall and ETc. Temperature-derived quantities (h2, h7,
    g_base) are left unperturbed (operational temperature forecasts are
    more accurate than precipitation forecasts).

    Parameters
    ----------
    sigma_base : float
        Base noise level. Default 0.15 (15% error at 1-day lead).
    seed : int or None
        Random seed for reproducibility. If None, not seeded.
    """

    def __init__(self, sigma_base=0.15, seed=None):
        self.sigma_base = sigma_base
        self.rng = np.random.default_rng(seed)

    def __call__(self, day, climate, precomputed, horizon):
        # Start from perfect forecast
        perfect = PerfectForecast()(day, climate, precomputed, horizon)

        # Generate noise
        j = np.arange(1, horizon + 1, dtype=float)
        sigma = self.sigma_base * np.sqrt(j)
        noise = self.rng.normal(0, sigma)

        # Apply to rainfall and ETc (multiplicative, clipped to prevent negatives)
        perfect['rainfall'] = np.maximum(perfect['rainfall'] * (1 + noise), 0)
        perfect['ETc'] = np.maximum(perfect['ETc'] * (1 + noise), 0)

        return perfect
