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
#   NoisyForecast    — multiplicative Gaussian noise on rainfall and ETc with
#                      AR(1) temporal persistence (ρ = 0.6 by default)
#
# v2.2 changes:
#   - PerfectForecast: removed dead Kc variable that was computed but never used.
#   - NoisyForecast: replaced independent per-call Gaussian noise with an
#     AR(1)-correlated process. The previous formulation re-drew fresh
#     independent noise on every step(), so the MPC's view of (e.g.) day d+5
#     jumped randomly between consecutive solves on days d, d+1, d+2.
#     Operational NWP forecast errors are strongly persistent at short lead
#     times (Buizza et al., 2005, Mon. Wea. Rev.), so the AR(1) form is more
#     physically representative.
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
            Each is an array of length `horizon`. End-of-season padding is
            handled by repeating the last available value (forward fill).
        """
        # Pad with last-day values if horizon extends beyond season
        def _slice_pad(arr, start, length):
            available = arr[start:start + length]
            if len(available) < length:
                pad = np.full(length - len(available),
                              available[-1] if len(available) > 0 else 0.0)
                return np.concatenate([available, pad])
            return available

        return {
            'rainfall':  _slice_pad(climate['rainfall'], day, horizon),
            'ETc':       _slice_pad(precomputed.Kc_ET, day, horizon),
            'radiation': _slice_pad(climate['radiation'], day, horizon),
            'h2':        _slice_pad(precomputed.h2, day, horizon),
            'h7':        _slice_pad(precomputed.h7, day, horizon),
            'g_base':    _slice_pad(precomputed.g_base, day, horizon),
        }


class NoisyForecast:
    """Multiplicative AR(1)-correlated Gaussian noise on rainfall and ETc.

    Real meteorological forecast errors are not independent across consecutive
    forecast issuances — operational ensemble prediction systems exhibit
    strong day-to-day persistence of forecast errors at short lead times
    (Buizza et al., 2005, Mon. Wea. Rev., 'A comparison of the ECMWF, MSC,
    and NCEP global ensemble prediction systems'). The previous implementation
    drew fresh independent noise on every call, which causes the MPC's view
    of (e.g.) day d+5 to jump randomly between consecutive solves on days d,
    d+1, d+2. This is physically unrealistic and induces erratic MPC
    behaviour because the controller treats each forecast as if the previous
    forecast's information had been entirely discarded.

    Noise model:
        ŵ(t+j | t) = w(t+j) × (1 + ε_j(t)),   j = 1..H

        Marginal:    ε_j(t) ~ N(0, σ_j²),     σ_j = σ_base · √j
                     (skill grows with lead time — standard assumption)

        Persistence: ε_j(t+1) = ρ · ε_j(t) + √(1 - ρ²) · η_j(t),
                     η_j(t) ~ N(0, σ_j²)
                     (preserves the marginal variance σ_j² in steady state)

    With ρ = 0.6, the noise envelope and marginal skill profile are preserved
    while introducing realistic temporal persistence consistent with daily
    precipitation forecast error autocorrelation at short lead times.

    A single noise process is shared between rainfall and ETc because forecast
    errors in these two variables are physically correlated through synoptic-
    scale weather systems — a wet system simultaneously increases rain and
    lowers reference ET through humidity and reduced solar radiation. This
    matches the original implementation's behaviour and is a deliberate
    simplification consistent with operational NWP error structure.

    Parameters
    ----------
    sigma_base : float
        Base noise level. Default 0.15 (15% error at 1-day lead).
    rho : float
        AR(1) persistence parameter, in [0, 1). Default 0.6 (Buizza et al.
        2005, daily precipitation forecast error autocorrelation at short
        lead times).
    seed : int or None
        Random seed for reproducibility. If None, not seeded.
    """

    def __init__(self, sigma_base=0.15, rho=0.6, seed=None):
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        self.sigma_base = sigma_base
        self.rho = rho
        self.rng = np.random.default_rng(seed)
        # Persistent noise vector — initialized lazily on first call so the
        # horizon length can vary if the controller is rebuilt with a
        # different Hp.
        self._eps = None
        self._horizon = None

    def reset(self, seed=None):
        """Reset the noise state. Call between independent episodes to
        prevent state leakage across runs.

        Parameters
        ----------
        seed : int or None
            If provided, re-seed the RNG. If None, keep the existing RNG
            state but clear the persistent noise vector.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._eps = None
        self._horizon = None

    def __call__(self, day, climate, precomputed, horizon):
        # Start from perfect forecast
        perfect = PerfectForecast()(day, climate, precomputed, horizon)

        # Lead-time-dependent marginal standard deviations
        j = np.arange(1, horizon + 1, dtype=float)
        sigma = self.sigma_base * np.sqrt(j)

        if self._eps is None or self._horizon != horizon:
            # Lazy initialize: draw from the stationary marginal distribution
            # so the AR(1) process is in equilibrium from step 0.
            self._eps = self.rng.normal(0.0, sigma)
            self._horizon = horizon
        else:
            # AR(1) update: ε(t+1) = ρ·ε(t) + √(1-ρ²)·η,  η ~ N(0, σ²)
            # The √(1-ρ²) scaling preserves the stationary marginal variance.
            innov = self.rng.normal(0.0, sigma)
            self._eps = self.rho * self._eps + np.sqrt(1.0 - self.rho ** 2) * innov

        # Apply (multiplicative, clipped to prevent negatives — rainfall and
        # ETc are physically non-negative)
        perfect['rainfall'] = np.maximum(perfect['rainfall'] * (1 + self._eps), 0)
        perfect['ETc']      = np.maximum(perfect['ETc']      * (1 + self._eps), 0)

        return perfect
