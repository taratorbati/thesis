# =============================================================================
# src/mpc/smoothing.py
# Smooth approximations of non-differentiable operators for IPOPT.
#
# The ABM uses max(), min(), clip(), and piecewise functions extensively.
# IPOPT's interior-point method can handle CasADi's native fmax/fmin, but
# may struggle at kink points. These smooth replacements provide C2-smooth
# alternatives that can be toggled on if IPOPT reports convergence issues.
#
# Each function works with both CasADi symbolics and numpy arrays.
# =============================================================================

import casadi as ca


def smooth_max(x, y, eps=0.01):
    """Smooth approximation of max(x, y).

    Uses the log-sum-exp trick:  max(x,y) ≈ eps * log(exp(x/eps) + exp(y/eps))
    But this overflows for large x. Instead use the equivalent form:
        max(x,y) ≈ 0.5 * (x + y + sqrt((x - y)^2 + eps^2))

    Parameters
    ----------
    x, y : CasADi SX/MX or float
    eps : float
        Smoothing parameter. Smaller = closer to true max but stiffer.
    """
    return 0.5 * (x + y + ca.sqrt((x - y)**2 + eps**2))


def smooth_max_zero(x, eps=0.01):
    """Smooth approximation of max(x, 0).

    smooth_max_zero(x) ≈ 0.5 * (x + sqrt(x^2 + eps^2))
    """
    return 0.5 * (x + ca.sqrt(x**2 + eps**2))


def smooth_min(x, y, eps=0.01):
    """Smooth approximation of min(x, y).

    min(x,y) = -max(-x, -y)
    """
    return 0.5 * (x + y - ca.sqrt((x - y)**2 + eps**2))


def smooth_clip(x, lo, hi, eps=0.01):
    """Smooth approximation of clip(x, lo, hi).

    clip(x, lo, hi) = min(max(x, lo), hi)
    """
    return smooth_min(smooth_max(x, lo, eps), hi, eps)
