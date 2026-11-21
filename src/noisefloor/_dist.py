"""Probability distributions used by noisefloor.

noisefloor deliberately depends only on numpy. The Normal distribution comes from
the standard library; Student's t is implemented here via the regularized
incomplete beta function (continued-fraction evaluation, Lentz's method).

Small evals matter most -- AIME has 30 problems, many internal regression suites
have fewer than 100 -- and at those sizes the Normal approximation is visibly
too optimistic, so a correct t distribution is not optional.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_STD_NORMAL = NormalDist()
_EPS = 1e-15
_TINY = 1e-300
_MAX_ITER = 500


def norm_cdf(x: float) -> float:
    return _STD_NORMAL.cdf(x)


def norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")
    return _STD_NORMAL.inv_cdf(p)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    # Use the symmetry relation where the continued fraction converges fastest.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(df) or df > 1e7:
        return norm_cdf(t)
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided p-value P(|T| >= |t|)."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(df) or df > 1e7:
        return 2.0 * (1.0 - norm_cdf(abs(t)))
    if t == 0.0:
        return 1.0
    x = df / (df + t * t)
    return min(1.0, betainc(df / 2.0, 0.5, x))


def t_ppf(p: float, df: float) -> float:
    """Quantile function (inverse CDF) of Student's t."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(df) or df > 1e7:
        return norm_ppf(p)
    if p == 0.5:
        return 0.0
    # Solve on the upper half and reflect: the t distribution is symmetric.
    target = p if p > 0.5 else 1.0 - p
    lo, hi = 0.0, max(1.0, norm_ppf(target) * 2.0)
    while t_cdf(hi, df) < target:
        lo, hi = hi, hi * 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    q = 0.5 * (lo + hi)
    return q if p > 0.5 else -q


def critical_value(confidence: float, df: float | None) -> float:
    """Two-sided critical value for a ``confidence``-level interval.

    ``df=None`` selects the Normal distribution.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    upper = 1.0 - (1.0 - confidence) / 2.0
    if df is None:
        return norm_ppf(upper)
    return t_ppf(upper, df)
