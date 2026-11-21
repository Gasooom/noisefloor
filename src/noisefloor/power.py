"""Power analysis: how big does an eval need to be?

The most expensive mistake in evaluation is running an eval that *cannot*
detect the effect you care about, then reading tea leaves in the result.
These functions answer, before you spend compute:

* how many questions do I need to detect a change of ``effect``?
* with ``n`` questions, what is the smallest change I can reliably detect?
* with ``n`` questions, what is my chance of detecting a change of ``effect``?

Planning formulas use the Normal approximation (standard practice for sample
size calculations). For binary scores, the variance of a paired difference is

    Var(d) = p1(1-p1) + p2(1-p2) - 2 * rho * sqrt(p1(1-p1) p2(1-p2))

where ``rho`` is the per-question correlation between the two models. Models
evaluated on the same questions are usually positively correlated (hard
questions are hard for everyone), which is exactly why paired designs win.
Clustering multiplies the variance by the design effect ``1 + (m - 1) * icc``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._dist import norm_cdf, norm_ppf
from .stats import Comparison


def design_effect(cluster_size: float, icc: float) -> float:
    """Variance inflation for clusters of mean size ``m`` and intra-class correlation ``icc``."""
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if not -1.0 <= icc <= 1.0:
        raise ValueError("icc must be in [-1, 1]")
    return max(0.0, 1.0 + (cluster_size - 1.0) * icc)


def _z(alpha: float, power: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    return norm_ppf(1.0 - alpha / 2.0) + norm_ppf(power)


def paired_variance(baseline_rate: float, candidate_rate: float, correlation: float = 0.0) -> float:
    """Per-question variance of the paired difference for binary scores."""
    for label, r in (("baseline_rate", baseline_rate), ("candidate_rate", candidate_rate)):
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"{label} must be in [0, 1], got {r}")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    v1 = baseline_rate * (1.0 - baseline_rate)
    v2 = candidate_rate * (1.0 - candidate_rate)
    return max(0.0, v1 + v2 - 2.0 * correlation * math.sqrt(v1 * v2))


def _variance(
    effect: float,
    sd: float | None,
    baseline_rate: float | None,
    correlation: float,
) -> float:
    if sd is not None:
        if sd < 0:
            raise ValueError("sd must be non-negative")
        return sd * sd
    if baseline_rate is None:
        raise ValueError("provide either sd (of per-question differences) or baseline_rate")
    return paired_variance(baseline_rate, baseline_rate + effect, correlation)


def sample_size(
    effect: float,
    *,
    sd: float | None = None,
    baseline_rate: float | None = None,
    correlation: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    design_effect: float = 1.0,
) -> int:
    """Questions needed to detect a paired difference of ``effect``.

    Give either ``sd`` (the standard deviation of per-question differences, e.g.
    from a pilot run) or ``baseline_rate`` for binary scores. ``correlation=0``
    is the conservative choice when you have no pilot data, and also equals the
    requirement *per model* for an unpaired design.
    """
    if effect == 0:
        raise ValueError("effect must be nonzero")
    var = _variance(effect, sd, baseline_rate, correlation) * design_effect
    return max(2, math.ceil(_z(alpha, power) ** 2 * var / (effect * effect)))


def minimum_detectable_effect(
    n: int,
    *,
    sd: float | None = None,
    baseline_rate: float | None = None,
    correlation: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    design_effect: float = 1.0,
) -> float:
    """Smallest improvement detectable with ``n`` questions at the given power.

    For binary scores the variance depends on the effect itself, so this solves
    ``effect = z * sqrt(Var(effect) * deff / n)`` by bisection.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    z = _z(alpha, power)
    if sd is not None:
        return z * sd * math.sqrt(design_effect / n)
    if baseline_rate is None:
        raise ValueError("provide either sd or baseline_rate")

    def gap(effect: float) -> float:
        var = paired_variance(baseline_rate, baseline_rate + effect, correlation) * design_effect
        return effect - z * math.sqrt(var / n)

    hi = 1.0 - baseline_rate
    if hi <= 0 or gap(hi) < 0:
        return math.inf  # cannot be detected even by going to 100%
    lo = 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi


def achieved_power(
    n: int,
    effect: float,
    *,
    sd: float | None = None,
    baseline_rate: float | None = None,
    correlation: float = 0.0,
    alpha: float = 0.05,
    design_effect: float = 1.0,
) -> float:
    """Probability that an eval of ``n`` questions detects a true change of ``effect``."""
    if n < 2:
        raise ValueError("n must be at least 2")
    var = _variance(effect, sd, baseline_rate, correlation) * design_effect
    if var == 0:
        return 1.0 if effect != 0 else alpha
    se = math.sqrt(var / n)
    z_a = norm_ppf(1.0 - alpha / 2.0)
    shift = abs(effect) / se
    return norm_cdf(shift - z_a) + norm_cdf(-shift - z_a)


def sample_size_for_margin(
    margin: float,
    *,
    rate: float = 0.5,
    sd: float | None = None,
    confidence: float = 0.95,
    design_effect: float = 1.0,
) -> int:
    """Questions needed so a single score's CI half-width is at most ``margin``."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    var = sd * sd if sd is not None else rate * (1.0 - rate)
    z = norm_ppf(1.0 - (1.0 - confidence) / 2.0)
    return max(2, math.ceil(z * z * var * design_effect / (margin * margin)))


@dataclass(frozen=True)
class PilotPlan:
    """Sample-size plan derived from an observed comparison (a pilot)."""

    effect: float
    n_current: int
    n_required: int
    power_current: float
    sd_diff_effective: float


def plan_from_comparison(
    comparison: Comparison, effect: float, *, alpha: float = 0.05, power: float = 0.8
) -> PilotPlan:
    """Use an existing paired comparison as a pilot study.

    The *effective* per-question SD (``se * sqrt(n)``) already includes the
    benefit of pairing and the cost of clustering, so it is the right input
    for planning a larger run of the same eval.
    """
    sd_eff = comparison.diff.se * math.sqrt(comparison.n)
    return PilotPlan(
        effect=effect,
        n_current=comparison.n,
        n_required=sample_size(effect, sd=sd_eff, alpha=alpha, power=power),
        power_current=achieved_power(comparison.n, effect, sd=sd_eff, alpha=alpha),
        sd_diff_effective=sd_eff,
    )
