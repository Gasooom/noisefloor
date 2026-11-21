"""Core estimators: means, standard errors, and paired comparisons.

The methods follow the recommendations in Miller (2024), "Adding Error Bars to
Evals: A Statistical Approach to Language Model Evaluations":

1. Report the standard error of the mean, not just the mean.
2. When questions are grouped (several questions per passage, several tests
   per function), use cluster-robust standard errors. Treating correlated
   questions as independent makes intervals too narrow, sometimes by a lot.
3. When two models answer the same questions, analyse the per-question
   *differences*. Question difficulty cancels out, which typically makes the
   comparison much more sensitive than comparing two independent means.
4. When questions are sampled more than once, separate within-question
   (sampling) variance from between-question variance, so you know whether
   to buy more questions or more samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._dist import critical_value, norm_ppf, t_sf_two_sided
from .data import EvalRun, FloatArray, align

Verdict = Literal["improvement", "regression", "inconclusive"]

#: Below this many clusters, cluster-robust SEs are known to be biased downward.
FEW_CLUSTERS = 20
SMALL_N = 30


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a standard error and a confidence interval."""

    mean: float
    se: float
    ci_low: float
    ci_high: float
    confidence: float
    n: int
    df: float
    method: Literal["iid", "clustered"]
    n_clusters: int | None = None

    @property
    def margin(self) -> float:
        """Half-width of the confidence interval."""
        return (self.ci_high - self.ci_low) / 2.0

    def contains(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high


def _standard_error(
    values: FloatArray, clusters: tuple[str, ...] | None
) -> tuple[float, float, Literal["iid", "clustered"], int | None]:
    """Return (se, degrees_of_freedom, method, n_clusters) for the mean of ``values``.

    The clustered estimator is the CR1 sandwich estimator for a mean:

        Var = G / (G - 1) * sum_c (sum_{i in c} (x_i - xbar))^2 / n^2

    With one question per cluster it reduces exactly to s^2 / n.
    """
    n = values.shape[0]
    if clusters is not None:
        labels, inverse = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
        g = len(labels)
        if g < n:
            if g < 2:
                return math.nan, math.nan, "clustered", g
            resid = values - values.mean()
            sums = np.bincount(inverse, weights=resid, minlength=g)
            var = (g / (g - 1)) * float(np.sum(sums * sums)) / (n * n)
            return math.sqrt(var), float(g - 1), "clustered", g
    if n < 2:
        return math.nan, math.nan, "iid", None
    return float(np.std(values, ddof=1)) / math.sqrt(n), float(n - 1), "iid", None


def estimate_mean(
    values: FloatArray,
    *,
    clusters: tuple[str, ...] | None = None,
    confidence: float = 0.95,
) -> Estimate:
    """Mean of ``values`` with a t-based confidence interval."""
    mean = float(np.mean(values))
    se, df, method, g = _standard_error(values, clusters)
    if math.isnan(se):
        lo = hi = math.nan
    else:
        crit = critical_value(confidence, df)
        lo, hi = mean - crit * se, mean + crit * se
    return Estimate(
        mean=mean,
        se=se,
        ci_low=lo,
        ci_high=hi,
        confidence=confidence,
        n=int(values.shape[0]),
        df=df,
        method=method,
        n_clusters=g,
    )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the CLT interval it never has zero width at 0% or 100% and never
    leaves [0, 1], which matters for tiny or saturated evals.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = critical_value(confidence, None)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class VarianceComponents:
    """Decomposition of score variance for runs with repeated samples.

    For a question answered K times, Var(question mean) = between + within / K.
    ``between`` is irreducible by resampling; only more questions shrink it.
    """

    mean_k: float
    within: float
    between: float
    se_by_k: dict[int, float]
    #: SE achievable with infinitely many samples per question (only more questions help).
    se_limit: float

    @property
    def within_share(self) -> float:
        """Share of single-sample variance that comes from sampling noise."""
        total = self.between + self.within
        return self.within / total if total > 0 else 0.0


def variance_components(
    run: EvalRun, se_now: float, ks: tuple[int, ...] = (1, 2, 4, 8, 16)
) -> VarianceComponents | None:
    """Estimate within/between-question variance; ``None`` if no question has K >= 2."""
    multi = run.n_samples >= 2
    if not np.any(multi):
        return None
    within = float(np.mean(run.within_var[multi]))
    filled = np.where(multi, run.within_var, within)
    between = max(0.0, float(np.var(run.scores, ddof=1)) - float(np.mean(filled / run.n_samples)))
    n = run.n
    iid_now = math.sqrt(max(between + float(np.mean(within / run.n_samples)), 1e-300) / n)
    # Carry any clustering inflation of the current data over to the projections.
    inflation = se_now / iid_now if iid_now > 0 and not math.isnan(se_now) else 1.0
    se_by_k = {k: inflation * math.sqrt((between + within / k) / n) for k in ks}
    return VarianceComponents(
        mean_k=float(np.mean(run.n_samples)),
        within=within,
        between=between,
        se_by_k=se_by_k,
        se_limit=inflation * math.sqrt(between / n),
    )


@dataclass(frozen=True)
class Summary:
    """Everything worth knowing about a single eval run."""

    name: str
    estimate: Estimate
    naive: Estimate | None
    wilson: tuple[float, float] | None
    components: VarianceComponents | None
    warnings: list[str]

    @property
    def design_effect(self) -> float | None:
        """Variance inflation from clustering, (SE_clustered / SE_iid)^2."""
        if self.naive is None or self.naive.se == 0:
            return None
        return (self.estimate.se / self.naive.se) ** 2


def summarize(run: EvalRun, *, confidence: float = 0.95, cluster: bool = True) -> Summary:
    """Mean, standard error, and diagnostics for one eval run."""
    clusters = run.clusters if cluster else None
    est = estimate_mean(run.scores, clusters=clusters, confidence=confidence)
    naive = None
    if est.method == "clustered":
        naive = estimate_mean(run.scores, clusters=None, confidence=confidence)
    wilson = None
    if run.is_binary and est.method == "iid" and bool(np.all(run.n_samples == 1)):
        wilson = wilson_interval(int(run.scores.sum()), run.n, confidence)

    warnings: list[str] = []
    if run.n < SMALL_N:
        warnings.append(
            f"only {run.n} questions: intervals are wide and rely on the t approximation; "
            "consider --bootstrap"
        )
    if est.se == 0:
        warnings.append("all questions have the same score; the CLT interval is degenerate")
    if est.method == "clustered" and est.n_clusters is not None and est.n_clusters < FEW_CLUSTERS:
        warnings.append(
            f"only {est.n_clusters} clusters: cluster-robust standard errors are biased "
            "downward with few clusters"
        )
    if cluster and run.clusters is not None and est.method == "iid":
        warnings.append("every question is its own cluster; clustering has no effect")
    return Summary(
        name=run.name,
        estimate=est,
        naive=naive,
        wilson=wilson,
        components=variance_components(run, est.se),
        warnings=warnings,
    )


@dataclass(frozen=True)
class Comparison:
    """Paired comparison of a candidate run against a baseline run."""

    baseline_name: str
    candidate_name: str
    baseline: Estimate
    candidate: Estimate
    diff: Estimate
    t_stat: float
    p_value: float
    unpaired_se: float
    correlation: float
    wins: int
    losses: int
    ties: int
    power: float
    mde: float
    dropped_baseline: int
    dropped_candidate: int
    warnings: list[str]

    @property
    def n(self) -> int:
        return self.diff.n

    @property
    def verdict(self) -> Verdict:
        if math.isnan(self.diff.ci_low):
            return "inconclusive"
        if self.diff.ci_low > 0:
            return "improvement"
        if self.diff.ci_high < 0:
            return "regression"
        return "inconclusive"

    @property
    def significant(self) -> bool:
        return self.verdict != "inconclusive"

    @property
    def variance_reduction(self) -> float:
        """Fraction of variance removed by pairing, vs. comparing independent means."""
        if self.unpaired_se == 0 or math.isnan(self.unpaired_se):
            return 0.0
        return 1.0 - (self.diff.se / self.unpaired_se) ** 2

    @property
    def n_required(self) -> int | None:
        """Questions needed to detect the *observed* difference at the run's power.

        Scales the current sample size by (MDE / |diff|)^2, holding the per-question
        variance and cluster structure fixed. ``None`` if the difference is zero.
        """
        if self.diff.mean == 0 or math.isnan(self.mde):
            return None
        return math.ceil(self.n * (self.mde / abs(self.diff.mean)) ** 2)


def compare(
    baseline: EvalRun,
    candidate: EvalRun,
    *,
    confidence: float = 0.95,
    cluster: bool = True,
    power: float = 0.8,
) -> Comparison:
    """Paired comparison: is ``candidate`` really better or worse than ``baseline``?"""
    al = align(baseline, candidate)
    a, b = al.baseline, al.candidate
    clusters = a.clusters if cluster else None
    d = b.scores - a.scores

    est_a = estimate_mean(a.scores, clusters=clusters, confidence=confidence)
    est_b = estimate_mean(b.scores, clusters=clusters, confidence=confidence)
    diff = estimate_mean(d, clusters=clusters, confidence=confidence)

    if diff.se > 0 and not math.isnan(diff.se):
        t_stat = diff.mean / diff.se
        p_value = t_sf_two_sided(t_stat, diff.df)
    elif diff.mean == 0 or math.isnan(diff.se):
        t_stat, p_value = 0.0, 1.0
    else:  # every question moved by the same nonzero amount
        t_stat, p_value = math.copysign(math.inf, diff.mean), 0.0

    unpaired = math.hypot(est_a.se, est_b.se)
    sa, sb = float(np.std(a.scores)), float(np.std(b.scores))
    corr = float(np.corrcoef(a.scores, b.scores)[0, 1]) if sa > 0 and sb > 0 else 0.0

    if math.isnan(diff.se):
        mde = math.nan
    else:
        mde = (critical_value(confidence, diff.df) + norm_ppf(power)) * diff.se

    warnings: list[str] = []
    if al.dropped_baseline or al.dropped_candidate:
        warnings.append(
            f"{al.dropped_baseline} baseline and {al.dropped_candidate} candidate questions "
            "had no match and were excluded"
        )
    if diff.n < SMALL_N:
        warnings.append(
            f"only {diff.n} paired questions: consider --bootstrap to check the t approximation"
        )
    if (
        diff.method == "clustered"
        and diff.n_clusters is not None
        and diff.n_clusters < FEW_CLUSTERS
    ):
        warnings.append(
            f"only {diff.n_clusters} clusters: cluster-robust standard errors are biased "
            "downward with few clusters"
        )
    return Comparison(
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        baseline=est_a,
        candidate=est_b,
        diff=diff,
        t_stat=t_stat,
        p_value=p_value,
        unpaired_se=unpaired,
        correlation=corr,
        wins=int(np.sum(d > 0)),
        losses=int(np.sum(d < 0)),
        ties=int(np.sum(d == 0)),
        power=power,
        mde=mde,
        dropped_baseline=al.dropped_baseline,
        dropped_candidate=al.dropped_candidate,
        warnings=warnings,
    )
