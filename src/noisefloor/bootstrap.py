"""Nonparametric bootstrap as an assumption-light cross-check.

The analytic intervals in :mod:`noisefloor.stats` rely on the central limit
theorem. For small evals, heavily skewed scores, or few clusters, the
bootstrap is a useful second opinion. When clusters are present, whole
clusters are resampled (the *cluster bootstrap*), preserving within-cluster
correlation.

Implementation note: instead of materialising resampled index arrays, each
replicate draws multinomial *weights* over clusters. The statistic for a
replicate is then ``(w @ cluster_sums) / (w @ cluster_sizes)``, which handles
unequal cluster sizes correctly and vectorises well. Replicates are processed
in chunks to bound memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import EvalRun, FloatArray, align

_MAX_CELLS = 4_000_000


@dataclass(frozen=True)
class BootstrapResult:
    statistic: float
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float
    n_resamples: int
    method: Literal["question", "cluster"]
    seed: int | None


def _replicates(
    values: FloatArray,
    clusters: tuple[str, ...] | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, Literal["question", "cluster"]]:
    if clusters is not None:
        _, inverse = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
        sums = np.bincount(inverse, weights=values)
        sizes = np.bincount(inverse).astype(np.float64)
        method: Literal["question", "cluster"] = "cluster"
    else:
        sums = values.astype(np.float64)
        sizes = np.ones_like(sums)
        method = "question"
    g = sums.shape[0]
    probs = np.full(g, 1.0 / g)
    out = np.empty(n_resamples, dtype=np.float64)
    chunk = max(1, _MAX_CELLS // g)
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        w = rng.multinomial(g, probs, size=stop - start).astype(np.float64)
        out[start:stop] = (w @ sums) / (w @ sizes)
    return out, method


def _summarise(
    stat: float,
    reps: FloatArray,
    *,
    null: float | None,
    confidence: float,
    n_resamples: int,
    method: Literal["question", "cluster"],
    seed: int | None,
) -> BootstrapResult:
    alpha = 1.0 - confidence
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    if null is None:
        p = float("nan")
    else:
        # Centred bootstrap test: how often does the resampled statistic stray
        # from the estimate by at least as much as the estimate strays from the null?
        extreme = np.abs(reps - stat) >= abs(stat - null) - 1e-12
        p = float((1 + np.sum(extreme)) / (n_resamples + 1))
    return BootstrapResult(
        statistic=stat,
        ci_low=float(lo),
        ci_high=float(hi),
        confidence=confidence,
        p_value=min(1.0, p),
        n_resamples=n_resamples,
        method=method,
        seed=seed,
    )


def bootstrap_mean(
    run: EvalRun,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    cluster: bool = True,
    seed: int | None = 0,
) -> BootstrapResult:
    """Percentile bootstrap interval for a run's mean score."""
    if n_resamples < 100:
        raise ValueError("use at least 100 bootstrap resamples")
    rng = np.random.default_rng(seed)
    clusters = run.clusters if cluster else None
    reps, method = _replicates(run.scores, clusters, n_resamples, rng)
    return _summarise(
        run.mean,
        reps,
        null=None,
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
        seed=seed,
    )


def bootstrap_diff(
    baseline: EvalRun,
    candidate: EvalRun,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    cluster: bool = True,
    seed: int | None = 0,
) -> BootstrapResult:
    """Paired bootstrap for ``mean(candidate) - mean(baseline)`` with a p-value."""
    if n_resamples < 100:
        raise ValueError("use at least 100 bootstrap resamples")
    al = align(baseline, candidate)
    d = al.candidate.scores - al.baseline.scores
    rng = np.random.default_rng(seed)
    clusters = al.baseline.clusters if cluster else None
    reps, method = _replicates(d, clusters, n_resamples, rng)
    return _summarise(
        float(np.mean(d)),
        reps,
        null=0.0,
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
        seed=seed,
    )
