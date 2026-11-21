"""Leaderboards with honest uncertainty: rank intervals and statistical tiers.

A leaderboard sorted by point estimate implies an ordering that the data often
cannot support. This module reports, for each model:

* a confidence interval on its *rank* (via bootstrap), and
* a *tier*: models within a tier cannot be distinguished from the tier's
  leader at the chosen significance level.

Two input modes:

* **per-question runs** (preferred): all models are resampled jointly over the
  shared questions, so rank intervals respect the paired structure, and tiers
  use paired tests.
* **summary scores** (score + n per model, e.g. scraped from a public
  leaderboard): models are treated as independent binomial estimates. This is
  conservative when the models actually share questions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._dist import critical_value, norm_cdf, norm_ppf
from .data import DataError, EvalRun
from .stats import Estimate, compare, estimate_mean

_MAX_CELLS = 4_000_000


@dataclass(frozen=True)
class Entry:
    name: str
    estimate: Estimate
    rank: int
    rank_low: int
    rank_high: int
    tier: int
    #: p-value of the test against this model's tier leader (None for leaders).
    p_vs_leader: float | None


@dataclass(frozen=True)
class Leaderboard:
    entries: list[Entry]
    mode: Literal["paired", "independent"]
    n_questions: int | None
    confidence: float
    n_resamples: int

    @property
    def n_tiers(self) -> int:
        return max(e.tier for e in self.entries)

    def tier(self, k: int) -> list[Entry]:
        return [e for e in self.entries if e.tier == k]


def _rank_matrix(means: np.ndarray) -> np.ndarray:
    """Competition ranks (1 = best) per row of a (B, M) matrix of means."""
    # rank = 1 + number of models strictly better
    return 1 + np.sum(means[:, None, :] > means[:, :, None], axis=2)


def _rank_intervals(reps: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    ranks = _rank_matrix(reps)
    alpha = 1.0 - confidence
    lo = np.floor(np.quantile(ranks, alpha / 2.0, axis=0, method="lower")).astype(int)
    hi = np.ceil(np.quantile(ranks, 1.0 - alpha / 2.0, axis=0, method="higher")).astype(int)
    return lo, hi


def from_runs(
    runs: Sequence[EvalRun],
    *,
    confidence: float = 0.95,
    cluster: bool = True,
    n_resamples: int = 2_000,
    seed: int | None = 0,
) -> Leaderboard:
    """Leaderboard from per-question results, restricted to shared questions."""
    if len(runs) < 2:
        raise DataError("a leaderboard needs at least two runs")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise DataError("run names must be unique")
    shared = [q for q in runs[0].ids if all(q in r._index for r in runs[1:])]
    if len(shared) < 2:
        raise DataError("runs share fewer than two questions")
    aligned = [r.subset(shared) for r in runs]
    clusters = next((r.clusters for r in aligned if r.clusters is not None), None)
    if not cluster:
        clusters = None
    aligned = [
        EvalRun(r.name, r.ids, r.scores, r.n_samples, r.within_var, clusters) for r in aligned
    ]

    scores = np.stack([r.scores for r in aligned], axis=1)  # (n, M)
    if clusters is not None:
        labels, inverse = np.unique(np.asarray(clusters, dtype=object), return_inverse=True)
        g = len(labels)
        sums = np.zeros((g, scores.shape[1]))
        np.add.at(sums, inverse, scores)
        sizes = np.bincount(inverse).astype(np.float64)
    else:
        sums, sizes = scores, np.ones(scores.shape[0])
        g = scores.shape[0]

    rng = np.random.default_rng(seed)
    reps = np.empty((n_resamples, scores.shape[1]))
    chunk = max(1, _MAX_CELLS // g)
    probs = np.full(g, 1.0 / g)
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        w = rng.multinomial(g, probs, size=stop - start).astype(np.float64)
        reps[start:stop] = (w @ sums) / (w @ sizes)[:, None]

    estimates = [estimate_mean(r.scores, clusters=clusters, confidence=confidence) for r in aligned]
    lo, hi = _rank_intervals(reps, confidence)
    order = sorted(range(len(runs)), key=lambda i: -estimates[i].mean)

    tiers: dict[int, int] = {}
    p_vs: dict[int, float | None] = {}
    leader = order[0]
    tier = 1
    tiers[leader], p_vs[leader] = 1, None
    for i in order[1:]:
        c = compare(aligned[leader], aligned[i], confidence=confidence, cluster=cluster)
        if c.diff.ci_high < 0:  # significantly worse than the tier leader
            tier += 1
            leader = i
            tiers[i], p_vs[i] = tier, None
        else:
            tiers[i], p_vs[i] = tier, c.p_value

    return _assemble(
        names,
        estimates,
        order,
        lo,
        hi,
        tiers,
        p_vs,
        mode="paired",
        n_questions=len(shared),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def from_summary(
    rows: Sequence[tuple[str, float, int]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int | None = 0,
) -> Leaderboard:
    """Leaderboard from ``(name, score, n)`` triples, assuming binary scores.

    ``score`` may be a fraction (0.734) or a percentage (73.4).
    """
    if len(rows) < 2:
        raise DataError("a leaderboard needs at least two models")
    names, means, ses, ns = [], [], [], []
    as_percent = any(s > 1.0 for _, s, _ in rows)
    for name, score, n in rows:
        p = score / 100.0 if as_percent else score
        if not 0.0 <= p <= 1.0:
            raise DataError(f"{name}: score {score} is not a valid accuracy")
        if n < 2:
            raise DataError(f"{name}: n must be at least 2")
        names.append(name)
        means.append(p)
        # Floor the variance so saturated scores (0% / 100%) still carry uncertainty.
        ses.append(math.sqrt(max(p * (1 - p), 0.25 / n) / n))
        ns.append(n)
    if len(set(names)) != len(names):
        raise DataError("model names must be unique")

    z = critical_value(confidence, None)
    estimates = [
        Estimate(
            mean=m,
            se=s,
            ci_low=max(0.0, m - z * s),
            ci_high=min(1.0, m + z * s),
            confidence=confidence,
            n=n,
            df=math.inf,
            method="iid",
        )
        for m, s, n in zip(means, ses, ns, strict=True)
    ]
    rng = np.random.default_rng(seed)
    reps = rng.normal(np.array(means), np.array(ses), size=(n_resamples, len(rows)))
    lo, hi = _rank_intervals(reps, confidence)
    order = sorted(range(len(rows)), key=lambda i: -means[i])

    tiers: dict[int, int] = {}
    p_vs: dict[int, float | None] = {}
    leader = order[0]
    tier = 1
    tiers[leader], p_vs[leader] = 1, None
    z_crit = norm_ppf(1.0 - (1.0 - confidence) / 2.0)
    for i in order[1:]:
        se = math.hypot(ses[leader], ses[i])
        zstat = (means[leader] - means[i]) / se
        if zstat > z_crit:
            tier += 1
            leader = i
            tiers[i], p_vs[i] = tier, None
        else:
            tiers[i], p_vs[i] = tier, 2.0 * (1.0 - norm_cdf(abs(zstat)))

    common_n = ns[0] if len(set(ns)) == 1 else None
    return _assemble(
        names,
        estimates,
        order,
        lo,
        hi,
        tiers,
        p_vs,
        mode="independent",
        n_questions=common_n,
        confidence=confidence,
        n_resamples=n_resamples,
    )


def _assemble(
    names: list[str],
    estimates: list[Estimate],
    order: list[int],
    lo: np.ndarray,
    hi: np.ndarray,
    tiers: dict[int, int],
    p_vs: dict[int, float | None],
    *,
    mode: Literal["paired", "independent"],
    n_questions: int | None,
    confidence: float,
    n_resamples: int,
) -> Leaderboard:
    entries = []
    for rank, i in enumerate(order, start=1):
        entries.append(
            Entry(
                name=names[i],
                estimate=estimates[i],
                rank=rank,
                rank_low=int(min(lo[i], rank)),
                rank_high=int(max(hi[i], rank)),
                tier=tiers[i],
                p_vs_leader=p_vs[i],
            )
        )
    return Leaderboard(
        entries=entries,
        mode=mode,
        n_questions=n_questions,
        confidence=confidence,
        n_resamples=n_resamples,
    )
