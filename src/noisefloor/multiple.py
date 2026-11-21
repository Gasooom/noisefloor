"""Multiple-comparison corrections.

Compare two models on 20 benchmarks at alpha = 0.05 and, even if they are
identical, you expect one "significant" win. Release notes that cherry-pick
the benchmarks that moved are a statistical artefact factory. Correct for it.

* ``holm``       -- controls the family-wise error rate (any false positive);
                    uniformly more powerful than Bonferroni. Use for ship gates.
* ``bh``         -- Benjamini-Hochberg; controls the false discovery rate.
                    Use for exploratory sweeps over many benchmarks/slices.
* ``bonferroni`` -- simplest, most conservative.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

Correction = Literal["holm", "bh", "bonferroni", "none"]
CORRECTIONS: tuple[Correction, ...] = ("holm", "bh", "bonferroni", "none")


def _validate(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if np.any((p < 0) | (p > 1) | np.isnan(p)):
        raise ValueError("pvalues must be in [0, 1]")
    return p


def bonferroni(pvalues: Sequence[float]) -> list[float]:
    p = _validate(pvalues)
    return np.minimum(1.0, p * p.size).tolist()  # type: ignore[no-any-return]


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values."""
    p = _validate(pvalues)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    out = np.empty(m)
    out[order] = adj_sorted
    return out.tolist()  # type: ignore[no-any-return]


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg (1995) adjusted p-values (q-values)."""
    p = _validate(pvalues)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    out = np.empty(m)
    out[order] = adj_sorted
    return out.tolist()  # type: ignore[no-any-return]


def adjust(pvalues: Sequence[float], method: Correction = "holm") -> list[float]:
    if method == "holm":
        return holm(pvalues)
    if method == "bh":
        return benjamini_hochberg(pvalues)
    if method == "bonferroni":
        return bonferroni(pvalues)
    if method == "none":
        return _validate(pvalues).tolist()  # type: ignore[no-any-return]
    raise ValueError(f"unknown correction {method!r}; choose from {', '.join(CORRECTIONS)}")
