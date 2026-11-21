"""Compare two models across a suite of benchmarks, with multiplicity control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .data import DataError, EvalRun, load
from .multiple import Correction, adjust
from .stats import Comparison, compare

SUFFIXES = (".jsonl", ".json", ".csv", ".tsv")


@dataclass(frozen=True)
class SuiteRow:
    benchmark: str
    comparison: Comparison
    p_adjusted: float
    significant: bool

    @property
    def direction(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "improvement" if self.comparison.diff.mean > 0 else "regression"


@dataclass(frozen=True)
class SuiteResult:
    rows: list[SuiteRow]
    correction: Correction
    alpha: float

    @property
    def improvements(self) -> list[SuiteRow]:
        return [r for r in self.rows if r.direction == "improvement"]

    @property
    def regressions(self) -> list[SuiteRow]:
        return [r for r in self.rows if r.direction == "regression"]

    @property
    def naive_significant(self) -> int:
        """How many benchmarks *look* significant before correction."""
        return sum(1 for r in self.rows if r.comparison.p_value < self.alpha)


def compare_suite(
    pairs: Mapping[str, tuple[EvalRun, EvalRun]],
    *,
    correction: Correction = "holm",
    alpha: float = 0.05,
    cluster: bool = True,
) -> SuiteResult:
    """Run a paired comparison per benchmark and adjust p-values across the suite."""
    if not pairs:
        raise DataError("suite is empty")
    names = sorted(pairs)
    comps = [
        compare(pairs[n][0], pairs[n][1], confidence=1 - alpha, cluster=cluster) for n in names
    ]
    adj = adjust([c.p_value for c in comps], correction)
    rows = [
        SuiteRow(benchmark=n, comparison=c, p_adjusted=p, significant=p < alpha)
        for n, c, p in zip(names, comps, adj, strict=True)
    ]
    return SuiteResult(rows=rows, correction=correction, alpha=alpha)


def _index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise DataError(f"{directory}: not a directory")
    found: dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in SUFFIXES and p.is_file():
            if p.stem in found:
                raise DataError(f"{directory}: two files for benchmark {p.stem!r}")
            found[p.stem] = p
    return found


def load_suite(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    **load_kwargs: str | None,
) -> tuple[dict[str, tuple[EvalRun, EvalRun]], list[str]]:
    """Pair benchmark files by stem across two directories.

    Returns the pairs and a list of benchmark names present in only one side.
    """
    base = _index(Path(baseline_dir))
    cand = _index(Path(candidate_dir))
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise DataError("no benchmark files with matching names in the two directories")
    unmatched = sorted(set(base) ^ set(cand))
    pairs = {
        name: (load(base[name], **load_kwargs), load(cand[name], **load_kwargs)) for name in shared
    }
    return pairs, unmatched
