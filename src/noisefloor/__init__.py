"""noisefloor: is your eval improvement real, or just noise?

Statistical rigor for LLM evaluations -- confidence intervals, cluster-robust
standard errors, paired comparisons, power analysis, multiple-comparison
control, leaderboard tiers, and a CI gate.

Quick start::

    import noisefloor as nf

    base = nf.load("baseline.jsonl")
    cand = nf.load("candidate.jsonl")
    result = nf.compare(base, cand)
    print(result.verdict, result.diff.mean, (result.diff.ci_low, result.diff.ci_high))
"""

from .bootstrap import BootstrapResult, bootstrap_diff, bootstrap_mean
from .data import DataError, EvalRun, align, from_records, load
from .gate import GatePolicy, GateResult, evaluate_gate
from .leaderboard import Leaderboard, from_runs, from_summary
from .multiple import adjust, benjamini_hochberg, bonferroni, holm
from .power import (
    achieved_power,
    design_effect,
    minimum_detectable_effect,
    plan_from_comparison,
    sample_size,
    sample_size_for_margin,
)
from .stats import Comparison, Estimate, Summary, compare, estimate_mean, summarize
from .suite import SuiteResult, compare_suite, load_suite

__version__ = "0.1.0"

__all__ = [
    "BootstrapResult",
    "Comparison",
    "DataError",
    "Estimate",
    "EvalRun",
    "GatePolicy",
    "GateResult",
    "Leaderboard",
    "SuiteResult",
    "Summary",
    "__version__",
    "achieved_power",
    "adjust",
    "align",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_diff",
    "bootstrap_mean",
    "compare",
    "compare_suite",
    "design_effect",
    "estimate_mean",
    "evaluate_gate",
    "from_records",
    "from_runs",
    "from_summary",
    "holm",
    "load",
    "load_suite",
    "minimum_detectable_effect",
    "plan_from_comparison",
    "sample_size",
    "sample_size_for_margin",
    "summarize",
]
