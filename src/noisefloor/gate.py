"""CI gate: turn a paired comparison into a pass/fail decision.

Three policies, from lenient to strict:

``fail_on_regression`` (default)
    Fail only when the candidate is *significantly* worse. Noise never blocks
    a merge, but a real regression always does.

``max_regression=m`` (non-inferiority)
    Fail unless we can rule out a regression larger than ``m``, i.e. unless
    the lower confidence bound of the difference is above ``-m``. This is the
    honest question for "this refactor should not hurt quality": absence of
    evidence of a regression is not evidence of its absence, and a small eval
    *will* fail this gate -- which is the point.

``require_improvement``
    Fail unless the candidate is significantly better. For changes whose
    whole justification is a quality gain.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import Comparison


@dataclass(frozen=True)
class GatePolicy:
    fail_on_regression: bool = True
    max_regression: float | None = None
    require_improvement: bool = False

    def __post_init__(self) -> None:
        if self.max_regression is not None and self.max_regression < 0:
            raise ValueError("max_regression must be non-negative")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]
    comparison: Comparison
    policy: GatePolicy

    @property
    def headline(self) -> str:
        return "PASS" if self.passed else "FAIL"


def evaluate_gate(comparison: Comparison, policy: GatePolicy | None = None) -> GateResult:
    policy = policy or GatePolicy()
    c = comparison
    reasons: list[str] = []
    passed = True
    level = f"{c.diff.confidence:.0%}"

    if policy.fail_on_regression and c.verdict == "regression":
        passed = False
        reasons.append(
            f"candidate is significantly worse than baseline ({level} CI entirely below zero)"
        )
    if policy.max_regression is not None and not c.diff.ci_low >= -policy.max_regression:
        passed = False
        reasons.append(
            f"cannot rule out a regression larger than {policy.max_regression:g} "
            f"(lower {level} bound is {c.diff.ci_low:+.4g}); "
            f"this eval can only rule out regressions larger than {-c.diff.ci_low:.4g}"
        )
    if policy.require_improvement and c.verdict != "improvement":
        passed = False
        reasons.append("candidate is not significantly better than baseline")
    if passed:
        if c.verdict == "improvement":
            reasons.append("candidate is significantly better than baseline")
        elif c.verdict == "regression":
            reasons.append("candidate is significantly worse, but the policy allows it")
        else:
            reasons.append("no significant difference detected")
    return GateResult(passed=passed, reasons=reasons, comparison=c, policy=policy)
