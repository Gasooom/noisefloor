import pytest

from noisefloor.data import EvalRun
from noisefloor.gate import GatePolicy, evaluate_gate
from noisefloor.stats import Comparison, compare

BASE = [0] * 60 + [1] * 40
BETTER = [1] * 30 + [0] * 30 + [1] * 40
WORSE = [0] * 60 + [1] * 10 + [0] * 30
NOISY = [1] + [0] * 59 + [1] * 39 + [0]


def _cmp(cand: list[int]) -> Comparison:
    return compare(EvalRun.from_scores(BASE), EvalRun.from_scores(cand))


def test_default_policy_blocks_only_real_regressions() -> None:
    assert evaluate_gate(_cmp(BETTER)).passed
    assert evaluate_gate(_cmp(NOISY)).passed
    res = evaluate_gate(_cmp(WORSE))
    assert not res.passed
    assert res.headline == "FAIL"
    assert "significantly worse" in res.reasons[0]


def test_non_inferiority_fails_small_noisy_evals() -> None:
    res = evaluate_gate(_cmp(NOISY), GatePolicy(max_regression=0.01))
    assert not res.passed
    assert "cannot rule out" in res.reasons[0]
    lenient = evaluate_gate(_cmp(NOISY), GatePolicy(max_regression=0.2))
    assert lenient.passed


def test_require_improvement() -> None:
    policy = GatePolicy(require_improvement=True)
    assert evaluate_gate(_cmp(BETTER), policy).passed
    assert not evaluate_gate(_cmp(NOISY), policy).passed


def test_allowing_regressions() -> None:
    res = evaluate_gate(_cmp(WORSE), GatePolicy(fail_on_regression=False))
    assert res.passed
    assert "policy allows" in res.reasons[0]


def test_invalid_policy() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        GatePolicy(max_regression=-0.1)
