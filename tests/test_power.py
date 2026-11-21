import math

import numpy as np
import pytest

from noisefloor.data import EvalRun
from noisefloor.power import (
    achieved_power,
    design_effect,
    minimum_detectable_effect,
    paired_variance,
    plan_from_comparison,
    sample_size,
    sample_size_for_margin,
)
from noisefloor.stats import compare


def test_sample_size_textbook_value() -> None:
    # Classic result: detecting d = 0.5 SD at alpha=.05, power=.8 needs
    # (1.95996 + 0.84162)^2 / 0.25 = 31.4 -> 32 observations.
    assert sample_size(0.5, sd=1.0) == 32


def test_binary_unpaired_matches_two_proportion_formula() -> None:
    p1, p2 = 0.70, 0.72
    z = 1.959964 + 0.841621
    expected = math.ceil(z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / 0.02**2)
    assert sample_size(0.02, baseline_rate=0.70) == expected
    assert expected > 8000  # a 2pp change on a 70% eval needs a big eval


def test_correlation_reduces_required_n() -> None:
    n0 = sample_size(0.03, baseline_rate=0.6, correlation=0.0)
    n5 = sample_size(0.03, baseline_rate=0.6, correlation=0.5)
    n8 = sample_size(0.03, baseline_rate=0.6, correlation=0.8)
    assert n0 > n5 > n8
    assert n5 == pytest.approx(n0 / 2, rel=0.05)


def test_design_effect() -> None:
    assert design_effect(1, 0.9) == 1.0
    assert design_effect(5, 0.25) == 2.0
    with pytest.raises(ValueError, match="cluster_size"):
        design_effect(0.5, 0.1)
    base = sample_size(0.05, baseline_rate=0.5)
    assert sample_size(0.05, baseline_rate=0.5, design_effect=2.0) == pytest.approx(2 * base, abs=1)


def test_mde_inverts_sample_size() -> None:
    for rate in (0.3, 0.5, 0.85):
        n = sample_size(0.04, baseline_rate=rate, correlation=0.4)
        mde = minimum_detectable_effect(n, baseline_rate=rate, correlation=0.4)
        assert mde == pytest.approx(0.04, rel=0.01)
        assert mde <= 0.04


def test_mde_with_sd_and_infinite_case() -> None:
    assert minimum_detectable_effect(32, sd=1.0) == pytest.approx(0.4953, abs=1e-3)
    assert minimum_detectable_effect(3, baseline_rate=0.99) == math.inf


def test_achieved_power_round_trip() -> None:
    n = sample_size(0.05, baseline_rate=0.5)
    assert achieved_power(n, 0.05, baseline_rate=0.5) == pytest.approx(0.8, abs=0.01)
    assert achieved_power(n, 0.0, sd=0.5) == pytest.approx(0.05, abs=1e-9)
    assert achieved_power(100, 0.02, baseline_rate=0.7) < 0.15


def test_margin_sizes() -> None:
    # +-3pp at 50% accuracy is the familiar ~1068 respondents of polling
    assert sample_size_for_margin(0.03) == 1068


def test_paired_variance_validation() -> None:
    assert paired_variance(0.5, 0.5, 1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="baseline_rate"):
        paired_variance(1.2, 0.5)
    with pytest.raises(ValueError, match="sd"):
        sample_size(0.1)
    with pytest.raises(ValueError, match="nonzero"):
        sample_size(0.0, sd=1)


def test_plan_from_pilot_comparison() -> None:
    rng = np.random.default_rng(5)
    p = rng.beta(1, 1, 300)
    a = (rng.random(300) < p).astype(float)
    b = (rng.random(300) < np.clip(p + 0.02, 0, 1)).astype(float)
    c = compare(EvalRun.from_scores(a), EvalRun.from_scores(b))
    plan = plan_from_comparison(c, 0.02)
    assert plan.n_current == 300
    assert plan.n_required > 300
    assert 0.05 < plan.power_current < 0.8
    assert plan.sd_diff_effective == pytest.approx(c.diff.se * math.sqrt(300))
