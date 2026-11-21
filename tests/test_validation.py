"""Monte Carlo checks of the statistical guarantees (see examples/validate.py).

Tolerances are about 3.5 binomial standard errors of the simulated rate, so a
correct implementation fails these by chance well under 1 time in 1000.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SIMS = 800


def _load_validate() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "examples" / "validate.py"
    spec = importlib.util.spec_from_file_location("nf_validate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load_validate()


def tol(rate: float, sims: int) -> float:
    return 3.5 * math.sqrt(rate * (1 - rate) / sims)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


def test_iid_coverage(rng: np.random.Generator) -> None:
    c = V.coverage_iid(SIMS, rng)
    assert abs(c.noisefloor - 0.95) < tol(0.95, SIMS)


def test_small_sample_t_interval(rng: np.random.Generator) -> None:
    c = V.coverage_small(SIMS, rng)
    assert abs(c.noisefloor - 0.95) < tol(0.95, SIMS)


def test_clustered_coverage_and_naive_failure(rng: np.random.Generator) -> None:
    c = V.coverage_clustered(SIMS, rng)
    assert abs(c.noisefloor - 0.95) < tol(0.95, SIMS) + 0.01
    assert c.shortcut is not None
    assert c.shortcut < 0.91  # ignoring clusters gives overconfident intervals


def test_false_alarm_rate(rng: np.random.Generator) -> None:
    c = V.false_alarm_rate(SIMS, rng)
    assert abs(c.noisefloor - 0.05) < tol(0.05, SIMS) + 0.01
    assert c.shortcut is not None
    assert c.shortcut > c.noisefloor


def test_planned_power(rng: np.random.Generator) -> None:
    sims = SIMS // 2
    c = V.power_check(sims, rng)
    assert abs(c.noisefloor - 0.80) < tol(0.80, sims)


def test_holm_controls_family_wise_error(rng: np.random.Generator) -> None:
    sims = 300
    c = V.fwer_check(sims, rng)
    assert c.noisefloor < 0.05 + tol(0.05, sims)
    assert c.shortcut is not None
    assert c.shortcut > 0.25  # 1 - 0.95**10 = 0.40 without correction
