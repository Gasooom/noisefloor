import numpy as np
import pytest

from noisefloor.data import DataError, EvalRun
from noisefloor.leaderboard import from_runs, from_summary


def _runs(effects: list[float], n: int = 400, seed: int = 0) -> list[EvalRun]:
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 1, n)
    out = []
    for i, e in enumerate(effects):
        s = (rng.random(n) < np.clip(p + e, 0, 1)).astype(float)
        out.append(EvalRun.from_scores(s, name=f"m{i}"))
    return out


def test_well_separated_models_get_distinct_tiers() -> None:
    lb = from_runs(_runs([0.0, 0.25, 0.5]))
    assert [e.name for e in lb.entries] == ["m2", "m1", "m0"]
    assert [e.tier for e in lb.entries] == [1, 2, 3]
    assert all(e.rank_low == e.rank_high == e.rank for e in lb.entries)
    assert lb.mode == "paired"
    assert lb.n_questions == 400


def test_near_identical_models_share_a_tier() -> None:
    lb = from_runs(_runs([0.0, 0.005, 0.01, 0.3]))
    assert lb.entries[0].name == "m3"
    rest = lb.entries[1:]
    assert len({e.tier for e in rest}) == 1
    # the noise-level models' rank intervals overlap
    assert any(e.rank_high - e.rank_low >= 1 for e in rest)
    assert lb.n_tiers == 2
    assert lb.entries[0].p_vs_leader is None
    assert all(e.p_vs_leader is not None for e in rest[1:])


def test_rank_intervals_contain_rank() -> None:
    lb = from_runs(_runs([0.0, 0.02, 0.04, 0.06, 0.08], n=150, seed=3))
    for e in lb.entries:
        assert e.rank_low <= e.rank <= e.rank_high


def test_from_runs_uses_shared_questions_only() -> None:
    a = EvalRun.from_scores([1, 0, 1, 1], ids=["a", "b", "c", "d"], name="A")
    b = EvalRun.from_scores([1, 1, 0], ids=["a", "b", "c"], name="B")
    lb = from_runs([a, b], n_resamples=200)
    assert lb.n_questions == 3


def test_from_runs_validation() -> None:
    r = EvalRun.from_scores([1, 0], name="x")
    with pytest.raises(DataError, match="at least two"):
        from_runs([r])
    with pytest.raises(DataError, match="unique"):
        from_runs([r, r])


def test_from_summary_ties_on_small_benchmark() -> None:
    # Four models on a 198-question benchmark, within a few points of each other.
    rows = [("alpha", 61.1, 198), ("beta", 59.6, 198), ("gamma", 58.1, 198), ("delta", 40.0, 198)]
    lb = from_summary(rows)
    assert lb.mode == "independent"
    assert [e.name for e in lb.entries] == ["alpha", "beta", "gamma", "delta"]
    assert [e.tier for e in lb.entries] == [1, 1, 1, 2]
    assert lb.entries[0].rank_high >= 2  # the "winner" is not clearly #1
    assert lb.entries[0].estimate.ci_low < lb.entries[2].estimate.mean


def test_from_summary_fraction_and_saturated_scores() -> None:
    lb = from_summary([("a", 1.0, 30), ("b", 0.9, 30)])
    assert lb.entries[0].estimate.se > 0
    assert lb.entries[0].estimate.ci_high == 1.0


def test_from_summary_validation() -> None:
    with pytest.raises(DataError, match="valid accuracy"):
        from_summary([("a", 50, 10), ("b", 120, 10)])
    with pytest.raises(DataError, match="n must"):
        from_summary([("a", 0.5, 1), ("b", 0.4, 10)])
