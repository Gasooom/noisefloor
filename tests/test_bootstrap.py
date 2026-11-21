import numpy as np
import pytest

from noisefloor.bootstrap import bootstrap_diff, bootstrap_mean
from noisefloor.data import EvalRun
from noisefloor.stats import compare, summarize


def test_bootstrap_mean_agrees_with_clt_for_large_n() -> None:
    rng = np.random.default_rng(3)
    run = EvalRun.from_scores((rng.random(2000) < 0.7).astype(float))
    b = bootstrap_mean(run, n_resamples=4000, seed=1)
    s = summarize(run).estimate
    assert b.method == "question"
    assert b.statistic == pytest.approx(s.mean)
    assert b.ci_low == pytest.approx(s.ci_low, abs=0.004)
    assert b.ci_high == pytest.approx(s.ci_high, abs=0.004)


def test_bootstrap_is_reproducible_with_seed() -> None:
    run = EvalRun.from_scores([0, 1, 1, 0, 1] * 20)
    a = bootstrap_mean(run, n_resamples=500, seed=123)
    b = bootstrap_mean(run, n_resamples=500, seed=123)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_cluster_bootstrap_is_wider_for_correlated_clusters() -> None:
    scores = [1, 1, 1, 1, 0, 0, 0, 0] * 25
    clusters = [f"c{i // 4}" for i in range(200)]
    run = EvalRun.from_scores(scores, clusters=clusters)
    clustered = bootstrap_mean(run, n_resamples=3000)
    naive = bootstrap_mean(run, n_resamples=3000, cluster=False)
    assert clustered.method == "cluster"
    assert (clustered.ci_high - clustered.ci_low) > 1.5 * (naive.ci_high - naive.ci_low)


def test_unequal_cluster_sizes_give_correct_point_estimate() -> None:
    scores = [1, 1, 1, 1, 1, 1, 0]  # cluster a has 6 questions, b has 1
    clusters = ["a"] * 6 + ["b"]
    run = EvalRun.from_scores(scores, clusters=clusters)
    b = bootstrap_mean(run, n_resamples=2000)
    assert b.statistic == pytest.approx(6 / 7)
    assert 0.0 <= b.ci_low <= b.ci_high <= 1.0


def test_bootstrap_diff_p_value_tracks_t_test() -> None:
    rng = np.random.default_rng(11)
    p = rng.beta(1, 1, size=400)
    a = (rng.random(400) < p).astype(float)
    b = (rng.random(400) < np.clip(p + 0.08, 0, 1)).astype(float)
    ra, rb = EvalRun.from_scores(a), EvalRun.from_scores(b)
    boot = bootstrap_diff(ra, rb, n_resamples=5000)
    t = compare(ra, rb)
    assert boot.statistic == pytest.approx(t.diff.mean)
    assert boot.ci_low == pytest.approx(t.diff.ci_low, abs=0.015)
    assert boot.ci_high == pytest.approx(t.diff.ci_high, abs=0.015)
    assert (boot.p_value < 0.05) == (t.p_value < 0.05)


def test_bootstrap_diff_null_effect_has_large_p() -> None:
    run = EvalRun.from_scores([0, 1] * 50)
    boot = bootstrap_diff(run, run, n_resamples=500)
    assert boot.p_value == 1.0


def test_too_few_resamples() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        bootstrap_mean(EvalRun.from_scores([0, 1]), n_resamples=10)
