import math

import numpy as np
import pytest

from noisefloor._dist import t_ppf
from noisefloor.data import EvalRun
from noisefloor.stats import compare, estimate_mean, summarize, wilson_interval


def test_iid_mean_matches_textbook() -> None:
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    est = estimate_mean(x)
    se = np.std(x, ddof=1) / math.sqrt(10)
    assert est.mean == pytest.approx(0.7)
    assert est.se == pytest.approx(se)
    assert est.df == 9
    assert est.method == "iid"
    assert est.margin == pytest.approx(t_ppf(0.975, 9) * se)
    assert est.contains(0.7)


def test_singleton_clusters_reduce_to_iid() -> None:
    rng = np.random.default_rng(0)
    x = rng.random(50)
    iid = estimate_mean(x)
    clustered = estimate_mean(x, clusters=tuple(str(i) for i in range(50)))
    assert clustered.method == "iid"
    assert clustered.se == pytest.approx(iid.se)


def test_clustered_se_hand_computed() -> None:
    # Two clusters of perfectly correlated questions.
    x = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    clusters = ("a", "a", "a", "b", "b", "b")
    est = estimate_mean(x, clusters=clusters)
    # residuals: +0.5 x3 in a, -0.5 x3 in b -> cluster sums +-1.5
    var = (2 / 1) * (1.5**2 + 1.5**2) / 36
    assert est.method == "clustered"
    assert est.n_clusters == 2
    assert est.df == 1
    assert est.se == pytest.approx(math.sqrt(var))
    # and it is much larger than the (wrong) iid SE
    assert est.se > 2 * estimate_mean(x).se


def test_wilson_interval_known_value() -> None:
    lo, hi = wilson_interval(8, 10, 0.95)
    assert lo == pytest.approx(0.4901625, abs=1e-6)
    assert hi == pytest.approx(0.9433178, abs=1e-6)
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo < 1.0


def test_summarize_flags_small_and_degenerate_runs() -> None:
    s = summarize(EvalRun.from_scores([1.0] * 12))
    assert any("same score" in w for w in s.warnings)
    assert any("only 12 questions" in w for w in s.warnings)
    assert s.wilson is not None
    assert s.wilson[0] < 1.0


def test_summarize_reports_design_effect() -> None:
    scores = [1, 1, 1, 1, 0, 0, 0, 0] * 10
    clusters = [f"c{i // 4}" for i in range(80)]
    s = summarize(EvalRun.from_scores(scores, clusters=clusters))
    assert s.naive is not None
    assert s.design_effect is not None
    assert s.design_effect > 3.0
    s_off = summarize(EvalRun.from_scores(scores, clusters=clusters), cluster=False)
    assert s_off.estimate.method == "iid"


def test_variance_components_recover_truth() -> None:
    rng = np.random.default_rng(42)
    n, k = 3000, 4
    p = rng.beta(2, 2, size=n)  # per-question true accuracy
    samples = rng.random((n, k)) < p[:, None]
    run = EvalRun.from_scores(samples.astype(float).tolist())
    s = summarize(run)
    vc = s.components
    assert vc is not None
    assert vc.mean_k == 4
    # Var(p) for Beta(2,2) = 1/20; E[p(1-p)] = 0.25 - 0.05 = 0.2
    assert vc.between == pytest.approx(0.05, abs=0.01)
    assert vc.within == pytest.approx(0.2, abs=0.01)
    assert vc.se_by_k[4] == pytest.approx(s.estimate.se, rel=0.05)
    assert vc.se_by_k[1] > vc.se_by_k[4] > vc.se_by_k[16] > vc.se_limit
    assert 0 < vc.within_share < 1


def test_no_components_without_resampling() -> None:
    assert summarize(EvalRun.from_scores([0, 1, 1, 0])).components is None


def test_compare_paired_matches_manual() -> None:
    a = EvalRun.from_scores([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], name="base")
    b = EvalRun.from_scores([1, 1, 1, 1, 0, 1, 1, 0, 1, 1], name="cand")
    c = compare(a, b)
    d = b.scores - a.scores
    se = np.std(d, ddof=1) / math.sqrt(10)
    assert c.diff.mean == pytest.approx(0.2)
    assert c.diff.se == pytest.approx(se)
    assert c.t_stat == pytest.approx(0.2 / se)
    assert (c.wins, c.losses, c.ties) == (2, 0, 8)
    assert c.baseline_name == "base"
    assert c.candidate_name == "cand"
    assert 0 < c.p_value < 1


def test_pairing_beats_unpaired_when_correlated() -> None:
    rng = np.random.default_rng(1)
    p = rng.beta(0.6, 0.6, size=500)  # strongly varying difficulty
    a = (rng.random(500) < p).astype(float)
    b = (rng.random(500) < np.clip(p + 0.05, 0, 1)).astype(float)
    c = compare(EvalRun.from_scores(a), EvalRun.from_scores(b))
    assert c.correlation > 0.3
    assert c.diff.se < c.unpaired_se
    assert c.variance_reduction > 0.2


def test_verdicts() -> None:
    base = [0] * 60 + [1] * 40
    better = [1] * 30 + [0] * 30 + [1] * 40
    worse = [0] * 60 + [1] * 10 + [0] * 30
    assert compare(EvalRun.from_scores(base), EvalRun.from_scores(better)).verdict == "improvement"
    assert compare(EvalRun.from_scores(base), EvalRun.from_scores(worse)).verdict == "regression"
    noisy = list(base)
    noisy[0], noisy[99] = 1, 0
    c = compare(EvalRun.from_scores(base), EvalRun.from_scores(noisy))
    assert c.verdict == "inconclusive"
    assert not c.significant
    assert c.n_required is None  # zero observed difference


def test_identical_runs() -> None:
    a = EvalRun.from_scores([1, 0, 1, 0])
    c = compare(a, a)
    assert c.p_value == 1.0
    assert c.verdict == "inconclusive"


def test_constant_shift_is_infinitely_significant() -> None:
    a = EvalRun.from_scores([0.1, 0.2, 0.3, 0.4] * 10)
    b = EvalRun.from_scores([0.2, 0.3, 0.4, 0.5] * 10)
    c = compare(a, b)
    assert c.p_value == 0.0
    assert c.verdict == "improvement"


def test_mde_and_required_n() -> None:
    rng = np.random.default_rng(7)
    a = (rng.random(200) < 0.6).astype(float)
    b = a.copy()
    flip = rng.choice(200, size=12, replace=False)
    b[flip[:8]] = 1.0
    b[flip[8:]] = 0.0
    c = compare(EvalRun.from_scores(a), EvalRun.from_scores(b))
    assert c.mde > 0
    if c.n_required is not None and abs(c.diff.mean) < c.mde:
        assert c.n_required > c.n


def test_compare_warns_on_dropped_questions() -> None:
    a = EvalRun.from_scores([1] * 40 + [0] * 5, ids=[str(i) for i in range(45)])
    b = EvalRun.from_scores([1] * 40, ids=[str(i) for i in range(40)])
    c = compare(a, b)
    assert c.dropped_baseline == 5
    assert any("no match" in w for w in c.warnings)


def test_clustered_comparison_uses_cluster_df() -> None:
    scores_a = [1, 1, 0, 0] * 25
    scores_b = [1, 1, 1, 0] * 25
    clusters = [f"p{i // 4}" for i in range(100)]
    c = compare(
        EvalRun.from_scores(scores_a, clusters=clusters),
        EvalRun.from_scores(scores_b, clusters=clusters),
    )
    assert c.diff.method == "clustered"
    assert c.diff.n_clusters == 25
    assert c.diff.df == 24
