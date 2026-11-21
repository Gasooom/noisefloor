import json
from pathlib import Path

import numpy as np
import pytest

from noisefloor.data import DataError, EvalRun
from noisefloor.multiple import adjust, benjamini_hochberg, bonferroni, holm
from noisefloor.suite import compare_suite, load_suite

P = [0.01, 0.04, 0.03, 0.005]


def test_holm_known_values() -> None:
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> cummax -> .02 .03 .06 .06
    assert holm(P) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_bh_known_values() -> None:
    # sorted p: .005 .01 .03 .04 -> *4/rank: .02 .02 .04 .04 -> cummin from top
    assert benjamini_hochberg(P) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bonferroni_and_none() -> None:
    assert bonferroni(P) == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert adjust(P, "none") == pytest.approx(P)


def test_ordering_invariants() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(50).tolist()
    h, b, bf = holm(p), benjamini_hochberg(p), bonferroni(p)
    eps = 1e-12
    for raw, hh, bb, ff in zip(p, h, b, bf, strict=True):
        assert raw <= bb + eps
        assert bb <= hh + eps
        assert hh <= ff + eps


def test_empty_and_invalid() -> None:
    assert holm([]) == []
    assert benjamini_hochberg([]) == []
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm([1.5])
    with pytest.raises(ValueError, match="unknown correction"):
        adjust([0.1], "fdr")  # type: ignore[arg-type]


def _pair(effect: float, seed: int, n: int = 300) -> tuple[EvalRun, EvalRun]:
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 1, n)
    a = (rng.random(n) < p).astype(float)
    b = (rng.random(n) < np.clip(p + effect, 0, 1)).astype(float)
    return EvalRun.from_scores(a), EvalRun.from_scores(b)


def test_suite_controls_false_positives() -> None:
    # 20 null benchmarks + one real effect
    pairs = {f"null{i:02d}": _pair(0.0, i) for i in range(20)}
    pairs["real"] = _pair(0.15, 999)
    res = compare_suite(pairs, correction="holm")
    assert [r.benchmark for r in res.improvements] == ["real"]
    assert not res.regressions
    assert res.naive_significant >= len(res.improvements)
    row = next(r for r in res.rows if r.benchmark == "real")
    assert row.p_adjusted >= row.comparison.p_value


def test_load_suite(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "cand").mkdir()
    for side, score in (("base", 0), ("cand", 1)):
        for bench in ("mmlu", "gsm8k"):
            rows = [{"id": i, "score": score} for i in range(5)]
            (tmp_path / side / f"{bench}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
            )
    (tmp_path / "base" / "only_here.jsonl").write_text('{"id": 1, "score": 1}', encoding="utf-8")
    pairs, unmatched = load_suite(tmp_path / "base", tmp_path / "cand")
    assert sorted(pairs) == ["gsm8k", "mmlu"]
    assert unmatched == ["only_here"]
    with pytest.raises(DataError, match="not a directory"):
        load_suite(tmp_path / "missing", tmp_path / "cand")
