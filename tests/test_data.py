import json
from pathlib import Path

import numpy as np
import pytest

from noisefloor.data import DataError, EvalRun, align, from_records, load


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_autodetects_fields_and_groups_samples() -> None:
    rows = [
        {"question_id": "q1", "correct": True},
        {"question_id": "q1", "correct": False},
        {"question_id": "q2", "correct": True},
        {"question_id": "q1", "correct": True},
    ]
    run = from_records(rows)
    assert run.ids == ("q1", "q2")
    np.testing.assert_allclose(run.scores, [2 / 3, 1.0])
    assert run.n_samples.tolist() == [3, 1]
    assert run.within_var[0] == pytest.approx(np.var([1, 0, 1], ddof=1))
    assert np.isnan(run.within_var[1])
    assert run.clusters is None


def test_string_and_bool_scores() -> None:
    rows = [
        {"id": 1, "score": "PASS"},
        {"id": 2, "score": "fail"},
        {"id": 3, "score": "0.25"},
        {"id": 4, "score": 1},
    ]
    assert from_records(rows).scores.tolist() == [1.0, 0.0, 0.25, 1.0]


def test_nested_fields_and_explicit_names() -> None:
    rows = [
        {"meta": {"uid": "a"}, "metrics": {"acc": 0.5}},
        {"meta": {"uid": "b"}, "metrics": {"acc": 1}},
    ]
    run = from_records(rows, id_field="meta.uid", score_field="metrics.acc")
    assert run.ids == ("a", "b")
    assert run.scores.tolist() == [0.5, 1.0]


def test_clusters_must_be_consistent() -> None:
    rows = [
        {"id": "q1", "score": 1, "cluster": "p1"},
        {"id": "q1", "score": 0, "cluster": "p2"},
    ]
    with pytest.raises(DataError, match="appears in clusters"):
        from_records(rows)


def test_missing_score_field_is_actionable() -> None:
    with pytest.raises(DataError, match="--score-field"):
        from_records([{"id": 1, "value": 3}])


def test_bad_scores_rejected() -> None:
    with pytest.raises(DataError, match="cannot interpret"):
        from_records([{"id": 1, "score": "maybe"}])
    with pytest.raises(DataError, match="finite"):
        from_records([{"id": 1, "score": float("nan")}])


def test_load_jsonl_csv_json(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "a.jsonl", [{"id": "x", "score": 1}, {"id": "y", "score": 0}])
    (tmp_path / "b.csv").write_text("id,score,cluster\nx,1,c1\ny,0,c1\n", encoding="utf-8")
    (tmp_path / "c.json").write_text(
        json.dumps({"results": [{"id": "x", "score": 1}, {"id": "y", "score": 0}]}),
        encoding="utf-8",
    )
    a = load(tmp_path / "a.jsonl")
    b = load(tmp_path / "b.csv")
    c = load(tmp_path / "c.json")
    assert a.name == "a"
    assert a.scores.tolist() == b.scores.tolist() == c.scores.tolist() == [1.0, 0.0]
    assert b.clusters == ("c1", "c1")


def test_invalid_jsonl_reports_line(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text('{"id": 1, "score": 1}\n{oops\n', encoding="utf-8")
    with pytest.raises(DataError, match="line 2"):
        load(tmp_path / "bad.jsonl")


def test_align_pairs_on_ids_and_counts_drops() -> None:
    a = EvalRun.from_scores([1, 0, 1], ids=["q1", "q2", "q3"], name="a")
    b = EvalRun.from_scores([0, 1, 1, 1], ids=["q3", "q2", "q4", "q1"], name="b")
    al = align(a, b)
    assert al.baseline.ids == al.candidate.ids == ("q1", "q2", "q3")
    assert al.candidate.scores.tolist() == [1.0, 1.0, 0.0]
    assert al.dropped_baseline == 0
    assert al.dropped_candidate == 1


def test_align_no_overlap() -> None:
    a = EvalRun.from_scores([1], ids=["x"])
    b = EvalRun.from_scores([1], ids=["y"])
    with pytest.raises(DataError, match="share no question ids"):
        align(a, b)


def test_align_propagates_clusters() -> None:
    a = EvalRun.from_scores([1, 0], ids=["q1", "q2"], clusters=["c", "d"])
    b = EvalRun.from_scores([1, 1], ids=["q1", "q2"])
    al = align(a, b)
    assert al.candidate.clusters == ("c", "d")


def test_duplicate_ids_rejected_in_constructor() -> None:
    with pytest.raises(DataError, match="unique"):
        EvalRun.from_scores([1, 0], ids=["q", "q"])
