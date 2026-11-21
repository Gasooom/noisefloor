"""Loading eval results into a canonical, per-question representation.

An eval run is a set of *questions*. Each question may be answered more than
once (K samples at temperature > 0), and questions may belong to *clusters*
(e.g. several questions about the same passage, or several unit tests for the
same function). Every statistic in noisefloor is computed from this
representation, so getting it right here matters:

* the unit of analysis is the question, not the row -- repeated samples are
  averaged per question, and their spread is kept as within-question variance;
* clusters are a property of a question and must be consistent across samples.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

ID_FIELDS = ("id", "question_id", "qid", "example_id", "task_id", "problem_id", "sample_id", "uid")
SCORE_FIELDS = ("score", "correct", "is_correct", "accuracy", "passed", "pass", "reward", "label")
CLUSTER_FIELDS = ("cluster", "cluster_id", "group", "group_id", "passage_id", "document_id")

_TRUE = {"true", "t", "yes", "y", "pass", "passed", "correct"}
_FALSE = {"false", "f", "no", "n", "fail", "failed", "incorrect"}


class DataError(ValueError):
    """Raised when eval data is malformed or ambiguous."""


@dataclass(frozen=True)
class EvalRun:
    """Per-question results of one eval run.

    Attributes:
        name: Human-readable label (usually the file stem).
        ids: Question identifiers, unique, in first-seen order.
        scores: Mean score per question across its samples.
        n_samples: Number of samples (K_i) per question.
        within_var: Unbiased variance across a question's samples (NaN if K_i < 2).
        clusters: Cluster label per question, or ``None`` when unclustered.
    """

    name: str
    ids: tuple[str, ...]
    scores: FloatArray
    n_samples: npt.NDArray[np.int64]
    within_var: FloatArray
    clusters: tuple[str, ...] | None = None
    _index: dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.ids)
        if n == 0:
            raise DataError(f"{self.name}: eval run has no questions")
        if len(set(self.ids)) != n:
            raise DataError(f"{self.name}: question ids must be unique")
        for label, arr in (
            ("scores", self.scores),
            ("n_samples", self.n_samples),
            ("within_var", self.within_var),
        ):
            if arr.shape != (n,):
                raise DataError(f"{self.name}: {label} has shape {arr.shape}, expected ({n},)")
        if not np.all(np.isfinite(self.scores)):
            raise DataError(f"{self.name}: scores must be finite")
        if self.clusters is not None and len(self.clusters) != n:
            raise DataError(f"{self.name}: clusters must have one label per question")
        object.__setattr__(self, "_index", {qid: i for i, qid in enumerate(self.ids)})

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def n_clusters(self) -> int | None:
        return None if self.clusters is None else len(set(self.clusters))

    @property
    def is_binary(self) -> bool:
        return bool(np.all((self.scores == 0.0) | (self.scores == 1.0)))

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    def index_of(self, qid: str) -> int:
        return self._index[qid]

    def subset(self, ids: Sequence[str]) -> EvalRun:
        """Return a new run restricted to ``ids`` (in that order)."""
        idx = np.fromiter((self._index[q] for q in ids), dtype=np.int64, count=len(ids))
        return EvalRun(
            name=self.name,
            ids=tuple(ids),
            scores=self.scores[idx],
            n_samples=self.n_samples[idx],
            within_var=self.within_var[idx],
            clusters=None if self.clusters is None else tuple(self.clusters[i] for i in idx),
        )

    @classmethod
    def from_scores(
        cls,
        scores: Iterable[float] | Iterable[Sequence[float]],
        *,
        name: str = "run",
        ids: Sequence[str] | None = None,
        clusters: Sequence[Any] | None = None,
    ) -> EvalRun:
        """Build a run from in-memory scores.

        ``scores`` is either one score per question, or a sequence of per-question
        sample lists (for K > 1 samples per question).
        """
        groups: list[list[float]] = []
        for item in scores:
            if isinstance(item, (int, float, np.integer, np.floating)):
                groups.append([float(item)])
            else:
                groups.append([float(v) for v in item])
        if ids is None:
            ids = [str(i) for i in range(len(groups))]
        if len(ids) != len(groups):
            raise DataError("ids and scores must have the same length")
        return _build(
            name,
            [str(q) for q in ids],
            groups,
            None if clusters is None else [str(c) for c in clusters],
        )


def _get_path(row: Mapping[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _detect(rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]) -> str | None:
    first = rows[0]
    for name in candidates:
        if name in first:
            return name
    return None


def _to_score(value: Any, *, where: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE:
            return 1.0
        if s in _FALSE:
            return 0.0
        try:
            out = float(s)
        except ValueError:
            raise DataError(f"{where}: cannot interpret score {value!r} as a number") from None
    else:
        raise DataError(f"{where}: cannot interpret score {value!r} as a number")
    if not math.isfinite(out):
        raise DataError(f"{where}: score must be finite, got {value!r}")
    return out


def _build(
    name: str,
    ids: list[str],
    groups: list[list[float]],
    clusters: list[str] | None,
) -> EvalRun:
    scores = np.array([float(np.mean(g)) for g in groups], dtype=np.float64)
    n_samples = np.array([len(g) for g in groups], dtype=np.int64)
    within = np.array(
        [float(np.var(g, ddof=1)) if len(g) > 1 else math.nan for g in groups],
        dtype=np.float64,
    )
    return EvalRun(
        name=name,
        ids=tuple(ids),
        scores=scores,
        n_samples=n_samples,
        within_var=within,
        clusters=None if clusters is None else tuple(clusters),
    )


def from_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str = "run",
    id_field: str | None = None,
    score_field: str | None = None,
    cluster_field: str | None = None,
) -> EvalRun:
    """Build an :class:`EvalRun` from row dicts.

    Rows sharing an id are treated as repeated samples of the same question.
    Field names are auto-detected unless given; dotted paths (``metrics.acc``)
    reach into nested JSON.
    """
    if not rows:
        raise DataError(f"{name}: no rows found")

    id_key = id_field or _detect(rows, ID_FIELDS)
    score_key = score_field or _detect(rows, SCORE_FIELDS)
    cluster_key = cluster_field if cluster_field is not None else _detect(rows, CLUSTER_FIELDS)
    if score_key is None:
        raise DataError(
            f"{name}: could not find a score column (tried {', '.join(SCORE_FIELDS)}); "
            "pass --score-field"
        )

    order: list[str] = []
    groups: dict[str, list[float]] = {}
    cluster_of: dict[str, str] = {}
    for lineno, row in enumerate(rows, start=1):
        where = f"{name} row {lineno}"
        if id_key is None:
            qid = str(lineno - 1)
        else:
            raw_id = _get_path(row, id_key)
            if raw_id is None or raw_id == "":
                raise DataError(f"{where}: missing id field {id_key!r}")
            qid = str(raw_id)
        raw_score = _get_path(row, score_key)
        if raw_score is None or raw_score == "":
            raise DataError(f"{where}: missing score field {score_key!r}")
        score = _to_score(raw_score, where=where)
        if qid not in groups:
            order.append(qid)
            groups[qid] = []
        groups[qid].append(score)

        if cluster_key:
            raw_cluster = _get_path(row, cluster_key)
            if raw_cluster is None or raw_cluster == "":
                raise DataError(f"{where}: missing cluster field {cluster_key!r}")
            label = str(raw_cluster)
            prev = cluster_of.setdefault(qid, label)
            if prev != label:
                raise DataError(
                    f"{where}: question {qid!r} appears in clusters {prev!r} and {label!r}"
                )

    clusters = [cluster_of[q] for q in order] if cluster_key else None
    return _build(name, order, [groups[q] for q in order], clusters)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read rows from a ``.jsonl``, ``.json`` (list of objects), or ``.csv`` file."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"{p}: file not found")
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix == ".csv" or suffix == ".tsv":
        delimiter = "\t" if suffix == ".tsv" else ","
        return [dict(r) for r in csv.DictReader(text.splitlines(), delimiter=delimiter)]
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, Mapping):
            for key in ("results", "samples", "data", "records"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise DataError(f"{p}: expected a JSON list of objects")
        return [dict(r) for r in data]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{p} line {lineno}: invalid JSON ({exc.msg})") from None
        if not isinstance(obj, dict):
            raise DataError(f"{p} line {lineno}: expected a JSON object")
        rows.append(obj)
    return rows


def load(
    path: str | Path,
    *,
    name: str | None = None,
    id_field: str | None = None,
    score_field: str | None = None,
    cluster_field: str | None = None,
) -> EvalRun:
    """Load an eval run from disk. See :func:`from_records` for field handling."""
    p = Path(path)
    return from_records(
        read_rows(p),
        name=name or p.stem,
        id_field=id_field,
        score_field=score_field,
        cluster_field=cluster_field,
    )


@dataclass(frozen=True)
class Alignment:
    """Two runs restricted to their shared questions, in the same order."""

    baseline: EvalRun
    candidate: EvalRun
    dropped_baseline: int
    dropped_candidate: int


def align(baseline: EvalRun, candidate: EvalRun) -> Alignment:
    """Pair two runs on question id.

    Pairing is what makes A/B comparisons on the same eval powerful: the
    question-level difficulty cancels out. Questions present in only one run
    are dropped (and counted, so reports can warn about it).
    """
    shared = [q for q in baseline.ids if q in candidate._index]
    if not shared:
        raise DataError(
            f"{baseline.name} and {candidate.name} share no question ids; "
            "check --id-field or that both runs used the same eval"
        )
    a = baseline.subset(shared)
    b = candidate.subset(shared)
    if a.clusters is not None and b.clusters is not None and a.clusters != b.clusters:
        raise DataError("baseline and candidate disagree about question clusters")
    if a.clusters is None and b.clusters is not None:
        a = EvalRun(a.name, a.ids, a.scores, a.n_samples, a.within_var, b.clusters)
    elif b.clusters is None and a.clusters is not None:
        b = EvalRun(b.name, b.ids, b.scores, b.n_samples, b.within_var, a.clusters)
    return Alignment(
        baseline=a,
        candidate=b,
        dropped_baseline=baseline.n - len(shared),
        dropped_candidate=candidate.n - len(shared),
    )
