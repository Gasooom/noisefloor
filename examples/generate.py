"""Generate the synthetic example datasets in examples/data/.

Everything here is simulated, with known ground truth, so the examples can
show what noisefloor gets right (and what naive analysis gets wrong):

* ``rag_baseline.jsonl`` / ``rag_candidate.jsonl``: a retrieval-QA eval with
  600 questions over 120 documents (5 questions each). Document difficulty
  is shared, so questions are clustered. The candidate is truly better by
  1.4 points -- real, but too small for this eval to confirm.
* ``rag_candidate_k5.jsonl``: the candidate sampled 5 times per question.
* ``suite/{main,pr}/``: 14 benchmarks; one has a real +10 point gain, one a
  real -8 point regression, the other 12 are unchanged.
* ``leaderboard.csv``: summary scores for six fictional models on a
  198-question benchmark.

Run:  python examples/generate.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "data"
SEED = 20261121


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def rag(rng: np.random.Generator) -> None:
    n_docs, per_doc = 120, 5
    doc_difficulty = rng.normal(0.0, 1.2, n_docs)
    q_difficulty = rng.normal(0.0, 0.8, (n_docs, per_doc))
    logit = 1.1 - doc_difficulty[:, None] - q_difficulty
    p_base = logistic(logit)
    p_cand = logistic(logit + 0.09)  # ~ +1.5pp on average

    base_rows, cand_rows, k5_rows = [], [], []
    for d in range(n_docs):
        for j in range(per_doc):
            qid = f"doc{d:03d}-q{j}"
            cluster = f"doc{d:03d}"
            base_rows.append(
                {"id": qid, "doc_id": cluster, "correct": bool(rng.random() < p_base[d, j])}
            )
            cand_rows.append(
                {"id": qid, "doc_id": cluster, "correct": bool(rng.random() < p_cand[d, j])}
            )
            for k in range(5):
                k5_rows.append(
                    {
                        "id": qid,
                        "doc_id": cluster,
                        "sample": k,
                        "correct": bool(rng.random() < p_cand[d, j]),
                    }
                )
    write_jsonl(OUT / "rag_baseline.jsonl", base_rows)
    write_jsonl(OUT / "rag_candidate.jsonl", cand_rows)
    write_jsonl(OUT / "rag_candidate_k5.jsonl", k5_rows)


def suite(rng: np.random.Generator) -> None:
    # 12 benchmarks with no true change, one real gain, one real regression.
    # With 12 null benchmarks at alpha = 0.05, the chance that at least one
    # *looks* significant is 1 - 0.95**12 = 46%: release notes write themselves.
    benchmarks = {
        "arithmetic": (400, 0.0),
        "code_repair": (300, 0.0),
        "commonsense": (500, 0.0),
        "data_extraction": (350, 0.0),
        "instruction_following": (500, 0.0),
        "long_context": (600, -0.08),
        "multilingual_qa": (600, 0.0),
        "planning": (300, 0.0),
        "reading_comprehension": (450, 0.0),
        "safety_refusals": (350, 0.0),
        "sql_generation": (400, 0.0),
        "summarization_judge": (400, 0.0),
        "tool_use": (600, 0.10),
        "translation": (500, 0.0),
    }
    for name, (n, effect) in benchmarks.items():
        p = rng.beta(0.7, 0.5, n)  # realistic spread of question difficulty
        a = rng.random(n) < p
        b = rng.random(n) < np.clip(p + effect, 0.0, 1.0)
        for side, scores in (("main", a), ("pr", b)):
            rows: list[dict[str, object]] = [
                {"id": f"{name}-{i:04d}", "score": int(x)} for i, x in enumerate(scores)
            ]
            write_jsonl(OUT / "suite" / side / f"{name}.jsonl", rows)


def leaderboard() -> None:
    rows = [
        ("aurora-xl", 64.6),
        ("basalt-pro", 63.1),
        ("cirrus-2", 62.6),
        ("delta-large", 58.1),
        ("ember-mini", 51.5),
        ("fjord-7b", 41.9),
    ]
    with (OUT / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "score", "n"])
        for name, score in rows:
            w.writerow([name, score, 198])


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    rag(rng)
    suite(rng)
    leaderboard()
    print(f"wrote example data to {OUT}")
