"""Monte Carlo validation of noisefloor's statistical guarantees.

Every claim noisefloor makes is a frequency claim: "95% intervals contain the
truth 95% of the time", "the gate raises a false alarm 5% of the time", "this
sample size gives 80% power". This script checks each claim by simulation on
data with known ground truth, and contrasts it with the common shortcut.

Run:  python examples/validate.py [--sims 4000]

The same checks run, with fewer simulations, in tests/test_validation.py.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import noisefloor as nf
from noisefloor.stats import estimate_mean

Z95 = 1.959964


@dataclass(frozen=True)
class Check:
    name: str
    target: str
    noisefloor: float
    shortcut: float | None
    shortcut_name: str = ""


def clustered_probs(
    rng: np.random.Generator,
    n_clusters: int,
    size: int,
    p: float = 0.7,
    cluster_sd: float = 1.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per-question success probabilities with shared per-cluster difficulty.

    Models e.g. a reading-comprehension eval: 5 questions per passage, and
    some passages are simply harder than others.
    """
    logit = np.log(p / (1 - p))
    shift = np.repeat(rng.normal(0.0, cluster_sd, n_clusters), size)
    noise = rng.normal(0.0, 0.3, n_clusters * size)
    probs = 1.0 / (1.0 + np.exp(-(logit + shift + noise)))
    clusters = tuple(f"c{i}" for i in np.repeat(np.arange(n_clusters), size))
    return probs, clusters


def population_mean(p: float = 0.7, cluster_sd: float = 1.0) -> float:
    """The estimand: expected accuracy over the whole question distribution."""
    rng = np.random.default_rng(12345)
    logit = (
        np.log(p / (1 - p)) + rng.normal(0, cluster_sd, 4_000_000) + rng.normal(0, 0.3, 4_000_000)
    )
    return float(np.mean(1.0 / (1.0 + np.exp(-logit))))


def draw(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    return (rng.random(probs.shape) < probs).astype(float)


# --------------------------------------------------------------------------- checks


def coverage_iid(sims: int, rng: np.random.Generator) -> Check:
    n, p = 100, 0.7
    hits = sum(estimate_mean(draw(rng, np.full(n, p))).contains(p) for _ in range(sims))
    return Check(f"95% CI coverage, i.i.d. questions (n={n})", "95%", hits / sims, None)


def coverage_small(sims: int, rng: np.random.Generator) -> Check:
    """LLM-judge ratings on a 1-10 scale, only 12 prompts."""
    n, truth = 12, 5.5  # uniform integer ratings 1..10
    hits_t = hits_z = 0
    for _ in range(sims):
        e = estimate_mean(rng.integers(1, 11, n).astype(float))
        hits_t += e.contains(truth)
        hits_z += abs(e.mean - truth) <= Z95 * e.se
    return Check(
        f"95% CI coverage, tiny judge-scored eval (n={n})",
        "95%",
        hits_t / sims,
        hits_z / sims,
        "z-interval",
    )


def coverage_clustered(sims: int, rng: np.random.Generator) -> Check:
    truth = population_mean()
    hits_c = hits_iid = 0
    for _ in range(sims):
        probs, clusters = clustered_probs(rng, 60, 5)
        x = draw(rng, probs)
        hits_c += estimate_mean(x, clusters=clusters).contains(truth)
        hits_iid += estimate_mean(x).contains(truth)
    return Check(
        "95% CI coverage, clustered eval (60 passages × 5 questions)",
        "95%",
        hits_c / sims,
        hits_iid / sims,
        "ignore clusters",
    )


def false_alarm_rate(sims: int, rng: np.random.Generator) -> Check:
    """Two models with identical average accuracy but different topic strengths.

    Model A gets +delta_c on passage c and model B gets -delta_c, with delta_c
    symmetric around zero, so by symmetry their expected accuracies are equal
    and any 'significant' difference is a false alarm.
    """
    alarms_c = alarms_iid = 0
    for _ in range(sims):
        probs, clusters = clustered_probs(rng, 60, 5)
        logit = np.log(probs / (1 - probs))
        delta = np.repeat(rng.normal(0.0, 0.5, 60), 5)
        a = nf.EvalRun.from_scores(draw(rng, 1 / (1 + np.exp(-(logit + delta)))), clusters=clusters)
        b = nf.EvalRun.from_scores(draw(rng, 1 / (1 + np.exp(-(logit - delta)))), clusters=clusters)
        alarms_c += nf.compare(a, b).significant
        alarms_iid += nf.compare(a, b, cluster=False).significant
    return Check(
        "false alarm rate, equally good models (clustered eval)",
        "5%",
        alarms_c / sims,
        alarms_iid / sims,
        "ignore clusters",
    )


def power_check(sims: int, rng: np.random.Generator) -> Check:
    effect, rate = 0.05, 0.6
    n = nf.sample_size(effect, baseline_rate=rate)
    detected = 0
    for _ in range(sims):
        a = nf.EvalRun.from_scores(draw(rng, np.full(n, rate)))
        b = nf.EvalRun.from_scores(draw(rng, np.full(n, rate + effect)))
        detected += nf.compare(a, b).verdict == "improvement"
    return Check(f"power at planned n = {n:,} for a +5pp gain", "80%", detected / sims, None)


def fwer_check(sims: int, rng: np.random.Generator) -> Check:
    k, n = 10, 200
    any_holm = any_raw = 0
    for _ in range(sims):
        pvals = []
        for _ in range(k):
            p = rng.beta(1, 1, n)
            a = nf.EvalRun.from_scores(draw(rng, p))
            b = nf.EvalRun.from_scores(draw(rng, p))
            pvals.append(nf.compare(a, b).p_value)
        any_holm += min(nf.holm(pvals)) < 0.05
        any_raw += min(pvals) < 0.05
    return Check(
        f"≥1 false 'win' across {k} unchanged benchmarks",
        "≤ 5%",
        any_holm / sims,
        any_raw / sims,
        "no correction",
    )


CHECKS: list[tuple[Callable[[int, np.random.Generator], Check], float]] = [
    (coverage_iid, 1.0),
    (coverage_small, 1.0),
    (coverage_clustered, 1.0),
    (false_alarm_rate, 1.0),
    (power_check, 0.5),
    (fwer_check, 0.25),
]


def run_all(sims: int, seed: int) -> list[Check]:
    rng = np.random.default_rng(seed)
    return [fn(max(100, int(sims * scale)), rng) for fn, scale in CHECKS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo validation of noisefloor")
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("| Check | Target | noisefloor | Common shortcut |")
    print("|---|:-:|:-:|:-:|")
    for c in run_all(args.sims, args.seed):
        shortcut = "n/a" if c.shortcut is None else f"{c.shortcut:.1%} ({c.shortcut_name})"
        print(f"| {c.name} | {c.target} | **{c.noisefloor:.1%}** | {shortcut} |")


if __name__ == "__main__":
    main()
