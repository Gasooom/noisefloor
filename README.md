<div align="center">

# noisefloor

**Is your eval improvement real, or just noise?**

Statistical rigor for LLM evaluations: honest error bars, paired comparisons,
cluster-robust standard errors, power analysis, multiple-comparison control,
leaderboard tiers, and a CI gate that blocks real regressions, not noise.

[![CI](https://github.com/Gasooom/noisefloor/actions/workflows/ci.yml/badge.svg)](https://github.com/Gasooom/noisefloor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-brightgreen)
![Typed](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![PyPI](https://img.shields.io/pypi/v/noisefloor-evals)](https://pypi.org/project/noisefloor-evals/)

</div>

---

## The problem

A team changes a prompt, swaps a model, or tweaks an agent. They rerun their
eval of a few hundred questions and see **72.0% → 76.2%**. They ship it.

Here is what noisefloor says about that exact result:

```console
$ noisefloor compare rag_baseline.jsonl rag_candidate.jsonl

noisefloor · compare  rag_baseline → rag_candidate  (600 paired questions, 120 clusters)

                     score             95% CI
  rag_baseline       72.0%       [67.3, 76.7]
  rag_candidate      76.2%       [71.9, 80.5]

  difference        +4.2pp       [-0.4, +8.8]   p = 0.076

  verdict  INCONCLUSIVE  the difference is within the noise floor

  · pairing removed 47% of the variance (per-question correlation 0.26)
  · wins / losses / ties: 98 / 73 / 429
  · smallest effect this eval can detect (80% power): ±6.6pp
  · to reliably detect a +4.2pp effect you need ~1,494 questions (you have 600)
```

This dataset is simulated, so we know the truth: the candidate really is
better, by **+1.4pp**. The observed +4.2pp is **3× the real effect**. An eval
this size cannot tell a real 1.4pp gain apart from luck, and noisefloor says so
instead of letting a lucky draw become a launch decision.

Most of the benchmarks people quote are smaller than their decisions assume:

| Benchmark (public size) | Questions | 95% error bar on one score (at 50%) | Smallest real gain you can reliably detect* |
|---|---:|---:|---:|
| AIME 2024 | 30 | ±18.7pp | 24.1pp |
| HumanEval | 164 | ±7.7pp | 10.8pp |
| GPQA Diamond | 198 | ±7.0pp | 9.9pp |
| SWE-bench Verified | 500 | ±4.4pp | 6.2pp |
| MMLU | 14,042 | ±0.8pp | 1.2pp |

<sub>*Paired comparison, per-question correlation 0.5, 80% power, α = 0.05. Computed with `noisefloor power`.</sub>

The statistics to handle this exist. Miller (2024), [*Adding Error Bars to
Evals*](https://arxiv.org/abs/2411.00640), lays them out for LLM evaluation.
What has been missing is a tool that brings them into the workflow engineers
actually use: a terminal, a pull request, a CI check. **noisefloor is that
tool.**

## What it does

| Command | Answers |
|---|---|
| `noisefloor summary` | What is my score, *with an honest error bar*? Is clustering inflating my confidence? Should I buy more questions or more samples per question? |
| `noisefloor compare` | Is B really better than A on the same questions? (paired test, cluster-robust) |
| `noisefloor gate` | Should this PR be allowed to merge? Exits `1` on policy failure and writes a PR comment. |
| `noisefloor suite` | Across 14 benchmarks, which changes survive multiple-comparison correction? |
| `noisefloor power` | How many questions do I need *before* I spend compute? |
| `noisefloor leaderboard` | Which models are actually distinguishable? Rank confidence intervals and statistical tiers. |

Also included: a **GitHub Action** ([docs](docs/github-action.md)), a typed
**Python API**, and a **Monte Carlo validation suite** that verifies every
guarantee against data with known ground truth.

## Install

```bash
pip install noisefloor-evals
```

The package is published as `noisefloor-evals`; you import it as `noisefloor`, and
the command is `noisefloor`.

Requires Python ≥ 3.10. The only runtime dependency is numpy; the Student-t
distribution is implemented from scratch, so there is no SciPy.

## Quick start

Your eval results just need one row per question (or per sample) with an id
and a score:

```jsonl
{"id": "doc017-q2", "doc_id": "doc017", "correct": true}
{"id": "doc017-q3", "doc_id": "doc017", "correct": false}
```

Common field names (`id`/`question_id`/`task_id`, `score`/`correct`/`passed`,
`cluster`/`doc_id`/`group`) are detected automatically. Booleans, `"pass"`/`"fail"`,
and continuous scores all work. Repeated ids are treated as repeated samples
of the same question. Use `--id-field`, `--score-field`, and `--cluster-field`
(dotted paths like `metrics.acc` work) for anything else.

Every example below runs on the files in [`examples/data`](examples/data).

### A score with an honest error bar

```console
$ noisefloor summary rag_candidate_k5.jsonl

noisefloor · summary  rag_candidate_k5  (600 questions, 120 clusters)

  score     72.6%   ± 3.5pp   95% CI [69.1, 76.1]   SE 1.8pp, clustered
  clustering inflates variance 2.38×; a naive i.i.d. analysis would claim ± 2.3pp

  resampling  5.0 samples/question; 76% of single-sample variance is sampling noise
    K=1: ±5.5pp  K=2: ±4.3pp  K=4: ±3.6pp  K=8: ±3.2pp  K=16: ±3.0pp  K=∞: ±2.7pp
    doubling samples/question would cut the error bar by only 10%, doubling questions by 29%. Buy more questions, not more samples.
```

The questions come in groups of five per document, so a naive analysis would
report an error bar about 35% too narrow. The variance decomposition then
answers a real budgeting question: with 5 samples per question already,
more resampling buys almost nothing.

### How many questions do I need?

```console
$ noisefloor power --effect 2pp --baseline-rate 70%

noisefloor · power  questions to detect +2.0pp (baseline 70.0%, 80% power, α = 0.05)

  correlation  questions
          0.0      8,077
          0.3      5,655
          0.5      4,040
          0.7      2,425
```

Run `--n 198` to get the smallest effect a 198-question eval can detect,
`--n` with `--effect` for achieved power, `--margin 2pp` to size a single
score, `--cluster-size`/`--icc` for clustered designs, or `--pilot base.jsonl
cand.jsonl` to plan from runs you already have.

### Many benchmarks without fooling yourself

```console
$ noisefloor suite runs/main/ runs/pr/ --correction bh
  ...
  4 of 14 benchmarks look significant at α = 0.05 before correction;
  2 survive Benjamini-Hochberg correction (1 better, 1 worse).
```

In this simulated suite exactly two benchmarks truly changed. Before
correction, four *look* significant, which is enough for a misleading release
note. After correction, exactly the two real effects remain.

### Leaderboards with rank uncertainty

```console
$ noisefloor leaderboard --summary leaderboard.csv

  tier  rank  rank CI  model          score          95% CI
     1     1      1–4  aurora-xl      64.6%    [57.9, 71.3]
     1     2      1–4  basalt-pro     63.1%    [56.4, 69.8]
     1     3      1–4  cirrus-2       62.6%    [55.9, 69.3]
     1     4      1–5  delta-large    58.1%    [51.2, 65.0]
  ─────────────────────────────────────────────────────────
     2     5      4–6  ember-mini     51.5%    [44.5, 58.5]
     2     6      5–6  fjord-7b       41.9%    [35.0, 48.8]
```

On a 198-question benchmark, the "#1 model" could plausibly be #4. Pass
per-question result files instead of `--summary` to get a paired analysis
with joint-bootstrap rank intervals.

## In CI: block real regressions, not noise

```yaml
- uses: Gasooom/noisefloor@main
  with:
    baseline: results/main.jsonl
    candidate: results/pr.jsonl
    max-regression: 1pp        # optional non-inferiority margin
```

The action prints the comparison, writes a job summary, keeps **one**
up-to-date comment on the PR, and fails only when the change fails your
policy:

| Policy | Fails when |
|---|---|
| default | the candidate is **significantly worse** |
| `max-regression: 1pp` | a regression > 1pp **cannot be ruled out** |
| `require-improvement: true` | the candidate is **not significantly better** |

See [docs/github-action.md](docs/github-action.md). This repository's own CI
dogfoods the action: it asserts that the noisy +4.2pp example passes and a real
regression fails.

## Python API

```python
import noisefloor as nf

base = nf.load("baseline.jsonl")
cand = nf.load("candidate.jsonl")

c = nf.compare(base, cand)
c.verdict                      # 'improvement' | 'regression' | 'inconclusive'
c.diff.mean, c.diff.ci_low, c.diff.ci_high, c.p_value
c.mde                          # smallest effect this eval can detect
c.n_required                   # questions needed to confirm the observed effect

nf.sample_size(0.02, baseline_rate=0.7, correlation=0.5)   # -> 4040
nf.evaluate_gate(c, nf.GatePolicy(max_regression=0.01)).passed

runs = {name: (nf.load(f"main/{name}.jsonl"), nf.load(f"pr/{name}.jsonl")) for name in names}
nf.compare_suite(runs, correction="holm").regressions
```

Every function is typed (`mypy --strict`) and returns frozen dataclasses.

## Does it actually work? (validation)

Every guarantee noisefloor makes is a frequency claim, so each one is checked
by simulation against data with known ground truth
([`examples/validate.py`](examples/validate.py), 4,000 simulations per row; also
run in CI):

| Check | Target | noisefloor | Common shortcut |
|---|:-:|:-:|:-:|
| 95% CI coverage, i.i.d. questions (n=100) | 95% | **94.9%** | n/a |
| 95% CI coverage, tiny judge-scored eval (n=12) | 95% | **94.5%** | 92.4% (z-interval) |
| 95% CI coverage, clustered eval (60 passages × 5 questions) | 95% | **95.2%** | 88.1% (ignore clusters) |
| False alarm rate, equally good models (clustered eval) | 5% | **5.1%** | 9.0% (ignore clusters) |
| Power at planned n = 1,468 for a +5pp gain | 80% | **79.1%** | n/a |
| ≥1 false "win" across 10 unchanged benchmarks | ≤ 5% | **5.1%** | 40.5% (no correction) |

The Student-t implementation is checked against published tables to 7
significant figures, and the estimators against hand-derived values. Run the
full suite with `pytest`, or only the fast tests with `pytest -m "not slow"`.

## How it works

The full derivations are in [docs/methodology.md](docs/methodology.md). In short:

- **Unit of analysis = question.** Repeated samples are averaged per question,
  and their spread is kept as within-question variance.
- **Cluster-robust (CR1) standard errors** with G−1 degrees of freedom when
  questions share a passage, repository, or conversation.
- **Paired differences** whenever two systems answer the same questions. Shared
  difficulty cancels, which often removes half the variance.
- **Student-t intervals** everywhere, plus Wilson intervals for small binary
  evals and a vectorized cluster bootstrap as a cross-check.
- **Power analysis** with correlation and design effects, including planning
  from a pilot run.
- **Holm / Benjamini–Hochberg** corrections across benchmarks.
- **Joint-bootstrap rank intervals** and paired-test tiers for leaderboards.

## Design principles

1. **Never print a bare number.** Every estimate comes with its interval.
2. **Say what to do next.** "Inconclusive" alone is useless. "You need ~1,500
   questions" or "buy questions, not samples" is actionable.
3. **Validate the guarantees, not just the code.** Unit tests show the code
   does what it says; Monte Carlo tests show that what it says is true.
4. **Boring to install.** Pure Python plus numpy, no services, no API keys, works
   offline on any eval harness's output.

## Roadmap

- [ ] Native readers for popular harness outputs (lm-evaluation-harness, Inspect, OpenAI Evals, promptfoo)
- [ ] Sequential testing (always-valid p-values) for evals that stream in
- [ ] LLM-as-judge noise: judge-resampling variance and inter-judge agreement
- [ ] Stratified / weighted estimates for evals that mix difficulty buckets
- [ ] HTML report with forest plots

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
New statistical features should come with a Monte Carlo check in
`tests/test_validation.py`.

## Citation

If noisefloor helps your research, please cite it (see [CITATION.cff](CITATION.cff))
and the paper that motivated it:

> Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* arXiv:2411.00640.

## License

MIT © Abualgasim Ibrahim
