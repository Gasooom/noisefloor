# Using noisefloor in CI

The repository ships a composite GitHub Action. Point it at the per-question
results of your baseline and of the candidate, and it will:

1. print the comparison in the job log,
2. write a Markdown report to the job summary,
3. post (and on later pushes, *update*) a single comment on the pull request,
4. fail the job only when the change fails your policy.

## Minimal workflow

```yaml
# .github/workflows/evals.yml
name: evals
on: pull_request

permissions:
  contents: read
  pull-requests: write   # to post the report comment

jobs:
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Produce per-question results for the PR however you normally do,
      # e.g. run your eval harness against the changed prompt/model.
      - run: python run_evals.py --out results/candidate.jsonl

      - uses: Gasooom/noisefloor@main
        with:
          baseline: results/baseline.jsonl     # e.g. cached from main
          candidate: results/candidate.jsonl
```

## Choosing a policy

| Goal | Inputs | Fails when |
|---|---|---|
| Never block on noise (default) | *(none)* | candidate is **significantly worse** |
| "This refactor must not hurt quality" | `max-regression: 1pp` | a regression > 1pp **cannot be ruled out** |
| "This change exists to improve quality" | `require-improvement: true` | candidate is **not significantly better** |
| Report only | `allow-regression: true` | never |

The non-inferiority policy is the honest version of "make sure it does not
get worse". On a small eval it will fail even when nothing changed, and the
report tells you how large an eval you need. That is intended: a 200-question
eval genuinely cannot promise that quality did not drop by 1pp.

## Several gates on one PR

Give each gate its own `comment-tag` so they update separate comments:

```yaml
      - uses: Gasooom/noisefloor@main
        with: { baseline: base/qa.jsonl, candidate: pr/qa.jsonl, comment-tag: qa }
      - uses: Gasooom/noisefloor@main
        with: { baseline: base/code.jsonl, candidate: pr/code.jsonl, comment-tag: code }
```

For many benchmarks at once, prefer `noisefloor suite` with a multiple-comparison
correction (see the README).

## Outputs

`passed`, `verdict`, `difference`, `ci-low`, `ci-high`, `p-value`, which you can
use in later steps, e.g. `if: steps.gate.outputs.verdict == 'improvement'`.

## Security notes

All inputs reach the shell through environment variables, never through
inline `${{ }}` interpolation, so a crafted file name or input cannot inject
shell commands. The comment step needs only `pull-requests: write`.
