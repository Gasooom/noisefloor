# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-21

### Added
- `summary`: t-based confidence intervals, CR1 cluster-robust standard errors,
  Wilson intervals, design effect, and within/between variance decomposition
  for repeated samples, with projections for K = 1 to 16.
- `compare`: paired comparisons with p-values, wins/losses/ties, variance
  removed by pairing, minimum detectable effect, and the sample size needed to
  confirm the observed effect.
- `gate`: CI gate with no-regression, non-inferiority (`--max-regression`), and
  require-improvement policies; Markdown output for PR comments.
- `suite`: per-benchmark comparisons with Holm, Benjamini–Hochberg, or
  Bonferroni correction.
- `leaderboard`: joint-bootstrap rank confidence intervals and statistical
  tiers, from per-question runs or from summary scores.
- `power`: sample size, minimum detectable effect, achieved power, margin
  sizing, design effects, and planning from a pilot run.
- Vectorized (cluster) bootstrap for means and paired differences.
- Dependency-free Student-t distribution.
- Composite GitHub Action with job summary, PR comment, and outputs.
- Monte Carlo validation suite and synthetic example datasets.
