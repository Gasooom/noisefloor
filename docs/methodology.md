# Methodology

This document states exactly what noisefloor computes, and why. The approach
follows Miller (2024), *Adding Error Bars to Evals*, with standard results
from survey statistics and multiple testing.

**Notation.** An eval has $n$ questions indexed by $i$. Question $i$ is
answered $K_i \ge 1$ times with scores $x_{ik}$ (binary or continuous), and
belongs to cluster $c(i)$ (a passage, a repository, a conversation); there are
$G$ clusters. The per-question score is $s_i = \frac{1}{K_i}\sum_k x_{ik}$.

The estimand is the **expected score over the distribution of questions**
that the eval samples from, not the score on these particular questions. That
is why the eval's size matters even when every question is answered
perfectly reliably.

---

## 1. Mean and standard error

$$\bar s = \frac{1}{n}\sum_i s_i, \qquad \widehat{\mathrm{SE}}_{\text{iid}} = \frac{\mathrm{sd}(s)}{\sqrt n}$$

Intervals use Student's $t$ with $n-1$ degrees of freedom:
$\bar s \pm t_{1-\alpha/2,\,n-1}\,\widehat{\mathrm{SE}}$. For $n=30$ the
critical value is 2.045 rather than 1.960; the Normal approximation is
noticeably overconfident on the small evals where precision matters most.
The $t$ distribution is implemented from the regularized incomplete beta
function, so the only dependency is numpy.

For binary, unclustered, single-sample runs noisefloor also reports the
**Wilson score interval**, which stays inside $[0,1]$ and never collapses to
zero width at 0% or 100%.

## 2. Clustered standard errors

Questions in the same cluster share difficulty, so they are positively
correlated and carry less information than independent questions. Treating
them as independent understates the standard error. noisefloor uses the CR1
cluster-robust (sandwich) estimator for a mean:

$$\widehat{\mathrm{Var}}_{\text{CR}}(\bar s) = \frac{G}{G-1}\cdot\frac{1}{n^2}\sum_{g=1}^{G}\Big(\sum_{i:\,c(i)=g}(s_i-\bar s)\Big)^2$$

with $t_{G-1}$ critical values (Cameron and Miller, 2015). With one question
per cluster it reduces exactly to $\mathrm{sd}(s)^2/n$. The reported **design
effect** is $(\widehat{\mathrm{SE}}_{\text{CR}}/\widehat{\mathrm{SE}}_{\text{iid}})^2$:
how much variance the clustering adds.

With fewer than about 20 clusters, cluster-robust SEs are biased downward,
and noisefloor warns about it.

## 3. Paired comparisons

When two systems answer the same questions, analyse the per-question
differences $d_i = s_i^{B} - s_i^{A}$:

$$\hat\Delta = \bar d,\qquad \widehat{\mathrm{SE}}(\hat\Delta) = \widehat{\mathrm{SE}}(d)\ \text{(iid or clustered, as above)}$$

Because

$$\mathrm{Var}(d) = \mathrm{Var}(s^A) + \mathrm{Var}(s^B) - 2\,\mathrm{Cov}(s^A, s^B),$$

and question difficulty makes the covariance positive, pairing removes a large
share of the variance. noisefloor reports this as *variance removed by
pairing*, $1 - \mathrm{SE}_{\text{paired}}^2 / (\mathrm{SE}_A^2 + \mathrm{SE}_B^2)$.

The test is a paired $t$-test, $t = \hat\Delta/\widehat{\mathrm{SE}}$, with
$n-1$ (or $G-1$) degrees of freedom. The **verdict** is *improvement* or
*regression* when the $(1-\alpha)$ interval excludes zero, and *inconclusive*
otherwise.

Pairing does not make clustering irrelevant. If two systems have
cluster-specific strengths (one better at math passages, the other at
history), the differences are themselves clustered, and ignoring this roughly
doubles the false-alarm rate (see the validation table in the README).

## 4. Resampling: more questions or more samples?

With $K$ samples per question,

$$\mathrm{Var}(s_i) = \sigma^2_{b} + \frac{\sigma^2_{w}}{K}$$

where $\sigma^2_w$ is the within-question (sampling) variance and $\sigma^2_b$
the between-question variance. noisefloor estimates
$\hat\sigma^2_w = \text{mean}_i\,\widehat{\mathrm{Var}}_k(x_{ik})$ over
questions with $K_i\ge2$ and
$\hat\sigma^2_b = \max(0,\ \widehat{\mathrm{Var}}(s_i) - \text{mean}_i\,\hat\sigma^2_{w}/K_i)$,
then projects the standard error for $K = 1, 2, 4, 8, 16, \infty$, scaled by
the observed clustering inflation. Resampling only shrinks the $\sigma^2_w/K$
term, so once it is small relative to $\sigma^2_b$, the only remedy is more
questions. The report says which regime you are in.

## 5. Power and sample size

To detect a paired difference $\delta$ with two-sided level $\alpha$ and power
$1-\beta$:

$$n = \left\lceil \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\ \sigma_d^2\ \mathrm{DEFF}}{\delta^2} \right\rceil$$

For binary scores with rates $p_1$ and $p_2 = p_1 + \delta$ and per-question
correlation $\rho$:

$$\sigma_d^2 = p_1(1-p_1) + p_2(1-p_2) - 2\rho\sqrt{p_1(1-p_1)\,p_2(1-p_2)}$$

and for clusters of mean size $m$ with intra-class correlation $\mathrm{ICC}$,
$\mathrm{DEFF} = 1 + (m-1)\,\mathrm{ICC}$ (Kish, 1965). $\rho = 0$ is the
conservative choice without pilot data, and equals the requirement *per
model* for an unpaired design.

The **minimum detectable effect** inverts this formula (solved by bisection
for binary scores, because $\sigma^2_d$ depends on $\delta$). **Achieved
power** is $\Phi(|\delta|/\mathrm{SE} - z_{1-\alpha/2}) + \Phi(-|\delta|/\mathrm{SE} - z_{1-\alpha/2})$.

`plan_from_comparison` treats an existing run as a pilot study. It uses the
*effective* per-question SD $\widehat{\mathrm{SE}}(\hat\Delta)\sqrt n$, which
already includes both the benefit of pairing and the cost of clustering.

## 6. Multiple comparisons

Reporting the benchmarks that moved out of a suite of $k$ inflates false
positives. At $k=10$ and $\alpha = 0.05$, the chance of at least one spurious
"win" is $1 - 0.95^{10} \approx 40\%$.

- **Holm (1979)** step-down: sort the p-values, set
  $\tilde p_{(j)} = \max_{l\le j} \min(1, (k-l+1)\,p_{(l)})$. This controls the
  family-wise error rate and is uniformly more powerful than Bonferroni. It is
  the default, and the right choice for ship/no-ship gates.
- **Benjamini–Hochberg (1995)**: $\tilde p_{(j)} = \min_{l \ge j} \min(1, k\,p_{(l)}/l)$.
  This controls the false discovery rate and suits exploratory sweeps over
  many benchmarks or slices.

## 7. Bootstrap

The (cluster) percentile bootstrap gives an assumption-light second opinion
(Efron and Tibshirani, 1993). Each replicate draws multinomial weights $w_g$
over clusters and computes $\sum_g w_g S_g / \sum_g w_g m_g$, where $S_g$ and
$m_g$ are the cluster's score sum and size. This handles unequal cluster sizes
exactly, without materialising index arrays. The p-value is the centred
bootstrap test $\Pr^*(|\hat\theta^* - \hat\theta| \ge |\hat\theta|)$.

## 8. Leaderboards

**Rank intervals.** All models are resampled *jointly* over the shared
questions (or clusters); each replicate re-ranks the models, and the
$\alpha/2$ and $1-\alpha/2$ quantiles of each model's rank form its rank
interval. Joint resampling preserves the correlation between models.

**Tiers.** Walking down the sorted list, a model starts a new tier when it is
significantly worse (paired test) than the current tier's leader. Models in a
tier cannot be distinguished from its leader. Tiers are a descriptive summary,
not a formal multiple-testing procedure.

**Summary mode** (only score and $n$ known) treats each model as an
independent binomial with variance floored at $0.25/n$ for saturated scores.
When the models actually shared questions this is conservative, because the
positive correlation that pairing would exploit is ignored.

## 9. CI gate policies

| Policy | Fails when |
|---|---|
| default | $\mathrm{CI}_{\text{high}}(\Delta) < 0$ (significant regression) |
| `max_regression = m` | $\mathrm{CI}_{\text{low}}(\Delta) < -m$ (regression larger than $m$ not ruled out) |
| `require_improvement` | $\mathrm{CI}_{\text{low}}(\Delta) \le 0$ |

The non-inferiority policy is a confidence-interval formulation of a
one-sided test at level $\alpha/2$.

## Validation

`examples/validate.py` checks the frequency guarantees above by Monte Carlo
simulation on data with known ground truth, and `tests/test_validation.py`
runs the same checks in CI.

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289–300.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Kish, L. (1965). *Survey Sampling*. Wiley.
- Miller, E. (2024). Adding error bars to evals: A statistical approach to language model evaluations. arXiv:2411.00640.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212.
