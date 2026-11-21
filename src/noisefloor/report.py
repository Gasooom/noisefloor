"""Rendering results as terminal text, GitHub-flavoured Markdown, or JSON.

Formatting conventions:

* Scores in [0, 1] are shown as percentages and differences as percentage
  points (pp). Other scales are shown as raw numbers.
* Every number that is an estimate is shown with its interval. A bare number
  is how people get fooled.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal

from .gate import GateResult
from .leaderboard import Leaderboard
from .power import PilotPlan
from .stats import Comparison, Estimate, Summary
from .suite import SuiteResult

Format = Literal["text", "markdown", "json"]
MARKER = "<!-- noisefloor-report -->"


# --------------------------------------------------------------------------- styling


@dataclass(frozen=True)
class Style:
    color: bool

    def _wrap(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.color else s

    def bold(self, s: str) -> str:
        return self._wrap("1", s)

    def dim(self, s: str) -> str:
        return self._wrap("2", s)

    def green(self, s: str) -> str:
        return self._wrap("32", s)

    def red(self, s: str) -> str:
        return self._wrap("31", s)

    def yellow(self, s: str) -> str:
        return self._wrap("33", s)

    def cyan(self, s: str) -> str:
        return self._wrap("36", s)

    @classmethod
    def auto(cls, force: bool | None = None) -> Style:
        if force is not None:
            return cls(color=force)
        if os.environ.get("NO_COLOR"):
            return cls(color=False)
        if os.environ.get("FORCE_COLOR"):
            return cls(color=True)
        return cls(color=sys.stdout.isatty())


PLAIN = Style(color=False)


# --------------------------------------------------------------------------- numbers


@dataclass(frozen=True)
class Units:
    """How to display scores: as percentages (for [0,1] scores) or raw values."""

    percent: bool

    def score(self, x: float) -> str:
        if math.isnan(x):
            return "n/a"
        return f"{100 * x:.1f}%" if self.percent else f"{x:.3f}"

    def delta(self, x: float) -> str:
        if math.isnan(x):
            return "n/a"
        if math.isinf(x):
            return "∞"
        return f"{100 * x:+.1f}pp" if self.percent else f"{x:+.3f}"

    def size(self, x: float) -> str:
        if math.isnan(x):
            return "n/a"
        if math.isinf(x):
            return "∞"
        return f"{100 * abs(x):.1f}pp" if self.percent else f"{abs(x):.3f}"

    def interval(self, lo: float, hi: float, *, signed: bool = False) -> str:
        if math.isnan(lo) or math.isnan(hi):
            return "[n/a]"
        if self.percent:
            fmt = "{:+.1f}" if signed else "{:.1f}"
            return f"[{fmt.format(100 * lo)}, {fmt.format(100 * hi)}]"
        fmt = "{:+.3f}" if signed else "{:.3f}"
        return f"[{fmt.format(lo)}, {fmt.format(hi)}]"

    @classmethod
    def for_estimates(cls, *ests: Estimate) -> Units:
        return cls(percent=all(-1e-9 <= e.mean <= 1 + 1e-9 for e in ests))


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    if p < 0.0001:
        return "< 0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def fmt_int(n: int | None) -> str:
    return "n/a" if n is None else f"{n:,}"


# --------------------------------------------------------------------------- JSON


def _clean(x: Any) -> Any:
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def estimate_dict(e: Estimate) -> dict[str, Any]:
    return {
        "mean": e.mean,
        "se": e.se,
        "ci": [e.ci_low, e.ci_high],
        "confidence": e.confidence,
        "n": e.n,
        "df": e.df,
        "method": e.method,
        "n_clusters": e.n_clusters,
    }


def summary_dict(s: Summary) -> dict[str, Any]:
    out: dict[str, Any] = {"name": s.name, "estimate": estimate_dict(s.estimate)}
    if s.naive is not None:
        out["naive_iid_estimate"] = estimate_dict(s.naive)
        out["design_effect"] = s.design_effect
    if s.wilson is not None:
        out["wilson_ci"] = list(s.wilson)
    if s.components is not None:
        vc = s.components
        out["variance_components"] = {
            "mean_samples_per_question": vc.mean_k,
            "within_question": vc.within,
            "between_question": vc.between,
            "projected_se_by_k": {str(k): v for k, v in vc.se_by_k.items()},
            "se_limit_infinite_k": vc.se_limit,
        }
    out["warnings"] = s.warnings
    return out


def comparison_dict(c: Comparison) -> dict[str, Any]:
    return {
        "baseline": {"name": c.baseline_name, **estimate_dict(c.baseline)},
        "candidate": {"name": c.candidate_name, **estimate_dict(c.candidate)},
        "difference": estimate_dict(c.diff),
        "verdict": c.verdict,
        "t_stat": c.t_stat,
        "p_value": c.p_value,
        "paired_questions": c.n,
        "wins": c.wins,
        "losses": c.losses,
        "ties": c.ties,
        "correlation": c.correlation,
        "unpaired_se": c.unpaired_se,
        "variance_reduction_from_pairing": c.variance_reduction,
        "power": c.power,
        "minimum_detectable_effect": c.mde,
        "questions_needed_for_observed_effect": c.n_required,
        "dropped": {"baseline": c.dropped_baseline, "candidate": c.dropped_candidate},
        "warnings": c.warnings,
    }


def to_json(obj: Any) -> str:
    return json.dumps(_clean(obj), indent=2)


# --------------------------------------------------------------------------- summary


def render_summary(s: Summary, fmt: Format = "text", style: Style = PLAIN) -> str:
    if fmt == "json":
        return to_json(summary_dict(s))
    e = s.estimate
    u = Units.for_estimates(e)
    level = f"{e.confidence:.0%}"
    clusters = f", {e.n_clusters} clusters" if e.method == "clustered" else ""

    if fmt == "markdown":
        lines = [
            f"### noisefloor: `{s.name}`",
            "",
            f"| Score | {level} CI | ± | SE | Questions |",
            "|---:|---:|---:|---:|---:|",
            f"| **{u.score(e.mean)}** | {u.interval(e.ci_low, e.ci_high)} | {u.size(e.margin)} "
            f"| {u.size(e.se)} | {fmt_int(e.n)}{clusters} |",
            "",
        ]
        if s.naive is not None and s.design_effect is not None:
            lines.append(
                f"Clustering inflates the variance {s.design_effect:.2f}×; an i.i.d. analysis "
                f"would have reported ±{u.size(s.naive.margin)}."
            )
        lines += [f"> ⚠️ {w}" for w in s.warnings]
        return "\n".join(lines).rstrip() + "\n"

    st = style
    out = [
        f"{st.bold('noisefloor')} {st.dim('·')} summary  {st.cyan(s.name)}  "
        f"{st.dim(f'({fmt_int(e.n)} questions{clusters})')}",
        "",
        f"  score     {st.bold(u.score(e.mean))}   ± {u.size(e.margin)}   "
        f"{st.dim(f'{level} CI')} {u.interval(e.ci_low, e.ci_high)}   "
        f"{st.dim(f'SE {u.size(e.se)}, {e.method}')}",
    ]
    if s.wilson is not None:
        out.append(f"  wilson    {st.dim(f'{level} CI')} {u.interval(*s.wilson)}")
    if s.naive is not None and s.design_effect is not None:
        out.append(
            f"  {st.yellow('clustering')} inflates variance {s.design_effect:.2f}×; "
            f"a naive i.i.d. analysis would claim ± {u.size(s.naive.margin)}"
        )
    if s.components is not None:
        vc = s.components
        out += [
            "",
            f"  {st.bold('resampling')}  {vc.mean_k:.1f} samples/question; "
            f"{vc.within_share:.0%} of single-sample variance is sampling noise",
        ]
        ks = sorted(vc.se_by_k)
        out.append(
            "    "
            + "  ".join(f"K={k}: ±{u.size(1.96 * vc.se_by_k[k])}" for k in ks)
            + f"  K=∞: ±{u.size(1.96 * vc.se_limit)}"
        )
        out.append("    " + st.dim(_resampling_advice(s)))
    for w in s.warnings:
        out.append(f"  {st.yellow('!')} {w}")
    return "\n".join(out) + "\n"


def _resampling_advice(s: Summary) -> str:
    vc = s.components
    assert vc is not None
    k = max(1.0, vc.mean_k)
    var_now = vc.between + vc.within / k
    var_double_k = vc.between + vc.within / (2 * k)
    gain_samples = 1 - math.sqrt(var_double_k / var_now) if var_now > 0 else 0.0
    gain_questions = 1 - 1 / math.sqrt(2)
    if gain_samples >= gain_questions * 0.8:
        return (
            f"doubling samples/question would cut the error bar by {gain_samples:.0%}; "
            f"doubling questions by {gain_questions:.0%}. Resampling is cheap leverage here."
        )
    return (
        f"doubling samples/question would cut the error bar by only {gain_samples:.0%}, "
        f"doubling questions by {gain_questions:.0%}. Buy more questions, not more samples."
    )


# --------------------------------------------------------------------------- compare


_VERDICT_TEXT = {
    "improvement": "candidate is significantly better",
    "regression": "candidate is significantly worse",
    "inconclusive": "the difference is within the noise floor",
}
_VERDICT_ICON = {"improvement": "✅", "regression": "❌", "inconclusive": "⚠️"}


def _compare_bullets(c: Comparison, u: Units) -> list[str]:
    b = []
    if c.correlation > 0 and c.variance_reduction > 0:
        b.append(
            f"pairing removed {c.variance_reduction:.0%} of the variance "
            f"(per-question correlation {c.correlation:.2f})"
        )
    b.append(f"wins / losses / ties: {c.wins} / {c.losses} / {c.ties}")
    b.append(f"smallest effect this eval can detect ({c.power:.0%} power): ±{u.size(c.mde)}")
    if c.verdict == "inconclusive" and c.n_required is not None:
        b.append(
            f"to reliably detect a {u.delta(c.diff.mean)} effect you need "
            f"~{fmt_int(c.n_required)} questions (you have {fmt_int(c.n)})"
        )
    return b


def render_comparison(
    c: Comparison,
    fmt: Format = "text",
    style: Style = PLAIN,
    gate: GateResult | None = None,
) -> str:
    if fmt == "json":
        payload = comparison_dict(c)
        if gate is not None:
            payload["gate"] = {"passed": gate.passed, "reasons": gate.reasons}
        return to_json(payload)
    u = Units.for_estimates(c.baseline, c.candidate)
    level = f"{c.diff.confidence:.0%}"
    clusters = f", {c.diff.n_clusters} clusters" if c.diff.method == "clustered" else ""
    if fmt == "markdown":
        return _comparison_markdown(c, u, level, clusters, gate)

    st = style
    v = c.verdict
    color = {"improvement": st.green, "regression": st.red, "inconclusive": st.yellow}[v]
    w = max(len(c.baseline_name), len(c.candidate_name), len("difference"))
    out = [
        f"{st.bold('noisefloor')} {st.dim('·')} compare  {st.cyan(c.baseline_name)} → "
        f"{st.cyan(c.candidate_name)}  {st.dim(f'({fmt_int(c.n)} paired questions{clusters})')}",
        "",
        f"  {'':<{w}}   {'score':>8}   {level + ' CI':>16}",
        f"  {c.baseline_name:<{w}}   {u.score(c.baseline.mean):>8}   "
        f"{u.interval(c.baseline.ci_low, c.baseline.ci_high):>16}",
        f"  {c.candidate_name:<{w}}   {u.score(c.candidate.mean):>8}   "
        f"{u.interval(c.candidate.ci_low, c.candidate.ci_high):>16}",
        "",
        f"  {'difference':<{w}}   {st.bold(color(f'{u.delta(c.diff.mean):>8}'))}   "
        f"{u.interval(c.diff.ci_low, c.diff.ci_high, signed=True):>16}   p = {fmt_p(c.p_value)}",
        "",
        f"  verdict  {color(st.bold(v.upper()))}  {_VERDICT_TEXT[v]}",
    ]
    if gate is not None:
        g = st.green if gate.passed else st.red
        out.append(f"  gate     {g(st.bold(gate.headline))}  {'; '.join(gate.reasons)}")
    out.append("")
    out += [f"  {st.dim('·')} {line}" for line in _compare_bullets(c, u)]
    out += [f"  {st.yellow('!')} {w_}" for w_ in c.warnings]
    return "\n".join(out) + "\n"


def _comparison_markdown(
    c: Comparison, u: Units, level: str, clusters: str, gate: GateResult | None
) -> str:
    v = c.verdict
    title = f"### noisefloor: `{c.baseline_name}` → `{c.candidate_name}`"
    lines = [MARKER, title, ""]
    if gate is not None:
        icon = "✅" if gate.passed else "❌"
        lines += [f"**Gate: {icon} {gate.headline}**: {'; '.join(gate.reasons)}", ""]
    lines += [
        f"| | Score | {level} CI |",
        "|---|---:|---:|",
        f"| `{c.baseline_name}` | {u.score(c.baseline.mean)} "
        f"| {u.interval(c.baseline.ci_low, c.baseline.ci_high)} |",
        f"| `{c.candidate_name}` | {u.score(c.candidate.mean)} "
        f"| {u.interval(c.candidate.ci_low, c.candidate.ci_high)} |",
        f"| **Δ** | **{u.delta(c.diff.mean)}** "
        f"| {u.interval(c.diff.ci_low, c.diff.ci_high, signed=True)} |",
        "",
        f"{_VERDICT_ICON[v]} **{v.capitalize()}**: {_VERDICT_TEXT[v]} "
        f"(p = {fmt_p(c.p_value)}, {fmt_int(c.n)} paired questions{clusters}).",
        "",
        "<details><summary>Details</summary>",
        "",
    ]
    lines += [f"- {b}" for b in _compare_bullets(c, u)]
    lines += [f"- ⚠️ {w}" for w in c.warnings]
    lines += [
        "",
        "</details>",
        "",
        "<sub>Generated by [noisefloor](https://github.com/Gasooom/noisefloor): "
        "statistical rigor for LLM evals.</sub>",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- suite


def render_suite(
    r: SuiteResult, fmt: Format = "text", style: Style = PLAIN, unmatched: list[str] | None = None
) -> str:
    if fmt == "json":
        return to_json(
            {
                "correction": r.correction,
                "alpha": r.alpha,
                "naive_significant": r.naive_significant,
                "improvements": [x.benchmark for x in r.improvements],
                "regressions": [x.benchmark for x in r.regressions],
                "unmatched": unmatched or [],
                "benchmarks": [
                    {
                        "benchmark": x.benchmark,
                        "p_adjusted": x.p_adjusted,
                        "significant": x.significant,
                        "direction": x.direction,
                        **comparison_dict(x.comparison),
                    }
                    for x in r.rows
                ],
            }
        )
    units = [Units.for_estimates(x.comparison.baseline, x.comparison.candidate) for x in r.rows]
    corr = {"holm": "Holm", "bh": "Benjamini-Hochberg", "bonferroni": "Bonferroni", "none": "no"}[
        r.correction
    ]
    survivors = len(r.improvements) + len(r.regressions)
    footer = (
        f"{r.naive_significant} of {len(r.rows)} benchmarks look significant at "
        f"α = {r.alpha:g} before correction; {survivors} "
        f"survive{'s' if survivors == 1 else ''} {corr} correction "
        f"({len(r.improvements)} better, {len(r.regressions)} worse)."
    )
    if fmt == "markdown":
        lines = [
            MARKER,
            "### noisefloor: benchmark suite",
            "",
            "| Benchmark | n | Baseline | Candidate | Δ | CI | p (adj.) | |",
            "|---|---:|---:|---:|---:|---:|---:|:-:|",
        ]
        for x, u in zip(r.rows, units, strict=True):
            c = x.comparison
            icon = {"improvement": "✅", "regression": "❌", "inconclusive": "·"}[x.direction]
            lines.append(
                f"| {x.benchmark} | {fmt_int(c.n)} | {u.score(c.baseline.mean)} "
                f"| {u.score(c.candidate.mean)} | {u.delta(c.diff.mean)} "
                f"| {u.interval(c.diff.ci_low, c.diff.ci_high, signed=True)} "
                f"| {fmt_p(x.p_adjusted)} | {icon} |"
            )
        lines += ["", footer]
        if unmatched:
            lines.append(f"\n> Skipped (present on one side only): {', '.join(unmatched)}")
        return "\n".join(lines) + "\n"

    st = style
    w = max(len("benchmark"), *(len(x.benchmark) for x in r.rows))
    out = [
        f"{st.bold('noisefloor')} {st.dim('·')} suite  "
        f"{st.dim(f'({len(r.rows)} benchmarks, {corr} correction)')}",
        "",
        f"  {'benchmark':<{w}}  {'n':>6}  {'Δ':>8}  {'CI':>16}  {'p':>8}  {'p adj':>8}",
    ]
    for x, u in zip(r.rows, units, strict=True):
        c = x.comparison
        color = {"improvement": st.green, "regression": st.red, "inconclusive": st.dim}[x.direction]
        out.append(
            f"  {x.benchmark:<{w}}  {fmt_int(c.n):>6}  {color(f'{u.delta(c.diff.mean):>8}')}  "
            f"{u.interval(c.diff.ci_low, c.diff.ci_high, signed=True):>16}  "
            f"{fmt_p(c.p_value):>8}  {fmt_p(x.p_adjusted):>8}"
        )
    out += ["", f"  {footer}"]
    if unmatched:
        out.append(f"  {st.yellow('!')} skipped (one side only): {', '.join(unmatched)}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- leaderboard


def render_leaderboard(lb: Leaderboard, fmt: Format = "text", style: Style = PLAIN) -> str:
    if fmt == "json":
        return to_json(
            {
                "mode": lb.mode,
                "n_questions": lb.n_questions,
                "confidence": lb.confidence,
                "n_tiers": lb.n_tiers,
                "entries": [
                    {
                        "name": e.name,
                        "rank": e.rank,
                        "rank_ci": [e.rank_low, e.rank_high],
                        "tier": e.tier,
                        "p_vs_tier_leader": e.p_vs_leader,
                        **estimate_dict(e.estimate),
                    }
                    for e in lb.entries
                ],
            }
        )
    u = Units.for_estimates(*(e.estimate for e in lb.entries))
    level = f"{lb.confidence:.0%}"
    n_models = len(lb.entries)
    footer = (
        f"{n_models} models fall into {lb.n_tiers} statistically distinguishable "
        f"tier{'s' if lb.n_tiers != 1 else ''}. "
        f"Models in the same tier are not significantly different from the tier's leader."
    )
    note = (
        "paired analysis over shared questions"
        if lb.mode == "paired"
        else "independent binomial approximation from summary scores"
    )

    def rank_range(lo: int, hi: int) -> str:
        return str(lo) if lo == hi else f"{lo}–{hi}"

    if fmt == "markdown":
        lines = [
            "### noisefloor: leaderboard",
            "",
            f"| Tier | Rank ({level} CI) | Model | Score | {level} CI |",
            "|:-:|:-:|---|---:|---:|",
        ]
        for e in lb.entries:
            lines.append(
                f"| {e.tier} | {e.rank} ({rank_range(e.rank_low, e.rank_high)}) | {e.name} "
                f"| {u.score(e.estimate.mean)} "
                f"| {u.interval(e.estimate.ci_low, e.estimate.ci_high)} |"
            )
        lines += ["", footer, "", f"<sub>{note}</sub>"]
        return "\n".join(lines) + "\n"

    st = style
    w = max(len("model"), *(len(e.name) for e in lb.entries))
    n_info = f", n = {fmt_int(lb.n_questions)}" if lb.n_questions else ""
    out = [
        f"{st.bold('noisefloor')} {st.dim('·')} leaderboard  {st.dim(f'({note}{n_info})')}",
        "",
        f"  {'tier':>4}  {'rank':>4}  {'rank CI':>7}  {'model':<{w}}  {'score':>7}  "
        f"{level + ' CI':>14}",
    ]
    prev_tier = None
    for e in lb.entries:
        if prev_tier is not None and e.tier != prev_tier:
            out.append(st.dim("  " + "─" * (4 + 2 + 4 + 2 + 7 + 2 + w + 2 + 7 + 2 + 14)))
        prev_tier = e.tier
        out.append(
            f"  {e.tier:>4}  {e.rank:>4}  {rank_range(e.rank_low, e.rank_high):>7}  "
            f"{e.name:<{w}}  {st.bold(f'{u.score(e.estimate.mean):>7}')}  "
            f"{u.interval(e.estimate.ci_low, e.estimate.ci_high):>14}"
        )
    out += ["", f"  {footer}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- power


@dataclass(frozen=True)
class PowerTable:
    """A small grid of power-analysis results for display."""

    title: str
    headers: list[str]
    rows: list[list[str]]
    notes: list[str]


def render_power(t: PowerTable, fmt: Format = "text", style: Style = PLAIN) -> str:
    if fmt == "json":
        return to_json(
            {
                "title": t.title,
                "rows": [dict(zip(t.headers, r, strict=True)) for r in t.rows],
                "notes": t.notes,
            }
        )
    if fmt == "markdown":
        lines = [
            f"### noisefloor: {t.title}",
            "",
            "| " + " | ".join(t.headers) + " |",
            "|" + "|".join("---:" for _ in t.headers) + "|",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in t.rows]
        lines += [""] + [f"- {n}" for n in t.notes]
        return "\n".join(lines) + "\n"
    widths = [max(len(h), *(len(r[i]) for r in t.rows)) for i, h in enumerate(t.headers)]
    st = style
    out = [f"{st.bold('noisefloor')} {st.dim('·')} power  {t.title}", ""]
    out.append("  " + "  ".join(st.dim(h.rjust(w)) for h, w in zip(t.headers, widths, strict=True)))
    for r in t.rows:
        out.append("  " + "  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)))
    if t.notes:
        out.append("")
        out += [f"  {st.dim('·')} {n}" for n in t.notes]
    return "\n".join(out) + "\n"


def pilot_table(plan: PilotPlan, percent: bool) -> PowerTable:
    u = Units(percent=percent)
    return PowerTable(
        title=f"plan from pilot run ({fmt_int(plan.n_current)} paired questions)",
        headers=["target effect", "power now", "questions needed (80% power)"],
        rows=[[u.delta(plan.effect), f"{plan.power_current:.0%}", fmt_int(plan.n_required)]],
        notes=[
            "uses the pilot's effective per-question SD of differences "
            f"({plan.sd_diff_effective:.3f}), which already includes pairing and clustering",
        ],
    )
