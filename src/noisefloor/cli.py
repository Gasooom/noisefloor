"""Command-line interface.

Exit codes: 0 success / gate passed, 1 gate failed, 2 usage or data error.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .bootstrap import bootstrap_diff, bootstrap_mean
from .data import DataError, EvalRun, load, read_rows
from .gate import GatePolicy, evaluate_gate
from .leaderboard import from_runs, from_summary
from .multiple import CORRECTIONS
from .power import (
    achieved_power,
    design_effect,
    minimum_detectable_effect,
    plan_from_comparison,
    sample_size,
    sample_size_for_margin,
)
from .report import (
    PowerTable,
    Style,
    Units,
    fmt_int,
    fmt_p,
    pilot_table,
    render_comparison,
    render_leaderboard,
    render_power,
    render_suite,
    render_summary,
)
from .stats import compare, summarize
from .suite import compare_suite, load_suite

EXIT_OK, EXIT_GATE_FAILED, EXIT_ERROR = 0, 1, 2
CORRELATIONS = (0.0, 0.3, 0.5, 0.7)
CORRELATION_NOTE = (
    "correlation = per-question correlation between the two models' scores. Models on the "
    "same questions are usually positively correlated; 0 is the conservative (unpaired) case"
)


# --------------------------------------------------------------------------- parsing helpers


def parse_amount(text: str) -> float:
    """Parse an effect or margin: ``2pp``, ``2%`` and ``0.02`` all mean 0.02."""
    s = text.strip().lower()
    try:
        if s.endswith("pp"):
            return float(s[:-2]) / 100.0
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid amount {text!r} (try 2pp or 0.02)") from None


def parse_rate(text: str) -> float:
    """Parse a rate: ``70%``, ``70`` and ``0.7`` all mean 0.7."""
    s = text.strip().lower()
    try:
        v = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid rate {text!r} (try 70% or 0.7)") from None
    if v > 1.0:
        v /= 100.0
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"rate {text!r} must be between 0 and 100%")
    return v


def parse_confidence(text: str) -> float:
    v = parse_rate(text)
    if not 0.5 <= v < 1.0:
        raise argparse.ArgumentTypeError("confidence must be in [50%, 100%)")
    return v


def _data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--id-field", help="question id field (default: auto-detect)")
    g.add_argument("--score-field", help="score field (default: auto-detect)")
    g.add_argument(
        "--cluster-field", help="cluster field (default: auto-detect; pass '' to disable)"
    )
    g.add_argument(
        "--no-cluster", action="store_true", help="ignore clusters even if present (not advised)"
    )


def _output_args(p: argparse.ArgumentParser, *, markdown_file: bool = False) -> None:
    g = p.add_argument_group("output")
    g.add_argument(
        "-f", "--format", choices=("text", "markdown", "json"), default="text", dest="fmt"
    )
    g.add_argument("-o", "--output", type=Path, help="write the report to a file, not stdout")
    g.add_argument("--color", action=argparse.BooleanOptionalAction, default=None)
    if markdown_file:
        g.add_argument(
            "--markdown",
            type=Path,
            metavar="FILE",
            help="additionally write a Markdown report (e.g. for a PR comment or job summary)",
        )


def _stat_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("statistics")
    g.add_argument("-c", "--confidence", type=parse_confidence, default=0.95, help="default: 95%%")
    g.add_argument(
        "--bootstrap",
        type=int,
        nargs="?",
        const=10_000,
        default=0,
        metavar="N",
        help="also run a bootstrap with N resamples (default N: 10000)",
    )
    g.add_argument("--seed", type=int, default=0, help="random seed for resampling")


def _load(path: Path, args: argparse.Namespace) -> EvalRun:
    return load(
        path,
        id_field=args.id_field,
        score_field=args.score_field,
        cluster_field=args.cluster_field,
    )


def _emit(args: argparse.Namespace, text: str) -> None:
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _style(args: argparse.Namespace) -> Style:
    if args.output or args.fmt != "text":
        return Style(color=False)
    return Style.auto(args.color)


# --------------------------------------------------------------------------- commands


def cmd_summary(args: argparse.Namespace) -> int:
    run = _load(args.run, args)
    s = summarize(run, confidence=args.confidence, cluster=not args.no_cluster)
    text = render_summary(s, args.fmt, _style(args))
    if args.bootstrap and args.fmt == "text":
        b = bootstrap_mean(
            run,
            n_resamples=args.bootstrap,
            confidence=args.confidence,
            cluster=not args.no_cluster,
            seed=args.seed,
        )
        u = Units.for_estimates(s.estimate)
        text += (
            f"  bootstrap ({b.method}, {fmt_int(b.n_resamples)} resamples) "
            f"{u.interval(b.ci_low, b.ci_high)}\n"
        )
    _emit(args, text)
    return EXIT_OK


def _distinct_names(
    base: EvalRun, cand: EvalRun, base_path: Path, cand_path: Path
) -> tuple[EvalRun, EvalRun]:
    """``main/qa.jsonl`` vs ``pr/qa.jsonl`` should not both be labelled ``qa``."""
    if base.name != cand.name:
        return base, cand
    b_name = f"{base_path.resolve().parent.name}/{base.name}"
    c_name = f"{cand_path.resolve().parent.name}/{cand.name}"
    if b_name == c_name:
        b_name, c_name = f"{base.name} (baseline)", f"{cand.name} (candidate)"
    return replace(base, name=b_name), replace(cand, name=c_name)


def _comparison(args: argparse.Namespace) -> tuple[Any, EvalRun, EvalRun]:
    base, cand = _distinct_names(
        _load(args.baseline, args), _load(args.candidate, args), args.baseline, args.candidate
    )
    c = compare(base, cand, confidence=args.confidence, cluster=not args.no_cluster)
    return c, base, cand


def _bootstrap_line(args: argparse.Namespace, base: EvalRun, cand: EvalRun, c: Any) -> str:
    b = bootstrap_diff(
        base,
        cand,
        n_resamples=args.bootstrap,
        confidence=args.confidence,
        cluster=not args.no_cluster,
        seed=args.seed,
    )
    u = Units.for_estimates(c.baseline, c.candidate)
    return (
        f"  · bootstrap ({b.method}, {fmt_int(b.n_resamples)} resamples): "
        f"{u.interval(b.ci_low, b.ci_high, signed=True)}, p ≈ {fmt_p(b.p_value)}\n"
    )


def cmd_compare(args: argparse.Namespace) -> int:
    c, base, cand = _comparison(args)
    text = render_comparison(c, args.fmt, _style(args))
    if args.bootstrap and args.fmt == "text":
        text += _bootstrap_line(args, base, cand, c)
    _emit(args, text)
    if args.markdown:
        args.markdown.write_text(render_comparison(c, "markdown"), encoding="utf-8")
    return EXIT_OK


def cmd_gate(args: argparse.Namespace) -> int:
    c, base, cand = _comparison(args)
    policy = GatePolicy(
        fail_on_regression=not args.allow_regression,
        max_regression=args.max_regression,
        require_improvement=args.require_improvement,
    )
    g = evaluate_gate(c, policy)
    text = render_comparison(c, args.fmt, _style(args), gate=g)
    if args.bootstrap and args.fmt == "text":
        text += _bootstrap_line(args, base, cand, c)
    _emit(args, text)
    if args.markdown:
        args.markdown.write_text(render_comparison(c, "markdown", gate=g), encoding="utf-8")
    return EXIT_OK if g.passed else EXIT_GATE_FAILED


def cmd_suite(args: argparse.Namespace) -> int:
    pairs, unmatched = load_suite(
        args.baseline_dir,
        args.candidate_dir,
        id_field=args.id_field,
        score_field=args.score_field,
        cluster_field=args.cluster_field,
    )
    r = compare_suite(
        pairs, correction=args.correction, alpha=1 - args.confidence, cluster=not args.no_cluster
    )
    _emit(args, render_suite(r, args.fmt, _style(args), unmatched))
    if args.markdown:
        args.markdown.write_text(render_suite(r, "markdown", unmatched=unmatched), encoding="utf-8")
    if args.fail_on_regression and r.regressions:
        return EXIT_GATE_FAILED
    return EXIT_OK


def _read_summary_csv(path: Path) -> list[tuple[str, float, int]]:
    rows = read_rows(path)
    if not rows:
        raise DataError(f"{path}: empty")
    keys = {k.lower(): k for k in rows[0]}

    def pick(*names: str) -> str:
        for n in names:
            if n in keys:
                return keys[n]
        raise DataError(f"{path}: needs a column named one of {', '.join(names)}")

    name_k = pick("model", "name", "system")
    score_k = pick("score", "accuracy", "acc", "pass@1", "mean")
    n_k = pick("n", "questions", "num_questions", "count")
    out = []
    for r in rows:
        try:
            out.append((str(r[name_k]), float(r[score_k]), int(float(r[n_k]))))
        except (TypeError, ValueError):
            raise DataError(f"{path}: could not parse row {r}") from None
    return out


def cmd_leaderboard(args: argparse.Namespace) -> int:
    if args.summary and args.runs:
        raise DataError("pass either result files or --summary, not both")
    if args.summary:
        lb = from_summary(
            _read_summary_csv(args.summary), confidence=args.confidence, seed=args.seed
        )
    else:
        if len(args.runs) < 2:
            raise DataError("pass at least two result files (or --summary scores.csv)")
        runs = [_load(p, args) for p in args.runs]
        lb = from_runs(
            runs, confidence=args.confidence, cluster=not args.no_cluster, seed=args.seed
        )
    _emit(args, render_leaderboard(lb, args.fmt, _style(args)))
    return EXIT_OK


def cmd_power(args: argparse.Namespace) -> int:
    deff = 1.0
    deff_note = None
    if args.cluster_size is not None or args.icc is not None:
        if args.cluster_size is None or args.icc is None:
            raise DataError("--cluster-size and --icc must be given together")
        deff = design_effect(args.cluster_size, args.icc)
        deff_note = (
            f"clusters of {args.cluster_size:g} questions with ICC {args.icc:g} "
            f"inflate variance {deff:.2f}× (design effect)"
        )
    alpha = 1 - args.confidence
    power = args.power
    u = Units(percent=args.sd is None)

    if args.pilot:
        if args.effect is None:
            raise DataError("--pilot needs --effect")
        base, cand = _load(args.pilot[0], args), _load(args.pilot[1], args)
        c = compare(base, cand, confidence=args.confidence, cluster=not args.no_cluster)
        plan = plan_from_comparison(c, args.effect, alpha=alpha, power=power)
        table = pilot_table(plan, percent=Units.for_estimates(c.baseline).percent)
        _emit(args, render_power(table, args.fmt, _style(args)))
        return EXIT_OK

    if args.margin is not None:
        rate = args.baseline_rate if args.baseline_rate is not None else 0.5
        n = sample_size_for_margin(
            args.margin, rate=rate, sd=args.sd, confidence=args.confidence, design_effect=deff
        )
        score_desc = u.score(rate) if args.sd is None else f"sd {args.sd:g}"
        table = PowerTable(
            title=f"questions for a ±{u.size(args.margin)} error bar",
            headers=["score", "confidence", "questions"],
            rows=[[score_desc, f"{args.confidence:.0%}", fmt_int(n)]],
            notes=[deff_note] if deff_note else [],
        )
        _emit(args, render_power(table, args.fmt, _style(args)))
        return EXIT_OK

    if args.sd is None and args.baseline_rate is None:
        raise DataError(
            "pass --baseline-rate (binary scores) or --sd (of per-question differences)"
        )
    if args.sd is not None:
        corrs: tuple[float, ...] = (0.0,)
        base_desc = f"sd {args.sd:g}"
        notes = []
    else:
        corrs = (args.correlation,) if args.correlation is not None else CORRELATIONS
        base_desc = f"baseline {u.score(args.baseline_rate)}"
        notes = [CORRELATION_NOTE]
    if deff_note:
        notes.append(deff_note)
    opts: dict[str, Any] = {
        "sd": args.sd,
        "baseline_rate": args.baseline_rate,
        "alpha": alpha,
        "design_effect": deff,
    }

    if args.effect is not None and args.n is not None:
        rows = [
            [f"{r:.1f}", f"{achieved_power(args.n, args.effect, correlation=r, **opts):.0%}"]
            for r in corrs
        ]
        title = f"power to detect {u.delta(args.effect)} with {fmt_int(args.n)} questions"
        table = PowerTable(
            title=f"{title} ({base_desc})",
            headers=["correlation", "power"],
            rows=rows,
            notes=notes,
        )
    elif args.effect is not None:
        rows = [
            [f"{r:.1f}", fmt_int(sample_size(args.effect, correlation=r, power=power, **opts))]
            for r in corrs
        ]
        table = PowerTable(
            title=f"questions to detect {u.delta(args.effect)} "
            f"({base_desc}, {power:.0%} power, α = {alpha:g})",
            headers=["correlation", "questions"],
            rows=rows,
            notes=notes,
        )
    elif args.n is not None:
        rows = []
        for r in corrs:
            mde = minimum_detectable_effect(args.n, correlation=r, power=power, **opts)
            rows.append([f"{r:.1f}", "not detectable" if math.isinf(mde) else u.size(mde)])
        table = PowerTable(
            title=f"smallest detectable effect with {fmt_int(args.n)} questions "
            f"({base_desc}, {power:.0%} power)",
            headers=["correlation", "min. detectable effect"],
            rows=rows,
            notes=notes,
        )
    else:
        raise DataError(
            "pass --effect (sample size), --n (detectable effect), both (power), "
            "--margin, or --pilot"
        )
    _emit(args, render_power(table, args.fmt, _style(args)))
    return EXIT_OK


# --------------------------------------------------------------------------- parser


EPILOG = """\
examples:
  noisefloor summary results.jsonl
  noisefloor compare baseline.jsonl candidate.jsonl --bootstrap
  noisefloor gate baseline.jsonl candidate.jsonl --max-regression 2pp --markdown comment.md
  noisefloor suite runs/main/ runs/pr-123/ --correction holm
  noisefloor power --effect 2pp --baseline-rate 70%
  noisefloor power --n 198 --baseline-rate 60%
  noisefloor leaderboard --summary scores.csv
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noisefloor",
        description="Is your eval improvement real, or just noise?",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"noisefloor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("summary", help="score with an honest error bar")
    p.add_argument("run", type=Path)
    _stat_args(p)
    _data_args(p)
    _output_args(p)
    p.set_defaults(func=cmd_summary)

    for name, helptext, func in (
        ("compare", "paired comparison of two runs on the same eval", cmd_compare),
        ("gate", "compare and exit 1 if the change fails the policy (for CI)", cmd_gate),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("baseline", type=Path)
        p.add_argument("candidate", type=Path)
        if name == "gate":
            g = p.add_argument_group("policy")
            g.add_argument(
                "--max-regression",
                type=parse_amount,
                metavar="AMOUNT",
                help="fail unless a regression larger than AMOUNT (e.g. 1pp) can be ruled out",
            )
            g.add_argument(
                "--require-improvement",
                action="store_true",
                help="fail unless the candidate is significantly better",
            )
            g.add_argument(
                "--allow-regression",
                action="store_true",
                help="do not fail on a significant regression",
            )
        _stat_args(p)
        _data_args(p)
        _output_args(p, markdown_file=True)
        p.set_defaults(func=func)

    p = sub.add_parser("suite", help="compare across many benchmarks with multiplicity control")
    p.add_argument("baseline_dir", type=Path)
    p.add_argument("candidate_dir", type=Path)
    p.add_argument("--correction", choices=CORRECTIONS, default="holm")
    p.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any benchmark regresses after correction",
    )
    p.add_argument("-c", "--confidence", type=parse_confidence, default=0.95)
    _data_args(p)
    _output_args(p, markdown_file=True)
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("leaderboard", help="rank models with rank CIs and statistical tiers")
    p.add_argument("runs", type=Path, nargs="*", help="per-question result files, one per model")
    p.add_argument("--summary", type=Path, help="CSV/JSONL with columns model, score, n")
    p.add_argument("-c", "--confidence", type=parse_confidence, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    _data_args(p)
    _output_args(p)
    p.set_defaults(func=cmd_leaderboard)

    p = sub.add_parser("power", help="how many questions do you need?")
    p.add_argument("--effect", type=parse_amount, help="effect to detect, e.g. 2pp")
    p.add_argument("--n", type=int, help="number of questions you have")
    p.add_argument("--margin", type=parse_amount, help="target ± error bar for a single score")
    p.add_argument("--baseline-rate", type=parse_rate, help="baseline accuracy, e.g. 70%%")
    p.add_argument("--sd", type=float, help="SD of per-question differences (continuous scores)")
    p.add_argument("--correlation", type=float, help="per-question correlation between models")
    p.add_argument("--cluster-size", type=float, help="average questions per cluster")
    p.add_argument("--icc", type=float, help="intra-cluster correlation")
    p.add_argument("--power", type=parse_rate, default=0.8, help="default: 80%%")
    p.add_argument("-c", "--confidence", type=parse_confidence, default=0.95)
    p.add_argument(
        "--pilot",
        type=Path,
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="plan from an existing pair of runs",
    )
    _data_args(p)
    _output_args(p)
    p.set_defaults(func=cmd_power)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cluster_field", None) == "":
        args.cluster_field = None
        args.no_cluster = True
    try:
        code: int = args.func(args)
    except (DataError, ValueError) as exc:
        print(f"noisefloor: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"noisefloor: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
