import json
from pathlib import Path

import numpy as np
import pytest

from noisefloor.cli import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK, main, parse_amount, parse_rate


def _write(path: Path, scores: list[int], clusters: list[str] | None = None) -> Path:
    rows = []
    for i, s in enumerate(scores):
        row: dict[str, object] = {"id": f"q{i}", "correct": bool(s)}
        if clusters is not None:
            row["cluster"] = clusters[i]
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def runs(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    p = rng.beta(1, 1, 300)
    base = (rng.random(300) < p).astype(int).tolist()
    noisy = (rng.random(300) < p).astype(int).tolist()
    worse = (rng.random(300) < np.clip(p - 0.3, 0, 1)).astype(int).tolist()
    return {
        "base": _write(tmp_path / "base.jsonl", base),
        "noisy": _write(tmp_path / "noisy.jsonl", noisy),
        "worse": _write(tmp_path / "worse.jsonl", worse),
    }


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def test_parse_amount_and_rate() -> None:
    assert parse_amount("2pp") == pytest.approx(0.02)
    assert parse_amount("2%") == pytest.approx(0.02)
    assert parse_amount("0.02") == pytest.approx(0.02)
    assert parse_rate("70%") == pytest.approx(0.7)
    assert parse_rate("70") == pytest.approx(0.7)
    assert parse_rate("0.7") == pytest.approx(0.7)


def test_summary_text_and_json(runs: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["summary", str(runs["base"]), "--no-color"], capsys)
    assert code == EXIT_OK
    assert "95% CI" in out
    code, out, _ = run(["summary", str(runs["base"]), "-f", "json"], capsys)
    data = json.loads(out)
    assert data["estimate"]["n"] == 300
    assert "wilson_ci" in data


def test_summary_bootstrap(runs: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["summary", str(runs["base"]), "--bootstrap", "500"], capsys)
    assert code == EXIT_OK
    assert "bootstrap" in out


def test_compare_formats(
    runs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    md = tmp_path / "out.md"
    args = ["compare", str(runs["base"]), str(runs["noisy"]), "--markdown", str(md)]
    code, out, _ = run(args, capsys)
    assert code == EXIT_OK
    assert "INCONCLUSIVE" in out
    text = md.read_text(encoding="utf-8")
    assert text.startswith("<!-- noisefloor-report -->")
    assert "| **Δ** |" in text
    code, out, _ = run(["compare", str(runs["base"]), str(runs["worse"]), "-f", "json"], capsys)
    assert json.loads(out)["verdict"] == "regression"


def test_gate_exit_codes(runs: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    base, noisy, worse = str(runs["base"]), str(runs["noisy"]), str(runs["worse"])
    assert run(["gate", base, noisy], capsys)[0] == EXIT_OK
    code, out, _ = run(["gate", base, worse], capsys)
    assert code == EXIT_GATE_FAILED
    assert "FAIL" in out
    assert run(["gate", base, worse, "--allow-regression"], capsys)[0] == EXIT_OK
    assert run(["gate", base, noisy, "--max-regression", "0.5pp"], capsys)[0] == EXIT_GATE_FAILED
    assert run(["gate", base, noisy, "--require-improvement"], capsys)[0] == EXIT_GATE_FAILED
    code, out, _ = run(["gate", base, worse, "-f", "json"], capsys)
    assert json.loads(out)["gate"]["passed"] is False


def test_output_file(
    runs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "report.txt"
    code, out, _ = run(
        ["compare", str(runs["base"]), str(runs["noisy"]), "-o", str(target)], capsys
    )
    assert code == EXIT_OK
    assert out == ""
    assert "\033[" not in target.read_text(encoding="utf-8")


def test_suite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for side in ("a", "b"):
        (tmp_path / side).mkdir()
    rng = np.random.default_rng(1)
    for bench in ("x", "y"):
        p = rng.beta(1, 1, 200)
        _write(tmp_path / "a" / f"{bench}.jsonl", (rng.random(200) < p).astype(int).tolist())
        shift = -0.4 if bench == "y" else 0.0
        b = (rng.random(200) < np.clip(p + shift, 0, 1)).astype(int).tolist()
        _write(tmp_path / "b" / f"{bench}.jsonl", b)
    args = ["suite", str(tmp_path / "a"), str(tmp_path / "b")]
    code, out, _ = run(args, capsys)
    assert code == EXIT_OK
    assert "Holm" in out
    assert run([*args, "--fail-on-regression"], capsys)[0] == EXIT_GATE_FAILED
    code, out, _ = run([*args, "-f", "json", "--correction", "bh"], capsys)
    assert json.loads(out)["regressions"] == ["y"]


def test_leaderboard(
    runs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(["leaderboard", *map(str, runs.values())], capsys)
    assert code == EXIT_OK
    assert "tier" in out
    csv_path = tmp_path / "lb.csv"
    csv_path.write_text("model,accuracy,n\nA,70.2,500\nB,69.8,500\nC,40.0,500\n", encoding="utf-8")
    code, out, _ = run(["leaderboard", "--summary", str(csv_path), "-f", "markdown"], capsys)
    assert code == EXIT_OK
    assert "| Tier |" in out
    code, _, err = run(["leaderboard", str(runs["base"])], capsys)
    assert code == EXIT_ERROR
    assert "at least two" in err


def test_power_modes(runs: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["power", "--effect", "2pp", "--baseline-rate", "70%"], capsys)
    assert code == EXIT_OK
    assert "8,077" in out
    code, out, _ = run(["power", "--n", "198", "--baseline-rate", "60%", "-f", "json"], capsys)
    assert len(json.loads(out)["rows"]) == 4
    code, out, _ = run(["power", "--n", "500", "--effect", "3pp", "--baseline-rate", "0.5"], capsys)
    assert "power" in out
    code, out, _ = run(["power", "--margin", "3pp"], capsys)
    assert "1,068" in out
    code, out, _ = run(["power", "--effect", "0.5", "--sd", "1"], capsys)
    assert "32" in out
    args = ["power", "--pilot", str(runs["base"]), str(runs["noisy"]), "--effect", "2pp"]
    code, out, _ = run(args, capsys)
    assert code == EXIT_OK
    assert "pilot" in out
    args = [
        "power",
        "--effect",
        "2pp",
        "--baseline-rate",
        "70%",
        "--cluster-size",
        "5",
        "--icc",
        "0.1",
    ]
    code, out, _ = run(args, capsys)
    assert "design effect" in out


def test_errors_are_reported_not_raised(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["summary", str(tmp_path / "nope.jsonl")], capsys)
    assert code == EXIT_ERROR
    assert "not found" in err
    code, _, err = run(["power"], capsys)
    assert code == EXIT_ERROR
    assert "--baseline-rate" in err
    code, _, err = run(
        ["power", "--cluster-size", "4", "--baseline-rate", "0.5", "--n", "9"], capsys
    )
    assert "together" in err


def test_clustered_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scores = [1, 1, 1, 0, 0, 0] * 20
    clusters = [f"c{i // 3}" for i in range(120)]
    path = _write(tmp_path / "c.jsonl", scores, clusters)
    code, out, _ = run(["summary", str(path)], capsys)
    assert code == EXIT_OK
    assert "clustered" in out
    assert "inflates variance" in out
    _code, out, _ = run(["summary", str(path), "--cluster-field", ""], capsys)
    assert "iid" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "noisefloor" in capsys.readouterr().out
