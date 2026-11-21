# Contributing to noisefloor

Thanks for helping make evals more honest.

## Development setup

```bash
git clone https://github.com/Gasooom/noisefloor
cd noisefloor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff format .
ruff check .
mypy                     # strict mode
pytest -m "not slow"     # fast unit tests (~1s)
pytest -m slow           # Monte Carlo validation (~30s)
```

CI runs all of the above on Python 3.10 to 3.13.

## Ground rules

- **Statistical changes need a simulation.** A unit test shows the code does
  what you meant; a Monte Carlo check in `tests/test_validation.py` shows that
  what you meant is true (coverage, error rates, power). Add one for any new
  estimator, interval, or test.
- **Cite your method.** Add the formula and a reference to
  `docs/methodology.md`.
- **Keep the dependency footprint at numpy.** If you need a special function,
  implement it in `_dist.py` and test it against published values.
- **Every number gets an interval.** Reports must never show a bare point
  estimate.
- **Errors should say what to do.** Prefer `"could not find a score column
  (tried ...); pass --score-field"` over a stack trace.

## Regenerating the example data

```bash
python examples/generate.py
python examples/validate.py --sims 4000   # refresh the README validation table
```
