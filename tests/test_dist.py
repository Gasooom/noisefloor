import math

import pytest

from noisefloor._dist import (
    betainc,
    critical_value,
    norm_ppf,
    t_cdf,
    t_ppf,
    t_sf_two_sided,
)

# Reference values from standard statistical tables (e.g. NIST/SEMATECH handbook).
T_TABLE = [
    (1, 0.975, 12.706204736),
    (2, 0.975, 4.302652730),
    (5, 0.975, 2.570581836),
    (10, 0.975, 2.228138852),
    (29, 0.975, 2.045229642),
    (30, 0.995, 2.749995653),
    (100, 0.95, 1.660234326),
    (1000, 0.975, 1.962339081),
]


@pytest.mark.parametrize(("df", "p", "expected"), T_TABLE)
def test_t_ppf_matches_tables(df: int, p: float, expected: float) -> None:
    assert t_ppf(p, df) == pytest.approx(expected, rel=1e-7)
    assert t_ppf(1 - p, df) == pytest.approx(-expected, rel=1e-7)


@pytest.mark.parametrize(("df", "p", "q"), T_TABLE)
def test_t_cdf_inverts_ppf(df: int, p: float, q: float) -> None:
    assert t_cdf(q, df) == pytest.approx(p, abs=1e-9)


def test_t_cdf_cauchy_closed_form() -> None:
    # df=1 is the Cauchy distribution: CDF = 1/2 + atan(t)/pi
    for t in [-5.0, -1.0, -0.3, 0.0, 0.7, 2.0, 10.0]:
        assert t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-12)


def test_t_two_sided_consistent_with_cdf() -> None:
    for df in [3, 12, 80]:
        for t in [0.1, 1.0, 2.5, 4.0]:
            assert t_sf_two_sided(t, df) == pytest.approx(2 * (1 - t_cdf(t, df)), abs=1e-10)
            assert t_sf_two_sided(-t, df) == pytest.approx(t_sf_two_sided(t, df))
    assert t_sf_two_sided(0.0, 5) == 1.0


def test_large_df_converges_to_normal() -> None:
    assert t_ppf(0.975, 1e9) == pytest.approx(norm_ppf(0.975))
    assert t_ppf(0.975, 1e5) == pytest.approx(1.959983, abs=1e-4)


def test_betainc_edges_and_symmetry() -> None:
    assert betainc(2, 3, 0.0) == 0.0
    assert betainc(2, 3, 1.0) == 1.0
    for a, b, x in [(0.5, 0.5, 0.3), (2.0, 5.0, 0.2), (10.0, 1.5, 0.9)]:
        assert betainc(a, b, x) == pytest.approx(1 - betainc(b, a, 1 - x), abs=1e-12)
    # I_x(1, 1) is the uniform CDF
    assert betainc(1, 1, 0.37) == pytest.approx(0.37)


def test_critical_value() -> None:
    assert critical_value(0.95, None) == pytest.approx(1.959964, abs=1e-6)
    assert critical_value(0.95, 10) == pytest.approx(2.228138852, rel=1e-7)
    with pytest.raises(ValueError, match="confidence"):
        critical_value(1.5, 10)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probability"):
        t_ppf(0.0, 5)
    with pytest.raises(ValueError, match="degrees"):
        t_cdf(1.0, 0)
