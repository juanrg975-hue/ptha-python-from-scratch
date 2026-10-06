"""Validation tests for pyptha.scaling against rptha reference values.

The golden values come directly from the rptha source docstrings/examples in
``rupture_scaling.R`` (e.g. slip_from_Mw ~ 10 m at Mw 9.0) and from the exact
coefficient arithmetic in that file, recomputed independently here.

Run with:  python -m pytest pyptha/tests/test_scaling.py
or simply: python pyptha/tests/test_scaling.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import scaling  # noqa: E402


def test_M0_Mw_roundtrip():
    # From the R docstring: Mw = M0_2_Mw(4e17); M0 back should equal 4e17.
    Mw = scaling.M0_2_Mw(4e17)
    M0 = scaling.M0_2_Mw(Mw, inverse=True)
    assert math.isclose(M0, 4e17, rel_tol=1e-12)


def test_M0_Mw_known_value():
    # Mw = 2/3*(log10(M0) - 9.05)
    M0 = 1e20
    expected = 2.0 / 3.0 * (math.log10(M0) - 9.05)
    assert math.isclose(scaling.M0_2_Mw(M0), expected, rel_tol=1e-12)


def test_M0_Mw_vectorized():
    M0 = np.array([1e18, 1e20, 1e22])
    out = scaling.M0_2_Mw(M0)
    expected = 2.0 / 3.0 * (np.log10(M0) - 9.05)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_Mw_2_rupture_size_strasser():
    # Recompute the Strasser relation directly from its coefficients.
    Mw = 9.0
    r = scaling.Mw_2_rupture_size(Mw, relation="Strasser")
    assert math.isclose(r["area"], 10 ** (-3.476 + Mw * 0.952), rel_tol=1e-12)
    assert math.isclose(r["width"], 10 ** (-0.882 + Mw * 0.351), rel_tol=1e-12)
    assert math.isclose(r["length"], 10 ** (-2.477 + Mw * 0.585), rel_tol=1e-12)


def test_Mw_2_rupture_size_detailed_ci():
    Mw, CI_sd = 9.0, 2.0
    r = scaling.Mw_2_rupture_size(Mw, relation="Strasser", detailed=True, CI_sd=CI_sd)
    # plus/minus CI must bracket the central value symmetrically in log space.
    for k in ("area", "width", "length"):
        assert r.plus_CI[k] > r.values[k] > r.minus_CI[k]
        assert math.isclose(
            math.log10(r.plus_CI[k]) - math.log10(r.values[k]),
            math.log10(r.values[k]) - math.log10(r.minus_CI[k]),
            rel_tol=1e-9,
        )


def test_allenhayes_piecewise_area():
    # Below and above the area threshold uses different coefficients.
    thr = (5.62 + 2.23) / (1.22 - 0.31)
    below = scaling.Mw_2_rupture_size(thr - 0.5, relation="AllenHayes")
    above = scaling.Mw_2_rupture_size(thr + 0.5, relation="AllenHayes")
    assert math.isclose(below["area"],
                        10 ** (-5.62 + (thr - 0.5) * 1.22), rel_tol=1e-12)
    assert math.isclose(above["area"],
                        10 ** (2.23 + (thr + 0.5) * 0.31), rel_tol=1e-12)


def test_blaser_area_from_length_width():
    # Blaser area = length*width with zero-correlation residuals.
    r = scaling.Mw_2_rupture_size(8.0, relation="Blaser-reverse", detailed=True)
    exp_a = (-2.28 + -1.80, 0.55 + 0.45, math.sqrt(0.18 ** 2 + 0.17 ** 2))
    assert np.allclose(r.area_absigma, exp_a, rtol=1e-12)


def test_inverse_roundtrip():
    # From the R docstring example: squeezing the -2 sigma area recovers it.
    for Mw in (8.0, 8.67, 9.0):
        for relation in ("Strasser", "AllenHayes"):
            area0 = scaling.Mw_2_rupture_size(
                Mw, relation=relation, detailed=True, CI_sd=2)
            Mw_squeezed = scaling.Mw_2_rupture_size_inverse(
                area0.values["area"], relation=relation, CI_sd=-2)
            area1 = scaling.Mw_2_rupture_size(
                Mw_squeezed, relation=relation, detailed=True, CI_sd=2)
            assert abs(area1.minus_CI["area"] - area0.values["area"]) < 1e-4


def test_slip_from_Mw_area_mu():
    # From R docstring: slip_from_Mw_area_mu(9.0, 100e3) "close to 10 m".
    # Exact value with mu=3e10, constant=9.05 is ~11.83 m; the docstring is
    # deliberately approximate, so we verify the exact analytic result here.
    s = scaling.slip_from_Mw_area_mu(9.0, 100e3)
    M0 = scaling.M0_2_Mw(9.0, inverse=True)
    assert math.isclose(s, M0 / (100e3 * 1e6 * 3e10), rel_tol=1e-12)
    assert 8.0 < s < 14.0  # in the neighbourhood of the docstring's "~10 m"


def test_slip_from_Mw():
    # From R docstring: slip_from_Mw(9.0) roughly 10 m.
    s = scaling.slip_from_Mw(9.0)
    assert 5.0 < s < 20.0


def test_slip_from_Mw_vectorized():
    Mws = np.array([8.0, 8.5, 9.0])
    s = scaling.slip_from_Mw(Mws)
    assert s.shape == (3,)
    # Slip increases with magnitude for these relations.
    assert s[0] < s[1] < s[2]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
