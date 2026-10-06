"""Validation tests for pyptha.unit_sources.

Validated against the analytic geometry of an idealised planar interface
built by make_planar_unit_source_grid: the summary-statistics routine must
recover the strike, dip, unit-source length/width and count that went in.

Run:  python pyptha/tests/test_unit_sources.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import unit_sources as us  # noqa: E402


def _build(strike=0.0, dip=20.0, na=6, nd=4, L=50.0, W=40.0, top=0.0):
    grid = us.make_planar_unit_source_grid(
        lon0=165.0, lat0=-45.0, strike=strike, dip=dip,
        n_alongstrike=na, n_downdip=nd,
        subfault_length=L, subfault_width=W, top_depth=top)
    return grid


def test_grid_shape():
    grid = _build(na=6, nd=4)
    assert grid.shape == (4 + 1, 3, 6 + 1)


def test_source_count():
    stats = us.discretized_source_approximate_summary_statistics(_build(na=6, nd=4))
    assert stats["subfault_number"].size == 6 * 4
    assert set(stats["downdip_number"]) == {1, 2, 3, 4}
    assert set(stats["alongstrike_number"]) == {1, 2, 3, 4, 5, 6}


def test_recovers_dip():
    # The summary routine measures dip with ellipsoidal surface distances
    # (distHaversine in R), while the planar builder places vertices on the
    # WGS84 ellipsoid; the small mismatch (< ~1 deg on a 20 deg interface) is
    # geodetic, not a port error. We check the dip is recovered to ~1 deg.
    for dip_in in (10.0, 20.0, 35.0):
        stats = us.discretized_source_approximate_summary_statistics(
            _build(dip=dip_in))
        assert np.all(np.abs(stats["dip"] - dip_in) < 1.5)
        # And the mean is very close to the target.
        assert abs(stats["dip"].mean() - dip_in) < 1.0


def test_recovers_strike():
    for strike_in in (0.0, 45.0, 120.0):
        stats = us.discretized_source_approximate_summary_statistics(
            _build(strike=strike_in))
        # Strike recovered within a couple of degrees (great-circle bearing).
        d = np.abs((stats["strike"] - strike_in + 180) % 360 - 180)
        assert np.all(d < 3.0)


def test_recovers_subfault_dimensions():
    L, W = 50.0, 40.0
    stats = us.discretized_source_approximate_summary_statistics(
        _build(L=L, W=W))
    # Length/width should be close to the requested unit-source size.
    assert np.allclose(stats["length"], L, rtol=0.05)
    assert np.allclose(stats["width"], W, rtol=0.05)


def test_depth_increases_downdip():
    stats = us.discretized_source_approximate_summary_statistics(
        _build(dip=25.0, nd=4, W=40.0))
    # Deeper down-dip rows must have larger centroid depth.
    for a in np.unique(stats["alongstrike_number"]):
        sel = stats["alongstrike_number"] == a
        d = stats["depth"][sel]
        order = np.argsort(stats["downdip_number"][sel])
        assert np.all(np.diff(d[order]) > 0)


def test_area_matches_length_times_width():
    stats = us.discretized_source_approximate_summary_statistics(_build())
    np.testing.assert_allclose(
        stats["area"], stats["length"] * stats["width"], rtol=1e-12)


def test_total_area_reasonable():
    # Total interface area ~ (na*L) x (nd*W) for a planar patch.
    na, nd, L, W = 6, 4, 50.0, 40.0
    stats = us.discretized_source_approximate_summary_statistics(
        _build(na=na, nd=nd, L=L, W=W, dip=15.0))
    total = stats["area"].sum()
    nominal = (na * L) * (nd * W)
    assert 0.85 * nominal < total < 1.15 * nominal


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
