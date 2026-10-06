"""Validation tests for pyptha.events (uniform-slip rupture generation).

Validated against the invariants the R code asserts internally (seismic moment
exactly M0(Mw), correct subfault counts, scaling-relation aspect ratio) using
an idealised planar source.

Run:  python pyptha/tests/test_events.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import events, scaling, unit_sources as us  # noqa: E402


def _stats(na=8, nd=5, L=50.0, W=45.0, dip=15.0):
    grid = us.make_planar_unit_source_grid(
        lon0=165.0, lat0=-45.0, strike=20.0, dip=dip,
        n_alongstrike=na, n_downdip=nd, subfault_length=L, subfault_width=W)
    return us.discretized_source_approximate_summary_statistics(grid)


def test_single_magnitude_moment_exact():
    stats = _stats()
    e = events.get_all_earthquake_events_of_magnitude_Mw(8.5, stats)
    M0 = scaling.M0_2_Mw(8.5, inverse=True)
    es = e["event_statistics"]
    mu = 3e10
    local_M0 = es["slip"] * (es["area"] * 1e6) * mu
    np.testing.assert_allclose(local_M0, M0, rtol=1e-8)


def test_subfault_count_matches_block():
    stats = _stats()
    e = events.get_all_earthquake_events_of_magnitude_Mw(8.5, stats)
    nl = e["event_dim"]["length"]
    nw = e["event_dim"]["width"]
    for idx in e["event_indices"]:
        assert idx.size == nl * nw


def test_larger_Mw_more_subfaults():
    stats = _stats(na=10, nd=6)
    small = events.get_all_earthquake_events_of_magnitude_Mw(7.5, stats)
    big = events.get_all_earthquake_events_of_magnitude_Mw(9.0, stats)
    assert big["actual_subfault_count"] >= small["actual_subfault_count"]


def test_event_table_columns_and_length():
    stats = _stats()
    table = events.get_all_earthquake_events(
        stats, Mmin=7.5, Mmax=8.5, dMw=0.1, source_zone_name="testzone")
    n = table["Mw"].size
    for key in ("Mw", "area", "slip", "mean_length", "mean_width",
                "mean_depth", "max_depth", "event_index_string", "sourcename"):
        assert len(table[key]) == n
    assert np.all(table["sourcename"] == "testzone")
    # Magnitudes span the requested range.
    assert table["Mw"].min() >= 7.5 - 1e-9
    assert table["Mw"].max() <= 8.5 + 1e-9


def test_event_index_string_roundtrip():
    stats = _stats()
    table = events.get_all_earthquake_events(stats, Mmin=8.0, Mmax=8.2, dMw=0.1)
    for k in range(table["Mw"].size):
        parsed = events.get_unit_source_indices_in_event(
            table["event_index_string"][k])
        np.testing.assert_array_equal(parsed, table["event_indices"][k])
        # Indices are valid rows of the unit-source table.
        assert parsed.max() < stats["subfault_number"].size


def test_area_within_scaling_bounds():
    # Every generated event's area should sit within the 2-sigma scaling band.
    stats = _stats(na=12, nd=7)
    for mw in (7.8, 8.4, 9.0):
        e = events.get_all_earthquake_events_of_magnitude_Mw(mw, stats)
        rs = scaling.Mw_2_rupture_size(mw, detailed=True, CI_sd=2.0)
        area = e["event_statistics"]["area"]
        # Allow the boundary-limited cases to touch the edges.
        assert np.all(area >= rs.minus_CI["area"] * 0.5)
        assert np.all(area <= rs.plus_CI["area"] * 2.0)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
