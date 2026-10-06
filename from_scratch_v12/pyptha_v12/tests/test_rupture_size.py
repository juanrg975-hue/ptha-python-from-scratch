"""v10: events.get_all_earthquake_events(_of_magnitude_Mw)'s rupture_size.

Two synthetic meshes, so the tests need no data file:
  * a uniform planar mesh (50 x 45 km cells, rptha's assumption holds);
  * a calabria2-like mesh: 20 x 5 cells whose length shrinks from 50 km at
    the trench to 14 km at depth and whose width changes 28 -> 72 -> 28 km
    along strike, so the cell area spans a factor ~10.

Run:  .venv/Scripts/python.exe -m pytest from_scratch_v12/pyptha_v12/tests/test_rupture_size.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import events, scaling, unit_sources as us  # noqa: E402

MWS = np.round(np.arange(7.2, 9.61, 0.1), 2)


def _uniform():
    grid = us.make_planar_unit_source_grid(
        lon0=165.0, lat0=-45.0, strike=20.0, dip=15.0,
        n_alongstrike=12, n_downdip=4, subfault_length=50.0, subfault_width=45.0)
    return us.discretized_source_approximate_summary_statistics(grid)


def _uneven(nd=5, na=20):
    """Summary statistics of an uneven mesh, down-dip index fastest."""
    dd, asn = np.meshgrid(np.arange(1, nd + 1), np.arange(1, na + 1), indexing="xy")
    dd, asn = dd.ravel(), asn.ravel()
    length = np.interp(dd, [1, nd], [50.0, 14.0])
    width = 28.0 + 44.0 * np.sin(np.pi * (asn - 0.5) / na)
    depth = 5.0 + 10.0 * (dd - 1)
    return {"length": length, "width": width, "depth": depth,
            "max_depth": depth + 5.0, "downdip_number": dd,
            "alongstrike_number": asn,
            "subfault_number": np.arange(1, dd.size + 1),
            "lon_c": asn * 0.3, "lat_c": -dd * 0.3}


def _area_ratio(e):
    return e["event_statistics"]["area"] / e["desired_ALW"]["area"]


def test_default_is_rptha():
    stats = _uneven()
    for mw in (7.2, 8.0, 8.8):
        a = events.get_all_earthquake_events_of_magnitude_Mw(mw, stats)
        b = events.get_all_earthquake_events_of_magnitude_Mw(
            mw, stats, rupture_size="rptha")
        assert a["topleft_indices"] == b["topleft_indices"]
        assert a["event_dim"] == b["event_dim"]
        assert len({(d["length"], d["width"]) for d in a["event_dims"]}) == 1


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        events.get_all_earthquake_events_of_magnitude_Mw(7.5, _uneven(),
                                                         rupture_size="mean")


@pytest.mark.parametrize("make", [_uniform, _uneven])
def test_local_blocks_fit_cover_and_keep_moment(make):
    stats = make()
    ncell = stats["length"].size
    for mw in MWS:
        e = events.get_all_earthquake_events_of_magnitude_Mw(
            mw, stats, rupture_size="local")
        assert e["event_dim"] is None
        covered = np.zeros(ncell, bool)
        keys = set()
        for idx, dim in zip(e["event_indices"], e["event_dims"]):
            assert idx.size == dim["length"] * dim["width"]
            covered[idx] = True
            keys.add(tuple(idx))
        assert covered.all(), f"Mw {mw}: some cells are in no rupture"
        assert len(keys) == len(e["event_indices"]), "duplicate ruptures"
        es = e["event_statistics"]
        M0 = es["slip"] * es["area"] * 1e6 * 3.0e10
        assert np.allclose(M0, scaling.M0_2_Mw(mw, inverse=True))


def test_local_follows_the_scaling_relation_on_uneven_cells():
    stats = _uneven()
    for mw in (7.5, 7.8, 8.1, 8.4):
        r = _area_ratio(events.get_all_earthquake_events_of_magnitude_Mw(mw, stats))
        lo = _area_ratio(events.get_all_earthquake_events_of_magnitude_Mw(
            mw, stats, rupture_size="local"))
        assert lo.max() / lo.min() < 2.0
        assert np.all(np.abs(np.log10(lo)) < np.log10(1.6))
        # rptha's single block is far off somewhere on this mesh
        assert r.max() / r.min() > 2.5


def test_local_matches_rptha_area_on_uniform_cells():
    stats = _uniform()
    for mw in (7.5, 8.0, 8.5):
        r = _area_ratio(events.get_all_earthquake_events_of_magnitude_Mw(mw, stats))
        lo = _area_ratio(events.get_all_earthquake_events_of_magnitude_Mw(
            mw, stats, rupture_size="local"))
        # equal cells: local's worst area is never worse than rptha's
        assert np.abs(np.log10(lo)).max() <= np.abs(np.log10(r)).max() + 1e-9


def test_rupture_longer_than_zone_is_the_whole_length():
    stats = _uneven()
    e = events.get_all_earthquake_events_of_magnitude_Mw(
        9.6, stats, rupture_size="local")
    nstrike = int(stats["alongstrike_number"].max())
    assert all(d["length"] == nstrike for d in e["event_dims"])
    assert all(tl[1] == 1 for tl in e["topleft_indices"])
    # rows chosen by area: the largest block, the whole zone
    assert e["event_statistics"]["area"].max() == pytest.approx(
        np.sum(stats["length"] * stats["width"]))


def test_event_table_passes_rupture_size_through():
    stats = _uneven()
    t_r = events.get_all_earthquake_events(stats, Mmin=7.2, Mmax=8.0, dMw=0.1)
    t_l = events.get_all_earthquake_events(stats, Mmin=7.2, Mmax=8.0, dMw=0.1,
                                           rupture_size="local")
    assert t_r["area"].size != t_l["area"].size or not np.allclose(t_r["area"], t_l["area"])
    for s, idx in zip(t_l["event_index_string"], t_l["event_indices"]):
        assert np.array_equal(events.get_unit_source_indices_in_event(s), idx)


# --- v10_q: the q weight (events.coverage_weights) ---------------------------

def test_coverage_weights_by_hand():
    # one magnitude, cells 0-3: ruptures {0,1} {1,2} {3}; counts 1, 2, 1, 1
    q = events.coverage_weights([7.2, 7.2, 7.2],
                                [np.array([0, 1]), np.array([1, 2]), np.array([3])], 4)
    assert np.allclose(q, [0.75, 0.75, 1.0])
    # magnitudes are counted separately
    q = events.coverage_weights([7.2, 7.3], [np.array([0]), np.array([0])], 1)
    assert np.allclose(q, [1.0, 1.0])


def test_coverage_weights_equal_on_rptha_interior():
    stats = _uniform()
    e = events.get_all_earthquake_events(stats, Mmin=7.6, Mmax=7.6, dMw=0.1)
    q = events.coverage_weights(e["Mw"], e["event_indices"], stats["length"].size)
    asn = stats["alongstrike_number"]
    # not touching the first or last column (fewer ruptures cover those)
    interior = np.array([asn[i].min() > 1 and asn[i].max() < asn.max()
                         for i in e["event_indices"]])
    assert any(interior)
    assert np.ptp(q[interior]) < 1e-12


def test_coverage_weights_balance_dense_and_sparse_parts():
    # with q, a cell's summed weight no longer grows with how many ruptures
    # cover it: compare the spread of per-cell coverage with and without q
    stats = _uneven()
    n = stats["length"].size
    for mw in (7.2, 7.5):
        e = events.get_all_earthquake_events(stats, Mmin=mw, Mmax=mw, dMw=0.1,
                                             rupture_size="local")
        q = events.coverage_weights(e["Mw"], e["event_indices"], n)
        raw, cor = np.zeros(n), np.zeros(n)
        for qi, idx in zip(q, e["event_indices"]):
            raw[idx] += 1.0
            cor[idx] += qi
        assert cor.max() / cor.min() < raw.max() / raw.min()
