"""Tests for lib/segmentation.py (v9, LEVEL 0).

These check the properties the rest of the pipeline depends on, not agreement
with PTHA18. Bird (2003)'s plate pairs and PTHA18's segment list are different
source models and are allowed to differ; validate_v9.py's `segmentation` check
prints that comparison, and lib/segmentation.py's docstring explains it.

What must hold for a segmentation to be usable at all:
  * the segments TILE the mesh -- every along-strike column belongs to exactly
    one segment. If they did not, the union-of-segments branch would either
    lose moment (a gap) or double-count it (an overlap), and LEVEL 0's "the
    union is a SUM over segments" would be wrong.
  * slices are 1-based, inclusive and inside the mesh, because that is what
    run_logic_tree.py's alongstrike_slice expects.
  * names are usable as JSON keys and file names.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_PKG, "lib"))

import segmentation as sg  # noqa: E402


def _straight_trench_grid(n_cols, lat0=-40.0, lat1=-15.0, lon=185.0):
    """A minimal (ndip+1, 3, nstrike+1) lattice running north along a meridian.

    Only the top row's lon/lat are read by the segmentation code, so the
    down-dip rows just repeat with a small offset.
    """
    lats = np.linspace(lat0, lat1, n_cols + 1)
    grid = np.zeros((3, 3, n_cols + 1))
    for r in range(3):
        grid[r, 0, :] = lon + 0.25 * r
        grid[r, 1, :] = lats
        grid[r, 2, :] = 10.0 * r
    return grid


def _assert_tiles(segs, n_cols):
    covered = []
    for s in segs:
        a, b = s["alongstrike_slice"]
        assert 1 <= a <= b <= n_cols, f"slice {a}-{b} outside 1..{n_cols}"
        covered += list(range(a, b + 1))
    assert covered == list(range(1, n_cols + 1)), (
        "segments must tile the mesh exactly, with no gap and no overlap")


def test_safe_name_is_usable_as_key_and_filename():
    # Bird writes the pair with a slash whose direction says which plate goes
    # under; neither form can appear in a filename.
    assert sg._safe_name("TO/PA") == "to_pa"
    assert sg._safe_name("NZ\\SA") == "nz_sa"
    for raw in ("TO/PA", "NZ\\SA", "OK/PA"):
        name = sg._safe_name(raw)
        assert not (set(name) & set('/\\:*?"<>| ')), name


def test_runs_groups_consecutive_labels():
    assert sg._runs(["a", "a", "b", "b", "b", "a"]) == [
        ("a", 1, 2), ("b", 3, 5), ("a", 6, 6)]


def test_column_centres_are_between_the_nodes():
    grid = _straight_trench_grid(4)
    lon, lat = sg.column_centres(grid)
    assert lon.size == 4 and lat.size == 4
    # each centre lies strictly between its two bounding nodes
    nodes = grid[0, 1, :]
    for i in range(4):
        assert min(nodes[i], nodes[i + 1]) < lat[i] < max(nodes[i], nodes[i + 1])


def test_column_centres_longitudes_are_in_0_360():
    grid = _straight_trench_grid(6, lon=-176.0)
    lon, _ = sg.column_centres(grid)
    assert np.all((lon >= 0.0) & (lon < 360.0))


def test_kermadectonga_splits_into_tonga_and_kermadec():
    """The worked example of the module docstring, on a synthetic trench.

    Bird has two subducting plate pairs along this trench, TO/PA in the north
    and KE/PA in the south, so a mesh spanning both must come back as two
    segments in north-to-south order.
    """
    grid = _straight_trench_grid(40, lat0=-40.0, lat1=-15.0, lon=185.0)
    segs = sg.bird_segments(grid, log=lambda *a: None)
    assert len(segs) == 2
    _assert_tiles(segs, 40)
    pairs = [s["plate_pair"] for s in segs]
    assert set(pairs) == {"KE/PA", "TO/PA"}
    # the grid runs south -> north, so Kermadec (south) comes first
    assert pairs[0] == "KE/PA"


def test_segments_always_tile_whatever_the_mesh_length():
    for n in (12, 25, 61, 100):
        grid = _straight_trench_grid(n, lat0=-40.0, lat1=-15.0, lon=185.0)
        segs = sg.bird_segments(grid, log=lambda *a: None)
        _assert_tiles(segs, n)


def test_min_columns_suppresses_short_segments():
    """A short run is absorbed, and the result still tiles.

    A segment of one or two columns cannot be constrained by its own
    seismicity, so bird_segments merges it into its longer neighbour rather
    than emitting it.
    """
    grid = _straight_trench_grid(60, lat0=-40.0, lat1=-15.0, lon=185.0)
    for min_cols in (2, 5, 10, 20):
        segs = sg.bird_segments(grid, min_columns=min_cols, log=lambda *a: None)
        _assert_tiles(segs, 60)
        if len(segs) > 1:
            shortest = min(b - a + 1 for a, b in
                           (s["alongstrike_slice"] for s in segs))
            assert shortest >= min_cols


def test_far_from_any_subduction_zone_gives_no_segmentation():
    """Mid-Atlantic: no subduction step within range, so nothing to segment."""
    grid = _straight_trench_grid(20, lat0=10.0, lat1=30.0, lon=330.0)
    segs = sg.bird_segments(grid, max_distance_deg=2.0, log=lambda *a: None)
    assert segs == []


def test_repeated_plate_pair_gets_distinct_keys():
    """Two disjoint stretches of the same pair must not collide on one key.

    southamerica really does this: Bird's NZ\\SA is interrupted by the
    Altiplano sliver NZ\\AP, so the pair appears twice along one trench. They
    are separate along-strike slices and must stay separate segments.
    """
    grid = _straight_trench_grid(90, lat0=-45.0, lat1=-2.0, lon=287.0)
    segs = sg.bird_segments(grid, log=lambda *a: None)
    keys = [s["segment_key"] for s in segs]
    assert len(keys) == len(set(keys)), f"duplicate segment keys: {keys}"
    _assert_tiles(segs, 90)


def test_ptha18_segments_rescale_onto_a_different_mesh(tmp_path):
    """PTHA18's indices are columns of PTHA18's mesh, so they must be rescaled.

    A from-scratch mesh generally has a different column count (kermadectonga2:
    73 vs 72). Reusing the raw integers would shift every boundary silently.
    """
    csv = tmp_path / "sourcezone_parameters.csv"
    csv.write_text(
        "sourcename,segment_name,segment_boundary_alongstrike_index_lower,"
        "segment_boundary_alongstrike_index_upper\n"
        "z,_a,1,25\n"
        "z,_b,26,50\n"
        "z,_c,51,100\n")

    same = sg.ptha18_segments("z", 100, str(csv), log=lambda *a: None)
    assert [s["alongstrike_slice"] for s in same] == [[1, 25], [26, 50], [51, 100]]

    # halve the mesh: the boundaries move with it, and still tile
    half = sg.ptha18_segments("z", 50, str(csv), log=lambda *a: None)
    _assert_tiles(half, 50)
    assert half[0]["alongstrike_slice"] == [1, 13] or half[0]["alongstrike_slice"] == [1, 12]
    assert half[-1]["alongstrike_slice"][1] == 50
    # the original indices are kept for the comparison
    assert half[0]["ptha18_slice"] == [1, 25]


def test_ptha18_segments_returns_empty_for_an_unsegmented_zone(tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text(
        "sourcename,segment_name,segment_boundary_alongstrike_index_lower,"
        "segment_boundary_alongstrike_index_upper\n"
        "z,,,\n")
    assert sg.ptha18_segments("z", 40, str(csv), log=lambda *a: None) == []


_SZP_HEADER = ("sourcename,segment_name,segment_boundary_alongstrike_index_lower,"
               "segment_boundary_alongstrike_index_upper,cmin,cpref,cmax,bmin,bmax,"
               "mw_max_observed,prob_Mmax_below_Mmin,scaling_relation\n")


def test_official_segments_block_keeps_ptha18_rows(tmp_path):
    """Step 8's official segmented input: PTHA18's own columns (not
    rescaled, it runs on PTHA18's mesh), coupling and mw_max_observed per
    segment, the zone's coupling prior settings, the session's GCMT events."""
    csv = tmp_path / "p.csv"
    csv.write_text(_SZP_HEADER
                   + "z,,,,0.2,0.3,0.7,0.7,1.2,8.1,0,Strasser\n"
                   + "z,_a,1,24,0.1,0.2,0.7,0.7,1.2,8,0,Strasser\n"
                   + "z,_b,25,72,0.4,0.54,0.7,0.7,1.2,8.2,0.1,Strasser\n")
    zone_rates = {"coupling": {"prior_type": "spreadsheet_and_uniform_50_50",
                               "uniform_range": [0.1, 1.3],
                               "spreadsheet_values": [0.2, 0.3, 0.7]},
                  "update_logic_tree_weights_with_data": True}
    gcmt = {"z_a": {"count": 4}, "z_b": {"count": 1}}
    block = sg.ptha18_official_segments_block("z", zone_rates, str(csv), gcmt.get,
                                              log=lambda *a: None)
    assert list(block) == ["a", "b"]
    assert block["a"]["alongstrike_slice"] == [1, 24]
    assert block["b"]["alongstrike_slice"] == [25, 72]
    b = block["b"]["rates"]
    assert b["coupling"]["spreadsheet_values"] == [0.4, 0.54, 0.7]
    assert b["coupling"]["prob_zero_coupling"] == 0.1
    assert b["coupling"]["uniform_range"] == [0.1, 1.3]
    assert b["mw_max_observed"] == 8.2 and b["observed_seismicity"] == {"count": 1}
    # the zone's own block is not modified by the per-segment copies
    assert zone_rates["coupling"]["spreadsheet_values"] == [0.2, 0.3, 0.7]
    # no convergence here: the engine averages the Bird profile per segment
    assert "tectonic_convergence_mm_per_yr" not in b


def test_official_segments_block_needs_gcmt_when_level3_is_on(tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text(_SZP_HEADER + "z,_a,1,10,0.1,0.2,0.7,0.7,1.2,8,0,Strasser\n")
    with pytest.raises(ValueError):
        sg.ptha18_official_segments_block(
            "z", {"coupling": {}, "update_logic_tree_weights_with_data": True},
            str(csv), lambda name: None, log=lambda *a: None)
    assert sg.ptha18_official_segments_block(
        "y", {"coupling": {}}, str(csv), lambda name: None, log=lambda *a: None) == {}


def test_engine_reproduces_ptha18_official_segment_tree(tmp_path):
    """The engine's segment path against PTHA18's own tree of kermadectonga2's
    _hikurangi segment (extracted from the saved session): every branch's 'a'
    and posterior weight. Catches a segment inheriting the zone's convergence
    (hikurangi's own is about 30 mm/yr against the zone's 99), its area, its
    Mw_max anchor, its coupling or its GCMT events. The edge correction is
    off: it moves how a rate is shared among scenarios, never 'a' or the
    weights."""
    root = os.path.dirname(_PKG)
    need = [os.path.join(root, "inputs", "input_kermadectonga2.json"),
            os.path.join(root, "official_ptha_data", "trees",
                         "logic_tree_branches_kermadectonga2_hikurangi_OFFICIAL.csv"),
            os.path.join(root, "official_ptha_data", "trees",
                         "official_gcmt_observations.csv")]
    if not all(os.path.exists(p) for p in need):
        pytest.skip("PTHA18's extracted kermadectonga2 inputs are not present")
    import pandas as pd
    sys.path.insert(0, os.path.join(_PKG, "validation"))
    sys.path.insert(0, os.path.join(_PKG, "python_logic_tree_v12"))
    import validate_v9
    import run_logic_tree as rlt
    cfg = validate_v9.official_segmented_input("kermadectonga2")
    seg = dict(cfg["segments"]["hikurangi"])
    seg["rates"] = dict(seg["rates"], edge_correction={"mode": "off"})
    rep = rlt.build_source_representation("kermadectonga2_hikurangi", cfg, seg,
                                          str(tmp_path))
    br = rep["branches"]
    p = pd.DataFrame([dict(par, a=br.a_parameter[i], posterior_prob=br.all_par_prob[i])
                      for i, par in enumerate(br.all_par_combo)])
    o = pd.read_csv(need[1])
    key = lambda d: list(zip(d.slip_rate.round(12), d.b.round(10), d.Mw_max.round(8),
                             d.Mw_frequency_distribution))
    o["k"], p["k"] = key(o), key(p)
    m = o.merge(p.drop_duplicates("k"), on="k", suffixes=("_o", "_p"))
    assert len(m) == len(o) == len(p)
    assert np.max(np.abs(m.a_o - m.a_p)) < 1e-9
    assert np.max(np.abs(m.posterior_prob_o - m.posterior_prob_p)) < 1e-12
    # rptha's segment is a stretch of the zone: the zone's whole scenario
    # table, and zero probability for a scenario entirely outside it
    from pyptha_v12 import events
    eq, ecp = rep["all_eq"], rep["ecp"]
    asn = rep["stats"]["alongstrike_number"].astype(int)
    a, b = seg["alongstrike_slice"]
    frac = np.array([np.mean((asn[events.get_unit_source_indices_in_event(s)] >= a)
                             & (asn[events.get_unit_source_indices_in_event(s)] <= b))
                     for s in eq["event_index_string"]])
    assert asn.max() == 72 and eq["Mw"].size > 2000
    live = rep["scenario_rates"] > 0
    assert np.all(ecp[live & (frac == 0)] == 0)
    assert np.all(ecp[live & (frac > 0)] > 0)


def test_bird_to_berryman_table_names_real_pairs_and_rows():
    """Every entry of berryman_params.BIRD_SEGMENT_TO_BERRYMAN is a plate pair
    Bird (2003) really has as a subduction step, and every row it names is a
    Berryman et al. (2015) Table 3.1 row; a mapped segment's coupling is its
    row's (Tonga: 0.1/0.2/0.7, the values PTHA18 gives its _tonga)."""
    import berryman_params as bp
    pairs = set(sg.load_bird_subduction_steps()[0])
    for zone, table in bp.BIRD_SEGMENT_TO_BERRYMAN.items():
        for pair, rows in table.items():
            assert pair in pairs, (zone, pair)
            assert rows and all(r in bp.BERRYMAN_ROWS for r in rows), (zone, rows)
    rows = bp.berryman_rows_for_bird_segment("kermadectonga2", "TO/PA")
    assert bp.berryman_segment_coupling(rows) == [0.1, 0.2, 0.7]
    assert bp.berryman_segment_mw_max_min(rows) == 8.0
    both = bp.berryman_rows_for_bird_segment("kermadectonga2", "KE/PA")
    assert bp.berryman_segment_mw_max_min(both) == 8.1
    assert bp.berryman_rows_for_bird_segment("sunda2", "SU/AU") is None


def test_compare_with_ptha18_reports_a_count_difference():
    """The kermadectonga2 case: the comparison must SAY the models differ."""
    bird = [{"alongstrike_slice": [1, 23]}, {"alongstrike_slice": [24, 73]}]
    ptha = [{"alongstrike_slice": [1, 24]}, {"alongstrike_slice": [25, 59]},
            {"alongstrike_slice": [60, 73]}]
    lines = []
    out = sg.compare_with_ptha18(bird, ptha, 73, log=lines.append)
    assert out["n_bird"] == 2 and out["n_ptha18"] == 3
    assert any("genuinely differ" in ln for ln in lines)
    # the shared boundary is found to be one column away
    assert out["matched"][0][2] == 1


# ---------------------------------------------------------------------------
# v9: segment boundaries from Berryman et al. (2015) Table 3.1 (the default)
# ---------------------------------------------------------------------------

def _grid_through(points, n_cols):
    """A 2-row mesh whose top edge runs through ``points`` (lon, lat)."""
    pts = np.asarray(points, float)
    seg = np.hypot(np.diff(pts[:, 0]) * np.cos(np.radians(pts[:-1, 1])), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n_cols + 1)
    lon, lat = np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])
    grid = np.zeros((3, 3, n_cols + 1))
    for r in range(3):
        grid[r, 0, :] = lon - 0.3 * r
        grid[r, 1, :] = lat
        grid[r, 2, :] = 10.0 * r
    return grid


def test_column_edge_position_follows_the_down_dip_line():
    """A point well down dip of the top edge goes to the column edge whose
    down-dip line passes through it, not to the nearest top-edge point
    (v9, zones cut at Berryman's points: hellenic2)."""
    n = 10
    grid = np.zeros((3, 3, n + 1))
    for r in range(3):                       # top edge 100-105E at 0N; edges run north-east
        grid[r, 0, :] = np.linspace(100.0, 105.0, n + 1) + 1.0 * r
        grid[r, 1, :] = 1.0 * r
        grid[r, 2, :] = 10.0 * r
    t, d = sg.column_edge_position(grid, 103.5, 1.5)   # on the edge that starts at 102.0E
    assert t == 4.0 and d < 1.0
    t_top, _ = sg.trench_position(grid, 103.5, 1.5)    # the top-edge rule: edge 7
    assert round(t_top) == 7


def test_berryman_segment_ends_chain_and_match_their_whole_margin():
    """Transcription check of Table 3.1: every segment of a zone starts where
    its neighbour ends, and a zone's outer ends are its Whole Margin row's."""
    import berryman_params as bp
    whole = {"kermadectonga2": ((175.503, -42.059), (-173.407, -14.584)),
             "kurilsjapan": ((141.992, 34.666), (164.066, 55.209)),
             "ryuku": ((122.501, 23.643), (138.674, 35.034)),
             "sunda2": ((92.068, 13.715), (120.886, -11.493)),
             "newhebrides2": ((164.612, -10.892), (174.277, -22.667)),
             "hellenic2": ((19.912, 37.731), (28.726, 36.579))}
    # Table 3.1 itself does not chain Hellenic: the western segment ends at
    # 25.288E and the eastern one starts at 25.228E (both 34.202N; checked on
    # the rendered page, rows 48-49). berryman_params uses their midpoint.
    gap_deg = {"hellenic2": 0.1}
    for zone, segs in bp.ZONE_BERRYMAN_SEGMENTS.items():
        ends = [bp.BERRYMAN_SEGMENT_TRENCH[r] for _, r in segs]
        tol = gap_deg.get(zone, 0.0)
        near = lambda p, q: abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol  # noqa: E731
        for a, b in zip(ends[:-1], ends[1:]):
            assert near(a[1], b[0]) or near(a[0], b[1]), (zone, a, b)
        for _, r in segs:
            assert r in bp.BERRYMAN_ROWS and r in bp.BERRYMAN_SEGMENT_PLATES, r
        if zone in whole:
            flat = {p for e in ends for p in e}
            assert set(whole[zone]) <= flat, zone


def test_berryman_splits_kermadectonga2_into_ptha18s_three_segments():
    """A mesh along Berryman's own Hikurangi-Kermadec-Tonga trench gets the
    three segments PTHA18 has, in trench order, each with its own row."""
    grid = _grid_through([(175.503, -42.059), (179.838, -37.476),
                          (185.015, -23.750), (186.593, -14.584)], 72)
    segs = sg.berryman_segments(grid, "kermadectonga2", log=lambda *a: None)
    _assert_tiles(segs, 72)
    assert [s["segment_key"] for s in segs] == ["hikurangi", "kermadec", "tonga"]
    assert [s["berryman_rows"] for s in segs] == [["H-K-T Hikurangi"], ["Kermadec"], ["Tonga"]]
    assert segs[0]["plate_pair"] == "PA\HF"


def test_berryman_drops_a_segment_beyond_the_mesh():
    """southamerica on a mesh that stops near 43 S (SLAB1.0): Berryman's two
    Patagonia segments lie beyond it and get no column."""
    grid = _grid_through([(-78.646, 7.337), (-81.599, -3.245), (-71.307, -21.965),
                          (-73.246, -34.290), (-74.5, -43.0)], 118)
    segs = sg.berryman_segments(grid, "southamerica", log=lambda *a: None)
    _assert_tiles(segs, 118)
    assert [s["segment_key"] for s in segs] == ["ecuador", "peru", "northern_chile", "central_chile"]


def test_zone_segments_dispatch_and_undivided_zone():
    grid = _grid_through([(163.2, -50.1), (168.8, -44.0)], 13)
    assert sg.zone_segments(grid, "puysegur2", "berryman", log=lambda *a: None) == []
    with pytest.raises(ValueError):
        sg.zone_segments(grid, "puysegur2", "ptha18", log=lambda *a: None)
