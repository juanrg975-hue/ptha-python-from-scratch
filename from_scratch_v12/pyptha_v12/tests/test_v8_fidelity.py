"""v8 regression tests: the guarantees v8's changes rest on.

Golden data in tests/data/ (all small):
  ptha18_contours_puysegur2.*          PTHA18's own contour shapefile
                                       (NCI DATA/SOURCEZONE_CONTOURS.zip)
  rptha_downdip_lines_puysegur2.csv    rptha's R output for those contours:
                                       create_downdip_lines_on_source_contours_
                                       improved(contours, 50), R 4.6.1,
                                       rptha 0.1.147, minpack.lm 1.2-4
  ptha18_unit_source_grid_puysegur2.csv  PTHA18's own unit-source grid
                                       (NCI SOURCE_ZONES/puysegur2/EQ_SOURCE/
                                       all_discretized_sources.RDS)
  ptha18_events_puysegur2.csv          PTHA18's own event table (NCI
                                       SOURCE_ZONES/puysegur2/TSUNAMI_EVENTS/
                                       all_uniform_slip_earthquake_events_
                                       puysegur2.nc): Mw, event_index_string,
                                       area, rate_annual
and the official statistics tables and inputs in <package>/inputs/.

Run:  .venv/Scripts/python.exe -m pytest from_scratch_v12/pyptha_v12/tests/test_v8_fidelity.py -q
"""

import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))       # from_scratch_v12/
ROOT = os.path.abspath(os.path.join(PKG, ".."))              # ptha18_logic_tree_test/
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))

from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import unit_sources as us  # noqa: E402

DATA = os.path.join(HERE, "data")


def _ptha18_contours():
    gpd = pytest.importorskip("geopandas")
    g = gpd.read_file(os.path.join(DATA, "ptha18_contours_puysegur2.shp"))
    return [(float(r["level"]), np.asarray(r.geometry.coords, float)[:, :2])
            for _, r in g.iterrows()]


def _hav_m(a, b):
    la1, la2 = np.radians(a[:, 1]), np.radians(b[:, 1])
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin(np.radians(b[:, 0] - a[:, 0]) / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(h))


def test_coloured_jacobian_is_minpack_fdjac2_bit_for_bit():
    contours = _ptha18_contours()
    interps, _ = cd._build_interpolators(contours)
    num_l, npv = len(interps), 15
    s0 = np.tile(np.linspace(0, 1, npv), (num_l, 1))

    def residual(free):
        sm = s0.copy()
        sm[:, 1:npv - 1] = free.reshape(num_l, npv - 2, order="F")
        return cd._quality_matrix(sm, interps)

    rng = np.random.RandomState(0)
    x = s0[:, 1:npv - 1].ravel(order="F") + 0.1 / (npv - 1) * rng.uniform(-1, 1, num_l * (npv - 2))
    f0 = residual(x)
    jc = cd._fd_jacobian(residual, x, f0, *cd._stencil_colouring(num_l, npv))
    jn = np.empty_like(jc)
    for j in range(x.size):                      # MINPACK fdjac2, one column at a time
        h = cd._SQRT_EPS * abs(x[j]) or cd._SQRT_EPS
        xx = x.copy()
        xx[j] = x[j] + h
        jn[:, j] = (residual(xx) - f0) / h
    assert np.array_equal(jc, jn)


def test_nls_lm_equals_true_lmdif():
    from scipy.optimize import leastsq
    contours = _ptha18_contours()
    interps, _ = cd._build_interpolators(contours)
    num_l, npv = len(interps), 15
    s0 = np.tile(np.linspace(0, 1, npv), (num_l, 1))

    def residual(free):
        sm = s0.copy()
        sm[:, 1:npv - 1] = free.reshape(num_l, npv - 2, order="F")
        return cd._quality_matrix(sm, interps)

    x0 = s0[:, 1:npv - 1].ravel(order="F")
    xa = leastsq(residual, x0, ftol=1e-3, xtol=cd._SQRT_EPS, gtol=0.0,
                 maxfev=100 * (x0.size + 1), factor=100)[0]
    xb, _ = cd._nls_lm(residual, x0, cd._stencil_colouring(num_l, npv), ftol=1e-3,
                       maxiter=10 ** 6)
    assert np.array_equal(xa, xb)


def test_downdip_lines_match_rptha_r_output():
    """Same contours, same algorithm: v8's lines sit on rptha's R lines."""
    contours = _ptha18_contours()
    interps, _ = cd._build_interpolators(contours)
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    top = np.asarray(min(contours, key=lambda c: c[0])[1])
    top_km = sum(g.inv(top[k, 0], top[k, 1], top[k + 1, 0], top[k + 1, 1])[2]
                 for k in range(len(top) - 1)) / 1000.0
    s = cd._optimise_s_matrix(interps, int(np.ceil(top_km / 50.0 + 1)), len(interps))
    x, y = cd._get_xy(s, interps)
    rows = list(csv.DictReader(open(os.path.join(DATA, "rptha_downdip_lines_puysegur2.csv"))))
    nc = max(int(r["col"]) for r in rows)
    nk = max(int(r["k"]) for r in rows)
    r_xy = np.empty((nk, 2, nc))
    for r in rows:
        r_xy[int(r["k"]) - 1, 0, int(r["col"]) - 1] = float(r["lon"])
        r_xy[int(r["k"]) - 1, 1, int(r["col"]) - 1] = float(r["lat"])
    assert x.shape == (nk, nc)
    d = _hav_m(np.column_stack([x.ravel(), y.ravel()]),
               np.column_stack([r_xy[:, 0, :].ravel(), r_xy[:, 1, :].ravel()]))
    assert d.max() < 1.0, f"max offset from rptha's R lines {d.max():.3f} m"


def test_mesh_reproduces_ptha18_grid():
    contours = _ptha18_contours()
    grid = cd.discretized_source_from_contours_optimal(
        contours, 50.0, desired_unit_source_width=35.0, verbose=False)
    rows = list(csv.DictReader(open(os.path.join(DATA, "ptha18_unit_source_grid_puysegur2.csv"))))
    nr = max(int(r["row"]) for r in rows)
    nc = max(int(r["col"]) for r in rows)
    off = np.empty((nr, 3, nc))
    for r in rows:
        i, j = int(r["row"]) - 1, int(r["col"]) - 1
        off[i] = off[i]
        off[i, 0, j], off[i, 1, j], off[i, 2, j] = float(r["lon"]), float(r["lat"]), float(r["depth"])
    assert grid.shape == off.shape
    d = _hav_m(np.column_stack([grid[:, 0, :].ravel(), grid[:, 1, :].ravel()]),
               np.column_stack([off[:, 0, :].ravel(), off[:, 1, :].ravel()]))
    assert d.max() < 5.0, f"max node offset from PTHA18's grid {d.max():.2f} m"
    assert cd.mesh_defects(grid)["total"] == 0


def test_unit_source_statistics_match_ptha18_table():
    netCDF4 = pytest.importorskip("netCDF4")
    nc_path = os.path.join(ROOT, "inputs", "geometry", "unit_source_statistics_puysegur2.nc")
    if not os.path.exists(nc_path):
        pytest.skip("official puysegur2 statistics table not present")
    rows = list(csv.DictReader(open(os.path.join(DATA, "ptha18_unit_source_grid_puysegur2.csv"))))
    nr = max(int(r["row"]) for r in rows)
    nc = max(int(r["col"]) for r in rows)
    grid = np.empty((nr, 3, nc))
    for r in rows:
        i, j = int(r["row"]) - 1, int(r["col"]) - 1
        grid[i, 0, j], grid[i, 1, j], grid[i, 2, j] = float(r["lon"]), float(r["lat"]), float(r["depth"])
    st = us.discretized_source_approximate_summary_statistics(grid)
    with netCDF4.Dataset(nc_path) as ds:
        off = {k: np.asarray(ds[k][:], float) for k in
               ("lon_c", "lat_c", "depth", "strike", "dip", "length", "width",
                "alongstrike_number", "downdip_number")}
    ko = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(off["alongstrike_number"], off["downdip_number"]))}
    ks = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(st["alongstrike_number"], st["downdip_number"]))}
    io, is_ = np.array([(ko[k], ks[k]) for k in ko]).T
    for k, tol in (("lon_c", 1e-9), ("lat_c", 1e-9), ("depth", 1e-9), ("dip", 1e-8),
                   ("length", 1e-7), ("width", 1e-7)):
        assert np.max(np.abs(off[k][io] - np.asarray(st[k], float)[is_])) < tol, k
    dstrike = np.abs((off["strike"][io] - np.asarray(st["strike"], float)[is_] + 180) % 360 - 180)
    assert dstrike.max() < 1e-8          # needs geosphere's ELLIPSOIDAL midPoint


def test_step1_contours_on_a_synthetic_slab():
    """A planar slab dipping 20 deg under a trench whose depth varies 4-9 km
    along strike: v8 must find the trench and put each level L at SLAB depth
    L + (local trench depth), as PTHA18's 'depth below the nearby trench'."""
    import slab_contours as sc
    x = np.arange(100.0, 106.0001, 0.02)
    y = np.arange(-2.0, 6.0001, 0.02)
    X, Y = np.meshgrid(x, y)
    # midway between two grid columns, so the raster's own edge line (half
    # a cell beyond the last valid centre) falls exactly on the trench
    trench_lon = 101.01
    trench_depth = 6.5 + 2.5 * np.sin(Y / 8.0 * 2 * np.pi)            # 4-9 km
    dist_km = (X - trench_lon) * 111.195 * np.cos(np.radians(Y))
    depth = trench_depth + dist_km * np.tan(np.radians(20.0))
    depth[dist_km < 0] = np.nan                                          # nothing seaward of the trench
    depth[depth > 120] = np.nan
    contours, info = sc.below_trench_contours(x, y, depth, 40.0, verbose=False)
    assert sc.check_contours(contours, info) == []
    assert [lv for lv, _ in contours] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    tr = contours[0][1]
    assert np.max(np.abs(tr[:, 0] - trench_lon)) < 0.05                # trench on the up-dip edge
    for lv, c in contours[1:]:
        lat = c[:, 1]
        inner = (lat > 0.0) & (lat < 4.0)
        expected_lon = trench_lon + (lv / np.tan(np.radians(20.0))) / (
            111.195 * np.cos(np.radians(lat[inner])))
        err_km = np.abs(c[inner, 0] - expected_lon) * 111.195 * np.cos(np.radians(lat[inner]))
        assert np.median(err_km) < 0.3, (lv, np.median(err_km))


def _official_nc_stats(zone):
    netCDF4 = pytest.importorskip("netCDF4")
    p = os.path.join(ROOT, "inputs", "geometry", f"unit_source_statistics_{zone}.nc")
    if not os.path.exists(p):
        pytest.skip(f"official {zone} statistics table not present")
    with netCDF4.Dataset(p) as ds:
        return {k: np.asarray(ds[k][:]) for k in ds.variables
                if ds[k].ndim == 1 and ds[k].dtype.kind in "fi"}


def test_event_table_is_row_identical_to_ptha18():
    """Same events in the same row order as rptha (expand.grid varies the
    down-dip index fastest), so event numbers match PTHA18's own tables."""
    from pyptha_v12 import events
    st = _official_nc_stats("puysegur2")
    for k in ("subfault_number", "downdip_number", "alongstrike_number"):
        st[k] = st[k].astype(int)
    rows = list(csv.DictReader(open(os.path.join(DATA, "ptha18_events_puysegur2.csv"))))
    ev = events.get_all_earthquake_events(
        st, Mmin=7.2, Mmax=9.8, dMw=0.1, mu=3e10, relation="Strasser",
        source_zone_name="puysegur2")
    assert list(ev["event_index_string"]) == [r["event_index_string"] for r in rows]
    area = np.array([float(r["area"]) for r in rows])
    assert np.max(np.abs(ev["area"] - area) / area) < 1e-12


def test_convergence_components_follow_compute_rates_all_sources():
    import bird_convergence as bc
    div, rl = bc.convergence_components([-30.0, 10.0, -20.0], [50.0, 5.0, -3.0], 90.0)
    assert div.tolist() == [30.0, 0.0, 20.0]        # a thrust counts convergence only
    cap = 30.0 * np.tan(np.radians(50.0))
    assert rl.tolist() == [cap, 0.0, -3.0]          # lateral capped at div*tan(50), sign kept


@pytest.mark.parametrize("zone", ["cascadia", "makran2", "philippine", "kermadectonga2",
                                  "kurilsjapan", "southamerica", "izumariana"])
def test_bird_convergence_reproduces_ptha18_convergence(zone):
    """On PTHA18's own mesh, --convergence bird-griffin (PTHA18's table, read
    from the package's own data/bird/, not from rptha) gives PTHA18's
    convergence (recovered from its saved session) to machine precision."""
    import bird_convergence as bc
    assert os.path.commonpath([bc.BIRD_ZIP, PKG]) == PKG
    inp = os.path.join(ROOT, "inputs", f"input_{zone}.json")
    if not os.path.exists(inp):
        pytest.skip(f"official {zone} input not present")
    col = bc.column_convergence(_official_nc_stats(zone), log=lambda *a: None,
                                table="bird-griffin")
    official = json.load(open(inp))["rates"]["tectonic_convergence_mm_per_yr"]
    assert abs(col["area_weighted_mean_mm_per_yr"] - official) / official < 1e-12


@pytest.mark.parametrize("zone", ["cascadia", "philippine", "kurilsjapan",
                                  "southamerica", "izumariana"])
def test_bird_public_table_matches_ptha18_where_ptha18_used_bird(zone):
    """--convergence bird reads Bird (2003)'s public catalogue from the
    package's own data/bird/; where PTHA18 kept Bird's own steps (these
    zones) it gives PTHA18's convergence exactly."""
    import bird_convergence as bc
    assert os.path.commonpath([bc.BIRD_RAW_ZIP, PKG]) == PKG
    inp = os.path.join(ROOT, "inputs", f"input_{zone}.json")
    if not os.path.exists(inp):
        pytest.skip(f"official {zone} input not present")
    col = bc.column_convergence(_official_nc_stats(zone), log=lambda *a: None,
                                table="bird")
    official = json.load(open(inp))["rates"]["tectonic_convergence_mm_per_yr"]
    assert abs(col["area_weighted_mean_mm_per_yr"] - official) / official < 1e-12


def test_saved_hs_features_give_the_same_comparison():
    """Step 7b saves per-field features; step 8 compares them with PTHA18's
    catalogue. The result equals comparing the full fields directly."""
    import hs_official_compare as hoc
    rng = np.random.default_rng(3)
    syn = [rng.random((rng.integers(1, 7), rng.integers(1, 12))) for _ in range(25)]
    off = [np.where(rng.random((4, 9)) > 0.4, rng.random((4, 9)), 0.0) for _ in range(30)]
    a = hoc.compare_to_official(syn, off)
    b = hoc.compare_features(hoc.field_features(syn), hoc.field_features(off))
    for side in ("synthetic", "official"):
        for k in ("n", "mean_peak_slip_m", "mean_active_cells", "mean_concentration"):
            assert a[side][k] == b[side][k]
    assert a["spectral_distance"] == b["spectral_distance"]


def _planar_slab(dip_deg_of_lat=lambda lat: 20.0 + 0 * lat, trench_lon=101.01):
    x = np.arange(100.0, 106.0001, 0.02)
    y = np.arange(-2.0, 6.0001, 0.02)
    X, Y = np.meshgrid(x, y)
    dist_km = (X - trench_lon) * 111.195 * np.cos(np.radians(Y))
    depth = 6.0 + dist_km * np.tan(np.radians(dip_deg_of_lat(Y)))
    depth[dist_km < 0] = np.nan
    depth[depth > 120] = np.nan
    return x, y, X, Y, depth


def test_step1_clip_line_through_the_slab_is_never_trench():
    """v9 (hellenic_east2): a clip window whose straight edge cuts THROUGH
    the slab, obliquely to the dip, leaves a shallow, up-dip facing edge that
    passes both trench tests. It must not become trench: it is the window,
    not the slab's own edge. Planar slab dipping 20 deg towards azimuth 300,
    trench along the line d = 0, clipped at 104.5E."""
    import slab_contours as sc
    x = np.arange(100.0, 106.0001, 0.02)
    y = np.arange(-2.0, 6.0001, 0.02)
    X, Y = np.meshgrid(x, y)
    ux, uy = -np.sin(np.radians(60.0)), np.cos(np.radians(60.0))   # down-dip direction
    d_km = ((X - 103.0) * ux + (Y - 0.0) * uy) * 111.195
    depth = 6.0 + d_km * np.tan(np.radians(20.0))
    depth[(d_km < 0) | (depth > 120)] = np.nan
    bbox = (100.0, 104.5, -2.0, 6.0)
    # without the rule the clip edge would be taken as trench ...
    masked = np.where(X <= bbox[1], depth, np.nan)
    old = sc.find_trench(x, y, masked)["trench_raw"]
    corner_lat = (bbox[1] - 103.0) * -ux / uy          # where the real trench meets the clip line
    on_clip = lambda t: (np.abs(t[:, 0] - bbox[1]) < 0.03) & (t[:, 1] > corner_lat + 0.1)  # noqa: E731
    assert on_clip(old).sum() > 5
    # ... with it, no trench point lies on the clip line
    contours, info = sc.below_trench_contours(x, y, depth, 40.0, bbox=bbox, verbose=False)
    assert sc.check_contours(contours, info) == []
    raw = info["diag"]["trench_raw"]
    assert on_clip(raw).sum() == 0
    assert info["clip_edge_shallow_km"] > 20.0
    # every level is then measured from the real trench (6 km deep): the
    # 20 km level sits 20 / tan(20 deg) = 55 km down dip of it everywhere
    lv20 = [c for lv, c in contours if lv == 20.0][0]
    d20 = ((lv20[:, 0] - 103.0) * ux + lv20[:, 1] * uy) * 111.195
    assert np.median(np.abs(d20 - 20.0 / np.tan(np.radians(20.0)))) < 1.0


def test_step1_trench_ends_cut_the_zone_at_the_given_points():
    """v9 (Berryman segment ends, hellenic): with trench_ends, every contour
    is cut along the down-dip line through each point, even when the point
    lies down dip of the raster's trench, and the cut is not moved."""
    import slab_contours as sc
    x, y, X, Y, depth = _planar_slab()          # trench at 101.01E, dips east
    ends = ((101.6, 0.0), (101.3, 4.0))          # 60 and 30 km down dip of it
    contours, info = sc.below_trench_contours(x, y, depth, 40.0, verbose=False,
                                              trench_ends=ends)
    assert sc.check_contours(contours, info) == []
    for e in ("start", "end"):
        assert info["ends"][e]["cut"] and "given end point" in info["ends"][e]["reason"]
        assert info["ends"][e]["point_distance_km"] < 3.0
    for lv, c in contours:                       # steepest descent runs due east
        assert abs(c[:, 1].min() - 0.0) < 0.06 and abs(c[:, 1].max() - 4.0) < 0.06, (lv, c[:, 1].min(), c[:, 1].max())


def test_step1_cuts_a_slanted_raster_end_square_and_keeps_a_clean_one():
    """The SLAB data stop along a slanted line at the north end: the mesh's
    end cells would be ~58 deg off square, so every contour is cut along
    the down-dip line from the trench's last point. The south end, where
    the data stop square to the contours, is kept."""
    import slab_contours as sc
    x, y, X, Y, depth = _planar_slab()
    depth[Y > 4.0 + 1.6 * (X - 101.0)] = np.nan
    contours, info = sc.below_trench_contours(x, y, depth, 40.0, verbose=False)
    assert sc.check_contours(contours, info) == []
    e = info["ends"]
    cut = [k for k in ("start", "end") if e[k]["cut"]]
    assert len(cut) == 1
    assert e[cut[0]]["deviation_deg"] > 45.0 >= e[cut[0]]["deviation_after_deg"]
    top = contours[0][1][:, 1].max()
    for lv, c in contours:
        assert c[:, 1].max() < top + 0.1, (lv, c[:, 1].max(), top)


def test_step1_cuts_a_zone_that_narrows_to_a_point():
    """North of 4N the plate steepens from 20 to 70 deg, so the zone narrows
    from ~110 km to ~15 km: the tip is cut back to where the width is 70%
    of the median."""
    import slab_contours as sc
    dip = lambda lat: np.where(lat < 4.0, 20.0, 20.0 + (lat - 4.0) / 2.0 * 50.0)  # noqa: E731
    x, y, X, Y, depth = _planar_slab(dip)
    contours, info = sc.below_trench_contours(x, y, depth, 40.0, verbose=False)
    assert sc.check_contours(contours, info) == []
    e = info["ends"]
    cut = [k for k in ("start", "end") if e[k]["cut"]]
    assert len(cut) == 1 and "narrows to a point" in e[cut[0]]["reason"]
    assert e[cut[0]]["tip_width_ratio"] < 0.5
    # The narrow tip (north of ~4.6N, under 60% of the width) is gone and the
    # full-width part (south of 4N) is kept. The cut lands at ~4.1N rather
    # than the 4.3N a straight east-west width would give, because the
    # steepest-descent lines bend north, into the steeper part, and so reach
    # the cutoff sooner.
    top = contours[0][1][:, 1].max()
    assert 3.95 < top < 4.6, top
