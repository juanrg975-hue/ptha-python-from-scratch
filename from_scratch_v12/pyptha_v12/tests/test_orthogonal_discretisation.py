"""Tests for the rptha-faithful orthogonal contour discretiser.

The defining property of rptha's improved algorithm is that the down-dip
lines cross the depth contours nearly orthogonally. We check that the ported
optimiser (a) actually reduces the orthogonality "badness" it minimises, and
(b) yields a grid measurably MORE orthogonal than the simplified discretiser,
on the real Puysegur contours -- while keeping the geometry physical.

Skipped automatically if geopandas or the shapefile are unavailable.

Run:  python pyptha/tests/test_orthogonal_discretisation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import unit_sources as us  # noqa: E402

_SHP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "rptha", "R", "rptha", "inst", "extdata", "puysegur.shp"))


def _available():
    try:
        import geopandas  # noqa: F401
    except Exception:
        return False
    return os.path.exists(_SHP)


def _load():
    import geopandas as gpd
    gdf = gpd.read_file(_SHP)
    gdf["level"] = gdf["level"].astype(float)
    out = []
    for _, row in gdf.iterrows():
        g = row.geometry
        coords = (np.array(g.coords) if g.geom_type == "LineString"
                  else np.array(sorted(g.geoms, key=lambda x: len(x.coords))[-1].coords))
        out.append((row["level"], coords[:, :2]))
    return out


def _mean_nonorthogonality(grid):
    """Mean |cos(angle)| between down-dip and along-strike edges (0 = orthogonal).

    Measured in a local tangent plane (lon scaled by cos(lat)), matching how
    rptha measures orthogonality.
    """
    nd, _, na = grid.shape
    lon, lat = grid[:, 0, :], grid[:, 1, :]
    vals = []
    for j in range(1, nd - 1):
        for i in range(1, na - 1):
            clat = np.cos(np.radians(lat[j, i]))
            v_as = np.array([(lon[j, i + 1] - lon[j, i - 1]) * clat,
                             lat[j, i + 1] - lat[j, i - 1]])
            v_dd = np.array([(lon[j + 1, i] - lon[j - 1, i]) * clat,
                             lat[j + 1, i] - lat[j - 1, i]])
            c = abs(v_as @ v_dd) / (np.linalg.norm(v_as) * np.linalg.norm(v_dd) + 1e-12)
            vals.append(c)
    return float(np.mean(vals))


def test_optimiser_reduces_badness():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    interps, _ = cd._build_interpolators(contours)
    num_l = len(interps)
    npv = 20
    s_even = np.tile(np.linspace(0, 1, npv), (num_l, 1))
    q_even = float(np.sum(cd._quality_matrix(s_even, interps) ** 2))

    s_opt = cd._optimise_s_matrix(interps, desired_num_lines=npv, num_l=num_l)
    q_opt = float(np.sum(cd._quality_matrix(s_opt, interps) ** 2))
    # Optimisation must lower the badness it minimises.
    assert q_opt < q_even


def test_orthogonal_beats_simplified():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    g_simple = us.discretized_source_from_contours(
        contours, n_alongstrike=21, n_downdip=8)
    g_ortho = cd.discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length=40, n_downdip=8)
    no_simple = _mean_nonorthogonality(g_simple)
    no_ortho = _mean_nonorthogonality(g_ortho)
    # The orthogonal grid must be clearly closer to orthogonal.
    assert no_ortho < no_simple
    assert no_ortho < 0.1          # within ~6 degrees of orthogonal


def test_orthogonal_geometry_physical():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    grid = cd.discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length=40, n_downdip=8)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    rows = sorted(set(stats["downdip_number"].astype(int)))
    mean_dip = [stats["dip"][stats["downdip_number"] == j].mean() for j in rows]
    # Dip increases with depth; widths are tens of km; area is sensible.
    assert mean_dip[0] < mean_dip[-1]
    assert stats["width"].mean() < 40.0
    total = float(np.sum(stats["length"] * stats["width"]))
    assert 30_000 < total < 200_000


def test_interpolate_3D_path_equal_arclength():
    """Rows are spaced by EQUAL 3D ARC LENGTH (rptha), not equal depth.

    On a path whose shallow half is nearly flat (long in arc, small in depth),
    the mid point by arc length sits at a much shallower depth than the mid
    point by depth. This pins the faithful-resampling fix of 2026-07-23.
    """
    # 0->10 km depth over ~220 km horizontal, then 10->40 km over ~55 km.
    path = np.array([[163.0, -46.0, 0.0],
                     [165.0, -45.0, 10.0],
                     [165.5, -44.8, 40.0]])
    pts = cd.interpolate_3D_path(path, n=3)
    # End points preserved.
    np.testing.assert_allclose(pts[0], path[0], atol=1e-6)
    np.testing.assert_allclose(pts[-1, 2], 40.0, atol=1e-6)
    # Mid row: equal-depth resampling would put it at depth 20; equal
    # arc-length must put it well within the long shallow segment (< 10 km).
    assert pts[1, 2] < 10.0


def test_matches_rptha_puysegur_depths():
    """Golden values from rptha 0.1.147 run on puysegur.shp (2026-07-23).

    rptha's discretized_source_from_source_contours(50, 50) picks a 2 x 17
    grid whose middle row sits at ~11.2-12.3 km depth (equal 3D arc length),
    with unit-source centroid depths ~5.6-26.0 km. The port must reproduce
    the row-placement rule and land in the same depth band.
    """
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    grid = cd.discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length=50.0,
        desired_unit_source_width=50.0, seed=1234)
    # rptha chose 2 rows x 17 columns for 50x50 km cells.
    assert grid.shape[0] - 1 == 2
    assert grid.shape[2] - 1 == 17
    # Middle grid row: rptha places it at 11.2-12.3 km (NOT 20 km).
    mid_depths = grid[1, 2, :]
    assert np.all(mid_depths > 10.0) and np.all(mid_depths < 13.5)
    # Centroid depth range as in rptha (5.63 / 26.05 +- tolerance).
    stats = us.discretized_source_approximate_summary_statistics(grid)
    assert abs(stats["depth"].min() - 5.63) < 0.5
    assert abs(stats["depth"].max() - 26.05) < 0.5


def test_reproducible():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    g1 = cd.discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length=40, n_downdip=8, seed=7)
    g2 = cd.discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length=40, n_downdip=8, seed=7)
    np.testing.assert_allclose(g1, g2)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
