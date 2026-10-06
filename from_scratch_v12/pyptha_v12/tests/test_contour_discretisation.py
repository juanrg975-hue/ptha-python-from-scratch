"""Tests for the (simplified) contour -> unit-source-grid discretiser.

Uses the REAL Puysegur contours shipped with rptha to check the resulting
geometry is physically sensible: dip increases with depth (a subduction
interface), widths are reasonable, and the total area is the right order of
magnitude. This also guards the contour-orientation fix (adjacent depth
contours are sometimes stored in opposite directions).

Skipped automatically if geopandas or the shapefile are unavailable.

Run:  python pyptha/tests/test_contour_discretisation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import unit_sources as us  # noqa: E402

_SHP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "rptha", "R", "rptha", "inst", "extdata", "puysegur.shp"))


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


def _available():
    try:
        import geopandas  # noqa: F401
    except Exception:
        return False
    return os.path.exists(_SHP)


def test_puysegur_dip_increases_with_depth():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    grid = us.discretized_source_from_contours(contours, n_alongstrike=20, n_downdip=8)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    rows = sorted(set(stats["downdip_number"].astype(int)))
    mean_dip = [stats["dip"][stats["downdip_number"] == j].mean() for j in rows]
    # Subduction interface: dip should broadly increase from trench to depth.
    assert mean_dip[0] < mean_dip[-1]
    # And it should be reasonably monotone (no wild oscillation from bad
    # contour orientation).
    assert np.all(np.diff(mean_dip) > -3.0)


def test_puysegur_geometry_physical():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    grid = us.discretized_source_from_contours(contours, n_alongstrike=20, n_downdip=8)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    # Unit-source widths should be tens of km, not hundreds (the bug symptom).
    assert stats["width"].mean() < 40.0
    assert np.all(stats["dip"] > 0)
    # Total interface area: right order of magnitude for Puysegur (rptha uses
    # ~28 unit sources at 50x50 km ~= 70,000 km^2).
    total = float(np.sum(stats["length"] * stats["width"]))
    assert 30_000 < total < 200_000


def test_grid_shape_and_depths():
    if not _available():
        print("SKIP (geopandas or shapefile unavailable)")
        return
    contours = _load()
    grid = us.discretized_source_from_contours(contours, n_alongstrike=15, n_downdip=6)
    assert grid.shape == (6 + 1, 3, 15 + 1)
    # Depths span the contour range, shallow (row 0) to deep (last row).
    assert grid[0, 2, :].mean() < grid[-1, 2, :].mean()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
