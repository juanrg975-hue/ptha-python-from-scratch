"""v12 --cell-size strasser: no unit source larger than the Mmin rupture / k."""

import numpy as np
import pytest

from pyptha_v12 import contour_discretisation as cd
from pyptha_v12 import unit_sources as us


def fan_contours(r_trench=330.0, r_deep=110.0, levels=np.arange(0, 61, 5), npts=80):
    """A fan-shaped zone: concentric arcs, the trench outermost (like calabria2,
    whose trench is ~4 times longer than its 60 km contour)."""
    lon0, lat0 = 20.0, 36.0
    theta = np.radians(np.linspace(200, 340, npts))
    out = []
    for lev in levels:
        r = r_trench - (r_trench - r_deep) * lev / levels[-1]
        lon = lon0 + r / (111.32 * np.cos(np.radians(lat0))) * np.cos(theta)
        lat = lat0 + r / 111.32 * np.sin(theta)
        out.append((float(lev), np.column_stack([lon, lat])))
    return out


def max_dims(grid):
    s = us.discretized_source_approximate_summary_statistics(grid)
    return float(np.max(s["length"])), float(np.max(s["width"]))


def test_cap_values():
    assert cd.strasser_cell_cap(7.2, 1.5) == pytest.approx((54.32 / 1.5, 44.15 / 1.5), rel=2e-3)
    assert cd.strasser_cell_cap(7.2, 2.0) == pytest.approx((27.16, 22.08), rel=2e-3)
    with pytest.raises(ValueError):
        cd.strasser_cell_cap(7.2, 0)


def test_bounded_mesh_meets_the_cap_where_the_mean_rule_does_not():
    contours = fan_contours()
    cap_l, cap_w = cd.strasser_cell_cap(7.2, 1.5)
    mean = cd.discretized_source_from_contours_optimal(
        contours, 50.0, 50.0, seed=1234, verbose=False, min_downdip=2)
    lm, wm = max_dims(mean)
    assert lm > cap_l * 1.02 or wm > cap_w * 1.02   # the classic mesh breaks the cap

    grid = cd.discretized_source_from_contours_bounded(
        contours, cap_l, cap_w, seed=1234, verbose=False, min_downdip=2)
    lb, wb = max_dims(grid)
    assert lb <= cap_l * 1.02 and wb <= cap_w * 1.02
    assert grid.shape[0] > mean.shape[0] and grid.shape[2] > mean.shape[2]
    assert cd.mesh_defects(grid)["total"] == 0
    # same zone: the total area hardly changes
    sm = us.discretized_source_approximate_summary_statistics(mean)
    sb = us.discretized_source_approximate_summary_statistics(grid)
    a_mean = float(np.sum(np.asarray(sm["length"]) * np.asarray(sm["width"])))
    a_bound = float(np.sum(np.asarray(sb["length"]) * np.asarray(sb["width"])))
    assert a_bound == pytest.approx(a_mean, rel=0.03)
