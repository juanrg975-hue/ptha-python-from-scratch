"""v8.2: the column rule of discretized_source_from_contours_optimal.

Two synthetic zones, so the tests need no data file:
  * a straight zone (parallel contours);
  * a tight arc (concentric contours, trench 3 times as long as the deep
    edge), the calabria2-like case column_rule="average" is meant for.

Run:  .venv/Scripts/python.exe -m pytest from_scratch_v12/pyptha_v12/tests/test_column_rule.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import contour_discretisation as cd  # noqa: E402

KM_PER_DEG = 111.195


def _straight_zone():
    """Planar interface dipping 15 deg, trench 600 km long along lat 0."""
    lon = np.linspace(0.0, 600.0 / KM_PER_DEG, 61)
    out = []
    for depth in (0.0, 10.0, 20.0, 30.0, 40.0):
        lat = -depth / np.tan(np.radians(15.0)) / KM_PER_DEG
        out.append((depth, np.column_stack([lon, np.full(lon.size, lat)])))
    return out


def _fan_zone():
    """Concentric arcs over 140 deg: trench radius 300 km, deep edge 100 km."""
    ang = np.radians(np.linspace(200.0, 340.0, 141))
    out = []
    for depth, radius in ((0.0, 300.0), (10.0, 250.0), (20.0, 200.0),
                          (30.0, 150.0), (40.0, 100.0)):
        x, y = radius * np.cos(ang), radius * np.sin(ang)
        out.append((depth, np.column_stack([x / KM_PER_DEG, y / KM_PER_DEG])))
    return out


def _mesh(contours, **kw):
    return cd.discretized_source_from_contours_optimal(
        contours, desired_unit_source_length=50.0, desired_unit_source_width=50.0,
        seed=1234, min_downdip=2, verbose=False, **kw)


def test_trench_is_the_default_and_the_v8_mesh():
    for contours in (_straight_zone(), _fan_zone()):
        assert np.array_equal(_mesh(contours), _mesh(contours, column_rule="trench"))


def test_average_centres_the_cells_of_a_fan():
    contours = _fan_zone()
    trench, average = _mesh(contours), _mesh(contours, column_rule="average")
    s_t, s_a = cd.mesh_shape(trench), cd.mesh_shape(average)
    assert s_t["taper"] > cd.PTHA18_MAX_TAPER                # a fan, outside PTHA18's range
    assert average.shape[2] < trench.shape[2]                # fewer columns
    assert average.shape[0] == trench.shape[0]               # same rows
    assert abs(np.log(s_a["median_length_km"] / 50.0)) < abs(np.log(s_t["median_length_km"] / 50.0))
    assert s_a["mean_deviation"] < s_t["mean_deviation"]
    assert cd.mesh_defects(average)["total"] == 0


def test_average_column_count():
    contours = _fan_zone()
    n = int(np.ceil(cd._mean_row_length_km(_mesh(contours)) / 50.0))
    assert _mesh(contours, column_rule="average").shape[2] - 1 == n


def test_straight_zone_shape_is_inside_ptha18_range():
    s = cd.mesh_shape(_mesh(_straight_zone()))
    assert s["taper"] < cd.PTHA18_MAX_TAPER
    assert s["width_ratio"] < cd.PTHA18_MAX_WIDTH_RATIO


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        _mesh(_straight_zone(), column_rule="auto")
