"""v10_q: the trench ramp rule (slab_contours.find_trench(ramp_rise_km=)).

A synthetic planar slab whose trench is 8 km deep along strike except at one
end, where the raster's edge climbs to 20 km over 80 km (the edge of the
SLAB data, not trench). With the rule that end is trimmed and the datum near
it is the real trench's; without it (None, v9) nothing changes.

Run:  .venv/Scripts/python.exe -m pytest from_scratch_v12/pyptha_v12/tests/test_trench_ramp.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lib")))

import slab_contours as sc  # noqa: E402

KM = 111.195


def _slab(ramp=True):
    x = np.arange(100.0, 106.0001, 0.02)
    y = np.arange(-2.0, 8.0001, 0.02)
    X, Y = np.meshgrid(x, y)
    trench_lon = 101.01
    dist_km = (X - trench_lon) * KM * np.cos(np.radians(Y))
    along_km = (Y - Y.min()) * KM               # 0 at the southern end
    trench_depth = np.full_like(X, 8.0)
    if ramp:                                    # 20 km at the south end, 8 km from 80 km on
        trench_depth = np.where(along_km < 80, 20.0 - 12.0 * along_km / 80.0, 8.0)
    depth = trench_depth + dist_km * np.tan(np.radians(15.0))
    depth[dist_km < 0] = np.nan
    depth[depth > 120] = np.nan
    return x, y, depth


def test_ramp_end_is_trimmed():
    x, y, depth = _slab()
    tr = sc.find_trench(x, y, depth, ramp_rise_km=3.0)
    start, end = tr["ramp_trim_km"]
    # 3 km above the normal 8 km is reached 60 km in; the 50 km running
    # median smears the edge by up to about 25 km
    assert 35.0 < start + end < 85.0
    assert min(start, end) == 0.0
    s = sc.line_km(tr["trench"])
    d = sc.running_median(tr["trench_depth_raw"], s, 50.0)
    assert d.max() < 8.0 + 3.0 + 1.0


def test_off_is_v9():
    x, y, depth = _slab()
    a = sc.find_trench(x, y, depth)
    b = sc.find_trench(x, y, depth, ramp_rise_km=None)
    assert np.array_equal(a["trench"], b["trench"])
    assert a["ramp_trim_km"] == [0.0, 0.0]


def test_no_ramp_nothing_to_trim():
    x, y, depth = _slab(ramp=False)
    a = sc.find_trench(x, y, depth)
    b = sc.find_trench(x, y, depth, ramp_rise_km=3.0)
    assert b["ramp_trim_km"] == [0.0, 0.0]
    assert np.array_equal(a["trench"], b["trench"])


def test_contours_use_the_real_trench_as_datum():
    # a point 20 km from the ramp end along strike, 60 km down dip: 17 + 16 =
    # 33 km below sea level. v9 measures it from the ramp (about 17 km), the
    # rule from the real trench further along (8 to 11 km), so its depth below
    # the trench grows by several km
    x, y, depth = _slab()
    _, off = sc.below_trench_contours(x, y, depth, 40.0, verbose=False)
    _, on = sc.below_trench_contours(x, y, depth, 40.0, verbose=False, ramp_rise_km=3.0)
    lat = -2.0 + 20.0 / KM
    lon = 101.01 + 60.0 / (KM * np.cos(np.radians(lat)))
    j, i = np.argmin(np.abs(y - lat)), np.argmin(np.abs(x - lon))
    assert on["datum_field"][j, i] - off["datum_field"][j, i] > 4.0
