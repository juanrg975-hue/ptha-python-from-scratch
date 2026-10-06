"""v12 mesh_file: an external quadrilateral mesh read into the node array."""

import os

import numpy as np
import pytest

from pyptha_v12 import mesh_file
from pyptha_v12 import unit_sources as us

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_V9 = os.path.dirname(_PKG)


def planar_grid():
    """A dipping planar fault, rptha's convention (dip to the right)."""
    return us.make_planar_unit_source_grid(
        lon0=170.0, lat0=-20.0, strike=20.0, dip=15.0, n_alongstrike=6,
        n_downdip=3, subfault_length=50.0, subfault_width=40.0, top_depth=0.0)


def write_dat(grid, path, order=None, start=0, reverse_ring=False,
              units="m", negative=True):
    nr, nc = grid.shape[0] - 1, grid.shape[2] - 1
    lines = []
    for i in range(nr):
        for j in range(nc):
            ring = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
            if reverse_ring:
                ring = ring[::-1]
            ring = ring[start:] + ring[:start]
            vals = []
            for a, b in ring:
                d = grid[a, 2, b] * (1000.0 if units == "m" else 1.0)
                vals += [grid[a, 0, b], grid[a, 1, b], -d if negative else d]
            lines.append(" ".join(f"{v:.12f}" for v in vals))
    if order is not None:
        lines = [lines[k] for k in order]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


@pytest.mark.parametrize("start,reverse_ring,units,negative,shuffle", [
    (0, False, "m", True, False),
    (2, False, "m", True, True),
    (1, True, "km", False, True),
    (3, True, "km", True, False),
])
def test_round_trip_any_order(tmp_path, start, reverse_ring, units, negative, shuffle):
    g = planar_grid()
    order = np.random.default_rng(3).permutation(18) if shuffle else None
    p = tmp_path / "m.dat"
    write_dat(g, p, order, start, reverse_ring, units, negative)
    out = mesh_file.read_quadrilateral_mesh(p, log=lambda *a: None)
    assert out.shape == g.shape
    assert np.allclose(out, g, atol=1e-9)


def test_columns_reversed_to_dip_right(tmp_path):
    g = planar_grid()
    flipped = g[:, :, ::-1].copy()   # dips to the LEFT of its strike
    p = tmp_path / "m.dat"
    write_dat(flipped, p)
    out = mesh_file.read_quadrilateral_mesh(p, log=lambda *a: None)
    assert np.allclose(out, g, atol=1e-9)


def test_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "bad.dat"
    p.write_text("1 2 3 4 5 6 7 8 9\n")
    with pytest.raises(ValueError, match="12 numbers"):
        mesh_file.read_quadrilateral_mesh(p, log=lambda *a: None)


def test_rejects_missing_cell(tmp_path):
    g = planar_grid()
    p = tmp_path / "hole.dat"
    write_dat(g, p, order=[k for k in range(18) if k != 7])
    with pytest.raises(ValueError):
        mesh_file.read_quadrilateral_mesh(p, log=lambda *a: None)


def test_alaska_file_matches_ptha18_table():
    """The file the colleagues sent is PTHA18's alaskaaleutians mesh: every
    unit source's centre, strike and length as in PTHA18's table; depth and
    dip too except the trench row (the file puts the trench at 0.1 km)."""
    nc = pytest.importorskip("netCDF4")
    dat = os.path.join(_V9, "alaskaaleutians_quadrilateral_coors.dat")
    ref = os.path.join(_PKG, "validation", "ptha18_reference", "nc",
                       "unit_source_statistics_alaskaaleutians.nc")
    if not (os.path.exists(dat) and os.path.exists(ref)):
        pytest.skip("the Alaska mesh or PTHA18's table is not present")
    st = us.discretized_source_approximate_summary_statistics(
        mesh_file.read_quadrilateral_mesh(dat, log=lambda *a: None))
    with nc.Dataset(ref) as d:
        d.set_auto_mask(False)
        o = {k: np.asarray(d[k][:], float) for k in ("lon_c", "lat_c", "depth", "dip",
                                                      "strike", "length", "downdip_number",
                                                      "alongstrike_number")}
    ko = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(o["alongstrike_number"], o["downdip_number"]))}
    idx = np.array([ko[(int(a), int(b))] for a, b in zip(st["alongstrike_number"], st["downdip_number"])])
    for f in ("lon_c", "lat_c"):
        assert np.max(np.abs(np.asarray(st[f]) - o[f][idx])) < 1e-6
    assert np.max(np.abs((np.asarray(st["strike"]) - o["strike"][idx] + 180) % 360 - 180)) < 1e-6
    deep = np.asarray(st["downdip_number"]) > 1
    for f in ("depth", "dip", "length"):
        assert np.max(np.abs(np.asarray(st[f])[deep] - o[f][idx][deep])) < 1e-6
    # trench row: the file's trench is at 0.1 km, PTHA18's at 0
    trench = ~deep
    assert np.max(np.abs(np.asarray(st["depth"])[trench] - o["depth"][idx][trench])) <= 0.05 + 1e-9
    assert np.max(np.abs(np.asarray(st["length"])[trench] - o["length"][idx][trench])) < 1e-3
