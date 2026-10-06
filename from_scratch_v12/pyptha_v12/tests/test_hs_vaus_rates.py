"""hs_vaus_rates: PTHA18's sharing of a FAUS rate among its HS/VAUS scenarios
(compute_rates_all_sources.R 1551-1600)."""

import os

import numpy as np
import pytest

from pyptha_v12 import hs_vaus_rates as hvr
from pyptha_v12.scaling import slip_from_Mw

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_NC = os.path.join(_PKG, os.pardir, os.pardir, "official_ptha_data", "public_nc")

FLAT = {"HS": (np.linspace(0, 1, 51), np.ones(51)),
        "VAUS": (np.linspace(0, 1, 51), np.ones(51))}


def test_table_shape_and_integral():
    w = hvr.load_peak_slip_weights()
    for kind in ("HS", "VAUS"):
        q, f = w[kind]
        assert q.size == 51 and np.allclose(q, np.linspace(0, 1, 51))
        assert np.all(f > 0)
        assert np.trapezoid(f, q) == pytest.approx(1.0, rel=1e-9)
    # PTHA18: HS too high-slip (weight falls with peak slip), VAUS too low
    assert w["HS"][1][-1] < w["HS"][1][0]
    assert w["VAUS"][1][40] > w["VAUS"][1][10]


def test_families_sum_to_one_and_limit_gets_zero():
    mw = 8.0
    limit = 7.5 * float(slip_from_Mw(mw))
    parent = [1, 1, 1, 2, 2]
    peak = [1.0, 2.0, limit * 1.01, 3.0, 4.0]
    w, above = hvr.family_weights(parent, peak, [mw] * 5, "VAUS")
    assert above.tolist() == [False, False, True, False, False]
    assert w[2] == 0.0
    assert w[:2].sum() == pytest.approx(1.0)
    assert w[3:].sum() == pytest.approx(1.0)


def test_family_all_above_limit_carries_nothing():
    mw = 7.5
    big = 10 * float(slip_from_Mw(mw))
    w, above = hvr.family_weights([5, 5], [big, big * 2], [mw, mw], "HS")
    assert above.all() and np.all(w == 0)


def test_ranks_ties_in_generation_order():
    # q_i = i / (n + 1): a weight rising with q gives later ties more
    rising = {"HS": (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
              "VAUS": FLAT["VAUS"]}
    w, _ = hvr.family_weights([0, 0, 0], [2.0, 2.0, 1.0], [8.0] * 3, "HS",
                              weights=rising)
    # ranks: third 1, first 2, second 3
    f = 1 + np.array([2, 3, 1]) / 4
    assert np.allclose(w, f / f.sum())


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        hvr.family_weights([0], [1.0], [8.0], "FAUS")


@pytest.mark.parametrize("zone", ["puysegur2", "kermadectonga2"])
@pytest.mark.parametrize("kind,name", [("HS", "stochastic"), ("VAUS", "variable_uniform")])
def test_reproduces_ptha18_published_rates(zone, kind, name):
    """Every published HS/VAUS rate from the published FAUS rates."""
    nc = pytest.importorskip("netCDF4")
    u_path = os.path.join(_NC, f"all_uniform_slip_earthquake_events_{zone}.nc")
    f_path = os.path.join(_NC, f"all_{name}_slip_earthquake_events_{zone}.nc")
    if not (os.path.exists(u_path) and os.path.exists(f_path)):
        pytest.skip("PTHA18's published event files are not present")
    with nc.Dataset(u_path) as u, nc.Dataset(f_path) as f:
        u.set_auto_mask(False)
        f.set_auto_mask(False)
        parent_rate = u["rate_annual"][:].astype(float)
        parent_mw = u["Mw"][:].astype(float)
        row = f["uniform_event_row"][:].astype(int)
        slip = nc.chartostring(f["event_slip_string"][:])
        rate = f["rate_annual"][:].astype(float)
    peak = np.array([max(float(x) for x in s.split("_") if x) for s in slip])
    w, _ = hvr.family_weights(row, peak, parent_mw[row - 1], kind)
    pred = w * parent_rate[row - 1]
    nz = rate > 0
    assert np.all(pred[~nz] == 0)
    assert np.max(np.abs(pred[nz] - rate[nz]) / rate[nz]) < 1e-9
