"""v11 variable shear modulus (PTHA18 report Section 3.7.5)."""

import os

import numpy as np
import pytest

from pyptha_v12 import hs_vaus_rates as hvr
from pyptha_v12 import rates
from pyptha_v12 import variable_mu as vmu
from pyptha_v12.scaling import M0_2_Mw

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_BASE = os.path.abspath(os.path.join(_PKG, os.pardir, os.pardir))


def test_rigidity_curve_knots_and_log_interpolation():
    mu = vmu.shear_modulus_depth([0, 5, 7.5, 15, 35, 100])
    assert np.allclose(mu, [10e9, 10e9, 10e9, 30e9, 67e9, 67e9])
    # halfway (in depth) between 7.5 and 15 km: geometric mean, not arithmetic
    assert vmu.shear_modulus_depth(11.25) == pytest.approx(np.sqrt(10e9 * 30e9))
    assert np.isnan(vmu.shear_modulus_depth(-1.0))
    with pytest.raises(ValueError):
        vmu.shear_modulus_depth(10.0, kind="nope")


def test_variable_mu_Mw_relabels_by_depth():
    area = np.array([1000.0, 1000.0])          # km2
    depth = np.array([5.0, 40.0])              # 10 GPa and 67 GPa
    slip = 2.0
    shallow = vmu.variable_mu_Mw(["1-"], [slip], area, depth)[0]
    deep = vmu.variable_mu_Mw(["2-"], [slip], area, depth)[0]
    constant = M0_2_Mw(1000e6 * slip * 3e10)
    assert shallow == pytest.approx(constant + 2 / 3 * np.log10(10 / 30))
    assert deep == pytest.approx(constant + 2 / 3 * np.log10(67 / 30))
    # a slip string (HS/VAUS) gives the moment-weighted sum
    both = vmu.variable_mu_Mw(["1-2-"], ["1_3_"], area, depth)[0]
    assert both == pytest.approx(M0_2_Mw(1000e6 * (1 * 10e9 + 3 * 67e9)))


def test_conditional_ecdf_matches_r_semantics():
    x = np.array([-0.2, 0.0, 0.1, 0.3, -0.1, 0.2])
    cond = np.array([7.2, 7.2, 7.2, 7.3, 7.3, 7.3])
    F = vmu.make_conditional_ecdf(x, cond)
    # R ecdf is right-continuous: P(x <= t)
    assert F(0.0, 7.2) == pytest.approx(2 / 3)
    assert F(-0.21, 7.2) == 0.0 and F(1.0, 7.2) == 1.0
    # linear between the two bins, clamped outside
    assert F(0.0, 7.25) == pytest.approx(0.5 * 2 / 3 + 0.5 * 1 / 3)
    assert F(0.0, 7.0) == F(0.0, 7.2) and F(0.0, 9.0) == F(0.0, 7.3)
    out = F(np.array([0.0, 0.3]), 7.3)
    assert np.allclose(out, [1 / 3, 1.0])
    with pytest.raises(ValueError):
        vmu.make_conditional_ecdf(x, [7.2, 7.2, 7.25, 7.3, 7.3, 7.6])


def test_rate_function_with_mw_error_uses_those_weights():
    b = rates.LogicTreeBranches(
        all_par_combo=[{}, {}], a_parameter=np.zeros(2),
        all_par_prob=np.array([0.5, 0.5]), all_par_prob_prior=np.array([0.5, 0.5]),
        Mw_seq=np.array([7.0, 8.0]),
        all_rate_matrix=np.array([[1.0, 0.1], [3.0, 0.3]]),
        all_par_prob_with_Mw_error=np.array([0.9, 0.1]))
    f = rates.MwRateFunction(b)
    g = f.with_mw_error()
    assert f(7.0) == pytest.approx(2.0)
    assert g(7.0) == pytest.approx(0.9 * 1 + 0.1 * 3)
    assert g(7.0, quantiles=[0.5])[0] == pytest.approx(1.0)
    assert f(7.0, quantiles=[0.6])[0] == pytest.approx(3.0)
    # without an error model the weights are the posterior
    b2 = rates.LogicTreeBranches(**{**b.__dict__, "all_par_prob_with_Mw_error": None})
    assert rates.MwRateFunction(b2).with_mw_error()(7.0) == pytest.approx(2.0)


def test_variable_mu_weight_curves_present():
    w = hvr.load_peak_slip_weights()
    for k in ("HS", "VAUS", "HS_variable_mu", "VAUS_variable_mu"):
        q, f = w[k]
        assert q.size == 51 and np.trapezoid(f, q) == pytest.approx(1.0, rel=1e-9)
    a, _ = hvr.family_weights([1, 1, 1], [1.0, 2.0, 3.0], [8.0] * 3, "VAUS")
    b, _ = hvr.family_weights([1, 1, 1], [1.0, 2.0, 3.0], [8.0] * 3, "VAUS", variable_mu=True)
    assert a.sum() == pytest.approx(1) and b.sum() == pytest.approx(1)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("zone", ["puysegur2", "kermadectonga2"])
def test_reproduces_ptha18_published_variable_mu_Mw(zone):
    nc = pytest.importorskip("netCDF4")
    uss = os.path.join(_BASE, "inputs", "geometry", f"unit_source_statistics_{zone}.nc")
    ev = os.path.join(_BASE, "official_ptha_data", "public_nc",
                      f"all_variable_uniform_slip_earthquake_events_{zone}.nc")
    if not (os.path.exists(uss) and os.path.exists(ev)):
        pytest.skip("PTHA18's published files are not present")
    with nc.Dataset(uss) as u:
        u.set_auto_mask(False)
        area = u["length"][:] * u["width"][:]
        depth = u["depth"][:].astype(float)
    with nc.Dataset(ev) as f:
        f.set_auto_mask(False)
        eis = [s.strip() for s in nc.chartostring(f["event_index_string"][:])]
        ess = [s.strip() for s in nc.chartostring(f["event_slip_string"][:])]
        off = f["variable_mu_Mw"][:].astype(float)
    assert np.max(np.abs(vmu.variable_mu_Mw(eis, ess, area, depth) - off)) < 1e-9
