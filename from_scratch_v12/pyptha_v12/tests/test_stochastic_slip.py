"""Validation tests for pyptha.stochastic_slip (SFFM generator).

Validated against the invariants stated in the rptha docstring example:
clipping produces patches of zero slip (min == 0), the peak is recentred to
the template's peak, and the mean slip is preserved.

Run:  python pyptha/tests/test_stochastic_slip.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import stochastic_slip as ss  # noqa: E402


def _template():
    tg = np.zeros((8, 12))
    tg[2, 4] = 1.0  # fixed peak-slip location, as in the R example
    xs = np.linspace(0, 120, tg.shape[1])
    ys = np.linspace(0, 50, tg.shape[0])
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    reg_par = (1 / 50 * dx, 1 / 20 * dy)  # numerical corner wavenumbers
    return tg, reg_par


def test_output_shape():
    tg, reg = _template()
    m = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(0))
    assert m.shape == tg.shape


def test_clipping_produces_zero_slip():
    # From the R docstring: min(random_slip_mat) == 0 (default clipping).
    tg, reg = _template()
    m = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(1))
    assert np.isclose(m.min(), 0.0)
    assert np.any(m == 0.0)


def test_mean_slip_preserved():
    tg, reg = _template()
    tg = tg * 3.7  # arbitrary total slip
    m = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(2))
    assert np.isclose(m.sum(), tg.sum())


def test_nonnegative():
    tg, reg = _template()
    m = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(3))
    assert np.all(m >= 0.0)


def test_peak_recentred_to_template_peak():
    tg, reg = _template()
    m = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(4))
    # The output peak should coincide with the template's peak (recentre_slip).
    assert np.unravel_index(np.argmax(m), m.shape) == (2, 4)


def test_reproducible_with_seed():
    tg, reg = _template()
    m1 = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(42))
    m2 = ss.sffm_simulate(reg, tg, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(m1, m2)


def test_wavenumber_grids():
    tg = np.zeros((4, 6))
    kx, ky = ss.sffm_get_numerical_wavenumbers(tg)
    assert kx.shape == tg.shape and ky.shape == tg.shape
    # kx varies along columns, is symmetric, and starts at 0.
    assert kx[0, 0] == 0.0
    assert np.allclose(kx[0], np.minimum(np.arange(6), 6 - np.arange(6)) / 6)
    assert np.allclose(ky[:, 0], np.minimum(np.arange(4), 4 - np.arange(4)) / 4)


def test_absolute_value_removal_option():
    # Changing the negative-slip removal to abs() removes the zero-slip patches.
    tg, reg = _template()
    pars = ss.sffm_get_default_model_parameters()
    pars.negative_slip_removal_function = np.abs
    pars.spatial_slip_decay = "none"
    pars.recentre_slip = False
    m = ss.sffm_simulate(reg, tg, sffm_pars=pars, rng=np.random.default_rng(5))
    assert np.all(m >= 0.0)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
