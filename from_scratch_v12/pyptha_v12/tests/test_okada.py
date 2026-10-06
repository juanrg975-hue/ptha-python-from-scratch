"""Validation tests for pyptha.okada against known Okada (1985) behaviour.

Since rptha/R/rptha is not compiled on this machine, we validate against the
physical properties of the Okada solution and the reference example in the
rptha docstring (a pure thrust fault produces a characteristic uplift lobe).
These checks pin down the sign conventions and the coordinate transform,
which are the error-prone parts of the port.

Run:  python pyptha/tests/test_okada.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import okada  # noqa: E402


def _grid(n=101, half=100.5e3):
    ax = np.linspace(-half, half, n)
    gx, gy = np.meshgrid(ax, ax)
    return gx.ravel(), gy.ravel()


def test_thrust_fault_uplift_and_subsidence():
    # rptha docstring example: pure thrust, dip 15 deg, dip-slip 1 m.
    # A thrust fault uplifts the hanging wall and subsides the footwall.
    rlon, rlat = _grid()
    out = okada.okada_tsunami(
        elon=0.0, elat=0.0, edep=30.0, strk=0.0, dip=15.0,
        lnth=1.0, wdt=1.0, disl1=0.0, disl2=1.0,
        rlon=rlon, rlat=rlat)
    z = out["zdsp"]
    assert np.max(z) > 0.0, "thrust fault must produce uplift somewhere"
    assert np.min(z) < 0.0, "thrust fault must produce subsidence somewhere"
    # Uplift should dominate the deformation volume for a thrust.
    assert np.max(z) > abs(np.min(z))


def test_output_shapes_and_finiteness():
    rlon, rlat = _grid(51)
    out = okada.okada_tsunami(
        0.0, 0.0, 20.0, 45.0, 30.0, 10.0, 8.0, 0.5, 1.0, rlon, rlat)
    for key in ("edsp", "ndsp", "zdsp"):
        assert out[key].shape == rlon.shape
        assert np.all(np.isfinite(out[key]))


def test_superposition_two_faults():
    # The two-fault result must equal the sum of the individual results.
    rlon, rlat = _grid(41)
    common = dict(rlon=rlon, rlat=rlat)

    a = okada.okada_tsunami(0.0, 0.0, 30.0, 0.0, 15.0, 1.0, 1.0, 0.0, 1.0,
                            **common)
    b = okada.okada_tsunami(0.0, 50e3, 25.0, 0.0, 12.0, 1.0, 1.0, 0.0, 1.0,
                            **common)
    both = okada.okada_tsunami(
        elon=[0.0, 0.0], elat=[0.0, 50e3], edep=[30.0, 25.0],
        strk=[0.0, 0.0], dip=[15.0, 12.0], lnth=[1.0, 1.0], wdt=[1.0, 1.0],
        disl1=[0.0, 0.0], disl2=[1.0, 1.0], **common)
    for key in ("edsp", "ndsp", "zdsp"):
        np.testing.assert_allclose(both[key], a[key] + b[key], atol=1e-12)


def test_strike_rotation_symmetry():
    # Rotating the fault strike by 90 deg and the observation grid by 90 deg
    # must rotate the vertical field the same way (z is a scalar field).
    n = 61
    rlon, rlat = _grid(n)
    z0 = okada.okada_tsunami(
        0.0, 0.0, 30.0, 0.0, 15.0, 20.0, 15.0, 0.0, 1.0,
        rlon, rlat)["zdsp"].reshape(n, n)
    # Strike 90: fault rotated; sample on a grid rotated by 90 (x->y, y->-x).
    z90 = okada.okada_tsunami(
        0.0, 0.0, 30.0, 90.0, 15.0, 20.0, 15.0, 0.0, 1.0,
        rlat, -rlon)["zdsp"].reshape(n, n)
    # These vertical fields should match closely.
    np.testing.assert_allclose(z0, z90, atol=1e-6)


def test_distance_cutoff_zeroes_far_field():
    # With a tight cutoff, far points get exactly zero contribution.
    rlon = np.array([0.0, 5e5])   # one near, one 500 km away
    rlat = np.array([0.0, 0.0])
    out = okada.okada_tsunami(
        0.0, 0.0, 30.0, 0.0, 15.0, 1.0, 1.0, 0.0, 1.0,
        rlon, rlat, dstmx=1.0, dstmx_min=20.0)  # cutoff ~30 km
    assert out["zdsp"][1] == 0.0
    assert out["zdsp"][0] != 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
