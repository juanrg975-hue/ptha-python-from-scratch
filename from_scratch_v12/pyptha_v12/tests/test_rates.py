"""Validation tests for pyptha.rates.

Golden references come from rptha's ``single_source_rate_computation.R``
(the Alaska example) and from the analytic properties of the Gutenberg-Richter
logic-tree machinery.

Run:  python pyptha/tests/test_rates.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import rates, scaling  # noqa: E402


# --- Gutenberg-Richter closed forms -----------------------------------------

def test_truncated_gr_endpoints():
    a, b, mwmin, mwmax = 4.0, 1.0, 7.5, 9.4
    # At Mw = Mw_min the exceedance rate is 10^(a-b*mwmin) - 10^(a-b*mwmax).
    r = rates.Mw_exceedance_rate_truncated_gutenberg_richter(mwmin, a, b, mwmin, mwmax)
    assert np.isclose(r, 10 ** (a - b * mwmin) - 10 ** (a - b * mwmax))
    # Above Mw_max the rate is exactly zero.
    assert rates.Mw_exceedance_rate_truncated_gutenberg_richter(
        mwmax + 0.1, a, b, mwmin, mwmax) == 0.0


def test_characteristic_gr_positive_at_mwmax():
    a, b, mwmin, mwmax = 4.0, 1.0, 7.5, 9.4
    r = rates.Mw_exceedance_rate_characteristic_gutenberg_richter(mwmax, a, b, mwmin, mwmax)
    assert r > 0.0  # characteristic form keeps finite rate at Mw_max


def test_gr_monotonic_decreasing():
    Mw = np.linspace(7.5, 9.4, 50)
    r = rates.Mw_exceedance_rate_truncated_gutenberg_richter(Mw, 4.0, 1.0, 7.5, 9.4)
    assert np.all(np.diff(r) <= 1e-15)


# --- conditional probabilities ----------------------------------------------

def test_conditional_prob_sums_to_one_per_bin():
    et = {"Mw": np.array([7.5, 7.5, 7.5, 7.6, 7.6]),
          "slip": np.array([1.0, 2.0, 4.0, 1.0, 3.0])}
    p = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    for mw in np.unique(et["Mw"]):
        assert np.isclose(p[et["Mw"] == mw].sum(), 1.0)
    # inverse_slip: bigger slip -> smaller conditional probability
    idx = np.where(et["Mw"] == 7.5)[0]
    assert p[idx][0] > p[idx][1] > p[idx][2]


# --- synthetic single-source event table ------------------------------------

def _synthetic_source(dMw=0.1, mw_lo=7.5, mw_hi=9.4, n_per_mw=10, area_total=2.0e5):
    """A simple evenly-spaced event table mimicking a discretised source.

    Each magnitude bin has n_per_mw events; area/slip follow the Strasser
    scaling relation for that Mw (so the moment balance is well posed).
    """
    mws = np.round(np.arange(mw_lo, mw_hi + 1e-9, dMw), 4)
    Mw, slip, area = [], [], []
    for mw in mws:
        sz = scaling.Mw_2_rupture_size(mw, relation="Strasser")
        s = scaling.slip_from_Mw_area_mu(mw, sz["area"])
        for _ in range(n_per_mw):
            Mw.append(mw)
            area.append(sz["area"])
            slip.append(s)
    return {"Mw": np.array(Mw), "slip": np.array(slip), "area": np.array(area)}, area_total


def test_rate_function_mean_monotone_and_positive():
    et, area = _synthetic_source()
    ecp = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=[0.04], slip_rate_prob=[1.0],
        b=[0.95], b_prob=[1.0],
        Mw_min=[7.5], Mw_min_prob=[1.0],
        Mw_max=[9.4], Mw_max_prob=[1.0],
        sourcezone_total_area=area,
        event_table=et, event_conditional_probabilities=ecp,
        Mw_frequency_distribution="truncated_gutenberg_richter")
    Mws = np.linspace(7.5, 9.3, 40)
    r = rate_fn(Mws)
    assert np.all(r > 0)
    assert np.all(np.diff(r) <= 1e-12)  # exceedance rate is non-increasing


def test_moment_balance_recovers_slip_rate():
    # If we integrate moment over the modelled rates, we must recover the
    # input long-term seismic slip rate (this is exactly what the 'a'
    # back-calculation enforces).
    et, area = _synthetic_source()
    ecp = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    slip_rate_in = 0.04
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=[slip_rate_in], slip_rate_prob=[1.0],
        b=[0.95], b_prob=[1.0], Mw_min=[7.5], Mw_min_prob=[1.0],
        Mw_max=[9.4], Mw_max_prob=[1.0], sourcezone_total_area=area,
        event_table=et, event_conditional_probabilities=ecp)

    br = rate_fn(return_all_logic_tree_branches=True)
    # Per-event rate = conditional_prob * (rate(Mw-dMw/2) - rate(Mw+dMw/2))
    Mfd = rates.Mw_exceedance_rate_truncated_gutenberg_richter
    par = br.all_par_combo[0]
    dMw = 0.1
    per_event = ecp * (
        Mfd(et["Mw"] - dMw / 2, par["a"], par["b"], par["Mw_min"], par["Mw_max"])
        - Mfd(et["Mw"] + dMw / 2, par["a"], par["b"], par["Mw_min"], par["Mw_max"]))
    # Moment balance: mu * slip_rate * Area == sum(mu * slip_i * area_i * rate_i)
    mu = 3e10
    lhs = mu * slip_rate_in * (area * 1e6)
    rhs = np.sum(mu * et["slip"] * (et["area"] * 1e6) * per_event)
    assert np.isclose(lhs, rhs, rtol=1e-6)


def test_logic_tree_bounds_bracket_mean():
    et, area = _synthetic_source()
    ecp = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=[0.03, 0.04, 0.05], slip_rate_prob=[1 / 3, 1 / 3, 1 / 3],
        b=[0.7, 0.95, 1.2], b_prob=[1 / 3, 1 / 3, 1 / 3],
        Mw_min=[7.5], Mw_min_prob=[1.0],
        Mw_max=[9.3, 9.4], Mw_max_prob=[0.5, 0.5],
        sourcezone_total_area=area, event_table=et,
        event_conditional_probabilities=ecp)
    out = rate_fn(np.array([8.0, 8.5, 9.0]), bounds=True)
    assert np.all(out["lower"] <= out["rate"] + 1e-15)
    assert np.all(out["rate"] <= out["upper"] + 1e-15)
    assert np.all(out["lower"] <= out["median"] + 1e-15)
    assert np.all(out["median"] <= out["upper"] + 1e-15)


def test_bayesian_update_shifts_weights():
    # An observed count of zero large events should down-weight high-rate
    # branches relative to the prior.
    et, area = _synthetic_source()
    ecp = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    common = dict(
        slip_rate=[0.03, 0.05], slip_rate_prob=[0.5, 0.5],
        b=[0.95], b_prob=[1.0], Mw_min=[7.5], Mw_min_prob=[1.0],
        Mw_max=[9.4], Mw_max_prob=[1.0], sourcezone_total_area=area,
        event_table=et, event_conditional_probabilities=ecp)
    prior_fn = rates.rate_of_earthquakes_greater_than_Mw_function(**common)
    post_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        update_logic_tree_weights_with_data=True,
        Mw_count_duration=(7.5, 0, 38), **common)
    prior = prior_fn(return_all_logic_tree_branches=True).all_par_prob
    post = post_fn(return_all_logic_tree_branches=True).all_par_prob
    # Weights must still be a valid distribution and must have changed.
    assert np.isclose(post.sum(), 1.0)
    assert not np.allclose(prior, post)


def test_alaska_like_return_period_order_of_magnitude():
    # The rptha Alaska example states rate(Mw>9.0) ~ 1/540 per year with its
    # full logic tree and real unit sources. Our synthetic source cannot
    # reproduce that exactly, but the machinery must give a physically
    # plausible large-event return period (centuries to millennia).
    et, area = _synthetic_source(area_total=3.5e5)
    ecp = rates.get_event_probabilities_conditional_on_Mw(et, "inverse_slip")
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=[0.05], slip_rate_prob=[1.0], b=[0.95], b_prob=[1.0],
        Mw_min=[7.5], Mw_min_prob=[1.0], Mw_max=[9.4], Mw_max_prob=[1.0],
        sourcezone_total_area=area, event_table=et,
        event_conditional_probabilities=ecp,
        account_for_moment_below_mwmin=True)
    r9 = float(rate_fn(9.0))
    return_period = 1.0 / r9
    assert 100 < return_period < 10000


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
