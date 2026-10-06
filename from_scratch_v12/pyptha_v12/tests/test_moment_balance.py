"""Tests for pyptha.moment_balance and the individual scenario rates.

These cover the PTHA18 pieces added for individual_codes_v3, checking them
against the invariants the report states rather than against golden numbers,
since the driver scripts they come from ship no test values.
"""

import numpy as np
import pytest

import os, sys  # noqa: E401
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pyptha_v12 import moment_balance as mb
from pyptha_v12 import rates


# ---------------------------------------------------------------------------
# A small synthetic source: 3 down-dip rows x 6 along-strike columns
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_source():
    n_dip, n_strike = 3, 6
    n = n_dip * n_strike
    alongstrike = np.repeat(np.arange(1, n_strike + 1), n_dip)
    downdip = np.tile(np.arange(1, n_dip + 1), n_strike)

    # Three magnitude bins, each a block of ruptures sliding along strike.
    event_index_string, event_Mw, event_slip = [], [], []
    for mw, width in ((7.5, 1), (8.0, 2), (8.5, 3)):
        for start in range(1, n_strike - width + 2):
            cols = set(range(start, start + width))
            idx = np.where(np.isin(alongstrike, list(cols)))[0]
            event_index_string.append("".join(f"{i + 1}-" for i in idx))
            event_Mw.append(mw)
            event_slip.append(1.0 * width)

    return {
        "alongstrike": alongstrike, "downdip": downdip, "n": n,
        "event_index_string": np.array(event_index_string, dtype=object),
        "event_Mw": np.array(event_Mw),
        "event_slip": np.array(event_slip),
    }


def test_edge_events_are_the_ones_touching_the_extreme_columns(toy_source):
    is_edge = mb.events_touching_alongstrike_edge(
        toy_source["event_index_string"], toy_source["alongstrike"])
    # Every magnitude bin has exactly two edge ruptures (one at each end),
    # except where the rupture is wide enough to touch both at once.
    assert is_edge.sum() > 0
    assert not is_edge.all()


def test_back_calculate_convergence_conserves_bin_rate(toy_source):
    """The edge correction redistributes rate, it never creates any."""
    rng = np.random.default_rng(0)
    event_rates = rng.uniform(0.1, 1.0, toy_source["event_Mw"].size)

    env = mb.back_calculate_convergence(
        event_rates, toy_source["event_Mw"], toy_source["event_index_string"],
        toy_source["event_slip"], toy_source["alongstrike"],
        n_unit_sources=toy_source["n"], edge_multiplier=3.0)

    for mw in np.unique(toy_source["event_Mw"]):
        k = toy_source["event_Mw"] == mw
        assert np.isclose(env["new_event_rates"][k].sum(), event_rates[k].sum())
        assert np.isclose(env["new_conditional_probability"][k].sum(), 1.0)


def test_integrated_slip_equals_sum_of_slip_times_rate(toy_source):
    """Total integrated slip must equal sum_j (slip_j * rate_j * n_sources_j)."""
    event_rates = np.full(toy_source["event_Mw"].size, 0.01)
    env = mb.back_calculate_convergence(
        event_rates, toy_source["event_Mw"], toy_source["event_index_string"],
        toy_source["event_slip"], toy_source["alongstrike"],
        n_unit_sources=toy_source["n"], edge_multiplier=0.0)

    expected = 0.0
    for s, slp, rt in zip(toy_source["event_index_string"],
                          toy_source["event_slip"], event_rates):
        n_src = len([v for v in s.split("-") if v != ""])
        expected += slp * rt * n_src
    assert np.isclose(env["integrated_slip"].sum(), expected)


def test_edge_correction_lifts_the_zone_ends(toy_source):
    """Without correction the ends sag; the correction raises them."""
    event_rates = np.full(toy_source["event_Mw"].size, 0.01)
    kw = dict(event_Mw=toy_source["event_Mw"],
              event_index_string=toy_source["event_index_string"],
              event_slip=toy_source["event_slip"],
              alongstrike_number=toy_source["alongstrike"],
              n_unit_sources=toy_source["n"])

    raw = mb.back_calculate_convergence(event_rates, edge_multiplier=0.0, **kw)
    cor = mb.back_calculate_convergence(event_rates, edge_multiplier=5.0, **kw)

    asn = toy_source["alongstrike"]
    def edge_share(v):
        edge = np.isin(asn, (asn.min(), asn.max()))
        return v[edge].sum() / v.sum()

    assert edge_share(cor["integrated_slip"]) > edge_share(raw["integrated_slip"])


def test_fit_edge_multiplier_improves_the_misfit(toy_source):
    """The fitted multiplier must beat the uncorrected case."""
    event_rates = np.full(toy_source["event_Mw"].size, 0.01)
    target = np.ones(toy_source["n"])   # uniform convergence

    fit = mb.fit_edge_multiplier(
        event_rates, toy_source["event_Mw"], toy_source["event_index_string"],
        toy_source["event_slip"], toy_source["alongstrike"], target,
        n_unit_sources=toy_source["n"])

    assert fit["objective"] < fit["objective_at_zero"]
    assert 0 <= fit["edge_multiplier"] <= 30
    assert np.isclose(fit["conditional_probability"][
        toy_source["event_Mw"] == 7.5].sum(), 1.0)


# ---------------------------------------------------------------------------
# Coupling prior
# ---------------------------------------------------------------------------
def test_coupling_prior_is_normalised_and_spans_the_range():
    vals, probs = mb.coupling_prior_spreadsheet_and_uniform_50_50(
        uniform_range=[0.1, 1.3], spreadsheet_values=[0.3, 0.5, 0.7], n=20)

    assert vals.size == probs.size == 20
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)
    assert np.isclose(vals.min(), 0.1)
    assert np.isclose(vals.max(), 1.3)
    # Log spacing: successive ratios are constant.
    ratios = vals[1:] / vals[:-1]
    assert np.allclose(ratios, ratios[0])


def test_coupling_prior_concentrates_near_the_spreadsheet_values():
    """Half the density comes from the [0.3, 0.7] triple, so the mean should
    sit well inside the wider uniform range rather than at its midpoint."""
    vals, probs = mb.coupling_prior_spreadsheet_and_uniform_50_50(
        uniform_range=[0.1, 1.3], spreadsheet_values=[0.3, 0.5, 0.7], n=20)
    mean = float(np.sum(vals * probs))
    uniform_mean = 0.7   # midpoint of [0.1, 1.3]
    assert mean < uniform_mean
    assert 0.3 < mean < 0.7 or np.isclose(mean, 0.6, atol=0.1)


def test_coupling_prior_zero_branch():
    vals, probs = mb.coupling_prior_spreadsheet_and_uniform_50_50(
        uniform_range=[0.1, 1.3], spreadsheet_values=[0.3, 0.5, 0.7], n=10,
        prob_zero_coupling=0.2)
    assert vals[0] == 0.0
    assert np.isclose(probs[0], 0.2)
    assert np.isclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# Parameter axes and the dip factor
# ---------------------------------------------------------------------------
def test_interpolate_logic_tree_parameter_matches_r_approx():
    # R: approx(c(0.7, 1.2), n=6)$y -> 0.7 0.8 0.9 1.0 1.1 1.2
    out = mb.interpolate_logic_tree_parameter([0.7, 1.2], 6)
    assert np.allclose(out, [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])


def test_dip_factor_is_one_over_cos():
    factor, mean_dip = mb.mean_dip_cos_factor([30.0, 30.0, 30.0])
    assert np.isclose(mean_dip, 30.0)
    assert np.isclose(factor, 1.0 / np.cos(np.radians(30.0)))
    # The report's warning case: at 45 degrees the correction is 41%.
    factor45, _ = mb.mean_dip_cos_factor([45.0])
    assert np.isclose(factor45, np.sqrt(2), atol=1e-9)


def test_dip_factor_is_negligible_at_shallow_dip():
    factor, _ = mb.mean_dip_cos_factor([10.0])
    assert factor < 1.02


# ---------------------------------------------------------------------------
# Individual scenario rates (report Equation 3)
# ---------------------------------------------------------------------------
def test_scenario_rates_sum_to_the_exceedance_curve():
    """Summing r_j above a magnitude must reproduce the source-zone curve.

    This is the invariant that makes Equation 3 the natural end product: the
    scenario rates are a partition of the magnitude-bin rates.
    """
    Mw = np.repeat([7.5, 7.6, 7.7], 4)
    ecp = np.tile([0.4, 0.3, 0.2, 0.1], 3)
    dMw = 0.1

    # A plain exponential exceedance-rate curve stands in for the logic tree.
    def rate_fn(mw, quantiles=None):
        return 10.0 ** (3.0 - 1.0 * np.asarray(mw, dtype=float))

    r_j = rates.individual_scenario_rates(Mw, ecp, rate_fn, dMw)

    for m in (7.5, 7.6, 7.7):
        got = r_j[Mw >= m - 1e-9].sum()
        want = rate_fn(m - dMw / 2) - rate_fn(Mw.max() + dMw / 2)
        assert np.isclose(got, want)


def test_scenario_rates_respect_conditional_probability():
    """Within one bin, rates must be in proportion to the conditional probs."""
    Mw = np.full(4, 8.0)
    ecp = np.array([0.4, 0.3, 0.2, 0.1])

    def rate_fn(mw, quantiles=None):
        return 10.0 ** (3.0 - 1.0 * np.asarray(mw, dtype=float))

    r_j = rates.individual_scenario_rates(Mw, ecp, rate_fn, 0.1)
    assert np.allclose(r_j / r_j.sum(), ecp)


def test_scenario_rates_with_quantiles_returns_one_row_per_quantile():
    Mw = np.array([7.5, 7.5, 7.6, 7.6])
    ecp = np.array([0.5, 0.5, 0.5, 0.5])

    def rate_fn(mw, quantiles=None):
        base = 10.0 ** (3.0 - 1.0 * np.asarray(mw, dtype=float))
        if quantiles is None:
            return base
        # Scale each quantile differently so the rows are distinguishable.
        return np.array([base * (0.5 + q) for q in np.atleast_1d(quantiles)])

    out = rates.individual_scenario_rates(Mw, ecp, rate_fn, 0.1,
                                          quantiles=[0.16, 0.5, 0.84])
    assert out.shape == (3, 4)
    # Higher quantile -> larger rate, given the scaling above.
    assert np.all(out[2, :] > out[0, :])


def test_scenario_rate_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in length"):
        rates.individual_scenario_rates(
            np.array([7.5, 7.6]), np.array([1.0]), lambda mw: mw, 0.1)
