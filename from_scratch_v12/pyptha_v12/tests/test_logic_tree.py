"""Tests for the logic-tree levels in pyptha.logic_tree.

Reference values in test_weighted_percentile_matches_R and
test_quantile_type6_matches_R were produced by running the R originals
(make_stage_vs_rate_curve_fast.R's weighted_percentile, and R's
quantile(type=6)) on the same inputs.
"""

import numpy as np
import pytest

import os, sys  # noqa: E401
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pyptha_v12 import logic_tree as lt


# ---------------------------------------------------------------------------
# LEVEL 0 -- segmentation
# ---------------------------------------------------------------------------

def test_segmentation_with_segments_is_50_50():
    out = lt.get_unsegmented_and_segmented_source_names_on_source_zone(
        "alaska", ["alaska", "alaska_west", "alaska_east", "puysegur"])
    assert out["unsegmented_weight"] == 0.5
    assert out["union_of_segments_weight"] == 0.5
    assert out["segments"] == ["alaska_west", "alaska_east"]


def test_segmentation_without_segments_is_100_0():
    out = lt.get_unsegmented_and_segmented_source_names_on_source_zone(
        "puysegur", ["alaska", "alaska_west", "puysegur"])
    assert out["unsegmented_weight"] == 1.0
    assert out["union_of_segments_weight"] == 0.0
    assert out["segments"] == []


def test_segmentation_requires_unique_source_zone():
    with pytest.raises(ValueError):
        lt.get_unsegmented_and_segmented_source_names_on_source_zone(
            "alaska", ["puysegur"])


# ---------------------------------------------------------------------------
# LEVEL 4 -- edge correction
# ---------------------------------------------------------------------------

def _toy_events():
    """Three scenarios in one Mw bin on a 3-column zone.

    Scenario 0 touches column 1, scenario 2 touches column 3: both are edge
    ruptures. Scenario 1 sits in the middle.
    """
    Mw = np.array([8.0, 8.0, 8.0])
    ecp = np.array([1 / 3, 1 / 3, 1 / 3])
    eis = ["1-", "2-", "3-"]
    alongstrike = np.array([1, 2, 3])
    return Mw, ecp, eis, alongstrike


def test_edge_correction_zero_multiplier_is_identity():
    Mw, ecp, eis, als = _toy_events()
    out = lt.edge_correct_conditional_probabilities(Mw, ecp, eis, als, 0.0)
    assert np.allclose(out, ecp)


def test_edge_correction_moves_weight_to_edges_and_renormalises():
    Mw, ecp, eis, als = _toy_events()
    out = lt.edge_correct_conditional_probabilities(Mw, ecp, eis, als, 1.0)
    # Still a probability distribution within the magnitude bin: the
    # correction redistributes rate, it does not create any.
    assert np.isclose(out.sum(), 1.0)
    # The two edge scenarios gained, the middle one lost.
    assert out[0] > ecp[0]
    assert out[2] > ecp[2]
    assert out[1] < ecp[1]
    # With multiplier 1 the edges are doubled before renormalising: 2:1:2.
    assert np.allclose(out, np.array([0.4, 0.2, 0.4]))


def test_edge_correction_renormalises_within_each_Mw_bin():
    Mw = np.array([8.0, 8.0, 8.5, 8.5])
    ecp = np.array([0.5, 0.5, 0.5, 0.5])
    eis = ["1-", "2-", "1-", "2-"]
    als = np.array([1, 2])
    out = lt.edge_correct_conditional_probabilities(Mw, ecp, eis, als, 0.5)
    for mw in (8.0, 8.5):
        assert np.isclose(out[Mw == mw].sum(), 1.0)


def test_edge_correction_rejects_negative_multiplier():
    Mw, ecp, eis, als = _toy_events()
    with pytest.raises(ValueError):
        lt.edge_correct_conditional_probabilities(Mw, ecp, eis, als, -1.0)


# ---------------------------------------------------------------------------
# LEVEL 5 -- weighted percentile
# ---------------------------------------------------------------------------

def test_weighted_percentile_matches_R():
    # Reference from R: weighted_percentile(c(3,1,2,5), c(.1,.4,.2,.3), p)
    vals = np.array([3.0, 1.0, 2.0, 5.0])
    w = np.array([0.1, 0.4, 0.2, 0.3])
    p = np.array([0.05, 0.5, 0.9, 1.0])
    expected = np.array([1.0, 2.0, 5.0, 5.0])
    for method in ("orig", "findinterval_search"):
        assert np.allclose(lt.weighted_percentile(vals, w, p, method), expected)


def test_weighted_percentile_methods_agree_on_random_input():
    rng = np.random.default_rng(0)
    vals = rng.random(60) * 10
    w = rng.random(60)
    p = np.linspace(0, 1, 21)
    a = lt.weighted_percentile(vals, w, p, "orig")
    b = lt.weighted_percentile(vals, w, p, "findinterval_search")
    assert np.allclose(a, b)


def test_weighted_percentile_ignores_zero_weight_values():
    # A huge value carrying zero weight must not affect any percentile.
    vals = np.array([1.0, 2.0, 3.0, 1e9])
    w = np.array([0.3, 0.3, 0.4, 0.0])
    p = np.array([0.1, 0.5, 0.99])
    out = lt.weighted_percentile(vals, w, p, "findinterval_search")
    assert out.max() <= 3.0


def test_weighted_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        lt.weighted_percentile([1.0, 2.0], [0.5, 0.5], [1.5])
    with pytest.raises(ValueError):
        lt.weighted_percentile([1.0, 2.0], [-0.5, 0.5], [0.5])


def test_quantile_type6_matches_R():
    # Reference from R: quantile(c(1,2,2,3,5,5,8), c(.025,.5,.975), type=6)
    x = np.array([1.0, 2, 2, 3, 5, 5, 8])
    out = lt._r_quantile_type6(x, [0.025, 0.5, 0.975])
    assert np.allclose(out, np.array([1.0, 3.0, 8.0]))


# ---------------------------------------------------------------------------
# LEVEL 5 -- the copula
# ---------------------------------------------------------------------------

def _toy_branches(rates_matrix, probs):
    rates_matrix = np.atleast_2d(np.asarray(rates_matrix, float))
    return {
        "threshold_stages": np.arange(rates_matrix.shape[0], dtype=float),
        "logic_tree_branch_exceedance_rates": rates_matrix,
        "logic_tree_branch_posterior_prob": np.asarray(probs, float),
    }


def test_percentiles_unsegmented_only():
    """With no segments all the weight must sit on the unsegmented branch."""
    unseg = _toy_branches([[1.0, 2.0, 3.0]], [1 / 3, 1 / 3, 1 / 3])
    out = lt.compute_exceedance_rate_percentiles_with_random_sampling(
        unseg, [], N=20000, unsegmented_wt=1.0, union_of_segments_wt=0.0,
        rng=np.random.default_rng(1))
    # Mean of a uniform draw over {1,2,3} is 2.
    assert abs(out["mean_exrate"][0] - 2.0) < 0.05
    # Percentiles stay inside the support and are ordered.
    assert out["percentile_exrate"][:, 0].min() >= 1.0
    assert out["percentile_exrate"][:, 0].max() <= 3.0
    assert np.all(np.diff(out["percentile_exrate"][:, 0]) >= 0)


def test_comonotonic_is_wider_than_independent():
    """The copula changes the spread, not the mean.

    Comonotonic reuses one uniform draw across segments, so their
    uncertainties reinforce; independent draws partially cancel.
    """
    rng_rates = np.linspace(1.0, 5.0, 20)
    seg_a = _toy_branches([rng_rates], np.full(20, 1 / 20))
    seg_b = _toy_branches([rng_rates], np.full(20, 1 / 20))
    unseg = _toy_branches([rng_rates * 2], np.full(20, 1 / 20))

    res = {}
    for cop in ("comonotonic", "independent"):
        res[cop] = lt.compute_exceedance_rate_percentiles_with_random_sampling(
            unseg, [seg_a, seg_b], N=40000, segments_copula_type=cop,
            rng=np.random.default_rng(42))

    # Same mean, to Monte Carlo accuracy.
    assert abs(res["comonotonic"]["mean_exrate"][0]
               - res["independent"]["mean_exrate"][0]) < 0.05

    # Measure the spread on p0.16-p0.84 rather than the outermost pair: with
    # a bounded branch set the 2.5/97.5 percentiles saturate at the support
    # limits under both copulas, which would hide the effect.
    def spread(r):
        return r["percentile_exrate"][3, 0] - r["percentile_exrate"][1, 0]

    assert spread(res["comonotonic"]) > spread(res["independent"])


def test_percentiles_reject_inconsistent_thresholds():
    unseg = _toy_branches([[1.0, 2.0]], [0.5, 0.5])
    seg = dict(_toy_branches([[1.0, 2.0]], [0.5, 0.5]))
    seg["threshold_stages"] = np.array([99.0])
    with pytest.raises(ValueError):
        lt.compute_exceedance_rate_percentiles_with_random_sampling(
            unseg, [seg, seg], N=1000, rng=np.random.default_rng(0))


def test_percentiles_reject_single_segment():
    """PTHA18 never has exactly one segment, so it signals an input error."""
    unseg = _toy_branches([[1.0, 2.0]], [0.5, 0.5])
    seg = _toy_branches([[1.0, 2.0]], [0.5, 0.5])
    with pytest.raises(ValueError):
        lt.compute_exceedance_rate_percentiles_with_random_sampling(
            unseg, [seg], N=1000, rng=np.random.default_rng(0))


def test_percentiles_reject_segment_weight_without_segments():
    unseg = _toy_branches([[1.0, 2.0]], [0.5, 0.5])
    with pytest.raises(ValueError):
        lt.compute_exceedance_rate_percentiles_with_random_sampling(
            unseg, [], N=1000, unsegmented_wt=0.5, union_of_segments_wt=0.5,
            rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# LEVEL 3 -- magnitude observation error
# ---------------------------------------------------------------------------

def test_exceedance_rate_of_observed_negligible_error_is_a_no_op():
    """With a vanishingly small error the observed rate equals the true rate.

    This mirrors the check in the R function's own documentation example.
    """
    from pyptha_v12.rates import Mw_exceedance_rate_truncated_gutenberg_richter as tgr

    def exceedance_true(x):
        return tgr(x, a=6.0, b=1.0, Mw_min=-np.inf, Mw_max=9.4)

    def tiny_error(x, mw_true):
        return np.clip((np.asarray(x, float) + 1e-4) / 2e-4, 0.0, 1.0)

    data_value = 7.6
    got = lt.exceedance_rate_of_observed(
        exceedance_true, tiny_error, data_value,
        (data_value - 1e-3, data_value + 1e-3), integration_dy=1e-5)
    assert abs(got - float(exceedance_true(data_value))) < 1e-8


def test_exceedance_rate_of_observed_symmetric_error_raises_the_rate():
    """For a GR model, symmetric magnitude errors inflate the observed rate.

    There are many more small events than large ones, so 'small event plus
    positive error' crosses the threshold more often than 'large event plus
    negative error' falls below it.
    """
    from pyptha_v12.rates import Mw_exceedance_rate_truncated_gutenberg_richter as tgr

    def exceedance_true(x):
        return tgr(x, a=6.0, b=1.0, Mw_min=-np.inf, Mw_max=9.4)

    def uniform_error(x, mw_true, h=0.4):
        return np.clip((np.asarray(x, float) + h) / (2 * h), 0.0, 1.0)

    data_value = 7.6
    with_error = lt.exceedance_rate_of_observed(
        exceedance_true, uniform_error, data_value,
        (data_value - 0.4, data_value + 0.4), integration_dy=1e-4)
    assert with_error > float(exceedance_true(data_value))
