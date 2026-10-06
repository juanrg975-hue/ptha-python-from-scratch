"""Logic-tree levels that sit outside ``rates.rate_of_earthquakes_greater_than_Mw_function``.

``rates.py`` already covers LEVEL 1 (the parameter grid), LEVEL 2 (deriving the
GR 'a' by seismic-moment balance) and the common case of LEVEL 3 (the Bayesian
weight update). This module adds the remaining levels of the PTHA18 logic tree:

    LEVEL 0  segmentation      -> unsegmented vs union-of-segments weights
             ``get_unsegmented_and_segmented_source_names_on_source_zone``
    LEVEL 3  full Bayesian update, including the magnitude-observation-error
             extension and the censored-time (inter-event) likelihood
             ``exceedance_rate_of_observed``, ``compute_updated_logic_tree_weights``
    LEVEL 4  edge correction   -> inflate the conditional probability of
             ruptures touching the along-strike edge of the source zone
             ``edge_correct_conditional_probabilities``
    LEVEL 5  combining segments -> epistemic percentiles via sampling with a
             comonotonic (or independent) copula
             ``weighted_percentile``,
             ``compute_exceedance_rate_percentiles_with_random_sampling``

These are literal ports of the R originals:
    rptha/R/rptha/R/rupture_probabilities.R                      (LEVEL 3)
    rptha/R/rptha/R/make_stage_vs_rate_curve_fast.R              (weighted_percentile)
    R/examples/austptha_template/EVENT_RATES/back_calculate_convergence.R (LEVEL 4)
    rptha/ptha_access/get_PTHA_results.R                         (LEVEL 5)
    rptha/ptha_access/get_detailed_PTHA18_source_zone_info.R     (LEVEL 0)

Note on ``threshold_stages``: in PTHA18 the LEVEL 5 percentile machinery is
applied to tsunami wave-height thresholds. This port keeps that argument
generic -- it is simply the axis along which exceedance rates are tabulated,
supplied by the caller. Nothing here reads tsunami data. Passing Mw thresholds
gives a purely seismic logic tree.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# LEVEL 0 -- segmentation
# ---------------------------------------------------------------------------


def get_unsegmented_and_segmented_source_names_on_source_zone(
        source_zone, all_source_names):
    """Split a source-zone name list into the unsegmented branch and its segments.

    Port of the R function of the same name. PTHA18 gives 50:50 weight to
    unsegmented vs the union-of-segments when a zone has segments, and 100:0
    when it does not.

    ``all_source_names`` is the list of available source/segment names, where
    segments are named ``<source_zone>_<segment>``.
    """
    all_source_names = list(all_source_names)
    if all_source_names.count(source_zone) != 1:
        raise ValueError(
            f"source_zone {source_zone!r} must appear exactly once in "
            f"all_source_names")

    segments = [n for n in all_source_names
                if n.startswith(source_zone + "_")]

    if segments:
        unsegmented_weight = 0.5
        union_of_segments_weight = 0.5
    else:
        unsegmented_weight = 1.0
        union_of_segments_weight = 0.0

    return {
        "unsegmented_name": source_zone,
        "unsegmented_weight": unsegmented_weight,
        "segments": segments,
        "union_of_segments_weight": union_of_segments_weight,
        # Each segment carries the FULL union-of-segments weight, not a share
        # of it. The union-of-segments branch is a SUM over segments (they all
        # occur), not a choice between them, so dividing by the segment count
        # would be wrong. See Davies & Griffin (2018) Section 3.7.6: "The
        # 'union-of-segments' model has a mean Mw-exceedance-rate curve derived
        # by summing the mean curves on each segment".
        "per_segment_weight": union_of_segments_weight,
    }


# ---------------------------------------------------------------------------
# LEVEL 3 -- magnitude observation error
# ---------------------------------------------------------------------------


def exceedance_rate_of_observed(exceedance_true, cdf_obs_error, data_value,
                                true_value_range_with_nontrivial_cdf_value,
                                integration_dy=0.01):
    """Exceedance rate of observations that carry a random error.

    Port of the R ``exceedance_rate_of_observed``. If the true values follow
    ``exceedance_true(y)`` and the observation is ``y + error`` with
    ``cdf_obs_error(x, y)`` the CDF of the error given the true value, this
    returns the rate at which the OBSERVED value exceeds ``data_value``:

        integral { (1 - cdf_obs_error(data_value - y | y))
                   * (-d(exceedance_true)/dy) } dy

    computed by numerical integration over the supplied range, plus the rate of
    events above the upper limit (which can never be pushed below data_value).
    """
    data_value = float(np.asarray(data_value).reshape(()))

    lower, upper = true_value_range_with_nontrivial_cdf_value
    n = int(np.floor((upper - lower) / integration_dy)) + 1
    value_true = lower + np.arange(n) * integration_dy

    cdf_complement = 1.0 - np.asarray(
        cdf_obs_error(data_value - value_true, value_true), dtype=float)

    incremental_rate = -(
        np.asarray(exceedance_true(value_true + integration_dy / 2), dtype=float)
        - np.asarray(exceedance_true(value_true - integration_dy / 2), dtype=float))

    tail = float(np.asarray(
        exceedance_true(upper + integration_dy / 2), dtype=float).reshape(()))
    return float(np.sum(incremental_rate * cdf_complement) + tail)


# ---------------------------------------------------------------------------
# LEVEL 4 -- edge correction
# ---------------------------------------------------------------------------


def edge_correct_conditional_probabilities(event_Mw, event_conditional_prob,
                                           event_index_string,
                                           alongstrike_number,
                                           edge_multiplier):
    """Inflate the conditional probability of ruptures touching the zone edge.

    Port of the ``edge_multiplier`` branch of the R
    ``back_calculate_convergence``. With uniform slip, unit sources near the
    along-strike edge of a source zone are covered by fewer events, so the
    back-calculated slip sags at the edges. Inflating the rate of edge-touching
    ruptures compensates, improving seismic-moment conservation.

    An event is 'on the edge' when any of its unit sources sits at the minimum
    or maximum along-strike index of the source zone. Within each magnitude bin
    the corrected probabilities are renormalised, so the total rate in the bin
    is unchanged -- this redistributes rate towards the edges, it does not add
    any.

    ``event_index_string`` holds the '-'-separated 1-based unit-source indices
    of each event, as produced by ``events.get_all_earthquake_events``.
    Returns the corrected conditional probabilities.
    """
    if edge_multiplier < 0:
        raise ValueError("edge_multiplier must be >= 0")

    event_Mw = np.asarray(event_Mw, dtype=float)
    ecp = np.asarray(event_conditional_prob, dtype=float)
    alongstrike_number = np.asarray(alongstrike_number)

    if edge_multiplier == 0:
        return ecp.copy()

    edge_values = (alongstrike_number.min(), alongstrike_number.max())
    # event_index_string looks like "1-2-3-", with a trailing separator that
    # R's strsplit drops.
    is_on_edge = np.array([
        np.any(np.isin(alongstrike_number[
            np.array([int(v) for v in s.split("-") if v != ""]) - 1],
            edge_values))
        for s in event_index_string])

    out = np.zeros_like(ecp)
    for mw in np.unique(event_Mw):
        k = np.where(event_Mw == mw)[0]
        if np.all(ecp[k] == 0):
            # Impossible magnitude bin: keep the R code's dummy uniform value.
            out[k] = 1.0 / k.size
            continue
        new = ecp[k] * (1.0 + edge_multiplier * is_on_edge[k])
        out[k] = new / new.sum()
    return out


# ---------------------------------------------------------------------------
# LEVEL 5 -- percentiles across segments, with a copula
# ---------------------------------------------------------------------------


def weighted_percentile(vals, weights, p, method="findinterval_search"):
    """Weighted empirical percentile of ``vals``.

    Port of the R ``weighted_percentile``. Reads the weighted ECDF of ``vals``
    at each probability in ``p``. ``method='orig'`` mirrors the simple (slow)
    R path; ``'findinterval_search'`` mirrors the fast path used by the
    percentile routine below. Both give the same answer.
    """
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    p = np.atleast_1d(np.asarray(p, dtype=float))

    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must lie in [0, 1]")
    if np.any(weights < 0):
        raise ValueError("weights must be >= 0")

    weights = weights / weights.sum()

    if method == "orig":
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        cum = np.cumsum(weights[order])
        cum = cum / cum.max()
        ind = np.array([np.sum(cum < pi) for pi in p])
        return sorted_vals[np.minimum(ind, sorted_vals.size - 1)]

    elif method == "findinterval_search":
        k = np.where(weights > 0)[0]
        order = np.argsort(vals[k], kind="stable")
        sorted_vals = vals[k][order]
        cum = np.cumsum(weights[k][order])
        cum = cum / cum.max()
        # R's findInterval(left.open=TRUE) then +1 -> searchsorted 'left'.
        ind = np.searchsorted(cum, p, side="left")
        return sorted_vals[np.minimum(ind, sorted_vals.size - 1)]

    raise ValueError(f"unknown method {method!r}")


def _r_quantile_type6(x, probs):
    """R's ``quantile(..., type=6)``, which the R original uses."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    # type 6: p(k) = k/(n+1); h = (n+1)p
    h = (n + 1) * probs
    h = np.clip(h, 1.0, float(n))
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, n)
    frac = h - lo
    return x[lo - 1] + frac * (x[hi - 1] - x[lo - 1])


def compute_exceedance_rate_percentiles_with_random_sampling(
        unsegmented_branch_exrates,
        segmented_branch_exrates,
        N=40000,
        unsegmented_wt=0.5,
        union_of_segments_wt=0.5,
        segments_copula_type="comonotonic",
        percentile_probs=(0.025, 0.16, 0.5, 0.84, 0.975),
        numerical_probs=None,
        use_numerical_probs=True,
        rng=None):
    """Combine unsegmented + segmented branches into epistemic percentiles.

    Port of the R ``compute_exceedance_rate_percentiles_with_random_sampling``.
    This is where the copula lives: with ``segments_copula_type='comonotonic'``
    the SAME uniform random draw is reused across every segment, so their
    percentiles move together; with ``'independent'`` each segment gets its own
    draw.

    Each ``*_branch_exrates`` argument is a dict with:
        ``threshold_stages``                 (n_threshold,)
        ``logic_tree_branch_exceedance_rates`` (n_threshold, n_branch)
        ``logic_tree_branch_posterior_prob``   (n_branch,)
    ``segmented_branch_exrates`` is a list of such dicts, one per segment (empty
    if the zone is unsegmented).

    The exceedance rate of the union-of-segments is the SUM over segments,
    sampled at the shared (comonotonic) or per-segment (independent)
    percentile. The unsegmented and segmented samples are then pooled in the
    given weights, and the requested percentiles read off the pooled sample.

    ``threshold_stages`` is whatever axis the caller tabulated the rates on --
    PTHA18 uses tsunami wave height, but nothing here requires that.
    """
    if abs(unsegmented_wt + union_of_segments_wt - 1.0) >= 0.1 / N:
        raise ValueError("unsegmented_wt + union_of_segments_wt must be 1")

    if rng is None:
        rng = np.random.default_rng()

    if numerical_probs is None:
        numerical_probs = 0.5 * (1 + np.sin(np.linspace(-np.pi / 2, np.pi / 2, 1000)))
    numerical_probs = np.asarray(numerical_probs, dtype=float)

    segmented_branch_exrates = list(segmented_branch_exrates)
    Nseg = len(segmented_branch_exrates)

    thresholds = np.asarray(unsegmented_branch_exrates["threshold_stages"],
                            dtype=float)
    for seg in segmented_branch_exrates:
        if not np.array_equal(np.asarray(seg["threshold_stages"], dtype=float),
                              thresholds):
            raise ValueError("all segments must share the same threshold_stages")

    Nu = int(round(N * unsegmented_wt))
    Ns = N - Nu

    # The uniform draws whose inverse-CDF lookup generates the samples.
    random_unsegmented = rng.random(Nu)

    random_segments = []
    if Nseg > 1:
        first = rng.random(Ns)
        random_segments.append(first)
        for _ in range(1, Nseg):
            if segments_copula_type == "comonotonic":
                # Perfectly correlated percentiles across segments.
                random_segments.append(first)
            elif segments_copula_type == "independent":
                random_segments.append(rng.random(Ns))
            else:
                raise ValueError(f"unknown copula type {segments_copula_type!r}")
    elif Nseg == 1:
        raise ValueError(
            "Only one segment: this is not how PTHA18 works, so it suggests "
            "an input error")
    else:
        if union_of_segments_wt != 0:
            raise ValueError(
                "If no segments are provided then union_of_segments_wt should "
                "be zero")

    unseg_rates = np.asarray(
        unsegmented_branch_exrates["logic_tree_branch_exceedance_rates"],
        dtype=float)
    unseg_prob = np.asarray(
        unsegmented_branch_exrates["logic_tree_branch_posterior_prob"],
        dtype=float)

    Nst = thresholds.size
    percentile_probs = np.atleast_1d(np.asarray(percentile_probs, dtype=float))
    mean_exrate = np.full(Nst, np.nan)
    percentile_exrate = np.full((percentile_probs.size, Nst), np.nan)

    for i in range(Nst):
        # --- unsegmented branch ---
        if Nu > 0:
            if use_numerical_probs:
                quant = weighted_percentile(unseg_rates[i, :], unseg_prob,
                                            numerical_probs,
                                            method="findinterval_search")
                probs_axis = numerical_probs
            else:
                order = np.argsort(unseg_rates[i, :], kind="stable")
                quant = unseg_rates[i, :][order]
                probs_axis = np.cumsum(unseg_prob[order])
            random_unsegmented_exrates = np.interp(
                random_unsegmented, probs_axis, quant,
                left=quant[0], right=quant[-1])
        else:
            random_unsegmented_exrates = np.empty(0)

        # --- union of segments: sum the per-segment rates ---
        if Ns > 0:
            random_segmented_exrates = np.zeros(Ns)
            for j, seg in enumerate(segmented_branch_exrates):
                seg_rates = np.asarray(
                    seg["logic_tree_branch_exceedance_rates"], dtype=float)
                seg_prob = np.asarray(
                    seg["logic_tree_branch_posterior_prob"], dtype=float)
                if use_numerical_probs:
                    quant_s = weighted_percentile(seg_rates[i, :], seg_prob,
                                                  numerical_probs,
                                                  method="findinterval_search")
                    probs_axis_s = numerical_probs
                else:
                    order = np.argsort(seg_rates[i, :], kind="stable")
                    quant_s = seg_rates[i, :][order]
                    probs_axis_s = np.cumsum(seg_prob[order])
                # Identical random_segments[j] across j -> comonotonic;
                # independent draws -> independent solution.
                random_segmented_exrates = random_segmented_exrates + np.interp(
                    random_segments[j], probs_axis_s, quant_s,
                    left=quant_s[0], right=quant_s[-1])
        else:
            random_segmented_exrates = np.empty(0)

        samples = np.concatenate([random_unsegmented_exrates,
                                  random_segmented_exrates])
        mean_exrate[i] = samples.mean()
        percentile_exrate[:, i] = _r_quantile_type6(samples, percentile_probs)

    return {
        "threshold_stages": thresholds,
        "mean_exrate": mean_exrate,
        "percentile_probs": percentile_probs,
        "percentile_exrate": percentile_exrate,
    }


# ---------------------------------------------------------------------------
# Bridge: logic-tree branches -> the (threshold x branch) matrix LEVEL 5 wants
# ---------------------------------------------------------------------------


def branch_exceedance_rates_on_thresholds(rate_fn, threshold_stages):
    """Tabulate every logic-tree branch's exceedance rate at each threshold.

    Builds the ``(n_threshold, n_branch)`` matrix that LEVEL 5 consumes,
    straight from an :class:`~pyptha.rates.MwRateFunction`. Here the thresholds
    are magnitudes, which is what makes the pipeline runnable without any
    tsunami input.

    In PTHA18 the equivalent matrix is built by
    ``random_scenario_exceedance_rates_all_logic_tree_branches``, where the
    thresholds are tsunami wave heights and the branch rates are weighted by
    the conditional probability of exceeding that height given the magnitude
    bin. With Mw thresholds that conditional probability is just an indicator,
    so the matrix reduces to the branch exceedance-rate curves evaluated at the
    thresholds -- which is what this does.
    """
    branches = rate_fn(return_all_logic_tree_branches=True)
    thresholds = np.atleast_1d(np.asarray(threshold_stages, dtype=float))

    n_branch = branches.all_rate_matrix.shape[0]
    out = np.empty((thresholds.size, n_branch))
    for i in range(n_branch):
        out[:, i] = np.interp(thresholds, branches.Mw_seq,
                              branches.all_rate_matrix[i, :],
                              left=branches.all_rate_matrix[i, 0], right=0.0)
        # Beyond a branch's own Mw_max the rate is exactly zero.
        out[:, i] *= (thresholds <= branches.Mw_seq[-1])

    return {
        "threshold_stages": thresholds,
        "logic_tree_branch_exceedance_rates": out,
        "logic_tree_branch_posterior_prob": branches.all_par_prob,
    }


# ---------------------------------------------------------------------------
# v9 (LEVEL 0): scenario-rate percentiles on a partially segmented zone
# ---------------------------------------------------------------------------


def scenario_rate_percentiles_partial_segmentation(
        unseg_rate_fn, seg_rate_fns, event_Mw, event_cond_probs, dMw,
        unsegmented_weight, segmented_weight, desired_quantiles,
        percentile_discretization=0.0025, log=print):
    """Port of compute_rates_all_sources.R's
    update_scenario_rate_percentiles_on_source_zones_with_partial_segmentation
    (lines 998-1409), constant shear modulus.

    On a zone with an unsegmented model (weight w_u) and a union of segments
    (weight w_s, the segments co-monotonic, so the union's percentile p is the
    sum of the segments' p), the p-th percentile of the ZONE's exceedance rate
    is not w_u * (unsegmented p) + w_s * (union p). For each magnitude-bin edge
    this finds the percentile of each model whose weighted combination hits
    p, evaluates each source's exceedance curve there, and turns the curve
    back into per-scenario rates, exactly as R does.

    unseg_rate_fn, seg_rate_fns: MwRateFunction of the unsegmented model and
      of each segment. event_Mw: the zone's scenario magnitudes (shared by all
      sources). event_cond_probs: [unsegmented, segment 1, ...] conditional
      probabilities on that table.
    Returns a list, one entry per source (unsegmented first), of arrays
    (len(desired_quantiles), n_event): the rates R writes to its files
    before scaling each by its row_weight.
    """
    event_Mw = np.asarray(event_Mw, dtype=float)
    lower_mw = event_Mw.min() - dMw / 2.0
    upper_mw = event_Mw.max() + dMw / 2.0
    n_mw = int(round((upper_mw - lower_mw) / dMw + 1))
    mw_seq = np.linspace(lower_mw, upper_mw, n_mw)
    dp = 1.0 / round(1.0 / percentile_discretization)   # 1/dp is an integer
    n_p = int(round(1.0 / dp))
    p_store = np.linspace(dp / 2.0, 1.0 - dp / 2.0, n_p)

    # (n_mw, n_p) exceedance rate at each stored percentile
    unseg = np.asarray(unseg_rate_fn(mw_seq, quantiles=p_store)).T
    segs = [np.asarray(f(mw_seq, quantiles=p_store)).T for f in seg_rate_fns]
    seg_sum = np.sum(segs, axis=0)

    eps = 1.0e-8
    # index into p_store of the percentile to evaluate each model at, or -1
    # (R: NA or a percentile of 0 -> rate 0)
    idx_u = np.full((len(desired_quantiles), n_mw), -1, dtype=int)
    idx_s = np.full((len(desired_quantiles), n_mw), -1, dtype=int)
    k_all = np.arange(n_p)
    for i in range(n_mw):
        ru, rs = unseg[i], seg_sum[i]
        uniq = np.unique(np.concatenate([[0.0], ru, rs]))
        # R: pc_value_unseg[j] = max(percentiles_to_store * (rate_unseg <= u_j))
        ku = np.where(ru[None, :] <= uniq[:, None], k_all[None, :], -1).max(axis=1)
        ks = np.where(rs[None, :] <= uniq[:, None], k_all[None, :], -1).max(axis=1)
        pu = np.where(ku >= 0, p_store[np.maximum(ku, 0)], 0.0)
        ps = np.where(ks >= 0, p_store[np.maximum(ks, 0)], 0.0)
        pc = pu * unsegmented_weight + ps * segmented_weight
        for j, q in enumerate(desired_quantiles):
            t = int(np.sum(pc <= q + eps))
            if t > 1:
                idx_s[j, i] = ks[t - 1]
                idx_u[j, i] = ku[t - 1]

    n_adjusted = 0
    out = []
    curves = [unseg] + segs
    for s, curve in enumerate(curves):
        idx = idx_u if s == 0 else idx_s
        rates_q = np.zeros((len(desired_quantiles), event_Mw.size))
        for j in range(len(desired_quantiles)):
            ex = np.where(idx[j] >= 0, curve[np.arange(n_mw), np.maximum(idx[j], 0)], 0.0)
            # R 1260-1335: never let one source's curve increase with Mw
            for k in range(1, n_mw):
                if ex[k] > ex[k - 1]:
                    n_adjusted += 1
                    limit = ex[k - 2] if k > 1 else np.inf
                    if ex[k] <= limit:
                        ex[k - 1] = ex[k]
                    else:
                        ex[k] = ex[k - 1]
            ecp = np.asarray(event_cond_probs[s], dtype=float)
            for b in range(n_mw - 1):
                sel = (event_Mw > mw_seq[b]) & (event_Mw < mw_seq[b + 1])
                rates_q[j, sel] = ecp[sel] * (ex[b] - ex[b + 1])
        out.append(rates_q)
    if n_adjusted:
        log(f"      LEVEL 0 percentiles: {n_adjusted} non-monotonic step(s) "
            f"adjusted, as R does")
    return out
