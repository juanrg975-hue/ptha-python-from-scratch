"""Earthquake rate models: Gutenberg-Richter + logic tree.

Python port of the rate machinery in ``rptha/R/rupture_probabilities.R`` that
is used by ``single_source_rate_computation.R``:

    get_event_probabilities_conditional_on_Mw
    Mw_exceedance_rate_truncated_gutenberg_richter
    Mw_exceedance_rate_characteristic_gutenberg_richter
    rate_of_earthquakes_greater_than_Mw_function
        (with the logic tree over slip_rate x b x Mw_min x Mw_max x Mfd,
         seismic-moment balance to solve the GR 'a', optional Bayesian
         update of the branch weights from an observed event count)

The result is an ``MwRateFunction`` object: call it with an array of Mw to get
the logic-tree-mean exceedance rate, or ask for bounds / quantiles / all
branches, matching the optional arguments of the R function.

The Bayesian weight update (``compute_updated_logic_tree_weights``) is a full
port: Poisson count likelihood, the censored inter-event-time likelihood, the
magnitude-density likelihood, and the magnitude-observation-error extension.

The remaining logic-tree levels (segmentation, edge correction, and combining
segments into epistemic percentiles with a copula) live in ``logic_tree.py``.
"""

from __future__ import annotations

import itertools
import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.stats import expon, poisson

from .scaling import M0_2_Mw

# ---------------------------------------------------------------------------
# Conditional probabilities within a magnitude bin
# ---------------------------------------------------------------------------


def get_event_probabilities_conditional_on_Mw(event_table,
                                               conditional_probability_model="all_equal"):
    """Conditional probability of each event given its magnitude bin.

    ``event_table`` is a dict/record array (or pandas DataFrame) with at least
    a ``Mw`` field, and ``slip`` if ``conditional_probability_model`` is
    'inverse_slip'. A callable may be supplied taking the sub-table of a fixed
    Mw and returning its conditional probabilities.

    Returns a numpy array of length ``len(Mw)`` that sums to 1 within each Mw.
    """
    Mw = np.asarray(event_table["Mw"], dtype=float)

    if conditional_probability_model == "all_equal":
        def pfun(idx):
            n = idx.size
            return np.full(n, 1.0 / n)
    elif conditional_probability_model == "inverse_slip":
        slip = np.asarray(event_table["slip"], dtype=float)

        def pfun(idx):
            inv = 1.0 / slip[idx]
            return inv / inv.sum()
    elif callable(conditional_probability_model):
        user_fun = conditional_probability_model

        def pfun(idx):
            return np.asarray(user_fun(idx), dtype=float)
    else:
        raise ValueError("conditional_probability_model value not recognized")

    out = np.full(Mw.size, np.nan)
    for mw in np.unique(Mw):
        idx = np.where(Mw == mw)[0]
        p = pfun(idx)
        if not np.isclose(p.sum(), 1.0):
            raise ValueError(
                f"event conditional probabilities with Mw = {mw} do not "
                f"sum to 1.0 (got {p.sum()})")
        out[idx] = p
    if np.any(np.isnan(out)):
        raise ValueError("NA values among event_conditional_probabilities")
    return out


# ---------------------------------------------------------------------------
# Gutenberg-Richter exceedance-rate variants
# ---------------------------------------------------------------------------


def Mw_exceedance_rate_truncated_gutenberg_richter(Mw, a, b, Mw_min, Mw_max):
    """Truncated GR: rate of events with magnitude > Mw."""
    Mw = np.asarray(Mw, dtype=float)
    n = 10.0 ** (a - b * np.maximum(Mw, Mw_min)) - 10.0 ** (a - b * Mw_max)
    return n * (Mw <= Mw_max)


def Mw_exceedance_rate_characteristic_gutenberg_richter(Mw, a, b, Mw_min, Mw_max):
    """Characteristic GR: finite rate at exactly Mw_max."""
    Mw = np.asarray(Mw, dtype=float)
    n = 10.0 ** (a - b * np.maximum(Mw, Mw_min))
    return n * (Mw <= Mw_max)


_MFD = {
    "truncated_gutenberg_richter": Mw_exceedance_rate_truncated_gutenberg_richter,
    "characteristic_gutenberg_richter":
        Mw_exceedance_rate_characteristic_gutenberg_richter,
}


def compute_moment_fraction_from_events_greater_or_equal_than_mwmin(
        Mws, rate_Mws, moment_Mws, mwmin):
    """Fraction of seismic moment carried by events with Mw >= mwmin.

    ``rate_Mws`` are per-bin rates (not exceedance rates). ``Mws`` must be
    evenly spaced.
    """
    Mws = np.asarray(Mws, dtype=float)
    dMw = Mws[1] - Mws[0]
    if not np.allclose(np.diff(Mws), dMw):
        raise ValueError("Mws must be evenly spaced")
    if mwmin < Mws[0] or mwmin > Mws[-1]:
        raise ValueError("mwmin not compatible with Mws")
    high = Mws >= mwmin
    return np.sum(rate_Mws[high] * moment_Mws[high]) / np.sum(rate_Mws * moment_Mws)


# ---------------------------------------------------------------------------
# Logic-tree rate function
# ---------------------------------------------------------------------------


@dataclass
class LogicTreeBranches:
    """Everything needed to evaluate the logic tree (return_all_logic_tree_branches)."""

    all_par_combo: list      # list of dicts, one per branch
    a_parameter: np.ndarray   # GR 'a' per branch
    all_par_prob: np.ndarray  # posterior weight per branch (sums to 1)
    all_par_prob_prior: np.ndarray  # prior weight per branch
    Mw_seq: np.ndarray        # magnitudes at which rates are tabulated
    all_rate_matrix: np.ndarray  # (n_branch, len(Mw_seq)) exceedance rates
    # posterior weights accounting for Mw observation error (equal to
    # all_par_prob when no error model was supplied)
    all_par_prob_with_Mw_error: np.ndarray = None


class MwRateFunction:
    """Callable Mw-exceedance-rate model over a logic tree.

    Instances are returned by :func:`rate_of_earthquakes_greater_than_Mw_function`.
    Calling the instance with an array of Mw returns the logic-tree-mean
    exceedance rate. Keyword flags mirror the R output function:

        rate_fn(Mw)                          -> mean exceedance rate
        rate_fn(Mw, bounds=True)             -> dict of mean/upper/lower/median
        rate_fn(Mw, quantiles=[...])         -> array (len(q), len(Mw))
        rate_fn(return_all_logic_tree_branches=True) -> LogicTreeBranches
    """

    def __init__(self, branches: LogicTreeBranches):
        self._b = branches

    def _interp(self, values, Mw):
        # rule=2 in R: clamp to end values outside the range. Above the
        # largest Mw_max of the logic tree the R function multiplies every
        # output by (Mw <= max_Mw_max), i.e. the rate is exactly zero.
        Mw = np.asarray(Mw, dtype=float)
        out = np.interp(Mw, self._b.Mw_seq, values,
                        left=values[0], right=values[-1])
        return out * (Mw <= self._b.Mw_seq[-1])

    def __call__(self, Mw=None, bounds=False, quantiles=None,
                 return_all_logic_tree_branches=False,
                 epistemic_nonzero_weight=False):
        b = self._b
        if return_all_logic_tree_branches:
            return b

        mean_rates = b.all_par_prob @ b.all_rate_matrix

        if epistemic_nonzero_weight:
            w = b.all_par_prob @ (b.all_rate_matrix > 0)
            return self._interp(w, Mw)

        if bounds:
            upper = b.all_rate_matrix.max(axis=0)
            lower = b.all_rate_matrix.min(axis=0)
            median = self._quantile_curve(0.5)
            return {
                "rate": self._interp(mean_rates, Mw),
                "upper": self._interp(upper, Mw),
                "lower": self._interp(lower, Mw),
                "median": self._interp(median, Mw),
            }

        if quantiles is not None:
            q = np.atleast_1d(np.asarray(quantiles, dtype=float))
            curves = self._quantile_curves(q)
            out = np.array([self._interp(c, Mw) for c in curves])
            return out

        return self._interp(mean_rates, Mw)

    def with_mw_error(self):
        """v11: the same branch curves weighted with all_par_prob_with_Mw_error
        (R's ``account_for_mw_obs_error=TRUE``: mean, quantiles, median and
        epistemic_nonzero_weight use those weights; upper/lower are the
        max/min over branches either way). PTHA18's variable shear modulus
        rates come from this (compute_rates_all_sources.R 749-869)."""
        b = self._b
        w = b.all_par_prob_with_Mw_error
        if w is None:
            w = b.all_par_prob
        return MwRateFunction(dataclasses.replace(b, all_par_prob=w))

    def _quantile_curve(self, p):
        """Inverse-quantile of the branch rates at each Mw (weighted by branch prob)."""
        return self._quantile_curves(np.atleast_1d(p))[0]

    def _quantile_curves(self, q):
        """_quantile_curve for every p in q at once: each Mw column is sorted
        once for all of them (v9; the same numbers as one call per p, which
        re-sorted 32,000 branches per p and per Mw)."""
        b = self._b
        rates = b.all_rate_matrix          # (n_branch, n_mw)
        prob = b.all_par_prob
        n_mw = rates.shape[1]
        out = np.empty((q.size, n_mw))
        for j in range(n_mw):
            col = rates[:, j]
            order = np.argsort(col, kind="stable")
            cum = np.cumsum(prob[order])
            ind = np.minimum(np.searchsorted(cum, q, side="left"), len(order) - 1)
            out[:, j] = col[order][ind]
        return out


def individual_scenario_rates(event_Mw, event_conditional_probabilities,
                              rate_fn, dMw, quantiles=None):
    """Rate (events/year) of each individual earthquake scenario.

    Davies & Griffin (2018), Equation 3:

        r_j = Pr(j | Mw = Mw_j) * [ GR(Mw_j - dMw/2) - GR(Mw_j + dMw/2) ]

    i.e. the rate of the scenario's magnitude bin, split among the scenarios in
    that bin according to their conditional probability. Summing ``r_j`` over
    all scenarios above a magnitude reproduces the source-zone exceedance rate,
    which is what makes this the natural end product of the rate calculation.

    ``rate_fn`` is an :class:`MwRateFunction`. By default the logic-tree MEAN
    curve is used. Passing ``quantiles`` instead evaluates the rates at those
    percentiles of the logic tree, returning an array of shape
    ``(len(quantiles), n_event)``.

    Note the report's caveat (Section 3.7.7): an individual ``r_j`` at the 16th
    percentile may exceed the same scenario's rate at the 50th, because these
    are increments of percentile curves rather than percentiles of the
    increments. Summing over scenarios restores the expected ordering.
    """
    event_Mw = np.asarray(event_Mw, dtype=float)
    ecp = np.asarray(event_conditional_probabilities, dtype=float)
    if event_Mw.size != ecp.size:
        raise ValueError("event_Mw and conditional probabilities differ in length")

    lower = event_Mw - dMw / 2.0
    upper = event_Mw + dMw / 2.0

    if quantiles is None:
        return ecp * (np.asarray(rate_fn(lower), dtype=float)
                      - np.asarray(rate_fn(upper), dtype=float))

    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    lo = np.atleast_2d(rate_fn(lower, quantiles=q))
    hi = np.atleast_2d(rate_fn(upper, quantiles=q))
    return ecp[None, :] * (lo - hi)


def rate_of_earthquakes_greater_than_Mw_function(
        slip_rate, slip_rate_prob, b, b_prob, Mw_min, Mw_min_prob,
        Mw_max, Mw_max_prob, sourcezone_total_area, event_table,
        event_conditional_probabilities, computational_increment=0.01,
        Mw_frequency_distribution="truncated_gutenberg_richter",
        Mw_frequency_distribution_prob=1.0,
        update_logic_tree_weights_with_data=False,
        Mw_count_duration=(np.nan, np.nan, np.nan),
        Mw_2_M0=None, account_for_moment_below_mwmin=False,
        mw_max_posterior_equals_mw_max_prior=False,
        Mw_obs_data=None, mw_observation_error_cdf=None):
    """Build an :class:`MwRateFunction` for a source zone.

    Direct port of the R function of the same name, covering the logic tree
    over (slip_rate, b, Mw_min, Mw_max, Mw_frequency_distribution), the
    seismic-moment balance that fixes the GR 'a' parameter for each branch,
    and (optionally) a Poisson Bayesian update of the branch weights from an
    observed event count.

    Parameters follow the R names. Vector parameters carry per-value
    probabilities that must sum to 1. ``event_table`` must provide ``Mw``,
    ``slip`` and ``area`` (km^2) fields.
    """
    slip_rate = np.atleast_1d(np.asarray(slip_rate, dtype=float))
    slip_rate_prob = np.atleast_1d(np.asarray(slip_rate_prob, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    b_prob = np.atleast_1d(np.asarray(b_prob, dtype=float))
    Mw_min = np.atleast_1d(np.asarray(Mw_min, dtype=float))
    Mw_min_prob = np.atleast_1d(np.asarray(Mw_min_prob, dtype=float))
    Mw_max = np.atleast_1d(np.asarray(Mw_max, dtype=float))
    Mw_max_prob = np.atleast_1d(np.asarray(Mw_max_prob, dtype=float))
    Mfd_names = np.atleast_1d(np.asarray(Mw_frequency_distribution))
    Mfd_prob = np.atleast_1d(np.asarray(Mw_frequency_distribution_prob, dtype=float))

    if Mw_2_M0 is None:
        def Mw_2_M0(x):
            return M0_2_Mw(x, inverse=True)

    # Length + probability-sum checks (matching the R stopifnot()s).
    for v, p, name in ((slip_rate, slip_rate_prob, "slip_rate"),
                       (b, b_prob, "b"), (Mw_min, Mw_min_prob, "Mw_min"),
                       (Mw_max, Mw_max_prob, "Mw_max"),
                       (Mfd_names, Mfd_prob, "Mw_frequency_distribution")):
        if v.size != p.size:
            raise ValueError(f"{name} and its probabilities differ in length")
        if not np.isclose(p.sum(), 1.0):
            raise ValueError(f"{name} probabilities must sum to 1")
    if not (Mw_max.min() > Mw_min.max()):
        raise ValueError("min(Mw_max) must exceed max(Mw_min)")

    # Full logic tree: cartesian product of parameter values and weights.
    combos = list(itertools.product(
        range(slip_rate.size), range(b.size), range(Mw_min.size),
        range(Mw_max.size), range(Mfd_names.size)))
    all_par_combo = []
    all_par_prob = np.empty(len(combos))
    for k, (si, bi, mi, xi, fi) in enumerate(combos):
        all_par_combo.append({
            "slip_rate": float(slip_rate[si]), "b": float(b[bi]),
            "Mw_min": float(Mw_min[mi]), "Mw_max": float(Mw_max[xi]),
            "Mw_frequency_distribution": str(Mfd_names[fi]),
        })
        all_par_prob[k] = (slip_rate_prob[si] * b_prob[bi] * Mw_min_prob[mi]
                           * Mw_max_prob[xi] * Mfd_prob[fi])
    if not np.isclose(all_par_prob.sum(), 1.0):
        raise ValueError("logic-tree branch probabilities do not sum to 1")

    # Magnitude sequence at which each branch curve is tabulated.
    min_Mw_min = float(Mw_min.min())
    max_Mw_max = float(Mw_max.max())
    npts = round((max_Mw_max - min_Mw_min) / computational_increment) + 1
    Mw_seq = np.linspace(min_Mw_min, max_Mw_max, npts)

    eq_Mw = np.asarray(event_table["Mw"], dtype=float)
    eq_slip = np.asarray(event_table["slip"], dtype=float)
    eq_area = np.asarray(event_table["area"], dtype=float)
    ecp = np.asarray(event_conditional_probabilities, dtype=float)

    table_Mw_values = np.unique(eq_Mw)
    incs = np.diff(table_Mw_values)
    if not np.allclose(incs, incs[0]):
        raise ValueError("event table Mw values must be evenly spaced")
    table_Mw_increment = float(incs[0])

    # Conditional-probability sanity check per magnitude bin.
    for mw in table_Mw_values:
        ii = np.where(eq_Mw == mw)[0]
        if not np.isclose(ecp[ii].sum(), 1.0):
            raise ValueError(f"conditional probabilities for Mw={mw} do not sum to 1")

    lower_Mw = eq_Mw - table_Mw_increment / 2.0
    upper_Mw = eq_Mw + table_Mw_increment / 2.0

    all_rate_matrix = np.empty((len(all_par_combo), Mw_seq.size))
    a_parameter = np.empty(len(all_par_combo))

    # The cartesian product means every (b, Mw_min, Mw_max, Mfd) combination
    # recurs once per slip_rate value -- combos is built with slip_rate as the
    # SLOWEST-varying itertools.product index, i.e. n_slip_rate contiguous
    # blocks that each cycle through the same (b, Mw_min, Mw_max, Mfd)
    # sub-pattern. moment_fraction and RHS below depend only on that
    # sub-pattern (never on slip_rate), and 'a' shifts additively with
    # log10(slip_rate) for fixed RHS -- so each sub-pattern's expensive work
    # (the event-table reduction, and account_for_moment_below_mwmin's fine
    # Mw scan) is done ONCE per unique (b, Mw_min, Mw_max, Mfd) tuple instead
    # of once per branch, then broadcast across slip_rate for free.
    n_slip = slip_rate.size
    n_sub = len(combos) // n_slip
    sub_combos = combos[:n_sub]  # combos[0:n_sub] holds si=0, every (bi,mi,xi,fi) once

    for k, (_, bi, mi, xi, fi) in enumerate(sub_combos):
        par0 = all_par_combo[k]
        Mfd = _MFD[par0["Mw_frequency_distribution"]]
        bb, mwmin, mwmax = par0["b"], par0["Mw_min"], par0["Mw_max"]

        if account_for_moment_below_mwmin:
            lower = min(mwmin, lower_Mw.min(), 0.0) - 0.001
            upper = max(mwmax, upper_Mw.max()) + 0.001
            broad = np.arange(lower, upper + 1e-9, 0.001)
            dmw0 = broad[1] - broad[0]
            broad_rates = (Mfd(broad - dmw0 / 2, 0.0, bb, broad[0], mwmax)
                           - Mfd(broad + dmw0 / 2, 0.0, bb, broad[0], mwmax))
            broad_moment = np.asarray(Mw_2_M0(broad), dtype=float)
            mf = compute_moment_fraction_from_events_greater_or_equal_than_mwmin(
                broad, broad_rates, broad_moment, lower_Mw.min())
            mf -= compute_moment_fraction_from_events_greater_or_equal_than_mwmin(
                broad, broad_rates, broad_moment, upper_Mw.max())
            if not (0.0 <= mf <= 1.0):
                raise ValueError("moment_fraction outside [0, 1]")
            moment_fraction = mf
        else:
            moment_fraction = 1.0

        # Per-event rate assuming a=0, then back-out 'a' to match slip.
        # Independent of slip_rate, so computed once per sub-combo.
        rate_a0 = (Mfd(lower_Mw, 0.0, bb, mwmin, mwmax)
                   - Mfd(upper_Mw, 0.0, bb, mwmin, mwmax))
        RHS = np.sum(eq_slip * (eq_area * 1e6) * rate_a0 * ecp)

        # LHS of the seismic-moment balance (area in m^2), vectorised over
        # every slip_rate value that shares this sub-combo.
        LHS = (sourcezone_total_area * 1e6) * slip_rate * moment_fraction
        a_vals = np.log10(LHS / RHS)

        rows = k + n_sub * np.arange(n_slip)
        a_parameter[rows] = a_vals
        all_rate_matrix[rows, :] = Mfd(
            Mw_seq[None, :], a_vals[:, None], bb, mwmin, mwmax)
        for row, a_val in zip(rows, a_vals):
            all_par_combo[row]["a"] = float(a_val)

    all_par_prob_prior = all_par_prob.copy()

    if update_logic_tree_weights_with_data:
        all_par_prob = compute_updated_logic_tree_weights(
            all_par_combo, all_par_prob_prior, Mw_count_duration,
            Mw_obs_data=Mw_obs_data,
            mw_max_posterior_equals_mw_max_prior=mw_max_posterior_equals_mw_max_prior)

        if mw_observation_error_cdf is None:
            # No error model supplied, so the weights are unchanged.
            all_par_prob_with_Mw_error = all_par_prob
        else:
            all_par_prob_with_Mw_error = compute_updated_logic_tree_weights(
                all_par_combo, all_par_prob_prior, Mw_count_duration,
                Mw_obs_data=Mw_obs_data,
                mw_max_posterior_equals_mw_max_prior=mw_max_posterior_equals_mw_max_prior,
                cdf_mw_observation_error=mw_observation_error_cdf,
                integration_dy=computational_increment / 2)
    else:
        all_par_prob_with_Mw_error = all_par_prob

    branches = LogicTreeBranches(
        all_par_combo=all_par_combo, a_parameter=a_parameter,
        all_par_prob=all_par_prob, all_par_prob_prior=all_par_prob_prior,
        all_par_prob_with_Mw_error=all_par_prob_with_Mw_error,
        Mw_seq=Mw_seq, all_rate_matrix=all_rate_matrix)
    return MwRateFunction(branches)


def compute_updated_logic_tree_weights(
        all_par_combo, prior, Mw_count_duration,
        Mw_obs_data=None, mw_max_posterior_equals_mw_max_prior=False,
        cdf_mw_observation_error=None, max_obs_mw_error=0.5,
        integration_dy=0.005):
    """Bayesian update of the logic-tree weights from observed seismicity.

    Full port of the R ``compute_updated_logic_tree_weights`` (LEVEL 3 of the
    PTHA18 logic tree). The likelihood has two components:

    * a TEMPORAL one -- either a Poisson count of events above a threshold over
      a fixed duration (the default, and what PTHA18 recommends), or, when
      ``Mw_obs_data['t']`` is supplied, an exponential inter-event-time
      likelihood with the first and last intervals censored;
    * a MAGNITUDE one -- when ``Mw_obs_data['Mw']`` is supplied, the density of
      each observed magnitude under the branch's GR curve.

    Passing ``cdf_mw_observation_error`` (a function ``F(x, y)`` giving the CDF
    of the magnitude error ``x`` when the true magnitude is ``y``) accounts for
    measurement error in the observed magnitudes, via
    :func:`~pyptha.logic_tree.exceedance_rate_of_observed`.

    ``mw_max_posterior_equals_mw_max_prior`` forces the posterior marginal over
    Mw_max back onto its prior. That is not a pure Bayesian update; it exists
    so a user can hold the Mw_max uncertainty fixed regardless of the data.
    """
    from .logic_tree import exceedance_rate_of_observed

    if np.any(np.isnan(np.asarray(Mw_count_duration, dtype=float))):
        raise ValueError("Must provide Mw_count_duration when updating weights")
    data_thresh, data_count, duration = Mw_count_duration
    data_count = int(data_count)

    if cdf_mw_observation_error is not None:
        # v11: the error CDF is evaluated on the same (x, y) points for every
        # branch (only the GR curve changes), so remember each result: the
        # same numbers, without one empirical-CDF lookup per branch.
        _raw_cdf, _cdf_cache = cdf_mw_observation_error, {}

        def cdf_mw_observation_error(x, y):
            xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            key = (xa.shape, xa.tobytes(), ya.shape, ya.tobytes())
            if key not in _cdf_cache:
                _cdf_cache[key] = np.asarray(_raw_cdf(xa, ya), dtype=float)
            return _cdf_cache[key]

    if Mw_obs_data is None:
        Mw_obs_data = {}
    obs_t = Mw_obs_data.get("t")
    obs_Mw = Mw_obs_data.get("Mw")

    n_branch = len(all_par_combo)
    model_rates = np.empty(n_branch)

    for i, par in enumerate(all_par_combo):
        Mfd = _MFD[par["Mw_frequency_distribution"]]
        a_par, b_par, mwmax = par["a"], par["b"], par["Mw_max"]

        if cdf_mw_observation_error is None:
            # Mw_min = -inf so the exceedance rate at the threshold is unclamped.
            model_rates[i] = Mfd(data_thresh, a_par, b_par, -np.inf, mwmax)
        else:
            # Check the error model really has the assumed finite support.
            if cdf_mw_observation_error(-max_obs_mw_error,
                                        data_thresh + max_obs_mw_error) > 0:
                raise ValueError("mw_observation_error is too large")
            if cdf_mw_observation_error(max_obs_mw_error,
                                        data_thresh - max_obs_mw_error) < 1:
                raise ValueError("mw_observation_error is too large")

            def Mfd_local(y, _a=a_par, _b=b_par, _mx=mwmax, _f=Mfd):
                return _f(y, _a, _b, -np.inf, _mx)

            model_rates[i] = exceedance_rate_of_observed(
                Mfd_local, cdf_mw_observation_error, data_thresh,
                (data_thresh - max_obs_mw_error, data_thresh + max_obs_mw_error),
                integration_dy=integration_dy)

    # --- temporal component of the likelihood ---
    if obs_t is None:
        log_like = poisson.logpmf(data_count, model_rates * duration)
    else:
        # Censored ML is biased for a Poisson process; the R code warns here too.
        obs_t = np.asarray(obs_t, dtype=float)
        if np.min(np.diff(obs_t)) < 0:
            raise ValueError("Mw_obs_data['t'] must be sorted")
        if obs_t.min() < 0 or obs_t.max() > duration:
            raise ValueError("Mw_obs_data['t'] outside the observation window")

        dts = np.diff(obs_t)
        first_lower = obs_t[0]
        last_lower = duration - obs_t[data_count - 1]

        log_like = np.empty(n_branch)
        for i, ri in enumerate(model_rates):
            if ri <= 0:
                log_like[i] = -np.inf
                continue
            # Exponential spacings, with the first/last intervals only known
            # to exceed their observed lower bound (upper-tail probability).
            log_like[i] = (np.sum(expon.logpdf(dts, scale=1.0 / ri))
                           - ri * first_lower - ri * last_lower)

    # --- magnitude component of the likelihood ---
    if obs_Mw is not None:
        obs_Mw = np.atleast_1d(np.asarray(obs_Mw, dtype=float))
        if obs_Mw.size != data_count:
            raise ValueError("len(Mw_obs_data['Mw']) must equal the data count")
        if np.any(obs_Mw < data_thresh):
            raise ValueError("Mw_obs_data['Mw'] below the data threshold")
        if obs_t is not None and obs_Mw.size != np.asarray(obs_t).size:
            raise ValueError("Mw_obs_data 'Mw' and 't' differ in length")

        for i, par in enumerate(all_par_combo):
            Mfd = _MFD[par["Mw_frequency_distribution"]]
            a_par, b_par = par["a"], par["b"]
            mwmin, mwmax = par["Mw_min"], par["Mw_max"]

            # Density above the threshold is -(1/GR(thresh)) * dGR/dMw,
            # since the CDF is 1 - GR(Mw)/GR(thresh).
            if cdf_mw_observation_error is None:
                gr_thresh = Mfd(data_thresh, a_par, b_par, mwmin, mwmax)
                eps = 1e-4
                dens = -1.0 / (gr_thresh * 2 * eps) * (
                    Mfd(obs_Mw + eps, a_par, b_par, mwmin, mwmax)
                    - Mfd(obs_Mw - eps, a_par, b_par, mwmin, mwmax))
            else:
                def Mfd_local(y, _a=a_par, _b=b_par, _mx=mwmax, _f=Mfd):
                    return _f(y, _a, _b, -np.inf, _mx)

                gr_thresh = exceedance_rate_of_observed(
                    Mfd_local, cdf_mw_observation_error, data_thresh,
                    (data_thresh - max_obs_mw_error,
                     data_thresh + max_obs_mw_error),
                    integration_dy=integration_dy)

                eps = integration_dy
                rate_plus = np.array([exceedance_rate_of_observed(
                    Mfd_local, cdf_mw_observation_error, x,
                    (x - max_obs_mw_error, x + max_obs_mw_error),
                    integration_dy=integration_dy) for x in obs_Mw + eps])
                rate_minus = np.array([exceedance_rate_of_observed(
                    Mfd_local, cdf_mw_observation_error, x,
                    (x - max_obs_mw_error, x + max_obs_mw_error),
                    integration_dy=integration_dy) for x in obs_Mw - eps])
                dens = -1.0 / (2 * eps * gr_thresh) * (rate_plus - rate_minus)

            if np.any(dens < 0):
                raise ValueError("negative magnitude density in the likelihood")
            with np.errstate(divide="ignore"):
                log_like[i] = log_like[i] + np.sum(np.log(dens))

    if not np.any(np.isfinite(log_like)):
        raise ValueError("data is impossible under every model")

    # Rescale for numerical stability; this cancels in the normalisation.
    mx = np.nanmax(log_like[np.isfinite(log_like)])
    w = prior * np.exp(log_like - mx)
    post = w / w.sum()

    if mw_max_posterior_equals_mw_max_prior:
        mwmax_all = np.array([p["Mw_max"] for p in all_par_combo])
        if not np.isnan(data_thresh):
            if np.any((prior > 0) & (mwmax_all < data_thresh)):
                raise ValueError(
                    "Mw_max prior puts non-zero weight on a value below the "
                    "observations, so mw_max_posterior_equals_mw_max_prior "
                    "cannot be satisfied")
        # Re-normalise so the marginal over Mw_max equals the prior marginal.
        for mx_val in np.unique(mwmax_all):
            sel = mwmax_all == mx_val
            post_mass = post[sel].sum()
            prior_mass = prior[sel].sum()
            if post_mass > 0:
                post[sel] = post[sel] / post_mass * prior_mass
        post = post / post.sum()

    return post
