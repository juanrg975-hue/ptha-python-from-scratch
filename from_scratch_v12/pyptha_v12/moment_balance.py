"""Seismic-moment balance diagnostics and the PTHA18 coupling prior.

These are the pieces of the PTHA18 rate calculation that live in the
`austptha_template/EVENT_RATES` driver scripts rather than in the rptha package
itself, so they were missing from the earlier ports:

    back_calculate_convergence          - long-term slip rate integrated onto
        each unit source, i.e. sum_j (slip_j * rate_j) for every unit source
        touched by event j. This is the quantity PTHA18 plots in its Figures
        38, 39 and 45 to show that moment conservation holds spatially.
    fit_edge_multiplier                 - the numerical fit of the edge
        correction factor e_j (Davies & Griffin 2018, Section 3.7.1.2), which
        the report specifies as "determined numerically to give the best
        agreement" between integrated slip and tectonic convergence.
    coupling_prior_spreadsheet_and_uniform_50_50 - the 50/50 composite prior on
        seismic coupling (Section 3.7.2.3).
    interpolate_logic_tree_parameter    - the sub-sampling that turns a few
        anchor values into the fine parameter grids PTHA18 uses.

Sources, traced line by line:
    R/examples/austptha_template/EVENT_RATES/back_calculate_convergence.R
    R/examples/austptha_template/EVENT_RATES/compute_rates_all_sources.R
        (coupling prior ~lines 223-290; edge multiplier fit ~lines 643-735)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .events import get_unit_source_indices_in_event

__all__ = [
    "events_touching_alongstrike_edge",
    "back_calculate_convergence",
    "fit_edge_multiplier",
    "interpolate_logic_tree_parameter",
    "coupling_prior_spreadsheet_and_uniform_50_50",
    "mean_dip_cos_factor",
]


# ---------------------------------------------------------------------------
# Which events touch the along-strike edge of the source zone
# ---------------------------------------------------------------------------


def events_touching_alongstrike_edge(event_index_string, alongstrike_number):
    """Boolean flag per event: does it touch the extreme along-strike column?

    Port of the ``is_on_edge`` calculation in ``back_calculate_convergence.R``.
    An event is 'on the edge' when any of its unit sources sits at the minimum
    or maximum along-strike index of the source zone.
    """
    alongstrike_number = np.asarray(alongstrike_number)
    edge_values = (alongstrike_number.min(), alongstrike_number.max())
    return np.array([
        bool(np.any(np.isin(
            alongstrike_number[get_unit_source_indices_in_event(s)],
            edge_values)))
        for s in event_index_string])


# ---------------------------------------------------------------------------
# Long-term integrated slip rate per unit source
# ---------------------------------------------------------------------------


def back_calculate_convergence(event_rates, event_Mw, event_index_string,
                               event_slip, alongstrike_number,
                               n_unit_sources=None, edge_multiplier=0.0):
    """Integrate the long-term slip rate onto each unit source.

    Port of ``back_calculate_convergence.R`` for the uniform-slip case. For
    every event j with rate r_j and (uniform) slip S_j, every unit source it
    ruptures accumulates ``S_j * r_j``. Summed over all events this gives the
    modelled long-term slip rate on each unit source, which seismic-moment
    conservation says should follow the coupled tectonic convergence.

    With ``edge_multiplier > 0`` the conditional probability of edge-touching
    events is inflated by ``(1 + edge_multiplier)`` and renormalised WITHIN
    each magnitude bin, so the total rate per bin is unchanged -- rate is
    redistributed towards the edges, never created.

    Returns a dict with the new per-event rates and conditional probabilities,
    and ``integrated_slip`` (one value per unit source).
    """
    event_rates = np.asarray(event_rates, dtype=float)
    event_Mw = np.asarray(event_Mw, dtype=float)
    event_slip = np.asarray(event_slip, dtype=float)
    alongstrike_number = np.asarray(alongstrike_number)

    if n_unit_sources is None:
        n_unit_sources = alongstrike_number.size

    if edge_multiplier < 0:
        raise ValueError("edge_multiplier must be >= 0")

    indices = [get_unit_source_indices_in_event(s) for s in event_index_string]
    is_on_edge = np.array([
        bool(np.any(np.isin(alongstrike_number[ind],
                            (alongstrike_number.min(),
                             alongstrike_number.max()))))
        for ind in indices])

    new_event_rates = np.zeros_like(event_rates)
    new_conditional_probability = np.zeros_like(event_rates)

    for mw in np.unique(event_Mw):
        k = np.where(event_Mw == mw)[0]
        if np.all(event_rates[k] == 0):
            # Impossible magnitude bin: R stores a dummy uniform value.
            new_conditional_probability[k] = 1.0 / k.size
            continue
        if edge_multiplier != 0:
            inflated = event_rates[k] * (1.0 + edge_multiplier * is_on_edge[k])
            # Renormalise so the bin total is preserved exactly.
            inflated = inflated * event_rates[k].sum() / inflated.sum()
            new_event_rates[k] = inflated
        else:
            new_event_rates[k] = event_rates[k]
        new_conditional_probability[k] = (new_event_rates[k]
                                          / new_event_rates[k].sum())

    integrated_slip = np.zeros(n_unit_sources)
    for ind, slp, rt in zip(indices, event_slip, new_event_rates):
        integrated_slip[ind] += slp * rt

    return {
        "integrated_slip": integrated_slip,
        "new_event_rates": new_event_rates,
        "new_conditional_probability": new_conditional_probability,
        "is_on_edge": is_on_edge,
    }


# ---------------------------------------------------------------------------
# Numerical fit of the edge multiplier
# ---------------------------------------------------------------------------


def fit_edge_multiplier(event_rates, event_Mw, event_index_string, event_slip,
                        alongstrike_number, target_convergence,
                        n_unit_sources=None, lower=0.0, upper=30.0,
                        is_in_segment=None):
    """Find the edge multiplier that best reproduces the convergence pattern.

    This is LEVEL 4 as PTHA18 actually does it (Davies & Griffin 2018, Section
    3.7.1.2): rather than the user picking a number, the multiplier is fitted
    so the SHAPE of the modelled integrated slip matches the SHAPE of the
    tectonic convergence rate. Both sides are normalised to sum to one before
    comparison, because the seismic coupling rescales the amplitude but not the
    shape. The objective is the least-squares difference, minimised over
    [0, 30], exactly as in ``compute_rates_all_sources.R``:

        sum( ( model/sum(model) - target/sum(target) )^2 )

    ``target_convergence`` is the desired convergence rate on each unit source
    (any units: only its shape matters).

    Returns a dict with the fitted multiplier, the objective at that point, and
    the resulting conditional probabilities.
    """
    target = np.asarray(target_convergence, dtype=float)
    if np.all(target == 0):
        raise ValueError("target_convergence is identically zero")

    if is_in_segment is None:
        is_in_segment = np.ones_like(target)
    is_in_segment = np.asarray(is_in_segment, dtype=float)

    target_shape = target / target.sum()

    def objective(edge_multiplier):
        env = back_calculate_convergence(
            event_rates, event_Mw, event_index_string, event_slip,
            alongstrike_number, n_unit_sources=n_unit_sources,
            edge_multiplier=float(edge_multiplier))
        model = env["integrated_slip"]
        if model.sum() <= 0:
            return np.inf
        return float(np.sum(is_in_segment
                            * (model / model.sum() - target_shape) ** 2))

    # R weeds out the degenerate case where the multiplier changes nothing,
    # always against a fixed 0.0 baseline (compute_rates_all_sources.R:692,
    # "f1 = fun_to_optimize(0.0)"), not against `lower`. Every caller today
    # passes lower=0.0, so this was previously a no-op difference -- fixed
    # here so it stays correct if a caller ever passes lower>0.
    f_lo, f_hi = objective(0.0), objective(10.0)
    if np.isclose(f_lo, f_hi):
        raise ValueError(
            "the edge multiplier has no impact on this source: every "
            "edge-touching event must have zero prior weight")

    # R: optimize(fun_to_optimize, lower=0, upper=30) (compute_rates_all_
    # sources.R:704), Brent's fmin with R's default tol = eps^0.25. scipy's
    # bounded method is the same algorithm, so with R's tolerance (v8; scipy's
    # own default is 1e-5) it lands on R's multiplier: per-event rates then
    # match PTHA18's published ones to ~1e-12 instead of ~1e-6.
    opt = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                          options={"xatol": np.finfo(float).eps ** 0.25})
    best = float(opt.x)

    if best < 1.0e-3 or best > upper - 1.0e-3:
        # R prints a warning rather than failing; keep that behaviour but make
        # it visible to the caller.
        hit_bound = True
    else:
        hit_bound = False

    env = back_calculate_convergence(
        event_rates, event_Mw, event_index_string, event_slip,
        alongstrike_number, n_unit_sources=n_unit_sources,
        edge_multiplier=best)

    return {
        "edge_multiplier": best,
        "objective": float(opt.fun),
        "objective_at_zero": f_lo,
        "hit_bound": hit_bound,
        "conditional_probability": env["new_conditional_probability"],
        "integrated_slip": env["integrated_slip"],
        "is_on_edge": env["is_on_edge"],
    }


# ---------------------------------------------------------------------------
# Logic-tree parameter grids
# ---------------------------------------------------------------------------


def interpolate_logic_tree_parameter(anchor_values, n, log_spacing=False):
    """Expand a few anchor values into an ``n``-value logic-tree axis.

    Port of R's ``approx(anchor_values, n=n)$y``, which PTHA18 uses to turn the
    handful of tabulated parameter values into the fine grids it samples (20
    b-values, 20 coupling values, 40 Mw_max values). With ``log_spacing=True``
    the interpolation happens in log space, which is what the coupling prior
    does so that small coupling values are resolved just as finely as large
    ones in a relative sense.
    """
    anchor = np.asarray(anchor_values, dtype=float)
    if anchor.size < 2:
        raise ValueError("need at least two anchor values")
    if n < anchor.size:
        raise ValueError("n must be at least the number of anchor values")

    if log_spacing:
        if np.any(anchor <= 0):
            raise ValueError("log spacing needs strictly positive values")
        return np.exp(np.interp(np.linspace(0, 1, n),
                                np.linspace(0, 1, anchor.size),
                                np.log(anchor)))

    return np.interp(np.linspace(0, 1, n),
                     np.linspace(0, 1, anchor.size), anchor)


def coupling_prior_spreadsheet_and_uniform_50_50(
        uniform_range, spreadsheet_values, n, prob_zero_coupling=0.0):
    """The PTHA18 composite prior on seismic coupling.

    Davies & Griffin (2018), Section 3.7.2.3: half the prior density comes from
    the Global Earthquake Model values (Berryman et al. 2015), whose
    lower/preferred/upper triple is read as the 0/50/100 points of a CDF; the
    other half is uniform over a wider range (PTHA18 uses [0.1, 1.3], the upper
    limit exceeding 1.0 to absorb possible under-estimation of the tectonic
    convergence rate).

    The two CDFs are averaged, coupling values are laid out with logarithmic
    spacing over the combined range, and the per-value probability is the
    numerical derivative of the averaged CDF (a centred difference between
    successive midpoints), exactly as in ``compute_rates_all_sources.R``.

    ``prob_zero_coupling`` optionally prepends an aseismic branch, matching the
    ``prob_Mmax_below_Mmin`` treatment in the same script.

    Returns ``(coupling_values, coupling_probabilities)``.
    """
    pr1 = np.asarray(uniform_range, dtype=float)
    pr2 = np.asarray(spreadsheet_values, dtype=float)
    if np.any(pr1 <= 0) or np.any(pr2 <= 0):
        raise ValueError("coupling values must be strictly positive")

    def make_ecdf(vals):
        # R's approxfun(vals, seq(0, 1, len=length(vals)), rule=2): linear
        # between the anchors, flat outside them.
        vals = np.asarray(vals, dtype=float)
        probs = np.linspace(0.0, 1.0, vals.size)

        def ecdf(x):
            return np.interp(np.asarray(x, dtype=float), vals, probs,
                             left=probs[0], right=probs[-1])
        return ecdf

    uniform_ecdf = make_ecdf(pr1)
    spreadsheet_ecdf = make_ecdf(pr2)

    def final_ecdf(x):
        return 0.5 * (uniform_ecdf(x) + spreadsheet_ecdf(x))

    combined = np.concatenate([pr1, pr2])
    range_c = np.array([combined.min(), combined.max()])
    coupling_vals = np.exp(np.linspace(np.log(range_c[0]),
                                       np.log(range_c[1]), n))

    ll = coupling_vals.size
    # Midpoints forward and backward, with the end intervals mirrored, so the
    # numerical derivative covers the whole support.
    fwd_partner = np.concatenate([coupling_vals[1:],
                                  [2 * coupling_vals[ll - 1]
                                   - coupling_vals[ll - 2]]])
    bwd_partner = np.concatenate([[2 * coupling_vals[0] - coupling_vals[1]],
                                  coupling_vals[:ll - 1]])
    coupling_forward = 0.5 * (coupling_vals + fwd_partner)
    coupling_backward = 0.5 * (coupling_vals + bwd_partner)

    coupling_p = final_ecdf(coupling_forward) - final_ecdf(coupling_backward)
    if coupling_p.sum() <= 0:
        raise ValueError("coupling prior has zero total mass")
    coupling_p = coupling_p / coupling_p.sum()

    if prob_zero_coupling > 0:
        if prob_zero_coupling > 1:
            raise ValueError("prob_zero_coupling must be in (0, 1]")
        coupling_vals = np.concatenate([[0.0], coupling_vals])
        coupling_p = np.concatenate([[prob_zero_coupling],
                                     coupling_p * (1 - prob_zero_coupling)])

    return coupling_vals, coupling_p


# ---------------------------------------------------------------------------
# Dip correction on the tectonic convergence rate
# ---------------------------------------------------------------------------


def mean_dip_cos_factor(dip_degrees, weights=None):
    """``1 / cos(mean dip)``, the factor relating convergence to fault slip.

    Tectonic convergence rates come from plate models as HORIZONTAL velocities,
    whereas earthquake slip lies in the fault plane. PTHA18 divides the
    convergence rate by ``cos(mean_dip)`` before the moment balance
    (``compute_rates_all_sources.R``: ``sourcepar$slip = sourcepar$slip/cos_dip
    * 1/1000``). At shallow dips the correction is negligible, but it reaches
    15% at 30 degrees and 41% at 45 degrees, so it matters on steep zones such
    as Puysegur.

    The mean dip is an angle average (via the mean unit vector), matching
    rptha's ``mean_angle``. ``weights`` optionally weights unit sources, which
    is how PTHA18 restricts the average to a segment.
    """
    dip = np.radians(np.asarray(dip_degrees, dtype=float))
    if weights is None:
        weights = np.ones_like(dip)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("dip weights sum to zero")

    mean_dip = np.arctan2(np.sum(weights * np.sin(dip)) / weights.sum(),
                          np.sum(weights * np.cos(dip)) / weights.sum())
    cos_dip = np.cos(mean_dip)
    if cos_dip <= 0:
        raise ValueError("mean dip >= 90 degrees")
    return float(1.0 / cos_dip), float(np.degrees(mean_dip))


def mw_max_anchor(area_km2, alongstrike_number, width_km,
                  mw_max_observed, scaling_relation="Strasser",
                  mw_observed_perturbation=0.05,
                  minimum_allowed_mw_max=7.35,
                  maximum_allowed_mw_max=9.6):
    """The ``Mw_max`` axis endpoints, derived the way PTHA18 derives them.

    ``compute_rates_all_sources.R`` does not take these as input; it computes
    them from the source geometry, so typing them into a config by hand is how
    a run silently stops matching the official tree.

    The lower anchor is the largest observed magnitude plus a small
    perturbation, floored at ``minimum_allowed_mw_max``. The upper anchor is
    the SMALLER of two scaling-relation bounds:

    * area at -1 standard deviation, via ``Mw_2_rupture_size_inverse``;
    * the magnitude at which the scaling width at -2 standard deviations equals
      the zone's mean along-strike-summed width.

    The width bound exists because PTHA18 simulates rupture width within +-2 SD.
    Without it, a zone only one or two unit sources deep admits magnitudes whose
    width must be truncated, producing implausibly large mean slip. The result
    is clipped at ``maximum_allowed_mw_max`` after interpolation, not here.

    Parameters
    ----------
    area_km2 : float
        Total source (or segment) area, matching ``sourcepar$area_in_segment``.
    alongstrike_number, width_km : array_like
        Per-unit-source along-strike column index and down-dip width. Widths are
        summed within each column, then averaged over columns.
    mw_max_observed : float
        Largest observed magnitude, from ``sourcezone_parameters.csv``.

    Returns
    -------
    (lower, upper) : tuple of float
        The two anchor values, ready for ``interpolate_logic_tree_parameter``.
    """
    from scipy.optimize import brentq

    from .scaling import Mw_2_rupture_size, Mw_2_rupture_size_inverse

    lower = max(float(mw_max_observed) + float(mw_observed_perturbation),
                float(minimum_allowed_mw_max))

    upper_area = float(Mw_2_rupture_size_inverse(
        float(area_km2), relation=scaling_relation, CI_sd=-1))

    col = np.asarray(alongstrike_number)
    wid = np.asarray(width_km, dtype=float)
    per_column = np.array([wid[col == c].sum() for c in np.unique(col)])
    mean_width = float(per_column.mean())

    def width_residual(mw):
        d = Mw_2_rupture_size(mw, relation=scaling_relation, detailed=True,
                              CI_sd=2)
        return float(d.minus_CI["width"]) - mean_width

    # rptha searches Mw in [2, 20]; the root is unique over that interval.
    upper_width = float(brentq(width_residual, 2.0, 20.0, xtol=1e-12))

    upper = min(upper_area, upper_width)
    if upper <= lower:
        raise ValueError(
            f"scaling-relation Mw_max ({upper:.4f}) is below the observed "
            f"floor ({lower:.4f}); the geometry cannot host the observed "
            f"largest earthquake")
    # The UNCLIPPED anchor is returned deliberately. compute_rates_all_sources.R
    # interpolates the 40 slots between the raw anchors and only then applies
    # pmin(..., MAXIMUM_ALLOWED_MW_MAX), under its own comment "Clip AFTER
    # interpolation" (line 358). The order matters whenever the raw upper anchor
    # exceeds the cap: clipping first spreads 40 distinct values evenly up to
    # 9.6, while clipping second bunches the top slots ONTO 9.6, so the axis
    # holds fewer distinct values and carries more weight at the cap. On
    # kurilsjapan the official axis has 22 distinct values out of 40, and
    # southamerica 18; clipping first gives 40 in both cases and shifts the
    # prior-weighted mean Mw_max by about 1%. The caller applies the cap.
    return lower, upper
