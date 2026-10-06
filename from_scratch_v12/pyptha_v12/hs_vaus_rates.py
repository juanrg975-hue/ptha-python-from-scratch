"""Rates of HS and VAUS scenarios from the rates of their FAUS parents.

Port of rptha's ``EVENT_RATES/compute_rates_all_sources.R`` lines 1551-1600
(PTHA18, fixed shear modulus). Each HS or VAUS scenario is drawn from one
FAUS rupture (its "parent", ``uniform_event_row`` in PTHA18's netCDF), and
the parent's rate is shared among the scenarios drawn from it:

1. A scenario whose peak slip is above ``PEAK_SLIP_LIMIT_FACTOR`` (7.5)
   times the scaling relation's mean slip for the parent's Mw
   (``slip_from_Mw``: the slip on the relation's median area) gets weight 0
   (rptha ``config_peak_slip_limit_factor.R``).
2. The others are ranked by peak slip within the family,
   q = rank / (n + 1) (ties in order of appearance, R's ``ties.method =
   'first'``), and weighted by f(q), a function PTHA18 fitted to DART buoy
   observations (``peak_slip_quantile_adjustment_factors.csv``, written by
   ``event_properties_and_GOF.R``): one for HS ("stochastic"), one for VAUS
   ("variable_uniform"), linear between 51 knots q = 0, 0.02, ..., 1.
3. The weights of a family are divided by their sum, so the family's
   scenarios together carry exactly the parent's rate (or nothing, if every
   one is above the limit).

The table itself is not published. ``data/ptha18_peak_slip_quantile_weights
.csv`` was recovered from PTHA18's published HS/VAUS rates
(``validation/recover_peak_slip_weights.py``): fitted on puysegur2 alone it
reproduces every kermadectonga2 HS and VAUS rate to 1e-14 (relative), and
the other way round. The function is only known up to a constant factor,
which step 3 cancels.

v11: PTHA18 also publishes rates for a depth-varying shear modulus
(``variable_mu_rate_annual``): the same rule on the variable shear modulus
FAUS rates, with a second pair of DART curves (PTHA18's
peak_slip_quantile_adjustment_factors_varyMu.csv, recovered the same way;
``variable_mu=True``). The peak-slip limit is the same in both cases
(compute_rates_all_sources.R 1645-1646).
"""

from __future__ import annotations

import csv
import os

import numpy as np

from .scaling import slip_from_Mw

PEAK_SLIP_LIMIT_FACTOR = 7.5

DEFAULT_WEIGHTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "ptha18_peak_slip_quantile_weights.csv")

_COLUMN = {"HS": "weight_stochastic", "VAUS": "weight_variable_uniform",
           "HS_variable_mu": "weight_stochastic_variable_mu",
           "VAUS_variable_mu": "weight_variable_uniform_variable_mu"}


def load_peak_slip_weights(path=DEFAULT_WEIGHTS_CSV):
    """{"HS": (q, f), "VAUS": (q, f), and v11 "HS_variable_mu",
    "VAUS_variable_mu" when the file has them} from the weights file."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    q = np.array([float(r["peak_slip_quantile"]) for r in rows])
    return {kind: (q, np.array([float(r[col]) for r in rows]))
            for kind, col in _COLUMN.items() if col in rows[0]}


def family_weights(parent, peak_slip, parent_Mw, kind, mu=3.0e10,
                   relation="Strasser", weights=None,
                   limit_factor=PEAK_SLIP_LIMIT_FACTOR, variable_mu=False):
    """Share of its parent's rate carried by each HS or VAUS scenario.

    ``parent``: the parent FAUS rupture of each scenario (any hashable id),
    ``peak_slip``: each scenario's peak slip (m; for VAUS its uniform slip),
    ``parent_Mw``: the parent's Mw (the same for a whole family), all in the
    order the scenarios were generated (that order breaks ties). ``kind`` is
    "HS" or "VAUS". Returns ``(weight, above_limit)``: weights sum to 1 over
    each family that has a scenario under the limit, and are 0 above it.
    v11 ``variable_mu=True``: PTHA18's curves for the variable shear modulus
    rates (the limit does not change).
    """
    if kind not in ("HS", "VAUS"):
        raise ValueError(f"kind must be 'HS' or 'VAUS', not {kind!r}")
    q_knots, f_knots = (weights or load_peak_slip_weights())[
        kind + ("_variable_mu" if variable_mu else "")]
    parent = np.asarray(parent)
    peak_slip = np.asarray(peak_slip, dtype=float)
    parent_Mw = np.asarray(parent_Mw, dtype=float)

    # the limit depends only on Mw: one scaling-relation call per magnitude
    mws, inv = np.unique(np.round(parent_Mw, 6), return_inverse=True)
    mean_slip = np.atleast_1d(slip_from_Mw(mws, mu=mu, relation=relation))
    above = peak_slip > limit_factor * mean_slip[inv]

    weight = np.zeros(peak_slip.size)
    _, fam = np.unique(parent, return_inverse=True)
    for k in np.unique(fam):
        ok = np.where((fam == k) & ~above)[0]  # generation order
        if ok.size == 0:
            continue
        rank = np.empty(ok.size)
        rank[np.argsort(peak_slip[ok], kind="stable")] = np.arange(1, ok.size + 1)
        f = np.interp(rank / (ok.size + 1), q_knots, f_knots)
        if f.sum() > 0:
            weight[ok] = f / f.sum()
    return weight, above
