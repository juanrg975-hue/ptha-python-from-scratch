"""Variable shear modulus (rigidity that grows with depth), PTHA18's way.

Port of the rptha / PTHA18 pieces that turn the constant-rigidity scenarios
into the "variable shear modulus" ones (Davies & Griffin 2018, PTHA18 report
Section 3.7.5):

    shear_modulus_depth      rptha ``unit_sources.R`` 1702: the rigidity curve,
                             an approximate fit to Bilek & Lay (1999): 10 GPa
                             down to 7.5 km, 30 GPa at 15 km, 67 GPa from
                             35 km, log-linear in between ("prem" is PREM).
    variable_mu_Mw           ``append_variable_mu_variables_to_event_netcdf.R``
                             90-114: a scenario's magnitude when each unit
                             source's moment uses the rigidity at its depth,
                             M0 = sum(area * slip * mu(depth)). The slip is
                             the scenario's own (computed with the constant
                             rigidity): PTHA18 keeps the scenarios and
                             "relabels" their magnitude.
    make_conditional_ecdf    rptha ``rupture_probabilities.R`` 1481: the
                             distribution of (variable - constant) magnitude
                             given the constant magnitude, from many
                             scenarios (PTHA18 uses its HS scenarios,
                             ``compute_rates_all_sources.R`` 476-516).

How PTHA18 uses them: rates stay functions of the CONSTANT-rigidity
magnitude, but the earthquakes in the catalogue (GCMT) have the real,
variable-rigidity magnitude, so LEVEL 3 treats the difference as an
observation error with that distribution (``rates.py``,
``all_par_prob_with_Mw_error``). The "variable_mu" rates are the same branch
curves averaged with those weights (``MwRateFunction.with_mw_error``).
Normal-fault zones (rake -90) keep the constant rigidity in PTHA18.
"""

from __future__ import annotations

import numpy as np

from .scaling import M0_2_Mw

# rptha shear_modulus_depth(type='default'), Pa
_DEPTHS_KM = np.array([0.0, 7.5, 15.0, 35.0, 9999.0])
_MU_PA = np.array([10.0, 10.0, 30.0, 67.0, 67.0]) * 1e9

# PREM, Dziewonski & Anderson (1981) Table 4 (rptha type='prem')
_PREM_DEPTHS = np.array([0, 3, 3.0001, 15, 15.0001, 24.4, 24.4001, 40, 60, 80, 80.001, 100.0])
_PREM_VS = np.array([0, 0, 3.191, 3.191, 3.889, 3.889, 4.438, 4.472, 4.464, 4.457, 4.376, 4.369])
_PREM_RHO = np.array([1.02, 1.02, 2.6, 2.6, 2.9, 2.9, 3.38, 3.38, 3.38, 3.37, 3.37, 3.37])


def shear_modulus_depth(depth_km, kind="default"):
    """Rigidity (Pa) at ``depth_km``, interpolated in log10 as rptha does.

    Like R's ``approx`` (rule=1) a depth outside the curve gives NaN; the
    default curve runs from 0 to 9999 km.
    """
    depth_km = np.asarray(depth_km, dtype=float)
    if kind == "default":
        d, mu = _DEPTHS_KM, _MU_PA
    elif kind == "prem":
        d, mu = _PREM_DEPTHS, (_PREM_VS * 1e3) ** 2 * _PREM_RHO * 1e3
    else:
        raise ValueError(f"shear modulus curve {kind!r} not recognised")
    with np.errstate(divide="ignore"):
        logmu = np.log10(mu)
    out = 10.0 ** np.interp(depth_km, d, logmu, left=np.nan, right=np.nan)
    return out


def parse_index_string(s):
    """'3-8-13-' -> array([3, 8, 13]) (1-based unit-source numbers)"""
    return np.array([int(v) for v in str(s).split("-") if v.strip()], dtype=int)


def parse_slip_string(s):
    """'1.2_0.5_' -> array([1.2, 0.5])"""
    return np.array([float(v) for v in str(s).split("_") if v.strip()], dtype=float)


def variable_mu_Mw(index_strings, slips, area_km2, depth_km, kind="default"):
    """Magnitude of each scenario with the depth-varying rigidity.

    ``index_strings``: each scenario's "-"-joined 1-based unit-source numbers
    (``event_index_string``); ``slips``: each scenario's "_"-joined slips
    (``event_slip_string``, HS/VAUS) or one uniform slip value (FAUS);
    ``area_km2`` / ``depth_km``: per unit source, indexed by unit-source
    number - 1 (the unit-source statistics table). Returns an array of Mw.
    """
    mu = shear_modulus_depth(depth_km, kind) * np.asarray(area_km2, dtype=float) * 1e6
    out = np.empty(len(index_strings))
    for i, (eis, s) in enumerate(zip(index_strings, slips)):
        idx = parse_index_string(eis) - 1
        slip = parse_slip_string(s) if isinstance(s, str) else float(s)
        out[i] = M0_2_Mw(np.sum(mu[idx] * slip))
    return out


def make_conditional_ecdf(x, conditional_var):
    """Port of rptha ``make_conditional_ecdf``: F(xs, cond) = the empirical
    CDF of ``x`` among the samples whose ``conditional_var`` equals each of
    the two binned values around ``cond``, linearly interpolated between
    them (clamped at the ends). The binned values must be evenly spaced.
    """
    x = np.asarray(x, dtype=float)
    cv = np.asarray(conditional_var, dtype=float)
    values = np.unique(cv)
    if values.size > 1:
        spacing = np.diff(values)
        if not np.allclose(spacing, spacing[0], rtol=0, atol=1e-8 * max(1.0, abs(spacing[0]))):
            raise ValueError("unique conditional_var values are not evenly spaced, "
                             "which make_conditional_ecdf requires")
        spacing = float(spacing[0])
    else:
        spacing = 1.0
    sorted_x = [np.sort(x[cv == v]) for v in values]
    n_val = values.size

    def cdf(xs, cond):
        xs = np.asarray(xs, dtype=float)
        cond = np.asarray(cond, dtype=float)
        xs, cond = np.broadcast_arrays(xs, cond)
        shape = xs.shape
        xs, cond = xs.ravel(), cond.ravel()
        # R findInterval: number of values <= cond (1-based lower index)
        lo = np.searchsorted(values, cond, side="right")
        hi = lo + 1
        lo = np.clip(lo, 1, n_val)
        hi = np.clip(hi, 1, n_val)
        w_lo = np.clip((values[hi - 1] - cond) / spacing, 0.0, 1.0)
        out_lo = np.empty(xs.size)
        out_hi = np.empty(xs.size)
        for i in range(n_val):
            s = sorted_x[i]
            k = lo == i + 1
            if k.any():
                out_lo[k] = np.searchsorted(s, xs[k], side="right") / s.size
            k = hi == i + 1
            if k.any():
                out_hi[k] = np.searchsorted(s, xs[k], side="right") / s.size
        return (w_lo * out_lo + (1.0 - w_lo) * out_hi).reshape(shape)

    return cdf
