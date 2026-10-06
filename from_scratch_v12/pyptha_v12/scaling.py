"""Earthquake rupture scaling relations.

Python port of ``rptha/R/rupture_scaling.R`` (Geoscience Australia PTHA).
Only the active (non-commented) functions are ported:

    M0_2_Mw                 moment  <->  moment magnitude
    Mw_2_rupture_size       Mw -> (area, width, length) via empirical relations
    Mw_2_rupture_size_inverse   area -> Mw (inverse of the area relation)
    slip_from_Mw_area_mu    mean slip given Mw and rupture area
    slip_from_Mw            mean slip given Mw (area from a scaling relation)

The numerical behaviour mirrors the R code exactly, including the piecewise
(magnitude-dependent) coefficients of the Allen & Hayes (2017) interface
relation. Validated against the golden values embedded in the rptha examples
and docstrings (see pyptha/tests/test_scaling.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Moment magnitude <-> seismic moment
# ---------------------------------------------------------------------------


def M0_2_Mw(M0, inverse: bool = False, constant: float = 9.05):
    """Convert seismic moment M0 (Nm) to moment magnitude Mw, or the reverse.

    Uses ``Mw = 2/3 * (log10(M0) - 9.05)`` (Hanks & Kanamori 1977;
    Bird & Kagan 2004). The final constant can be overridden.

    Parameters
    ----------
    M0 : float or array-like
        Seismic moment in Nm when ``inverse=False``; otherwise it is treated
        as Mw and M0 (Nm) is returned.
    inverse : bool
        If False, return Mw given M0. If True, return M0 given Mw.
    constant : float
        The additive constant in the relation (default 9.05).
    """
    M0 = np.asarray(M0, dtype=float)
    if inverse:
        Mw = M0
        out = 10.0 ** (Mw * 3.0 / 2.0 + constant)
    else:
        out = 2.0 / 3.0 * (np.log10(M0) - constant)
    # Preserve scalar-in / scalar-out behaviour like R.
    if out.ndim == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# Rupture size scaling relations
# ---------------------------------------------------------------------------

# Each relation maps to (area_absigma, width_absigma, length_absigma), where
# each entry is (intercept a, slope b, log10 sigma). For relations with a
# magnitude-dependent (piecewise) coefficient, a callable is used instead.

_STATIC_RELATIONS = {
    "Strasser": {
        "area": (-3.476, 0.952, 0.304),
        "width": (-0.882, 0.351, 0.173),
        "length": (-2.477, 0.585, 0.180),
    },
    "Strasser-intraslab": {
        "area": (-3.225, 0.890, 0.184),
        "width": (-1.058, 0.356, 0.067),
        "length": (-2.350, 0.562, 0.146),
    },
    "AllenHayes-inslab": {
        "length": (-3.03, 0.63, 0.14),
        "width": (-1.01, 0.35, 0.15),
        "area": (-3.89, 0.96, 0.19),
    },
    "AllenHayes-outer-rise": {
        "length": (-2.87, 0.63, 0.08),
        "width": (-1.18, 0.35, 0.08),
        "area": (-3.89, 0.96, 0.11),
    },
    "Thingbaijam-subduction": {
        "length": (-2.412, 0.583, 0.107),
        "width": (-0.880, 0.366, 0.099),
        "area": (-3.292, 0.949, 0.150),
    },
    "Thingbaijam-normal": {
        "length": (-1.722, 0.485, 0.128),
        "width": (-0.829, 0.323, 0.128),
        "area": (-2.551, 0.808, 0.181),
    },
}

# Blaser relations: area is derived from length and width assuming zero
# correlation of residuals (matching the R code).
_BLASER = {
    "Blaser-reverse": {
        "length": (-2.28, 0.55, 0.18),
        "width": (-1.80, 0.45, 0.17),
    },
    "Blaser-normal": {
        "length": (-1.61, 0.46, 0.17),
        "width": (-1.08, 0.34, 0.16),
    },
}
for _name, _d in _BLASER.items():
    _l, _w = _d["length"], _d["width"]
    _d["area"] = (
        _l[0] + _w[0],
        _l[1] + _w[1],
        math.sqrt(_l[2] ** 2 + _w[2] ** 2),
    )
_STATIC_RELATIONS.update(_BLASER)

# Allen & Hayes (2017) interface: area and width are piecewise-linear in Mw.
# Thresholds set to the exact line-segment intersections, as in the R code.
_AH_AREA_MW_THRESHOLD = (5.62 + 2.23) / (1.22 - 0.31)
_AH_WIDTH_MW_THRESHOLD = (2.29 + 1.91) / 0.48


def _allenhayes_absigma(Mw: float):
    """Return (area, width, length) absigma tuples for the AllenHayes relation."""
    if Mw > _AH_AREA_MW_THRESHOLD:
        area = (2.23, 0.31, 0.256)
    else:
        area = (-5.62, 1.22, 0.256)

    length = (-2.90, 0.63, 0.182)

    if Mw > _AH_WIDTH_MW_THRESHOLD:
        width = (2.29, 0.0, 0.137)
    else:
        width = (-1.91, 0.48, 0.137)

    return area, width, length


@dataclass
class RuptureSize:
    """Detailed output of :func:`Mw_2_rupture_size` (``detailed=True``)."""

    values: dict          # {'area', 'width', 'length'} in km^2 / km / km
    log10_sigmas: dict     # log10 standard deviation of each quantity
    plus_CI: dict          # +CI_sd bound
    minus_CI: dict         # -CI_sd bound
    area_absigma: tuple = field(default=())
    width_absigma: tuple = field(default=())
    length_absigma: tuple = field(default=())
    relation: str = ""


def _relation_absigma(Mw: float, relation: str):
    """Resolve (area, width, length) absigma tuples for a relation at this Mw."""
    if relation == "AllenHayes":
        return _allenhayes_absigma(Mw)
    if relation in _STATIC_RELATIONS:
        d = _STATIC_RELATIONS[relation]
        return d["area"], d["width"], d["length"]
    raise ValueError(f"Relation value {relation!r} not recognized")


def Mw_2_rupture_size(Mw: float, relation: str = "Strasser",
                      detailed: bool = False, CI_sd: float = 1.0):
    """Compute rupture area (km^2), width (km) and length (km) from Mw.

    Parameters
    ----------
    Mw : float
        Moment magnitude (scalar only, matching the R ``length(Mw) == 1`` check).
    relation : str
        Scaling relation name. Supported: 'Strasser' (default),
        'Strasser-intraslab', 'AllenHayes', 'AllenHayes-inslab',
        'AllenHayes-outer-rise', 'Blaser-reverse', 'Blaser-normal',
        'Thingbaijam-subduction', 'Thingbaijam-normal'.
    detailed : bool
        If False, return a dict with area/width/length. If True, return a
        :class:`RuptureSize` with confidence-interval information.
    CI_sd : float
        Number of standard deviations for the confidence interval (detailed).

    Returns
    -------
    dict or RuptureSize
    """
    Mw = float(Mw)
    area_absigma, width_absigma, length_absigma = _relation_absigma(Mw, relation)

    area = 10.0 ** (area_absigma[0] + Mw * area_absigma[1])
    width = 10.0 ** (width_absigma[0] + Mw * width_absigma[1])
    length = 10.0 ** (length_absigma[0] + Mw * length_absigma[1])

    values = {"area": area, "width": width, "length": length}
    if not detailed:
        return values

    log10_sigmas = {
        "area": area_absigma[2],
        "width": width_absigma[2],
        "length": length_absigma[2],
    }
    plus_CI = {k: 10.0 ** (math.log10(values[k]) + CI_sd * log10_sigmas[k])
               for k in values}
    minus_CI = {k: 10.0 ** (math.log10(values[k]) - CI_sd * log10_sigmas[k])
                for k in values}

    return RuptureSize(
        values=values,
        log10_sigmas=log10_sigmas,
        plus_CI=plus_CI,
        minus_CI=minus_CI,
        area_absigma=area_absigma,
        width_absigma=width_absigma,
        length_absigma=length_absigma,
        relation=relation,
    )


def Mw_2_rupture_size_inverse(area: float, relation: str = "Strasser",
                              CI_sd: float = 0.0):
    """Inverse of the area scaling relation: given area (km^2), return Mw.

    Positive ``CI_sd`` corresponds to lower Mw, negative to higher Mw
    (matching the R convention).
    """
    if np.ndim(CI_sd) > 0:
        raise ValueError("CI_sd must be scalar")
    if np.ndim(area) > 0:
        raise ValueError("area must be scalar")

    # Coefficients evaluated at Mw=6.0 (below every piecewise threshold).
    area_coef = Mw_2_rupture_size(6.0, relation=relation, detailed=True).area_absigma

    Mw = (math.log10(area) - area_coef[0] - CI_sd * area_coef[2]) / area_coef[1]

    if relation == "AllenHayes":
        if Mw > _AH_AREA_MW_THRESHOLD:
            area_coef = Mw_2_rupture_size(
                Mw, relation=relation, detailed=True).area_absigma
            Mw = (math.log10(area) - area_coef[0]
                  - CI_sd * area_coef[2]) / area_coef[1]

    return Mw


# ---------------------------------------------------------------------------
# Slip relations
# ---------------------------------------------------------------------------


def slip_from_Mw_area_mu(Mw: float, area: float, mu: float = 3e10,
                         constant: float = 9.05) -> float:
    """Mean slip (m) on a rupture of given Mw and area (km^2).

    ``mu`` is the shear modulus in Pascals.
    """
    M0 = M0_2_Mw(Mw, inverse=True, constant=constant)
    area_m2 = area * 1e6
    return M0 / (area_m2 * mu)


def slip_from_Mw(Mw, mu: float = 3e10, relation: str = "Strasser",
                 area_function: Callable[[float], float] | None = None,
                 constant: float = 9.05):
    """Mean slip (m) given Mw, using a scaling relation for the area.

    ``area_function`` maps Mw -> area (km^2); by default it uses the area from
    :func:`Mw_2_rupture_size` for the given ``relation``. ``Mw`` may be a
    scalar or an array.
    """
    if area_function is None:
        def area_function(m):  # noqa: E306
            return Mw_2_rupture_size(m, relation=relation)["area"]

    Mw_arr = np.atleast_1d(np.asarray(Mw, dtype=float))
    area = np.array([area_function(float(m)) for m in Mw_arr]) * 1e6
    M0 = np.asarray(M0_2_Mw(Mw_arr, inverse=True, constant=constant), dtype=float)
    slip = M0 / (mu * area)

    if np.ndim(Mw) == 0:
        return float(slip[0])
    return slip
