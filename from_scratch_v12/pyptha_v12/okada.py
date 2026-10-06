"""Okada (1985) surface deformation for rectangular faults - pure NumPy.

Direct port of rptha's ``src/okada_tsunami_fortran.f`` (subroutines
``SRECTG``, ``SRECTF`` and the ``fault_disp`` driver). No Fortran compiler
required; the numerics reproduce the Okada (1985) analytic half-space
solution exactly as coded there, including:

  * the medium constant ``alp = mu/(lambda+mu)``; rptha uses ``alp = 0.5``
    (lambda = mu, Poisson's ratio = 0.25),
  * the coordinate transform that moves the fault *centroid* to Okada's
    lower reference corner and rotates into the strike-aligned frame,
  * the strike rotation of the horizontal displacements back to E/N.

Public entry point :func:`okada_tsunami` mirrors the R function of the same
name (see ``R/rptha/R/okada_tsunami.R``): given one or more rectangular
sub-faults and a set of observation points, it returns the east, north and
vertical surface displacements summed over all sub-faults.

Reference example (from the R docstring) is reproduced in the tests:
a pure thrust fault (dip 15, dip-slip 1 m) gives an uplift lobe.
"""

from __future__ import annotations

import numpy as np

_PI2 = 6.283185307179586  # 2*pi, matching the Fortran literal
_DTR = np.pi / 180.0


def _srectg(alp, xi, et, q, sd, cd, disl1, disl2, disl3):
    """Indefinite integral of surface displacement (Okada 1985, SRECTG).

    All inputs are numpy arrays (broadcast together) except the scalar
    ``alp``, ``disl*``. Returns only U1, U2, U3 (displacements); strain/tilt
    terms are not needed for tsunami initial conditions and are omitted.

    This is a faithful transcription of the SRECTG subroutine, restricted to
    the displacement outputs.
    """
    xi = np.asarray(xi, dtype=float)
    et = np.asarray(et, dtype=float)
    q = np.asarray(q, dtype=float)

    xi2 = xi * xi
    et2 = et * et
    q2 = q * q
    r2 = xi2 + et2 + q2
    r = np.sqrt(r2)
    r3 = r * r2
    d = et * sd - q * cd
    y = et * cd + q * sd
    ret = r + et
    ret = np.where(ret < 0.0, 0.0, ret)
    rd = r + d
    rrd = 1.0 / (r * rd)

    # TT = atan(xi*et/(q*r)), 0 where q == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(q != 0.0, np.arctan(xi * et / (q * r)), 0.0)

    # RE = 1/RET (0 if RET == 0); DLE = log(RET) or -log(R-ET)
    re = np.where(ret != 0.0, 1.0 / np.where(ret != 0.0, ret, 1.0), 0.0)
    dle = np.where(ret != 0.0,
                   np.log(np.where(ret != 0.0, ret, 1.0)),
                   -np.log(np.where(ret != 0.0, 1.0, r - et)))

    rrx = 1.0 / (r * (r + xi))
    rinv = 1.0 / r
    rre = re * rinv
    axi = (2.0 * r + xi) * rrx * rrx * rinv
    aet = (2.0 * r + et) * rre * rre * rinv

    # Coefficients A1..C3 differ for inclined vs vertical faults (CD == 0).
    inclined = (cd != 0.0)

    # --- inclined branch ---
    if inclined:
        cdinv = 1.0 / cd
        td = sd * cdinv
        x = np.sqrt(xi2 + q2)
        a5 = np.where(
            xi != 0.0,
            alp * 2.0 * cdinv * np.arctan(
                (et * (x + q * cd) + x * (r + x) * sd)
                / np.where(xi != 0.0, xi * (r + x) * cd, 1.0)),
            0.0,
        )
        a4 = alp * cdinv * (np.log(rd) - sd * dle)
        a3 = alp * (y / rd * cdinv - dle) + td * a4
        a1 = -alp * cdinv * xi / rd - td * a5
        c1 = alp * cdinv * xi * (rrd - sd * rre)
        c3 = alp * cdinv * (q * rre - y * rrd)
        b1 = alp * cdinv * (xi2 * rrd - 1.0) / rd - td * c3
        b2 = alp * cdinv * xi * y * rrd / rd - td * c1
    else:
        rd2 = rd * rd
        a1 = -alp * 0.5 * xi * q / rd2
        a3 = alp * 0.5 * (et / rd + y * q / rd2 - dle)
        a4 = -alp * q / rd
        a5 = -alp * xi * sd / rd
        b1 = alp * 0.5 * q / rd2 * (2.0 * xi2 * rrd - 1.0)
        b2 = alp * 0.5 * xi * sd / rd2 * (2.0 * q2 * rrd - 1.0)
        c1 = alp * xi * q * rrd / rd
        c3 = alp * sd / rd * (xi2 * rrd - 1.0)

    a2 = -alp * dle - a3
    b3 = -alp * xi * rre - b2
    b4 = -alp * (cd * rinv + q * sd * rre) - b1
    c2 = alp * (-sd * rinv + q * cd * rre) - c3

    u1 = np.zeros_like(r)
    u2 = np.zeros_like(r)
    u3 = np.zeros_like(r)

    # Strike-slip contribution
    if disl1 != 0.0:
        un = disl1 / _PI2
        req = rre * q
        u1 = u1 - un * (req * xi + tt + a1 * sd)
        u2 = u2 - un * (req * y + q * cd * re + a2 * sd)
        u3 = u3 - un * (req * d + q * sd * re + a4 * sd)

    # Dip-slip contribution
    if disl2 != 0.0:
        un = disl2 / _PI2
        sdcd = sd * cd
        u1 = u1 - un * (q / r - a3 * sdcd)
        u2 = u2 - un * (y * q * rrx + cd * tt - a1 * sdcd)
        u3 = u3 - un * (d * q * rrx + sd * tt - a5 * sdcd)

    # Tensile contribution
    if disl3 != 0.0:
        un = disl3 / _PI2
        sdsd = sd * sd
        u1 = u1 + un * (q2 * rre - a3 * sdsd)
        u2 = u2 + un * (-d * q * rrx - sd * (xi * q * rre - tt) - a1 * sdsd)
        u3 = u3 + un * (y * q * rrx + cd * (xi * q * rre - tt) - a5 * sdsd)

    return u1, u2, u3


def _srectf(alp, x, y, dep, al1, al2, aw1, aw2, sd, cd, disl1, disl2, disl3):
    """Surface displacement of a rectangular fault (Okada 1985, SRECTF).

    Sums the SRECTG indefinite integral over the four fault corners with the
    correct signs. ``x``, ``y`` are observation-point arrays in the fault's
    strike-aligned frame; the rest are scalars for a single sub-fault.
    """
    p = y * cd + dep * sd
    q = y * sd - dep * cd

    u1 = np.zeros_like(x)
    u2 = np.zeros_like(x)
    u3 = np.zeros_like(x)

    for k, aw in ((1, aw1), (2, aw2)):
        et = p - aw
        for j, al in ((1, al1), (2, al2)):
            xi = x - al
            sign = -1.0 if (j + k) == 3 else 1.0
            du1, du2, du3 = _srectg(alp, xi, et, q, sd, cd,
                                    disl1, disl2, disl3)
            u1 = u1 + sign * du1
            u2 = u2 + sign * du2
            u3 = u3 + sign * du3

    return u1, u2, u3


def okada_tsunami(elon, elat, edep, strk, dip, lnth, wdt, disl1, disl2,
                  rlon, rlat, dstmx=9.0e20, dstmx_min=20.0, alp=0.5):
    """Surface displacement from rectangular sub-faults (Okada 1985).

    Port of rptha's ``okada_tsunami``. Sub-fault parameters are given at the
    rupture *centroid*; the routine internally translates to Okada's lower
    reference corner and rotates into the strike-aligned frame, exactly as
    the Fortran ``fault_disp`` driver does.

    Parameters
    ----------
    elon, elat : array-like
        Centroid x, y of each sub-fault, in metres.
    edep : array-like
        Centroid depth of each sub-fault, in km.
    strk : array-like
        Strike (degrees clockwise from north).
    dip : array-like
        Dip (degrees below horizontal, dipping to the right along strike).
    lnth, wdt : array-like
        Sub-fault length and width, in km.
    disl1, disl2 : array-like
        Along-strike and up-dip dislocation, in m.
    rlon, rlat : array-like
        Observation-point x, y, in metres.
    dstmx : float
        Optional cutoff distance (as a multiple of depth) beyond which a
        sub-fault's contribution is ignored; saves time. Default effectively
        infinite.
    dstmx_min : float
        Minimum value of ``dstmx*depth`` (km).
    alp : float
        Medium constant mu/(lambda+mu). Default 0.5 (Poisson 0.25), as rptha.

    Returns
    -------
    dict with keys ``edsp``, ``ndsp``, ``zdsp`` : east, north and vertical
    surface displacement at each observation point (m).
    """
    elon = np.atleast_1d(np.asarray(elon, dtype=float))
    elat = np.atleast_1d(np.asarray(elat, dtype=float))
    edep = np.atleast_1d(np.asarray(edep, dtype=float))
    strk = np.atleast_1d(np.asarray(strk, dtype=float))
    dip = np.atleast_1d(np.asarray(dip, dtype=float))
    lnth = np.atleast_1d(np.asarray(lnth, dtype=float))
    wdt = np.atleast_1d(np.asarray(wdt, dtype=float))
    disl1 = np.atleast_1d(np.asarray(disl1, dtype=float))
    disl2 = np.atleast_1d(np.asarray(disl2, dtype=float))
    rlon = np.atleast_1d(np.asarray(rlon, dtype=float))
    rlat = np.atleast_1d(np.asarray(rlat, dtype=float))

    n = elon.size
    for name, arr in (("elat", elat), ("edep", edep), ("strk", strk),
                      ("dip", dip), ("lnth", lnth), ("wdt", wdt),
                      ("disl1", disl1), ("disl2", disl2)):
        if arr.size != n:
            raise ValueError(f"sub-fault variable {name!r} has length "
                             f"{arr.size}, expected {n}")
    if rlat.size != rlon.size:
        raise ValueError("len(rlon) != len(rlat)")

    # Convert dstmx to an absolute distance per sub-fault (km).
    dstmx_abs = np.maximum(dstmx * edep, dstmx_min)

    edsp = np.zeros(rlon.size)
    ndsp = np.zeros(rlon.size)
    zdsp = np.zeros(rlon.size)

    for i in range(n):
        cd = np.cos(dip[i] * _DTR)
        sd = np.sin(dip[i] * _DTR)
        ss = np.sin(strk[i] * _DTR)
        cs = np.cos(strk[i] * _DTR)

        # Half diagonal length of the surface projection of the slip area.
        tmp1 = 0.5 * np.sqrt(lnth[i] ** 2 + (cd * wdt[i]) ** 2)
        tmp2 = np.arctan2(cd * wdt[i], lnth[i])
        odep = edep[i] + 0.5 * wdt[i] * sd  # origin depth in Okada frame

        # Origin of the Okada reference frame (metres).
        oy = elat[i] - 1000.0 * tmp1 * np.cos(tmp2 - strk[i] * _DTR)
        ox = elon[i] + 1000.0 * tmp1 * np.sin(tmp2 - strk[i] * _DTR)

        # Translate observation points to the new origin (km).
        x = (rlon - ox) * 0.001
        y = (rlat - oy) * 0.001
        d = np.sqrt(x ** 2 + y ** 2)

        # Distance cutoff mask.
        if dstmx_abs[i] > 0.0:
            keep = ~(d > dstmx_abs[i])
        else:
            keep = np.ones_like(d, dtype=bool)

        if not np.any(keep):
            continue

        xk = x[keep]
        yk = y[keep]
        dk = d[keep]

        # Rotate into the strike-aligned Okada frame.
        az = 90.0 - np.arctan2(yk, xk) / _DTR
        yr = dk * np.sin(_DTR * (strk[i] - az))
        xr = dk * np.cos(_DTR * (strk[i] - az))

        u1, u2, u3 = _srectf(
            alp, xr, yr, odep, 0.0, lnth[i], 0.0, wdt[i],
            sd, cd, disl1[i], disl2[i], 0.0)

        # Rotate horizontal displacements back to E/N and accumulate.
        edsp[keep] += -u2 * cs + u1 * ss
        ndsp[keep] += u2 * ss + u1 * cs
        zdsp[keep] += u3

    return {"edsp": edsp, "ndsp": ndsp, "zdsp": zdsp}
