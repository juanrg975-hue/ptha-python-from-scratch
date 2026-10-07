"""Orthogonal contour -> unit-source-grid discretisation (rptha-faithful).

Python port of rptha's improved down-dip-line algorithm
(``rptha/R/downdip_3d_lines_on_source.R``):

    make_source_contours_interpolation_function_list  (+ orientation logic)
    get_quality_matrix          - orthogonality + regularity "badness" measure
    create_downdip_lines_on_source_contours_improved  - multigrid optimisation

The idea: describe the position of each down-dip line by a set of normalised
distances ``s`` in [0, 1] along every depth contour (one row per contour).
Optimise those distances so the resulting grid lines cross the contours as
orthogonally as possible while keeping unit-source sizes even. rptha uses the
Levenberg-Marquardt least-squares solver ``minpack.lm::nls.lm`` (MINPACK
``lmdif``). v8 reproduces that call step for step through
``scipy.optimize.leastsq`` (see :func:`_nls_lm`): same finite-difference
Jacobian (bit for bit), same tolerances, same 50-iteration cap, same ``Inf``
residual for invalid points. On PTHA18's own contours the resulting down-dip
lines match rptha's R output to within metres (0.00 m on puysegur2 and
makran2), where v5-v7 differed by 0.2-1.4 km.

The public entry point :func:`discretized_source_from_contours_orthogonal`
returns a ``unit_source_grid`` of shape ``(n_downdip+1, 3, n_alongstrike+1)``,
the same layout everything else in pyptha consumes.

This is the "as rptha does it" counterpart to the simpler
``unit_sources.discretized_source_from_contours``.
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")
_R = 6378137.0
_DEG2RAD = np.pi / 180.0

# rptha's own geodesy is SPHERICAL everywhere it uses geosphere
# (distHaversine, gcIntermediate, f=0 at every call site) -- _GEOD (above)
# is WGS84-ellipsoidal, correctly matching R's OWN ellipsoidal calls
# (sp::SpatialLinesLengths(longlat=TRUE), used for top_len_km/contour arc
# length), but interpolate_gc_path wraps geosphere::gcIntermediate, which
# is spherical-only -- no ellipsoidal option exists in geosphere. Using
# _GEOD there was an undocumented inconsistency: quantified on a real
# ~198 km Kermadec-area segment, ellipsoidal vs spherical interpolation
# differs by ~4 m pointwise and ~537 m (~0.27%) in total arc length.
_GEOD_SPHERE = Geod(a=_R, f=0.0)

# MINPACK fdjac2's finite-difference step with epsfcn = 0 (nls.lm's default).
_SQRT_EPS = float(np.sqrt(np.finfo(float).eps))
# _quality_matrix's residual for an invalid (non-monotone) s_matrix: Inf,
# exactly as R (get_quality_matrix returns s_matrix * 0 + Inf). v5-v7 used a
# finite 1e100 on the belief that scipy's MINPACK rejects non-finite values;
# scipy.optimize.leastsq does not, and the difference is NOT cosmetic: with
# Inf, MINPACK's residual norm of a rejected trial is NaN, so its test
# "p1*fnorm1 >= fnorm" is FALSE and the trust region shrinks by
# 0.5*dirder/(dirder + 0.5*actred); with 1e100 the test is TRUE and it always
# shrinks by 0.1. That sent the Python optimiser down a different path from
# R's whenever a trial step was invalid (verified on makran2: first step RSS
# 3.595 vs R's 3.668; with Inf, 3.66832243 vs 3.66832247 and the same final
# optimum to 1e-9).
_BAD = np.inf


# ---------------------------------------------------------------------------
# Local tangent-plane projection (port of spherical_to_cartesian2d_coordinates)
# ---------------------------------------------------------------------------


def _spherical_to_cartesian2d(coords_lonlat, origin_lonlat):
    """Local equirectangular projection about origin(s). Port of rptha's routine.

    ``coords_lonlat`` and ``origin_lonlat`` are (N, 2) arrays (origin may be a
    single (2,) row broadcast to all). Returns (N, 2) local x, y in metres.
    """
    coords = np.atleast_2d(np.asarray(coords_lonlat, dtype=float))
    origin = np.atleast_2d(np.asarray(origin_lonlat, dtype=float))
    dlon = (coords[:, 0] - origin[:, 0]) * _DEG2RAD
    dlat = (coords[:, 1] - origin[:, 1]) * _DEG2RAD
    x = _R * dlon * np.cos(origin[:, 1] * _DEG2RAD)
    y = _R * dlat
    return np.column_stack([x, y])


def _dist_haversine(p1, p2):
    """Spherical haversine distance, geosphere::distHaversine (r = 6378137),
    as used throughout rptha's discretiser.

    Written with geosphere's own expression order (``2 * atan2(sqrt(a),
    sqrt(1 - a)) * r``, v8) rather than the algebraically equal
    ``2 r asin(sqrt(a))``: the down-dip-line optimiser differentiates the
    contour parametrisation numerically, which amplifies last-bit
    differences by ~1e8, so matching R's arithmetic keeps its iterations on
    R's path for longer."""
    lon1, lat1 = np.radians(p1[0]), np.radians(p1[1])
    lon2, lat2 = np.radians(p2[0]), np.radians(p2[1])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    a = np.minimum(a, 1.0)
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * _R


def _bearing(p1, p2):
    """Spherical initial bearing, exactly geosphere::bearing(f=0)."""
    lon1, lat1 = np.radians(p1[0]), np.radians(p1[1])
    lon2, lat2 = np.radians(p2[0]), np.radians(p2[1])
    dlon = lon2 - lon1
    yy = np.sin(dlon) * np.cos(lat2)
    xx = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(yy, xx)) % 360.0


# ---------------------------------------------------------------------------
# Contour interpolators (port of make_source_contours_interpolation_function_list)
# ---------------------------------------------------------------------------


def _line_interpolator(coords):
    """Map s in [0, 1] -> (lon, lat) along a polyline, by arc length.

    Port of ``make_line_interpolation_fun``: linear interpolation of lon and
    lat against normalised cumulative distance.
    """
    coords = np.asarray(coords, dtype=float)
    seg = np.array([_dist_haversine(coords[k], coords[k + 1])
                    for k in range(len(coords) - 1)])
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    s = dist / dist[-1]
    # R's approxfun collapses tied x values (zero-length segments) with
    # ties = mean; consecutive duplicate vertices have identical y, so
    # keeping the first of each run is the same thing.
    keep = np.concatenate([[True], np.diff(s) > 0])
    s, cx, cy = s[keep], coords[keep, 0], coords[keep, 1]
    last = s.size - 1

    def approx(v, y):
        # stats::approx1 (R/src/library/stats/src/approx.c), linear branch:
        # y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i])), with exact
        # knot hits returning the knot value. Outside [0, 1] R gives NA
        # (rule = 1); the optimiser never asks there (s is pinned at 0 and
        # 1 and must be monotone), so the end values are returned instead.
        i = np.clip(np.searchsorted(s, v, side="right") - 1, 0, last - 1)
        j = i + 1
        out = y[i] + (y[j] - y[i]) * ((v - s[i]) / (s[j] - s[i]))
        out = np.where(v == s[j], y[j], out)
        out = np.where(v == s[i], y[i], out)
        out = np.where(v <= 0.0, y[0], out)
        return np.where(v >= 1.0, y[last], out)

    def f(sv):
        sv = np.atleast_1d(np.asarray(sv, dtype=float))
        return np.column_stack([approx(sv, cx), approx(sv, cy)])

    return f


def _build_interpolators(contours):
    """Build one s->(lon,lat) interpolator per contour, consistently oriented.

    Port of ``make_source_contours_interpolation_function_list``: sort contours
    by depth, orient the top line along strike, then flip any deeper contour
    that runs the opposite way.
    """
    ordered = sorted(contours, key=lambda c: float(c[0]))
    depths = [float(d) for d, _ in ordered]
    lines = [np.asarray(c, dtype=float) for _, c in ordered]

    top = lines[0]
    nc = len(top)
    top_bearing = _bearing(top[0, :2], top[min(nc, 10) - 1, :2])

    bot = lines[-1]
    d1 = _dist_haversine(top[0, :2], bot[0, :2])
    d2 = _dist_haversine(top[0, :2], bot[-1, :2])
    if d1 < d2:
        updip_bearing = _bearing(bot[0, :2], top[0, :2])
    else:
        updip_bearing = _bearing(bot[-1, :2], top[0, :2])

    if (top_bearing - updip_bearing) % 360 >= 180:
        top = top[::-1]
        lines[0] = top

    interps = [_line_interpolator(top[:, :2])]
    for i in range(1, len(lines)):
        nxt = lines[i]
        da = _dist_haversine(top[0, :2], nxt[0, :2])
        db = _dist_haversine(top[0, :2], nxt[-1, :2])
        if db <= da:
            nxt = nxt[::-1]
            lines[i] = nxt
        interps.append(_line_interpolator(nxt[:, :2]))
    return interps, depths


def _get_xy(s_matrix, interps):
    """Evaluate all interpolators at the given normalised distances.

    ``s_matrix`` is (num_l, np); returns (x, y) each (num_l, np).
    """
    num_l, npv = s_matrix.shape
    x = np.empty_like(s_matrix)
    y = np.empty_like(s_matrix)
    for i in range(num_l):
        pts = interps[i](s_matrix[i])
        x[i] = pts[:, 0]
        y[i] = pts[:, 1]
    return x, y


# ---------------------------------------------------------------------------
# Quality (badness) measure  (port of get_quality_matrix)
# ---------------------------------------------------------------------------


def _quality_matrix(s_matrix, interps):
    """Per-node 'badness': non-orthogonality + uneven spacing. Port of R.

    Lower is better; the optimiser drives this toward zero. Returns a flat
    array (the residual vector nls.lm minimises).
    """
    # R: get_xy_coords stops on any non-increasing row -> the caller returns
    # a q_matrix of Inf, so nls.lm rejects the step (see _BAD).
    if np.any(np.diff(s_matrix, axis=1) < 0):
        return np.full(s_matrix.size, _BAD)
    try:
        x, y = _get_xy(s_matrix, interps)
    except Exception:
        return np.full(s_matrix.size, _BAD)

    num_l, npv = s_matrix.shape
    j = np.arange(npv)
    jp1 = np.minimum(j + 1, npv - 1)
    jm1 = np.maximum(j - 1, 0)
    i_idx = np.arange(num_l)
    ip1_i = np.minimum(num_l - 1, i_idx + 1)
    im1_i = np.maximum(0, i_idx - 1)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # Vectorised over BOTH i (down-dip row) and j (along-strike column):
        # every (i, j) node's four neighbours at once, instead of a Python
        # loop over i alone. _spherical_to_cartesian2d already broadcasts
        # elementwise, so flattening (num_l, npv) -> (num_l*npv,) is exact,
        # not an approximation -- same arithmetic, same order of operations
        # per node, just done for every row in one call instead of num_l
        # separate calls.
        centre = np.column_stack([x.ravel(), y.ravel()])
        x_ip1 = x[ip1_i][:, j]
        y_ip1 = y[ip1_i][:, j]
        x_im1 = x[im1_i][:, j]
        y_im1 = y[im1_i][:, j]
        x_jp1 = x[:, jp1]
        y_jp1 = y[:, jp1]
        x_jm1 = x[:, jm1]
        y_jm1 = y[:, jm1]

        c_ip1 = _spherical_to_cartesian2d(
            np.column_stack([x_ip1.ravel(), y_ip1.ravel()]), centre)
        c_im1 = _spherical_to_cartesian2d(
            np.column_stack([x_im1.ravel(), y_im1.ravel()]), centre)
        c_jp1 = _spherical_to_cartesian2d(
            np.column_stack([x_jp1.ravel(), y_jp1.ravel()]), centre)
        c_jm1 = _spherical_to_cartesian2d(
            np.column_stack([x_jm1.ravel(), y_jm1.ravel()]), centre)

        v1 = c_ip1 - c_im1                    # down-dip direction
        v2 = c_jp1 - c_jm1                    # along-strike direction
        v1A = c_ip1                            # (centre is origin -> 0)

        dot_1A_2 = np.sum(v1A * v2, axis=1)
        sq_1A = np.sum(v1A * v1A, axis=1)
        sq_2 = np.sum(v2 * v2, axis=1)
        dot_1_2 = np.sum(v1 * v2, axis=1)
        sq_1 = np.sum(v1 * v1, axis=1)

        # Exactly R's expression, including its operation order
        # (sqrt(a * b), not sqrt(a) * sqrt(b); no extra epsilon on the
        # second term).
        ortho = (0.25 * np.abs(dot_1A_2) / (1.0e-12 + np.sqrt(sq_1A * sq_2))
                 + np.abs(dot_1_2) / np.sqrt(sq_1 * sq_2))
        ortho = ortho.reshape(num_l, npv)

        # Relative-spacing term, exactly R's double-exp formula (no
        # clipping: overflow -> inf is a valid 'very bad' residual, and
        # degenerate spacings are already rejected via the monotonic
        # check above).
        sp_next = np.abs(s_matrix[:, jp1] - s_matrix[:, j]) * (npv - 1) + (jp1 == j)
        sp_prev = np.abs(s_matrix[:, j] - s_matrix[:, jm1]) * (npv - 1) + (jm1 == j)
        even = (0.5 * (np.exp(np.exp(np.abs(np.log(sp_next))) - 1) - 1)
                + 0.5 * (np.exp(np.exp(np.abs(np.log(sp_prev))) - 1) - 1))

        q = ortho + even

    # No clean-up of Inf/NaN (v5-v7 mapped both to 1e100): R returns the raw
    # values, and MINPACK's norm treats them differently (see _BAD).
    # R returns c(q_matrix): column-major flattening. The residual ORDER
    # matters to Levenberg-Marquardt's finite-difference Jacobian scaling, so
    # match it exactly.
    return q.ravel(order="F")


# ---------------------------------------------------------------------------
# Multigrid optimisation of the down-dip lines
# ---------------------------------------------------------------------------


def _stencil_colouring(num_l, npv):
    """Column grouping that makes the finite-difference Jacobian cheap.

    Residual (i, j) of :func:`_quality_matrix` reads ``s`` only at (i, j),
    (i+-1, j) and (i, j+-1): a 5-point stencil, neighbours clamped at the
    edges. Colour free parameter (i, j) with ``c = (i + 2j) mod 5``; the five
    stencil points of every residual then carry five DIFFERENT colours, so
    perturbing all parameters of one colour at once changes each residual
    through exactly one of them. One function call per colour therefore
    gives, bit for bit, the columns MINPACK's ``fdjac2`` builds with one call
    per parameter.

    Returns ``(colour_of_param, owner)``. Parameters are indexed like R's
    ``c(s_matrix[, 2:(np-1)])`` (column-major); residuals like R's
    ``c(q_matrix)``. ``owner[c, r]`` is the colour-``c`` parameter residual
    ``r`` depends on, or -1.
    """
    pidx = np.full((num_l, npv), -1, dtype=int)
    colour = np.full((num_l, npv), -1, dtype=int)
    for j in range(1, npv - 1):
        for i in range(num_l):
            pidx[i, j] = i + num_l * (j - 1)
            colour[i, j] = (i + 2 * j) % 5
    colour_of_param = np.empty(num_l * (npv - 2), dtype=int)
    colour_of_param[pidx[:, 1:npv - 1].ravel(order="F")] = \
        colour[:, 1:npv - 1].ravel(order="F")
    owner = np.full((5, num_l * npv), -1, dtype=int)
    for j in range(npv):
        for i in range(num_l):
            r = i + num_l * j
            for a, b in ((i, j), (min(i + 1, num_l - 1), j), (max(i - 1, 0), j),
                         (i, min(j + 1, npv - 1)), (i, max(j - 1, 0))):
                p = pidx[a, b]
                if p >= 0:
                    owner[colour[a, b], r] = p
    return colour_of_param, owner


def _fd_jacobian(residual, x, f0, colour_of_param, owner):
    """Forward-difference Jacobian, exactly MINPACK ``fdjac2`` with epsfcn=0.

    Column p is ``(f(x + h_p e_p) - f(x)) / h_p`` with ``h_p = sqrt(eps)|x_p|``
    (``sqrt(eps)`` if ``x_p == 0``), the step R's ``nls.lm(jac = NULL)``
    uses. Evaluated five parameters-groups at a time (see
    :func:`_stencil_colouring`), which is exact because each residual depends
    on one parameter per group. The one non-local case is the monotonicity
    guard of :func:`_quality_matrix`, which returns ``_BAD`` everywhere; if a
    grouped step trips it, that group is redone one column at a time, which
    is what ``fdjac2`` itself would have seen.
    """
    m, n = f0.size, x.size
    h = _SQRT_EPS * np.abs(x)
    h[h == 0.0] = _SQRT_EPS
    jac = np.zeros((m, n))
    rows = np.arange(m)
    f0_bad = bool(np.all(f0 >= _BAD))
    for c in range(5):
        cols = np.nonzero(colour_of_param == c)[0]
        if cols.size == 0:
            continue
        xp = x.copy()
        xp[cols] = x[cols] + h[cols]
        fp = residual(xp)
        if (not f0_bad) and np.all(fp >= _BAD):
            for p in cols:
                xq = x.copy()
                xq[p] = x[p] + h[p]
                jac[:, p] = (residual(xq) - f0) / h[p]
            continue
        own = owner[c]
        k = own >= 0
        jac[rows[k], own[k]] = (fp[k] - f0[k]) / h[own[k]]
    return jac


def _nls_lm(residual, x0, colouring, ftol, maxiter=50, maxfev=None):
    """``minpack.lm::nls.lm(par, fn, control=list(ftol=..., ...))`` with
    ``jac = NULL``, reproduced step for step.

    That call is MINPACK's Levenberg-Marquardt ``lmdif`` (forward-difference
    Jacobian, epsfcn = 0), with nls.lm's defaults ptol = sqrt(eps),
    gtol = 0, factor = 100, automatic variable scaling (mode 1), plus two
    aborts minpack.lm adds around it (``nls_lm.c`` / ``fcn_lmdif.c``):

    * ``maxiter`` (default 50): the callback stops lmdif as soon as the
      ``maxiter``-th Jacobian has been computed, i.e. at most ``maxiter - 1``
      steps are taken;
    * ``maxfev`` (default ``100 * (n + 1)``), counted the lmdif way: every
      finite-difference Jacobian costs n evaluations.

    Here the Jacobian is built by :func:`_fd_jacobian` (bit-identical to
    fdjac2, five calls instead of n) and passed to MINPACK's ``lmder``
    through :func:`scipy.optimize.leastsq`, which runs the same algorithm
    as lmdif given the same Jacobian. An abort is reproduced by handing
    MINPACK a zero Jacobian: its gradient test (``gnorm <= gtol = 0``) then
    stops it at the current point before any further step, which is exactly
    where nls.lm returns. ``maxfev`` is honoured at Jacobian boundaries
    (lmdif checks it after each trial point; only the retry path, which R
    itself randomises without a seed, can reach it).

    Returns ``(x, niter)`` with ``niter`` as nls.lm reports it (the number of
    Jacobians computed), which is what rptha tests ``== 1``.
    """
    from scipy.optimize import leastsq

    n = x0.size
    if maxfev is None:
        maxfev = 100 * (n + 1)
    colour_of_param, owner = colouring
    state = {"niter": 0, "nfev": 0, "jx": None, "jac": None,
             "fx": None, "f": None}

    def fun(x):
        state["nfev"] += 1
        f = residual(x)
        state["fx"], state["f"] = x.copy(), f
        return f

    def dfun(x):
        # The same x twice in a row is scipy's own shape check followed by
        # lmder's first iteration, not a new iteration.
        if state["jx"] is not None and np.array_equal(state["jx"], x):
            return state["jac"]
        state["niter"] += 1
        lmdif_nfev = state["nfev"] + n * state["niter"]
        if state["niter"] >= maxiter or lmdif_nfev > maxfev:
            jac = np.zeros((state["f"].size if state["f"] is not None
                            else residual(x).size, n))
        else:
            f0 = (state["f"] if state["fx"] is not None
                  and np.array_equal(state["fx"], x) else residual(x))
            jac = _fd_jacobian(residual, x, f0, colour_of_param, owner)
        state["jx"], state["jac"] = x.copy(), jac
        return jac

    x, _ier = leastsq(fun, x0, Dfun=dfun, full_output=False, col_deriv=0,
                      ftol=ftol, xtol=_SQRT_EPS, gtol=0.0,
                      maxfev=10 ** 9, factor=100, diag=None)
    return np.asarray(x, dtype=float), state["niter"]


def _nearest_point_init_s_matrix(interps, npv, num_l, n_dense=2000):
    """Alternative starting s_matrix: align each row to the SHALLOWEST row by
    nearest-point matching, instead of rptha's uniform ``seq(0,1,len=npv)``
    applied identically to every row.

    NOT part of rptha. Used only as one of the repair candidates of
    :func:`discretized_source_from_contours_optimal`, and only when rptha's
    own mesh comes out invalid; never by the faithful port
    (:func:`discretized_source_from_contours_orthogonal`).

    Row 0 is the uniform reference ``linspace(0,1,npv)``; each deeper row i's
    entry j is the s-value (on that row's OWN interpolator) of the point
    nearest to shallow row's point j, made strictly increasing so the
    optimiser's own validity check accepts it.
    """
    ref_s = np.linspace(0.0, 1.0, npv)
    s_matrix = np.tile(ref_s, (num_l, 1))
    dense_s = np.linspace(0.0, 1.0, n_dense)
    shallow_pts = interps[0](ref_s)

    for i in range(1, num_l):
        dense_pts = interps[i](dense_s)
        row_s = np.empty(npv)
        for j in range(npv):
            d2 = ((dense_pts[:, 0] - shallow_pts[j, 0]) ** 2
                  + (dense_pts[:, 1] - shallow_pts[j, 1]) ** 2)
            row_s[j] = dense_s[np.argmin(d2)]
        row_s[0], row_s[-1] = 0.0, 1.0
        row_s = np.maximum.accumulate(row_s)
        eps = 1.0e-6
        for j in range(1, npv):
            if row_s[j] <= row_s[j - 1]:
                row_s[j] = row_s[j - 1] + eps
        row_s = row_s / row_s[-1]
        s_matrix[i] = row_s
    return s_matrix


def _optimise_s_matrix(interps, desired_num_lines, num_l, seed=1234,
                       init="uniform", verbose=False):
    """Solve for the normalised distances defining orthogonal down-dip lines.

    Replica of the loop in ``create_downdip_lines_on_source_contours_improved``
    (rptha, ``downdip_3d_lines_on_source.R`` lines 334-418):

    - multigrid over ``floor(desired_num_lines * c(1/8, 1/4, 1/2, 1))``,
      dropping levels below 3;
    - first initial condition ``seq(0, 1, len=np)`` on every row; later
      levels interpolate the previous optimum with ``approx(prev, n=np)``;
    - free parameters are the interior columns of every row, flattened
      column-major like R's ``c(s_matrix[, 2:(np-1)])``; columns 1 and np
      stay pinned at 0 and 1;
    - main solve ``nls.lm(ftol = 1e-3)`` with nls.lm's defaults, including
      its 50-iteration cap (see :func:`_nls_lm`);
    - if that solve stopped after its first iteration (``niter == 1``), two
      perturbed re-solves, amplitude ``0.2/(np-1)`` then a quarter of that,
      each ``nls.lm(ftol = 1e-6, maxiter = 250, maxfev = 9e4)``. R draws the
      signs with ``sample(c(-1,1))`` and no seed; here a seeded generator
      keeps runs reproducible;
    - invalid (non-monotone) rows are rejected inside the residual, as R's
      ``stop()`` -> ``Inf``; no clipping or repair.

    One addition that is NOT in rptha, and can only matter on the perturbed
    re-solve path: a level result that is non-monotone, non-finite or worse
    than the level's own starting point is discarded in favour of that
    starting point (it is always valid). Found necessary in v5, when a
    badly-scaled perturbation (since removed) scrambled southamerica.

    ``init="nearest"`` (not rptha) seeds only the coarsest level from
    :func:`_nearest_point_init_s_matrix`; used only by the repair path of
    :func:`discretized_source_from_contours_optimal`.

    v7 wrapped each level in a child process with a 90 s wall-clock timeout
    and a perturbed restart on expiry. That machinery is gone: nls.lm's own
    iteration cap bounds every solve, and the grouped Jacobian makes a level
    of southamerica's size take seconds, not minutes, so the result no
    longer depends on how fast the machine is.
    """
    rng = np.random.RandomState(seed)
    fracs = np.floor(desired_num_lines * np.array([1 / 8, 1 / 4, 1 / 2, 1])).astype(int)
    fracs = fracs[fracs >= 3]
    if fracs.size == 0:
        raise ValueError("desired_unit_source_length is too small")

    new_s = None
    for npv in fracs:
        if new_s is None:
            if init == "nearest":
                s_matrix = _nearest_point_init_s_matrix(interps, npv, num_l)
            else:
                s_matrix = np.tile(np.linspace(0, 1, npv), (num_l, 1))
        else:
            # R: approx(new_s_matrix[i,], n=np) -- interpolate the previous
            # row (y vs its index) at np equally spaced index positions.
            s_matrix = np.empty((num_l, npv))
            xs_old = np.linspace(0, 1, new_s.shape[1])
            xs_new = np.linspace(0, 1, npv)
            for i in range(num_l):
                s_matrix[i] = np.interp(xs_new, xs_old, new_s[i])

        interior = slice(1, npv - 1)
        colouring = _stencil_colouring(num_l, npv)

        def residual(free, _s=s_matrix, _npv=npv):
            sm = _s.copy()
            sm[:, 1:_npv - 1] = free.reshape(num_l, _npv - 2, order="F")
            return _quality_matrix(sm, interps)

        x0 = s_matrix[:, interior].ravel(order="F")
        best, niter = _nls_lm(residual, x0, colouring, ftol=1e-3)
        if verbose:
            print(f"      level npv={npv}: nls.lm niter={niter}", flush=True)
        if niter == 1:
            ascale = 0.2 / (npv - 1)
            for sc in (ascale, ascale / 4):
                pert = best + sc * rng.choice([-1.0, 1.0], size=best.size)
                best, _ = _nls_lm(residual, pert, colouring, ftol=1e-6,
                                  maxiter=250, maxfev=90000)

        new_s = s_matrix.copy()
        new_s[:, interior] = best.reshape(num_l, npv - 2, order="F")

        start_cost = float(np.sum(_quality_matrix(s_matrix, interps) ** 2))
        new_cost = float(np.sum(_quality_matrix(new_s, interps) ** 2))
        monotone = bool(np.all(np.diff(new_s, axis=1) >= 0))
        if (not monotone) or (not np.isfinite(new_cost)) or new_cost > start_cost:
            why = ("non-monotone" if not monotone
                   else f"cost {new_cost:.6g} > start {start_cost:.6g}")
            print(f"    note: mesh optimiser rejected its own result at "
                  f"npv={npv} ({why}); keeping this level's starting grid",
                  flush=True)
            new_s = s_matrix

    return new_s


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def interpolate_gc_path(surface_path, n=50):
    """n+2 points along the great circle between two (lon, lat) points.

    Port of ``geometric_util.R::interpolate_gc_path`` (which wraps
    geosphere::gcIntermediate with addStartEnd=TRUE), including the
    longitude-continuity correction for paths crossing the dateline.
    Uses _GEOD_SPHERE (spherical), NOT _GEOD (WGS84-ellipsoidal): geosphere
    has no ellipsoidal gcIntermediate, so R's own path here is spherical --
    see _GEOD_SPHERE's own comment for the measured impact of getting this
    wrong.
    """
    p1, p2 = np.asarray(surface_path, dtype=float)[:2]
    inter = _GEOD_SPHERE.npts(p1[0], p1[1], p2[0], p2[1], n)
    path = np.array([[p1[0], p1[1]]] + inter + [[p2[0], p2[1]]])

    jumps = np.diff(path[:, 0])
    if np.any(np.abs(jumps) > 180):
        offset = np.concatenate([[0.0], jumps * (np.abs(jumps) > 180)])
        offset = np.round(offset / 360.0) * 360.0
        path[:, 0] = path[:, 0] - np.cumsum(offset)
    return path


def interpolate_3D_path(threeD_path, n=100, depth_in_km=True, ndense=None):
    """n points along a 3D (lon, lat, depth) path, equally spaced by 3D arc length.

    Faithful port of ``geometric_util.R::interpolate_3D_path``: cumulative
    down-dip distances via :func:`unit_sources.distance_down_depth`, a dense
    great-circle path between every vertex pair (depth and distance linear
    along each segment), then linear interpolation of lon/lat/depth against
    the cumulative distance at ``n`` equally spaced distances.
    """
    from .unit_sources import distance_down_depth

    path = np.asarray(threeD_path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("threeD_path must have 3 columns")
    if path.shape[0] < 2:
        raise ValueError("threeD_path must have at least 2 points")
    if ndense is None:
        ndense = n * 10

    npts = path.shape[0]
    increments = np.zeros(npts)
    for i in range(1, npts):
        increments[i] = distance_down_depth(path[i - 1], path[i],
                                            depth_in_km=depth_in_km)
    path_distance = np.cumsum(increments)

    # Dense path: [lon, lat, depth, cumulative_distance] rows.
    fine = [np.concatenate([path[0], [path_distance[0]]])]
    for i in range(1, npts):
        seg2d = interpolate_gc_path(path[i - 1:i + 1, :2], n=ndense)[1:ndense + 1]
        seg_dep = np.linspace(path[i - 1, 2], path[i, 2], ndense + 2)[1:ndense + 1]
        seg_dst = np.linspace(path_distance[i - 1], path_distance[i],
                              ndense + 2)[1:ndense + 1]
        fine.append(np.column_stack([seg2d, seg_dep, seg_dst]))
        fine.append(np.concatenate([path[i], [path_distance[i]]]))
    fine = np.vstack([np.atleast_2d(f) for f in fine])

    # R's approx(..., n=n) evaluates at n equally spaced x over the x-range.
    xout = np.linspace(fine[0, 3], fine[-1, 3], n)
    out = np.column_stack([np.interp(xout, fine[:, 3], fine[:, k])
                           for k in range(3)])
    return out


def discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length, desired_unit_source_width=None,
        n_downdip=None, seed=1234, _init="uniform", min_downdip=None,
        n_alongstrike=None):
    """Build a unit_source_grid with rptha-style orthogonal down-dip lines.

    Follows ``discretized_source_from_source_contours``: optimised down-dip
    lines crossing every contour, then each down-dip transect is resampled at
    equal 3D arc-length spacing with :func:`interpolate_3D_path`, exactly as
    rptha does (NOT by uniform depth).

    Parameters
    ----------
    contours : list of (depth_km, coords) pairs
        ``coords`` is an (N, 2) array of (lon, lat). Order is irrelevant.
    desired_unit_source_length : float
        Target along-strike unit-source length (km); sets the number of
        down-dip lines from the top-contour length.
    desired_unit_source_width : float, optional
        Target down-dip unit-source width (km). Sets the number of rows via
        rptha's rule: ``max(round(mean_transect_length/width - 1), 0) + 2``
        grid rows. Defaults to ``desired_unit_source_length``. Ignored if
        ``n_downdip`` is given.
    n_downdip : int, optional
        Number of unit sources down dip (overrides the width-based rule).
    seed : int
        Seed for the perturbation retries (reproducibility).
    min_downdip : int, optional
        NOT rptha: if the width-based rule gives fewer rows than this, use
        this many instead. PTHA18 did the same thing by hand on the one zone
        it applied to (ReportPTHA.pdf p.14: Puysegur's width was cut to
        35 km "to ensure we had 2 rows of unit-sources in the down-dip
        direction"). None (default) is the faithful rule.
    n_alongstrike : int, optional
        NOT rptha (v8.2): number of unit sources along strike, overriding
        the top-contour rule above, the way ``n_downdip`` overrides the row
        rule. Used only by ``column_rule="average"`` of
        :func:`discretized_source_from_contours_optimal`. None (default) is
        the faithful rule.
    _init : str
        Internal -- "uniform" (default) is the faithful rptha port. Callers
        should not pass "nearest" directly; use
        :func:`discretized_source_from_contours_optimal` instead, which sets
        it and documents why.

    Returns
    -------
    unit_source_grid : ndarray, shape (n_downdip+1, 3, n_alongstrike+1)
    """
    from .unit_sources import distance_down_depth

    interps, depths = _build_interpolators(contours)
    num_l = len(interps)
    if num_l < 2:
        raise ValueError("need at least two depth contours")
    if desired_unit_source_width is None:
        desired_unit_source_width = desired_unit_source_length

    # Number of along-strike lines from the top-contour length. R computes it
    # with sp::SpatialLinesLengths(longlat=TRUE) over the RAW contour
    # vertices — an ellipsoidal distance (matches WGS84 geodesic to ~1e-5
    # relative, e.g. Puysegur: sp 817.0268 km vs WGS84 817.0225 km), far
    # inside the ceiling() rule's slack.
    top_raw = np.asarray(min(contours, key=lambda c: float(c[0]))[1], dtype=float)
    top_len_km = sum(
        _GEOD.inv(top_raw[k, 0], top_raw[k, 1],
                  top_raw[k + 1, 0], top_raw[k + 1, 1])[2]
        for k in range(len(top_raw) - 1)) / 1000.0
    desired_num_lines = int(np.ceil(top_len_km / desired_unit_source_length + 1))
    if n_alongstrike is not None:
        desired_num_lines = int(n_alongstrike) + 1

    s_matrix = _optimise_s_matrix(interps, desired_num_lines, num_l, seed=seed,
                                  init=_init)
    npv = s_matrix.shape[1]

    # Contour crossings -> down-dip transects (one 3D path per along-strike line).
    x, y = _get_xy(s_matrix, interps)
    depths_sorted = np.asarray(sorted(depths), dtype=float)
    transects = [np.column_stack([x[:, i], y[:, i], depths_sorted])
                 for i in range(npv)]

    # rptha: number of rows from the mean 3D transect length and desired width.
    if n_downdip is None:
        lengths_m = [sum(distance_down_depth(t[k], t[k + 1])
                         for k in range(num_l - 1)) for t in transects]
        mean_len_km = float(np.mean(lengths_m)) / 1000.0
        number_of_mid_lines = max(
            round(mean_len_km / desired_unit_source_width - 1), 0)
        rows = number_of_mid_lines + 2
        if min_downdip is not None and rows - 1 < min_downdip:
            print(f"    down-dip rows: the {desired_unit_source_width:g} km width "
                  f"rule gives {rows - 1} on a {mean_len_km:.0f} km mean "
                  f"transect; using {min_downdip} (PTHA18's minimum, "
                  f"ReportPTHA p.14)", flush=True)
            rows = min_downdip + 1
    else:
        rows = n_downdip + 1

    # rptha: resample each transect at equal 3D arc-length spacing.
    grid = np.empty((rows, 3, npv))
    for i, t in enumerate(transects):
        pts = interpolate_3D_path(t, n=rows, depth_in_km=True)
        grid[:, 0, i] = pts[:, 0]
        grid[:, 1, i] = pts[:, 1]
        grid[:, 2, i] = pts[:, 2]
    return grid


# ---------------------------------------------------------------------------
# "optimal": rptha's own mesh, with repairs used only if it is broken.
# NOT part of rptha (--discretizer optimal).
# ---------------------------------------------------------------------------


def _cumulative_km(pts):
    """Cumulative along-line distance (km) for a dense (N, 2) lon/lat path."""
    seg = np.array([_dist_haversine(pts[k], pts[k + 1])
                    for k in range(len(pts) - 1)]) / 1000.0
    return np.concatenate([[0.0], np.cumsum(seg)])


def _arc_fraction_of(pt, dense_pts, cum_km):
    """Where `pt` projects onto a dense path, as a fraction of its length."""
    d2 = ((dense_pts[:, 0] - pt[0]) ** 2 + (dense_pts[:, 1] - pt[1]) ** 2)
    return float(cum_km[int(np.argmin(d2))] / cum_km[-1])


def align_contour_ends(contours, n_dense=4000):
    """Trim every contour back to the along-strike span ALL levels share.

    This is the fix behind ``--discretizer optimal``, and it addresses the
    real cause of the folded tip cells -- which is not the optimiser.

    rptha's optimiser holds ``s = 0`` and ``s = 1`` fixed on EVERY contour
    row; only interior columns are free parameters (see
    ``create_downdip_lines_on_source_contours_improved``, whose
    ``moving_par`` is ``s_matrix[, 2:(np-1)]``). So the first and last
    down-dip column are *required* to join the contours' own endpoints,
    whatever those are. When SLAB resolves the shallow interface over a
    shorter along-strike span than the deep one, those endpoints are far
    apart ALONG the arc, and the edge column is forced to run along the
    arc instead of across it. Measured on kermadectonga2 (--ptha=false):
    the last column deviates 89.7 deg from orthogonal and is 209 km long
    against ~107 km mid-arc -- the visible southern-tip spike.

    Measured end gaps, shallow contour's endpoint to each deeper one's:

        kermadectonga2   start 121 km   end 209 km   (asymmetric -> folds)
        kurilsjapan      start 147 km   end 146 km   (symmetric  -> clean)
        puysegur2        start  33 km   end  57 km   (asymmetric -> bowtie)

    kurilsjapan is the control: its ends are near-symmetric, its mesh has
    no fold, and this function leaves it essentially untouched.

    Method: give every contour an along-arc coordinate, project each
    contour's two endpoints onto every other contour, and keep only the
    sub-arc inside the innermost projection at each end. All levels then
    start and finish on a common across-strike transect, so the edge
    columns come out orthogonal like the interior ones.

    Note this trims *along strike* only. The across-strike span (the
    down-dip width, which is what carries the area) is untouched, so the
    area cost is small and bounded by how badly the ends disagreed:
    measured -3.7% on kermadectonga2, -4.8% on puysegur2, ~0% on
    kurilsjapan. Discarding the folded columns outright instead costs
    -24% and -14% on the same two zones.
    """
    interps, depths = _build_interpolators(contours)
    n = len(interps)
    ds = np.linspace(0.0, 1.0, n_dense)
    dense = [f(ds) for f in interps]
    cum = [_cumulative_km(p) for p in dense]

    lo_frac = np.zeros(n)
    hi_frac = np.ones(n)
    for i in range(n):
        starts = [_arc_fraction_of(dense[k][0], dense[i], cum[i])
                  for k in range(n)]
        ends = [_arc_fraction_of(dense[k][-1], dense[i], cum[i])
                for k in range(n)]
        lo_frac[i] = max(0.0, max(starts))
        hi_frac[i] = min(1.0, min(ends))
        if hi_frac[i] - lo_frac[i] < 0.5:
            # Refuse to cut more than half the arc away: that would mean
            # the levels barely overlap, which is a data problem (a bad
            # contour extraction), not something to silently paper over.
            raise ValueError(
                f"align_contour_ends: contour at {depths[i]:g} km would keep "
                f"only {100*(hi_frac[i]-lo_frac[i]):.0f}% of its length; the "
                f"depth levels barely overlap along strike -- check step 1's "
                f"contour extraction for this zone")

    out = []
    for i, (d, _) in enumerate(sorted(contours, key=lambda c: float(c[0]))):
        ss = np.linspace(lo_frac[i], hi_frac[i], n_dense)
        out.append((d, interps[i](ss)))
    return out


def _cell_polygons(grid):
    nr, _, nc = grid.shape
    for r in range(nr - 1):
        for j in range(nc - 1):
            yield [(grid[r, 0, j], grid[r, 1, j]),
                   (grid[r, 0, j + 1], grid[r, 1, j + 1]),
                   (grid[r + 1, 0, j + 1], grid[r + 1, 1, j + 1]),
                   (grid[r + 1, 0, j], grid[r + 1, 1, j])]


def mesh_defects(grid):
    """What would make a unit-source grid unusable, counted per cell.

    * ``bowties``: cells whose quadrilateral self-intersects (a folded cell);
    * ``degenerate``: cells with less than 2% of the median cell area (a
      column collapsed onto its neighbour);
    * ``overlap_fraction``: how much of the summed cell area is covered twice
      (two columns crossing without either cell being a bow-tie);
    * ``nonfinite``: grid nodes that are NaN or infinite.

    ``total`` is the number of defective cells, plus 1 if the overlap
    exceeds 0.1%. Areas are compared in a local projection (longitude scaled
    by cos(mean latitude)); only ratios are used.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if not np.all(np.isfinite(grid)):
        n = int(np.sum(~np.isfinite(grid)))
        return {"bowties": 0, "degenerate": 0, "overlap_fraction": 0.0,
                "nonfinite": n, "total": n, "area_proj": 0.0}
    lat0 = np.radians(np.mean(grid[:, 1, :]))
    polys, bow = [], 0
    for q in _cell_polygons(grid):
        pq = Polygon([(lo * np.cos(lat0), la) for lo, la in q])
        if not pq.is_valid:
            bow += 1
            pq = pq.buffer(0)
        polys.append(pq)
    areas = np.array([abs(p.area) for p in polys])
    med = float(np.median(areas)) if areas.size else 0.0
    degen = int(np.sum(areas < 0.02 * med)) if med > 0 else 0
    tot = float(areas.sum())
    union = float(unary_union(polys).area) if polys else 0.0
    overlap = max(0.0, 1.0 - union / tot) if tot > 0 else 0.0
    total = bow + degen + (1 if overlap > 1e-3 else 0)
    return {"bowties": bow, "degenerate": degen, "overlap_fraction": overlap,
            "nonfinite": 0, "total": total, "area_proj": tot}


# ---------------------------------------------------------------------------
# v8.2: the column rule (--columns trench|average) and the mesh-shape check.
# ---------------------------------------------------------------------------

# Measured on PTHA18's 43 published unit-source meshes (NCI
# unit_source_statistics_<zone>.nc), with mesh_shape() below: the largest
# taper (sandwich), the widest range of column widths (alaskaaleutians) and
# the median of the mean deviation from 50 x 50. Printed by step 2 for
# comparison; they do not change any mesh.
PTHA18_MAX_TAPER = 1.17
PTHA18_MAX_WIDTH_RATIO = 3.26
PTHA18_MEDIAN_DEVIATION = 0.21


def mesh_shape(grid, target_km=50.0):
    """How far a unit-source grid is from cells of target_km x target_km.

    * ``taper``: median over columns of (trench-row cell length / deepest-row
      cell length). 1 = the cells keep their along-strike length with depth.
    * ``width_ratio``: widest column's full down-dip width / narrowest one's.
    * ``near_pct``: % of cells whose length and width are both 30-70 km.
    * ``mean_deviation``: mean over cells of max(|log2(L/50)|, |log2(W/50)|);
      0 = every cell 50 x 50, 1 = a typical cell has a side of 25 or 100 km.
    """
    from .unit_sources import discretized_source_approximate_summary_statistics
    s = discretized_source_approximate_summary_statistics(grid)
    nd, na = grid.shape[0] - 1, grid.shape[2] - 1
    L, W = np.asarray(s["length"]), np.asarray(s["width"])
    Lm, Wm = L.reshape(na, nd), W.reshape(na, nd)
    dev = np.maximum(np.abs(np.log2(L / target_km)), np.abs(np.log2(W / target_km)))
    cw = Wm.sum(axis=1)
    return {"n_alongstrike": na, "n_downdip": nd,
            "taper": float(np.median(Lm[:, 0] / Lm[:, -1])) if nd > 1 else 1.0,
            "width_ratio": float(cw.max() / cw.min()),
            "near_pct": float(100.0 * np.mean((L > 30) & (L < 70) & (W > 30) & (W < 70))),
            "mean_deviation": float(np.mean(dev)),
            "median_length_km": float(np.median(L)), "median_width_km": float(np.median(W))}


def _mean_row_length_km(grid):
    """Average 3D along-strike length of the grid's cell rows."""
    from .unit_sources import distance_down_depth
    rl = np.array([sum(distance_down_depth(grid[r, :, j], grid[r, :, j + 1])
                       for j in range(grid.shape[2] - 1))
                   for r in range(grid.shape[0])]) / 1000.0
    return float(np.mean(0.5 * (rl[:-1] + rl[1:])))


def discretized_source_from_contours_optimal(
        contours, desired_unit_source_length, desired_unit_source_width=None,
        n_downdip=None, seed=1234, verbose=True, min_downdip=None,
        column_rule="trench"):
    """The "optimal" mesh (see :func:`_optimal_core`), with a column rule (v8.2).

    ``column_rule`` sets how many columns (unit sources along strike) the
    mesh gets; everything else is unchanged:

    * ``"trench"`` (default): rptha's and PTHA18's rule, the trench length
      divided by the desired length, rounded up. The v8 mesh, bit for bit.
    * ``"average"``: the AVERAGE row length divided by the desired length,
      rounded up: the same rule applied to the average cell row instead of
      the trench row, the way rptha already counts the rows from the average
      down-dip width. NOT PTHA18's procedure; meant for zones whose mesh
      tapers far more than any PTHA18 mesh (step 2 prints the taper next to
      PTHA18's maximum, 1.17).

    Why ``"average"`` exists: every row of an rptha mesh has the same number
    of cells, so on a tight arc the deep cells are much shorter than the
    trench cells. On calabria2 (trench 980 km, 60 km contour 243 km) the
    trench rule gives 20 columns: about 50 km at the trench and 12 km at
    depth. The average row gives 12 columns: about 50 km in the middle rows
    (median cell length 26 -> 45 km). It cannot remove the taper, only centre
    it, and it does nothing for a zone whose width changes along strike
    (caribbean2), because every column keeps the same rows.
    """
    if column_rule not in ("trench", "average"):
        raise ValueError(f"column_rule must be 'trench' or 'average', not {column_rule!r}")
    kw = dict(desired_unit_source_width=desired_unit_source_width,
              n_downdip=n_downdip, seed=seed, verbose=verbose,
              min_downdip=min_downdip)
    base = _optimal_core(contours, desired_unit_source_length, **kw)
    if column_rule == "trench":
        return base
    n_base = base.shape[2] - 1
    mean_row = _mean_row_length_km(base)
    n = max(1, int(np.ceil(mean_row / desired_unit_source_length)))
    if verbose:
        print(f"    column rule 'average': average row {mean_row:.0f} km / "
              f"{desired_unit_source_length:g} km -> {n} columns (rptha's "
              f"trench rule: {n_base})", flush=True)
    if n == n_base:
        return base
    return _optimal_core(contours, desired_unit_source_length, n_alongstrike=n, **kw)


def strasser_cell_cap(cap_mw, k, relation="Strasser"):
    """(max length, max width) of a cell in km for ``--cell-size strasser``:
    the scaling relation's rupture at ``cap_mw`` (the smallest magnitude the
    run builds) divided by ``k``. Strasser at Mw 7.2: 54.3 x 44.2 km, so
    k = 1.5 gives 36.2 x 29.5 km and k = 2 gives 27.2 x 22.1 km."""
    from .scaling import Mw_2_rupture_size
    if not k > 0:
        raise ValueError(f"cell k must be positive, not {k!r}")
    size = Mw_2_rupture_size(float(cap_mw), relation=relation)
    return size["length"] / k, size["width"] / k


def discretized_source_from_contours_bounded(
        contours, max_length, max_width, seed=1234, verbose=True,
        min_downdip=None, max_tries=40):
    """v12 ``--cell-size strasser``: the "optimal" mesh with NO cell longer
    than ``max_length`` or wider than ``max_width`` (km).

    rptha's rule (``--cell-size mean``) counts the columns from the trench
    length and the rows from the MEAN down-dip length, so on a fan-shaped
    zone (calabria2: trench 981 km, 60 km contour 243 km) or one whose width
    changes along strike (antilles2, alaskaaleutians) some cells come out far
    larger than the target: calabria2's trench cells reach 4042 km2, more
    than a whole Mw 7.2 rupture, and alaskaaleutians' widest cells 97 km. This
    keeps the same mesher and the same rows x columns structure and only asks
    for more of them: it starts from rptha's rule with the caps as targets,
    then raises the row count until the widest cell (v12's own unit-source
    width) meets ``max_width`` and the column count until the longest cell
    meets ``max_length`` (2% tolerance), re-meshing each time.

    The cells stay unequal (a rows x columns mesh has as many columns at
    depth as at the trench), but none is larger than the cap, so local-size
    ruptures (``rupture_size="local"``) and the HS/VAUS footprints can match
    the scaling relation everywhere. Validated on 10 zones in
    ``examples_newtest/`` (see its html/report_strasser_meshes.html).
    """
    from .unit_sources import discretized_source_approximate_summary_statistics as stats

    def dims(g):
        s = stats(g)
        return float(np.max(s["length"])), float(np.max(s["width"]))

    grid = discretized_source_from_contours_optimal(
        contours, max_length, max_width, n_downdip=None, seed=seed,
        verbose=False, min_downdip=min_downdip, column_rule="trench")
    ncol, nrow = grid.shape[2] - 1, grid.shape[0] - 1
    for tries in range(1, max_tries + 1):
        L, W = dims(grid)
        if verbose:
            print(f"    cell cap: {ncol} x {nrow} cells, longest {L:.1f} km "
                  f"(cap {max_length:.1f}), widest {W:.1f} km (cap {max_width:.1f})",
                  flush=True)
        ok_l, ok_w = L <= max_length * 1.02, W <= max_width * 1.02
        if ok_l and ok_w:
            return grid
        if not ok_w:
            nrow = max(nrow + 1, int(np.ceil(nrow * W / max_width)))
        if not ok_l:
            ncol = max(ncol + 1, int(np.ceil(ncol * np.sqrt(L / max_length))))
        grid = _optimal_core(contours, max_length, desired_unit_source_width=max_width,
                             n_downdip=nrow, seed=seed, verbose=False,
                             min_downdip=min_downdip, n_alongstrike=ncol)
    raise ValueError(f"cell cap not met after {max_tries} meshes "
                     f"(last {ncol} x {nrow}: longest {L:.1f} km, widest {W:.1f} km)")


def _optimal_core(
        contours, desired_unit_source_length, desired_unit_source_width=None,
        n_downdip=None, seed=1234, verbose=True, min_downdip=None,
        n_alongstrike=None):
    """rptha's own mesh, repaired only if it is actually broken (v8).

    1. Build rptha's mesh with
       :func:`discretized_source_from_contours_orthogonal`, which reproduces
       rptha's R ``create_downdip_lines_on_source_contours_improved`` +
       ``discretized_source_from_source_contours`` to the metre (see
       :func:`_nls_lm`). On PTHA18's own contours it reproduces the official
       PTHA18 grid on all 14 zones tested (identical dimensions, area within
       0.13%).
    2. If :func:`mesh_defects` finds nothing wrong with it, return it
       UNCHANGED. That is the mesh PTHA18's automatic method gives.
    3. Only if it is broken (folded, collapsed or overlapping cells), try
       repairs in order of how little they depart from rptha, and return the
       first defect-free one:
         a. the same contours, optimiser started from nearest-point matching
            (:func:`_nearest_point_init_s_matrix`) instead of rptha's uniform
            start: same objective, different starting guess;
         b. contours trimmed to the along-strike span all levels share
            (:func:`align_contour_ends`), uniform start;
         c. both.
       If none is defect-free, the one with the fewest defects (ties: the
       larger area) is returned, with a printed warning.

    Why this replaces the v4-v7 rule. v4-v7 always built both the plain and
    the end-aligned mesh and kept the aligned one whenever its worst column
    was less skewed, even when the plain mesh had no defect at all. On
    PTHA18's own contours that rule moved away from PTHA18 on every zone it
    touched: cascadia lost 6 of its 22 columns (-13.8% area), puysegur2
    -2.5%, solomon2 -1.5%, izumariana -1.1%. The folded tips that motivated
    it came from v3-v7's fragmented contours (see lib/slab_contours.py), not
    from rptha's method.
    """
    kw = dict(desired_unit_source_width=desired_unit_source_width,
              n_downdip=n_downdip, seed=seed, min_downdip=min_downdip,
              n_alongstrike=n_alongstrike)
    base = discretized_source_from_contours_orthogonal(
        contours, desired_unit_source_length, **kw)
    d0 = mesh_defects(base)
    if d0["total"] == 0:
        if verbose:
            print("    optimal: rptha's own mesh has no folded, collapsed or "
                  "overlapping cells -> kept unchanged", flush=True)
        return base

    if verbose:
        print(f"    optimal: rptha's own mesh has defects ({d0['bowties']} "
              f"bow-tied, {d0['degenerate']} collapsed, overlap "
              f"{100 * d0['overlap_fraction']:.2f}%) -> trying repairs",
              flush=True)
    tried = [("rptha", base, d0)]
    try:
        aligned = align_contour_ends(contours)
    except ValueError as exc:
        aligned = None
        if verbose:
            print(f"    optimal: end alignment not applicable ({exc})")
    plans = [("rptha + nearest-point start", contours, "nearest"),
             ("end-aligned contours", aligned, "uniform"),
             ("end-aligned + nearest-point start", aligned, "nearest")]
    for name, cont, init in plans:
        if cont is None:
            continue
        g = discretized_source_from_contours_orthogonal(
            cont, desired_unit_source_length, _init=init, **kw)
        d = mesh_defects(g)
        tried.append((name, g, d))
        if verbose:
            print(f"    optimal: {name}: {d['bowties']} bow-tied, "
                  f"{d['degenerate']} collapsed, overlap "
                  f"{100 * d['overlap_fraction']:.2f}%", flush=True)
        if d["total"] == 0:
            if verbose:
                print(f"    optimal: using '{name}'", flush=True)
            return g
    name, g, d = min(tried, key=lambda t: (t[2]["total"], -t[2]["area_proj"]))
    print(f"    WARNING optimal: no repair removed every defect; using "
          f"'{name}' ({d['total']} defective cell(s)). Inspect the mesh "
          f"figure before trusting this zone.", flush=True)
    return g


# ---------------------------------------------------------------------------
# The deprecated discretiser (--discretizer mid)
# Port of rptha's create_downdip_lines_on_source_contours(), used by
# discretized_source_from_source_contours(improved_downdip_lines = FALSE).
# Superseded by the orthogonal/_improved() method above and not called by
# PTHA18's own template (make_initial_downdip_lines.R calls only _improved).
# Kept for direct mesh comparison against the LM/orthogonal method.
# ---------------------------------------------------------------------------


def _resample_geodesic_by_spacing(coords, n_out):
    """n_out evenly (geodesic-)spaced points along a polyline.

    Port of rptha's ``approxSpatialLines(..., longlat=TRUE)``: total arc
    length is divided into n_out-1 equal segments and lon/lat are each
    linearly interpolated against cumulative distance, as
    ``approx(seg_lnth, coords, out_lnth, rule=2)`` does. Lengths are WGS84
    geodesic, like ``sp::LineLength(longlat=TRUE)``.
    """
    coords = np.asarray(coords, dtype=float)
    seg = np.array([_GEOD.inv(coords[k, 0], coords[k, 1],
                              coords[k + 1, 0], coords[k + 1, 1])[2]
                    for k in range(len(coords) - 1)]) / 1000.0
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    xout = np.linspace(0.0, dist[-1], n_out)
    lon = np.interp(xout, dist, coords[:, 0])
    lat = np.interp(xout, dist, coords[:, 1])
    return np.column_stack([lon, lat])


def _extend_line(coords, frac):
    """R's extend_line_fraction for a contour (geometric_util.R lines
    188-198): prepend/append a point along the 'endpoints + midpoint'
    direction, scaled by `frac`."""
    c = np.asarray(coords, dtype=float)
    lc = len(c)
    lch = max(lc // 2, 2) - 1                      # R index max(floor(lc/2),2)
    dx1 = (c[0] - c[lch]) * 2
    lch = min(lch, lc - 2)                         # R: min(lch, lc - 1)
    dx2 = (c[-1] - c[lch]) * 2
    return np.vstack([c[0] + dx1 * frac, c, c[-1] + dx2 * frac])


def _intersect_dipcut_with_contours(p1, p2, contours_sorted, n=1000,
                                    extend_line_fraction=1.0e-6):
    """Port of ``intersect_surface_path_with_depth_contours``.

    The straight cut p1 -> p2 (extended by `extend_line_fraction` at both
    ends, as are the contours) is densified along its great circle with
    ``interpolate_gc_path(n = 1000)`` and intersected with every ORIGINAL
    contour polyline. R stops with "Non-unique contour levels" if a contour
    is crossed more than once; so does this port. A contour the cut misses
    simply contributes no point, as in R.
    """
    from shapely.geometry import LineString

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    q1, q2 = p1 - d * extend_line_fraction, p2 + d * extend_line_fraction
    cut = LineString(interpolate_gc_path(np.array([q1, q2]), n=n))
    pts = []
    for depth, coords in contours_sorted:
        inter = cut.intersection(LineString(_extend_line(coords, extend_line_fraction)))
        if inter.is_empty:
            continue
        geoms = list(getattr(inter, "geoms", [inter]))
        if len(geoms) != 1 or geoms[0].geom_type != "Point":
            raise ValueError(
                f"deprecated discretiser: the cut from {tuple(np.round(p1, 3))} "
                f"to {tuple(np.round(p2, 3))} crosses the {depth:g} km contour "
                f"{len(geoms)} times (R stops here with 'Non-unique contour "
                f"levels in intersection')")
        pts.append((geoms[0].x, geoms[0].y, float(depth)))
    pts.sort(key=lambda r: r[2])
    return np.array(pts)


def discretized_source_from_contours_mid(
        contours, desired_unit_source_length, desired_unit_source_width=None,
        n_downdip=None, min_downdip=None):
    """Unit-source grid with rptha's deprecated down-dip lines.

    Port of ``create_downdip_lines_on_source_contours()`` as rptha actually
    runs it. That routine builds three candidate cuts per column ('eq_spacing',
    'nearest', 'mid') but then uses ``down_dip_line_type = 'eq_spacing'``: the
    ``'mid'`` assignment is commented out in the R source
    (downdip_3d_lines_on_source.R, lines 583-584). So every cut joins the i-th
    equally spaced point of the shallowest contour to the i-th equally spaced
    point of the deepest one, and is then intersected with every contour
    (:func:`_intersect_dipcut_with_contours`). v4-v7's port used the 'mid'
    compromise instead and said R did; v8 follows R. The CLI keeps the name
    ``--discretizer mid`` for compatibility.

    Not what PTHA18 used (its template calls the _improved method); kept for
    comparison. ``orthogonal_near_trench`` (default FALSE in R) is not ported.
    ``min_downdip`` as in :func:`discretized_source_from_contours_orthogonal`.
    """
    from .unit_sources import distance_down_depth

    contours_sorted = sorted(((float(d), np.asarray(c, dtype=float)[:, :2])
                              for d, c in contours), key=lambda t: t[0])
    if len(contours_sorted) < 2:
        raise ValueError("need at least two depth contours")
    if desired_unit_source_width is None:
        desired_unit_source_width = desired_unit_source_length

    shallow_raw = contours_sorted[0][1]
    deep_raw = contours_sorted[-1][1]
    top_len_km = sum(
        _GEOD.inv(shallow_raw[k, 0], shallow_raw[k, 1],
                  shallow_raw[k + 1, 0], shallow_raw[k + 1, 1])[2]
        for k in range(len(shallow_raw) - 1)) / 1000.0
    npv = int(np.ceil(top_len_km / desired_unit_source_length)) + 1
    shallow = _resample_geodesic_by_spacing(shallow_raw, npv)
    deep = _resample_geodesic_by_spacing(deep_raw, npv)

    # R: reverse the deep line if its first point is not the one nearest the
    # shallow line's first point (distCosine, a sphere)
    if _dist_haversine(shallow[0], deep[0]) > _dist_haversine(shallow[0], deep[-1]):
        deep = deep[::-1]
    # R: make sure the lines run along strike (b1 + 90 should be near b2)
    mi = max(npv // 2, 2) - 1
    b1 = _bearing(shallow[0], shallow[mi])
    b2 = _bearing(shallow[0], deep[0])
    angle_diff = (b2 - (b1 + 90.0)) % 360.0
    if not (angle_diff < 90.0 or angle_diff > 270.0):
        shallow = shallow[::-1]
        deep = deep[::-1]

    transects = [_intersect_dipcut_with_contours(shallow[i], deep[i], contours_sorted)
                 for i in range(npv)]

    lengths_km = [sum(distance_down_depth(t[k], t[k + 1]) for k in range(len(t) - 1)) / 1000.0
                  for t in transects]
    if n_downdip is None:
        mean_len_km = float(np.mean(lengths_km))
        rows = max(round(mean_len_km / desired_unit_source_width - 1), 0) + 2
        if min_downdip is not None and rows - 1 < min_downdip:
            rows = min_downdip + 1
    else:
        rows = n_downdip + 1

    grid = np.empty((rows, 3, npv))
    for i, t in enumerate(transects):
        pts = interpolate_3D_path(t, n=rows, depth_in_km=True)
        grid[:, 0, i] = pts[:, 0]
        grid[:, 1, i] = pts[:, 1]
        grid[:, 2, i] = pts[:, 2]
    return grid
