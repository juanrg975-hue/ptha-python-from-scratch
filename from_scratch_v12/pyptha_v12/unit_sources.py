"""Unit-source geometry: grid -> per-subfault summary statistics.

Python port of the core of ``rptha/R/unit_sources.R``. The part that a
new-zone rupture/rate workflow actually consumes is the summary-statistics
table (centroid lon/lat/depth, strike, dip, length, width, area per unit
source). That is ported here as
:func:`discretized_source_approximate_summary_statistics`, a faithful
transcription of the R routine of the same name, with the geodesy done via
``pyproj``: spherical (r=6378137, f=0) wherever R's ``geosphere`` calls
(``distHaversine``, ``areaPolygon(..., f=0)``) are spherical -- see
``_GEOD_SPHERE`` below -- and WGS84-ellipsoidal (``_GEOD``) only where R
leaves geosphere at its ellipsoidal default: ``midPoint`` (for the strike).
v4 used the WGS84 ellipsoid for the spherical calls too, an undocumented
divergence from R fixed in v5 -- see from_scratch_v12/html/docs/code_map.html; v5-v7
then made midPoint spherical as well, which v8 reverts (see _midpoint).
Checked cell by cell against PTHA18's published unit_source_statistics
tables (v8): every field agrees to <= 4e-10 on 9 zones.

Scope note (deliberate): the full ``discretized_source_from_source_contours``
pipeline (shapefile contours -> interpolated down-dip lines -> 3D grid) rests
on several thousand lines of spherical-geometry helpers
(``downdip_3d_lines_on_source.R``, ``contour_interpolator.R``,
``geometric_util.R``). Porting all of that is out of proportion to the task,
so instead this module works from a ``unit_source_grid`` that is either
(a) loaded from an existing rptha discretisation, or
(b) built for an idealised planar/regular interface with
    :func:`make_planar_unit_source_grid`.
Both feed the same summary-statistics routine that events and rates need.

``unit_source_grid`` layout matches rptha: a float array of shape
``(n_downdip + 1, 3, n_alongstrike + 1)`` whose ``[j, :, i]`` entry is the
(lon, lat, depth) vertex at down-dip index ``j`` and along-strike index ``i``.
Depths are in km by default.
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")
# rptha's geosphere calls (distHaversine, areaPolygon(..., f=0)) are all
# SPHERICAL, r=6378137 -- not the WGS84 ellipsoid _GEOD above. Used only by
# _dist_haversine/_area_polygon below (which feed dip and width), matching
# the sphere contour_discretisation.py's own _dist_haversine already uses.
# See from_scratch_v12/html/docs/code_map.html: v4 used _GEOD (ellipsoidal) for these two,
# an undocumented, real divergence from R -- fixed here.
_GEOD_SPHERE = Geod(a=6378137.0, f=0.0)
_R = 6378137.0  # Earth radius (m), matching rptha's default
_DEG2RAD = np.pi / 180.0


# ---------------------------------------------------------------------------
# Geodesy helpers (port of the geosphere/geometric_util pieces we use)
# ---------------------------------------------------------------------------


def _midpoint(p1, p2):
    """Geodesic midpoint of two (lon, lat) points, degrees, on the WGS84
    ELLIPSOID, exactly geosphere::midPoint as rptha calls it (unit_sources.R
    line 404, no f= argument). geosphere's midPoint signature is
    ``midPoint(p1, p2, a = 6378137, f = 1/298.257223563)``: inverse geodesic,
    then destPoint half way along it. v5-v7 used a sphere here on the belief
    that midPoint had no f= argument; with the ellipsoid the unit-source
    strike matches PTHA18's published unit_source_statistics_*.nc to 1e-10
    deg on every zone checked, against up to 4.5e-4 deg with the sphere."""
    az, _, dist = _GEOD.inv(p1[0], p1[1], p2[0], p2[1])
    lon, lat, _ = _GEOD.fwd(p1[0], p1[1], az, dist / 2.0)
    return np.array([lon, lat])


def _bearing(p1, p2):
    """Initial great-circle bearing from p1 to p2 (deg clockwise from
    north), exactly geosphere::bearing(..., f=0) -- rptha's own call sites
    (unit_sources.R:27-29,404,710-711) all pass f=0, so this uses
    _GEOD_SPHERE, not the WGS84 _GEOD."""
    az12, _, _ = _GEOD_SPHERE.inv(p1[0], p1[1], p2[0], p2[1])
    return az12 % 360.0


def _dist_haversine(p1, p2):
    """Spherical surface distance (m) between two (lon, lat) points, exactly
    geosphere::distHaversine (r=6378137) -- matches R, not the WGS84
    ellipsoid _GEOD above. Feeds the dip calculation."""
    _, _, dist = _GEOD_SPHERE.inv(p1[0], p1[1], p2[0], p2[1])
    return dist


def _area_polygon(lonlat):
    """Absolute spherical area (m^2) of a polygon given as (lon, lat) rows,
    exactly geosphere::areaPolygon(coords, f=0) -- matches R, not the WGS84
    ellipsoid _GEOD above. Feeds the width calculation."""
    lons = np.asarray(lonlat)[:, 0]
    lats = np.asarray(lonlat)[:, 1]
    area, _ = _GEOD_SPHERE.polygon_area_perimeter(lons, lats)
    return abs(area)


def distance_down_depth(p1, p2, depth_in_km=True, n=1000):
    """3D distance between two (lon, lat, depth) points along a great circle.

    Faithful port of ``geometric_util.R::distance_down_depth``: interpolate the
    surface path, assume depth varies linearly with along-path distance, then
    sum the 3D chord lengths in an Earth-centred Cartesian frame. R's own
    ``distance_down_depth`` interpolates via ``geosphere::gcIntermediate``,
    which is intrinsically spherical (r=6378137, no ellipsoidal variant
    exists in geosphere) -- so this uses _GEOD_SPHERE, not the WGS84 _GEOD.
    """
    p1 = np.asarray(p1, dtype=float).copy()
    p2 = np.asarray(p2, dtype=float).copy()

    inter = _GEOD_SPHERE.npts(p1[0], p1[1], p2[0], p2[1], n)
    path = np.array([[p1[0], p1[1]]] + inter + [[p2[0], p2[1]]])

    d1 = p1[2] * 1000.0 if depth_in_km else p1[2]
    d2 = p2[2] * 1000.0 if depth_in_km else p2[2]
    depth = np.linspace(d1, d2, n + 2)

    lon = path[:, 0] * _DEG2RAD
    lat = path[:, 1] * _DEG2RAD
    rr = _R - depth
    xs = rr * np.sin(lon) * np.cos(lat)
    ys = rr * np.cos(lon) * np.cos(lat)
    zs = rr * np.sin(lat)
    return np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2))


def _mean_angle(angles_rad):
    """Mean of angles (radians) via the complex-mean method, in radians."""
    z = np.mean(np.exp(1j * np.asarray(angles_rad)))
    return np.arctan2(z.imag, z.real)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def discretized_source_approximate_summary_statistics(
        unit_source_grid, default_rake=90.0, default_slip=1.0,
        depth_in_km=True):
    """Per-unit-source summary statistics from a unit_source_grid.

    Faithful transcription of the R routine. Returns a dict of numpy arrays
    (one entry per unit source), with the same keys as the R data.frame:
    ``lon_c, lat_c, depth, strike, dip, rake, slip, length, width,
    downdip_number, alongstrike_number, subfault_number, max_depth``.
    Lengths/widths are in km, area follow-on is (length*width).

    ``unit_source_grid`` has shape ``(n_downdip+1, 3, n_alongstrike+1)``.
    """
    g = np.asarray(unit_source_grid, dtype=float)
    if g.ndim != 3 or g.shape[1] != 3:
        raise ValueError("unit_source_grid must have shape "
                         "(n_downdip+1, 3, n_alongstrike+1)")

    n_along = g.shape[2] - 1
    n_down = g.shape[0] - 1
    num = n_along * n_down

    keys = ["lon_c", "lat_c", "depth", "strike", "dip", "rake", "slip",
            "length", "width", "downdip_number", "alongstrike_number",
            "subfault_number", "max_depth"]
    out = {k: np.full(num, np.nan) for k in keys}
    out["rake"][:] = default_rake
    out["slip"][:] = default_slip

    depth_scale = 1000.0 if depth_in_km else 1.0

    counter = 0
    # Loop order (along-strike outer, down-dip inner) matches the R routine.
    for i in range(n_along):
        for j in range(n_down):
            # Unit-source corner polygon: (lon, lat, depth) rows.
            sc = np.array([g[j, :, i], g[j + 1, :, i],
                           g[j + 1, :, i + 1], g[j, :, i + 1]])

            out["subfault_number"][counter] = counter + 1
            out["downdip_number"][counter] = j + 1
            out["alongstrike_number"][counter] = i + 1

            out["lon_c"][counter] = sc[:, 0].mean()
            out["lat_c"][counter] = sc[:, 1].mean()
            out["depth"][counter] = sc[:, 2].mean()
            out["max_depth"][counter] = sc[:, 2].max()

            # Strike: bearing from the top-edge midpoint towards corner 4.
            mid = _midpoint(sc[0, :2], sc[3, :2])
            strike = _bearing(mid, sc[3, :2])
            if not (0.0 <= strike < 360.0):
                strike = strike % 360.0
            out["strike"][counter] = strike

            # Dip: mean of left/right down-dip slopes (angle-averaged).
            dl = np.arctan2((sc[1, 2] - sc[0, 2]) * depth_scale,
                            _dist_haversine(sc[0, :2], sc[1, :2]))
            dr = np.arctan2((sc[2, 2] - sc[3, 2]) * depth_scale,
                            _dist_haversine(sc[3, :2], sc[2, :2]))
            dip = _mean_angle([dl, dr]) / _DEG2RAD
            out["dip"][counter] = dip

            # Length: mean of the two along-strike edge lengths (3D).
            len0 = 0.5 * (distance_down_depth(sc[0], sc[3], depth_in_km)
                          + distance_down_depth(sc[1], sc[2], depth_in_km))

            surface_area = _area_polygon(sc[:, :2])
            sloping_area = surface_area * np.sqrt(1.0 + np.tan(dip * _DEG2RAD) ** 2)
            width0 = sloping_area / len0

            out["length"][counter] = len0 / 1000.0
            out["width"][counter] = width0 / 1000.0
            counter += 1

    # Convenience: sloping area (km^2), used by rate/event calculations.
    out["area"] = out["length"] * out["width"]
    return out


# ---------------------------------------------------------------------------
# Simple grid builder for idealised interfaces
# ---------------------------------------------------------------------------


def make_planar_unit_source_grid(lon0, lat0, strike, dip, n_alongstrike,
                                 n_downdip, subfault_length, subfault_width,
                                 top_depth=0.0):
    """Build a unit_source_grid for an idealised planar dipping interface.

    Useful for testing and for approximate new-zone geometries where a full
    contour discretisation is not warranted. The interface starts at
    ``(lon0, lat0, top_depth)`` (the shallow along-strike corner), extends
    ``n_alongstrike`` unit sources along ``strike`` and ``n_downdip`` unit
    sources down the ``dip`` direction.

    Returns an array of shape ``(n_downdip+1, 3, n_alongstrike+1)``, matching
    the layout expected by
    :func:`discretized_source_approximate_summary_statistics`.
    """
    strike_rad = strike * _DEG2RAD
    dip_rad = dip * _DEG2RAD
    downdip_az = (strike + 90.0) % 360.0  # dip direction (to the right of strike)

    # Horizontal projection of one down-dip step, and its depth increment (km).
    horiz_step_m = subfault_width * 1000.0 * np.cos(dip_rad)
    depth_step_km = subfault_width * np.sin(dip_rad)
    along_step_m = subfault_length * 1000.0

    grid = np.empty((n_downdip + 1, 3, n_alongstrike + 1))
    for j in range(n_downdip + 1):
        depth = top_depth + j * depth_step_km
        # Move j down-dip steps from the top edge, keeping lon0/lat0 as anchor.
        if j == 0:
            base_lon, base_lat = lon0, lat0
        else:
            base_lon, base_lat, _ = _GEOD.fwd(lon0, lat0, downdip_az,
                                              j * horiz_step_m)
        for i in range(n_alongstrike + 1):
            if i == 0:
                lon, lat = base_lon, base_lat
            else:
                lon, lat, _ = _GEOD.fwd(base_lon, base_lat, strike,
                                        i * along_step_m)
            grid[j, 0, i] = lon
            grid[j, 1, i] = lat
            grid[j, 2, i] = depth
    return grid


# ---------------------------------------------------------------------------
# Grid from real depth contours (simplified discretisation)
# ---------------------------------------------------------------------------


def _resample_line_by_arclength(coords, n_points):
    """Resample a (lon, lat) polyline to ``n_points`` equal geodesic spacing."""
    coords = np.asarray(coords, dtype=float)
    # Cumulative geodesic distance along the line.
    seg = np.array([_dist_haversine(coords[k], coords[k + 1])
                    for k in range(len(coords) - 1)])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n_points)
    out = np.empty((n_points, 2))
    for m, t in enumerate(targets):
        k = np.searchsorted(cum, t, side="right") - 1
        k = min(max(k, 0), len(coords) - 2)
        span = cum[k + 1] - cum[k]
        frac = 0.0 if span == 0 else (t - cum[k]) / span
        # Interpolate along the great circle of that segment.
        az, _, dist = _GEOD.inv(coords[k, 0], coords[k, 1],
                                coords[k + 1, 0], coords[k + 1, 1])
        lon, lat, _ = _GEOD.fwd(coords[k, 0], coords[k, 1], az, dist * frac)
        out[m] = (lon, lat)
    return out


def discretized_source_from_contours(contours, n_alongstrike,
                                     n_downdip=None):
    """Build a unit_source_grid from a set of depth contours.

    This is a *simplified* stand-in for rptha's
    ``discretized_source_from_source_contours``. It does NOT reproduce rptha's
    down-dip-line orthogonalisation; it simply resamples each depth contour to
    a common number of along-strike points and (optionally) interpolates extra
    rows between contours. For reasonably regular subduction contours this
    gives a usable grid; for strongly curved or irregular interfaces prefer a
    grid produced by rptha itself.

    Parameters
    ----------
    contours : list of (depth_km, coords) pairs
        ``coords`` is an (N, 2) array of (lon, lat) for that depth contour.
        Order does not matter; contours are sorted by depth (shallow first).
    n_alongstrike : int
        Number of unit sources along strike (the grid gets n_alongstrike+1
        points along strike).
    n_downdip : int, optional
        Number of unit sources down dip. Defaults to (number of contours - 1),
        i.e. one row of unit sources between each adjacent contour pair. If
        larger, contours are linearly interpolated (by depth) to add rows.

    Returns
    -------
    unit_source_grid : ndarray, shape (n_downdip+1, 3, n_alongstrike+1)
        (lon, lat, depth-km) vertices, ready for
        :func:`discretized_source_approximate_summary_statistics`.
    """
    ordered = sorted(contours, key=lambda c: float(c[0]))
    depths = np.array([float(d) for d, _ in ordered])
    n_contours = len(ordered)
    if n_contours < 2:
        raise ValueError("need at least two depth contours")

    # Orient all contours consistently along strike. Contour data often stores
    # adjacent depth lines in opposite directions; if we don't fix that, the
    # shallow end of one contour gets paired with the deep zone's far end,
    # producing nonsensical unit-source widths. Use the shallowest contour's
    # first endpoint as the reference "start" corner and flip any contour whose
    # start is closer to the reference's other end.
    ref = np.asarray(ordered[0][1], dtype=float)
    ref_start = ref[0]
    oriented = []
    for d, c in ordered:
        c = np.asarray(c, dtype=float)
        d_start = _dist_haversine(c[0], ref_start)
        d_end = _dist_haversine(c[-1], ref_start)
        if d_end < d_start:
            c = c[::-1]
        oriented.append((d, c))
    ordered = oriented

    # Resample every contour to the same along-strike point count.
    npts = n_alongstrike + 1
    resampled = np.array([_resample_line_by_arclength(c, npts) for _, c in ordered])
    # resampled: (n_contours, npts, 2)

    if n_downdip is None:
        n_downdip = n_contours - 1

    rows = n_downdip + 1
    # Target depths for the grid rows, evenly spaced from shallowest to deepest.
    row_depths = np.linspace(depths[0], depths[-1], rows)

    grid = np.empty((rows, 3, npts))
    for r, dep in enumerate(row_depths):
        # Linear interpolation between bracketing contours by depth.
        k = np.searchsorted(depths, dep, side="right") - 1
        k = min(max(k, 0), n_contours - 2)
        span = depths[k + 1] - depths[k]
        frac = 0.0 if span == 0 else (dep - depths[k]) / span
        lonlat = (1 - frac) * resampled[k] + frac * resampled[k + 1]
        grid[r, 0, :] = lonlat[:, 0]
        grid[r, 1, :] = lonlat[:, 1]
        grid[r, 2, :] = dep
    return grid
