"""SLAB depth raster -> PTHA18-style "depth below the nearby trench" contours.

New in v8. This replaces everything v4-v7 did in step 1 to turn a SLAB2.0 /
SLAB1.0 raster into the depth contours the discretiser meshes
(SHALLOWEST_KM, N_CONTOUR_LEVELS, fragment chaining, the end-to-end and
width-ratio checks, the nudging/raising/level-dropping retry ladder, and the
per-zone hand clips).

What PTHA18 actually did
------------------------
ReportPTHA.pdf section 3.1: "earthquake slip could occur between the trench
(depth = 0 km) and a source-zone specific maximum depth below the trench",
and "The source-zone geometries were converted into elevation contours
giving the 'depth below the nearby trench'".

PTHA18's own contour shapefiles (DATA/SOURCEZONE_CONTOURS.zip on NCI) show
exactly how: every zone has a 0 km line that IS the trench, then one
unbroken contour every 5 km down to the cutoff. Sampling the SLAB raster
along those contours shows the datum is LOCAL: level L sits where
SLAB depth minus the SLAB depth at the nearest trench point equals L (the
residual of that rule is +-0.3 to +-1 km on puysegur2, kurilsjapan,
makran2 and southamerica), whereas one constant offset for the whole zone,
which is what v3-v7 used, leaves a 3-7 km spread along strike. And the
0 km line lies on the up-dip edge of the SLAB raster itself.

Why the constant datum broke v3-v7
----------------------------------
The trench depth varies along strike (kermadectonga2 5-11 km, southamerica
2-10 km below sea level). With one constant offset, a shallow "below trench"
level asks SLAB for a depth that simply does not exist wherever the local
trench is deeper than it, so the shallow contours came out in fragments,
with holes and straight-line bridges. Everything v4-v7 added to step 1 was
a work-around for that one defect, and each work-around cost area: the
shallowest level had to be raised until it was continuous (calabria2 ended
up starting 38 km below the trench, 1 row, 12 unit sources), and the level
count was cut until a width check passed (southamerica 3 rows, -41.8% area).

With the local datum every level exists all along the arc by construction:
at the trench D = 0, and D grows down dip. No floor, no retries, no clips.

Method
------
1. Keep the largest connected block of valid raster cells.
2. Trace its outer boundary and classify every boundary vertex:
   * shallow: its (smoothed) edge depth is within `shallow_margin_km` of the
     shallow end of all edge depths (5th percentile);
   * up-dip facing: the large-scale depth gradient, sampled just inside,
     points INTO the slab (cosine with the inward normal >= 0.5);
   The trench is the longest run of such vertices, where short shallow gaps
   (flat accretionary prisms, e.g. Calabria and Makran, give locally noisy
   gradients) are bridged, and whose two ends are trimmed back to where the
   gradient is clearly up-dip (cosine >= 0.7): a lateral clip edge runs
   down dip, a trench runs along strike.
   With a clip window (bbox), the vertices on the window's own straight
   edge, where the raster still has slab on the other side, are never
   trench and never bridged: the window cuts THROUGH the slab there, and
   that cut can be shallow enough and face up dip enough to pass both tests
   (hellenic_east2: 129 km of its "trench" were the 29.126E clip line, 14-22
   km deep against 11-15 km for the real trench, so half the zone was
   measured from a datum 6 km too deep and its 25-30 km contours looped).
   The zone's end is then cut along a down-dip line by step 6 below, as at
   any other crooked end. Without a bbox nothing changes.
3. Trench depth = depth of the nearest valid cell to each trench vertex,
   running median over 50 km along the trench.
4. Datum field: every valid cell takes the trench depth of its nearest
   trench point (PTHA18's "nearby trench"), smoothed with a 25 km Gaussian
   so the contours do not inherit a kink where the nearest point jumps
   between two stretches of trench. Only the datum is smoothed, never the
   slab depth.
5. D = depth - datum. Contour D at spacing, 2*spacing, ..., cutoff; the
   0 km contour is the trench line itself. Each level keeps its longest
   piece (any other piece longer than 5% of it is reported).
6. Clean ends (trim_ends): where the SLAB raster, or the clip window, ends
   the zone crookedly (the mesh's end cells would be more than 45 deg off
   square) or the zone narrows to a point, every contour is cut along one
   down-dip line, as PTHA18 did by hand. Other ends are kept as they are.

The result is what step 2's discretiser consumes. Validated against PTHA18's
own contours and meshes: see html/docs/step1.html.
"""

import numpy as np

R_KM = 6378.137
KM_PER_DEG = np.pi * R_KM / 180.0


# ---------------------------------------------------------------------------
# small geometry helpers
# ---------------------------------------------------------------------------

def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                            np.sin(la)])


def _hav_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def line_km(c):
    """Cumulative along-line distance (km) of an (N, 2) lon/lat polyline."""
    c = np.asarray(c, dtype=float)
    return np.concatenate([[0.0], np.cumsum(
        _hav_km(c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]))])


def resample_line(c, spacing_km):
    """Equally spaced points (by arc length) along a polyline."""
    s = line_km(c)
    n = max(2, int(np.ceil(s[-1] / spacing_km)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])])


def drop_repeated_vertices(c):
    """Remove consecutive duplicate vertices (zero-length segments)."""
    c = np.asarray(c, dtype=float)
    keep = np.concatenate([[True], np.any(np.diff(c, axis=0) != 0, axis=1)])
    return c[keep]


def _contour_paths(x, y, field, level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cs = ax.contour(x, y, field, levels=[level])
    segs = [np.asarray(s, dtype=float) for s in cs.allsegs[0] if len(s) >= 2]
    plt.close(fig)
    return segs


def _bilin(x, y, f, lon, lat):
    i = np.clip(np.searchsorted(x, lon) - 1, 0, x.size - 2)
    j = np.clip(np.searchsorted(y, lat) - 1, 0, y.size - 2)
    tx = np.clip((lon - x[i]) / (x[i + 1] - x[i]), 0, 1)
    ty = np.clip((lat - y[j]) / (y[j + 1] - y[j]), 0, 1)
    return (f[j, i] * (1 - tx) * (1 - ty) + f[j, i + 1] * tx * (1 - ty)
            + f[j + 1, i] * (1 - tx) * ty + f[j + 1, i + 1] * tx * ty)


def largest_component(mask):
    """The largest 8-connected block of True cells."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _boundary_loop(x, y, mask):
    """Closed outer boundary of `mask`, padded so it always closes."""
    dx, dy = x[1] - x[0], y[1] - y[0]
    xp = np.concatenate([[x[0] - dx], x, [x[-1] + dx]])
    yp = np.concatenate([[y[0] - dy], y, [y[-1] + dy]])
    mp = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2))
    mp[1:-1, 1:-1] = mask
    loop = max(_contour_paths(xp, yp, mp, 0.5), key=len)
    if np.hypot(*(loop[0] - loop[-1])) < 1e-9:
        loop = loop[:-1]
    return loop


def _circular_smooth(a, w):
    if w <= 0:
        return a.copy()
    k = np.ones(2 * w + 1) / (2 * w + 1)
    out = np.empty_like(a)
    for c in range(a.shape[1]):
        ext = np.concatenate([a[-w:, c], a[:, c], a[:w, c]])
        out[:, c] = np.convolve(ext, k, mode="valid")
    return out


def _depth_gradient(x, y, depth, sigma_cells):
    """d(depth)/d(east km), d(depth)/d(north km), NaN-aware smoothing."""
    from scipy import ndimage
    valid = np.isfinite(depth)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    sm = ndimage.gaussian_filter(depth[tuple(idx)], sigma_cells)
    gy, gx = np.gradient(sm)
    gx = gx / ((x[1] - x[0]) * KM_PER_DEG * np.cos(np.radians(y))[:, None])
    gy = gy / ((y[1] - y[0]) * KM_PER_DEG)
    return gx, gy


def _runs(flags):
    """Circular runs of True in `flags` as (start, length)."""
    n = flags.size
    if flags.all():
        return [(0, n)]
    if not flags.any():
        return []
    start = int(np.argmax(~flags))
    out, k = [], 0
    while k < n:
        if flags[(start + k) % n]:
            j = k
            while j < n and flags[(start + j) % n]:
                j += 1
            out.append(((start + k) % n, j - k))
            k = j
        else:
            k += 1
    return out


# ---------------------------------------------------------------------------
# 1. the trench
# ---------------------------------------------------------------------------

def _ramp_trim(s, depth_ms, rise_km, ref_km, max_km):
    """v10_q ramp rule: how far to trim each end of a trench (km).

    An end is trimmed back, by at most `max_km`, while its running-median
    depth `depth_ms` is more than `rise_km` deeper than the trench's normal
    depth there: the larger of the trench's median between `ref_km` (from
    that end) and the whole trench's median. A trench end that climbs in
    depth like a ramp is not trench but the edge of the SLAB data (hellenic:
    19.7 km below sea level at the Kefalonia end against 11.5 km on the rest
    of the trench, which became the datum of half the zone and piled the
    25-35 km contours into one line).
    """
    whole = float(np.median(depth_ms))
    out = []
    for dist in (s, s[-1] - s):
        inward = (dist >= ref_km[0]) & (dist <= ref_km[1])
        normal = max(float(np.median(depth_ms[inward])) if inward.any() else whole, whole)
        order = np.argsort(dist)
        trim = 0.0
        for k in order:
            if dist[k] > max_km or depth_ms[k] <= normal + rise_km:
                break
            trim = float(dist[k])
        out.append(trim)
    return out


def find_trench(x, y, depth, shallow_margin_km=12.0, updip_cos=0.5,
                end_cos=0.7, gap_km=250.0, smooth_km=15.0, cut=None,
                ramp_rise_km=None, ramp_ref_km=(100.0, 200.0), ramp_max_km=100.0):
    """Locate the trench (up-dip edge) of a SLAB depth raster.

    `depth` is positive-down, km below sea level, NaN off the slab.
    `cut` (optional, 2D bool like `depth`): cells that ARE slab in the raster
    but were removed by a clip window. A boundary vertex next to one lies on
    the clip window's straight edge, a line drawn through the slab, not on
    the slab's own edge, so it is never trench, however shallow it is (v9:
    hellenic_east2's clip at 29.126E cut the slab 14-22 km deep and 129 km of
    it passed both tests below, which shifted the datum of half the zone).
    `ramp_rise_km` (v10_q, optional): trim each end of the trench where it
    climbs in depth like a ramp (see _ramp_trim); None keeps v9's trench.
    Returns a dict: 'trench' (N, 2) smoothed lon/lat line, 'trench_depth_raw'
    per vertex, plus diagnostics ('loop', 'score', 'edge_depth',
    'threshold_km', 'n_runs', 'length_km', 'clip_edge_km', 'ramp_trim_km').
    """
    from scipy.spatial import cKDTree

    valid = largest_component(np.isfinite(depth))
    loop = _boundary_loop(x, y, valid)
    n = len(loop)
    lat0 = np.radians(np.mean(loop[:, 1]))
    xy = np.column_stack([loop[:, 0] * np.cos(lat0), loop[:, 1]]) * KM_PER_DEG
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    step = float(np.median(seg))

    # inward normal from the (smoothed) boundary tangent and the loop's
    # orientation
    xs = _circular_smooth(xy, max(1, int(round(smooth_km / step))))
    tang = np.roll(xs, -1, axis=0) - np.roll(xs, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    left = np.column_stack([-tang[:, 1], tang[:, 0]])
    n_in = left if area2 > 0 else -left

    # large-scale (~20 km) depth gradient a little inside the edge
    cell_km = abs(y[1] - y[0]) * KM_PER_DEG
    gx, gy = _depth_gradient(x, y, np.where(valid, depth, np.nan),
                             sigma_cells=max(2.0, 20.0 / cell_km))
    probe = max(3.0 * cell_km, 15.0)
    plon = loop[:, 0] + n_in[:, 0] * probe / (KM_PER_DEG * np.cos(np.radians(loop[:, 1])))
    plat = loop[:, 1] + n_in[:, 1] * probe / KM_PER_DEG
    g = np.column_stack([_bilin(x, y, gx, plon, plat), _bilin(x, y, gy, plon, plat)])
    score = np.sum(g * n_in, axis=1) / np.maximum(np.linalg.norm(g, axis=1), 1e-12)

    # edge depth: nearest valid cell, then a 50 km running median
    jj, ii = np.nonzero(valid)
    _, k0 = cKDTree(np.column_stack([x[ii], y[jj]])).query(loop)
    edep = depth[jj[k0], ii[k0]]
    wk = max(1, int(round(25.0 / step)))
    ext = np.concatenate([edep[-wk:], edep, edep[:wk]])
    edep_s = np.array([np.median(ext[i:i + 2 * wk + 1]) for i in range(n)])
    threshold = max(float(np.percentile(edep_s, 5)) + shallow_margin_km, 15.0)
    shallow = edep_s <= threshold

    is_tr = shallow & (score >= updip_cos)

    # edge of the clip window, not of the slab: never trench, never bridged
    on_cut = np.zeros(n, dtype=bool)
    if cut is not None and np.any(cut):
        cj, ci = np.nonzero(cut)
        dist, _ = cKDTree(np.column_stack([x[ci], y[cj]])).query(loop)
        on_cut = dist <= 1.5 * max(abs(x[1] - x[0]), abs(y[1] - y[0]))
        is_tr &= ~on_cut

    def run_km(i0, m):
        return float(seg[(i0 + np.arange(m)) % n].sum())

    for i0, m in _runs(~is_tr):
        gap = (i0 + np.arange(m)) % n
        if run_km(i0, m) < gap_km and shallow[gap].all() and not on_cut[gap].any():
            is_tr[gap] = True
    runs = _runs(is_tr)
    if not runs:
        raise ValueError("no trench-like (shallow, up-dip facing) edge found "
                         "on the raster boundary")
    i0, m = max(runs, key=lambda r: run_km(*r))
    idx = (i0 + np.arange(m)) % n
    a, b = 0, len(idx)
    while a < b and score[idx[a]] < end_cos:
        a += 1
    while b > a and score[idx[b - 1]] < end_cos:
        b -= 1
    idx = idx[a:b]
    if len(idx) < 3:
        raise ValueError("the trench-like edge vanished after end trimming")

    raw = loop[idx]
    # Depth AT the edge: the nearest valid cell centre sits up to one cell
    # down dip of the edge line, so its depth reads slightly deep (0.4 km at
    # 20 deg dip on a 0.02-deg grid; more on coarser grids); extrapolate it
    # to the edge point along the (smoothed) local depth gradient.
    # The slope comes from a least-squares plane through the valid cells of
    # the 5x5 window around that cell (the smoothed gradient above is
    # flattened next to the edge by the fill outside the slab).
    ci, cj = ii[k0[idx]], jj[k0[idx]]
    edge_depth = np.empty(len(idx))
    for q in range(len(idx)):
        i0, j0 = ci[q], cj[q]
        jsl = slice(max(j0 - 2, 0), min(j0 + 3, depth.shape[0]))
        isl = slice(max(i0 - 2, 0), min(i0 + 3, depth.shape[1]))
        win = depth[jsl, isl]
        wy, wx = np.meshgrid(y[jsl], x[isl], indexing="ij")
        ok = np.isfinite(win)
        ex = (wx[ok] - x[i0]) * KM_PER_DEG * np.cos(np.radians(y[j0]))
        ey = (wy[ok] - y[j0]) * KM_PER_DEG
        px = (raw[q, 0] - x[i0]) * KM_PER_DEG * np.cos(np.radians(y[j0]))
        py = (raw[q, 1] - y[j0]) * KM_PER_DEG
        if ok.sum() >= 3:
            A = np.column_stack([np.ones(ok.sum()), ex, ey])
            coef = np.linalg.lstsq(A, win[ok], rcond=None)[0]
            edge_depth[q] = coef[0] + coef[1] * px + coef[2] * py
        else:
            edge_depth[q] = depth[j0, i0]
    ramp_trim = [0.0, 0.0]
    if ramp_rise_km is not None:
        s_raw = line_km(raw)
        ramp_trim = _ramp_trim(s_raw, running_median(edge_depth, s_raw, 50.0),
                               ramp_rise_km, ramp_ref_km, ramp_max_km)
        keep = (s_raw > ramp_trim[0]) & (s_raw < s_raw[-1] - ramp_trim[1])
        if any(ramp_trim):
            k0, k1 = np.nonzero(keep)[0][[0, -1]]
            raw, edge_depth = raw[k0:k1 + 1], edge_depth[k0:k1 + 1]
    trench = raw.copy()
    w = max(1, int(round(smooth_km / step)))
    if len(trench) > 2 * w + 1:
        ker = np.ones(2 * w + 1) / (2 * w + 1)
        for c in range(2):
            pad = np.concatenate([np.full(w, raw[0, c]), raw[:, c], np.full(w, raw[-1, c])])
            trench[:, c] = np.convolve(pad, ker, mode="valid")
        trench[0], trench[-1] = raw[0], raw[-1]
    return {
        "trench": trench, "trench_raw": raw, "trench_depth_raw": edge_depth,
        "loop": loop, "score": score, "is_trench": is_tr, "edge_depth": edep_s,
        "threshold_km": threshold, "n_runs": len(runs),
        "length_km": float(line_km(trench)[-1]), "loop_length_km": float(seg.sum()),
        "clip_edge_km": float(seg[on_cut].sum()),
        "clip_edge_shallow_km": float(seg[on_cut & shallow & (score >= updip_cos)].sum()),
        "ramp_trim_km": ramp_trim,
    }


def running_median(values, dist_km, window_km):
    out = np.empty_like(values, dtype=float)
    for k in range(len(values)):
        m = np.abs(dist_km - dist_km[k]) <= window_km / 2
        out[k] = np.median(values[m])
    return out


# ---------------------------------------------------------------------------
# 2. depth below the nearby trench
# ---------------------------------------------------------------------------

def depth_below_trench(x, y, depth, trench, trench_depth, smooth_km=25.0):
    """PTHA18's 'depth below the nearby trench' on every valid cell."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    X, Y = np.meshgrid(x, y)
    ok = np.isfinite(depth)
    _, k = cKDTree(_xyz(trench[:, 0], trench[:, 1])).query(_xyz(X[ok], Y[ok]))
    datum = np.zeros(depth.shape)
    datum[ok] = trench_depth[k]
    if smooth_km > 0:
        sig = smooth_km / (abs(y[1] - y[0]) * KM_PER_DEG)
        w = ok.astype(float)
        num = ndimage.gaussian_filter(datum * w, sig)
        den = ndimage.gaussian_filter(w, sig)
        datum = np.where(ok, num / np.maximum(den, 1e-12), 0.0)
    D = np.full(depth.shape, np.nan)
    D[ok] = depth[ok] - datum[ok]
    return D


# ---------------------------------------------------------------------------
# 3. clean ends: cut the contours along a down-dip line at each end
# ---------------------------------------------------------------------------
#
# PTHA18 ended every zone by hand with a straight line across the contours
# (compare its SOURCEZONE_CONTOURS with the raw SLAB data). Without that, a
# zone ends wherever the SLAB raster (or the clip window) stops: the deeper
# contours run on past the trench's last point, or stop short of it, and at
# some tips they converge to a point. The mesh then ends in slanted or
# sliver-shaped cells. This reproduces the hand edit with one rule: cut every
# contour along the same down-dip line, a line of steepest descent of the
# depth below the trench, which crosses every contour at a right angle.

# PTHA18's hand-edited ends are 2-42 degrees off square (14 zones; cascadia's
# last column 53); the artefact ends of the SLAB-derived zones 53-86.
MAX_END_DEVIATION_DEG = 45.0


def down_dip_transects(x, y, D, trench, cutoff_km, spacing_km=5.0,
                       step_km=1.0, sigma_km=10.0, max_km=1500.0):
    """Lines of steepest descent of D, one from every `spacing_km` of trench.

    Each is followed until D reaches `cutoff_km` ("complete") or until it
    leaves the slab first (the SLAB model, or the clip window, stops before
    the cutoff there).

    Returns (s_km, start_points (N, 2), complete (N,), length_km (N,),
    paths: list of (M, 2) lon/lat arrays).
    """
    from scipy import ndimage
    valid = np.isfinite(D)
    cell_km = abs(y[1] - y[0]) * KM_PER_DEG
    gx, gy = _depth_gradient(x, y, D, max(1.0, sigma_km / cell_km))
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    Df = D[tuple(idx)]
    pts = resample_line(trench, spacing_km)
    s = line_km(pts)
    n = len(pts)
    lon, lat = pts[:, 0].copy(), pts[:, 1].copy()
    active = np.ones(n, dtype=bool)
    complete = np.zeros(n, dtype=bool)
    length = np.zeros(n)
    paths = [[(lon[k], lat[k])] for k in range(n)]
    for _ in range(int(max_km / step_km)):
        a = np.nonzero(active)[0]
        if a.size == 0:
            break
        ux, uy = _bilin(x, y, gx, lon[a], lat[a]), _bilin(x, y, gy, lon[a], lat[a])
        nrm = np.hypot(ux, uy)
        stuck = ~np.isfinite(nrm) | (nrm == 0)
        nrm = np.where(stuck, 1.0, nrm)
        lon[a] += step_km * ux / nrm / (KM_PER_DEG * np.cos(np.radians(lat[a])))
        lat[a] += step_km * uy / nrm / KM_PER_DEG
        length[a] += step_km
        for k in a:
            paths[k].append((lon[k], lat[k]))
        off = ((lon[a] < x[0]) | (lon[a] > x[-1]) | (lat[a] < y[0]) | (lat[a] > y[-1]))
        i = np.clip(np.round((lon[a] - x[0]) / (x[1] - x[0])).astype(int), 0, x.size - 1)
        j = np.clip(np.round((lat[a] - y[0]) / (y[1] - y[0])).astype(int), 0, y.size - 1)
        inside = valid[j, i] & ~off
        # the trench lies on the raster's edge, so the first cell or two may
        # be off the slab; after that, leaving it ends the line
        left = off | (~inside & (length[a] > 3.0 * cell_km))
        done = inside & (_bilin(x, y, Df, lon[a], lat[a]) >= cutoff_km)
        complete[a[done]] = True
        active[a[done | left | stuck]] = False
    return s, pts, complete, length, [np.asarray(p) for p in paths]


def _end_deviation(contours, tline, end, rows, along_km=50.0):
    """How far from square the end cells of the mesh would be, in degrees
    (0 = rectangular corners), if the zone kept its natural end.

    rptha pins the mesh's end column to the contours' end points and places
    the next column about one unit-source length (`along_km`) in along each
    contour; both are then cut into `rows` equal pieces down dip. This
    rebuilds those two columns from the contours alone and returns the worst
    corner of the cells between them, the same measure as for a real mesh.
    """
    from shapely.geometry import Point
    ends, inner = [], []
    for _, c in contours:
        c = np.asarray(c, dtype=float)
        same = tline.project(Point(c[0])) <= tline.project(Point(c[-1]))
        if (end == "start") != same:
            c = c[::-1]                     # now it runs from this end inward
        cum = line_km(c)
        d = min(along_km, cum[-1])
        ends.append(c[0])
        inner.append([np.interp(d, cum, c[:, 0]), np.interp(d, cum, c[:, 1])])

    def nodes(p, n):
        p = np.asarray(p)
        cum = line_km(p)
        t = np.linspace(0.0, cum[-1], n + 1)
        return np.column_stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])])

    def corner(p, a, b):
        k = np.cos(np.radians(p[1]))
        u = np.array([(a[0] - p[0]) * k, a[1] - p[1]])
        v = np.array([(b[0] - p[0]) * k, b[1] - p[1]])
        cosang = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
        return abs(90.0 - np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    E, C = nodes(ends, rows), nodes(inner, rows)
    return max(max(corner(E[r], E[r + 1], C[r]), corner(E[r + 1], E[r], C[r + 1]))
               for r in range(rows))


def _end_cuts(s, complete, length, deviation, max_deviation=MAX_END_DEVIATION_DEG,
              taper_tip=0.5, taper_keep=0.7, smooth_km=25.0):
    """Where to cut each end of the zone, if at all, and why.

    An end is cut only if it is an artefact:
    * it narrows to a point (smoothed width at the outermost complete
      transect below `taper_tip` of the zone's median width): the cut goes
      where the width is back to `taper_keep` of the median. PTHA18's own
      hand edit of kermadectonga2's southern tip is at 74%; this rule cuts
      at 70%, 10 km away;
    * or its natural end is far from square (`deviation` above
      `max_deviation` degrees): the cut goes along the outermost transect
      that reaches the cutoff. PTHA18's hand-edited ends are 2-42 degrees
      off square (cascadia's last column 53), the artefact ends 55-86.
    Any other end is kept as the SLAB data give it, as PTHA18 kept
    makran2's slanted ends.

    `deviation` is {"start": deg, "end": deg}. Returns None if there is no
    complete transect, else {"start": ..., "end": ...} with index None for
    an end that stays as it is.
    """
    if not complete.any():
        return None
    med = float(np.median(length[complete]))
    ok = np.nonzero(complete)[0]
    width = np.full(s.size, np.nan)
    for k in ok:
        near = ok[np.abs(s[ok] - s[k]) <= smooth_km / 2]
        width[k] = np.median(length[near])
    out = {"median_width_km": med}
    for name, order in (("start", ok), ("end", ok[::-1])):
        k = int(order[0])
        tip = float(width[k] / med)
        rec = {"index": None, "tip_width_ratio": tip, "deviation_deg": deviation[name]}
        if tip < taper_tip:
            back = [int(q) for q in order if width[q] >= taper_keep * med]
            if back:
                rec.update(index=back[0], reason=(
                    f"the zone narrows to a point there ({100 * tip:.0f}% of its "
                    f"median width at the tip): cut where it is back to "
                    f"{100 * taper_keep:.0f}%"))
        elif deviation[name] > max_deviation:
            rec.update(index=k, reason=(
                f"its natural end is {deviation[name]:.0f} deg off square: cut along "
                + ("the down-dip line from the trench's last point"
                   if k in (0, s.size - 1) else
                   "the last down-dip line that reaches the cutoff inside the SLAB data")))
        else:
            rec["reason"] = (f"kept as it is ({deviation[name]:.0f} deg off square, "
                             f"within PTHA18's own range)")
        out[name] = rec
    a = out["start"]["index"] if out["start"]["index"] is not None else 0
    b = out["end"]["index"] if out["end"]["index"] is not None else s.size - 1
    if a >= b:
        return None
    return out


def _cuts_at_points(s, complete, paths, points, deviation):
    """Ends fixed in advance (v9, Berryman et al. 2015 segment ends): each
    end of the zone is cut along the complete down-dip transect that passes
    closest to the given lon/lat point. The point need not be on this
    trench: Berryman's trench end points can lie well down dip of the SLAB
    raster's up-dip edge (hellenic: 20-220 km), and the line of steepest
    descent through the point is the cut that point defines. Same return
    shape as _end_cuts; the end with the smaller trench position is 'start'.
    """
    ok = np.nonzero(complete)[0]
    if ok.size < 2:
        return None
    found = []
    for lon, lat in points:
        k_lat = np.cos(np.radians(lat))
        best = (np.inf, None)
        for k in ok:
            p = paths[k]
            dx = (p[:, 0] - lon) * k_lat * KM_PER_DEG
            dy = (p[:, 1] - lat) * KM_PER_DEG
            # distance to the polyline (segments, not only vertices)
            ax, ay, bx, by = dx[:-1], dy[:-1], dx[1:], dy[1:]
            ux, uy = bx - ax, by - ay
            t = np.clip(-(ax * ux + ay * uy) / np.maximum(ux * ux + uy * uy, 1e-12), 0, 1)
            d = float(np.min(np.hypot(ax + t * ux, ay + t * uy))) if len(p) > 1 else float(np.hypot(dx[0], dy[0]))
            if d < best[0]:
                best = (d, int(k))
        found.append((best[1], best[0], (float(lon), float(lat))))
    found.sort(key=lambda f: f[0])
    if found[0][0] == found[1][0]:
        return None
    out = {"median_width_km": None}
    for name, (k, d, pt) in zip(("start", "end"), found):
        out[name] = {"index": k, "tip_width_ratio": None, "deviation_deg": deviation[name],
                     "point": pt, "point_distance_km": d, "trench_s_km": float(s[k]),
                     "reason": (f"cut at the given end point ({pt[0]:.3f}E, {pt[1]:.3f}N): along "
                                f"the down-dip line that passes {d:.0f} km from it")}
    return out


def _point_at_level(path, x, y, Df, level):
    """Point of a transect where D first reaches `level`."""
    d = _bilin(x, y, Df, path[:, 0], path[:, 1])
    k = int(np.argmax(d >= level)) if np.any(d >= level) else len(path) - 1
    if k == 0:
        return path[0]
    t = (level - d[k - 1]) / max(d[k] - d[k - 1], 1e-12)
    return path[k - 1] + np.clip(t, 0, 1) * (path[k] - path[k - 1])


def trim_ends(x, y, D, contours, cutoff_km, spacing_km=5.0, cut_points=None):
    """Cut every contour along one down-dip line at each end of the zone that
    is an artefact (see _end_cuts); a clean end is left as it is.

    cut_points: optional ((lon, lat), (lon, lat)). Both ends are then cut
    where these points say, whatever they look like (see _cuts_at_points),
    and the cut is not moved: the zone is defined by those points.

    Returns (contours, info). info['start'] / info['end'] give the reason,
    the natural end's deviation from square and, for a cut end, the trench
    and contour length removed and the cut line; info is None if nothing
    could be assessed, or {'skipped': why} if a cut could not be applied.
    """
    from scipy import ndimage
    from shapely.geometry import LineString, Point
    from shapely.ops import substring

    trench = contours[0][1]
    tline = LineString(trench)
    s, pts, complete, length, paths = down_dip_transects(
        x, y, D, trench, cutoff_km, spacing_km=spacing_km)
    # the row count step 2's width rule would give at 50 km (at least 2)
    rows = max(2, int(round(float(np.median(length[complete])) / 50.0))) if complete.any() else 2
    deviation = {e: _end_deviation(contours, tline, e, rows) for e in ("start", "end")}
    if cut_points is not None:
        cuts = _cuts_at_points(s, complete, paths, cut_points, deviation)
        if cuts is None:
            return contours, {"skipped": "the two end points fall on the same down-dip line"}
    else:
        cuts = _end_cuts(s, complete, length, deviation)
    if cuts is None:
        return contours, None
    valid = np.isfinite(D)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
    Df = D[tuple(idx)]

    def cut_at(line, level, k):
        """Along-line position where transect k crosses this contour."""
        if level == 0:
            return line.project(Point(pts[k]))
        target = Point(_point_at_level(paths[k], x, y, Df, level))
        hit = line.intersection(LineString(paths[k]))
        cand = [g for g in getattr(hit, "geoms", [hit]) if g.geom_type == "Point"]
        p = min(cand, key=target.distance) if cand else None
        if p is None or p.distance(target) > 0.05:      # ~5 km: no clean crossing
            p = line.interpolate(line.project(target))
            if p.distance(target) > 0.05:
                return None
        return line.project(p)

    def apply(k0, k1):
        """Contours cut at transects k0 / k1 (None: that end stays), and the
        longest piece removed at each end; (None, why) if a cut fails."""
        out = []
        removed = {"start": 0.0, "end": 0.0}
        for level, c in contours:
            line = LineString(c)
            # contours are not necessarily oriented like the trench
            same = tline.project(Point(c[0])) <= tline.project(Point(c[-1]))
            a = cut_at(line, level, k0) if k0 is not None else (0.0 if same else line.length)
            b = cut_at(line, level, k1) if k1 is not None else (line.length if same else 0.0)
            if a is None or b is None:
                return None, f"the {level:g} km contour has no clean crossing with the cut line"
            lo, hi = sorted((a, b))
            if hi - lo <= 0:
                return None, f"the cut lines leave nothing of the {level:g} km contour"
            head = line_km(np.asarray(substring(line, 0, lo).coords))[-1] if lo > 0 else 0.0
            tail = (line_km(np.asarray(substring(line, hi, line.length).coords))[-1]
                    if hi < line.length else 0.0)
            removed["start"] = max(removed["start"], head if same else tail)
            removed["end"] = max(removed["end"], tail if same else head)
            out.append((level, drop_repeated_vertices(np.asarray(
                substring(line, lo, hi).coords))))
        return out, removed

    # A cut end must itself come out within max_deviation of square; if it
    # does not (the contours still bend there), the cut moves inward one
    # transect at a time, up to 150 km, as a hand edit would.
    k0, k1 = cuts["start"]["index"], cuts["end"]["index"]
    after = {"start": None, "end": None}
    for name in ("start", "end"):
        k = k0 if name == "start" else k1
        if k is None or cut_points is not None:
            continue
        step = 1 if name == "start" else -1
        best = None
        for m in range(int(150.0 / spacing_km) + 1):
            kk = k + step * m
            if not (0 <= kk < s.size) or not complete[kk]:
                continue
            res, _ = apply(kk, k1) if name == "start" else apply(k0, kk)
            if res is None:
                continue
            dev = _end_deviation(res, LineString(res[0][1]), name, rows)
            if best is None or dev < best[1]:
                best = (kk, dev)
            if dev <= MAX_END_DEVIATION_DEG:
                break
        if best is not None:
            if name == "start":
                k0 = best[0]
            else:
                k1 = best[0]
            after[name] = best[1]

    out, removed = apply(k0, k1)
    if out is None:
        return contours, {"skipped": removed}
    if cut_points is not None:
        tl = LineString(out[0][1])
        after = {e: _end_deviation(out, tl, e, rows) for e in ("start", "end")}
    info = {"median_width_km": cuts["median_width_km"]}
    for name, k, trench_cut in (("start", k0, s[k0] if k0 is not None else 0.0),
                                ("end", k1, s[-1] - s[k1] if k1 is not None else 0.0)):
        info[name] = {"cut": k is not None,
                      "trench_km": float(trench_cut),
                      "contours_km": float(removed[name]),
                      "reason": cuts[name]["reason"],
                      "tip_width_ratio": cuts[name]["tip_width_ratio"],
                      "deviation_deg": cuts[name]["deviation_deg"],
                      "deviation_after_deg": after[name],
                      "cut_line": paths[k] if k is not None else None}
        if cut_points is not None:
            info[name].update(point=cuts[name]["point"],
                              point_distance_km=cuts[name]["point_distance_km"])
    info["transects"] = {"s_km": s, "complete": complete, "length_km": length}
    return out, info


# ---------------------------------------------------------------------------
# 4. the contours
# ---------------------------------------------------------------------------

def below_trench_contours(x, y, depth_msl, cutoff_km, spacing_km=5.0,
                          bbox=None, verbose=True, clean_ends=True, trench_ends=None,
                          ramp_rise_km=None):
    """Contours at 0, spacing, ..., cutoff km below the nearby trench.

    Parameters
    ----------
    x, y : 1D ascending lon / lat of the raster.
    depth_msl : 2D, positive-down km below sea level, NaN off the slab.
    cutoff_km : seismogenic cutoff, km below the trench (Berryman-derived).
    spacing_km : contour spacing (PTHA18 used 5 km).
    bbox : optional (lon_min, lon_max, lat_min, lat_max); the raster is
        masked to it first. For a raster that covers more than the zone.
    clean_ends : cut every contour along one down-dip line at each end of
        the zone (trim_ends), as PTHA18 did by hand.
    trench_ends : optional ((lon, lat), (lon, lat)), the zone's two ends as
        published (v9: Berryman et al. 2015's segment end points, for a zone
        that is one part of a SLAB region, e.g. hellenic_west2). The trench,
        datum and contours are built on the whole raster (or bbox), then
        every contour is cut along the down-dip line through each point;
        the automatic end cleaning is not used.
    ramp_rise_km : v10_q, optional. Trim each end of the trench where it
        climbs in depth like a ramp by more than this many km (find_trench,
        _ramp_trim), before the datum is built. None (the default) keeps
        v9's trench; generate.py --trench-ramp on writes 3.0.

    Returns (contours, info). `contours` is a list of (level_km, (N, 2)
    lon/lat) with level 0 the trench; `info` holds the diagnostics.
    """
    depth = np.array(depth_msl, dtype=float)
    cut = None
    if bbox is not None:
        lo0, lo1, la0, la1 = bbox
        X, Y = np.meshgrid(x, y)
        outside = (X < lo0) | (X > lo1) | (Y < la0) | (Y > la1)
        cut = np.isfinite(depth) & outside   # slab the clip window removed
        depth[outside] = np.nan
    valid = largest_component(np.isfinite(depth))
    depth = np.where(valid, depth, np.nan)

    tr = find_trench(x, y, depth, cut=cut, ramp_rise_km=ramp_rise_km)
    s = line_km(tr["trench"])
    tdep = running_median(tr["trench_depth_raw"], s, 50.0)
    trench = resample_line(tr["trench"], 2.0)
    tdep = np.interp(line_km(trench), s, tdep)
    D = depth_below_trench(x, y, depth, trench, tdep)

    contours = [(0.0, drop_repeated_vertices(trench))]
    levels = []
    lv = spacing_km
    while lv <= cutoff_km + 1e-9:
        segs = _contour_paths(x, y, D, lv)
        if not segs:
            levels.append({"level": lv, "found": False})
            lv += spacing_km
            continue
        lens = sorted(((line_km(c)[-1], c) for c in segs), key=lambda t: -t[0])
        main = drop_repeated_vertices(lens[0][1])
        extra = [float(round(l, 1)) for l, _ in lens[1:] if l > 0.05 * lens[0][0]]
        closed = bool(np.hypot(*(main[0] - main[-1])) < 1e-9)
        levels.append({"level": lv, "found": True, "extra_pieces_km": extra,
                       "closed": closed})
        contours.append((float(lv), main))
        lv += spacing_km

    untrimmed_km = float(line_km(trench)[-1])
    untrimmed = list(contours)
    ends = None
    if trench_ends is not None:
        if not all(L["found"] and not L["closed"] for L in levels):
            raise ValueError("trench_ends: every level must be one open line before the cut")
        contours, ends = trim_ends(x, y, D, contours, cutoff_km, spacing_km=spacing_km,
                                   cut_points=trench_ends)
        if ends is None or "skipped" in ends:
            raise ValueError(f"trench_ends {trench_ends}: the cut could not be applied "
                             f"({(ends or {}).get('skipped', 'no complete down-dip line')})")
    elif clean_ends and all(L["found"] and not L["closed"] for L in levels):
        contours, ends = trim_ends(x, y, D, contours, cutoff_km, spacing_km=spacing_km)

    # Distance (in cells) from every valid cell to the nearest cell off the
    # slab: a contour whose INTERIOR hugs that edge is not a real depth line
    # but the edge of the SLAB model (or of the clip window) at that depth.
    from scipy import ndimage
    edge_cells = ndimage.distance_transform_edt(np.isfinite(D))
    for L, (_, main) in zip([L for L in levels if L["found"]], contours[1:]):
        # fraction of the contour's interior (its first and last 10% by
        # length excluded: a contour legitimately ends on the edge) that lies
        # within 1.5 cells of the edge of the slab model
        cum = line_km(main)
        inner = (cum > 0.1 * cum[-1]) & (cum < 0.9 * cum[-1])
        ii = np.clip(np.searchsorted(x, main[:, 0]), 0, x.size - 1)
        jj = np.clip(np.searchsorted(y, main[:, 1]), 0, y.size - 1)
        hug = edge_cells[jj, ii] < 1.5
        L.update({"length_km": float(cum[-1]), "n_points": int(len(main)),
                  "edge_fraction": float(hug[inner].mean()) if inner.any() else 0.0})

    info = {
        "trench_length_km": float(line_km(contours[0][1])[-1]),
        "trench_length_untrimmed_km": untrimmed_km,
        "ends": ends,
        "contours_untrimmed": untrimmed,
        "trench_depth_msl": {"p5": float(np.percentile(tdep, 5)),
                             "median": float(np.median(tdep)),
                             "p95": float(np.percentile(tdep, 95))},
        "edge_threshold_km": tr["threshold_km"],
        "trench_runs": tr["n_runs"],
        "clip_edge_km": tr["clip_edge_km"],
        "clip_edge_shallow_km": tr["clip_edge_shallow_km"],
        "ramp_trim_km": tr["ramp_trim_km"],
        "levels": levels,
        "diag": tr,
        "datum_field": D,
    }
    if verbose:
        dd = info["trench_depth_msl"]
        print(f"    trench: {untrimmed_km:.0f} km of the raster's "
              f"{tr['loop_length_km']:.0f} km outer edge; its depth below sea "
              f"level p5/median/p95 = {dd['p5']:.1f}/{dd['median']:.1f}/"
              f"{dd['p95']:.1f} km (this is the local datum)")
        if ramp_rise_km is not None:
            a, b = tr["ramp_trim_km"]
            print(f"    ramp rule (a trench end more than {ramp_rise_km:g} km deeper than the "
                  f"trench's normal depth is not trench): "
                  + (f"start trimmed by {a:.0f} km, end by {b:.0f} km" if a or b else "nothing to trim"))
        if tr["clip_edge_km"] > 0:
            print(f"    clip window: {tr['clip_edge_km']:.0f} km of the edge is the "
                  f"clip line cutting through the slab, never trench "
                  f"({tr['clip_edge_shallow_km']:.0f} km of it would otherwise "
                  f"have passed the trench tests)")
        if ends and "start" in ends:
            print("    zone ends fixed by the given end points (trench_ends):"
                  if trench_ends is not None else
                  f"    clean ends (an artefact end is cut along a down-dip line; "
                  f"median width {ends['median_width_km']:.0f} km):")
            for name in ("start", "end"):
                e = ends[name]
                if e["cut"]:
                    print(f"      {name:5s}: CUT, {e['reason']}; trench shortened by "
                          f"{e['trench_km']:.0f} km, deeper contours by up to "
                          f"{e['contours_km']:.0f} km; the new end is "
                          f"{e['deviation_after_deg']:.0f} deg off square")
                else:
                    print(f"      {name:5s}: {e['reason']}")
            print(f"    trench after the cuts: {info['trench_length_km']:.0f} km")
        elif ends:
            print(f"    clean ends: NOT applied ({ends['skipped']})")
        for L in levels:
            if not L["found"]:
                print(f"    {L['level']:5.1f} km below trench: NOT FOUND")
                continue
            msg = (f"    {L['level']:5.1f} km below trench: one line, "
                   f"{L['length_km']:.0f} km, {L['n_points']} points")
            if L["extra_pieces_km"]:
                msg += (f"  (ignored {len(L['extra_pieces_km'])} smaller "
                        f"piece(s): {L['extra_pieces_km']} km)")
            if L["closed"]:
                msg += "  WARNING: closed loop"
            print(msg)
    return contours, info


def check_contours(contours, info):
    """Problems that must stop the run (returned as a list of strings)."""
    problems = []
    missing = [L["level"] for L in info["levels"] if not L["found"]]
    if missing:
        problems.append(f"levels {missing} km below trench are not in the "
                        f"raster at all (the SLAB model stops shallower than "
                        f"the cutoff)")
    # A level much shorter than the one 5 km above it means the SLAB model
    # stops shallower than that depth along most of the zone. Compared level
    # by level, not with the trench: a zone may narrow gradually with depth
    # (calabria2's trench wraps round the Ionian prism, so its deepest level
    # is 30% of the trench), which the discretiser meshes without defects.
    prev = info["trench_length_km"]
    for L in info["levels"]:
        if L["found"] and L["length_km"] < 0.3 * prev:
            problems.append(f"the {L['level']:g} km level is only "
                            f"{L['length_km']:.0f} km long against {prev:.0f} km "
                            f"for the level above it")
        if L["found"]:
            prev = L["length_km"]
        if L["found"] and L["closed"]:
            problems.append(f"the {L['level']:g} km level is a closed loop")
        # A level in two sizeable pieces is broken (a gap in the SLAB model,
        # or a clip window cutting through it): keeping only the longer piece
        # would silently shorten that row of the mesh. No zone tested has any
        # second piece at all, so 10% is not a tuned threshold.
        big = [e for e in (L.get("extra_pieces_km") or [])
               if L["found"] and e > 0.10 * L["length_km"]]
        if big:
            problems.append(f"the {L['level']:g} km level comes in pieces "
                            f"({L['length_km']:.0f} km + "
                            f"{' + '.join(f'{e:.0f}' for e in big)} km): the "
                            f"SLAB model (or the clip window) breaks it")
        if L["found"] and L.get("edge_fraction", 0.0) > 0.05:
            problems.append(f"{100 * L['edge_fraction']:.0f}% of the "
                            f"{L['level']:g} km level runs along the edge of "
                            f"the SLAB model (or of the clip window): the "
                            f"raster stops at that depth there, so the level "
                            f"is not a real depth contour")
    return problems
