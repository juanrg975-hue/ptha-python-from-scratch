"""Figures of step1.html ("1. Contours from SLAB"), made from the real data.

Run from ptha18_logic_tree_test/ (ROOT):
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_step1.py

Every figure calls lib/slab_contours.py exactly as step 1 does
(below_trench_contours(x, y, depth, cutoff, spacing_km=5.0, bbox=CLIP_BBOX),
with depth = -z of the SLAB raster and the example's own cutoff and clip), and
then reads the intermediate results that function returns in `info`. The
PTHA18 contours come from the files validate_v9.py cached.
Writes html/docs/img/step1_*.png and prints the numbers quoted in the page.
"""

import glob
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PKG = os.path.join(ROOT, "from_scratch_v12")
OUT = os.path.join(HERE, "..", "img")
sys.path.insert(0, os.path.join(PKG, "lib"))
from official_geometry_params import read_grid  # noqa: E402
import slab_contours as sc  # noqa: E402

REF = os.path.join(PKG, "validation", "ptha18_reference", "contours", "SOURCEZONE_CONTOURS")
KER = os.path.join(ROOT, "slab_vs_ptha", "data", "slab2", "ker_slab2_dep_02.24.18.grd")
CAL = os.path.join(PKG, "data", "slab2", "cal_slab2_dep_02.24.18.grd")
CAR = os.path.join(PKG, "data", "slab2", "car_slab2_dep_02.24.18.grd")

TRENCH = "#c2185b"     # as in the step 1 figure
CONT = "#1f3b73"
CUT = "#e65100"
OURS = "#eb6834"       # this run
PTHA = "#2a78d6"       # PTHA18
GREY = "#8a93a3"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white"})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote img/{name}  ({os.path.getsize(p) / 1e3:.0f} KB)")


def aspect(ax, lat):
    ax.set_aspect(1 / np.cos(np.radians(lat)))


def run_step1(grd, cutoff, bbox=None):
    """Exactly what step1_fetch_slab2.py does with the raster."""
    x, y, z = read_grid(grd)
    depth = -z
    contours, info = sc.below_trench_contours(x, y, depth, cutoff, spacing_km=5.0,
                                              bbox=bbox, verbose=False)
    return x, y, depth, contours, info


def masked_depth(x, y, depth, bbox=None):
    """The depth below_trench_contours works on: clip window, largest block."""
    d = np.array(depth, dtype=float)
    if bbox is not None:
        X, Y = np.meshgrid(x, y)
        d[(X < bbox[0]) | (X > bbox[1]) | (Y < bbox[2]) | (Y > bbox[3])] = np.nan
    valid = sc.largest_component(np.isfinite(d))
    return np.where(valid, d, np.nan)


# ---------------------------------------------------------------------------
# boundary classification (method steps 1-2)
# ---------------------------------------------------------------------------

def boundary_classes(info):
    d = info["diag"]
    loop, score, is_tr = d["loop"], d["score"], d["is_trench"]
    shallow = d["edge_depth"] <= d["threshold_km"]
    up = score >= 0.5
    from scipy.spatial import cKDTree
    _, idx = cKDTree(loop).query(d["trench_raw"])
    chosen = np.zeros(len(loop), bool)
    chosen[idx] = True
    run = np.zeros(len(loop), bool)
    for i0, m in sc._runs(is_tr):
        members = (i0 + np.arange(m)) % len(loop)
        if chosen[members].any():
            run[members] = True
    cls = np.full(len(loop), "deep", dtype=object)
    cls[shallow & ~up] = "lateral"
    cls[is_tr & ~run] = "otherrun"
    cls[run & ~chosen] = "trimmed"
    cls[chosen] = "trench"
    cls[is_tr & ~(shallow & up) & ~(run & ~chosen)] = "bridged"
    return cls, shallow, up, chosen


CLS_STYLE = {
    "deep": ("#b8bec9", "deeper than the threshold: not trench"),
    "lateral": ("#d9822b", "shallow, but the slab does not deepen inward (lateral edge)"),
    "otherrun": ("#f4a6c6", "a shorter trench-like run (not kept)"),
    "bridged": ("#8a5cd1", "short gap bridged into the trench (shallow, noisy gradient)"),
    "trimmed": ("#e8c21a", "end of the run trimmed (cosine below 0.7)"),
    "trench": (TRENCH, "the trench: longest shallow, up-dip facing run"),
}


def fig_classes(name, x, y, depth, info, title, fname, lat_ref, box=None):
    d = info["diag"]
    cls, shallow, up, chosen = boundary_classes(info)
    loop = d["loop"]
    s = sc.line_km(np.vstack([loop, loop[:1]]))[:-1]
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], hspace=0.3, wspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    dm = masked_depth(x, y, depth)
    pc = ax.pcolormesh(x, y, np.where(dm < 150, dm, np.nan), cmap="YlGnBu", shading="auto",
                       vmin=0, vmax=150, alpha=0.8)
    cb = fig.colorbar(pc, ax=ax, shrink=0.5, pad=0.03, fraction=0.05)
    cb.set_label("SLAB depth below sea level (km)")
    for k, (c, lab) in CLS_STYLE.items():
        m = cls == k
        if m.any():
            ax.scatter(loop[m, 0], loop[m, 1], s=9 if k != "trench" else 12, color=c,
                       label=f"{lab} ({m.sum()})", zorder=3)
    ax.plot(*loop[0], "k^", ms=8, zorder=4)
    ax.annotate("edge km 0", loop[0], xytext=(8, -4), textcoords="offset points", fontsize=9)
    if box:
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
    aspect(ax, lat_ref)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"{title}: outer edge, classified")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.12), fontsize=8.5, frameon=False)

    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    for a in (a1, a2):
        a.fill_between(s, 0, 1, where=chosen, transform=a.get_xaxis_transform(),
                       color=TRENCH, alpha=0.10, lw=0)
    a1.plot(s, d["edge_depth"], color="#1d2433", lw=1.2, label="edge depth (50 km running median)")
    a1.axhline(d["threshold_km"], color="#c43d3d", ls="--", lw=1.2,
               label=f"threshold = max(5th percentile + 12, 15) = {d['threshold_km']:.1f} km")
    a1.set_ylabel("depth below sea level (km)")
    a1.set_ylim(min(0, np.nanmin(d["edge_depth"])), min(200, np.nanmax(d["edge_depth"]) * 1.05))
    a1.invert_yaxis()
    a1.legend(fontsize=8.5, loc="lower right")
    a1.set_title("along the edge: how deep is it here?")
    a2.plot(s, d["score"], color="#1d2433", lw=0.9, label="cosine(depth gradient, inward normal)")
    a2.axhline(0.5, color="#c43d3d", ls="--", lw=1.2, label="0.5: counts as up-dip facing")
    a2.axhline(0.7, color="#b7791f", ls=":", lw=1.4, label="0.7: the trench's two ends must reach this")
    a2.set_ylim(-1.05, 1.05)
    a2.set_ylabel("up-dip score")
    a2.set_xlabel("distance along the raster's outer edge from 'edge km 0' (km)")
    a2.legend(fontsize=8.5, loc="lower right")
    a2.set_title("along the edge: does the slab get deeper going inward? (+1 = yes)")
    a1.text(0.01, 0.97, "shaded: the part kept as trench", transform=a1.transAxes, fontsize=9,
            color=TRENCH, va="top")
    save(fig, fname)
    n = {k: int((cls == k).sum()) for k in CLS_STYLE}
    print(f"  {name}: loop {len(loop)} vertices, {s[-1] + 0:.0f} km; threshold {d['threshold_km']:.2f} km; "
          f"runs {d['n_runs']}; classes {n}")
    seg = sc.line_km(d["trench_raw"])[-1]
    print(f"  {name}: kept trench (raw) {seg:.0f} km")


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    print("kermadectonga2 (cutoff 40 km, no clip), as in kermadectonga2_v9/steps/step1")
    x, y, depth, contours, info = run_step1(KER, 40.0)
    d = info["diag"]
    dm = masked_depth(x, y, depth)
    D = info["datum_field"]
    unt = info["contours_untrimmed"]
    ends = info["ends"]
    print("  trench untrimmed %.0f km, after cuts %.0f km" % (info["trench_length_untrimmed_km"],
                                                              info["trench_length_km"]))

    # 1. boundary classification ------------------------------------------------
    fig_classes("kermadectonga2", x, y, depth, info, "kermadectonga2",
                "step1_ker_boundary_classes.png", -29)

    # 2. trench depth along strike ---------------------------------------------
    from scipy.spatial import cKDTree
    s_raw = sc.line_km(d["trench"])
    raw = d["trench_depth_raw"]
    med = sc.running_median(raw, s_raw, 50.0)
    valid = np.isfinite(dm)
    jj, ii = np.nonzero(valid)
    _, k0 = cKDTree(np.column_stack([x[ii], y[jj]])).query(d["trench_raw"])
    nearest = dm[jj[k0], ii[k0]]
    print("  edge depth: extrapolated median %.2f, nearest-cell median %.2f, "
          "median(nearest - extrapolated) %.2f km" % (np.median(raw), np.median(nearest),
                                                      np.median(nearest - raw)))
    print("  running median p5/median/p95 on 2-km trench: ", info["trench_depth_msl"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(s_raw, nearest, ".", color="#b8bec9", ms=3, label="depth of the nearest cell centre (not used)")
    ax.plot(s_raw, raw, ".", color="#5b6475", ms=3, label="edge depth, extrapolated to the edge line (5 x 5 plane)")
    ax.plot(s_raw, med, "-", color=TRENCH, lw=2.2, label="50 km running median = the local datum")
    ax.axhline(8.0, color="#c43d3d", ls="--", lw=1.5, label="one trench depth for the whole zone (Berryman's 'Tonga', 8 km)")
    L = s_raw[-1]
    cs, ce = ends["start"]["trench_km"], ends["end"]["trench_km"]
    ax.axvspan(0, cs, color="#f0f2f5", zorder=0)
    ax.axvspan(L - ce, L, color="#f0f2f5", zorder=0)
    ax.text(cs / 2, 16.6, "cut later\n(clean ends)", ha="center", fontsize=8.5, color="#5b6475")
    ax.invert_yaxis()
    ax.set_ylim(17, 2.5)
    ax.set_xlim(0, L)
    ax.set_xlabel("distance along the trench (km), from the south end (New Zealand) to the north end (Tonga)")
    ax.set_ylabel("trench depth below sea level (km)")
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.set_title("kermadectonga2: the trench is not at one depth")
    save(fig, "step1_ker_trench_depth.png")

    # 3. constant datum vs local datum -----------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 9.6), sharey=True)
    colors = ["#c43d3d", "#2f6fdf", "#2e9e6b", "#8a5cd1", "#d9822b", "#1d2433"]
    for ax in axs:
        ax.pcolormesh(x, y, np.where(dm < 120, dm, np.nan), cmap="Greys", shading="auto",
                      alpha=0.35, vmin=0, vmax=150)
        aspect(ax, -29)
        ax.set_xlabel("longitude (deg)")
        ax.set_xlim(172, 189)
    for L_, lw in ((0, 2.6), (5, 1.2)):
        segs = sc._contour_paths(x, y, dm, 8.0 + L_)
        segs = sorted(segs, key=lambda c: -sc.line_km(c)[-1])
        lens = [sc.line_km(c)[-1] for c in segs]
        print(f"  constant datum 8 km, level {L_} (= {8 + L_} km below sea level): "
              f"{len(segs)} pieces, {[round(v) for v in lens]} km")
        for i, c in enumerate(segs):
            axs[0].plot(c[:, 0], c[:, 1], color=colors[i % len(colors)] if L_ == 0 else "#1d2433",
                        lw=lw, ls="-" if L_ == 0 else "--")
    axs[0].plot([], [], color="#c43d3d", lw=2.6, label="'0 km' = 8 km below sea level:\n6 pieces (one colour each)")
    axs[0].plot([], [], color="#1d2433", lw=1.2, ls="--", label="'5 km' = 13 km below sea level:\n2 pieces")
    axs[0].set_title("if one datum were used for the whole zone (8 km)")
    axs[0].annotate("gap: the trench is 13 km deep here,\nso 13 km below sea level is\nthe edge of the data, not a line",
                    xy=(184.6, -25.2), xytext=(172.6, -21.8), fontsize=9, color="#c43d3d",
                    arrowprops=dict(arrowstyle="->", color="#c43d3d", lw=1.4))
    axs[0].annotate("most of the arc has no\n'0 km' line at all", xy=(186.6, -20.0), xytext=(172.6, -17.0),
                    fontsize=9, color="#c43d3d", arrowprops=dict(arrowstyle="->", color="#c43d3d", lw=1.4))
    axs[0].legend(loc="lower right", fontsize=9)
    axs[0].set_ylabel("latitude (deg)")
    for lv, c in unt[:2]:
        axs[1].plot(c[:, 0], c[:, 1], color=TRENCH if lv == 0 else "#1d2433",
                    lw=2.6 if lv == 0 else 1.2, ls="-" if lv == 0 else "--",
                    label="0 km: the trench, one line" if lv == 0 else "5 km below the nearby trench: one line")
    axs[1].set_title("step 1: depth below the NEARBY trench")
    axs[1].legend(loc="lower right", fontsize=9)
    save(fig, "step1_ker_constant_vs_local.png")

    # 4. depth, datum, D --------------------------------------------------------
    datum = np.where(np.isfinite(D), dm - D, np.nan)
    print("  datum field range on valid cells: %.2f to %.2f km" % (np.nanmin(datum), np.nanmax(datum)))
    fig, axs = plt.subplots(1, 3, figsize=(13, 8.6), sharey=True)
    p0 = axs[0].pcolormesh(x, y, np.where(dm <= 60, dm, np.nan), cmap="YlGnBu", shading="auto", vmin=0, vmax=60)
    axs[0].set_title("1. SLAB depth below sea level\n(only the first 60 km shown)")
    fig.colorbar(p0, ax=axs[0], orientation="horizontal", pad=0.07, shrink=0.9, label="km below sea level")
    p1 = axs[1].pcolormesh(x, y, np.where(dm <= 60, datum, np.nan), cmap="magma_r", shading="auto",
                           vmin=4, vmax=12)
    axs[1].plot(unt[0][1][:, 0], unt[0][1][:, 1], color=TRENCH, lw=1.2)
    axs[1].set_title("2. datum: trench depth at the nearest\ntrench point, smoothed 25 km")
    fig.colorbar(p1, ax=axs[1], orientation="horizontal", pad=0.07, shrink=0.9, label="km below sea level")
    p2 = axs[2].pcolormesh(x, y, np.where(dm <= 60, D, np.nan), cmap="viridis", shading="auto", vmin=-2, vmax=50)
    for lv, c in unt:
        axs[2].plot(c[:, 0], c[:, 1], color="white" if lv else TRENCH, lw=0.8 if lv else 1.4)
    axs[2].set_title("3. D = depth - datum, contoured\nat 0, 5, ..., 40 km")
    fig.colorbar(p2, ax=axs[2], orientation="horizontal", pad=0.07, shrink=0.9, label="km below the nearby trench")
    for ax in axs:
        aspect(ax, -29)
        ax.set_xlim(172, 189)
        ax.set_xlabel("longitude (deg)")
    axs[0].set_ylabel("latitude (deg)")
    save(fig, "step1_ker_datum_fields.png")

    # 5. transects, widths and the end cuts -----------------------------------
    trench_u = unt[0][1]
    s, pts, complete, length, paths = sc.down_dip_transects(x, y, D, trench_u, 40.0, spacing_km=5.0)
    medw = float(np.median(length[complete]))
    ok = np.nonzero(complete)[0]
    width = np.full(s.size, np.nan)
    for k in ok:
        near = ok[np.abs(s[ok] - s[k]) <= 12.5]
        width[k] = np.median(length[near])
    print("  transects: %d, complete %d, median width %.1f km (info says %.1f)" % (
        s.size, complete.sum(), medw, ends["median_width_km"]))
    print("  first complete transect width ratio %.3f, last %.3f" % (width[ok[0]] / medw, width[ok[-1]] / medw))

    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.28, wspace=0.12)
    boxes = [("south end: narrows to a point", (169.6, 176.2, -43.9, -39.4), -41.5, "start"),
             ("north end: crooked edge of the data", (183.3, 188.1, -17.6, -14.1), -16, "end")]
    for col, (ttl, (lo0, lo1, la0, la1), latr, e) in enumerate(boxes):
        ax = fig.add_subplot(gs[0, col])
        ax.pcolormesh(x, y, np.where(dm < 120, dm, np.nan), cmap="Greys", shading="auto",
                      alpha=0.3, vmin=0, vmax=150)
        for k, p in enumerate(paths):
            if not ((p[:, 0] > lo0) & (p[:, 0] < lo1) & (p[:, 1] > la0) & (p[:, 1] < la1)).any():
                continue
            ax.plot(p[:, 0], p[:, 1], color="#2e9e6b" if complete[k] else GREY,
                    lw=0.7, alpha=0.9 if complete[k] else 0.7)
        for lv, c in unt:
            ax.plot(c[:, 0], c[:, 1], ":", color="0.35", lw=0.9)
        for lv, c in contours:
            ax.plot(c[:, 0], c[:, 1], color=TRENCH if lv == 0 else CONT, lw=2.2 if lv == 0 else 1.0)
        cl = ends[e]["cut_line"]
        ax.plot(cl[:, 0], cl[:, 1], color=CUT, lw=3)
        ax.set_xlim(lo0, lo1)
        ax.set_ylim(la0, la1)
        aspect(ax, latr)
        ax.set_title(ttl)
        ax.set_xlabel("longitude (deg)")
    fig.axes[0].set_ylabel("latitude (deg)")
    ax0 = fig.axes[0]
    ax0.plot([], [], color="#2e9e6b", lw=1, label="down-dip line that reaches 40 km")
    ax0.plot([], [], color=GREY, lw=1, label="down-dip line that leaves the data first")
    ax0.plot([], [], ":", color="0.35", label="contours before the cut")
    ax0.plot([], [], color=CONT, label="contours after the cut")
    ax0.plot([], [], color=CUT, lw=3, label="the cut (one down-dip line)")
    ax0.legend(loc="upper left", fontsize=8.5)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(s[complete], length[complete], ".", color="#2e9e6b", ms=3, label="length of each down-dip line that reaches 40 km")
    ax.plot(s, width, "-", color="#1d2433", lw=1.4, label="width: median over 25 km of trench")
    ax.plot(s[~complete], np.zeros((~complete).sum()) + 5, "|", color=GREY, ms=6, label="lines that leave the data first")
    for f, lab, cc in ((1.0, "median width %.0f km" % medw, "#1d2433"), (0.7, "70 %: where a taper cut goes", "#b7791f"),
                       (0.5, "50 %: below this the tip is a taper", "#c43d3d")):
        ax.axhline(f * medw, color=cc, ls="--", lw=1.1, label=lab)
    ax.axvline(ends["start"]["trench_km"], color=CUT, lw=2.5)
    ax.axvline(s[-1] - ends["end"]["trench_km"], color=CUT, lw=2.5)
    ax.text(ends["start"]["trench_km"] + 20, 1.05 * np.nanmax(width), "south cut: %.0f km of trench removed" % ends["start"]["trench_km"],
            color=CUT, fontsize=9)
    ax.text(s[-1] - 25, 1.05 * np.nanmax(width), "north cut: %.0f km" % ends["end"]["trench_km"], color=CUT, fontsize=9, ha="right")
    ax.set_xlim(0, s[-1])
    ax.set_ylim(0, 1.12 * np.nanmax(width))
    ax.set_xlabel("distance along the trench from the south end (km)")
    ax.set_ylabel("width of the zone (km)")
    ax.legend(fontsize=8.5, ncol=3, loc="lower center")
    save(fig, "step1_ker_transects.png")

    # 6. the end-deviation metric: the mini-mesh --------------------------------
    from shapely.geometry import LineString, Point
    rows = max(2, int(round(medw / 50.0)))
    print("  rows used by the metric:", rows)

    def mini_mesh(cts, end, along_km=50.0):
        """The two end columns _end_deviation rebuilds (same code)."""
        tline = LineString(cts[0][1])
        E_, C_ = [], []
        for _, c in cts:
            c = np.asarray(c, dtype=float)
            same = tline.project(Point(c[0])) <= tline.project(Point(c[-1]))
            if (end == "start") != same:
                c = c[::-1]
            cum = sc.line_km(c)
            dd = min(along_km, cum[-1])
            E_.append(c[0])
            C_.append([np.interp(dd, cum, c[:, 0]), np.interp(dd, cum, c[:, 1])])

        def nodes(p, n):
            p = np.asarray(p)
            cum = sc.line_km(p)
            t = np.linspace(0.0, cum[-1], n + 1)
            return np.column_stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])])

        def corner(p, a, b):
            k = np.cos(np.radians(p[1]))
            u = np.array([(a[0] - p[0]) * k, a[1] - p[1]])
            v = np.array([(b[0] - p[0]) * k, b[1] - p[1]])
            ca = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
            return abs(90.0 - np.degrees(np.arccos(np.clip(ca, -1.0, 1.0))))
        E, C = nodes(E_, rows), nodes(C_, rows)
        cor = [(corner(E[r], E[r + 1], C[r]), E[r]) for r in range(rows)] + \
              [(corner(E[r + 1], E[r], C[r + 1]), E[r + 1]) for r in range(rows)]
        worst = max(cor, key=lambda t: t[0])
        dev = sc._end_deviation(cts, tline, end, rows)
        assert abs(dev - worst[0]) < 1e-9, (dev, worst[0])
        return E, C, np.asarray(E_), worst

    fig, axs = plt.subplots(2, 2, figsize=(11, 10.5))
    axs = axs.ravel()
    panels = [("north end, before: ", unt, "end", boxes[1][1], -16),
              ("north end, after: ", contours, "end", boxes[1][1], -16),
              ("south end, before: ", unt, "start", (169.6, 176.2, -43.9, -39.4), -41.5),
              ("south end, after: ", contours, "start", (169.6, 176.2, -43.9, -39.4), -41.5)]
    for ax, (ttl, cts, e, bb, latr) in zip(axs, panels):
        E, C, Eraw, worst = mini_mesh(cts, e)
        for lv, c in cts:
            ax.plot(c[:, 0], c[:, 1], color=TRENCH if lv == 0 else CONT, lw=1.8 if lv == 0 else 0.7)
        polys = [[E[r], E[r + 1], C[r + 1], C[r]] for r in range(rows)]
        ax.add_collection(PolyCollection(polys, facecolor="#fcf0e2", edgecolor=CUT, lw=1.6, alpha=0.9, zorder=3))
        ax.plot(E[:, 0], E[:, 1], "o", color=CUT, ms=5, zorder=4)
        ax.plot(C[:, 0], C[:, 1], "s", color=CUT, ms=4, zorder=4)
        ax.plot(Eraw[:, 0], Eraw[:, 1], "x", color="#1d2433", ms=5, zorder=4)
        ax.plot(*worst[1], "o", ms=16, mfc="none", mec="#c43d3d", mew=2, zorder=5)
        cxy = np.vstack([E, C]).mean(axis=0)
        ax.set_xlim(cxy[0] - 1.0, cxy[0] + 1.0)
        ax.set_ylim(cxy[1] - 0.8, cxy[1] + 0.8)
        aspect(ax, latr)
        ax.set_title(f"{ttl}worst corner {worst[0]:.0f} deg off square", fontsize=10.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlabel("longitude (deg)")
        ax.set_ylabel("latitude (deg)")
        print(f"  {ttl}{worst[0]:.1f} deg")
    axs[0].plot([], [], "x", color="#1d2433", label="end point of each contour")
    axs[0].plot([], [], "o", color=CUT, label="end column: those ends, split in 3 rows")
    axs[0].plot([], [], "s", color=CUT, label="next column, 50 km in")
    axs[0].plot([], [], "o", ms=10, mfc="none", mec="#c43d3d", mew=2, label="worst corner")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(hspace=0.3, bottom=0.09)
    save(fig, "step1_ker_minimesh.png")

    # 7. trench offset vs PTHA18 --------------------------------------------------
    import geopandas as gpd
    ref = gpd.read_file(os.path.join(REF, "kermadectonga2.shp"))
    R = {round(r.level, 1): np.asarray(r.geometry.coords) for r in ref.itertuples()}
    tr0 = contours[0][1]
    k = int(np.argmin(np.abs(tr0[:, 1] + 24.3)))
    cx, cy = tr0[k]
    lo0, lo1, la0, la1 = cx - 0.42, cx + 0.33, cy - 0.33, cy + 0.33
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    xe = np.concatenate([x - 0.025, [x[-1] + 0.025]])
    ye = np.concatenate([y - 0.025, [y[-1] + 0.025]])
    ax.pcolormesh(xe, ye, np.where(np.isfinite(dm), dm, np.nan), cmap="YlGnBu", vmin=0, vmax=40,
                  edgecolors="white", linewidth=0.6, alpha=0.75)
    X, Y = np.meshgrid(x, y)
    inb = (X > lo0) & (X < lo1) & (Y > la0) & (Y < la1) & np.isfinite(dm)
    ax.plot(X[inb], Y[inb], ".", color="#1d2433", ms=3)
    loop = d["loop"]
    ax.plot(loop[:, 0], loop[:, 1], color="0.35", lw=1, ls=":")
    ax.plot(tr0[:, 0], tr0[:, 1], color=OURS, lw=3, label="this run: 0 km line (the trench)")
    ax.plot(R[0.0][:, 0], R[0.0][:, 1], color=PTHA, lw=3, label="PTHA18: 0 km line")
    c5 = [c for lv, c in contours if lv == 5.0][0]
    ax.plot(c5[:, 0], c5[:, 1], color=OURS, lw=1.4, ls="--", label="this run: 5 km")
    ax.plot(R[5.0][:, 0], R[5.0][:, 1], color=PTHA, lw=1.4, ls="--", label="PTHA18: 5 km")
    ax.plot([], [], ":", color="0.35", label="outer boundary of the valid cells")
    ax.plot([], [], ".", color="#1d2433", label="cell centres (0.05 deg = 5.5 km apart)")
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)
    aspect(ax, cy)
    ax.set_xlabel("longitude (deg, 0-360)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("kermadectonga2 near %.1f S: where the two 0 km lines sit" % -cy)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    save(fig, "step1_ker_trench_offset.png")

    # PTHA18 0 km points on valid cells
    ii_ = np.clip(np.round((R[0.0][:, 0] - x[0]) / 0.05).astype(int), 0, x.size - 1)
    jj_ = np.clip(np.round((R[0.0][:, 1] - y[0]) / 0.05).astype(int), 0, y.size - 1)
    rs = sc.resample_line(R[0.0], 1.0)
    ii2 = np.clip(np.round((rs[:, 0] - x[0]) / 0.05).astype(int), 0, x.size - 1)
    jj2 = np.clip(np.round((rs[:, 1] - y[0]) / 0.05).astype(int), 0, y.size - 1)
    os_ = sc.resample_line(tr0, 1.0)
    ii3 = np.clip(np.round((os_[:, 0] - x[0]) / 0.05).astype(int), 0, x.size - 1)
    jj3 = np.clip(np.round((os_[:, 1] - y[0]) / 0.05).astype(int), 0, y.size - 1)
    print("  PTHA18 0 km vertices on a valid cell: %.1f%% (%d vertices); every km along it: %.1f%%; "
          "this run's 0 km, every km: %.1f%%" % (
              100 * np.isfinite(dm[jj_, ii_]).mean(), len(R[0.0]), 100 * np.isfinite(dm[jj2, ii2]).mean(),
              100 * np.isfinite(dm[jj3, ii3]).mean()))

    # 8. contours over the raster vs PTHA18 (south end and a middle stretch) -----
    fig, axs = plt.subplots(1, 2, figsize=(12, 7.2))
    for ax, (ttl, (lo0, lo1, la0, la1), latr) in zip(axs, [
            ("south end: the automatic cut and PTHA18's hand cut", (171.0, 176.6, -43.2, -38.6), -41),
            ("north end", (183.6, 188.0, -17.3, -14.0), -15.5)]):
        ax.pcolormesh(x, y, np.where(dm < 120, dm, np.nan), cmap="Greys", shading="auto", alpha=0.3,
                      vmin=0, vmax=150)
        for lv, c in contours:
            ax.plot(c[:, 0], c[:, 1], color=OURS, lw=2 if lv == 0 else 0.9)
        for lv, c in R.items():
            ax.plot(c[:, 0], c[:, 1], color=PTHA, lw=2 if lv == 0 else 0.9, ls="--")
        ax.set_xlim(lo0, lo1)
        ax.set_ylim(la0, la1)
        aspect(ax, latr)
        ax.set_title(ttl)
        ax.set_xlabel("longitude (deg)")
    axs[0].set_ylabel("latitude (deg)")
    axs[0].plot([], [], color=OURS, label="this run (0, 5, ..., 40 km)")
    axs[0].plot([], [], color=PTHA, ls="--", label="PTHA18's published contours")
    axs[0].legend(loc="upper left", fontsize=9)
    save(fig, "step1_ker_vs_ptha18_ends.png")

    # distance of the cuts to PTHA18's ends, along the trench
    tl = LineString(contours[0][1])
    p_s, p_n = Point(R[0.0][np.argmin(R[0.0][:, 1])]), Point(R[0.0][np.argmax(R[0.0][:, 1])])
    print("  PTHA18 0 km ends: south", np.round(p_s.coords[0], 3), "north", np.round(p_n.coords[0], 3))
    print("  this run 0 km ends: start", np.round(contours[0][1][0], 3), "end", np.round(contours[0][1][-1], 3))

    # calabria2 ------------------------------------------------------------------
    print("calabria2 (cutoff 60 km, no clip)")
    xc, yc, dc, cc, ic = run_step1(CAL, 60.0)
    fig_classes("calabria2", xc, yc, dc, ic, "calabria2", "step1_cal_boundary_classes.png", 38,
                box=(11.5, 21.0, 34.8, 42.2))

    # antilles2 --clip -----------------------------------------------------------
    print("antilles2 clip windows (cutoff 45 km)")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)
    for ax, bb in zip(axs, [(297.0, 307.0, 5.0, 18.5), (283.0, 307.0, 5.0, 18.5)]):
        xa, ya, da, ca, ia = run_step1(CAR, 45.0, bbox=bb)
        probs = sc.check_contours(ca, ia)
        print(f"  clip {bb}: trench {ia['trench_length_km']:.0f} km, check_contours problems: {probs}")
        ax.pcolormesh(xa, ya, np.where(da < 150, da, np.nan), cmap="Greys", shading="auto", alpha=0.12,
                      vmin=0, vmax=150)
        used = masked_depth(xa, ya, da, bb)
        ax.pcolormesh(xa, ya, np.where(used < 150, used, np.nan), cmap="YlGnBu", shading="auto", alpha=0.6,
                      vmin=0, vmax=150)
        for lv, c in ia["contours_untrimmed"]:
            ax.plot(c[:, 0], c[:, 1], ":", color="0.3", lw=0.8)
        ax.add_patch(plt.Rectangle((bb[0], bb[2]), bb[1] - bb[0], bb[3] - bb[2], fill=False,
                                   ec="#2f6fdf", lw=2, ls="--"))
        for lv, c in ca:
            ax.plot(c[:, 0], c[:, 1], color=TRENCH if lv == 0 else CONT, lw=2.2 if lv == 0 else 0.8)
        for e in ("start", "end"):
            if ia["ends"] and ia["ends"].get(e, {}).get("cut"):
                cl = ia["ends"][e]["cut_line"]
                ax.plot(cl[:, 0], cl[:, 1], color=CUT, lw=2.5)
        ax.set_title("--clip %g,%g,%g,%g" % bb + ("\nchecks: none failed" if not probs else "\nchecks: " + "; ".join(probs)),
                     fontsize=10.5)
        aspect(ax, 14)
        ax.set_xlim(283, 307)
        ax.set_ylim(5, 26)
        ax.set_xlabel("longitude (deg, 0-360)")
    axs[0].set_ylabel("latitude (deg)")
    axs[0].plot([], [], color="#2f6fdf", ls="--", lw=2, label="the --clip window")
    axs[0].plot([], [], color=TRENCH, lw=2.2, label="trench found")
    axs[0].plot([], [], color=CONT, lw=0.8, label="contours")
    axs[0].plot([], [], color=CUT, lw=2.5, label="end cuts")
    axs[0].plot([], [], ":", color="0.3", lw=0.8, label="removed by the end cuts")
    axs[0].plot([], [], "s", color="#7fcdbb", ms=9, label="SLAB cells step 1 uses")
    axs[0].legend(loc="upper left", fontsize=9)
    axs[1].annotate("a strip of the Puerto Rico slab is inside\nthe window and joined to the Antilles slab;\nno check flags it. Here the north taper\ncut happens to remove it", xy=(293, 18.35), xytext=(283.6, 21.4),
                    arrowprops=dict(arrowstyle="->", color="#c43d3d", lw=1.5), color="#c43d3d", fontsize=10)
    save(fig, "step1_antilles2_clip.png")

    # accuracy vs PTHA18 ---------------------------------------------------------
    print("distance to PTHA18's contours, per level (median over this run's line, 1 km steps)")
    accuracy()

    # copies of the step 1 figures of the examples --------------------------------
    for z in ("kermadectonga2",):
        src = os.path.join(ROOT, f"{z}_v9", "figures", "step1_trench_and_contours.png")
        dst = os.path.join(OUT, f"step1_{z}_trench_and_contours.png")
        shutil.copyfile(src, dst)
        print(f"  copied {z}_v9/figures/step1_trench_and_contours.png -> img/{os.path.basename(dst)}")


def accuracy():
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import LineString  # noqa: F401
    from shapely.ops import transform

    def unwrap(g):
        return transform(lambda a, b, c=None: (np.where(np.asarray(a) < 0, np.asarray(a) + 360, a), b), g)

    table = {}
    for z in ["kermadectonga2", "puysegur2", "makran2", "kurilsjapan", "southamerica"]:
        ours = gpd.read_file(glob.glob(os.path.join(ROOT, f"{z}_v9", "inputs", "geometry", "*contours*.shp"))[0])
        ref = gpd.read_file(os.path.join(REF, f"{z}.shp"))
        ours["geometry"] = ours.geometry.apply(unwrap)
        ref["geometry"] = ref.geometry.apply(unwrap)
        b = ref.total_bounds
        tr = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lon_0={(b[0] + b[2]) / 2} "
                                  f"+lat_0={(b[1] + b[3]) / 2} +units=km", always_xy=True)
        O = {round(r.level, 1): transform(tr.transform, r.geometry) for r in ours.itertuples()}
        Rf = {round(r.level, 1): transform(tr.transform, r.geometry) for r in ref.itertuples()}
        deepest = Rf[max(Rf)]
        rows = []
        for L in sorted(set(O) & set(Rf)):
            a, bl = O[L], Rf[L]
            d, sg = [], []
            for t in np.arange(0, a.length, 1.0):
                p = a.interpolate(t)
                s_ = bl.project(p)
                if s_ <= 1e-6 or s_ >= bl.length - 1e-6:
                    continue                       # beyond PTHA18's along-strike extent
                q = bl.interpolate(s_)
                d.append(p.distance(q))
                # + = this run's line is on the trench (seaward, up-dip) side of PTHA18's
                if L == 0:
                    sg.append(1 if p.distance(deepest) > q.distance(deepest) else -1)
                else:
                    sg.append(1 if p.distance(Rf[0.0]) < q.distance(Rf[0.0]) else -1)
            d, sg = np.array(d), np.array(sg)
            rows.append((L, float(np.median(d)), float(np.median(d * sg))))
        table[z] = rows
        print("  %-15s " % z + "  ".join("%g: %.1f (%+.1f)" % r for r in rows))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cols = {"kermadectonga2": "#c2185b", "puysegur2": "#2e9e6b", "makran2": "#d9822b",
            "kurilsjapan": "#2f6fdf", "southamerica": "#8a5cd1"}
    for z, rows in table.items():
        r = np.array(rows)
        ax.plot(r[:, 0], r[:, 2], "o-", color=cols[z], label=z, lw=1.6, ms=5)
    ax.axhline(0, color="#1d2433", lw=0.8)
    ax.set_xlabel("contour level (km below the nearby trench)")
    ax.set_ylabel("median signed offset (km)")
    ax.text(0.99, 0.96, "+ : this run's line lies on the trench side of PTHA18's", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#5b6475")
    ax.text(0.99, 0.04, "- : this run's line lies deeper (landward) than PTHA18's", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#5b6475")
    ax.set_ylim(-4.5, 5.5)
    ax.set_xticks(range(0, 60, 5))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    save(fig, "step1_accuracy_vs_ptha18.png")


if __name__ == "__main__":
    main()
