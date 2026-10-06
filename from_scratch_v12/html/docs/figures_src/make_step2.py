"""Figures for html/docs/step2.html (the mesh and --discretizer).

Run from ptha18_logic_tree_test/ (ROOT):

    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_step2.py

Everything is computed from real files with the package's own functions
(pyptha_v12/contour_discretisation.py, lib/slab_contours.py):

  step2_kermadectonga2_build.png   contours -> down-dip lines -> cells, on the
                                   kermadectonga2_v9 example (step 1's shapefile)
  step2_southamerica_repair.png    rptha's mesh on southamerica's contours
                                   BEFORE step 1's clean ends (it has defects),
                                   what --discretizer optimal does with it, and
                                   the final v8 mesh
  step2_kermadectonga2_lm_vs_mid.png  optimal (= lm here) vs mid on kermadectonga2
  step2_final_meshes.png           the 8 examples' final meshes (saved grids)

Slow parts (the southamerica meshes, about 10 minutes the first time) are
cached in STEP2_FIG_CACHE (default: a folder in the system temp directory).
The numbers quoted on the page are printed to the console.
"""

import contextlib
import io
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # from_scratch_v12/
ROOT = os.path.dirname(PKG)                                       # ptha18_logic_tree_test/
OUT = os.path.join(HERE, "..", "img")
CACHE = os.environ.get("STEP2_FIG_CACHE",
                       os.path.join(tempfile.gettempdir(), "ptha_step2_fig_cache"))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))

from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import unit_sources as us  # noqa: E402

INK, MUTED, LINE = "#1d2433", "#5b6475", "#dde1e8"
SCRATCH, OFFICIAL = "#eb6834", "#2a78d6"
BAD, WARN, GOOD = "#c43d3d", "#b7791f", "#2e9e6b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 9.5,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "axes.edgecolor": "#c3c8d2", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
})
DEPTH_CMAP = plt.get_cmap("Oranges")
EXAMPLES = ["kermadectonga2", "southamerica", "kurilsjapan", "makran2",
            "puysegur2", "calabria2", "caribbean2", "antilles2"]


# ---------------------------------------------------------------- helpers
def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)
    val = fn()
    with open(p, "wb") as fh:
        pickle.dump(val, fh)
    return val


def shapefile_contours(zone):
    """Step 1's contours, read exactly as step 2 reads them."""
    import geopandas as gpd
    g = gpd.read_file(os.path.join(ROOT, f"{zone}_v9", "inputs", "geometry",
                                   f"{zone}_slab2_contours.shp"))
    out = []
    for _, r in g.iterrows():
        geom = r.geometry
        if geom.geom_type != "LineString":
            geom = sorted(geom.geoms, key=lambda q: len(q.coords))[-1]
        out.append((float(r["level"]), np.asarray(geom.coords)[:, :2]))
    return sorted(out, key=lambda t: t[0])


def step1_contours_both(zone):
    """(clean-ended, untrimmed) contours, recomputed with step 1's own call."""
    def run():
        import slab_contours as sc
        from official_geometry_params import read_grid
        info = json.load(open(os.path.join(ROOT, f"{zone}_v9", "data",
                                           "step1_contours_info.json")))
        x, y, z = read_grid(os.path.join(ROOT, info["raster"]))
        cont, inf = sc.below_trench_contours(
            x, y, -z, info["cutoff_km_below_trench"], bbox=info["clip_bbox"],
            verbose=False)
        return cont, inf["contours_untrimmed"]
    return cached(f"{zone}_step1_contours", run)


def saved_grid(zone):
    return np.load(os.path.join(ROOT, f"{zone}_v9", "data", "slab2",
                                "unit_source_grid.npy"))


def cell_quads(grid):
    nr, _, nc = grid.shape
    quads, depth = [], []
    for r in range(nr - 1):
        for j in range(nc - 1):
            q = [(grid[r, 0, j], grid[r, 1, j]), (grid[r, 0, j + 1], grid[r, 1, j + 1]),
                 (grid[r + 1, 0, j + 1], grid[r + 1, 1, j + 1]), (grid[r + 1, 0, j], grid[r + 1, 1, j])]
            quads.append(q)
            depth.append(np.mean([grid[r, 2, j], grid[r, 2, j + 1],
                                  grid[r + 1, 2, j], grid[r + 1, 2, j + 1]]))
    return quads, np.array(depth)


def cell_flags(grid):
    """Per-cell bow-tie / collapsed flags with mesh_defects' own criteria."""
    from shapely.geometry import Polygon
    lat0 = np.radians(np.mean(grid[:, 1, :]))
    quads, _ = cell_quads(grid)
    bow, areas = [], []
    for q in quads:
        p = Polygon([(lo * np.cos(lat0), la) for lo, la in q])
        bow.append(not p.is_valid)
        areas.append(abs((p if p.is_valid else p.buffer(0)).area))
    areas = np.array(areas)
    coll = areas < 0.02 * np.median(areas)
    d = cd.mesh_defects(grid)
    assert d["bowties"] == int(np.sum(bow)) and d["degenerate"] == int(np.sum(coll))
    return np.array(bow), coll, d


def area_dip(grid):
    st = us.discretized_source_approximate_summary_statistics(grid)
    r = np.radians(st["dip"])
    return (float(np.sum(st["length"] * st["width"])),
            float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r))))))


def corner_devs(grid):
    """|corner angle - 90 deg| of every cell corner (local projection)."""
    lat0 = np.radians(np.mean(grid[:, 1, :]))
    quads, _ = cell_quads(grid)
    out = []
    for q in quads:
        P = np.array([(lo * np.cos(lat0), la) for lo, la in q])
        for k in range(4):
            u, v = P[k - 1] - P[k], P[(k + 1) % 4] - P[k]
            cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            out.append(abs(90 - np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return np.array(out)


def geo_aspect(ax, lat):
    ax.set_aspect(1.0 / np.cos(np.radians(lat)))


def draw_cells(ax, grid, vmax, flags=None, lw=0.5, alpha=1.0):
    quads, depth = cell_quads(grid)
    faces = DEPTH_CMAP(0.12 + 0.75 * np.clip(depth / vmax, 0, 1))
    faces[:, 3] = alpha
    edges = np.array([matplotlib.colors.to_rgba("#6b4a3a")] * len(quads))
    lws = np.full(len(quads), lw)
    if flags is not None:
        bow, coll = flags
        for i in np.nonzero(coll)[0]:
            faces[i] = matplotlib.colors.to_rgba(WARN)
            edges[i] = matplotlib.colors.to_rgba(WARN)
            lws[i] = 2.2
        for i in np.nonzero(bow)[0]:
            faces[i] = matplotlib.colors.to_rgba(BAD)
            edges[i] = matplotlib.colors.to_rgba(BAD)
            lws[i] = 2.2
    ax.add_collection(PolyCollection(quads, facecolors=faces, edgecolors=edges,
                                     linewidths=lws))


def style(ax, title=None):
    if title:
        ax.set_title(title, loc="left")
    ax.grid(color="#eef0f4", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)


# ------------------------------------------------ figure 1: kermadectonga2
def fig_build():
    zone = "kermadectonga2"
    contours = shapefile_contours(zone)
    saved = saved_grid(zone)

    def run():
        interps, depths = cd._build_interpolators(contours)
        num_l = len(interps)
        top = contours[0][1]
        top_len = sum(cd._GEOD.inv(top[k, 0], top[k, 1], top[k + 1, 0], top[k + 1, 1])[2]
                      for k in range(len(top) - 1)) / 1000.0
        dnl = int(np.ceil(top_len / 50.0 + 1))
        s = cd._optimise_s_matrix(interps, dnl, num_l, seed=1234, verbose=True)
        npv = s.shape[1]
        cost0 = float(np.sum(cd._quality_matrix(np.tile(np.linspace(0, 1, npv), (num_l, 1)), interps) ** 2))
        cost1 = float(np.sum(cd._quality_matrix(s, interps) ** 2))
        x, y = cd._get_xy(s, interps)
        x0, y0 = cd._get_xy(np.tile(np.linspace(0, 1, npv), (num_l, 1)), interps)
        return dict(top_len=top_len, dnl=dnl, s=s, x=x, y=y, x0=x0, y0=y0,
                    cost0=cost0, cost1=cost1, depths=sorted(depths))
    r = cached("kt2_build", run)
    x, y, s = r["x"], r["y"], r["s"]
    npv = s.shape[1]
    ds = np.asarray(r["depths"])
    lengths = []
    rebuilt = np.empty((saved.shape[0], 3, npv))
    for i in range(npv):
        t = np.column_stack([x[:, i], y[:, i], ds])
        lengths.append(sum(us.distance_down_depth(t[k], t[k + 1]) for k in range(len(ds) - 1)) / 1000)
        pts = cd.interpolate_3D_path(t, n=saved.shape[0])
        rebuilt[:, :, i] = pts
    mean_len = float(np.mean(lengths))
    print(f"[kermadectonga2] top contour {r['top_len']:.1f} km -> desired_num_lines "
          f"= ceil({r['top_len']:.1f}/50 + 1) = {r['dnl']}; multigrid levels "
          f"{[int(v) for v in np.floor(r['dnl'] * np.array([1/8, 1/4, 1/2, 1])) if v >= 3]}")
    print(f"[kermadectonga2] badness (sum of squares) evenly spaced start {r['cost0']:.4g}"
          f" -> optimised {r['cost1']:.4g}")
    print(f"[kermadectonga2] mean 3D down-dip transect {mean_len:.1f} km (min {min(lengths):.1f},"
          f" max {max(lengths):.1f}) -> rows = max(round({mean_len:.1f}/50 - 1), 0) + 2 = "
          f"{max(round(mean_len / 50 - 1), 0) + 2} nodes")
    print(f"[kermadectonga2] rebuilt grid vs saved step-2 grid: max |diff| "
          f"{np.max(np.abs(rebuilt - saved)):.2e} (deg / km)")
    print("[kermadectonga2] s-matrix, first 5 columns (rows = contours 0..40 km):")
    print(np.round(s[:, :5], 4))
    print(f"[kermadectonga2] s at column 40: {np.round(s[:, 40], 4)}")

    lat_lo, lat_hi, lon_lo, lon_hi = -27.2, -22.3, 182.4, 186.6
    fig = plt.figure(figsize=(12.6, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    draw_cells(ax0, saved, 40.0, lw=0.25)
    ax0.add_patch(Rectangle((lon_lo, lat_lo), lon_hi - lon_lo, lat_hi - lat_lo,
                            fill=False, ec=INK, lw=1.4))
    ax0.set_xlim(172, 190); ax0.set_ylim(-43.5, -13.5)
    geo_aspect(ax0, -28)
    style(ax0, "Whole zone")
    ax0.text(lon_lo - 0.4, (lat_lo + lat_hi) / 2, "zoom", fontsize=9, color=INK, ha="right", va="center")
    ax0.set_ylabel("latitude (deg)")

    cols = [DEPTH_CMAP(0.25 + 0.7 * d / 40.0) for d in ds]
    titles = ["1. Depth contours (step 1)\n0, 5, ..., 40 km below the trench",
              "2. Down-dip lines (dark)\ndots: where each crosses a contour",
              f"3. Cells (unit sources)\neach line cut into {saved.shape[0]-1} equal 3D pieces"]
    axes = [fig.add_subplot(gs[k]) for k in (1, 2, 3)]
    for k, ax in enumerate(axes):
        style(ax, titles[k])
        ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
        geo_aspect(ax, -25)
        ax.set_xlabel("longitude (deg E)")
        if k > 0:
            ax.set_yticklabels([])
    # panel 1: contours
    for (d, c), col in zip(contours, cols):
        axes[0].plot(c[:, 0], c[:, 1], color=col, lw=1.3)
    for d, c in contours:
        if d in (0.0, 40.0):
            k = np.argmin(np.abs(c[:, 1] - (lat_lo + 0.35)))
            axes[0].annotate("trench (0 km)" if d == 0 else "40 km", c[k],
                             xytext=(8 if d == 0 else -6, 0), textcoords="offset points", ha="left" if d == 0 else "right",
                             fontsize=9, color=INK, va="center")
    # panel 2: contours + down-dip lines
    for (d, c), col in zip(contours, cols):
        axes[1].plot(c[:, 0], c[:, 1], color=col, lw=0.9, alpha=0.8)
    for i in range(npv):
        axes[1].plot(x[:, i], y[:, i], color=INK, lw=1.0)
        axes[1].plot(x[:, i], y[:, i], "o", ms=2.4, color=INK)
    # panel 3: cells
    draw_cells(axes[2], saved, 40.0, lw=0.7)
    for i in range(npv):
        axes[2].plot(saved[:, 0, i], saved[:, 1, i], "o", ms=2.6, color=INK)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(DEPTH_CMAP(np.linspace(0.12, 0.87, 64))),
                               norm=plt.Normalize(0, 40))
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("depth below the trench (km)", color=MUTED)
    fig.savefig(os.path.join(OUT, "step2_kermadectonga2_build.png"), dpi=115, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------- figure 2: southamerica repair
def fig_repair():
    zone = "southamerica"
    final_contours, untrimmed = step1_contours_both(zone)

    def run_lm():
        return cd.discretized_source_from_contours_orthogonal(
            untrimmed, 50.0, desired_unit_source_width=50.0, min_downdip=2)

    def run_opt():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g = cd.discretized_source_from_contours_optimal(
                untrimmed, 50.0, desired_unit_source_width=50.0, min_downdip=2)
        return g, buf.getvalue()

    def run_aligned():
        return cd.align_contour_ends(untrimmed)

    lm = cached("sam_untrimmed_lm", run_lm)
    opt, log = cached("sam_untrimmed_optimal", run_opt)
    aligned = cached("sam_untrimmed_aligned", run_aligned)
    final = saved_grid(zone)
    print("[southamerica] optimal's own log on the untrimmed contours:")
    print(log.rstrip())
    fl = cell_flags(lm)
    fo = cell_flags(opt)
    for name, g, f in (("rptha (lm), untrimmed", lm, fl), ("optimal, untrimmed", opt, fo),
                       ("final v8 (clean ends)", final, cell_flags(final))):
        a, dip = area_dip(g)
        print(f"[southamerica] {name:24s} {g.shape[2]-1} x {g.shape[0]-1} cells, "
              f"{int(f[0].sum())} bow-tie, {int(f[1].sum())} collapsed, overlap "
              f"{100*f[2]['overlap_fraction']:.3f}%, area {a:,.0f} km2, mean dip {dip:.2f}")
    for lv, c in untrimmed[:1] + untrimmed[-1:]:
        print(f"[southamerica] untrimmed {lv:g} km contour south end at lat {min(c[0,1], c[-1,1]):.2f}")
    for lv, c in aligned[:1] + aligned[-1:]:
        print(f"[southamerica] end-aligned {lv:g} km contour south end at lat {min(c[0,1], c[-1,1]):.2f}")
    for lv, c in final_contours[:1] + final_contours[-1:]:
        print(f"[southamerica] clean-ended {lv:g} km contour south end at lat {min(c[0,1], c[-1,1]):.2f}")

    lat_lo, lat_hi, lon_lo, lon_hi = -45.5, -24.5, 280.5, 293.0
    NL = chr(10)
    zb = (285.95, 286.75, -39.55, -38.05)          # zoom box around the bow-tie
    fig = plt.figure(figsize=(13.2, 7.4))
    gsp = fig.add_gridspec(1, 4, wspace=0.22)
    axes = [fig.add_subplot(gsp[k]) for k in range(4)]
    ax_a, ax_z, ax_b, ax_c = axes
    panels = [(ax_a, lm, fl, "a. rptha's mesh (lm): defects"),
              (ax_b, opt, fo, "c. optimal: repaired"),
              (ax_c, final, None, "d. What v8 really uses")]
    for ax, g, f, t in panels:
        style(ax, t)
        draw_cells(ax, g, 55.0, flags=None if f is None else f[:2], lw=0.45)
        ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
        geo_aspect(ax, -35)
        ax.set_xlabel("longitude (deg E)")
    for ax in (ax_b, ax_c):
        ax.set_yticklabels([])
    ax_a.set_ylabel("latitude (deg)")
    for lv, c in untrimmed[:1] + untrimmed[-1:]:
        ax_a.plot(c[:, 0], c[:, 1], color=OFFICIAL, lw=1.0, ls="--")
    ax_a.add_patch(Rectangle((zb[0], zb[2]), zb[1] - zb[0], zb[3] - zb[2], fill=False, ec=INK, lw=1.3))
    bow, coll = fl[0], fl[1]
    quads, _ = cell_quads(lm)
    for i in np.nonzero(bow | coll)[0]:
        cx = np.mean([p[0] for p in quads[i]]); cy = np.mean([p[1] for p in quads[i]])
        lab = "bow-tie (see b)" if bow[i] else "collapsed"
        ax_a.annotate(lab, (cx, cy), xytext=(min(cx + 1.5, 290.2), cy + 0.5), fontsize=8.5,
                      color=BAD if bow[i] else WARN, fontweight="bold",
                      arrowprops=dict(arrowstyle="-", color=BAD if bow[i] else WARN, lw=1))
    # zoom panel: the columns of rptha's mesh around the bow-tie
    style(ax_z, NL.join(["b. zoom on the box in a", "the red bow-tie cell"]))
    draw_cells(ax_z, lm, 55.0, flags=fl[:2], lw=0.8)
    for j in range(lm.shape[2]):
        ax_z.plot(lm[:, 0, j], lm[:, 1, j], color=INK, lw=1.1)
        ax_z.plot(lm[:, 0, j], lm[:, 1, j], "o", ms=3, color=INK)
    ax_z.set_xlim(zb[0], zb[1]); ax_z.set_ylim(zb[2], zb[3])
    geo_aspect(ax_z, -38.6)
    ax_z.yaxis.tick_right()
    ax_z.set_xticks([286.0, 286.25, 286.5, 286.75])
    ax_z.set_xlabel("longitude (deg E)")
    NL = chr(10)
    ax_z.text(0.04, 0.985, NL.join(["dark lines = down-dip lines", "(columns). Down dip they",
                                   "bend north and run along", "the arc, so two of them",
                                   "cross: the red cell's sides", "cross each other."]),
              transform=ax_z.transAxes, fontsize=8.5, color=INK, va="top",
              bbox=dict(fc="white", ec=LINE, alpha=0.92))
    chosen = [ln for ln in log.splitlines() if "using" in ln]
    chosen = chosen[-1].split("using", 1)[1].strip().strip("'") if chosen else "?"
    ax_a.set_title(NL.join(["a. rptha's mesh (lm)",
                            f"{lm.shape[2]-1} x {lm.shape[0]-1} cells, {int(fl[0].sum())} bow-tie, "
                            f"{int(fl[1].sum())} collapsed"]), loc="left")
    ax_b.set_title(NL.join(["c. --discretizer optimal",
                            f"{opt.shape[2]-1} x {opt.shape[0]-1} cells, 0 defects,", f"repair: {chosen}"]),
                   loc="left")
    ax_c.set_title(NL.join(["d. what v8 really uses",
                            f"{final.shape[2]-1} x {final.shape[0]-1} cells, 0 defects,",
                            "crooked end cut in step 1"]), loc="left")
    ax_a.text(0.03, 0.985, NL.join(["dashed: 0 km line (trench)", "and 55 km contour"]),
              transform=ax_a.transAxes, fontsize=8.5, color=OFFICIAL, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step2_southamerica_repair.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return lm, opt, aligned, untrimmed


# ----------------------------------------------- figure 3: lm vs mid
def fig_lm_mid():
    """optimal (= lm on these contours) vs mid: kermadectonga2 zoom, plus
    corner statistics on four zones."""
    for z in ("kermadectonga2", "makran2", "puysegur2", "calabria2"):
        lm_z = saved_grid(z)   # step 2 used optimal, which kept rptha's lm mesh unchanged
        mid_z = cached(f"{z}_mid", lambda z=z: cd.discretized_source_from_contours_mid(
            shapefile_contours(z), 50.0, 50.0, min_downdip=2))
        for name, g in (("lm", lm_z), ("mid", mid_z)):
            dv = corner_devs(g)
            a, dip = area_dip(g)
            print(f"[{z}] {name:4s} {g.shape[2]-1} x {g.shape[0]-1} cells, area {a:,.0f} km2, "
                  f"mean dip {dip:.2f}, corner off square: median {np.median(dv):.1f}, "
                  f"90th pct {np.percentile(dv, 90):.1f}, max {dv.max():.1f} deg, "
                  f"defects {cd.mesh_defects(g)['total']}")
    zone = "kermadectonga2"
    contours = shapefile_contours(zone)
    lm = saved_grid(zone)
    mid = cached(f"{zone}_mid", None)
    lat_lo, lat_hi, lon_lo, lon_hi = -27.2, -22.3, 182.4, 186.6
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.2), sharey=True)
    t = []
    for name, g in (("a. optimal = lm (rptha's method)", lm), ("b. mid (rptha's old method)", mid)):
        dv = corner_devs(g)
        t.append(f"{name}\ncell corners off square: median {np.median(dv):.1f} deg")
    t.append("c. both down-dip lines, with\nthe contours (thin)")
    for k, ax in enumerate(axes):
        style(ax, t[k])
        ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
        geo_aspect(ax, -25)
        ax.set_xlabel("longitude (deg E)")
    axes[0].set_ylabel("latitude (deg)")
    draw_cells(axes[0], lm, 40.0, lw=0.7)
    draw_cells(axes[1], mid, 40.0, lw=0.7)
    for d, c in contours:
        axes[2].plot(c[:, 0], c[:, 1], color="#b9bfca", lw=0.7)
    for g, col, lab in ((lm, SCRATCH, "lm (and optimal)"), (mid, OFFICIAL, "mid")):
        for i in range(g.shape[2]):
            axes[2].plot(g[:, 0, i], g[:, 1, i], color=col, lw=1.4)
        axes[2].plot([], [], color=col, lw=1.8, label=lab)
    axes[2].legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step2_kermadectonga2_lm_vs_mid.png"), dpi=115, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------- figure 4: final meshes
def fig_final():
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.6))
    for ax, z in zip(axes.ravel(), EXAMPLES):
        g = saved_grid(z)
        draw_cells(ax, g, float(g[:, 2, :].max()), lw=0.3)
        lo, la = g[:, 0, :], g[:, 1, :]
        pad = 0.06 * max(lo.max() - lo.min(), la.max() - la.min())
        ax.set_xlim(lo.min() - pad, lo.max() + pad); ax.set_ylim(la.min() - pad, la.max() + pad)
        geo_aspect(ax, float(la.mean()))
        a, dip = area_dip(g)
        style(ax, f"{z}: {g.shape[2]-1} x {g.shape[0]-1} cells" + chr(10)
              + f"{a:,.0f} km2, mean dip {dip:.1f} deg")
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step2_final_meshes.png"), dpi=105, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"cache: {CACHE}")
    fig_build()
    fig_lm_mid()
    fig_final()
    fig_repair()
    for f in sorted(os.listdir(OUT)):
        if f.startswith("step2_"):
            print(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) / 1024:.0f} KB")
