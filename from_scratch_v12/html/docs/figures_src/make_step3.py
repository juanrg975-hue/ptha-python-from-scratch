"""Figures of html/docs/step3.html (plate convergence), from the real data.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_step3.py

Writes into from_scratch_v12/html/docs/img/:
  step3_kermadectonga2_matching.png   Bird segments near kermadectonga2, the
                                      top-edge points and their matches
  step3_kermadectonga2_per_column.png the per-column values of this run
  step3_official_vs_this_run.png      per-column convergent slip, this run's
                                      mesh vs PTHA18's mesh, 5 zones

Everything is recomputed with lib/bird_convergence.py (the code step 3 and
step 8 run): on the examples' meshes with Bird (2003)'s public catalogue, as
their step 3 did (--ptha false), and on PTHA18's unit-source tables
(inputs/geometry/unit_source_statistics_<zone>.nc) with PTHA18's own Bird
table, as step 8 does.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ROOT = os.path.dirname(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))

from pyptha_v12 import unit_sources as us  # noqa: E402
import bird_convergence as bc  # noqa: E402

OUT = os.path.join(HERE, "..", "img")
os.makedirs(OUT, exist_ok=True)

SCRATCH = "#eb6834"
OFFICIAL = "#2a78d6"
INK = "#1d2433"
MUTED = "#5b6475"
LINE = "#dde1e8"
CONV = "#d9822b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#b9c0cc",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 12, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white"})


def scratch_stats(folder):
    grid = np.load(os.path.join(ROOT, folder, "data", "slab2", "unit_source_grid.npy"))
    return grid, us.discretized_source_approximate_summary_statistics(grid)


def official_stats(zone):
    with netCDF4.Dataset(os.path.join(ROOT, "inputs", "geometry",
                                      f"unit_source_statistics_{zone}.nc")) as ds:
        return {k: np.asarray(ds[k][:], float) for k in (
            "lon_c", "lat_c", "strike", "dip", "width", "length", "rake",
            "downdip_number", "alongstrike_number")}


def top_edge_points(st):
    """Top-edge point of every column (as bird_convergence.column_convergence)."""
    top = np.where(np.asarray(st["downdip_number"]).astype(int) == 1)[0]
    order = np.argsort(np.asarray(st["alongstrike_number"])[top])
    top = top[order]
    lon, lat = bc.top_edge_point(st["lon_c"][top], st["lat_c"][top], st["strike"][top],
                                 st["dip"][top], st["width"][top])
    return np.asarray(lon) % 360.0, np.asarray(lat)


def bird_table():
    """Bird (2003)'s public catalogue, convergent types: what step 3 reads
    with --ptha false (every example)."""
    return bc._add_midpoints(bc.load_bird_raw_segments())


# ---------------------------------------------------------------------------
# 1. kermadectonga2: Bird segments, top-edge points, matches
# ---------------------------------------------------------------------------
def fig_matching():
    grid, st = scratch_stats("kermadectonga2_v9")
    col = bc.column_convergence(st, log=lambda *a: None, table="bird")
    lon_range, lat_range = bc.bounding_box(st["lon_c"], st["lat_c"])
    df = bird_table()
    used = df[bc._in_box(df, lon_range, lat_range)].copy()
    e_lon, e_lat = top_edge_points(st)
    ub_lon = used["mid_lon"].to_numpy() % 360.0
    ub_lat = used["mid_lat"].to_numpy()
    match = [int(np.argmin(bc.dist_haversine(a, b, used["mid_lon"].to_numpy(), ub_lat)))
             for a, b in zip(e_lon, e_lat)]

    nr, _, nc = grid.shape
    polys = []
    for r in range(nr - 1):
        for j in range(nc - 1):
            q = np.array([grid[r, :2, j], grid[r, :2, j + 1],
                          grid[r + 1, :2, j + 1], grid[r + 1, :2, j]])
            q[:, 0] %= 360.0
            polys.append(q)

    def seg_lines(d):
        a = d[["Long1", "Lat1"]].to_numpy().copy()
        b = d[["Long2", "Lat2"]].to_numpy().copy()
        a[:, 0] %= 360.0
        b[:, 0] %= 360.0
        return np.stack([a, b], axis=1)

    conv = np.maximum(0.0, -used["Div_vel"].to_numpy())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "conv", plt.get_cmap("YlOrBr")(np.linspace(0.35, 1.0, 256)))
    norm = plt.Normalize(0, 250)
    zoom = (184.9, 190.4, -18.5, -13.6)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, (x0, x1, y0, y1), title in (
            (axes[0], (171.5, 190.5, -44.0, -13.0), "Whole zone"),
            (axes[1], zoom, "Zoom: the northern end (Tonga), columns 1-8")):
        ax.add_collection(PolyCollection(polys, facecolor="#fde6d8", edgecolor=SCRATCH,
                                         linewidth=0.5, zorder=1))
        lc = LineCollection(seg_lines(used), colors=cmap(norm(conv)),
                            linewidths=3.4 if ax is axes[1] else 2.4, zorder=3)
        ax.add_collection(lc)
        for k, i in enumerate(match):
            ax.plot([e_lon[k], ub_lon[i]], [e_lat[k], ub_lat[i]], color=INK, lw=0.9, zorder=4)
        ax.scatter(e_lon, e_lat, s=16 if ax is axes[0] else 36, color=INK, zorder=5,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(ub_lon[match], ub_lat[match], s=18 if ax is axes[0] else 46,
                   marker="D", facecolor="white", edgecolor=INK, linewidth=1.0, zorder=5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (y0 + y1))))
        ax.set_title(title, loc="left")
        ax.set_xlabel("longitude (deg E, 0-360)")
        ax.grid(color="#eef0f4", lw=0.6)
    axes[0].set_ylabel("latitude (deg N)")
    axes[0].add_patch(plt.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                                    fill=False, ec=INK, lw=1.2, ls="--", zorder=6))
    axes[0].annotate("zoom (right)", (zoom[0], -17.0), (177.5, -17.0), fontsize=10, color=INK,
                     va="center", arrowprops=dict(arrowstyle="->", color=INK))
    axes[0].annotate("column 1 (north)", (e_lon[0], e_lat[0]), (177.0, -14.0), fontsize=10,
                     color=INK, va="center", arrowprops=dict(arrowstyle="->", color=MUTED))
    axes[0].annotate(f"column 73 (south):\nnearest convergent\nstep {col['distance_km'][-1]:.0f} km away,\n"
                     f"{col['convergent_slip_mm_per_yr'][-1]:.1f} mm/yr",
                     (e_lon[-1], e_lat[-1]), (171.9, -33.0), fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED))
    axes[0].annotate("Bird segments from the\nNew Hebrides zone are in\nthe search box too",
                     (171.8, -23.1), (171.9, -27.0), fontsize=10, color=MUTED,
                     arrowprops=dict(arrowstyle="->", color=MUTED))

    # labels in the zoom: column number, div and convergent slip
    div = col["div_mm_per_yr"]
    slip = col["convergent_slip_mm_per_yr"]
    ypos = {1: -13.85, 2: -14.2, 3: -14.55, 4: -15.3, 5: -16.0, 6: -16.7, 7: -17.35, 8: -18.0}
    for c, y in ypos.items():
        k = c - 1
        axes[1].annotate(f"col {c}: div {div[k]:.0f}, slip {slip[k]:.0f}",
                         (e_lon[k], e_lat[k]), (188.15, y), fontsize=9.5,
                         color=INK, ha="left", va="center",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LINE, lw=0.6),
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), zorder=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=30)
    cb.set_label("Bird segment colour: max(0, -Div_vel), mm/yr")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="", color=INK, markeredgecolor="white", label="top-edge point of a column"),
        Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK, label="matched segment midpoint"),
        Line2D([], [], color=INK, lw=0.9, label="match (nearest, haversine)"),
        Line2D([], [], color=cmap(0.6), lw=3, label="Bird step, convergent type (public catalogue)"),
        plt.Rectangle((0, 0), 1, 1, fc="#fde6d8", ec=SCRATCH, label="this run's mesh (219 cells)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle("kermadectonga2_v9: each column's top edge is matched to the nearest Bird (2003) segment",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    p = os.path.join(OUT, "step3_kermadectonga2_matching.png")
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 2. kermadectonga2: the per-column values
# ---------------------------------------------------------------------------
def fig_per_column():
    with open(os.path.join(ROOT, "kermadectonga2_v9", "data", "convergence_per_column.json")) as f:
        j = json.load(f)
    n = np.asarray(j["alongstrike_number"])
    div = np.asarray(j["div_mm_per_yr"])
    slip = np.asarray(j["convergent_slip_mm_per_yr"])
    mean = float(j["area_weighted_mean_mm_per_yr"])

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(n, slip, width=0.8, color="#f5c9a0", edgecolor="none",
           label="convergent slip sqrt(div² + rl²): the event weights")
    ax.bar(n, div, width=0.8, color=CONV, edgecolor="none",
           label="div: convergent part only, the LEVEL 4 target shape")
    ax.axhline(mean, color=INK, lw=1.4, ls="--",
               label=f"area-weighted mean of the slip: {mean:.2f} mm/yr (data/convergence.txt)")
    ax.annotate(f"col 1: div {div[0]:.0f}, sideways speed capped,\nslip {slip[0]:.1f}",
                (1, slip[0]), (1.5, 292), fontsize=10, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate(f"col 4: highest slip, {slip[3]:.1f}", (4, slip[3]), (14, 262), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate(f"cols 71-73: slow end near New Zealand,\n{slip[70]:.1f} to {slip[72]:.1f} mm/yr",
                (72, slip[71] + 2), (52, 150), fontsize=10, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.annotate(f"col 43: div {div[42]:.1f} but slip {slip[42]:.1f}\n(large sideways part)",
                (43, slip[42]), (30, 185), fontsize=10, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(0, 74)
    ax.set_ylim(0, 325)
    ax.set_xlabel("along-strike column (1 = north, Tonga; 73 = south, near New Zealand)")
    ax.set_ylabel("mm/yr (horizontal)")
    ax.legend(loc="upper right", fontsize=9.5, frameon=False, bbox_to_anchor=(1.0, 1.02))
    ax.grid(axis="y", color="#eef0f4", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("kermadectonga2_v9: Bird convergence per column (data/convergence_per_column.json)",
                 loc="left")
    p = os.path.join(OUT, "step3_kermadectonga2_per_column.png")
    fig.savefig(p, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 3. this run's mesh vs PTHA18's mesh, per column, five zones
# ---------------------------------------------------------------------------
ZONES = [("kermadectonga2", "kermadectonga2_v9", "lat"),
         ("southamerica", "southamerica_v9", "lat"),
         ("kurilsjapan", "kurilsjapan_v9", "lat"),
         ("makran2", "makran2_v9", "lon"),
         ("puysegur2", "puysegur2_v9", "lat")]


def fig_official():
    fig, axes = plt.subplots(len(ZONES), 1, figsize=(12, 15.5))
    rows = []
    for ax, (zone, folder, xk) in zip(axes, ZONES):
        _, st_s = scratch_stats(folder)
        st_o = official_stats(zone)
        res = {}
        # each mesh with the Bird table its own run used: this run (--ptha
        # false) the public catalogue, PTHA18's mesh (step 8) PTHA18's table
        for tag, st, colr, table in (("this run", st_s, SCRATCH, "bird"),
                                     ("PTHA18 mesh", st_o, OFFICIAL, "bird-griffin")):
            c = bc.column_convergence(st, log=lambda *a: None, table=table)
            lon, lat = top_edge_points(st)
            x = lat if xk == "lat" else lon
            ax.plot(x, c["convergent_slip_mm_per_yr"], color=colr, lw=2, marker="o", ms=3.5,
                    label=f"{tag}: {len(x)} columns, area-weighted mean "
                          f"{c['area_weighted_mean_mm_per_yr']:.2f} mm/yr")
            ax.axhline(c["area_weighted_mean_mm_per_yr"], color=colr, lw=1, ls="--")
            res[tag] = (c["area_weighted_mean_mm_per_yr"], len(x), float(x.min()), float(x.max()))
        if zone == "puysegur2":
            ax.axhline(35.0, color=MUTED, lw=1.6, ls=":")
            ax.set_ylim(0, 42)
            ax.text(-47.9, 36.3,
                    "PTHA18 did not use Bird here: constant 35 mm/yr", color=MUTED, fontsize=10)
        ax.set_title(zone, loc="left")
        ax.set_xlabel("latitude of the column's top edge (deg N)" if xk == "lat"
                      else "longitude of the column's top edge (deg E)")
        ax.set_ylabel("mm/yr")
        ax.grid(color="#eef0f4", lw=0.6)
        ax.legend(loc="best", fontsize=9.5, framealpha=0.9)
        if zone != "puysegur2":
            ax.set_ylim(bottom=0)
        rows.append((zone, res))
    axes[1].annotate("PTHA18's mesh continues south to -57.6 deg;\nthis run's SLAB mesh stops near -43 deg",
                     (-45, 30), (-40, 10), fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.suptitle("Bird convergent slip per column: this run's mesh (orange) vs PTHA18's mesh (blue), "
                 "same code (lib/bird_convergence.py)", x=0.02, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = os.path.join(OUT, "step3_official_vs_this_run.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    for zone, res in rows:
        print(zone, {k: tuple(round(v, 4) for v in val) for k, val in res.items()})
    return p


if __name__ == "__main__":
    for fn in (fig_matching, fig_per_column, fig_official):
        path = fn()
        print("wrote", os.path.relpath(path, ROOT), f"{os.path.getsize(path) / 1024:.0f} KB")
