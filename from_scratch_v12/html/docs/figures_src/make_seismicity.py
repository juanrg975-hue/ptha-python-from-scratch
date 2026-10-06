"""Figure of html/docs/seismicity.html (steps 4-5), from the real data.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_seismicity.py

Writes from_scratch_v12/html/docs/img/seismicity_kermadectonga2_selection.png:
the GCMT events of Mw >= 7.15 in PTHA18's window near kermadectonga2_v9's
mesh, the ones step 5 selects, and the ones it rejects, coloured by the FIRST
rule that rejects them, in the order step 5 applies the rules (depth, then
the 0.4 deg zone, then the focal mechanism).

The rule is not re-typed here: the functions and constants are imported from
the example's own generated step 5 script (kermadectonga2_v9/steps/
step5_subset_gcmt.py), so the figure applies exactly the code that ran.
It also prints the one event PTHA18 selected on southamerica that the
southamerica_v9 run does not, and why.
"""

import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from shapely.geometry import Point  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ROOT = os.path.dirname(PKG)
OUT = os.path.join(HERE, "..", "img")
os.makedirs(OUT, exist_ok=True)

SCRATCH = "#eb6834"
INK = "#1d2433"
MUTED = "#5b6475"
SEIS = "#8a5cd1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#b9c0cc",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 12, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white"})


def load_step5(folder):
    sys.dont_write_bytecode = True      # never write into the example folder
    path = os.path.join(ROOT, folder, "steps", "step5_subset_gcmt.py")
    spec = importlib.util.spec_from_file_location(f"step5_{folder}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify(folder, s5=None, near_deg=2.5, cat=None):
    """Every Mw >= threshold event in the window whose hypocentre or centroid
    lies within near_deg of the mesh, with the first failing rule."""
    s5 = s5 or load_step5(folder)
    if cat is None:
        cat = s5.parse_ndk_full(os.path.join(ROOT, folder, "data", "gcmt"))
    grid = np.load(os.path.join(ROOT, folder, "data", "slab2", "unit_source_grid.npy"))
    from pyptha_v12 import unit_sources as us
    stats = us.discretized_source_approximate_summary_statistics(grid)
    us_df = pd.DataFrame({"lon": stats["lon_c"], "lat": stats["lat_c"],
                          "strike": stats["strike"]})
    region = s5.buffered_cells(grid)
    wide = region.buffer(near_deg - s5.BUFFER_DEG)
    ref_lon = float(np.mean(grid[:, 0, :]))
    rows = []
    for _, r in cat[cat["Mw"] >= s5.THRESHOLD_MW].iterrows():
        h = Point(s5.near_lon(r["hypo_lon"], ref_lon), r["hypo_lat"])
        c = Point(s5.near_lon(r["cent_lon"], ref_lon), r["cent_lat"])
        if not (wide.contains(h) or wide.contains(c)):
            continue
        if r["depth_km"] > s5.MAX_DEPTH_KM:
            why = "deeper than 71 km"
        elif not (region.contains(h) or region.contains(c)):
            why = "outside the 0.4 deg zone"
        else:
            strike = s5.nearest_unit_source_strike(r["cent_lon"], r["cent_lat"], us_df)
            if s5.qualifies(r, strike):
                why = "selected"
            elif not any(s5.angle_diff(rk, 90.0) <= s5.RAKE_TOL_DEG
                         for rk in (r["rake1"], r["rake2"])):
                why = "no thrust-like plane"
            else:
                why = "strike not aligned"
        rows.append({**r.to_dict(), "why": why,
                     "hx": h.x, "hy": h.y, "cx": c.x, "cy": c.y})
    return grid, region, pd.DataFrame(rows), cat


STYLE = {
    "selected": dict(color=SEIS, marker="*", s=230, edgecolor=INK, lw=0.8, z=6),
    "deeper than 71 km": dict(color="#b9c0cc", marker="o", s=34, edgecolor="#8a93a3", lw=0.6, z=3),
    "outside the 0.4 deg zone": dict(color="#2a78d6", marker="s", s=42, edgecolor="white", lw=0.6, z=4),
    "no thrust-like plane": dict(color="#c43d3d", marker="v", s=60, edgecolor="white", lw=0.6, z=5),
    "strike not aligned": dict(color="#b7791f", marker="D", s=46, edgecolor="white", lw=0.6, z=5),
}


def fig_kermadec():
    grid, region, ev, _ = classify("kermadectonga2_v9", near_deg=4.0)
    counts = ev["why"].value_counts().to_dict()
    print("kermadectonga2 events near the mesh (Mw >= 7.15):", counts)

    nr, _, nc = grid.shape
    polys = [np.array([grid[r, :2, j], grid[r, :2, j + 1], grid[r + 1, :2, j + 1],
                       grid[r + 1, :2, j]]) for r in range(nr - 1) for j in range(nc - 1)]

    fig = plt.figure(figsize=(13, 9.6))
    ax = fig.add_axes([0.05, 0.04, 0.48, 0.84])
    at = fig.add_axes([0.61, 0.56, 0.37, 0.34])
    ax.add_collection(PolyCollection(polys, facecolor="#fde6d8", edgecolor=SCRATCH, lw=0.4, zorder=1))
    geoms = getattr(region, "geoms", [region])
    for i, g in enumerate(geoms):
        x, y = g.exterior.xy
        ax.plot(x, y, color=SEIS, lw=1.3, ls="--", zorder=2,
                label="0.4 deg zone (cells buffered, joined)" if i == 0 else None)
    for why, st in STYLE.items():
        d = ev[ev["why"] == why]
        if not len(d):
            continue
        ax.scatter(d["cx"], d["cy"], c=st["color"], marker=st["marker"], s=st["s"],
                   edgecolors=st["edgecolor"], linewidths=st["lw"], zorder=st["z"],
                   label=f"{why}: {len(d)}")
    sel = ev[ev["why"] == "selected"].sort_values("Mw")
    for _, r in sel.iterrows():
        ax.plot([r["hx"], r["cx"]], [r["hy"], r["cy"]], color=INK, lw=0.8, zorder=5)
        ax.scatter([r["hx"]], [r["hy"]], s=14, color=INK, zorder=6)
    # isolated events: label to the right; the cluster near -29 deg: labels
    # stacked on the left, each joined to its star
    slot = -25.8
    for _, r in sel.sort_values("cy", ascending=False).iterrows():
        txt = f"Mw {r['Mw']:.2f}, {str(r['date'])[:10]}"
        if r["cy"] > -26:
            ax.annotate(txt, (r["cx"], r["cy"]), (r["cx"] + 0.9, r["cy"]), fontsize=9,
                        ha="left", va="center", color=INK, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        else:
            ax.annotate(txt, (r["cx"], r["cy"]), (179.6, slot), fontsize=9, ha="right",
                        va="center", color=INK, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
            slot -= 0.95
    ax.scatter([], [], s=14, color=INK, label="hypocentre (joined to its centroid)")
    ax.set_xlim(169.0, 193.0)
    ax.set_ylim(-46.5, -10.3)
    ax.set_aspect(1.0 / np.cos(np.radians(-29)))
    ax.set_xlabel("longitude (deg E, 0-360)")
    ax.set_ylabel("latitude (deg N)")
    ax.grid(color="#eef0f4", lw=0.6)
    ax.set_title("Mw >= 7.15, 1976-01-01 to 2017-03-01, within 4 deg of the mesh\n"
                 "symbols at the centroid, coloured by the first rule that rejects them",
                 loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # timeline of the selected events
    d = pd.to_datetime(sel["date"])
    at.axvspan(pd.Timestamp("1976-01-01"), pd.Timestamp("2017-03-01"), color="#f1eafb", zorder=0)
    at.vlines(d, 7.15, sel["Mw"], color=SEIS, lw=1.5)
    at.scatter(d, sel["Mw"], marker="*", s=150, color=SEIS, edgecolor=INK, lw=0.6, zorder=3)
    at.axhline(7.15, color=INK, lw=1, ls="--")
    at.text(pd.Timestamp("1977-06-01"), 7.17, "threshold Mw 7.15", fontsize=9, color=INK)
    at.set_xlim(pd.Timestamp("1974-01-01"), pd.Timestamp("2019-06-01"))
    at.set_ylim(7.1, 8.12)
    at.set_ylabel("Mw")
    at.set_title("The 9 selected events in time: LEVEL 3's data\n"
                 "count 9 in 41.1636 years (shaded window)", loc="left", fontsize=11)
    at.grid(axis="y", color="#eef0f4", lw=0.6)

    fig.text(0.61, 0.43,
             "How to read the map (counts in its legend)\n"
             "1. Grey circles: too deep (> 71 km). Most are the\n"
             "   deep earthquakes inside the sinking plate.\n"
             "2. Blue squares: shallow enough, but neither the\n"
             "   hypocentre nor the centroid is inside the 0.4 deg zone\n"
             f"   ({counts.get('outside the 0.4 deg zone', 0)} here).\n"
             "3. Red triangles: inside, but no nodal plane has a\n"
             "   rake within 50 deg of 90 (not a thrust).\n"
             "4. Gold diamonds: a thrust-like plane exists, but its\n"
             "   strike is > 50 deg off the nearest cell's strike.\n"
             "5. Purple stars: pass every test and are counted.",
             fontsize=10, va="top", color=INK, linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.6", fc="#f6f7f9", ec="#dde1e8"))
    fig.suptitle("kermadectonga2_v9: which GCMT earthquakes step 5 keeps, and why the others are left out",
                 x=0.02, y=0.985, ha="left", fontsize=13, fontweight="bold")
    p = os.path.join(OUT, "seismicity_kermadectonga2_selection.png")
    fig.savefig(p, dpi=112, bbox_inches="tight")
    plt.close(fig)
    return p


def southamerica_missing():
    """The one official southamerica event the from-scratch mesh does not select."""
    off = pd.read_csv(os.path.join(ROOT, "official_ptha_data", "trees",
                                   "official_gcmt_observations.csv"))
    off = off[off["source"] == "southamerica"]["Mw"].to_numpy()
    grid, _, ev, cat = classify("southamerica_v9", near_deg=6.0)
    mine = pd.read_csv(os.path.join(ROOT, "southamerica_v9", "data", "gcmt",
                                    "southamerica_gcmt_subset.csv"))["Mw"].to_numpy()
    for m in off:
        if np.min(np.abs(mine - m)) > 1e-9:
            row = cat.iloc[int(np.argmin(np.abs(cat["Mw"] - m)))]
            e = ev.iloc[int(np.argmin(np.abs(ev["Mw"] - m)))] if len(ev) else None
            print("southamerica: official-only event", row["date"].date(), f"Mw {m:.4f}",
                  f"hypo ({row['hypo_lat']}, {row['hypo_lon']}) centroid ({row['cent_lat']}, "
                  f"{row['cent_lon']}) depth {row['depth_km']} km",
                  "; this run:", None if e is None or abs(e["Mw"] - m) > 1e-9 else e["why"],
                  f"; this run's mesh latitudes {np.nanmin(grid[:, 1, :]):.2f} to "
                  f"{np.nanmax(grid[:, 1, :]):.2f}")


if __name__ == "__main__":
    sys.path.insert(0, PKG)
    path = fig_kermadec()
    print("wrote", os.path.relpath(path, ROOT), f"{os.path.getsize(path) / 1024:.0f} KB")
    southamerica_missing()
