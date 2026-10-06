"""Figures of html/docs/v10_q.html (what is new in v10 and v10_q).

Run from V9/ (it reads the example folders next to from_scratch_v12):
    ../../../.venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_v10q.py

Writes html/docs/img/v10q_*.png:
  v10q_ramp_profile.png        hellenic's trench depth from the Kefalonia end, the ramp rule's threshold
  v10q_ramp_mesh.png           hellenic_west2_v9 against hellenic_west2_v10_q (ramp rule)
  v10q_calabria_slip.png       slip range per magnitude, calabria2 v9 against v10_q
  v10q_calabria_rows.png       share of the Mw 7.2-7.6 rate per row, calabria2 v9 / v10 / v10_q
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PKG = "from_scratch_v12"
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))
import slab_contours as sc  # noqa: E402
from pyptha_v12 import events as ev, unit_sources as us  # noqa: E402
from pyptha_v12.scaling import Mw_2_rupture_size  # noqa: E402

IMG = os.path.join(PKG, "html", "docs", "img")
COL = {"v9": "#2a78d6", "v10": "#eb6834", "v10_q": "#1baf7a"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a93a3", "axes.grid": True, "grid.color": "#e1e0d9"})


def ramp_profile():
    with netCDF4.Dataset(os.path.join(PKG, "data", "slab2", "hel_slab2_dep_02.24.18.grd")) as ds:
        n = list(ds.variables)
        x = np.asarray(ds[n[0]][:], float)
        y = np.asarray(ds[n[1]][:], float)
        z = np.ma.filled(ds[n[2]][:].astype(float), np.nan)
    if y[0] > y[-1]:
        y, z = y[::-1], z[::-1]
    d = -z
    d = np.where(sc.largest_component(np.isfinite(d)), d, np.nan)
    tr = sc.find_trench(x, y, d)
    s = sc.line_km(tr["trench_raw"])
    med = sc.running_median(tr["trench_depth_raw"], s, 50.0)
    trim = sc._ramp_trim(s, med, 3.0, (100.0, 200.0), 100.0)[0]
    normal = max(float(np.median(med[(s >= 100) & (s <= 200)])), float(np.median(med)))
    k = s <= 400
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(s[k], med[k], color="#0b0b0b", lw=2, label="trench depth (50 km running median)")
    ax.axhline(normal, color="#52514e", ls="--", lw=1.2, label=f"normal depth there: {normal:.1f} km")
    ax.axhline(normal + 3, color="#b8312f", ls=":", lw=1.5, label=f"threshold = normal + 3 km ({normal + 3:.1f} km)")
    ax.axvspan(0, trim, color="#f9cdb8", alpha=0.7, label=f"trimmed: {trim:.0f} km")
    ax.invert_yaxis()
    ax.set_xlabel("km along the trench from its Kefalonia (NW) end")
    ax.set_ylabel("km below sea level")
    ax.set_title("hellenic: the trench end that climbs like a ramp", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "v10q_ramp_profile.png"), dpi=110)
    plt.close(fig)
    return trim


def ramp_mesh():
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, folder, title in zip(axs, ["hellenic_west2_v9", "hellenic_west2_v10_q"],
                                 ["v9: trench includes the Kefalonia ramp", "v10_q: ramp rule (71 km trimmed)"]):
        g = np.load(os.path.join(folder, "data", "slab2", "unit_source_grid.npy"))
        for j in range(g.shape[0]):
            ax.plot(g[j, 0], g[j, 1], color="#52514e", lw=0.7)
        for i in range(g.shape[2]):
            ax.plot(g[:, 0, i], g[:, 1, i], color="#52514e", lw=0.7)
        ax.plot(g[0, 0], g[0, 1], color="#d6336c", lw=2.6, label="trench (top edge)")
        ax.set_title(f"{title}  ({g.shape[2] - 1} x {g.shape[0] - 1} cells)", loc="left", fontsize=11)
        ax.set_aspect(1 / np.cos(np.radians(36.5)))
        ax.set_xlim(18.3, 26)
        ax.set_ylim(32.3, 39)
        ax.grid(False)
    axs[0].annotate("crease", xy=(22.75, 36.5), xytext=(23.6, 37.6), fontsize=11, color="#b8312f",
                    arrowprops=dict(arrowstyle="->", color="#b8312f"))
    axs[0].legend(fontsize=9, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "v10q_ramp_mesh.png"), dpi=100)
    plt.close(fig)


def calabria():
    g = np.load("calabria2_v9/data/slab2/unit_source_grid.npy")
    s = us.discretized_source_approximate_summary_statistics(g)
    area = s["length"] * s["width"]
    dd = s["downdip_number"].astype(int)
    runs = {"v9": ("calabria2_v9", "rptha"), "v10": ("calabria2_v10", "local"), "v10_q": ("calabria2_v10_q", "local")}
    sc_ = {k: pd.read_csv(f"{f}/outputs/scenario_rates_calabria2.csv") for k, (f, _) in runs.items()}
    mws = np.round(np.arange(7.2, 9.01, 0.1), 1)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for off, k in ((-0.02, "v9"), (0.02, "v10_q")):
        t = sc_[k]
        for mw in mws:
            m = t[np.isclose(t.Mw, mw)]
            ax.plot([mw + off] * 2, [m.slip_m.min(), m.slip_m.max()], color=COL[k], lw=3, solid_capstyle="round")
            ax.plot(mw + off, np.median(m.slip_m), "o", color=COL[k], ms=4)
        ax.plot([], [], color=COL[k], lw=3, label=k)
    st = [7.08e19 * 10 ** (1.5 * (mw - 7.2)) / (3e10 * Mw_2_rupture_size(mw)["area"] * 1e6) for mw in mws]
    ax.plot(mws, st, ls="--", color="#52514e", lw=1.3, label="slip of a rupture with Strasser's area")
    ax.set_yscale("log")
    ax.set_xlabel("magnitude Mw")
    ax.set_ylabel("slip (m)")
    ax.set_title("calabria2: slip range of the ruptures of each magnitude (bar = min to max, dot = median)", loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "v10q_calabria_slip.png"), dpi=110)
    plt.close(fig)

    shares = {}
    for k, (f, rule) in runs.items():
        t = sc_[k]
        tab = ev.get_all_earthquake_events(s, Mmin=7.2, Mmax=9.8, dMw=0.1, rupture_size=rule)
        assert np.allclose(tab["area"], t.area_km2)
        cr = np.zeros(area.size)
        for e in np.where(t.Mw.values < 7.65)[0]:
            i = tab["event_indices"][e]
            cr[i] += t.rate_mean.values[e] * area[i] / area[i].sum()
        shares[k] = [cr[dd == r].sum() / cr.sum() for r in range(1, 6)]
    row_area = [area[dd == r].sum() / area.sum() for r in range(1, 6)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.26
    for j, k in enumerate(runs):
        ax.bar(np.arange(5) + (j - 1) * w, shares[k], width=w - 0.02, color=COL[k], label=k)
    for r in range(5):
        ax.plot([r - 1.6 * w, r + 1.6 * w], [row_area[r]] * 2, color="#0b0b0b", lw=2)
    ax.plot([], [], color="#0b0b0b", lw=2, label="share of the fault area")
    ax.set_xticks(range(5), ["row 1 (trench)", "row 2", "row 3", "row 4", "row 5 (deepest)"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("calabria2: share of the Mw 7.2-7.6 rate in each row", loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "v10q_calabria_rows.png"), dpi=110)
    plt.close(fig)
    return shares, row_area


if __name__ == "__main__":
    print("ramp trimmed", round(ramp_profile()), "km")
    ramp_mesh()
    print(calabria())
