"""Figures for engine.html (steps 6-7: the logic tree and the rates).

Reads only existing files of the kermadectonga2_v9 example (its outputs,
outputs_official, input JSON and step 2's saved mesh) and writes PNGs into
html/docs/img/engine_*.png. Nothing in the example folder is modified.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_engine.py
(with .venv/Scripts/python.exe from there)
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, ".."))
PKG = os.path.abspath(os.path.join(DOCS, "..", ".."))          # from_scratch_v12/
ROOT = os.path.abspath(os.path.join(PKG, ".."))                  # ptha18_logic_tree_test/
IMG = os.path.join(DOCS, "img")
os.makedirs(IMG, exist_ok=True)
sys.path.insert(0, PKG)

from pyptha_v12 import events, unit_sources as us  # noqa: E402

ZONE = "kermadectonga2"
EX = os.path.join(ROOT, "kermadectonga2_v9")
OUT = os.path.join(EX, "outputs")
OFF = os.path.join(EX, "outputs_official")

SCR = "#eb6834"    # this run
OFFC = "#2a78d6"   # PTHA18 inputs (step 8)
ENG = "#2f6fdf"
GREY = "#8a93a3"
INK = "#1d2433"
MUTED = "#5b6475"
GOOD = "#2e9e6b"
SEIS = "#8a5cd1"
CONV = "#d9822b"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#b8bfca", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e3e6ec", "grid.linewidth": 0.8,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#dde1e8",
})
DPI = 120


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1e3:.0f} KB)")


cfg = json.load(open(os.path.join(EX, "inputs", f"input_{ZONE}_scratch.json")))
rc = cfg["rates"]
grid = np.load(os.path.join(EX, "data", "slab2", "unit_source_grid.npy"))
stats = us.discretized_source_approximate_summary_statistics(grid)
sr = pd.read_csv(os.path.join(OUT, f"scenario_rates_{ZONE}.csv"))
br = pd.read_csv(os.path.join(OUT, f"logic_tree_branches_{ZONE}.csv"))
cp = pd.read_csv(os.path.join(OUT, f"conditional_prob_{ZONE}.csv"))

# ---------------------------------------------------------------------------
# 1. The exceedance-rate curve: mean, percentiles, step 8, GCMT
# ---------------------------------------------------------------------------
curve = pd.read_csv(os.path.join(OUT, "rate_curves.csv"))
pct = pd.read_csv(os.path.join(OUT, "exceedance_rate_percentiles.csv"))
curve_off = pd.read_csv(os.path.join(OFF, "rate_curves.csv"))
pct_off = pd.read_csv(os.path.join(OFF, "exceedance_rate_percentiles.csv"))
obs = np.array(rc["observed_seismicity"]["observed_Mw"])
dur = rc["observed_seismicity"]["duration_years"]

fig, ax = plt.subplots(figsize=(10, 5.6))
th = pct.threshold_Mw.values
ax.fill_between(th, pct["p0.025"].clip(lower=1e-6), pct["p0.975"], color=SCR, alpha=0.13,
                label="this run: 2.5 to 97.5 percentile (LEVEL 5)")
ax.fill_between(th, pct["p0.16"].clip(lower=1e-6), pct["p0.84"], color=SCR, alpha=0.25,
                label="this run: 16 to 84 percentile (LEVEL 5)")
ax.semilogy(curve.Mw, curve.exceedance_rate_unsegmented, color=SCR, lw=2.4,
            label="this run: mean (rate_curves.csv)")
ax.semilogy(th, pct["p0.5"], "o--", color=SCR, lw=1.2, ms=5, label="this run: median")
ax.semilogy(curve_off.Mw, curve_off.exceedance_rate_unsegmented, color=OFFC, lw=1.8, ls="-",
            label="step 8 (PTHA18's inputs, same engine): mean")
mo = np.arange(7.15, obs.max() + 1e-9, 0.01)
ax.semilogy(mo, [(obs >= m).sum() / dur for m in mo], color=GOOD, lw=2.0, drawstyle="steps-post",
            label=f"GCMT: {obs.size} observed earthquakes / {dur:.1f} yr")
ax.set_xlim(7.15, 9.8)
ax.set_ylim(1e-5, 1.2)
ax.set_xlabel("Magnitude Mw")
ax.set_ylabel("Rate of earthquakes with magnitude > Mw (per year)")
ax.set_title("kermadectonga2: how often an earthquake larger than Mw happens")
r72 = float(curve.exceedance_rate_unsegmented[np.isclose(curve.Mw, 7.2)].iloc[0])
ax.annotate(f"Mw > 7.2: {r72:.3f} per year\n(about one every {1 / r72:.1f} years)",
            xy=(7.2, r72), xytext=(7.55, 0.45), fontsize=10.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED))
r88 = float(curve.exceedance_rate_unsegmented[np.isclose(curve.Mw, 8.8)].iloc[0])
ax.annotate(f"Mw > 8.8: {r88:.4f} per year\n(about one every {1 / r88:.0f} years)",
            xy=(8.8, r88), xytext=(8.95, 0.03), fontsize=10.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.annotate("zero above 9.6:\nthe largest Mw_max\nof the tree", xy=(9.6, 1.2e-5), xytext=(8.8, 2.2e-5),
            fontsize=10, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.legend(fontsize=9.3, loc="lower left")
save(fig, "engine_kermadectonga2_rate_curve.png")

# ---------------------------------------------------------------------------
# 2. Scenarios per magnitude and block sizes
# ---------------------------------------------------------------------------
mws = np.round(np.arange(7.2, 9.8 + 1e-9, 0.1), 1)
counts, dims = [], []
for m in mws:
    e = events.get_all_earthquake_events_of_magnitude_Mw(m, stats)
    counts.append(len(e["topleft_indices"]))
    dims.append((e["event_dim"]["length"], e["event_dim"]["width"]))
counts = np.array(counts)
assert counts.sum() == len(sr), (counts.sum(), len(sr))
nzero = sr.groupby(sr.Mw.round(1)).rate_mean.apply(lambda r: int((r == 0).sum())).reindex(mws, fill_value=0).values

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(mws, counts - nzero, width=0.075, color=ENG, label="scenarios with a rate > 0")
ax.bar(mws, nzero, width=0.075, bottom=counts - nzero, color="#c9ced8",
       label="scenarios with rate 0 (Mw above every Mw_max)")
prev = None
for m, c, d in zip(mws, counts, dims):
    if d != prev:
        ax.text(m, c + 6, f"{d[0]}x{d[1]}", ha="center", va="bottom", fontsize=9, color=INK, rotation=90)
    prev = d
ax.set_xlabel("Magnitude Mw (27 values, 7.2 to 9.8 in steps of 0.1)")
ax.set_ylabel("Number of scenarios")
ax.set_ylim(0, 290)
ax.set_xticks(mws)
ax.set_xticklabels([f"{m:.1f}" for m in mws], rotation=90, fontsize=9)
ax.set_title(f"kermadectonga2: {counts.sum():,} scenarios on a {int(stats['alongstrike_number'].max())} x "
             f"{int(stats['downdip_number'].max())} mesh. Label = block size (cells along strike x down dip), "
             "shown where it changes", fontsize=11)
ax.legend(fontsize=9.5, loc="upper right")
save(fig, "engine_kermadectonga2_scenarios_per_mw.png")

# ---------------------------------------------------------------------------
# 3. One scenario drawn on the mesh, columns coloured by convergence
# ---------------------------------------------------------------------------
lon = grid[:, 0, :]
lat = grid[:, 1, :]
n_dd, n_as = lon.shape[0] - 1, lon.shape[1] - 1
prof = np.array(rc["convergent_slip_profile"])
all_eq = events.get_all_earthquake_events(stats, Mmin=7.2, Mmax=9.8, dMw=0.1, source_zone_name=ZONE)
assert np.allclose(np.round(all_eq["area"], 6), np.round(sr.area_km2.values, 6))


def cell_poly(i, j):
    return np.array([[lon[i, j], lat[i, j]], [lon[i, j + 1], lat[i, j + 1]],
                     [lon[i + 1, j + 1], lat[i + 1, j + 1]], [lon[i + 1, j], lat[i + 1, j]]])


fig, axs = plt.subplots(1, 2, figsize=(12, 7.2), gridspec_kw={"width_ratios": [1, 1]})
cmap = plt.get_cmap("Oranges")
norm = matplotlib.colors.Normalize(0, prof.max())
sub = sr[np.isclose(sr.Mw, 8.5)]
first = np.array([int(stats["alongstrike_number"][all_eq["event_indices"][i]].min()) for i in sub.index])
mid = sub.index[(first == 34) & (sub.mean_depth_km.values < sub.mean_depth_km.median())]
picks = [int(sub.rate_mean.idxmax()), int(mid[0])]
notes = ["the most likely Mw 8.5 scenario:\ntouches the north edge (LEVEL 4\nlifts edge scenarios) where Bird\nconvergence is fastest",
         "a Mw 8.5 scenario in the middle\nof the zone (columns 34-40,\nthe up-dip position)"]
for ax, ev_i, note in zip(axs, picks, notes):
    for i in range(n_dd):
        for j in range(n_as):
            ax.add_patch(Polygon(cell_poly(i, j), closed=True, facecolor=cmap(norm(prof[j])),
                                 edgecolor="#9aa3b2", lw=0.4))
    idx = all_eq["event_indices"][ev_i]
    for k in idx:
        i = int(stats["downdip_number"][k]) - 1
        j = int(stats["alongstrike_number"][k]) - 1
        ax.add_patch(Polygon(cell_poly(i, j), closed=True, facecolor="none", edgecolor=ENG, lw=2.0))
    ax.plot(lon[0], lat[0], color=INK, lw=1.4)
    row = sr.iloc[ev_i]
    ax.set_title(f"scenario {int(row.event_id)}: Mw {row.Mw:.1f}, {len(idx)} cells\n"
                 f"slip {row.slip_m:.2f} m, rate {row.rate_mean:.2e} per year", fontsize=11)
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlim(lon.min() - 1.0, lon.max() + 1.0)
    ax.set_ylim(lat.min() - 0.8, lat.max() + 0.8)
    ax.set_xlabel("Longitude (deg E)")
    ax.text(lon[0, 0] - 0.7, lat[0, 0] + 0.1, "north end (column 1)", fontsize=9.5, color=MUTED,
            va="center", ha="right")
    ax.text(lon[0, -1] + 1.8, lat[0, -1] - 0.1, "south end (column 73)", fontsize=9.5, color=MUTED,
            va="center", ha="left")
    ax.text(0.04, 0.55, note, transform=ax.transAxes, fontsize=9.5, color=INK,
            bbox=dict(boxstyle="round", fc="white", ec="#dde1e8"))
axs[0].set_ylabel("Latitude (deg N)")
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=axs, shrink=0.7, pad=0.02)
cb.set_label("Bird convergent slip of the column (mm/yr)")
fig.suptitle("Blue outline: the scenario's cells. Black line: the trench (top edge of the mesh).",
             fontsize=11, y=0.02)
save(fig, "engine_kermadectonga2_scenario_map.png")

# ---------------------------------------------------------------------------
# 4. Conditional probability along strike, Mw 8.5, raw vs edge-corrected
# ---------------------------------------------------------------------------
summ = pd.read_csv(os.path.join(OUT, "logic_tree_summary.csv"))
edge = float(summ.value[summ.item == f"edge_multiplier_{ZONE}"].iloc[0])
m = 8.5
k = np.where(np.isclose(sr.Mw.values, m))[0]
first_col = np.array([int(stats["alongstrike_number"][all_eq["event_indices"][i]].min()) for i in k])
ncols_block = int(stats["alongstrike_number"][all_eq["event_indices"][k[0]]].max() - first_col[0] + 1)
cols = np.unique(first_col)
raw = np.array([cp.ecp_raw.values[k][first_col == c].sum() for c in cols])
cor = np.array([cp.ecp_edge_corrected.values[k][first_col == c].sum() for c in cols])
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.bar(cols - 0.2, raw, width=0.4, color=GREY, label="before LEVEL 4 (area x convergent slip only)")
ax.bar(cols + 0.2, cor, width=0.4, color=ENG, label=f"after LEVEL 4 (edge-touching scenarios x (1 + {edge:.2f}), renormalised)")
ax.set_xlabel(f"First along-strike column of the scenario (each Mw {m} scenario is {ncols_block} columns long)")
ax.set_ylabel("Conditional probability\n(both down-dip positions summed)")
ax.set_title(f"kermadectonga2, Mw {m}: how the rate of the magnitude bin is split between its {k.size} scenarios")
ax2 = ax.twinx()
ax2.plot(np.arange(1, prof.size + 1), prof, color=CONV, lw=1.6, label="Bird convergent slip (mm/yr)")
ax2.set_ylabel("Convergent slip (mm/yr)", color=CONV)
ax2.grid(False)
ax2.set_ylim(0, prof.max() * 1.15)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9.3, loc="upper center")
ax.set_ylim(0, max(cor.max(), raw.max()) * 1.35)
save(fig, "engine_kermadectonga2_conditional_prob.png")

# ---------------------------------------------------------------------------
# 5. Prior vs posterior: coupling and Mw_max
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
c = br.groupby("coupling")[["prior_prob", "posterior_prob"]].sum()
x = np.arange(c.shape[0])
axs[0].bar(x - 0.2, c.prior_prob, width=0.4, color=GREY, label="prior (before GCMT)")
axs[0].bar(x + 0.2, c.posterior_prob, width=0.4, color=SEIS, label="posterior (after GCMT)")
axs[0].set_xticks(x)
axs[0].set_xticklabels([f"{v:.2f}" for v in c.index], rotation=90, fontsize=9)
axs[0].set_xlabel("Coupling value (20 values, log-spaced from 0.10 to 1.30)")
axs[0].set_ylabel("Weight (sums to 1)")
pm = (br.coupling * br.prior_prob).sum()
qm = (br.coupling * br.posterior_prob).sum()
axs[0].set_title(f"Coupling: mean {pm:.2f} before, {qm:.2f} after")
axs[0].legend(fontsize=9.5)
mm = br.groupby("Mw_max")[["prior_prob", "posterior_prob"]].sum()
axs[1].plot(mm.index, mm.prior_prob, "o-", color=GREY, ms=4, label="prior")
axs[1].plot(mm.index, mm.posterior_prob, "o-", color=SEIS, ms=4, label="posterior")
axs[1].annotate("4 of the 40 slots are\nclipped onto 9.6", xy=(9.6, mm.prior_prob.iloc[-1]),
                xytext=(8.75, 0.15), fontsize=9.5, color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED))
axs[1].set_xlabel("Mw_max value")
axs[1].set_ylabel("Weight")
pmx = (br.Mw_max * br.prior_prob).sum()
qmx = (br.Mw_max * br.posterior_prob).sum()
axs[1].set_title(f"Mw_max: mean {pmx:.2f} before, {qmx:.2f} after")
axs[1].legend(fontsize=9.5)
fig.tight_layout()
save(fig, "engine_kermadectonga2_prior_posterior.png")

# ---------------------------------------------------------------------------
# 6. Truncated vs characteristic Gutenberg-Richter, one real branch pair
# ---------------------------------------------------------------------------
t = br.iloc[10544]
ch = br.iloc[10545]
assert t.Mw_max == ch.Mw_max and t.b == ch.b and t.coupling == ch.coupling
mw = np.linspace(7.15, 9.8, 600)


def trunc(mv, a, b, mx):
    return np.where(mv <= mx, 10 ** (a - b * np.maximum(mv, 7.15)) - 10 ** (a - b * mx), 0.0)


def chara(mv, a, b, mx):
    return np.where(mv <= mx, 10 ** (a - b * np.maximum(mv, 7.15)), 0.0)


fig, ax = plt.subplots(figsize=(9.5, 5))
yt = trunc(mw, t.a, t.b, t.Mw_max)
yc = chara(mw, ch.a, ch.b, ch.Mw_max)
ax.semilogy(mw[yt > 0], yt[yt > 0], color=ENG, lw=2.4,
            label=f"truncated GR (branch {int(t.branch_id)}, a = {t.a:.3f})")
ax.semilogy(mw[yc > 0], yc[yc > 0], color=CONV, lw=2.4,
            label=f"characteristic GR (branch {int(ch.branch_id)}, a = {ch.a:.3f})")
ax.axvline(t.Mw_max, color=MUTED, ls=":", lw=1.2)
ax.text(t.Mw_max + 0.03, 1.5e-5, f"Mw_max = {t.Mw_max:.2f}", rotation=90, ha="left", va="bottom", color=MUTED, fontsize=10)
ax.annotate("truncated: the curve bends\ndown to zero at Mw_max", xy=(9.4, trunc(np.array([9.4]), t.a, t.b, t.Mw_max)[0]),
            xytext=(8.05, 2e-5), fontsize=10, color=ENG, arrowprops=dict(arrowstyle="->", color=ENG))
ax.annotate("characteristic: a finite rate\nright up to Mw_max, then a drop", xy=(t.Mw_max, chara(np.array([t.Mw_max - 1e-6]), ch.a, ch.b, ch.Mw_max)[0]),
            xytext=(7.5, 2.5e-4), fontsize=10, color=CONV, arrowprops=dict(arrowstyle="->", color=CONV))
ax.set_xlim(7.15, 9.8)
ax.set_ylim(1e-5, 1)
ax.set_xlabel("Magnitude Mw")
ax.set_ylabel("Rate of earthquakes > Mw (per year)")
ax.set_title(f"Same coupling ({t.coupling:.3f}), b ({t.b:.3f}) and Mw_max: the two GR shapes")
ax.legend(fontsize=9.5, loc="upper right")
save(fig, "engine_gr_truncated_vs_characteristic.png")

# ---------------------------------------------------------------------------
# 7. The engine's own figures, copied
# ---------------------------------------------------------------------------
for src, dst in ((f"fig_branch_fan_{ZONE}.png", f"engine_{ZONE}_branch_fan.png"),
                 (f"fig_moment_balance_{ZONE}.png", f"engine_{ZONE}_moment_balance.png")):
    shutil.copyfile(os.path.join(OUT, "figures", src), os.path.join(IMG, dst))
    print(f"copied {src} -> img/{dst}")
