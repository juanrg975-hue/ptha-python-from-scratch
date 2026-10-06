"""Figures of validation.html.

1. validation_mesh_puysegur2.png: the mesh this package builds from PTHA18's
   own contours (NCI SOURCEZONE_CONTOURS) against PTHA18's published grid
   (NCI unit_source_grid), node by node; the same comparison validate_v9.py's
   mesh check makes.
2. validation_events_puysegur2.png: every scenario's mean annual rate from the
   engine on PTHA18's official input against PTHA18's published rate_annual.
   Needs the engine output validate_v9.py's events check leaves in
   runs/python/puysegur2_v8_validation_events/ (run that check first).

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_validation.py
"""
import os
import sys

import numpy as np
import pandas as pd
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(DOCS, "..", "..", ".."))
PKG = os.path.join(ROOT, "from_scratch_v12")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))
sys.path.insert(0, os.path.join(PKG, "validation"))
from validate_v9 import ptha18_contours, official_grid_lonlat, hav_m  # noqa: E402
from pyptha_v12 import contour_discretisation as cd  # noqa: E402

OURS, PTHA = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})

# 1. mesh
z = "puysegur2"
grid = cd.discretized_source_from_contours_optimal(ptha18_contours(z), 50.0,
                                                   desired_unit_source_width=35.0, verbose=False)
off = official_grid_lonlat(z)
d = hav_m(grid[:, :2, :].transpose(0, 2, 1).reshape(-1, 2), off.transpose(0, 2, 1).reshape(-1, 2))
print(f"mesh {z}: {grid.shape} vs {off.shape}; node distance median {np.median(d):.2f} m, max {d.max():.2f} m")
fig, ax = plt.subplots(figsize=(6.2, 6.4))
for lev, c in ptha18_contours(z):
    ax.plot(c[:, 0], c[:, 1], color="#c3c2b7", lw=0.8)
for r in range(off.shape[0]):
    ax.plot(off[r, 0], off[r, 1], color=PTHA, lw=4.5, alpha=0.45)
for j in range(off.shape[2]):
    ax.plot(off[:, 0, j], off[:, 1, j], color=PTHA, lw=4.5, alpha=0.45)
for r in range(grid.shape[0]):
    ax.plot(grid[r, 0], grid[r, 1], color=OURS, lw=1.2)
for j in range(grid.shape[2]):
    ax.plot(grid[:, 0, j], grid[:, 1, j], color=OURS, lw=1.2)
ax.plot([], [], color=PTHA, lw=4.5, alpha=0.45, label="PTHA18's published grid")
ax.plot([], [], color=OURS, lw=1.2, label="this code, on PTHA18's contours")
ax.plot([], [], color="#c3c2b7", lw=0.8, label="PTHA18's contours (0 to 40 km)")
ax.set_aspect(1 / np.cos(np.radians(np.mean(grid[:, 1, :]))))
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title(f"puysegur2: {grid.shape[2]-1} x {grid.shape[0]-1} cells both; nodes within {100 * d.max():.0f} cm")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(color="#e1e0d9", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "img", "validation_mesh_puysegur2.png"), dpi=115)
plt.close(fig)

# 2. events
p = pd.read_csv(os.path.join(ROOT, "runs", "python", f"{z}_v8_validation_events", f"scenario_rates_{z}.csv"))
with netCDF4.Dataset(os.path.join(PKG, "validation", "ptha18_reference", "events",
                                  f"all_uniform_slip_earthquake_events_{z}.nc")) as ds:
    ra = np.asarray(ds["rate_annual"][:], float)[:len(p)]
    mw = np.asarray(ds["Mw"][:], float)[:len(p)]
b = p["rate_mean"].to_numpy()
nz = ra > 0
rel = np.abs(b - ra)[nz] / ra[nz]
print(f"events {z}: {len(p)} scenarios, {nz.sum()} with a rate; max rel diff {rel.max():.1e}, median {np.median(rel):.1e}")
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
ax = axs[0]
sc = ax.scatter(ra[nz], b[nz], c=mw[nz], s=14, cmap="viridis")
lo, hi = ra[nz].min() / 2, ra[nz].max() * 2
ax.plot([lo, hi], [lo, hi], color="#5b6475", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("PTHA18's published rate_annual (per year)")
ax.set_ylabel("this engine's rate_mean (per year)")
ax.set_title(f"every scenario of {z} ({nz.sum()} with a rate)")
cb = fig.colorbar(sc, ax=ax); cb.set_label("Mw")
ax = axs[1]
ax.semilogy(mw[nz], np.maximum(rel, 1e-17), "o", ms=3, color=OURS)
ax.axhline(1e-7, color="#c43d3d", ls="--", lw=1)
ax.text(mw[nz].min(), 1.6e-7, "PASS threshold 1e-7", color="#c43d3d", fontsize=9)
ax.set_xlabel("Mw")
ax.set_ylabel("relative difference")
ax.set_title(f"relative difference: at most {rel.max():.1e}")
ax.grid(color="#e1e0d9", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "img", "validation_events_puysegur2.png"), dpi=115)
plt.close(fig)
print("done")
