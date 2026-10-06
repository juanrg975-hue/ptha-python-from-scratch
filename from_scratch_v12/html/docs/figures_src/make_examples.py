"""Data and images for html/docs/examples.html.

Reads every example folder's own files (run_final.log, outputs/,
outputs_official/, data/step1_contours_info.json, figures/) and writes
  html/docs/img/examples_<zone>_<name>.png   (resized copies of the figures)
  html/docs/img/examples_overview.png        (all 8 meshes, same scale per panel)
  html/docs/figures_src/examples_data.json   (the numbers the page quotes)

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_examples.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
IMG = os.path.join(DOCS, "img")
ROOT = os.path.abspath(os.path.join(DOCS, "..", "..", ".."))
FOLDERS = ["kermadectonga2_v9", "southamerica_v9", "kurilsjapan_v9", "makran2_v9",
           "puysegur2_v9", "calabria2_v9", "caribbean2_v9", "antilles2_v9"]
FIGS = ["step1_trench_and_contours", "scratch_mesh_map", "mesh_comparison_map",
        "scratch_mesh_map_convergence"]


def grab(pat, text, cast=str):
    m = re.search(pat, text)
    return cast(m.group(1).replace(",", "")) if m else None


def rate_at(path, mw):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    i = (d["Mw"] - mw).abs().idxmin()
    return float(d["exceedance_rate_unsegmented"].iloc[i])


def summary(path):
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    return dict(zip(d["item"], d["value"]))


def official_area(base, zone):
    """PTHA18's mesh area, from the cell table step 8 saved for the report."""
    p = os.path.join(base, "outputs_official", "ptha18_reference",
                     f"unit_source_statistics_{zone}.nc")
    if not os.path.exists(p):
        return None
    import netCDF4
    with netCDF4.Dataset(p) as ds:
        return round(float((np.asarray(ds["length"][:]) * np.asarray(ds["width"][:])).sum()), 1)


def shrink(src, dst, max_w=1400):
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(dst, optimize=True)


os.makedirs(IMG, exist_ok=True)
# Folders whose current contents are a user's own experiment, not the
# documented run: their record and images are kept as they were.
KEEP = set()
_old_path = os.path.join(HERE, "examples_data.json")
old = json.load(open(_old_path)) if os.path.exists(_old_path) else {}
data = {}
for f in FOLDERS:
    if f in KEEP and f in old:
        data[f] = old[f]
        print(f, "kept from the previous examples_data.json")
        continue
    base = os.path.join(ROOT, f)
    log = open(os.path.join(base, "run_final.log"), encoding="utf-8", errors="replace").read()
    # kurilsjapan_v9 ran steps 1-3 once before the --auto-clip run: read the last run only
    log = log[log.rfind("STEP 1 - "):] if "STEP 1 - " in log else log
    info = json.load(open(os.path.join(base, "data", "step1_contours_info.json")))
    zone = info["zone"]
    s = summary(os.path.join(base, "outputs", "logic_tree_summary.csv"))
    so = summary(os.path.join(base, "outputs_official", "logic_tree_summary.csv"))
    g = np.load(os.path.join(base, "data", "slab2", "unit_source_grid.npy"))
    ends = {}
    for k, e in (info.get("ends") or {}).items():
        ends[k] = {x: e.get(x) for x in ("cut", "reason", "trench_km", "contours_km",
                                         "deviation_deg", "deviation_after_deg", "tip_width_ratio")}
    rec = {
        "zone": zone,
        "rows": g.shape[0] - 1, "cols": g.shape[2] - 1,
        "optimal": grab(r"    (optimal: [^\n]*)", log),
        "mesh_check": grab(r"mesh check: ([^\n]*)", log),
        "area": grab(r"total area\s+([\d,\.]+) km", log, float),
        "area_off": official_area(base, zone),
        "dip": s.get(f"mean_dip_deg_{zone}"), "dip_off": so.get(f"mean_dip_deg_{zone}"),
        "conv": s.get(f"convergence_mm_per_yr_{zone}"), "conv_off": so.get(f"convergence_mm_per_yr_{zone}"),
        "edge": s.get(f"edge_multiplier_{zone}"), "edge_off": so.get(f"edge_multiplier_{zone}"),
        "tvd": s.get(f"prior_posterior_tvd_{zone}"),
        "gcmt": grab(r"LEVEL 3 data: (\d+) events", log, int),
        "trench_km": info.get("trench_length_km") or info.get("levels", [{}])[0].get("length_km"),
        "cutoff_km": info.get("cutoff_km"),
        "ends": ends,
    }
    for mw in (7.2, 8.0, 9.0):
        rec[f"r{mw}"] = rate_at(os.path.join(base, "outputs", "rate_curves.csv"), mw)
        rec[f"r{mw}_off"] = rate_at(os.path.join(base, "outputs_official", "rate_curves.csv"), mw)
    cmd = None
    readme = os.path.join(base, "README.md")
    if os.path.exists(readme):
        m = re.search(r"(\.venv/Scripts/python\.exe from_scratch_v12/generate\.py[^\n]*)", open(readme, encoding="utf-8").read())
        cmd = m.group(1).strip() if m else None
    rec["command"] = cmd
    for name in FIGS:
        src = os.path.join(base, "figures", name + ".png")
        if name == "scratch_mesh_map_convergence" and zone not in ("kermadectonga2", "southamerica", "calabria2"):
            continue  # the page shows convergence maps for these three only
        if os.path.exists(src):
            shrink(src, os.path.join(IMG, f"examples_{zone if f != 'caribbean2_v9' else 'caribbean2'}_{name}.png"))
    data[f] = rec
    print(f, rec["rows"], rec["cols"], rec["area"], rec["optimal"])

json.dump(data, open(os.path.join(HERE, "examples_data.json"), "w"), indent=1, default=float)

# small multiples: every mesh with its top edge, each panel with equal km scale in x and y
fig, axes = plt.subplots(2, 4, figsize=(13, 7.2))
for ax, f in zip(axes.ravel(), FOLDERS):
    g = np.load(os.path.join(ROOT, f, "data", "slab2", "unit_source_grid.npy"))
    lon, lat = g[:, 0, :], g[:, 1, :]
    for r in range(g.shape[0]):
        ax.plot(lon[r], lat[r], color="#eb6834", lw=0.6)
    for c in range(g.shape[2]):
        ax.plot(lon[:, c], lat[:, c], color="#eb6834", lw=0.6)
    ax.plot(lon[0], lat[0], color="#1d2433", lw=2.0)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.set_title(f"{f}  ({g.shape[2]-1} x {g.shape[0]-1})", fontsize=10)
    ax.tick_params(labelsize=7.5)
    ax.grid(color="#e1e0d9", lw=0.5)
fig.suptitle("The 8 example meshes (thick dark line: the top edge, on the trench)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "examples_overview.png"), dpi=110, facecolor="white")
print("done")
