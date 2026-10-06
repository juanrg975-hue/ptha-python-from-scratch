"""Which convergence source PTHA18 used on each of its subduction zones.

For every unsegmented PTHA18 zone with a thrust interface (rake 90) this
reads PTHA18's own mesh (unit_source_statistics_<zone>.nc, downloaded once
from NCI into validation/ptha18_reference/nc/ unless already in inputs/),
matches every column's top edge to the nearest row of PTHA18's Bird+Griffin
table exactly as step 3 does, and records who supplied the matched rows
(the table's 'collator': Bird2003_subset, JG = Jonathan Griffin, GD = Gareth
Davies). It also computes the zone's convergence with both tables of
generate.py --convergence ("bird", "bird-griffin"). "bird-griffin" on
PTHA18's mesh is PTHA18's own convergence (checked to 1e-15 on 8 zones).

Writes
  from_scratch_v12/data/bird/ptha18_convergence_sources.csv   (read by generate.py)
  from_scratch_v12/html/docs/figures_src/convergence_sources.json

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_convergence_sources.py
"""
import csv
import json
import os
import sys
import zipfile

import netCDF4
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ROOT = os.path.dirname(PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))
sys.path.insert(0, os.path.join(PKG, "validation"))
import bird_convergence as bc  # noqa: E402
from validate_v9 import fetch, THREDDS, REF  # noqa: E402

SZP = os.path.join(ROOT, "rptha", "R", "examples", "austptha_template", "DATA",
                   "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")
KEYS = ("lon_c", "lat_c", "strike", "dip", "width", "length", "rake",
        "downdip_number", "alongstrike_number")


def zones():
    d = pd.read_csv(SZP, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    u = d[d["segment_name"].isna() | (d["segment_name"].astype(str).str.strip() == "")]
    u = u[u["rake"] == 90.0]
    return [(r.sourcename.strip(), int(r.use_bird_convergence), float(r.row_weight),
             r.tectonic_slip) for r in u.itertuples()]


def stats(zone):
    p = os.path.join(ROOT, "inputs", "geometry", f"unit_source_statistics_{zone}.nc")
    if not os.path.exists(p):
        p = fetch(f"{THREDDS}/SOURCE_ZONES/{zone}/TSUNAMI_EVENTS/unit_source_statistics_{zone}.nc",
                  os.path.join(REF, "nc", f"unit_source_statistics_{zone}.nc"))
    with netCDF4.Dataset(p) as ds:
        return {k: np.asarray(ds[k][:], float) for k in KEYS}


with zipfile.ZipFile(bc.BIRD_ZIP) as zf:
    MERGED = bc._add_midpoints(pd.read_csv(zf.open(
        next(n for n in zf.namelist() if n.endswith(".csv")))))

quiet = lambda *a, **k: None  # noqa: E731
rows = []
for zone, use_bird, weight, slip in zones():
    try:
        s = stats(zone)
    except Exception as exc:
        print(f"{zone}: no PTHA18 mesh ({exc})")
        continue
    lr, la = bc.bounding_box(s["lon_c"], s["lat_c"])
    m = MERGED[bc._in_box(MERGED, lr, la)]
    m = m[m["class"].astype(str) != "Normal"]
    top = np.where(s["downdip_number"].astype(int) == 1)[0]
    who, names = [], []
    for k in top:
        e = bc.top_edge_point(s["lon_c"][k], s["lat_c"][k], s["strike"][k], s["dip"][k], s["width"][k])
        r = m.iloc[int(np.argmin(bc.dist_haversine(e[0], e[1], m.mid_lon.values, m.mid_lat.values)))]
        who.append(r.collator)
        if r.collator != "Bird2003_subset":
            names.append(str(r["name"]))
    n = len(top)
    counts = {c: who.count(c) for c in ("Bird2003_subset", "JG", "GD")}
    bg = bc.column_convergence(s, log=quiet, table="bird-griffin")["area_weighted_mean_mm_per_yr"]
    try:
        b = bc.column_convergence(s, log=quiet, table="bird")["area_weighted_mean_mm_per_yr"]
    except SystemExit:
        b = float("nan")
    if counts["Bird2003_subset"] == n:
        source = "bird"
    elif counts["Bird2003_subset"] == 0:
        source = "griffin"
    else:
        source = "mixed"
    lon = float(np.mean(s["lon_c"]) % 360.0)
    rows.append({
        "zone": zone, "row_weight": weight, "use_bird_convergence": use_bird,
        "constant_mm_per_yr": "" if use_bird else float(slip),
        "columns": n, "cols_bird": counts["Bird2003_subset"], "cols_griffin": counts["JG"],
        "cols_davies": counts["GD"], "traces": ";".join(sorted(set(names))),
        "source": source, "conv_bird_griffin": round(bg, 2),
        "conv_bird": round(b, 2) if b == b else "",
        "lon_c": round(lon, 1), "lat_c": round(float(np.mean(s["lat_c"])), 1)})
    print(f"{zone:22s} {source:8s} Bird {counts['Bird2003_subset']:3d} JG {counts['JG']:3d} "
          f"GD {counts['GD']:3d}  bird-griffin {bg:7.2f}  bird {b:7.2f}  "
          f"({lon:.1f}, {np.mean(s['lat_c']):.1f})  {';'.join(sorted(set(names)))}")

out_csv = os.path.join(PKG, "data", "bird", "ptha18_convergence_sources.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump(rows, open(os.path.join(HERE, "convergence_sources.json"), "w"), indent=1)
print("wrote", out_csv, len(rows), "zones")
