"""Re-run the checks behind html/docs/validation.html against PTHA18's own files.

    .venv/Scripts/python.exe from_scratch_v12/validation/validate_v9.py            # all checks
    .venv/Scripts/python.exe from_scratch_v12/validation/validate_v9.py mesh       # one check

Checks (each prints one line per zone and a PASS/FAIL verdict):

  mesh     v8's discretiser on PTHA18's OWN contours (NCI DATA/
           SOURCEZONE_CONTOURS.zip) vs PTHA18's published unit-source grid
           (NCI SOURCE_ZONES/<zone>/EQ_SOURCE/unit_source_grid/<zone>.shp) and
           statistics table (unit_source_statistics_<zone>.nc): same
           dimensions, same area and mean dip, node positions.
  engine   v8's logic-tree engine on the official inputs in inputs/ vs
           PTHA18's official branch tables (official_ptha_data/trees/): the GR
           'a' value, prior and posterior weight of every one of the 32,000
           branches.
  events   the rate of EVERY earthquake scenario, mean and LEVEL 5
           percentiles: v8's engine on the official input (plus PTHA18's
           Bird model where PTHA18 used it, exactly as step 8 adds it) vs
           PTHA18's published rate_annual* columns (NCI SOURCE_ZONES/<zone>/
           TSUNAMI_EVENTS/all_uniform_slip_earthquake_events_<zone>.nc), on
           the zones PTHA18 did not segment. Also checks that v8's event
           table lists the same events in the same row order.
  gcmt     step 5's event-selection rule, applied to PTHA18's own mesh, vs
           the events PTHA18 itself selected (official_gcmt_observations.csv),
           for the zone and (v9) every PTHA18 segment on its own columns.
           Needs GCMT .ndk files: pass --ndk DIR (any example's data/gcmt/).
  segmentation       v9: Bird's segments tile each example folder's mesh;
                     prints them beside PTHA18's (informational).
  berryman_segments  v9: the default boundaries, Berryman et al. (2015)
                     Table 3.1's end points, on PTHA18's own mesh vs PTHA18's
                     indices: within 2 columns wherever both split, 9 zones.
  official_segments  v9: the engine on PTHA18's own segmented input (as step
                     8 builds it) vs PTHA18's tree of every segment, 7 zones.
  events_segmented   v9: every scenario's rate and 5 percentiles of those 7
                     SEGMENTED zones vs PTHA18's published rate_annual*.
  v8parity           v9: an unsegmented v9 folder (<zone>_v9) vs v8's
                     (<zone>_v8), bit for bit.
  unseg_view         v9: a segmented folder's unsegmented branch
                     (<zone>_v9seg) vs the unsegmented folder (<zone>_v9),
                     bit for bit: what report_unsegmented.html is built on.

official_segments and events_segmented need PTHA18's extracted trees of
kermadectonga2 kurilsjapan izumariana ryuku sunda2 southamerica
alaskaaleutians (official_ptha_data/fetch_official_inputs.py, R).

Reference files are downloaded once into validation/ptha18_reference/ from
the public NCI THREDDS server (no account needed).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))

REF = os.path.join(HERE, "ptha18_reference")
THREDDS = "https://thredds.nci.org.au/thredds/fileServer/fj6/PTHA/AustPTHA_1"
MESH_ZONES = ["puysegur2", "cascadia", "makran2", "kermadectonga2", "newhebrides2",
              "philippine", "izumariana", "solomon2", "newguinea2", "mexico",
              "kurilsjapan", "alaskaaleutians", "sunda2", "southamerica"]
WIDTH = {"puysegur2": 35.0}          # sourcezone_parameters.csv; 50 km elsewhere


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                timeout=600) as r:
        blob = r.read()
    with open(dest + ".part", "wb") as fh:
        fh.write(blob)
    os.replace(dest + ".part", dest)
    return dest


def ptha18_contours(zone):
    z = fetch(f"{THREDDS}/DATA/SOURCEZONE_CONTOURS.zip", os.path.join(REF, "SOURCEZONE_CONTOURS.zip"))
    out = os.path.join(REF, "contours")
    if not os.path.exists(os.path.join(out, "SOURCEZONE_CONTOURS", zone + ".shp")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    import geopandas as gpd
    g = gpd.read_file(os.path.join(out, "SOURCEZONE_CONTOURS", zone + ".shp"))
    return [(float(r["level"]), np.asarray(r.geometry.coords, float)[:, :2]) for _, r in g.iterrows()]


def official_grid_lonlat(zone):
    """(rows+1, 2, cols+1) lon/lat of PTHA18's grid, from its polygon shapefile."""
    for e in ("shp", "shx", "dbf", "prj"):
        fetch(f"{THREDDS}/SOURCE_ZONES/{zone}/EQ_SOURCE/unit_source_grid/{zone}.{e}",
              os.path.join(REF, "unit_source_grid", f"{zone}.{e}"))
    import geopandas as gpd
    g = gpd.read_file(os.path.join(REF, "unit_source_grid", f"{zone}.shp"))
    nd, na = int(g["dwndp_n"].max()), int(g["alngst_"].max())
    out = np.full((nd + 1, 2, na + 1), np.nan)
    for _, r in g.iterrows():
        i, j = int(r["dwndp_n"]) - 1, int(r["alngst_"]) - 1
        c = np.asarray(r.geometry.exterior.coords)[:4]
        out[i, :, j], out[i, :, j + 1], out[i + 1, :, j + 1], out[i + 1, :, j] = c[0], c[1], c[2], c[3]
    return out


def official_stats(zone):
    import netCDF4
    p = os.path.join(ROOT, "inputs", "geometry", f"unit_source_statistics_{zone}.nc")
    if not os.path.exists(p):
        p = fetch(f"{THREDDS}/SOURCE_ZONES/{zone}/TSUNAMI_EVENTS/unit_source_statistics_{zone}.nc",
                  os.path.join(REF, "nc", f"unit_source_statistics_{zone}.nc"))
    with netCDF4.Dataset(p) as ds:
        return {k: np.asarray(ds[k][:], float) for k in
                ("lon_c", "lat_c", "strike", "dip", "length", "width", "depth",
                 "alongstrike_number", "downdip_number")}


def mean_angle(d):
    r = np.radians(d)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


def hav_m(a, b):
    la1, la2 = np.radians(a[:, 1]), np.radians(b[:, 1])
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin(np.radians(b[:, 0] - a[:, 0]) / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.minimum(1.0, h)))


def check_mesh(zones):
    from pyptha_v12 import contour_discretisation as cd
    from pyptha_v12 import unit_sources as us
    ok_all = True
    print("\nMESH: v8 discretiser on PTHA18's contours vs PTHA18's published grid")
    for z in zones:
        grid = cd.discretized_source_from_contours_optimal(
            ptha18_contours(z), 50.0, desired_unit_source_width=WIDTH.get(z, 50.0), verbose=False)
        off = official_grid_lonlat(z)
        st = us.discretized_source_approximate_summary_statistics(grid)
        o = official_stats(z)
        area, oarea = float(np.sum(st["length"] * st["width"])), float(np.sum(o["length"] * o["width"]))
        same = grid.shape[0] == off.shape[0] and grid.shape[2] == off.shape[2]
        med = mx = np.nan
        if same:
            a = np.column_stack([grid[:, 0, :].ravel(), grid[:, 1, :].ravel()])
            b = np.column_stack([np.mod(off[:, 0, :].ravel(), 360), off[:, 1, :].ravel()])
            a[:, 0] = np.mod(a[:, 0], 360)
            d1 = hav_m(a, b)
            b2 = np.column_stack([np.mod(off[:, 0, ::-1].ravel(), 360), off[:, 1, ::-1].ravel()])
            d2 = hav_m(a, b2)
            d = d1 if d1.mean() < d2.mean() else d2
            med, mx = float(np.median(d)), float(d.max())
        good = same and abs(area / oarea - 1) < 2e-3 and abs(mean_angle(st["dip"]) - mean_angle(o["dip"])) < 0.05
        ok_all &= good
        print(f"  {z:16s} dims {grid.shape[0]-1}x{grid.shape[2]-1} vs {off.shape[0]-1}x{off.shape[2]-1}  "
              f"area {100*(area/oarea-1):+.3f}%  dip {mean_angle(st['dip']):.3f}/{mean_angle(o['dip']):.3f}  "
              f"node offset median {med:.1f} m max {mx:.0f} m  {'PASS' if good else 'FAIL'}")
    return ok_all


def require_extraction(check, zones, need_branches=False, need_gcmt=False):
    """engine, events and gcmt compare against PTHA18's own inputs and tree,
    which fetch_official_inputs.py extracts from the saved session (R + the
    RData, as step 8 does). Stop with that command if any file is missing."""
    trees = os.path.join(ROOT, "official_ptha_data", "trees")
    missing = []
    for z in zones:
        if not need_gcmt and not os.path.exists(os.path.join(ROOT, "inputs", f"input_{z}.json")):
            missing.append(f"inputs/input_{z}.json")
        if need_branches and not os.path.exists(os.path.join(trees, f"logic_tree_branches_{z}_OFFICIAL.csv")):
            missing.append(f"official_ptha_data/trees/logic_tree_branches_{z}_OFFICIAL.csv")
    if need_gcmt and not os.path.exists(os.path.join(trees, "official_gcmt_observations.csv")):
        missing.append("official_ptha_data/trees/official_gcmt_observations.csv")
    if missing:
        raise SystemExit(
            f"\n{check.upper()}: PTHA18's extracted inputs are not here yet:\n  "
            + "\n  ".join(missing)
            + "\nExtract them once (needs R, rptha and the saved session, see RData.txt):\n"
            f"  .venv/Scripts/python.exe official_ptha_data/fetch_official_inputs.py {' '.join(zones)}\n"
            "The mesh check needs none of this (it downloads PTHA18's published files itself).")


def check_engine(zones=("puysegur2", "kermadectonga2", "kurilsjapan", "southamerica")):
    require_extraction("engine", zones, need_branches=True)
    import pandas as pd
    ok_all = True
    print("\nENGINE: v8 logic tree on the official inputs vs PTHA18's official branches")
    runner = os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py")
    for z in zones:
        cfg = json.load(open(os.path.join(ROOT, "inputs", f"input_{z}.json")))
        cfg["run_name"] = f"{z}_v8_validation"
        tmp = os.path.join(HERE, f"_input_{z}_v8_validation.json")
        json.dump(cfg, open(tmp, "w"), indent=1)
        subprocess.run([sys.executable, runner, tmp], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        os.remove(tmp)
        p = pd.read_csv(os.path.join(ROOT, "runs", "python", cfg["run_name"], f"logic_tree_branches_{z}.csv"))
        o = pd.read_csv(os.path.join(ROOT, "official_ptha_data", "trees", f"logic_tree_branches_{z}_OFFICIAL.csv"))
        key = lambda d: list(zip(d.slip_rate.round(12), d.b.round(10), d.Mw_max.round(8), d.Mw_frequency_distribution))
        o["k"], p["k"] = key(o), key(p)
        m = o.merge(p.drop_duplicates("k"), on="k", suffixes=("_o", "_p"))
        da = float(np.max(np.abs(m.a_o - m.a_p)))
        dp = float(np.max(np.abs(m.posterior_prob_o - m.posterior_prob_p)))
        good = len(p) == len(o) and da < 1e-9 and dp < 1e-12
        ok_all &= good
        print(f"  {z:16s} branches {len(p)}/{len(o)}  max|a diff| {da:.1e}  "
              f"max|posterior diff| {dp:.1e}  {'PASS' if good else 'FAIL'}")
    return ok_all


def check_events(zones=("puysegur2", "cascadia", "makran2", "philippine")):
    require_extraction("events", zones)
    """Zones PTHA18 did not segment: their published per-event rates are the
    unsegmented model alone, which is what the engine computes."""
    import netCDF4
    import pandas as pd
    import bird_convergence as bc
    from pyptha_v12 import events
    ok_all = True
    print("\nEVENTS: v8 per-scenario rates on the official inputs vs PTHA18's published rate_annual")
    runner = os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py")
    for z in zones:
        cfg = json.load(open(os.path.join(ROOT, "inputs", f"input_{z}.json")))
        with netCDF4.Dataset(os.path.join(ROOT, cfg["geometry"]["official_statistics_nc"])) as ds:
            st = {k: np.asarray(ds[k][:]) for k in ds.variables
                  if ds[k].ndim == 1 and ds[k].dtype.kind in "fi"}
        for k in ("subfault_number", "downdip_number", "alongstrike_number"):
            st[k] = st[k].astype(int)
        model = "constant"
        if bc.ptha18_uses_bird_convergence(z):
            cfg["rates"].update(bc.rates_entries(bc.column_convergence(st, log=lambda *a: None)))
            model = "Bird"
        cfg["run_name"] = f"{z}_v8_validation_events"
        tmp = os.path.join(HERE, f"_input_{z}_v8_validation_events.json")
        json.dump(cfg, open(tmp, "w"), indent=1)
        subprocess.run([sys.executable, runner, tmp], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        os.remove(tmp)
        p = pd.read_csv(os.path.join(ROOT, "runs", "python", cfg["run_name"], f"scenario_rates_{z}.csv"))

        nc = fetch(f"{THREDDS}/SOURCE_ZONES/{z}/TSUNAMI_EVENTS/all_uniform_slip_earthquake_events_{z}.nc",
                   os.path.join(REF, "events", f"all_uniform_slip_earthquake_events_{z}.nc"))
        # PTHA18's mean rate and its LEVEL 5 percentiles, against the engine's
        columns = {"rate_annual": "rate_mean", "rate_annual_lower_ci": "rate_p0.025",
                   "rate_annual_16pc": "rate_p0.16", "rate_annual_median": "rate_p0.5",
                   "rate_annual_84pc": "rate_p0.84", "rate_annual_upper_ci": "rate_p0.975"}
        with netCDF4.Dataset(nc) as ds:
            eis = [s.strip() for s in netCDF4.chartostring(ds["event_index_string"][:])]
            off = {k: np.asarray(ds[k][:], float) for k in columns}
        ev = events.get_all_earthquake_events(st, Mmin=7.2, Mmax=9.8, dMw=0.1, mu=3e10,
                                              relation="Strasser", source_zone_name=z)
        same_order = list(ev["event_index_string"]) == eis
        n = len(p)                                  # the engine's range stops at Mw 9.6
        worst = {}
        zero_mismatch = 0
        for ok_, pk in columns.items():
            a, b = off[ok_][:n], p[pk].to_numpy()
            zero_mismatch += int(np.sum((a == 0) != (b == 0)))
            nz = a > 0
            worst[ok_] = float(np.max(np.abs(b - a)[nz] / a[nz]))
        good = same_order and zero_mismatch == 0 and max(worst.values()) < 1e-7
        ok_all &= good
        print(f"  {z:16s} {model:8s} events {n}/{len(eis)}, same rows as PTHA18: {same_order}  "
              f"max rel diff: mean rate {worst['rate_annual']:.1e}, percentiles "
              f"{max(v for k, v in worst.items() if k != 'rate_annual'):.1e}, "
              f"zero/non-zero mismatches {zero_mismatch}  {'PASS' if good else 'FAIL'}")
    return ok_all


def check_gcmt(ndk_dir, zones=("kermadectonga2", "puysegur2", "kurilsjapan", "southamerica",
                                "makran2", "cascadia", "izumariana", "philippine")):
    require_extraction("gcmt", zones, need_gcmt=True)
    import pandas as pd
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from pyptha_v12 import contour_discretisation as cd
    from pyptha_v12 import unit_sources as us
    print("\nGCMT: step 5's rule on PTHA18's own mesh vs PTHA18's own selected events")
    lines = []
    for f in sorted(glob.glob(os.path.join(ndk_dir, "*.ndk"))):
        lines += open(f, encoding="latin-1").read().splitlines()
    re1 = re.compile(r"^\S+\s+(\d{4})/(\d{2})/(\d{2})\s+\S+\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)")
    ev = []
    for i in range(len(lines) // 5):
        b = lines[i * 5:(i + 1) * 5]
        m = re1.match(b[0])
        if not m:
            continue
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if not ("1976-01-01" <= date <= "2017-03-01"):
            continue
        try:
            p3, p4, p5 = b[2].split(), b[3].split(), b[4].split()
            e, mt = float(p4[0]), [float(v) for v in p4[1:13:2]]
            m0 = np.sqrt(0.5 * (mt[0] ** 2 + mt[1] ** 2 + mt[2] ** 2) + mt[3] ** 2 + mt[4] ** 2 + mt[5] ** 2) * 10 ** e * 1e-7
            s1, _, r1, s2, _, r2 = (float(v) for v in p5[-6:])
            ev.append((date, float(m.group(4)), float(m.group(5)), float(m.group(6)), float(p3[3]), float(p3[5]),
                       (2 / 3) * (np.log10(m0) - 9.05), s1, r1, s2, r2))
        except (ValueError, IndexError):
            continue
    cat = pd.DataFrame(ev, columns=["date", "hlat", "hlon", "hdep", "clat", "clon", "Mw", "s1", "r1", "s2", "r2"])
    cat = cat.drop_duplicates(["date", "hlat", "hlon", "hdep"])
    cat = cat[cat.Mw >= 7.15]
    off = pd.read_csv(os.path.join(ROOT, "official_ptha_data", "trees", "official_gcmt_observations.csv"))
    ang = lambda a, b: np.abs((a - b + 180) % 360 - 180)
    def select(grid, ref):
        st = us.discretized_source_approximate_summary_statistics(grid)
        nr, _, nc = grid.shape
        region = unary_union([Polygon([(grid[r, 0, j], grid[r, 1, j]), (grid[r, 0, j + 1], grid[r, 1, j + 1]),
                                       (grid[r + 1, 0, j + 1], grid[r + 1, 1, j + 1]), (grid[r + 1, 0, j], grid[r + 1, 1, j])]
                                      ).buffer(0.4, quad_segs=5) for r in range(nr - 1) for j in range(nc - 1)])
        near = lambda lo: lo + 360 * np.round((ref - lo) / 360)
        keep = []
        for _, e in cat.iterrows():
            if e.hdep > 71:
                continue
            if not (region.contains(Point(near(e.hlon), e.hlat)) or region.contains(Point(near(e.clon), e.clat))):
                continue
            d = hav_m(np.column_stack([np.full(len(st["lon_c"]), e.clon), np.full(len(st["lon_c"]), e.clat)]),
                      np.column_stack([st["lon_c"], st["lat_c"]]))
            s = st["strike"][int(np.argmin(d))] % 360
            if (ang(e.r1, 90) <= 50 and ang(e.s1, s) <= 50) or (ang(e.r2, 90) <= 50 and ang(e.s2, s) <= 50):
                keep.append(e.Mw)
        return np.sort(np.array(keep))

    import segmentation as sg
    sys.path.insert(0, os.path.join(ROOT, "official_ptha_data"))
    from fetch_official_inputs import SZP
    ok_all = True
    for z in zones:
        grid = cd.discretized_source_from_contours_optimal(
            ptha18_contours(z), 50.0, desired_unit_source_width=WIDTH.get(z, 50.0), verbose=False)
        ref = float(np.mean(grid[:, 0, :]))
        # the zone, then (v9) every PTHA18 segment on its own columns, the
        # rule of gcmt_subsetter.R 106-142 that step 5 applies to a segment
        parts = [(z, grid)] + [
            (f"{z}_{s['name']}", grid[:, :, (s["ptha18_slice"][0] - 1):s["ptha18_slice"][1]])
            for s in sg.ptha18_segments(z, grid.shape[2] - 1, SZP, log=lambda *a: None)]
        for name, g in parts:
            o = np.sort(off[off.source == name]["Mw"].dropna().to_numpy())
            k = select(g, ref)
            good = len(k) == len(o) and np.allclose(k, o, atol=1e-9)
            ok_all &= good
            print(f"  {name:28s} official {len(o):2d} events, step 5 rule {len(k):2d}  "
                  f"{'PASS (identical list)' if good else 'FAIL'}")
    return ok_all


def check_segmentation(zones=("kermadectonga2", "izumariana", "ryuku",
                              "kurilsjapan", "sunda2", "southamerica",
                              "alaskaaleutians")):
    """v9 (LEVEL 0). Bird-derived segment boundaries vs PTHA18's own.

    This check does NOT demand a match. Bird (2003)'s plate pairs and PTHA18's
    segment list are two different source models, and lib/segmentation.py's
    docstring explains where they part company (kermadectonga2's _hikurangi has
    no Bird plate boundary of its own; alaskaaleutians is a single NA/PA pair
    for its whole length). Forcing agreement would mean reverse-engineering
    PTHA18's answer and then reporting it as independent.

    What it DOES enforce is that the Bird segmentation is well formed, which is
    what the rest of the pipeline relies on:
      * the segments tile the mesh exactly -- every column in one segment, no
        gaps and no overlaps, so the union-of-segments branch loses no moment;
      * no segment is shorter than the minimum;
      * the slices are within the mesh.
    It then PRINTS the comparison with PTHA18 so the difference stays visible.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
    import numpy as np
    import segmentation as sg

    szp = os.path.join(ROOT, "rptha", "R", "examples", "austptha_template",
                       "DATA", "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")
    print("\nSEGMENTATION (v9, LEVEL 0): Bird (2003) plate pairs vs PTHA18")
    print("  a differing segment count is a model difference, not a failure "
          "(see lib/segmentation.py)")

    ok = True
    for zone in zones:
        grids = glob.glob(os.path.join(ROOT, f"{zone}_v*", "data", "slab2",
                                       "unit_source_grid.npy"))
        if not grids:
            print(f"  {zone:<18} skipped (no example folder with a mesh)")
            continue
        grid = np.load(sorted(grids)[0])
        n = grid.shape[2] - 1
        segs = sg.bird_segments(grid, log=lambda *a: None)

        covered = []
        for s in segs:
            a_, b_ = s["alongstrike_slice"]
            covered += list(range(a_, b_ + 1))
        tiled = covered == list(range(1, n + 1))
        ok = ok and tiled

        pt = sg.ptha18_segments(zone, n, szp, log=lambda *a: None) if os.path.exists(szp) else []
        verdict = "ok" if tiled else "NOT TILED"
        print(f"  {zone:<18} mesh {n:>4} cols | Bird {len(segs)} seg | "
              f"PTHA18 {len(pt) if pt else '-'} seg | tiling {verdict}")
        if pt and segs:
            bb = [s["alongstrike_slice"][1] for s in segs[:-1]]
            pb = [s["alongstrike_slice"][1] for s in pt[:-1]]
            for p in pb:
                near = min(bb, key=lambda x: abs(x - p)) if bb else None
                if near is not None:
                    print(f"      PTHA18 boundary {p:>4} -> nearest Bird "
                          f"{near:>4}  ({abs(near - p)} columns)")
    print(f"  -> segmentation {'PASS' if ok else 'FAIL'} "
          f"(well-formedness; the comparison above is informational)")
    return ok


def check_berryman_segments(zones=("kermadectonga2", "kurilsjapan", "izumariana", "ryuku",
                                   "sunda2", "southamerica", "alaskaaleutians",
                                   "newhebrides2", "solomon2")):
    """v9 (LEVEL 0). The default segment boundaries, Berryman et al. (2015)
    Table 3.1's trench end points (segmentation.berryman_segments), placed on
    PTHA18's OWN published mesh, vs PTHA18's own boundary indices
    (sourcezone_parameters.csv). PTHA18 took its segments from that table, so
    wherever both have the same two neighbouring segments the boundary must
    land within 2 columns of PTHA18's. Boundaries PTHA18 does not share
    (it merged Patagonia North+South and Alaska's four eastern segments, and
    added Arakan on sunda2) are printed, not judged."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
    import segmentation as sg
    szp = os.path.join(ROOT, "rptha", "R", "examples", "austptha_template",
                       "DATA", "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")
    print("\nBERRYMAN SEGMENTS (v9, LEVEL 0): Table 3.1 end points on PTHA18's own "
          "mesh vs PTHA18's boundary indices")
    ok = True
    for z in zones:
        g = official_grid_lonlat(z)
        n = g.shape[2] - 1
        b = sg.berryman_segments(g, z, log=lambda *a: None)
        p = sg.ptha18_segments(z, n, szp, log=lambda *a: None)
        bb = {(x["segment_key"], y["segment_key"]): x["alongstrike_slice"][1]
              for x, y in zip(b[:-1], b[1:])}
        # a segment PTHA18 merged from Berryman's is named after the
        # Berryman segment that sits at the boundary in question
        alias = {"patagonia": "patagonia_north", "eastern": "shumagin"}
        pb = {(alias.get(x["name"], x["name"]), alias.get(y["name"], y["name"])):
              x["ptha18_slice"][1] for x, y in zip(p[:-1], p[1:])}
        worst, lines = 0, []
        for pair, col in pb.items():
            if pair in bb:
                worst = max(worst, abs(bb[pair] - col))
                lines.append(f"{pair[0]}|{pair[1]} {bb[pair]} vs {col}")
            else:
                lines.append(f"{pair[0]}|{pair[1]} PTHA18 only ({col})")
        good = worst <= 2
        ok &= good
        print(f"  {z:<16} {n:>4} cols, Berryman {len(b)} seg, PTHA18 {len(p)}: "
              f"worst {worst} col  {'PASS' if good else 'FAIL'}  ({'; '.join(lines)})")
    print(f"  -> berryman_segments {'PASS' if ok else 'FAIL'}")
    return ok


def official_segmented_input(z):
    """PTHA18's own segmented run of zone z, as step 8 builds it for a
    segmented example: the official input, PTHA18's Bird model on its mesh,
    and every segment of sourcezone_parameters.csv with its session GCMT."""
    import netCDF4
    import bird_convergence as bc
    import segmentation as sg
    sys.path.insert(0, os.path.join(ROOT, "official_ptha_data"))
    from fetch_official_inputs import SZP, read_gcmt_observations
    cfg = json.load(open(os.path.join(ROOT, "inputs", f"input_{z}.json")))
    if bc.ptha18_uses_bird_convergence(z):
        with netCDF4.Dataset(os.path.join(ROOT, cfg["geometry"]["official_statistics_nc"])) as ds:
            st = {k: np.asarray(ds[k][:], float) for k in (
                "lon_c", "lat_c", "strike", "dip", "width", "length", "rake",
                "downdip_number", "alongstrike_number")}
        cfg["rates"].update(bc.rates_entries(bc.column_convergence(st, log=lambda *a: None)))
    trees = os.path.join(ROOT, "official_ptha_data", "trees")
    cfg["segments"] = sg.ptha18_official_segments_block(
        z, cfg["rates"], SZP, lambda name: read_gcmt_observations(trees, name),
        log=lambda *a: None)
    return cfg


def check_official_segments(zones=("kermadectonga2", "kurilsjapan", "izumariana", "ryuku",
                                  "sunda2", "southamerica", "alaskaaleutians")):
    """v9 (LEVEL 0). The engine on PTHA18's OWN segmented input vs PTHA18's
    official tree of every segment (logic_tree_branches_<zone>_<segment>_
    OFFICIAL.csv, extracted from the saved session): 'a' and posterior weight
    of every branch. This is what licenses step 8's official segmented run as
    PTHA18's answer, and it exercises the engine's segment path (its own
    convergence, area, Mw_max anchor, coupling and GCMT events) against R."""
    require_extraction("official_segments", zones, need_branches=True)
    import pandas as pd
    ok_all = True
    print("\nOFFICIAL SEGMENTS: engine on PTHA18's segmented inputs vs PTHA18's segment trees")
    runner = os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py")
    trees = os.path.join(ROOT, "official_ptha_data", "trees")
    for z in zones:
        cfg = official_segmented_input(z)
        cfg["run_name"] = f"{z}_v9_validation_segments"
        cfg["percentiles"]["N"] = 2000          # LEVEL 5 is not compared here
        tmp = os.path.join(HERE, f"_input_{z}_v9_validation_segments.json")
        json.dump(cfg, open(tmp, "w"), indent=1)
        subprocess.run([sys.executable, runner, tmp], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        os.remove(tmp)
        out = os.path.join(ROOT, "runs", "python", cfg["run_name"])
        for rep in [z] + [f"{z}_{s}" for s in cfg["segments"]]:
            p = pd.read_csv(os.path.join(out, f"logic_tree_branches_{rep}.csv"))
            o = pd.read_csv(os.path.join(trees, f"logic_tree_branches_{rep}_OFFICIAL.csv"))
            key = lambda d: list(zip(d.slip_rate.round(12), d.b.round(10), d.Mw_max.round(8), d.Mw_frequency_distribution))
            o["k"], p["k"] = key(o), key(p)
            m = o.merge(p.drop_duplicates("k"), on="k", suffixes=("_o", "_p"))
            da = float(np.max(np.abs(m.a_o - m.a_p))) if len(m) else np.inf
            dp = float(np.max(np.abs(m.posterior_prob_o - m.posterior_prob_p))) if len(m) else np.inf
            good = len(p) == len(o) == len(m) and da < 1e-9 and dp < 1e-12
            ok_all &= good
            print(f"  {rep:28s} branches {len(m)}/{len(o)}  max|a diff| {da:.1e}  "
                  f"max|posterior diff| {dp:.1e}  {'PASS' if good else 'FAIL'}")
    return ok_all


def check_events_segmented(zones=("kermadectonga2", "kurilsjapan", "izumariana", "ryuku",
                                 "sunda2", "southamerica", "alaskaaleutians")):
    """v9 (LEVEL 0). The rate of every scenario of a SEGMENTED zone: the
    engine on PTHA18's own segmented input (official_segmented_input, as
    step 8 builds it) vs PTHA18's published rate_annual and its LEVEL 5
    percentiles (NCI all_uniform_slip_earthquake_events_<zone>.nc). These
    published rates are 0.5 x unsegmented + 0.5 x every segment the scenario
    touches, with R's partial-segmentation percentiles, so this checks the
    whole segmented path scenario by scenario."""
    require_extraction("events_segmented", zones, need_branches=True)
    import netCDF4
    import pandas as pd
    ok_all = True
    print("\nEVENTS (segmented zones): every scenario's rate vs PTHA18's published rate_annual")
    runner = os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py")
    columns = {"rate_annual": "rate_mean", "rate_annual_lower_ci": "rate_p0.025",
               "rate_annual_16pc": "rate_p0.16", "rate_annual_median": "rate_p0.5",
               "rate_annual_84pc": "rate_p0.84", "rate_annual_upper_ci": "rate_p0.975"}
    for z in zones:
        cfg = official_segmented_input(z)
        cfg["run_name"] = f"{z}_v9_validation_events_segmented"
        cfg["percentiles"]["N"] = 2000          # LEVEL 5 curves are not compared here
        tmp = os.path.join(HERE, f"_input_{z}_v9_validation_events_segmented.json")
        json.dump(cfg, open(tmp, "w"), indent=1)
        subprocess.run([sys.executable, runner, tmp], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        os.remove(tmp)
        p = pd.read_csv(os.path.join(ROOT, "runs", "python", cfg["run_name"],
                                     f"scenario_rates_{z}_source_zone.csv"))
        nc = fetch(f"{THREDDS}/SOURCE_ZONES/{z}/TSUNAMI_EVENTS/all_uniform_slip_earthquake_events_{z}.nc",
                   os.path.join(REF, "events", f"all_uniform_slip_earthquake_events_{z}.nc"))
        with netCDF4.Dataset(nc) as ds:
            off = {k: np.asarray(ds[k][:], float) for k in columns}
            off_mw = np.round(np.asarray(ds["Mw"][:], float), 3)
        n = len(p)
        same_mw = bool(np.allclose(off_mw[:n], p["Mw"].to_numpy(), rtol=0, atol=1e-9))
        worst, zero_mismatch = {}, 0
        for ok_, pk in columns.items():
            a, b = off[ok_][:n], p[pk].to_numpy()
            zero_mismatch += int(np.sum((a == 0) != (b == 0)))
            nz = a > 0
            worst[ok_] = float(np.max(np.abs(b - a)[nz] / a[nz]))
        good = same_mw and zero_mismatch == 0 and max(worst.values()) < 1e-6
        ok_all &= good
        print(f"  {z:16s} {len(cfg['segments'])} segments, events {n}/{off_mw.size}, same Mw rows: {same_mw}  "
              f"max rel diff: mean {worst['rate_annual']:.1e}, percentiles "
              f"{max(v for k, v in worst.items() if k != 'rate_annual'):.1e}, "
              f"zero/non-zero mismatches {zero_mismatch}  {'PASS' if good else 'FAIL'}")
    return ok_all


def check_v8_parity(zone="kermadectonga2"):
    """v9 (LEVEL 0). An UNSEGMENTED v9 run must reproduce v8 exactly.

    v9 adds segmentation without changing how an unsegmented zone is
    computed. Since 2026-09-30 a v9 run does NOT give v8's numbers, because
    step 6 feeds it different Berryman values (coupling from the Whole Margin
    row, mw_max_observed as the largest known earthquake), so this runs v9's
    ENGINE on the v8 folder's own input JSON and compares with the v8 folder's
    outputs, column by column.

    The bar is bit-identity, not a tolerance: on an unsegmented input the
    segment code never executes, so any difference at all would mean v9 had
    changed the unsegmented path.

    Needs the v8 folder to have been run; skipped otherwise.
    """
    import numpy as np
    import pandas as pd

    v8 = os.path.join(ROOT, f"{zone}_v8", "outputs")
    inputs = glob.glob(os.path.join(ROOT, f"{zone}_v8", "inputs", "input_*_scratch.json"))
    print(f"\nV8 PARITY: v9's engine on {zone}_v8's own input vs {zone}_v8's outputs")
    if not (os.path.isdir(v8) and inputs):
        print(f"  skipped: need {zone}_v8/ with its inputs and outputs (run it with v8)")
        return True
    cfg = json.load(open(inputs[0]))
    cfg["run_name"] = f"{zone}_v9_v8parity"
    tmp = os.path.join(HERE, f"_input_{zone}_v9_v8parity.json")
    json.dump(cfg, open(tmp, "w"), indent=1)
    subprocess.run([sys.executable, os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py"),
                    tmp], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    os.remove(tmp)
    v9 = os.path.join(ROOT, "runs", "python", cfg["run_name"])

    files = ["rate_curves.csv", f"logic_tree_branches_{zone}.csv",
             f"scenario_rates_{zone}.csv", "exceedance_rate_percentiles.csv",
             f"conditional_prob_{zone}.csv"]
    ok = True
    for f in files:
        pa, pb = os.path.join(v8, f), os.path.join(v9, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  {f:<46} skipped (missing)")
            continue
        da, db = pd.read_csv(pa), pd.read_csv(pb)
        if "threshold_Mw" in da.columns:
            # example folders run before 2026-09-30 have 6 percentile
            # magnitudes, later ones 27: compare the ones both have
            common = np.intersect1d(da.threshold_Mw.round(6), db.threshold_Mw.round(6))
            da = da[da.threshold_Mw.round(6).isin(common)].reset_index(drop=True)
            db = db[db.threshold_Mw.round(6).isin(common)].reset_index(drop=True)
        if da.shape != db.shape:
            print(f"  {f:<46} FAIL shape {da.shape} vs {db.shape}")
            ok = False
            continue
        num = da.select_dtypes(include=[np.number]).columns
        m = float(np.nanmax(np.abs(da[num].values - db[num].values))) if len(num) else 0.0
        print(f"  {f:<46} rows={len(da):>6}  max diff {m:.3e}"
              f"  {'ok' if m == 0.0 else 'FAIL'}")
        ok = ok and m == 0.0
    print(f"  -> v8 parity {'PASS (bit-identical)' if ok else 'FAIL'}")
    return ok


def check_unseg_view(zones=("kermadectonga2", "kurilsjapan")):
    """v9 (LEVEL 0). A segmented run's unsegmented branch IS an unsegmented run.

    Step 9 builds report_unsegmented.html from a segmented folder's files
    instead of a second run, which is only honest if those files are what an
    unsegmented run of the same zone writes. This compares <zone>_v9seg with
    <zone>_v9 (same steps 1-6 apart from the segments), for this run and for
    step 8's official run: the unsegmented branch's tables, its rate curve and
    its percentile band (exceedance_rate_percentiles_unsegmented.csv, written
    by the engine only on a segmented run) must be bit-identical.

    Skips a zone without both folders.
    """
    import pandas as pd

    print("\nUNSEGMENTED VIEW: <zone>_v9seg's unsegmented branch vs <zone>_v9")
    ok = True
    for zone in zones:
        seg, plain = os.path.join(ROOT, f"{zone}_v9seg"), os.path.join(ROOT, f"{zone}_v9")
        if not (os.path.isdir(seg) and os.path.isdir(plain)):
            print(f"  {zone:<16} skipped: need {zone}_v9/ and {zone}_v9seg/")
            continue
        for out in ("outputs", "outputs_official"):
            pairs = [(f, f) for f in (f"logic_tree_branches_{zone}.csv",
                                      f"conditional_prob_{zone}.csv",
                                      f"integrated_slip_{zone}.csv",
                                      f"scenario_rates_{zone}.csv")]
            pairs.append(("exceedance_rate_percentiles_unsegmented.csv",
                          "exceedance_rate_percentiles.csv"))
            bad = []
            for fs, fp in pairs:
                a, b = os.path.join(seg, out, fs), os.path.join(plain, out, fp)
                if not (os.path.exists(a) and os.path.exists(b)):
                    bad.append(f"{fs} missing")
                elif open(a, "rb").read() != open(b, "rb").read():
                    bad.append(fs)
            ra = pd.read_csv(os.path.join(seg, out, "rate_curves.csv"))
            rb = pd.read_csv(os.path.join(plain, out, "rate_curves.csv"))
            if not ra[["Mw", "exceedance_rate_unsegmented"]].equals(rb):
                bad.append("rate_curves.csv (unsegmented column)")
            print(f"  {zone:<16} {out:<17} {len(pairs) + 1} tables  "
                  + ("ok (bit-identical)" if not bad else "FAIL: " + ", ".join(bad)))
            ok = ok and not bad
    print(f"  -> unsegmented view {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*",
                    default=["mesh", "engine", "events", "gcmt",
                             "segmentation", "berryman_segments", "official_segments",
                             "events_segmented",
                             "v8parity", "unseg_view"])
    ap.add_argument("--ndk", default=None, help="folder with GCMT .ndk files (for the gcmt check)")
    ap.add_argument("--zones", default=None, help="comma-separated zones for the mesh check")
    a = ap.parse_args()
    results = {}
    if "mesh" in a.checks:
        results["mesh"] = check_mesh(a.zones.split(",") if a.zones else MESH_ZONES)
    if "engine" in a.checks:
        results["engine"] = check_engine()
    if "events" in a.checks:
        results["events"] = check_events()
    if "gcmt" in a.checks:
        if a.ndk:
            results["gcmt"] = check_gcmt(a.ndk)
        else:
            print("\nGCMT: skipped (pass --ndk <folder with .ndk files>, e.g. an example's data/gcmt)")
    if "segmentation" in a.checks:
        results["segmentation"] = check_segmentation()
    if "berryman_segments" in a.checks:
        results["berryman_segments"] = check_berryman_segments()
    if "official_segments" in a.checks:
        results["official_segments"] = check_official_segments()
    if "events_segmented" in a.checks:
        results["events_segmented"] = check_events_segmented()
    if "v8parity" in a.checks:
        results["v8parity"] = check_v8_parity()
    if "unseg_view" in a.checks:
        results["unseg_view"] = check_unseg_view()
    print("\nSUMMARY: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
