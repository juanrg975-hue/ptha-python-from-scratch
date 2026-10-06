"""STEP 1 - Build Kermadec interface contours from the SLAB depth GRID.

This is LEVEL 0: before any seismicity, any b-value or any rate, we need the
shape of the fault. Everything downstream (total area, mean dip, the Mw_max
bounds) is measured off this surface.

What this script produces (v8)
------------------------------
Depth contours of the plate interface, in PTHA18's own convention: "depth
below the nearby trench", one contour every CONTOUR_SPACING_KM from the
trench itself (0 km) down to the seismogenic cutoff. That is exactly the
shape of PTHA18's own contour files (DATA/SOURCEZONE_CONTOURS on NCI): a
0 km line that is the trench, then one unbroken line every 5 km.

How (see from_scratch_v12/lib/slab_contours.py for the details)
---------------------------------------------------------------
  1. finds (or downloads) the region's SLAB2.0 / SLAB1.0 depth raster;
  2. the seismogenic cutoff from Berryman et al. (2015), as PTHA18 did;
  3. finds the TRENCH: the part of the raster's outer edge that is shallow
     and faces up dip (the slab gets deeper going into it);
  4. measures every raster cell's depth below its NEAREST trench point,
     which is PTHA18's "depth below the nearby trench" (a local datum, not
     one number for the whole zone);
  5. contours that field at 5, 10, ..., cutoff km; the 0 km contour is the
     trench line;
  6. cleans the zone's two ends (v8): where the SLAB raster (or --clip)
     ends the zone crookedly, so that the mesh's end cells would be more
     than 45 deg off square, or where the zone narrows to a point, every
     contour is cut along one down-dip line (a line of steepest descent,
     perpendicular to every contour), which is what PTHA18 did by hand;
     an end that is already clean is kept as it is;
  7. writes the contours (with a 'level' column, km below trench), a
     figure showing the trench, the contours and any end cut, and a JSON
     summary.

Why not v7's method
-------------------
v3-v7 converted "below trench" to "below sea level" with ONE constant
offset per zone (Berryman's trench depth). But the real trench depth varies
along strike (kermadectonga2: 5-11 km below sea level; southamerica 2-10 km),
so a shallow level asked SLAB for a depth that does not exist wherever the
local trench is deeper: the shallow contours came out broken into pieces.
v4-v7 then had to raise the shallowest level until it was continuous and
cut the number of levels until a width check passed -- which threw away the
shallow band PTHA18 models (calabria2 started 38 km below the trench, with
one row of 12 unit sources) -- and some zones still needed a hand-made clip.
With the local datum every level exists all along the arc by construction.

The antimeridian
----------------
Some zones straddle longitude 180 (Kermadec-Tonga runs from about -175 deg in
the north to +171 in the south). The discretisation code takes plain
differences of longitude with no wrap-around handling, so a contour jumping
from +180 to -180 reads as a 355-degree physical gap and produces a mangled
grid. Recasting every longitude to [0, 360) removes the jump; for a zone that
never goes negative this is a no-op.

If the automatic download fails (blocked by the host, not this script)
---------------------------------------------------------------------
Both downloads in this script are plain, unauthenticated HTTP GETs -- no
account or API key is needed, ever. But ScienceBase (the SLAB2.0 host) sits
behind Cloudflare, which sometimes blocks scripted requests outright with a
403 whose body is a "Just a moment..." challenge page, not a real error from
ScienceBase itself; earthquake.usgs.gov (the SLAB1.0 host) can likewise
return a 403 from its own web-application firewall. Either block is
server-side and this script cannot solve it by retrying, changing its
User-Agent, or waiting a few seconds -- it can persist for hours. The error
message this script raises when either download fails already includes
the same link and folder given below. When it happens, get the file by hand
in a normal browser (which passes the challenge automatically) instead:

  Where to put the file: this script checks THIS zone's own
  examples/kermadectonga2/data/slab2/ (or .../slab1/) FIRST -- generate.py creates both
  folders, and the automatic download saves there too -- and falls back to
  the SHARED from_scratch_v12/data/slab2/ (or .../slab1/) only if nothing is
  there. Nothing writes the shared folder automatically: put a .grd there
  by hand to reuse it in every folder of the same region.

  SLAB2.0 (zone names ending in '2', e.g. calabria2, kermadectonga2):
    This region's item:         https://www.sciencebase.gov/catalog/item/5aa318e1e4b0b1c392ea3f10
    Generic search, any region: https://www.sciencebase.gov/catalog/items?q=Slab2+Kermadec
    Download the item archive (the "Download all files" zip, tens of MB)
    and extract the single "<region>_slab2_dep_*.grd" member into:
        examples/kermadectonga2/data/slab2/          <- this zone's own folder, checked first
    (the exact filename does not matter; this script finds it by the
    "<region>_slab2_dep_" prefix). Re-run this script afterwards.

  SLAB1.0 (zone names without a '2', e.g. cascadia, kurilsjapan):
    Generic search, any region: https://earthquake.usgs.gov/data/slab/models.php
    Direct per-region URL:      https://earthquake.usgs.gov/static/lfs/data/slab/models/ker_slab1.0_clip.grd
    Save the downloaded file into:
        examples/kermadectonga2/data/slab1/ker_slab1.0_clip.grd   <- this zone's own folder, checked first
    (this exact filename IS required for SLAB1.0 -- see
    download_depth_grid_slab1() in official_geometry_params.py).

Run:  .venv/Scripts/python.exe examples/kermadectonga2/steps/step1_fetch_slab2.py
"""

import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))  # helper modules
from official_geometry_params import (find_depth_grid_for_zone,  # noqa: E402
                                      download_depth_grid_slab1, read_grid,
                                      slab_product_for_zone,
                                      ZONE_TO_SLAB2_REGION)
from berryman_params import berryman_cutoff_km, BERRYMAN_ROWS  # noqa: E402
import slab_contours as sc  # noqa: E402

PTHA18_ZONE_NAME = "kermadectonga2"
SLAB2_PREFIX = "ker"
SCIBASE_ITEM = "5aa318e1e4b0b1c392ea3f10"

RAW_DIR = os.path.join(EXAMPLE, "data", "slab2")
GEOMETRY_DIR = os.path.join(EXAMPLE, "inputs", "geometry")
FIGURES_DIR = os.path.join(EXAMPLE, "figures")
OUT_SHP = os.path.join(GEOMETRY_DIR, "kermadectonga2_slab2_contours.shp")
OUT_INFO = os.path.join(EXAMPLE, "data", "step1_contours_info.json")

# Contour spacing in km below the trench. PTHA18 used 5 km (DATA/
# SOURCEZONE_CONTOURS: 0, 5, 10, ... down to the cutoff on almost every
# zone). The number of down-dip ROWS does not come from here: step 2 sets it
# from the mesh's own down-dip length and the unit-source width, as rptha
# does. These contours only describe the surface.
CONTOUR_SPACING_KM = 5.0

# Optional lon/lat window (lon_min, lon_max, lat_min, lat_max) applied to the
# raster before anything else, for a SLAB region that covers more than the
# source zone you want (e.g. only the Lesser Antilles part of "Caribbean").
# None = the whole raster. Set with generate.py --clip or edit here.
CLIP_BBOX = None

# v12: generate.py --mesh-file: the geometry is an external quadrilateral
# mesh (inputs/geometry/<file>, one line per unit source with its 4 corners
# as lon, lat, depth), not SLAB. None: SLAB, as before.
MESH_FILE = None
MESH_DEPTH_UNITS = 'auto'

# Optional ((lon, lat), (lon, lat)): the zone's two ends, for a zone that is
# one part of a SLAB region (v9: Berryman et al. 2015 Table 3.1's segment end
# points for hellenic_west2 / hellenic_east2 / hellenic2). The trench, datum
# and contours are built on the whole raster (or CLIP_BBOX), then every
# contour is cut along the down-dip line through each point. None = the zone
# ends where the raster ends (and the automatic end cleaning applies). Set
# with generate.py --trench-ends or edit here.
TRENCH_ENDS = None

# v10_q ramp rule: an end of the trench that climbs in depth like a ramp, more
# than this many km deeper than the trench's normal depth, is not trench but
# the edge of the SLAB data, and is trimmed (at most 100 km) before the local
# datum is built (slab_contours.find_trench). Without it, hellenic's
# Kefalonia end (19.7 km below sea level against 11.5 km on the rest of the
# trench) became the datum of half the zone and piled its 25-35 km contours
# into one line. None = v9's trench. Set with generate.py --trench-ramp.
TRENCH_RAMP_RISE_KM = 3.0

# Where step 3 writes a plate-boundary-change finding, if it detects one.
# --auto-clip (below) turns that finding into a CLIP_BBOX without you having
# to read any coordinates off a map.
PLATE_CHANGE_JSON = os.path.join(EXAMPLE, "data", "plate_boundary_change.json")

ITEM = "https://www.sciencebase.gov/catalog/item"


def auto_clip_from_step3():
    """The clip window implied by step 3's plate-boundary finding, or None.

    This is the --auto-clip path: step 3 detects that the last (or first)
    few along-strike columns of the mesh match Bird segments belonging to a
    DIFFERENT plate pair -- i.e. the mesh has run past the end of its own
    subduction zone onto the next one -- and records which columns and the
    lon/lat box of the columns that remain. Here that box simply becomes
    CLIP_BBOX, so step 1 re-contours only the part of the raster that
    belongs to this zone.

    Requires step 3 to have run at least once on the UNCLIPPED mesh: the
    finding is made from the mesh, so the normal order is a full run, read
    the report, then re-run with --auto-clip. Returns (bbox, finding) or
    (None, reason).
    """
    if not os.path.exists(PLATE_CHANGE_JSON):
        return None, (
            "--auto-clip needs step 3's plate-boundary finding, and "
            f"{os.path.relpath(PLATE_CHANGE_JSON, ROOT)} is not there. Either "
            "step 3 has not run yet on this mesh (run the pipeline once "
            "first, then re-run with --auto-clip), or it ran and found "
            "nothing to cut -- in which case there is nothing for "
            "--auto-clip to do.")
    with open(PLATE_CHANGE_JSON) as f:
        found = json.load(f)
    if not found.get("auto_clip_separable"):
        return None, (
            "step 3 did flag a plate-boundary change here, but a lon/lat box "
            "cannot separate the flagged columns from the rest of this zone "
            "(its arc doubles back on itself, so the tail sits inside the "
            "bounding box of the part you keep). Cut it with an explicit "
            "--clip LON_MIN,LON_MAX,LAT_MIN,LAT_MAX instead; the flagged "
            f"columns are {found.get('columns')}.")
    bbox = found.get("auto_clip_bbox")
    if not bbox:
        return None, ("step 3's finding has no clip window recorded; re-run "
                      "step 3 to refresh it.")
    return tuple(bbox), found


def _cloudflare_exit():
    raise SystemExit(
        "ScienceBase is behind Cloudflare's bot check right now (HTTP 403, "
        "\"Just a moment...\" challenge page) -- this is not a ScienceBase "
        "error and will not clear by retrying, changing the User-Agent, or "
        "waiting a few seconds; it can persist for hours. Get the file by "
        "hand in a normal browser instead (it passes the challenge "
        "automatically):\n\n"
        f"  This region's item:         https://www.sciencebase.gov/catalog/item/{SCIBASE_ITEM}\n"
        "  Generic search, any region: https://www.sciencebase.gov/catalog/items?q=Slab2+Kermadec\n\n"
        "  Download the item archive (the \"Download all files\" zip, tens "
        "of MB) and extract the single \"<region>_slab2_dep_*.grd\" member "
        "into:\n"
        f"      {RAW_DIR}\n\n"
        "  (the exact filename does not matter; this script finds it by "
        "the \"<region>_slab2_dep_\" prefix). Re-run this script "
        "afterwards. See this script's module docstring (\"If the "
        "automatic download fails\") for the SLAB1.0 case too.") from None


def download_depth_grid(region):
    """Fetch the region's SLAB2.0 depth raster into this example's cache.

    The ScienceBase item id is known from generate.py (SCIBASE_ITEM), so the
    whole-item zip is downloaded directly and the .grd extracted (the
    per-file URLs 404 for these items). Some published .grd members are
    zero-length upstream, in which case the .xyz twin holds identical values
    and the raster is rebuilt from it.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    zpath = os.path.join(RAW_DIR, f"_{region}_sciencebase_item.zip")
    if not os.path.exists(zpath) or os.path.getsize(zpath) == 0:
        zip_url = f"{ITEM.replace('/item', '/file/get')}/{SCIBASE_ITEM}"
        print(f"  downloading ScienceBase item {SCIBASE_ITEM} (tens of MB) ...")
        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                blob = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 403 and b"Just a moment" in e.read():
                _cloudflare_exit()
            raise
        with open(zpath, "wb") as fh:
            fh.write(blob)
        print(f"  {len(blob):,} bytes -> {zpath}")

    with zipfile.ZipFile(zpath) as zf:
        prefix = f"{region}_slab2_dep_"
        cands = [n for n in zf.namelist()
                 if os.path.basename(n).startswith(prefix) and n.endswith(".grd")]
        if len(cands) != 1:
            raise SystemExit(f"expected one {prefix}*.grd, got {cands}")
        name = os.path.basename(cands[0])
        dest = os.path.join(RAW_DIR, name)
        if zf.getinfo(cands[0]).file_size == 0:
            xyz = cands[0][:-4] + ".xyz"
            print(f"  {name} is empty upstream - rebuilding from "
                  f"{os.path.basename(xyz)}")
            _grd_from_xyz(zf, xyz, dest)
        else:
            with zf.open(cands[0]) as src, open(dest, "wb") as fh:
                fh.write(src.read())
        print(f"  extracted {name} ({os.path.getsize(dest) / 1e3:,.0f} KB)")
    return dest


def _grd_from_xyz(zf, member, dest):
    """Write a NetCDF grid equivalent to a SLAB2.0 .grd, from its .xyz twin."""
    import netCDF4

    raw = zf.read(member).decode("utf-8", "replace").strip().splitlines()
    rows = []
    for line in raw:
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    arr = np.array(rows, dtype=float)
    xs = np.unique(arr[:, 0])
    ys = np.unique(arr[:, 1])
    z = np.full((ys.size, xs.size), np.nan)
    xi = np.searchsorted(xs, arr[:, 0])
    yi = np.searchsorted(ys, arr[:, 1])
    z[yi, xi] = arr[:, 2]
    with netCDF4.Dataset(dest, "w", format="NETCDF4") as ds:
        ds.createDimension("x", xs.size)
        ds.createDimension("y", ys.size)
        ds.createVariable("x", "f8", ("x",))[:] = xs
        ds.createVariable("y", "f8", ("y",))[:] = ys
        ds.createVariable("z", "f8", ("y", "x"))[:] = z


def unwrap_antimeridian(coords):
    """Shift negative longitudes by +360 so the arc is continuous."""
    return [(lon + 360.0 if lon < 0 else lon, lat) for lon, lat in coords]


def plot_trench_and_contours(x, y, depth, contours, info, out_png):
    """Map: the raster, its edge coloured by trench score, the trench, the contours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = info["diag"]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.pcolormesh(x, y, np.where(depth < 120, depth, np.nan), cmap="Greys",
                  shading="auto", alpha=0.55)
    loop = d["loop"]
    edge = ax.scatter(loop[:, 0], loop[:, 1], c=d["score"], cmap="coolwarm_r",
                      vmin=-1, vmax=1, s=4)
    plt.colorbar(edge, ax=ax, shrink=0.6,
                 label="raster edge: +1 = slab gets deeper going in (up-dip edge)")
    ends = info.get("ends") or {}
    if any(ends.get(e, {}).get("cut") for e in ("start", "end")):
        for lv, c in info["contours_untrimmed"]:
            ax.plot(c[:, 0], c[:, 1], ":", color="0.45", lw=0.8)
        ax.plot([], [], ":", color="0.45", lw=0.8, label="removed by the end cuts")
    for lv, c in contours[1:]:
        ax.plot(c[:, 0], c[:, 1], "-", color="#1f3b73", lw=0.8)
    tr = contours[0][1]
    ax.plot(tr[:, 0], tr[:, 1], "-", color="#c2185b", lw=2.5, label="trench (0 km)")
    ax.plot([], [], "-", color="#1f3b73", lw=0.8,
            label=f"every {CONTOUR_SPACING_KM:g} km below the trench")
    first = True
    for e in ("start", "end"):
        if ends.get(e, {}).get("cut"):
            cl = ends[e]["cut_line"]
            ax.plot(cl[:, 0], cl[:, 1], "-", color="#e65100", lw=2.5,
                    label="end cut (down-dip line)" if first else None)
            first = False
    if TRENCH_ENDS is not None:
        for k, (plon, plat) in enumerate(TRENCH_ENDS):
            ax.plot(plon, plat, "*", color="#6a1b9a", ms=14, mec="white",
                    label="given zone end points (TRENCH_ENDS)" if k == 0 else None)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(loop[:, 1]))))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("kermadectonga2: trench found on the SLAB raster edge and "
                 "depth-below-trench contours", fontsize=10)
    fig.savefig(out_png, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="STEP 1 - interface geometry")
    ap.add_argument("--auto-clip", action="store_true",
                    help="cut the along-strike columns that step 3 flagged as "
                         "belonging to a DIFFERENT plate boundary, using the "
                         "window step 3 derived from the mesh itself -- no "
                         "lon/lat needed. Run the pipeline once first so step "
                         "3 can make the finding.")
    args = ap.parse_args(argv)

    global CLIP_BBOX

    print("=" * 70)
    print("STEP 1 - Kermadec interface geometry")
    print("=" * 70)

    if MESH_FILE is not None:
        # v12: the geometry is the external mesh step 2 reads; SLAB, the
        # trench and the contours are not used
        print(f"\n  v12: geometry from the external mesh inputs/geometry/{MESH_FILE} "
              f"(generate.py --mesh-file), not from SLAB: nothing to fetch here. "
              f"Step 2 reads the mesh.")
        os.makedirs(os.path.join(EXAMPLE, "data"), exist_ok=True)
        with open(os.path.join(EXAMPLE, "data", "step1_contours_info.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"mesh_file": MESH_FILE}, fh, indent=1)
        print("\nNext:  step2_build_grid.py")
        return 0

    if args.auto_clip:
        bbox, info = auto_clip_from_step3()
        if bbox is None:
            raise SystemExit(f"\n--auto-clip: {info}")
        tail = "first" if info["end"] == "start" else "last"
        print(f"\n  --auto-clip: step 3 flagged this mesh's {tail} "
              f"{info['n_suspect']} of {info['n_total']} along-strike")
        print(f"  columns (numbers {info['columns'][0]}-{info['columns'][-1]}) "
              f"as sitting on a different plate")
        print(f"  boundary: the convergent component steps by "
              f"{info['jump_mm_per_yr']:.1f} mm/yr across column "
              f"{info['boundary_column']}")
        print(f"  ({info['div_inside']:.1f} mm/yr on this zone's side, "
              f"{info['div_outside']:.1f} on the far side).")
        print(f"  Clipping the raster to lon {bbox[0]:.3f}..{bbox[1]:.3f}, "
              f"lat {bbox[2]:.3f}..{bbox[3]:.3f}")
        print(f"  and rebuilding the contours from that window alone.")
        if CLIP_BBOX is not None:
            print(f"  (this REPLACES the CLIP_BBOX already set here, "
                  f"{CLIP_BBOX})")
        CLIP_BBOX = bbox
        print("\n  Re-run steps 2 and 3 after this so the mesh and the "
              "convergence")
        print("  are rebuilt on the clipped contours (step_total.py does "
              "that for you).")

    region = ZONE_TO_SLAB2_REGION.get(PTHA18_ZONE_NAME, SLAB2_PREFIX)
    product = slab_product_for_zone(PTHA18_ZONE_NAME)
    print(f"  SLAB product: {product}  (PTHA18's own naming rule: a zone "
          f"name ending in '2' used SLAB2.0, this one is '{PTHA18_ZONE_NAME}'"
          f" -- see ReportPTHA.pdf p.11-12)")

    # --- 1. the raster ---
    grd, _ = find_depth_grid_for_zone(PTHA18_ZONE_NAME, region,
                                      example_dir=EXAMPLE)
    if grd is None:
        print(f"  no cached {product} depth raster for region '{region}' "
              f"- fetching")
        if product == "SLAB1.0":
            grd = download_depth_grid_slab1(region, example_dir=EXAMPLE)
        else:
            grd = download_depth_grid(region)
    else:
        print(f"  using depth raster: {os.path.relpath(grd, ROOT)}")
    x, y, z = read_grid(grd)
    depth = -z  # km below sea level, positive down
    print(f"  grid {x.size} x {y.size} at {abs(x[1] - x[0]):.2f} deg, "
          f"depth {np.nanmin(depth):.1f} to {np.nanmax(depth):.1f} km below MSL")
    if CLIP_BBOX is not None:
        print(f"  CLIP_BBOX = {CLIP_BBOX} (lon_min, lon_max, lat_min, lat_max): "
              f"only this window of the raster is used")
    if TRENCH_ENDS is not None:
        print(f"  TRENCH_ENDS = {TRENCH_ENDS}: the zone is cut along the "
              f"down-dip lines through these two points")

    # --- 2. the cutoff, derived from Berryman et al. (2015) ---
    # ReportPTHA.pdf p.11: the dominant segment's down-dip depth MAX minus
    # its trench depth, rounded up to a multiple of 5 km -- see
    # berryman_params.py's module docstring for the citation and the exact
    # match against every zone with an official cutoff.
    cutoff, dominant_segment = berryman_cutoff_km(PTHA18_ZONE_NAME)
    if cutoff is None:
        raise SystemExit(
            f"'{PTHA18_ZONE_NAME}' has no Berryman et al. (2015) segment "
            f"mapping, so there is no seismogenic cutoff for it. Add one in "
            f"from_scratch_v12/lib/berryman_params.py (ZONE_TO_BERRYMAN_SEGMENTS "
            f"and ZONE_DOMINANT_SEGMENT) and re-run.")
    berry_trench = BERRYMAN_ROWS[dominant_segment].get("trench_depth_km")
    print(f"\n  seismogenic cutoff: {cutoff:g} km below the trench")
    print(f"    (Berryman et al. 2015, dominant segment '{dominant_segment}': "
          f"down-dip depth max minus trench depth, rounded up to 5 km)")

    # --- 3-5. trench, local datum, contours ---
    print(f"\n  contours every {CONTOUR_SPACING_KM:g} km below the NEARBY "
          f"trench (PTHA18's convention):")
    contours, info = sc.below_trench_contours(
        x, y, depth, cutoff, spacing_km=CONTOUR_SPACING_KM, bbox=CLIP_BBOX,
        trench_ends=TRENCH_ENDS, ramp_rise_km=TRENCH_RAMP_RISE_KM)
    if berry_trench is not None:
        print(f"    for reference, Berryman's trench depth for "
              f"'{dominant_segment}' is {berry_trench:g} km (used only for "
              f"the cutoff above; the contours use the measured local trench)")
    problems = sc.check_contours(contours, info)
    if problems:
        raise SystemExit("step 1 cannot build a valid contour set:\n  - "
                         + "\n  - ".join(problems)
                         + "\n  See the figure in figures/ and the module "
                           "docstring of from_scratch_v12/lib/slab_contours.py.")

    # --- 6. write ---
    import geopandas as gpd
    from shapely.geometry import LineString

    rows = [{"level": round(float(lv), 3),
             "geometry": LineString(unwrap_antimeridian(c))}
            for lv, c in contours]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    os.makedirs(GEOMETRY_DIR, exist_ok=True)
    gdf.to_file(OUT_SHP)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_png = os.path.join(FIGURES_DIR, "step1_trench_and_contours.png")
    plot_trench_and_contours(x, y, depth, contours, info, fig_png)

    summary = {
        "zone": PTHA18_ZONE_NAME,
        "raster": os.path.relpath(grd, ROOT),
        "clip_bbox": CLIP_BBOX,
        "trench_ends": TRENCH_ENDS,
        "trench_ramp_rise_km": TRENCH_RAMP_RISE_KM,
        "ramp_trim_km": info["ramp_trim_km"],
        "cutoff_km_below_trench": cutoff,
        "berryman_dominant_segment": dominant_segment,
        "berryman_trench_depth_km": berry_trench,
        "contour_spacing_km": CONTOUR_SPACING_KM,
        "trench_length_km": info["trench_length_km"],
        "trench_length_untrimmed_km": info["trench_length_untrimmed_km"],
        "trench_depth_below_msl_km": info["trench_depth_msl"],
        "levels": info["levels"],
        "ends": ({e: {k: v for k, v in info["ends"][e].items() if k != "cut_line"}
                  for e in ("start", "end")}
                 if info["ends"] and "start" in info["ends"] else info["ends"]),
    }
    with open(OUT_INFO, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    print(f"\n  wrote {len(gdf)} contours, 0-{cutoff:g} km below the trench")
    print(f"  -> {OUT_SHP}")
    print(f"  -> {fig_png}")
    print(f"  -> {OUT_INFO}")
    print(f"\n  extent: lon {lon_min:.2f} to {lon_max:.2f} (unwrapped, so "
          f"values above 180 are west of the antimeridian)")
    print(f"          lat {lat_min:.2f} to {lat_max:.2f}")
    print("\nNext:  step2_build_grid.py")


if __name__ == "__main__":
    sys.exit(main())
