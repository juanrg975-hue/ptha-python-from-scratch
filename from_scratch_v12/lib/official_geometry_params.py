"""Per-zone geometry parameters recovered from PTHA18's own published files.

This module exists because from_scratch v1 guessed two numbers that turn out
to be readable from data already in the package. Both guesses biased the mesh
in ways that dominated every "SLAB2.0 vs PTHA18" difference the v1 reports
attributed to geometry.

What was wrong in v1, and what replaces it
------------------------------------------
  MAX_DEPTH_KM = 100      a documented placeholder, applied to every zone.
                          The real per-zone cutoff is 30-55 km. For
                          kermadectonga2 it is 40, so v1 meshed 2.5x too deep.
                          Depth cutoff sets the total fault AREA, and area
                          scales the whole rate curve almost linearly, so this
                          was the single largest error in v1.

  DESIRED_WIDTH_KM = 50   applied to every zone. sourcezone_parameters.csv
                          gives puysegur2 a width of 35, not 50, and the
                          published mesh measures 35.82 km mean width there.

Where the cutoff comes from
---------------------------
The published unit_source_statistics_<zone>.nc carries a `max_depth` variable,
computed (see pyptha/unit_sources.py) as the maximum corner depth of each
unit-source cell. In the DEEPEST down-dip row that is by construction the
bottom edge of the mesh, i.e. the zone's seismogenic depth cutoff -- and it is
constant across that row.

Two independent checks confirm the reading:

  1. Reconstructing the bottom edge geometrically from other variables,
     centroid_depth + width*sin(dip)/2, reproduces it to better than 0.3%.
  2. Every zone's value is an exact multiple of 5 km, which is what the
     PTHA18 report describes: the Berryman et al. (2015) seismogenic limit
     minus the trench depth below MSL, "rounding up to a multiple of 5 km"
     (Davies & Griffin 2018, p17).

The report also states 55 km below the trench for a deep zone (p54), matching
the 55 recovered here for kurilsjapan and southamerica.

So these are READ, not typed: `cutoff_km()` prefers to read the live .nc file
and only falls back to the table below when that file is absent. The table is
a cache of what the files say, not an independent source of truth; `verify()`
re-derives it and complains if the two ever disagree.

Datum
-----
PTHA18 depths are measured BELOW THE NEARBY TRENCH; SLAB2.0/SLAB1.0 publish
depth below mean sea level. The offset is the local trench depth, which
varies along strike (2-11 km below sea level on the zones tested). Step 1
converts with that LOCAL value (lib/slab_contours.py, v8); this module only
supplies PTHA18-convention numbers and file lookups.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)  # from_scratch_v12/ (this file lives in lib/)
ROOT = os.path.dirname(PKG)
# v11: PTHA18's extracted files (rptha/, inputs/, official_ptha_data/) live in
# ptha18_logic_tree_test/; the package may sit there or one level down (V9/).
DATA_ROOT = next((r for r in (ROOT, os.path.dirname(ROOT))
                  if os.path.isdir(os.path.join(r, "rptha"))
                  or os.path.isdir(os.path.join(r, "official_ptha_data"))), ROOT)

# Cache of what the published .nc files say. Regenerate with verify().
# zone -> (cutoff_km, n_downdip_rows)
PUBLISHED_CUTOFF_KM = {
    "alaskaaleutians": 45.0,
    "arutrough": 30.0,
    "cascadia": 30.0,
    "kermadectonga2": 40.0,
    "kurilsjapan": 55.0,
    "mexico": 50.0,
    "newhebrides": 40.0,
    "puysegur2": 40.0,
    "southamerica": 55.0,
}

SZP = os.path.join(DATA_ROOT, "rptha", "R", "examples", "austptha_template", "DATA",
                   "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")

# PTHA18 source zone -> SLAB2.0 three-letter region code, for the continuous
# depth/dip/strike rasters. The two projects carve the world up differently, so
# this is a judgement rather than a lookup: PTHA18's kurilsjapan spans the Japan
# and Kuril arcs, which SLAB2.0 publishes as one "kur" region. Mirrors
# slab_vs_ptha/compare_geometry.py's ZONE_TO_REGION.
ZONE_TO_SLAB2_REGION = {
    "kurilsjapan": "kur",
    "kermadectonga2": "ker",
    "southamerica": "sam",
    "alaskaaleutians": "alu",
    "puysegur2": "puy",
}

# Same three-letter codes, reused for SLAB1.0: both products carve the world
# up the same way for every zone this package has tested (kur, ker, sam, alu,
# puy all resolve on both earthquake.usgs.gov/static/lfs/data/slab/models/
# and ScienceBase). Kept as a separate table rather than aliased to
# ZONE_TO_SLAB2_REGION in case a future zone's SLAB1.0 and SLAB2.0 region
# codes ever diverge.
ZONE_TO_SLAB1_REGION = dict(ZONE_TO_SLAB2_REGION)


def slab_product_for_zone(zone):
    """Which SLAB product PTHA18's own official mesh was built from.

    PTHA18's report (ReportPTHA.pdf, p.11) says: "we chose to update to
    SLAB2.0 only at source-zones near to Australia where the geometry
    changed substantially from SLAB1.0." Figure 3's caption (p.12) makes the
    rule explicit: "Those with names finishing in '2' used SLAB2.0 -- other
    source-zones either used SLAB1.0 or prescribed linear or parabolic
    profiles." Appendix A's Table 2 (p.131) confirms it in the official
    per-zone table: kermadectonga2, newhebrides2, puysegur2, solomon2,
    sunda2 all carry the "2" suffix; kurilsjapan does not.

    So the rule is exactly the zone name's own suffix, not a lookup table
    that could drift out of sync with it: a PTHA18 zone name ending in '2'
    used SLAB2.0, and one that does not almost certainly used SLAB1.0 (or,
    for zones neither SLAB product covers, a prescribed linear/parabolic
    profile PTHA18 built by hand -- this function cannot detect that third
    case, only distinguish the two SLAB products).
    """
    return "SLAB2.0" if zone.endswith("2") else "SLAB1.0"


def find_depth_grid(region, product="SLAB2.0", example_dir=None):
    """Path to a downloaded depth raster for `region`, or None.

    `product` selects which cache/filename pattern to look for:
    "SLAB2.0" (default, unchanged behaviour) or "SLAB1.0".

    `example_dir`, if given, is this run's own generated zone folder (e.g.
    calabria2_v8/): its data/slab2/ (or data/slab1/) is where step 1's
    automatic download saves the raster, for both products. Checked FIRST,
    so a per-folder file wins over a shared one of the same name.

    Then, for SLAB2.0, slab_vs_ptha/data/slab2/ if that study exists, and
    finally this package's SHARED from_scratch_v12/data/slab2/ (or .../slab1/).
    Nothing writes the shared folder automatically; a .grd placed there by
    hand is reused by every folder of that region.
    """
    import glob

    if product == "SLAB1.0":
        bases = []
        if example_dir:
            bases.append(os.path.join(example_dir, "data", "slab1"))
        bases.append(os.path.join(PKG, "data", "slab1"))
        pattern = f"{region}_slab1.0_clip.grd"
    else:
        bases = []
        if example_dir:
            bases.append(os.path.join(example_dir, "data", "slab2"))
        bases.append(os.path.join(ROOT, "slab_vs_ptha", "data", "slab2"))
        if DATA_ROOT != ROOT:  # v11: the package one level down (V9/)
            bases.append(os.path.join(DATA_ROOT, "slab_vs_ptha", "data", "slab2"))
        bases.append(os.path.join(PKG, "data", "slab2"))
        pattern = f"{region}_slab2_dep_*.grd"

    for base in bases:
        hits = [p for p in sorted(glob.glob(os.path.join(base, pattern)))
                if os.path.getsize(p) > 0]
        if hits:
            return hits[0]
    return None


def find_depth_grid_for_zone(zone, region=None, example_dir=None):
    """find_depth_grid, but picks SLAB1.0 vs SLAB2.0 automatically from the
    zone name (see slab_product_for_zone). Returns (path_or_None, product).

    example_dir: see find_depth_grid's docstring -- pass this run's own
    zone folder so a manually-downloaded .grd placed in ITS data/slab2/ (or
    .../slab1/) is found before falling back to the shared cache.
    """
    region = region or ZONE_TO_SLAB2_REGION.get(zone) or ZONE_TO_SLAB1_REGION.get(zone)
    product = slab_product_for_zone(zone)
    return find_depth_grid(region, product=product,
                           example_dir=example_dir), product


def download_depth_grid_slab1(region, out_dir=None, example_dir=None):
    """Fetch `region`'s SLAB1.0 depth raster.

    Unlike SLAB2.0 (a ScienceBase item you must resolve and unzip), SLAB1.0's
    per-region grids are single files at a fixed URL pattern on USGS's own
    file store -- confirmed live for every region code this package uses
    (ker, puy returns 404 since Puysegur was never in SLAB1.0's original
    coverage and PTHA18's own report explains why that is consistent: it is
    one of the zones "near Australia" updated straight to SLAB2.0; kur, sam,
    alu all resolve).

    A 403 here (unlike SLAB2.0's own Cloudflare 403, which always carries a
    "Just a moment..." challenge body) usually means USGS's own WAF blocking
    the request outright rather than the region being genuinely absent, and
    it can clear or not clear on its own -- so the message always gives the
    manual-download link and the exact folder to save the file into, the
    same way the SLAB2.0 path does, rather than only the "maybe this region
    was never modelled" explanation.

    `example_dir`, if given, is this run's own generated zone folder: the
    download is saved in its data/slab1/, the same as SLAB2.0 saves in
    data/slab2/, so each run folder holds the raster it used.
    find_depth_grid still also looks in the shared from_scratch_v12/data/slab1/,
    for a file dropped there by hand to serve every folder.
    """
    import urllib.request

    shared = os.path.join(PKG, "data", "slab1", f"{region}_slab1.0_clip.grd")
    if out_dir is None:
        out_dir = (os.path.join(example_dir, "data", "slab1") if example_dir
                   else os.path.dirname(shared))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{region}_slab1.0_clip.grd"
    dest = os.path.join(out_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    url = (f"https://earthquake.usgs.gov/static/lfs/data/slab/models/"
           f"{region}_slab1.0_clip.grd")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            blob = resp.read()
    except Exception as e:
        example_line = ""
        if example_dir:
            example_line = (
                f"  This run's own folder (checked first, use this one):\n"
                f"      {dest}\n\n")
        raise SystemExit(
            f"could not download SLAB1.0 grid for region '{region}' from "
            f"{url}: {e}\n\n"
            "Get the file by hand instead (a normal browser download, no "
            "account needed):\n\n"
            f"  Direct URL:      {url}\n"
            "  If that 404s, browse all SLAB1.0 regions here: "
            "https://earthquake.usgs.gov/data/slab/models.php\n\n"
            "  Save the downloaded file, with this exact name "
            f"('{fname}'), into:\n\n"
            f"{example_line}"
            f"  Or the shared cache (reused by every folder of this region):\n"
            f"      {shared}\n\n"
            "  Re-run this step afterwards.\n\n"
            "If the URL genuinely 404s (not a 403 block) for this region, "
            "PTHA18's official mesh here may instead be a prescribed "
            "linear/parabolic profile this package cannot reconstruct (see "
            "ReportPTHA.pdf p.11) -- puysegur is the one zone this package "
            "has confirmed that for.")
    with open(dest, "wb") as fh:
        fh.write(blob)
    return dest


def read_grid(path):
    """Load a SLAB1.0 or SLAB2.0 .grd (plain NetCDF) as (x, y, z), NaN off-slab.

    Both products publish the same three variables (x, y, z), verified
    directly against a downloaded SLAB1.0 grid, so one reader covers both.
    """
    import netCDF4

    with netCDF4.Dataset(path) as f:
        x = np.asarray(f["x"][:], dtype=float)
        y = np.asarray(f["y"][:], dtype=float)
        z = np.ma.filled(np.asarray(f["z"][:], dtype=float), np.nan)
    return x, y, z


def official_nc(zone):
    """Path to the published unit-source table, or None if not downloaded."""
    p = os.path.join(DATA_ROOT, "inputs", "geometry",
                     f"unit_source_statistics_{zone}.nc")
    return p if os.path.exists(p) else None


def cutoff_from_nc(zone):
    """Read the zone's seismogenic cutoff from the published .nc file.

    Returns (cutoff_km, n_rows, consistency_km) where `consistency_km` is the
    largest disagreement between the stored max_depth and the bottom edge
    reconstructed from centroid depth, width and dip. A large value there
    would mean this reading is not what we think it is, so it is returned
    rather than hidden.
    """
    path = official_nc(zone)
    if path is None:
        return None
    import netCDF4

    with netCDF4.Dataset(path) as f:
        md = np.asarray(f["max_depth"][:], dtype=float)
        d = np.asarray(f["depth"][:], dtype=float)
        w = np.asarray(f["width"][:], dtype=float)
        dip = np.asarray(f["dip"][:], dtype=float)
        dd = np.asarray(f["downdip_number"][:], dtype=int)

    last = dd.max()
    m = dd == last
    cutoff = float(np.unique(md[m])[0]) if np.unique(md[m]).size == 1 \
        else float(md[m].max())
    reconstructed = d[m] + w[m] * np.sin(np.radians(dip[m])) / 2.0
    consistency = float(np.abs(reconstructed - md[m]).max())
    return cutoff, int(last), consistency


def cutoff_km(zone, allow_table=True):
    """The zone's seismogenic depth cutoff in km, PTHA18 convention.

    Prefers the published file; falls back to the cached table. Raises if
    neither is available, because guessing is what v1 did.
    """
    got = cutoff_from_nc(zone)
    if got is not None:
        return got[0], "read from the published unit_source_statistics .nc"
    if allow_table and zone in PUBLISHED_CUTOFF_KM:
        return (PUBLISHED_CUTOFF_KM[zone],
                "from the cached table in official_geometry_params.py "
                "(the .nc file is not downloaded)")
    raise SystemExit(
        f"no seismogenic depth cutoff known for '{zone}'.\n"
        f"  Download the official table with:\n"
        f"    .venv/Scripts/python.exe official_ptha_data/"
        f"fetch_official_inputs.py {zone}\n"
        f"  This number sets the fault area and therefore the whole rate "
        f"curve; v2 refuses to invent one.")


def n_downdip_rows(zone):
    """How many down-dip rows the official mesh uses, or None."""
    got = cutoff_from_nc(zone)
    return got[1] if got else None


def width_km(zone):
    """approx_unit_source_width for the zone, from sourcezone_parameters.csv.

    Not a constant: puysegur2 uses 35 where the other interface zones use 50.
    v1 hardcoded 50 everywhere.
    """
    import csv

    with open(SZP, newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v.strip() if v else "") for k, v in raw.items()}
            if row["sourcename"] == zone and not row.get("segment_name"):
                return float(row["approx_unit_source_width"])
    raise SystemExit(
        f"'{zone}' has no unsegmented row in sourcezone_parameters.csv")


def scaling_relation_and_shear_modulus(zone):
    """(scaling_relation, shear_modulus_Pa) for `zone`'s unsegmented row in
    sourcezone_parameters.csv.

    make_all_earthquake_events.R:98-105 reads these PER ZONE
    (config.R:15,19: scaling_relation_type = all_sourcezone_par$scaling_
    relation[source_rows]; shear_modulus = all_sourcezone_par$shear_modulus
    [source_rows]*1e+10 -- the CSV's own value is in units of 1e10 Pa, so it
    is multiplied by 1e10 here to match) and passes them into
    get_all_earthquake_events (mu=, relation=). Not a constant across the
    whole catalogue: every zone this project currently runs uses Strasser/
    3 (->3e10), but outerrisesunda and arutrough use AllenHayes-outer-rise/
    Blaser-normal and 6 (->6e10) -- neither is discretised by this from-
    scratch pipeline today, but a caller that hardcodes Strasser/3e10
    instead of reading this table would silently diverge if one ever were.

    If `zone` has no unsegmented row at all (a zone PTHA18 never modelled),
    falls back to ("Strasser", 3e10) -- not an invented number: it is the
    value every zone this project currently runs already resolves to, and
    Strasser et al. (2010) is the general-purpose scaling relation PTHA18
    itself falls back to absent zone-specific literature (AllenHayes/Blaser
    are used only for two non-interface zones, see above). Prints a note so
    this fallback is never silent.
    """
    import csv

    with open(SZP, newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v.strip() if v else "") for k, v in raw.items()}
            if row["sourcename"] == zone and not row.get("segment_name"):
                return row["scaling_relation"], float(row["shear_modulus"]) * 1e10
    print(f"  note: '{zone}' has no row in sourcezone_parameters.csv -- "
          f"using the generic default scaling_relation='Strasser', "
          f"shear_modulus=3e10 Pa (same as every zone this project "
          f"currently runs)")
    return "Strasser", 3.0e10


def official_area_and_dip(zone):
    """Total area (km2) and angle-weighted mean dip of the published mesh.

    These are the two quantities the moment balance actually consumes, so
    they are what a from-scratch mesh should be judged against -- not the
    cell boundaries, which cannot be reproduced at all.
    """
    path = official_nc(zone)
    if path is None:
        return None
    import netCDF4

    with netCDF4.Dataset(path) as f:
        L = np.asarray(f["length"][:], dtype=float)
        W = np.asarray(f["width"][:], dtype=float)
        dip = np.radians(np.asarray(f["dip"][:], dtype=float))
        d = np.asarray(f["depth"][:], dtype=float)
    return {
        "area_km2": float((L * W).sum()),
        "mean_dip_deg": float(np.degrees(
            np.arctan2(np.sin(dip).mean(), np.cos(dip).mean()))),
        "n_unit_sources": int(L.size),
        "depth_min_km": float(d.min()),
        "depth_max_km": float(d.max()),
    }


def verify():
    """Re-derive every cached cutoff from the .nc files and report."""
    print("Seismogenic depth cutoffs, re-derived from the published files")
    print("=" * 74)
    print(f"{'zone':18s} {'cached':>7s} {'from .nc':>9s} {'rows':>5s} "
          f"{'x5?':>4s} {'edge check':>11s}")
    print("-" * 74)
    bad = []
    for zone in sorted(PUBLISHED_CUTOFF_KM):
        got = cutoff_from_nc(zone)
        cached = PUBLISHED_CUTOFF_KM[zone]
        if got is None:
            print(f"{zone:18s} {cached:7.0f} {'(no file)':>9s}")
            continue
        cutoff, rows, consistency = got
        is5 = "yes" if abs(cutoff / 5 - round(cutoff / 5)) < 1e-9 else "NO"
        flag = "" if abs(cutoff - cached) < 1e-9 else "  <-- MISMATCH"
        if flag:
            bad.append(zone)
        print(f"{zone:18s} {cached:7.0f} {cutoff:9.2f} {rows:5d} {is5:>4s} "
              f"{consistency:10.3f}km{flag}")
    print("-" * 74)
    print("'edge check' = max |stored max_depth - (centroid + W*sin(dip)/2)|")
    if bad:
        raise SystemExit(f"cached table disagrees with the files: {bad}")
    print("all cached values match the published files")


if __name__ == "__main__":
    verify()
