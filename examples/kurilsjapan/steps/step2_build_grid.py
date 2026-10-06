"""STEP 2 - Discretise the interface into unit sources.

Step 1 gave us depth contours: a set of lines on the sphere. That is a surface,
but not yet a model. This step cuts that surface into a grid of rectangular
patches ("unit sources"), which is the object every later level is built on.

What comes out of here, and why it matters
------------------------------------------
Two numbers dominate everything downstream:

  * TOTAL AREA -- the left-hand side of the seismic moment balance is
    (area x slip_rate x rigidity). Area scales the whole rate curve almost
    linearly: get it 10% wrong and the rates are ~10% wrong.

  * MEAN DIP -- the convergence recovered in step 3 is divided by cos(dip) to
    turn a horizontal plate velocity into slip along an inclined fault. A dip
    error propagates straight into the slip rate.

Comparing against the official values
--------------------------------------
This generated copy does NOT hardcode the official area/mean dip. They follow
from PTHA18's published unit-source table (unit_source_statistics_<zone>.nc on
NCI), which step 8 runs through the engine. Run step 8 (step8_official.py)
first if you want that comparison; this script will then read
runs/python/kurilsjapan_official/logic_tree_summary.csv and print the gap.
Without step 8, it just prints this run's own numbers.

On the mean dip
---------------
It is an ANGLE-weighted mean (via the mean of the unit vectors), not an
arithmetic one, matching the convention PTHA18 itself uses for its published
mean_dip figures.

How the grid size is chosen, and the one thing that cannot be reproduced
------------------------------------------------------------------------
rptha picks the number of along-strike columns from the length of the shallowest
contour divided by the desired unit-source length, and the number of down-dip
rows from the mean transect length.

v1 passed 50 km for BOTH length and width on every zone. The width is now read
per zone from sourcezone_parameters.csv, because it is not always 50:
puysegur2 uses 35, and the published mesh measures 35.82 km mean width there.

The length target is a different matter. PTHA18's own documentation says
approx_unit_source_length is IGNORED, because the along-strike spacing comes
from a hand-edited down-dip-lines shapefile (SOURCE_ZONES/<zone>/EQ_SOURCE/
unit_source_grid/<zone>.shp on NCI THREDDS -- published, but this script does
not read it as a geometry source; see from_scratch_v12/html/docs/official.html). What is not published
anywhere is the EDITING PROCESS itself: which kinks were smoothed, which
seams were moved and why. So this script takes the automatic branch of
rptha's discretiser, which PTHA18 did not take. The cell boundaries will
therefore differ from the official ones no matter what, and that is not
fixable without re-deriving the hand-editing PTHA18 applied.

What IS comparable, and what step 2 now checks, is total area and mean dip --
the only two mesh quantities the moment balance actually consumes.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/kurilsjapan/steps/step2_build_grid.py
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))
# v8 uses its OWN forked copy of pyptha (from_scratch_v12/pyptha_v12), not the
# shared one other versions read, so earlier versions are unaffected.
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))  # helper modules

from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import grid_cache  # noqa: E402
from pyptha_v12 import unit_sources as us  # noqa: E402

IN_SHP = os.path.join(EXAMPLE, "inputs", "geometry",
                      "kurilsjapan_slab2_contours.shp")
OUT_NPY = os.path.join(EXAMPLE, "data", "slab2", "unit_source_grid.npy")

# v12: generate.py --mesh-file: the geometry is an external quadrilateral
# mesh (inputs/geometry/<file>, one line per unit source with its 4 corners
# as lon, lat, depth), not SLAB. None: SLAB, as before.
MESH_FILE = None
MESH_DEPTH_UNITS = 'auto'

from official_geometry_params import (width_km, official_area_and_dip,  # noqa: E402
                                      n_downdip_rows)
from plot_meshes import fig_scratch_mesh_map, scratch_cell_polygons  # noqa: E402

FIGURES_DIR = os.path.join(EXAMPLE, "figures")

PTHA18_ZONE_NAME = "kurilsjapan"

# --ptha flag from generate.py (see its module docstring). True (default):
# may fix the down-dip row count to PTHA18's own published value below,
# same as v2 did. False: always lets rptha's own width-based rule infer the
# row count, even when the official table is downloaded.
USE_PTHA = False

# --discretizer flag from generate.py (see its module docstring).
# "optimal" (default): rptha's own mesh (the "lm" one below); only if that
# mesh has folded, collapsed or overlapping cells, a sequence of repairs is
# tried and the first defect-free result is used (see
# contour_discretisation.discretized_source_from_contours_optimal).
# "lm": rptha's method exactly as PTHA18's own template
# (make_initial_downdip_lines.R) calls it, no repairs.
# "mid": rptha's OLDER, deprecated method; PTHA18 does not use it.
DISCRETIZER = 'optimal'

# --columns flag from generate.py (v8.2; used by DISCRETIZER="optimal" only).
# How many columns (unit sources along strike) the mesh gets:
# "trench" (default): rptha's and PTHA18's rule, trench length /
#   DESIRED_LENGTH_KM, rounded up. The mesh PTHA18's method gives.
# "average": the AVERAGE row length / DESIRED_LENGTH_KM. Not PTHA18's
#   procedure: for a zone whose mesh tapers far more than PTHA18's (the
#   "taper" line printed below; PTHA18's 43 meshes: at most 1.17), it puts
#   the ~50 km cells in the middle rows instead of at the trench (calabria2:
#   20 -> 12 columns, median cell length 26 -> 45 km). See
#   contour_discretisation.discretized_source_from_contours_optimal.
COLUMN_RULE = 'trench'

# PTHA18's target unit-source size. Length is 50 km everywhere (though PTHA18
# ignores it -- see the docstring).
# Width: with USE_PTHA=True (default), per-zone and READ from
# sourcezone_parameters.csv (a resolution parameter, not a "declared"
# physical value, but still a PTHA18-internal file). With USE_PTHA=False,
# 50 km fixed for every zone instead -- the "target unit-source size" the
# PTHA18 report itself documents as the general default (Strasser et al.
# 2010 scaling relations are calibrated around that scale), not read from
# any PTHA18-internal file. This will not match puysegur2's real 35 km,
# and that is the point: --ptha=false does not know puysegur2 is narrow
# without reading PTHA18's own table for it.
DESIRED_LENGTH_KM = 50.0
DESIRED_WIDTH_KM = width_km(PTHA18_ZONE_NAME) if USE_PTHA else 50.0
SEED = 1234

# PTHA18's minimum of 2 down-dip rows. ReportPTHA.pdf p.14: "The only
# exception is the Puysegur trench where a narrower down-dip width (35 km)
# was used to ensure we had 2 rows of unit-sources in the down-dip
# direction." With --ptha false every zone gets 50 km, so a steep, narrow
# zone would come out with a single row; this applies PTHA18's own stated
# requirement instead (2 equal rows, which is what 35 km gave Puysegur).
# The mesh step prints a line whenever it applies. None disables it.
MIN_DOWNDIP_ROWS = 2

# Manual override for the down-dip row count. None (default) leaves rptha's
# own width-based rule in charge (mean 3D transect length / DESIRED_WIDTH_KM,
# see discretized_source_from_contours_orthogonal's docstring) -- unchanged
# behaviour, kept because it is the faithful port.
#
# UNITS: this is n_downdip, the number of down-dip CELLS (unit sources), the
# same parameter discretized_source_from_contours_orthogonal/_optimal take.
# Internally rows = n_downdip + 1 NODES are built, then
# grid.shape[0] - 1 = n_downdip cells come back out -- so N_DOWNDIP_OVERRIDE=4
# gives 4 down-dip cells (5 nodes), matching what the "grid: ... down dip"
# print line and the official-mesh comparison both report in CELLS.
#
# Set an integer here only if you want a different row count than the
# width-based rule gives. The rule rounds (round(mean/width - 1)), so a mean
# transect length close to a multiple of DESIRED_WIDTH_KM can land either
# side of a boundary between two zones that look alike; one row more or less
# moves total area by roughly 1/n_downdip. (v5-v7 also warned that the
# optimiser itself could land either side run to run; v8's optimiser is
# deterministic, so the same inputs always give the same count.)
N_DOWNDIP_OVERRIDE = None

OFFICIAL_SUMMARY_CSV = os.path.join(
    ROOT, "runs", "python", f"{PTHA18_ZONE_NAME}_official", "logic_tree_summary.csv")


def load_contours():
    """Read step 1's shapefile back as (depth_km, coords) pairs."""
    import geopandas as gpd

    if not os.path.exists(IN_SHP):
        raise SystemExit(
            f"missing {IN_SHP}\nRun step1_fetch_slab2.py first.")

    gdf = gpd.read_file(IN_SHP)
    gdf["level"] = gdf["level"].astype(float)

    contours = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "LineString":
            coords = np.array(geom.coords)
        else:
            # A depth level split into several arcs: take the longest, which
            # is the main trace. The short ones are detached slivers.
            parts = sorted(geom.geoms, key=lambda g: len(g.coords))
            coords = np.array(parts[-1].coords)
        contours.append((row["level"], coords[:, :2]))

    depths = sorted(c[0] for c in contours)
    print(f"  {len(contours)} contours at depths {depths} km")
    return contours


def angle_weighted_mean_dip(dip_deg):
    """Mean of the dip directions as unit vectors, in degrees.

    This is the convention behind PTHA18's own published mean_dip figures.
    """
    r = np.radians(dip_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))


def read_official_area_and_dip():
    """If step 8 has already run, read its area/mean-dip for comparison.

    Returns (area_km2, mean_dip_deg) or (None, None) if step 8 has not run
    yet -- that is a normal state, not an error; steps 1-7 and 9 do not need
    step 8 at all.
    """
    if not os.path.exists(OFFICIAL_SUMMARY_CSV):
        return None, None
    area = dip = None
    with open(OFFICIAL_SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            item = row["item"]
            if item.startswith("mean_dip_deg_"):
                dip = float(row["value"])
            # area is not written directly to logic_tree_summary.csv; it is
            # derivable from a_min/a_max only indirectly, so we do not try to
            # recover it here. mean dip is the one number both this script
            # and step 8's run write in a directly comparable form.
    return area, dip


def main_mesh_file():
    """v12: the unit sources are the external mesh's cells, used as given."""
    from pyptha_v12 import mesh_file
    path = os.path.join(EXAMPLE, "inputs", "geometry", MESH_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\n(generate.py --mesh-file copies it here)")
    print(f"\n  v12: unit sources from the external mesh inputs/geometry/{MESH_FILE}")
    grid = mesh_file.read_quadrilateral_mesh(path, depth_units=MESH_DEPTH_UNITS)
    n_downdip, n_alongstrike = grid.shape[0] - 1, grid.shape[2] - 1
    defects = cd.mesh_defects(grid)
    print(f"  mesh check: {defects}")
    stats = us.discretized_source_approximate_summary_statistics(grid)
    area = float(np.sum(stats["length"] * stats["width"]))
    print(f"  area {area:,.0f} km2, mean dip {angle_weighted_mean_dip(stats['dip']):.2f} deg, "
          f"cells {np.min(stats['length']):.1f}-{np.max(stats['length']):.1f} km along strike x "
          f"{np.min(stats['width']):.1f}-{np.max(stats['width']):.1f} km down dip, "
          f"trench depth {grid[0, 2, :].min():.2f}-{grid[0, 2, :].max():.2f} km")
    import hashlib
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    grid_cache.save(OUT_NPY, grid, {"mesh_file": MESH_FILE, "mesh_file_sha256": sha,
                                    "mesh_depth_units": MESH_DEPTH_UNITS})
    print(f"\n  saved grid -> {OUT_NPY}")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    scratch_polys, scratch_depth = scratch_cell_polygons(grid)
    mesh_fig = fig_scratch_mesh_map(
        scratch_polys, scratch_depth, PTHA18_ZONE_NAME, n_downdip,
        n_alongstrike, FIGURES_DIR, out_name="scratch_mesh_map.png")
    print(f"  saved mesh figure -> {mesh_fig}")
    print("\nNext:  step3_convergence.py")
    return 0


def main():
    print("=" * 70)
    print("STEP 2 - Discretise the interface into unit sources")
    print("=" * 70)
    if MESH_FILE is not None:
        return main_mesh_file()

    contours = load_contours()

    print(f"\n  discretising at {DESIRED_LENGTH_KM:g} km along strike, "
          f"{DESIRED_WIDTH_KM:g} km down dip ...")
    # NOTE on --ptha (see generate.py): with USE_PTHA=True (default), this
    # fixes the down-dip row count to PTHA18's own published value rather
    # than letting rptha's own width-based rule infer one, same as v2 did.
    # The rule that infers it (mean 3D transect length / desired width,
    # rptha-style -- see contour_discretisation.py's
    # discretized_source_from_contours_orthogonal) previously produced a
    # degenerate mesh (30 rows against an official 3), but that failure came
    # from v1's 20-km-spaced SLAB2.0 shapefile contours feeding it a wildly
    # wrong transect length, not from the rule itself -- with step 1's
    # PTHA18-style contours (one unbroken line every 5 km below the nearby
    # trench, v8) the same rule reproduces PTHA18's row counts.
    # With USE_PTHA=False this NEVER reads n_downdip_rows, so a zone with
    # no PTHA18 mesh at all can still run -- n_downdip=None below hands
    # control back to that rule.
    official_rows = n_downdip_rows(PTHA18_ZONE_NAME) if USE_PTHA else None
    if official_rows is not None:
        print(f"  fixing down-dip rows to {official_rows} (the official mesh's "
              f"row count, --ptha=true)")
    elif N_DOWNDIP_OVERRIDE is not None:
        official_rows = N_DOWNDIP_OVERRIDE
        print(f"  fixing down-dip rows to {official_rows} "
              f"(N_DOWNDIP_OVERRIDE, set by hand in this script -- rptha's "
              f"own width-based rule is otherwise left untouched)")
    else:
        print(f"  letting rptha's width-based rule infer the down-dip row "
              f"count"
              + ("" if USE_PTHA else " (--ptha=false: never reads the "
                                      "official row count, even if "
                                      "downloaded)"))

    if DISCRETIZER == "optimal":
        print(f"  --discretizer=optimal: rptha's own mesh, repaired only if it "
              f"has folded, collapsed or overlapping cells (see "
              f"contour_discretisation.discretized_source_from_contours_optimal)")
        grid = cd.discretized_source_from_contours_optimal(
            contours,
            desired_unit_source_length=DESIRED_LENGTH_KM,
            desired_unit_source_width=DESIRED_WIDTH_KM,
            n_downdip=official_rows,
            seed=SEED, min_downdip=MIN_DOWNDIP_ROWS,
            column_rule=COLUMN_RULE)
    elif DISCRETIZER == "lm":
        grid = cd.discretized_source_from_contours_orthogonal(
            contours,
            desired_unit_source_length=DESIRED_LENGTH_KM,
            desired_unit_source_width=DESIRED_WIDTH_KM,
            n_downdip=official_rows,
            seed=SEED, min_downdip=MIN_DOWNDIP_ROWS)
    else:
        print(f"  --discretizer=mid: using rptha's older, deprecated "
              f"nearest/equal-spacing compromise method (no orthogonality "
              f"optimisation) -- PTHA18's own template does not use this one")
        grid = cd.discretized_source_from_contours_mid(
            contours,
            desired_unit_source_length=DESIRED_LENGTH_KM,
            desired_unit_source_width=DESIRED_WIDTH_KM,
            n_downdip=official_rows, min_downdip=MIN_DOWNDIP_ROWS)

    n_downdip = grid.shape[0] - 1
    n_alongstrike = grid.shape[2] - 1
    print(f"  grid: {n_alongstrike} along strike x {n_downdip} down dip "
          f"= {n_alongstrike * n_downdip} unit sources")
    defects = cd.mesh_defects(grid)
    print(f"  mesh check: {defects['bowties']} folded (bow-tied) cells, "
          f"{defects['degenerate']} collapsed cells, "
          f"{100 * defects['overlap_fraction']:.2f}% overlapping area"
          + ("" if defects["total"] == 0 else "   <-- inspect the mesh figure"))
    # v8.2: how close the cells are to DESIRED_LENGTH_KM x DESIRED_WIDTH_KM,
    # next to the range of PTHA18's own 43 published meshes. Prints only.
    shape = cd.mesh_shape(grid, DESIRED_LENGTH_KM)
    fan = max(shape['taper'], 1 / shape['taper']) > cd.PTHA18_MAX_TAPER
    wide = shape['width_ratio'] > cd.PTHA18_MAX_WIDTH_RATIO
    flag = lambda bad: "   <-- outside PTHA18's range" if bad else ""
    print("  mesh shape (PTHA18's 43 meshes in brackets):")
    print(f"    taper (trench-row / deepest-row cell length) {shape['taper']:5.2f}  "
          f"(at most {cd.PTHA18_MAX_TAPER:g})" + flag(fan))
    print(f"    width ratio (widest / narrowest column)     {shape['width_ratio']:5.2f}  "
          f"(at most {cd.PTHA18_MAX_WIDTH_RATIO:g})" + flag(wide))
    print(f"    cells with both sides 30-70 km              {shape['near_pct']:5.1f}%  "
          f"(85% or more on 38 of 43)")
    print(f"    mean deviation from 50 x 50 (0 = perfect)    {shape['mean_deviation']:5.2f}  "
          f"(median {cd.PTHA18_MEDIAN_DEVIATION:g})")
    if fan and DISCRETIZER == "optimal" and COLUMN_RULE == "trench":
        print("    note: this mesh tapers more than any PTHA18 mesh; generate.py "
              "--columns average (or COLUMN_RULE = 'average' above) counts the "
              "columns from the average row instead of the trench")
    if wide:
        print("    note: the zone's down-dip width changes more along strike than in "
              "any PTHA18 zone; every column must keep the same rows, so no column "
              "or row count can make these cells 50 x 50")

    stats = us.discretized_source_approximate_summary_statistics(grid)

    area = float(np.sum(stats["length"] * stats["width"]))
    dip_ang = angle_weighted_mean_dip(stats["dip"])
    dip_arith = float(np.mean(stats["dip"]))

    print("\n  --- per-unit-source statistics ---")
    print(f"  length  {stats['length'].min():6.1f} - {stats['length'].max():6.1f} km")
    print(f"  width   {stats['width'].min():6.1f} - {stats['width'].max():6.1f} km")
    print(f"  depth   {stats['depth'].min():6.1f} - {stats['depth'].max():6.1f} km")
    print(f"  dip     {stats['dip'].min():6.1f} - {stats['dip'].max():6.1f} deg")

    print("\n  --- the two numbers that drive everything ---")
    print(f"  total area   {area:12,.1f} km^2")
    print(f"  mean dip     {dip_ang:12.4f} deg   "
          f"(arithmetic mean {dip_arith:.4f} deg, shown for contrast; the "
          f"official convention is the angle-weighted one above)")

    # --- compare against the published mesh, on the two quantities that
    # --- actually reach the moment balance. Only with --ptha true: with
    # --ptha false no PTHA18 file is read here (the report shows the
    # comparison from step 8's outputs).
    off = official_area_and_dip(PTHA18_ZONE_NAME) if USE_PTHA else None
    if off is not None:
        d_area = 100.0 * (area / off["area_km2"] - 1.0)
        d_dip = dip_ang - off["mean_dip_deg"]
        off_rows = n_downdip_rows(PTHA18_ZONE_NAME)
        print("\n  --- against the published PTHA18 mesh ---")
        print(f"  {'':16s} {'this run':>14s} {'official':>14s} {'gap':>12s}")
        print(f"  {'total area km2':16s} {area:14,.1f} "
              f"{off['area_km2']:14,.1f} {d_area:+11.2f}%")
        print(f"  {'mean dip deg':16s} {dip_ang:14.4f} "
              f"{off['mean_dip_deg']:14.4f} {d_dip:+11.4f}")
        print(f"  {'unit sources':16s} {n_alongstrike * n_downdip:14d} "
              f"{off['n_unit_sources']:14d}")
        print(f"  {'down-dip rows':16s} {n_downdip:14d} {off_rows:14d}")
        print(f"  {'depth range km':16s} "
              f"{stats['depth'].min():6.1f}-{stats['depth'].max():<7.1f} "
              f"{off['depth_min_km']:6.1f}-{off['depth_max_km']:<7.1f}")
        print("\n  Area is what matters: it scales the rate curve almost")
        print("  linearly. Mean dip enters only through 1/cos(dip), which is")
        print("  nearly flat, so a few tenths of a degree is negligible.")
    elif USE_PTHA:
        _, official_dip = read_official_area_and_dip()
        if official_dip is not None:
            print(f"\n  official mean dip (from step 8's run): "
                 f"{official_dip:.4f} deg ({dip_ang - official_dip:+.4f})")
        else:
            print(f"\n  (no published table for {PTHA18_ZONE_NAME}; download it "
                 f"with official_ptha_data/fetch_official_inputs.py for a "
                 f"comparison)")

    # Saved with a fingerprint (contour file hash + every meshing parameter)
    # so step 7 and step 9 reuse THIS mesh instead of rebuilding it; they
    # rebuild only if the input JSON asks for different parameters.
    fp = grid_cache.fingerprint(
        IN_SHP, method=DISCRETIZER, desired_unit_source_length=DESIRED_LENGTH_KM,
        desired_unit_source_width=DESIRED_WIDTH_KM, n_downdip=official_rows,
        min_downdip=MIN_DOWNDIP_ROWS, seed=SEED,
        # v8.2: only a non-rptha column rule enters the fingerprint, so a
        # "trench" mesh keeps exactly the v8 fingerprint.
        **({} if (DISCRETIZER != "optimal" or COLUMN_RULE == "trench")
           else {"column_rule": COLUMN_RULE}))
    grid_cache.save(OUT_NPY, grid, fp)
    print(f"\n  saved grid -> {OUT_NPY} (+ fingerprint {os.path.basename(grid_cache.meta_path(OUT_NPY))})")

    # Draw the mesh itself, not just its summary statistics. Needs nothing
    # but this run's own grid -- no official/PTHA18 data of any kind -- so
    # it always renders, on any zone, with or without --ptha. See
    # plot_meshes.py's module docstring.
    #
    # PRELIMINARY ONLY: this is a quick look at the mesh THIS SCRIPT built
    # from DESIRED_WIDTH_KM above. If you hand-edit desired_unit_source_width
    # (or anything else) in inputs/kurilsjapan_scratch.json AFTER running step 6,
    # this figure goes stale -- it does not know about that edit. step 9
    # rebuilds the mesh from that JSON directly (via run_logic_tree.py's own
    # build_grid()) and overwrites this exact file with the mesh step 7
    # actually used. Treat this one as "what step 2 would compute before any
    # manual edits", not the final picture.
    os.makedirs(FIGURES_DIR, exist_ok=True)
    scratch_polys, scratch_depth = scratch_cell_polygons(grid)
    mesh_fig = fig_scratch_mesh_map(
        scratch_polys, scratch_depth, PTHA18_ZONE_NAME, n_downdip,
        n_alongstrike, FIGURES_DIR, out_name="scratch_mesh_map.png")
    print(f"  saved mesh figure -> {mesh_fig}  (preliminary -- step 9 "
          f"rebuilds this from the final input JSON)")

    print("\n  How to read the gaps above:")
    print("  the CELL BOUNDARIES will not match: PTHA18 meshed its own hand-edited")
    print("  contours (published on NCI, DATA/SOURCEZONE_CONTOURS.zip; fed those,")
    print("  this discretiser reproduces PTHA18's mesh), while this run builds")
    print("  its contours from SLAB. Area and mean dip CAN match, and they are")
    print("  the only two mesh quantities the moment balance consumes. Judge")
    print("  this mesh on those, not on cells.")
    print("\nNext:  step3_convergence.py")


if __name__ == "__main__":
    sys.exit(main())
