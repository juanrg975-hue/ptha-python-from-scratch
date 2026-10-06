"""STEP 3 - Tectonic convergence rate from Bird (2003).

LEVEL 1 input. The moment balance in step 7 needs a plate convergence rate in
mm/yr: it is the physical driver of how fast the fault accumulates the moment
that earthquakes later release.

Where the data comes from
--------------------------
Bird, P. (2003), "An updated digital model of plate boundaries", G-cubed
4(3). Each boundary step is one short segment (Long1,Lat1)-(Long2,Lat2) with
a right-lateral component (RL_vel) and a divergent component (Div_vel,
negative = convergent). Which table is read is generate.py's --convergence
(BIRD_TABLE below; this folder was generated with "'bird'"):

  bird           (default) Bird's public catalogue as published, every step
                 of a convergent boundary type (subduction, oceanic or
                 continental convergent):
                     from_scratch_v12/data/bird/PB2002_steps.dat.txt.zip
  bird-griffin   The table PTHA18 used: Bird's steps it kept plus Jonathan
                 Griffin's source-zone traces with their own plate rates, with
                 a class column ('Normal', 'Thrust', 'Megathrust' or blank):
                     from_scratch_v12/data/bird/bird_griffin_traces_table.csv.zip
                 (a copy of rptha's sourcezone_traces_table_merged.csv.zip).

Both are in from_scratch_v12/data/bird/, so nothing is downloaded and nothing
is read from rptha. On the zones where PTHA18 used Griffin's traces the two
give very different convergences (newguinea2: 38 vs 96 mm/yr); which zones
those are: from_scratch_v12/html/docs/convergence_sources.html.

PTHA18's own method (lib/bird_convergence.py)
---------------------------------------------
Ported from rptha/R/examples/austptha_template/EVENT_RATES/
make_spatially_variable_source_zone_convergence_rates.R and
compute_rates_all_sources.R lines 371-415:

1. Each TOP-EDGE unit source (downdip_number == 1) is moved from its
   centroid to its top edge: destPoint(centroid, strike - 90,
   width/2 * cos(dip)).
2. The nearest Bird segment to that point is found by great-circle
   (Haversine) distance to the segment's great-circle midpoint, as
   geosphere computes both.
3. A thrust source ignores class=='Normal' rows (outer-rise bending).
4. div_vec = max(0, -Div_vel) (only convergence counts), and the lateral
   component is capped at div_vec * tan(50 deg) (config.R's rake_deviation);
   the convergent slip is sqrt(div_vec^2 + rl_vec^2).
5. Every unit source takes its column's top-edge match.

The zone's convergence is the AREA-WEIGHTED mean of that convergent slip.
Run on PTHA18's own meshes this reproduces the official convergence to
machine precision on all 7 Bird zones tested (cascadia, makran2, philippine,
kermadectonga2, kurilsjapan, southamerica, izumariana).

v8 changes
----------
* data/convergence.txt now holds the area-weighted mean itself, i.e. the
  HORIZONTAL plate rate, which is what the engine's
  tectonic_convergence_mm_per_yr means. The engine divides by cos(mean dip)
  once, as R does (compute_rates_all_sources.R line 432). v3-v7 wrote the
  value already divided by cos(mean dip), so the dip correction was applied
  twice: +0.6% at makran2's 6 deg mean dip, +4% at 16 deg (kermadectonga2,
  southamerica, kurilsjapan), +24% at puysegur2's 36 deg.
* data/convergence_per_column.json (new) keeps the per-column values. Step 6
  passes them to the engine, which then uses PTHA18's Bird model: event
  probabilities weighted by the convergent slip under each rupture, and the
  convergent component as the LEVEL 4 target shape. With the flat target and
  1/slip weights of v3-v7, per-event rates differed from PTHA18's published
  ones by a median 4-26% on the Bird zones; with this model they match to
  about 1e-6.
* Bird segment centroids are great-circle midpoints (geosphere midPoint with
  f = 0) and distances are Haversine, as in R, instead of the arithmetic
  lon/lat mean and the WGS84 geodesic distance.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/alaskaaleutians_slab/steps/step3_convergence.py
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))  # helper modules

from pyptha_v12 import unit_sources as us  # noqa: E402
import bird_convergence as bc  # noqa: E402
from berryman_params import zone_plate_pairs  # noqa: E402

GRID_NPY = os.path.join(EXAMPLE, "data", "slab2", "unit_source_grid.npy")
OUT_TXT = os.path.join(EXAMPLE, "data", "convergence.txt")
OUT_PERCELL_NPY = os.path.join(EXAMPLE, "data", "convergence_per_cell.npy")
OUT_PERCOLUMN_JSON = os.path.join(EXAMPLE, "data", "convergence_per_column.json")
# The plate-boundary-change finding (v8.1), written whenever one is detected
# and deleted when there is none, so step 1 --auto-clip and step 9 always see
# the CURRENT mesh's verdict rather than a stale one.
OUT_PLATE_CHANGE_JSON = os.path.join(EXAMPLE, "data", "plate_boundary_change.json")

PTHA18_ZONE_NAME = "alaskaaleutians"
# --ptha (see generate.py): True only adds a comparison line with step 8's
# convergence to this log. It does not choose the Bird table (BIRD_TABLE does).
USE_PTHA = False
# generate.py --convergence: "bird" (Bird's public catalogue, default) or
# "bird-griffin" (PTHA18's Bird + Griffin table). Edit and re-run steps 3,
# 6 (--force), 7 and 9 to switch in this folder.
BIRD_TABLE = 'bird'
OFFICIAL_SUMMARY_CSV = os.path.join(
    ROOT, "runs", "python", f"{PTHA18_ZONE_NAME}_official", "logic_tree_summary.csv")

# Manual override for the convergence (mm/yr, horizontal, before the
# 1/cos(dip) correction the engine applies). None (default): use the Bird
# (2003) match this script computes below. A number here switches the run to
# PTHA18's CONSTANT-convergence model (use_bird_convergence = 0, what PTHA18
# did for puysegur with 35 mm/yr): that rate everywhere, 1/slip event
# weights, and a flat LEVEL 4 target. Bird's per-cell matching is still
# computed and saved for the report's map.
CONVERGENCE_OVERRIDE_MM_PER_YR = None


def _wrap(text, width=66):
    """Wrap one message line to the log's width."""
    import textwrap
    return textwrap.wrap(text, width) or [""]


def read_official_convergence():
    """If step 8 has already run, read its convergence for comparison."""
    if not os.path.exists(OFFICIAL_SUMMARY_CSV):
        return None
    with open(OFFICIAL_SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["item"].startswith("convergence_mm_per_yr_"):
                return float(row["value"])
    return None


def main():
    print("=" * 70)
    print("STEP 3 - Tectonic convergence from Bird (2003)")
    print("=" * 70)

    if not os.path.exists(GRID_NPY):
        raise SystemExit(f"missing {GRID_NPY}\nRun step2_build_grid.py first.")
    grid = np.load(GRID_NPY)
    stats = us.discretized_source_approximate_summary_statistics(grid)

    print("\n  matching each TOP-EDGE unit source to its nearest Bird segment "
          "(great-circle distance) ...")
    print("  table: " + ("Bird (2003) + Griffin's traces, PTHA18's table (--convergence bird-griffin)"
                         if BIRD_TABLE == "bird-griffin" else
                         "Bird (2003)'s public catalogue (--convergence bird)"))
    # v9: a zone defined by Berryman's segments is matched only to Bird steps
    # between its own plates (None for every other zone: plain nearest step)
    plate_pairs = zone_plate_pairs("alaskaaleutians") if BIRD_TABLE == "bird" else None
    if plate_pairs:
        print(f"  only Bird steps between this zone's plates (Berryman et al. 2015: "
              f"{', '.join(plate_pairs)})")
    col = bc.column_convergence(stats, table=BIRD_TABLE, plate_pairs=plate_pairs)
    slip = col["per_unit_source_convergent_slip"]
    convergence = col["area_weighted_mean_mm_per_yr"]

    r = np.radians(stats["dip"])
    mean_dip = float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))))

    print(f"\n  per-column convergent slip:  {slip.min():.1f} - {slip.max():.1f} mm/yr")
    print(f"  top edge to Bird segment:    {col['distance_km'].max():.1f} km at most")
    print(f"  area-weighted convergence:   {convergence:.4f} mm/yr (horizontal)")
    print(f"  mean dip:                    {mean_dip:.4f} deg; the engine applies "
          f"1/cos(dip) = {1.0 / np.cos(np.radians(mean_dip)):.4f} once")

    # --- does this mesh run past its own plate boundary? (v8.1) ---
    # The Bird match never fails: a mesh that overshoots its zone silently
    # reads the NEXT boundary's convergence at a perfectly normal match
    # distance. See lib/bird_convergence.py's detector section.
    plate_change = bc.detect_plate_boundary_change(col)
    if plate_change is not None:
        grid_for_clip = grid
        bbox = bc.auto_clip_bbox(plate_change, grid_for_clip)
        separable = bc.auto_clip_is_separable(plate_change, grid_for_clip, bbox)
        plate_change["auto_clip_bbox"] = list(bbox) if bbox else None
        plate_change["auto_clip_separable"] = bool(separable)
        print("\n  " + "!" * 66)
        for line in bc.plate_boundary_change_message(plate_change,
                                                     PTHA18_ZONE_NAME):
            for chunk in _wrap(line):
                print(f"  {chunk}")
            print()
        if not separable:
            print("  NOTE: a lon/lat box cannot separate this tail from the "
                  "rest of")
            print("  the zone (the arc doubles back), so --auto-clip will "
                  "refuse and")
            print("  ask for an explicit --clip instead.")
            print()
        print("  " + "!" * 66)
        with open(OUT_PLATE_CHANGE_JSON, "w") as f:
            json.dump(plate_change, f, indent=1)
        print(f"\n  saved the finding -> {OUT_PLATE_CHANGE_JSON}")
    else:
        # No finding now: drop any finding from an earlier mesh.
        if os.path.exists(OUT_PLATE_CHANGE_JSON):
            os.remove(OUT_PLATE_CHANGE_JSON)
        print("\n  plate-boundary check: every along-strike column matches "
              "the same")
        print("  plate pair -- no sign this mesh runs past its own zone.")

    use_bird = CONVERGENCE_OVERRIDE_MM_PER_YR is None
    if not use_bird:
        print(f"\n  CONVERGENCE_OVERRIDE_MM_PER_YR is set: {convergence:.4f} mm/yr "
              f"(Bird) replaced by {CONVERGENCE_OVERRIDE_MM_PER_YR:g} mm/yr, "
              f"with PTHA18's constant-convergence model (see its comment)")
        convergence = float(CONVERGENCE_OVERRIDE_MM_PER_YR)

    # With --ptha false nothing of PTHA18 is read here; the comparison with
    # PTHA18's convergence is in the report, from step 8's outputs.
    if USE_PTHA:
        official = read_official_convergence()
        if official is not None:
            print(f"\n  official (from step 8's run): {official:.4f} mm/yr")
            print(f"  difference: {100.0 * (convergence - official) / official:+.1f}%")
            print("  (same method and same units; a gap comes from the mesh, e.g.")
            print("  its along-strike extent, or from a zone where PTHA18 used a")
            print("  constant rate instead of Bird: puysegur, 35 mm/yr)")
        else:
            print("\n  (run step8_official.py first for a comparison against "
                  "PTHA18's own convergence)")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w") as f:
        f.write(f"{convergence:.10f}\n")
    print(f"\n  saved -> {OUT_TXT}")

    # Per-cell convergent slip (mm/yr), in the unit-source order of
    # discretized_source_approximate_summary_statistics(grid). Step 9 maps it.
    np.save(OUT_PERCELL_NPY, slip)
    print(f"  saved per-cell convergence -> {OUT_PERCELL_NPY}")

    with open(OUT_PERCOLUMN_JSON, "w") as f:
        json.dump({
            "_what_this_is": (
                "Bird (2003) convergence per along-strike column of step 2's "
                "mesh (lib/bird_convergence.py). Step 6 passes the two "
                "profiles to the engine when use_bird_convergence is true."),
            "bird_table": BIRD_TABLE,
            "use_bird_convergence": use_bird,
            "alongstrike_number": [int(v) for v in col["alongstrike_number"]],
            "div_mm_per_yr": [float(v) for v in col["div_mm_per_yr"]],
            "convergent_slip_mm_per_yr": [float(v) for v in col["convergent_slip_mm_per_yr"]],
            "distance_km": [float(v) for v in col["distance_km"]],
            "area_weighted_mean_mm_per_yr": col["area_weighted_mean_mm_per_yr"],
        }, f, indent=1)
    print(f"  saved per-column convergence -> {OUT_PERCOLUMN_JSON}")
    print("\nNext:  step4_fetch_gcmt.py")


if __name__ == "__main__":
    sys.exit(main())
