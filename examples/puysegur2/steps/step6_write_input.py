"""STEP 6 - Assemble everything into one run_logic_tree.py input JSON.

Every level's inputs so far have lived in separate files: the geometry
(step 2), the convergence (step 3), the GCMT subset (step 5). This step
collects them into the single JSON that python_logic_tree/run_logic_tree.py
expects, in the same schema the official comparisons use.

What is DERIVED here (from public data, steps 1-5) vs. what is DECLARED
(a literature value this script reads from the actual primary source, not
from any PTHA18-internal file, because no from-scratch exercise can derive
it from geometry or a catalogue alone)
-------------------------------------------------------------------------
  derived   geometry           step 1-2, SLAB2.0/SLAB1.0
  derived   convergence        step 3, Bird (2003), with its per-column
                               profile (PTHA18's Bird event weights and
                               LEVEL 4 target, v8)
  derived   observed_seismicity step 4-5, GCMT

  declared  coupling (cmin/cpref/cmax)   read from berryman_params.py
  declared  b_anchor (bmin/bpref/bmax)   read from berryman_params.py
  derived   mw_max_observed              max(Berryman's largest row, GCMT max)
  declared  n_logic_tree_bins            resolution, not physics
  derived   segments (v9, --segmented true only)  Bird (2003) plate pairs;
                               per segment its GCMT events (step 5), its
                               Berryman row(s) for coupling and Mw_max
                               observed; see build_segments()

v3 vs v2: where the "declared" numbers come from
--------------------------------------------------
v2 read coupling/b_anchor/mw_max_observed from
    rptha/R/examples/austptha_template/DATA/SOURCEZONE_PARAMETERS/
        sourcezone_parameters.csv
PTHA18's OWN internal table -- a fine literature lookup for a zone PTHA18
already covers, but useless for a zone it never modelled (no row exists).

v3 instead reads the same three numbers from the actual paper PTHA18 itself
cites for them: Berryman et al. (2015) "The GEM Faulted Earth Subduction
Interface Characterisation Project, Version 2.0", GNS Science Miscellaneous
Series 80 -- see from_scratch_v12/lib/berryman_params.py for the full citation,
the verified table values, and the derivation rule for the seismogenic
cutoff (which is ALSO now read from Berryman via this module, not from
PTHA18's published unit_source_statistics_*.nc -- see step2_build_grid.py).
This is the general-purpose version: it works for any subduction zone this
package's authors add a Berryman Table 3.1 row for, not just the ones
PTHA18 happened to already model.

The parts of the official run that CANNOT legitimately be looked up this way
-- geometry, convergence, the GCMT subset -- are exactly the parts this
example rebuilds in steps 1-5 instead of reading from any file.

Editing the output JSON by hand
--------------------------------
This step REFUSES to overwrite inputs/puysegur2_scratch.json if it already
exists. Edit any field by hand (desired_unit_source_width, coupling,
mw_max_observed, whatever) and every later step (7, 9, ...) reads that same
file as-is -- nothing re-derives it from Berryman/SLAB/GCMT behind your
back. To discard your edits and regenerate from scratch, either delete the
file yourself or pass --force.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/puysegur2/steps/step6_write_input.py
    .venv/Scripts/python.exe examples/puysegur2/steps/step6_write_input.py --force   # overwrite an edited JSON
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))  # ptha18_logic_tree_test/
V5_DIR = os.path.join(ROOT, "from_scratch_v12")

sys.path.insert(0, V5_DIR)
sys.path.insert(0, os.path.join(V5_DIR, "lib"))  # helper modules
from berryman_params import (berryman_coupling, berryman_b_anchor,  # noqa: E402
                             berryman_mw_max_observed)
from official_geometry_params import (width_km, n_downdip_rows,  # noqa: E402
                                      scaling_relation_and_shear_modulus)
import bird_convergence as bc  # noqa: E402

CONV_TXT = os.path.join(EXAMPLE, "data", "convergence.txt")
CONV_COLUMNS_JSON = os.path.join(EXAMPLE, "data", "convergence_per_column.json")
GCMT_CSV = os.path.join(EXAMPLE, "data", "gcmt", "puysegur2_gcmt_subset.csv")
GRID_NPY = os.path.join(EXAMPLE, "data", "slab2", "unit_source_grid.npy")
SHP = os.path.join(EXAMPLE, "inputs", "geometry", "puysegur2_slab2_contours.shp")

# v12: generate.py --mesh-file: the geometry is an external quadrilateral
# mesh (inputs/geometry/<file>, one line per unit source with its 4 corners
# as lon, lat, depth), not SLAB. None: SLAB, as before.
MESH_FILE = None
MESH_DEPTH_UNITS = 'auto'
OFFICIAL_SUMMARY_CSV = os.path.join(
    ROOT, "runs", "python", "puysegur2_official", "logic_tree_summary.csv")

OUT_JSON_NAME = "input_puysegur2_scratch.json"
OUT_JSON = os.path.join(EXAMPLE, "inputs", OUT_JSON_NAME)

# The engine writes to runs/python/<run_name>/. v3-v7 used "<zone>_scratch",
# shared by every version and every example folder of the same zone, so each
# run overwrote the previous one there; v8 adds this folder's name.
RUN_NAME = "puysegur2_scratch_puysegur2"
PTHA18_ZONE_NAME = "puysegur2"

# --ptha flag from generate.py (see its module docstring).
USE_PTHA = False

# --segmented flag from generate.py (v9, LEVEL 0). False reproduces v8.
SEGMENTED = False
# v9: where the segments end, "berryman" (Table 3.1 end points) or "bird"
SEGMENT_BOUNDARIES = "berryman"

# --discretizer flag from generate.py (see its module docstring and
# step2_build_grid.py.tmpl's). Written into the geometry block below so
# run_logic_tree.py's build_grid() -- what step 7/9 actually mesh rates
# on -- rebuilds the SAME mesh step 2 already built and validated, instead
# of silently defaulting to "lm".
DISCRETIZER = 'optimal'

def _read_step2_constant(name):
    """Read one of step2_build_grid.py's own constants by TEXT, not import.

    This used to be a second copy of the constant, kept in sync by hand --
    which failed exactly as designed to fail: setting it in step 2 alone
    left this script's own copy at None, so the JSON it writes (what step
    7/9 actually mesh rates on) silently disagreed with the mesh step 2
    had already built and validated. One override, read from its one
    source of truth, cannot go out of sync like that.

    Importing step2_build_grid as a module would work (it is in the same
    steps/ directory, already on sys.path) but costs several seconds just
    to read one constant, because step 2's own top-level imports pull in
    matplotlib and the whole pyptha_v12.contour_discretisation stack.
    Reading the source as text and extracting the assignment costs well
    under a millisecond and runs none of step 2's code, so there is no
    risk of a stray side effect either.
    """
    import re

    path = os.path.join(HERE, "step2_build_grid.py")
    src = open(path, encoding="utf-8").read()
    m = re.search(rf"^{name}\s*=\s*(.+?)\s*(#.*)?$", src, re.M)
    if not m:
        raise SystemExit(
            f"could not find {name} in {path} -- has it been "
            f"renamed or removed? step 6 needs to read step 2's own value "
            f"so the two never disagree.")
    return eval(m.group(1), {"__builtins__": {}})


# --ptha=false has no official row count to fall back on, so this reads
# step 2's own manual override (see _read_step2_constant's
# docstring) instead of carrying a second, independently-editable copy of
# the same setting -- the two used to desync exactly because they were two
# copies. None (default) leaves build_grid()'s own width-based rule in
# charge, same as step 2's default.
#
# Written into the geometry block below for the same reason DISCRETIZER is:
# without it, this JSON -- the one run_logic_tree.py's build_grid() (step
# 7/9, i.e. what the report and the real rates are built from) actually
# meshes on -- never carries step 2's override, so the engine silently
# falls back to the automatic row-count rule instead, which can disagree
# with step 2's own figure (see this file's own n_downdip comment below for
# the analogous --ptha=true desync this mirrors).
N_DOWNDIP_OVERRIDE = _read_step2_constant("N_DOWNDIP_OVERRIDE")
# PTHA18's minimum of 2 down-dip rows (ReportPTHA p.14), read from step 2 for
# the same reason: the engine must mesh exactly as step 2 did.
MIN_DOWNDIP_ROWS = _read_step2_constant("MIN_DOWNDIP_ROWS")


def _step2_value(name, default):
    """Step 2's own value of `name` when it is a plain value (for example
    after a hand edit such as DESIRED_WIDTH_KM = 35.0), else `default`, the
    value step 2's generated expression gives (width_km(...) with --ptha
    true cannot be evaluated as text)."""
    try:
        return _read_step2_constant(name)
    except Exception:
        return default


# The other meshing settings, read from step 2 for the same reason, so that
# a hand edit there reaches the engine instead of being silently replaced
# by this script's own copy.
DESIRED_LENGTH_KM = _step2_value("DESIRED_LENGTH_KM", 50.0)
DESIRED_WIDTH_KM = _step2_value(
    "DESIRED_WIDTH_KM", width_km(PTHA18_ZONE_NAME) if USE_PTHA else 50.0)
MESH_SEED = _step2_value("SEED", 1234)
DISCRETIZER = _step2_value("DISCRETIZER", DISCRETIZER)
# v8.2: the column rule step 2 meshed with (see its COLUMN_RULE comment).
# A step 2 without the constant (a v8.1 folder) used rptha's own rule.
COLUMN_RULE = _step2_value("COLUMN_RULE", "trench")

# --- Declared literature values (see module docstring) -----------------
# Rupture scaling relation and shear modulus for the event table. --ptha
# false: Strasser et al. (2010) and 30 GPa, the general-purpose values for a
# subduction interface. --ptha true: PTHA18's own per-zone row of
# sourcezone_parameters.csv (Strasser and 3e10 Pa on every interface zone).
if USE_PTHA:
    SCALING_RELATION, SHEAR_MODULUS_PA = scaling_relation_and_shear_modulus(PTHA18_ZONE_NAME)
else:
    SCALING_RELATION, SHEAR_MODULUS_PA = "Strasser", 3.0e10

# v10: how many cells each uniform-slip rupture gets (generate.py
# --rupture-size). "rptha": rptha's and PTHA18's rule, one block per
# magnitude from the zone's mean cell size. "local": a block per placement
# from the real km of the cells there (pyptha_v12/events.py, _local_blocks),
# for meshes whose cells differ a lot. Not PTHA18's procedure.
RUPTURE_SIZE = 'local'

# PTHA18's own observation window; matches step 4. See that script's module
# docstring for why v2 uses the official end date rather than today.
WINDOW_START = "1976-01-01"
WINDOW_END = "2017-03-01"
THRESHOLD_MW = 7.15

# Magnitudes at which LEVEL 5 computes the epistemic percentiles: every 0.1
# from 7.2 to 9.8, the whole range of the scenarios (and the engine's own
# default). Until 2026-09-30 only 7.2, 7.6, ..., 9.2, which made the report's
# band a few straight segments that stopped at 9.2. Step 8 uses the same list.
PERCENTILE_MW = [round(7.2 + 0.1 * i, 1) for i in range(27)]


def read_sourcezone_row():
    """This zone's coupling (cmin/cpref/cmax), b-value anchor
    (bmin/bpref/bmax) and mw_max_observed, read from Berryman et al. (2015)
    directly via berryman_params.py -- NOT from PTHA18's own
    sourcezone_parameters.csv (that is what v2 does; see this module's
    docstring for why v3 changes this).

    v9 (2026-09-30): coupling is Berryman's Whole Margin row when Table 3.1
    has one for the zone (as PTHA18 does, berryman_params.ZONE_WHOLE_MARGIN),
    otherwise the mean of its segment rows; b is the mean of Berryman's rows
    (unchanged); mw_max_observed is the largest Mmax-min of the zone's rows,
    and main() then takes the larger of that and the zone's own GCMT maximum
    (the largest earthquake known on the zone). They can still differ from
    PTHA18's table, which sometimes hand-adjusts mw_max_observed (its notes
    column, e.g. southamerica's 1960 Chile treated as Mw 9.2).
    """
    coupling = berryman_coupling(PTHA18_ZONE_NAME)
    b_anchor = berryman_b_anchor(PTHA18_ZONE_NAME)
    mw_max_observed = berryman_mw_max_observed(PTHA18_ZONE_NAME)
    return coupling, b_anchor, mw_max_observed


def read_convergence():
    if not os.path.exists(CONV_TXT):
        raise SystemExit(f"missing {CONV_TXT}\nRun step3_convergence.py first.")
    with open(CONV_TXT) as f:
        return float(f.read().strip())


def read_convergence_columns():
    """Step 3's per-column Bird profile (v8). A folder whose step 3 ran
    before v8 has no such file: re-run step 3."""
    if not os.path.exists(CONV_COLUMNS_JSON):
        raise SystemExit(f"missing {CONV_COLUMNS_JSON}\n"
                         f"Run step3_convergence.py (v8) first.")
    with open(CONV_COLUMNS_JSON) as f:
        return json.load(f)


def read_gcmt_subset():
    if not os.path.exists(GCMT_CSV):
        raise SystemExit(f"missing {GCMT_CSV}\nRun step5_subset_gcmt.py first.")
    duration_years = (pd.Timestamp(WINDOW_END) - pd.Timestamp(WINDOW_START)).days / 365.25
    try:
        df = pd.read_csv(GCMT_CSV)
        mws = sorted(float(m) for m in df["Mw"]) if len(df) else []
    except pd.errors.EmptyDataError:
        # A zero-event result is a legitimate outcome (some zones genuinely
        # have no qualifying GCMT event), not a broken file.
        mws = []
    return {
        "threshold_Mw": THRESHOLD_MW,
        "count": len(mws),
        "duration_years": duration_years,
        "observed_Mw": mws,
    }


def read_segment_gcmt(segment_key):
    """The per-segment catalogue step 5 wrote, in the same shape as the zone's.

    v9 (LEVEL 0). A segment must declare its OWN observed seismicity:
    run_logic_tree.py raises rather than let a segment inherit the zone's,
    because that would credit every segment with earthquakes that happened in
    the others and inflate its rate.
    """
    path = GCMT_CSV.replace(".csv", "_" + segment_key + ".csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\nRun step5_subset_gcmt.py first (with a "
            f"segmented run: it writes one catalogue per segment).")
    duration_years = (pd.Timestamp(WINDOW_END) - pd.Timestamp(WINDOW_START)).days / 365.25
    try:
        df = pd.read_csv(path)
        mws = sorted(float(m) for m in df["Mw"]) if len(df) else []
    except pd.errors.EmptyDataError:
        mws = []
    return {
        "threshold_Mw": THRESHOLD_MW,
        "count": len(mws),
        "duration_years": duration_years,
        "observed_Mw": mws,
    }


def build_segments():
    """The `segments` block of the config: one entry per along-strike segment.

    v9 (LEVEL 0). Returns {} when the run is unsegmented, which makes the
    config free of segments, as v8's.

    Each segment gets:
      * ``alongstrike_slice``   -- its columns (lib/segmentation.py
                                   zone_segments): by default between the
                                   trench end points Berryman et al. (2015)
                                   Table 3.1 gives the segment, or, with
                                   --segment-boundaries bird, where Bird
                                   (2003)'s plate pair changes. PUBLIC DATA
                                   either way: this is what keeps a segmented
                                   run honest under --ptha false.
      * ``observed_seismicity`` -- the earthquakes PTHA18's rule selects on
                                   its own columns (step 5).
      * coupling                -- its OWN Berryman et al. (2015) row, as
                                   PTHA18 gives each of its segments (a
                                   Berryman segment IS a row; a Bird segment
                                   takes the row(s) BIRD_SEGMENT_TO_BERRYMAN
                                   names); the zone's values where there is
                                   none.
      * b-value                 -- the zone's, as in PTHA18 (every segment row
                                   of sourcezone_parameters.csv has the zone's
                                   b range).

    Mw_max_observed is the LARGER of two numbers, never the zone's own:
      * the largest magnitude in the segment's own GCMT catalogue (the
        catalogue threshold, Mw 7.15, when it hosted nothing);
      * Berryman's value for the segment: the largest Mmax-min of its own
        rows (berryman_params.berryman_segment_mw_max_min: the largest
        earthquake thought to have occurred there, ReportPTHA 3.7.2.1), or,
        for a segment the table does not map, the smallest Mmax-min of the
        zone's rows (berryman_segment_mw_max_floor). GCMT starts in 1976,
        so on its own it would tell kurilsjapan's Kurils-Kamchatka segment
        that nothing above Mw 8.33 had happened there, 24 years after
        Kamchatka 1952 (Mw 9.0).
    If the segment's geometry cannot host Berryman's floor (the engine's own
    mw_max_anchor test), its catalogue value is used instead, with a warning:
    a stretch too small for the zone's smallest Mmax-min is most likely not
    part of the zone (kurilsjapan's Izu-Bonin tail, which step 3 flags and
    --auto-clip removes).
    """
    if not SEGMENTED:
        return {}, None

    sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))
    import segmentation as sg
    from berryman_params import (berryman_segment_mw_max_floor,
                                 berryman_rows_for_bird_segment,
                                 berryman_segment_coupling,
                                 berryman_segment_mw_max_min)
    from pyptha_v12 import moment_balance as mb
    from pyptha_v12 import unit_sources as us

    grid = np.load(GRID_NPY)
    segs = sg.zone_segments(grid, PTHA18_ZONE_NAME, SEGMENT_BOUNDARIES,
                            log=lambda m: print(m))
    if len(segs) < 2:
        print("  -> a single segment: running this zone unsegmented")
        return {}, None

    zone_floor = berryman_segment_mw_max_floor(PTHA18_ZONE_NAME)
    print("  segment mw_max_observed: max(own GCMT maximum, Berryman Mmax-min "
          "of its own rows, or the zone's smallest when it has none)")

    def can_host(slice_, mw):
        # the same slice, area and anchor test the engine applies
        i0, i1 = int(slice_[0]), int(slice_[1])
        st = us.discretized_source_approximate_summary_statistics(
            grid[:, :, (i0 - 1):(i1 + 1)])
        try:
            mb.mw_max_anchor(float(np.sum(st["length"] * st["width"])),
                             st["alongstrike_number"], st["width"],
                             mw_max_observed=mw, scaling_relation=SCALING_RELATION)
            return True
        except ValueError:
            return False

    out = {}
    for s in segs:
        key = s["segment_key"]
        gcmt = read_segment_gcmt(key)
        # the segment's own largest observed magnitude, or the catalogue's
        # threshold when it hosted nothing (a segment with no earthquake
        # cannot evidence a large one). It must always be written: a segment
        # without the key inherits the ZONE's mw_max_observed in the engine,
        # which gave kurilsjapan's 5-column, 0-event Izu-Bonin tail a floor
        # of Mw 9.05 its area cannot host (engine error). With the threshold
        # the floor is the method's own minimum, max(7.15 + 0.05, 7.35).
        gcmt_max = max(gcmt["observed_Mw"]) if gcmt["observed_Mw"] else THRESHOLD_MW
        rows = (s.get("berryman_rows")
                or berryman_rows_for_bird_segment(PTHA18_ZONE_NAME, s["plate_pair"]))
        berry_floor = berryman_segment_mw_max_min(rows) if rows else zone_floor
        mw_max_obs = gcmt_max
        if berry_floor is not None and berry_floor > gcmt_max:
            if can_host(s["alongstrike_slice"], berry_floor):
                mw_max_obs = berry_floor
            else:
                print(f"  WARNING segment {key}: its geometry cannot host "
                      f"Berryman's floor Mw {berry_floor:g}, so it keeps its "
                      f"own catalogue value {gcmt_max:g}. A stretch this small "
                      f"is probably not part of the zone: see step 3's "
                      f"plate-boundary check and --auto-clip.")
        print(f"    {key:10s} mw_max_observed {mw_max_obs:g} (GCMT {gcmt_max:g}"
              + (f", Berryman {berry_floor:g})" if berry_floor is not None else ")"))
        entry = {
            "alongstrike_slice": s["alongstrike_slice"],
            "rates": {
                "observed_seismicity": gcmt,
            },
            "_plate_pair": s["plate_pair"],
            "_berryman_rows": rows or [],
            "_boundaries": SEGMENT_BOUNDARIES,
            "_provenance": (
                f"DERIVED: along-strike columns "
                f"{s['alongstrike_slice'][0]}-{s['alongstrike_slice'][1]} of "
                f"this run's own mesh, "
                + (f"between the trench end points Berryman et al. (2015) "
                   f"Table 3.1 gives segment '{rows[0]}'"
                   if SEGMENT_BOUNDARIES == "berryman" else
                   f"from Bird (2003) plate pair '{s['plate_pair']}' "
                   f"(PB2002_steps.dat, class SUB)")
                + ". No PTHA18 file was read to place this boundary."),
        }
        if mw_max_obs is not None:
            entry["rates"]["mw_max_observed"] = mw_max_obs
        if rows:
            c = berryman_segment_coupling(rows)
            entry["rates"]["coupling"] = {
                "prior_type": "spreadsheet_and_uniform_50_50",
                "uniform_range": [0.1, 1.3],
                "spreadsheet_values": c,
                "prob_zero_coupling": 0.0,
            }
            print(f"    {key:10s} coupling {c[0]:g}/{c[1]:g}/{c[2]:g} from Berryman "
                  f"{' + '.join(rows)}")
        else:
            print(f"    {key:10s} coupling: the zone's (no Berryman row mapped to "
                  f"plate pair {s['plate_pair']})")
        out[key] = entry
    return out, segs


def read_official_convergence():
    """If step 8 has already run, read its convergence for the provenance
    note below. Returns None if step 8 has not run."""
    if not os.path.exists(OFFICIAL_SUMMARY_CSV):
        return None
    with open(OFFICIAL_SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["item"].startswith("convergence_mm_per_yr_"):
                return float(row["value"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite inputs/puysegur2_scratch.json even if it "
                         "already exists (discards any hand edits)")
    args = ap.parse_args()

    print("=" * 70)
    print("STEP 6 - Write the run_logic_tree.py input JSON")
    print("=" * 70)

    if os.path.exists(OUT_JSON) and not args.force:
        print(f"\n  {OUT_JSON} already exists -- NOT overwriting it.")
        print(f"  If you edited it by hand, every later step reads that "
              f"file as-is; nothing re-derives it behind your back.")
        print(f"  To discard those edits and regenerate from Berryman/SLAB/"
              f"GCMT, either delete the file or re-run with --force.")
        return

    if MESH_FILE is None and not os.path.exists(SHP):
        raise SystemExit(f"missing {SHP}\nRun step1_fetch_slab2.py first.")

    coupling, b_anchor, mw_max_observed = read_sourcezone_row()
    convergence = read_convergence()
    conv_columns = read_convergence_columns()
    use_bird = bool(conv_columns["use_bird_convergence"])
    gcmt = read_gcmt_subset()
    segments, seg_meta = build_segments()
    official_convergence = read_official_convergence() if USE_PTHA else None

    # v9: the largest earthquake known on the zone, Berryman's or GCMT's
    berryman_mw_max = mw_max_observed
    gcmt_mw_max = max(gcmt["observed_Mw"]) if gcmt["observed_Mw"] else None
    if mw_max_observed is not None and gcmt_mw_max is not None:
        mw_max_observed = max(mw_max_observed, gcmt_mw_max)

    print(f"  coupling (cmin/cpref/cmax): {coupling}  (berryman_params.py)")
    print(f"  b (bmin/bpref/bmax): {b_anchor}  (berryman_params.py)")
    if mw_max_observed is None and gcmt_mw_max is not None:
        # a zone Berryman does not treat: its own largest GCMT event
        mw_max_observed = gcmt_mw_max
    print(f"  mw_max_observed: "
          + (f"{mw_max_observed:g}" if mw_max_observed is not None else "none (set it by hand)")
          + "  (larger of Berryman "
          + (f"{berryman_mw_max:g}" if berryman_mw_max is not None else "none")
          + " and GCMT "
          + (f"{gcmt_mw_max:g}" if gcmt_mw_max is not None else "none") + ")")
    print(f"  convergence: {convergence:.4f} mm/yr  (step 3, horizontal; the "
          f"engine applies 1/cos(mean dip))")
    print("  event weights and LEVEL 4 target: "
          + ("Bird per-column profile (PTHA18's use_bird_convergence = 1 model)"
             if use_bird else "constant convergence (1/slip weights, flat target)"))
    print(f"  GCMT: {gcmt['count']} events above Mw {gcmt['threshold_Mw']:g} "
          f"in {gcmt['duration_years']:.4f} yr  (step 5)")

    shp_rel = os.path.relpath(SHP, ROOT).replace(os.sep, "/")

    conv_provenance = (
        f"DERIVED in step3_convergence.py: Bird (2003) velocities nearest each "
        f"column's top edge, area-weighted over the unit sources "
        f"(lib/bird_convergence.py, PTHA18's method). Horizontal rate: the "
        f"engine divides it by cos(mean dip) once. This run: "
        f"{convergence:.4f} mm/yr."
        + (" The per-column profiles in this block select PTHA18's Bird model "
           "for event probabilities and the LEVEL 4 target."
           if use_bird else
           " CONVERGENCE_OVERRIDE_MM_PER_YR was set in step 3, so this run "
           "uses PTHA18's constant-convergence model.")
    )
    if official_convergence is not None:
        conv_provenance += (
            f" Official (from step 8's run): {official_convergence:.4f} mm/yr."
        )
    else:
        conv_provenance += (
            " Official value not available -- run step8_official.py for a "
            "comparison."
        )

    cfg = {
        "_what_this_is": (
            f"{PTHA18_ZONE_NAME}, built FROM SCRATCH by examples/puysegur2/: "
            "geometry from SLAB2.0/SLAB1.0 (step 1-2), convergence "
            "area-weighted from Bird (2003) (step 3), observed seismicity "
            "from a from-scratch GCMT subset (step 4-5). Coupling, "
            "b_anchor and mw_max_observed are read from Berryman et al. "
            "(2015) directly via berryman_params.py, not from any "
            "PTHA18-internal file (see step6_write_input.py's module "
            "docstring). This is NOT the official input and will not "
            "reproduce PTHA18's published rate curve exactly."
        ),
        "run_name": RUN_NAME,
        "source_zone": PTHA18_ZONE_NAME,
        "geometry": {
            "mode": "shapefile",
            "shapefile": shp_rel,
            "desired_unit_source_length": DESIRED_LENGTH_KM,
            # width_km() reads sourcezone_parameters.csv (a PTHA18-internal
            # file, though a resolution parameter, not a "declared" physical
            # value). With --ptha=false this is never read: 50 km fixed
            # instead, matching step2_build_grid.py.tmpl's same choice --
            # see its module docstring for why.
            "desired_unit_source_width": DESIRED_WIDTH_KM,
            # n_downdip: step2_build_grid.py.tmpl already fixes this to the
            # official row count when USE_PTHA is True (its own log prints
            # "fixing down-dip rows to N"), but this JSON -- the one
            # run_logic_tree.py (step 7/9) actually meshes rates on -- never
            # carried that value, so the engine silently fell back to
            # build_grid()'s own automatic row-count rule instead. That rule
            # depends only on desired_unit_source_width, not on the official
            # row count, so it can (and did, for kermadectonga2) disagree
            # with step 2's own figure: 2 rows here vs. step 2's 3. Passing
            # it through explicitly keeps this JSON's mesh identical to the
            # one step 2 already validated.
            #
            # With --ptha=false there is no official row count to fall back
            # on, so N_DOWNDIP_OVERRIDE (this script's own copy, kept in
            # sync by hand with step2_build_grid.py's) takes over instead --
            # same desync, same fix, just for the manual-override path
            # rather than the official one.
            #
            # v8: the SAME fallback order step 2 uses (official rows if
            # --ptha true and the table exists, else N_DOWNDIP_OVERRIDE); v7
            # wrote None here with --ptha true on a zone PTHA18 never meshed,
            # while step 2 had used the override.
            "n_downdip": (
                (n_downdip_rows(PTHA18_ZONE_NAME) if USE_PTHA else None)
                or N_DOWNDIP_OVERRIDE),
            "min_downdip": MIN_DOWNDIP_ROWS,
            "seed": MESH_SEED,
            # Same desync this file's n_downdip comment already describes,
            # for the discretiser choice: step2_build_grid.py.tmpl meshes
            # with --discretizer DISCRETIZER (see generate.py), but until
            # this field existed, run_logic_tree.py's build_grid() -- what
            # step 7/9 actually mesh rates on -- had no way to know that and
            # silently fell back to its own default ("lm"), even when this
            # run was generated with --discretizer optimal. Confirmed to
            # matter: on puysegur2 --ptha=false, step 2's own optimal mesh
            # is 13x1 cells (area 38,910 km2), while the unpatched
            # build_grid() rebuilt 15x1 cells (area 40,879 km2, the plain-lm
            # mesh with its self-intersecting tip cell) for the actual rate
            # calculation. Passing it through keeps this JSON's mesh
            # identical to the one step 2 already validated.
            "discretizer": DISCRETIZER,
            # v8.2: how many columns the "optimal" mesh gets; without it the
            # engine would rebuild with rptha's trench rule and mesh the
            # rates on a different grid from step 2's.
            "column_rule": COLUMN_RULE,
            # v8: where step 2 saved the mesh, with its fingerprint (contour
            # file hash + the meshing parameters above) alongside. The engine
            # reuses it when the fingerprint still matches the fields above,
            # so steps 7 and 9 use exactly the mesh step 2 built and checked;
            # edit any meshing field and it rebuilds instead.
            "unit_source_grid_npy": os.path.relpath(
                os.path.join(EXAMPLE, "data", "slab2", "unit_source_grid.npy"),
                ROOT).replace(os.sep, "/"),
        },
        # PTHA18 builds FAUS events from Mw 7.2 to 9.8 in steps of 0.1
        # (SOURCE_ZONES/TEMPLATE/TSUNAMI_EVENTS/config.R lines 37-39). Events
        # above the largest Mw_max (9.6) get zero rate, so this does not
        # change any rate; v3-v7 stopped at 9.6, so their event tables
        # lacked PTHA18's Mw 9.7-9.8 rows (v8 check: with Mmax 9.6 the event
        # set is otherwise identical to PTHA18's published tables).
        "events": {
            "Mmin": 7.2,
            "Mmax": 9.8,
            "dMw": 0.1,
            "source_zone_name": PTHA18_ZONE_NAME,
            # Read by the engine and step 7b (see SCALING_RELATION above).
            "scaling_relation": SCALING_RELATION,
            "shear_modulus_Pa": SHEAR_MODULUS_PA,
            # v10, read by the engine, step 7b and step 9 (see RUPTURE_SIZE).
            "rupture_size": RUPTURE_SIZE,
        },
        "rates": {
            "tectonic_convergence_mm_per_yr": convergence,
            "coupling": {
                "prior_type": "spreadsheet_and_uniform_50_50",
                "uniform_range": [0.1, 1.3],
                "spreadsheet_values": coupling,
                "prob_zero_coupling": 0.0,
            },
            "b_anchor": b_anchor,
            "Mw_max_anchor": "derive",
            "Mw_min": [7.15],
            "Mw_min_prob": [1.0],
            "Mw_frequency_distribution": [
                "truncated_gutenberg_richter", "characteristic_gutenberg_richter"],
            "Mw_frequency_distribution_prob": [0.7, 0.3],
            "n_logic_tree_bins": {"coupling": 20, "b": 20, "Mw_max": 40},
            "update_logic_tree_weights_with_data": True,
            "mw_max_posterior_equals_mw_max_prior": False,
            "edge_correction": {"mode": "fit", "lower": 0.0, "upper": 30.0},
            "account_for_moment_below_mwmin": True,
            "conditional_probability_model": "inverse_slip",
            "mw_max_observed": mw_max_observed,
            "scaling_relation": SCALING_RELATION,
            "observed_seismicity": gcmt,
        },
        "segments": segments,
        "percentiles": {
            "threshold_Mw": PERCENTILE_MW,
            "copula": "comonotonic",
            "N": 40000,
            "percentile_probs": [0.025, 0.16, 0.5, 0.84, 0.975],
            "seed": 123,
        },
        "outputs": {"write_branch_rate_curves": False},
        "_convergence_provenance": conv_provenance,
        "_observed_seismicity_provenance": (
            f"DERIVED in step4-5: GCMT catalogue downloaded fresh and "
            f"filtered by this example's own reimplementation of "
            f"gcmt_subsetter.R (buffer 0.4 deg, rake and strike within 50 "
            f"deg of either nodal plane, depth <= 71 km). {gcmt['count']} "
            f"event(s) above Mw {gcmt['threshold_Mw']:g} in "
            f"{gcmt['duration_years']:.4f} years, window "
            f"[{WINDOW_START}, {WINDOW_END}), PTHA18's own window, so the count "
            f"and the duration compare directly with an official run's. See "
            f"step5_subset_gcmt.py's module docstring."
        ),
        "_coupling_provenance": (
            f"DECLARED, not derived: read directly from Berryman et al. "
            f"(2015) Table 3.1 via berryman_params.py's "
            f"berryman_coupling('{PTHA18_ZONE_NAME}') -- Berryman's Whole "
            f"Margin row for the zone where Table 3.1 has one (v9), else "
            f"the mean across this zone's merged Berryman segments; NOT "
            f"from PTHA18's own sourcezone_parameters.csv (that is v2's "
            f"approach). See "
            f"berryman_params.py's module docstring for the exact "
            f"citation, table values and averaging rule, and for why "
            f"coupling cannot be derived from geometry or a catalogue at "
            f"all."
        ),
        "_b_anchor_provenance": (
            f"DECLARED, not derived: bmin/bpref/bmax = {b_anchor}, read "
            f"directly from Berryman et al. (2015) Table 3.1 via "
            f"berryman_params.py's berryman_b_anchor('{PTHA18_ZONE_NAME}') "
            f"-- mean across this zone's merged Berryman segments."
        ),
        "_mw_max_observed_provenance": (
            f"DERIVED: the larger of Berryman et al. (2015)'s largest "
            f"Mmax-min for the zone ({berryman_mw_max:g}, "
            f"berryman_mw_max_observed) and the largest magnitude in step 5's "
            f"GCMT subset ("
            + (f"{gcmt_mw_max:g}" if gcmt_mw_max is not None else "no event")
            + f"): the largest earthquake known on the zone (ReportPTHA "
            f"Section 3.7.2.1)."
        ),
    }

    # R picks the event-probability model and the LEVEL 4 target from
    # use_bird_convergence (compute_rates_all_sources.R:371-420, 537-573);
    # "inverse_slip" above is its constant-convergence choice.
    if use_bird:
        cfg["rates"].update(bc.rates_entries(conv_columns))

    if MESH_FILE is not None:
        # v12: the engine reads the external mesh directly
        cfg["geometry"] = {
            "mode": "mesh_file",
            "mesh_file": os.path.relpath(os.path.join(EXAMPLE, "inputs", "geometry", MESH_FILE),
                                         ROOT).replace(os.sep, "/"),
            "mesh_depth_units": MESH_DEPTH_UNITS,
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  wrote -> {OUT_JSON}")
    print(f"  run_name: {RUN_NAME}")
    print("\nNext:  step7_run.py")


if __name__ == "__main__":
    sys.exit(main())
