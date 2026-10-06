"""Generate a from-scratch PTHA18 example folder, v11, for any
SLAB2.0/SLAB1.0 zone.

v11 = v10_q plus step 7c, the variable shear modulus (2026-10-06):
PTHA18's variable_mu rates (rigidity from each cell's depth, each
scenario's magnitude relabelled, LEVEL 3 with that difference as a
catalogue magnitude error), next to the unchanged constant-rigidity ones.
README: "What v11 changes", "Known limitations (v11)".

v10_q = v10 plus (1) q: with --rupture-size local, each rupture's
conditional-probability weight is multiplied by the mean over its cells of
1 / (number of same-magnitude ruptures containing the cell), so places
covered by more ruptures (smaller cells) do not get more rate; and (2) the
trench ramp rule, --trench-ramp on (default): a trench end more than 3 km
deeper than the trench's normal depth is the edge of the SLAB data and is
trimmed (at most 100 km) before the local datum is built (hellenic 71 km,
caribbean/antilles 68 km, kermadectonga2 11 km, other zones unchanged).
README: "Trench ramp rule (v10_q)", "What v10_q changes", "Known
limitations (v10_q)".

v10 = v9 plus --rupture-size rptha|local (default rptha = v9): with local,
each uniform-slip rupture is sized from the real km of its own cells instead
of one cell count per magnitude from the zone's mean cell size (README,
"What v10 changes"). The case for both, zone by zone:
V9/html/v10q_rationale.html.

v9 = v8 plus LEVEL 0 segmentation (--segmented true): segment boundaries
from Berryman et al. (2015) Table 3.1's trench end points (default; or Bird
(2003)'s plate pairs, --segment-boundaries bird), and PTHA18's segment method checked against
rptha (each segment on the zone's full scenario table weighted by the part
of each scenario inside it, its own convergence, GCMT events, Berryman row
and Mw_max; the zone's scenario rates with PTHA18's partial-segmentation
percentiles). Step 8 of a segmented example runs PTHA18's own segmented
model. Zone coupling is Berryman's Whole Margin row where it has one, and
mw_max_observed the largest earthquake known on the zone (Berryman's largest
row or GCMT); both differ from v8's (README, What v9 adds). Recipes:
from_scratch_v12/README.md, "What v9 adds".

v8 = a full audit of v7 against rptha's R source and PTHA18's own published
files, with the unit-source mesh rebuilt the way PTHA18 built it. Every
change, with the evidence for it, is in from_scratch_v12/html/docs/index.html.
In short:

  * Step 1 builds PTHA18's contours: "depth below the NEARBY trench", the
    trench itself as the 0 km line, then one unbroken line every 5 km to the
    seismogenic cutoff (lib/slab_contours.py). v3-v7 used ONE trench depth
    for the whole zone, which broke the shallow contours wherever the real
    trench is deeper; v4-v7 then threw the shallow band away and piled up
    retries and hand clips. Gone: SHALLOWEST_KM, N_CONTOUR_LEVELS, the
    width-ratio checks, the retry ladder, per-zone clips (an optional,
    generic --clip replaces the last one).
  * The mesh optimiser now reproduces rptha's minpack.lm::nls.lm call step
    for step (the 50-iteration cap, R's Inf for invalid points, MINPACK's
    exact finite differences; 10-20x faster), so its down-dip lines match
    rptha's R output to within metres (v7: 0.2-1.4 km).
  * --discretizer optimal keeps rptha's own mesh whenever it has no folded,
    collapsed or overlapping cell, and repairs only a broken one. Fed
    PTHA18's own contours it reproduces the official mesh on all 14 zones
    tested (dimensions identical, area within 0.004%).
  * Clean ends: where the SLAB data end a zone crookedly (end cells more
    than 45 deg off square) or the zone narrows to a point, step 1 cuts
    every contour along one down-dip line, as PTHA18 did by hand; a clean
    end is kept. On kermadectonga2 the cuts land within 10 km of PTHA18's
    hand-made ends.
  * PTHA18's documented minimum of 2 down-dip rows is applied (ReportPTHA
    p.14, the reason for Puysegur's 35 km width).
  * Step 2's mesh is saved with a fingerprint and reused by steps 7 and 9.
  * Rates: step 3 hands the engine the horizontal Bird convergence (v3-v7
    divided it by cos(mean dip) and the engine divided again: +0.6% to +24%
    on the slip rate), and the engine uses PTHA18's Bird model for event
    probabilities and the LEVEL 4 target (lib/bird_convergence.py). On
    PTHA18's own inputs every scenario rate now matches PTHA18's published
    one to ~1e-6 (v7: median 4-26% off on Bird zones).
  * Events are listed in rptha's row order, so event numbers match PTHA18's
    tables; step 5 applies gcmt_subsetter.R's exact rule (PTHA18's own event
    list on all 8 zones tested); step 8 runs the same engine as step 7.

v6 = v5's templates and engine, unchanged, plus one new step: STEP 7b
generates HETEROGENEOUS-slip (HS) AND variable-area-uniform-slip (VAUS)
variants of the same FAUS ruptures step 7 already builds. HS uses the SFFM
generator (pyptha_v12/stochastic_slip.py, a port of rptha's
sffm_fit_simulate_earthquake.R -- the S_NCF method of Davies et al. 2015),
ported in v5 but never wired into any step. VAUS is derived from each HS
field exactly as rptha's make_all_earthquake_events.R does (not an
independent stochastic model -- see step7b_stochastic_slip.py's module
docstring). v6 wires both in as an ADDITIONAL, clearly-labelled output, not
a replacement:

  - LEVELs 0-5 (segmentation, logic tree, moment balance, Bayesian update,
    edge correction, percentiles) still run on FAUS (Fixed Area Uniform
    Slip) ruptures only, exactly as v5 and as PTHA18 itself does. Every
    rate number in rate_curves.csv / exceedance_rate_percentiles.csv is
    FAUS, unchanged by step 7b.
  - step 7b takes every FAUS placement the event table produced and, for
    each one, generates rptha's own 15-per-placement (floored at 200/
    magnitude) heterogeneous-slip realisations (same total slip / same Mw,
    different spatial pattern), then derives the matching VAUS field from
    each one -- both as an illustrative addition, more realistic tsunami
    initial conditions, not a different rate model. Written to
    outputs/hs_slip_fields/ and outputs/vaus_slip_fields/, never read back
    into the logic tree.
  - step9_report.py labels every rupture-derived figure/table FAUS, HS or
    VAUS explicitly, so the three are never conflated.

v5 = v4's templates, unchanged, plus three engine fixes found by an
independent fidelity audit against the rptha R source (spherical vs
ellipsoidal geodesy in unit_sources.py, computational_increment=0.02, and
fit_edge_multiplier's degeneracy-check baseline). v5 carries its own forked
copy of pyptha/python_logic_tree (from_scratch_v12/pyptha_v12,
from_scratch_v12/python_logic_tree_v12) so v1-v4 and the shared engine are
completely unaffected. See from_scratch_v12/html/docs/code_map.html for the full detail on
each fix, and what was independently verified as ALREADY faithful.

v4 = v3 plus one fix: the mesh no longer folds at the along-strike tips.

What v4 changes, relative to v3
-------------------------------
v3's meshes carry a geometric artefact at the ends of the arc: on
kermadectonga2 the last down-dip column sits 89.7 deg from orthogonal
(it runs ALONG the arc instead of across it) and is 209 km long against
~107 km mid-arc, which is the visible spike at the southern tip; on
puysegur2 the same effect bow-ties one unit source outright.

The cause is not the optimiser. rptha pins ``s = 0`` and ``s = 1`` on
every contour row -- only interior columns are free parameters (see
``create_downdip_lines_on_source_contours_improved``, whose
``moving_par`` is ``s_matrix[, 2:(np-1)]``). So the edge columns are
*required* to join the contours' own endpoints. Where SLAB resolves the
shallow interface over a shorter along-strike span than the deep one,
those endpoints are far apart along the arc:

    zone             shallow-to-deep end gap     mesh under v3
    kermadectonga2   start 121 km, end 209 km    severe tip fold
    puysegur2        start  33 km, end  57 km    one bow-tied cell
    kurilsjapan      start 147 km, end 146 km    clean (symmetric)

v4 adds ``--discretizer optimal`` (the new default), which trims the
contours to the along-strike span all depth levels share before handing
them to the SAME optimiser, so those edge columns join points that face
each other. Because aligning helps only where the ends disagree -- doing
it to kurilsjapan, which is already clean, made it worse -- optimal
builds both meshes and keeps the aligned one only when it measurably
improves (fewer self-intersecting cells, or lower worst-column badness).

Measured, --ptha=false:

    zone             worst column      bow-ties    area
    kermadectonga2   1.087 -> 0.239    0 -> 0      -3.7%
    puysegur2        0.862 -> 0.337    1 -> 0      -4.8%
    kurilsjapan      0.196 (lm kept)   0           unchanged

``--discretizer lm`` reproduces v3 exactly, and ``mid`` is rptha's older
deprecated method; both are kept for comparison.

A second fix lands in ``official_geometry_params.shallowest_full_arc_km``:
it used to measure the shallowest usable contour from ALL fragments
stacked together, so a level SLAB had broken into four disjoint pieces
counted as spanning the whole arc. It now measures the longest SINGLE
fragment. ``step1_fetch_slab2.py`` also refuses outright, rather than
silently joining, if a level it is asked to extract turns out to be
fragmented across a large gap -- joining those draws straight lines that
do not follow the interface.

Everything else is v3, unchanged
--------------------------------
Same nine-step pipeline, same Berryman et al. (2015) inputs for coupling,
b-value, mw_max_observed, the seismogenic cutoff and the datum, same
--ptha flag. See from_scratch_v3/generate.py for that lineage.

What v4 still cannot do, by construction
----------------------------------------
Match PTHA18's cell boundaries: those come from a hand-edited shapefile
whose editing process was never published. PTHA18's own template
(``make_initial_downdip_lines.R``) calls the automatic discretiser and
then says the result is "typically ... used to define the unit source
lateral boundaries, optionally with further manual editing" -- so the
published mesh is the automatic one plus edits nobody outside can
reproduce. Judge v4 the same way as v3: on total area and mean dip.

Zones PTHA18 never modelled at all (new in v6)
------------------------------------------------
`--zone` used to require an exact sourcezone_parameters.csv sourcename --
PTHA18's own internal zone list. With `--ptha false`, it no longer does:
any SLAB2.0/SLAB1.0 region can be run under any zone name, including one
PTHA18 never modelled, PROVIDED that name also has a berryman_params.py
mapping (ZONE_TO_BERRYMAN_SEGMENTS / ZONE_DOMINANT_SEGMENT). If it does
not, generate.py stops with a message telling you so -- the seismogenic
cutoff and datum (needed by step 1, before any input JSON exists to hand-
edit) have no safe generic default, so they must come from a real
Berryman et al. (2015) Table 3.1 row, added to berryman_params.py the same
way every other zone in it was (see BERRYMAN_TABLE_SOURCE in that module's
docstring). "calabria" is one such zone, added this way (BERRYMAN_ROWS
already had a "Calabria" row; only the zone-name mapping was missing).

Three other literature inputs (coupling, b-value, scaling_relation/
shear_modulus) DO have a documented generic default and fall back to it
automatically, with a printed note, when the zone has no mapping:
coupling -> [0.1, 1.3] (PTHA18's own uniform prior, already applied to
every zone as half of a 50/50 blend regardless of its own spreadsheet
value); b-value -> [0.7, 0.95, 1.2] (the range most "no differentiation"
Berryman rows already carry); scaling_relation/shear_modulus -> Strasser/
3e10 (what every zone this project currently runs already resolves to;
sourcezone_parameters.csv itself has no possible row for a zone PTHA18
never modelled, so this one can never come from Berryman either). See
report_geometry_inputs()'s output for which of these applied.

Step 8 (the official PTHA18 comparison) is never available for such a
zone -- there is nothing to compare against -- and step 9's report simply
omits that section, exactly as it already does for a zone PTHA18 retired.

Usage
-----
    .venv/Scripts/python.exe from_scratch_v12/generate.py <name> [--zone Z]
        [--folder F] [--ptha true|false] [--segmented true|false]
        [--segment-boundaries berryman|bird]
        [--columns trench|average] [--convergence bird|bird-griffin]
        [--rupture-size rptha|local] [--trench-ramp on|off]
        [--discretizer optimal|lm|mid]
        [--clip LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]
        [--trench-ends LON1,LAT1,LON2,LAT2]

    .venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kermadectonga2_v9 --ptha false
    .venv/Scripts/python.exe from_scratch_v12/generate.py kermadec --zone kermadectonga2 --folder kermadectonga2_v9seg --ptha false --segmented true
    .venv/Scripts/python.exe from_scratch_v12/generate.py calabria --zone calabria2 --folder calabria2_v9 --ptha false
    .venv/Scripts/python.exe from_scratch_v12/generate.py caribbean --zone antilles2 --folder antilles2_v9 --ptha false --clip 297,307,5,18.5
    .venv/Scripts/python.exe from_scratch_v12/generate.py hellenic --zone hellenic_west2 --folder hellenic_west2_v9 --ptha false
    .venv/Scripts/python.exe from_scratch_v12/generate.py hellenic --zone hellenic2 --folder hellenic2_v9seg --ptha false --segmented true
    then: .venv/Scripts/python.exe <folder>/steps/step_total.py --skip-hs
"""

import argparse
import csv
import html
import os
import re
import shlex
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
TEMPLATES_DIR = os.path.join(HERE, "templates")

# This package's own names, substituted into every generated step so an
# example built by v9 runs v9's engine (see the mapping in main()).
PKG = os.path.basename(HERE)
PYPKG = "pyptha_v12"
LTPKG = "python_logic_tree_v12"
# v12: rptha/ next to the package, or one level up (package in V9/)
_RPTHA_ROOT = next((r for r in (ROOT, os.path.dirname(ROOT))
                    if os.path.isdir(os.path.join(r, "rptha"))), ROOT)
SZP = os.path.join(_RPTHA_ROOT, "rptha", "R", "examples", "austptha_template", "DATA",
                   "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))  # helper modules
from slab2_catalog import by_id, find_region, guess_ptha18_zones  # noqa: E402
from official_geometry_params import width_km, official_area_and_dip  # noqa: E402
from berryman_params import (berryman_cutoff_km, berryman_coupling,  # noqa: E402
                             berryman_b_anchor, berryman_mw_max_observed,
                             ZONE_TO_BERRYMAN_SEGMENTS, ZONE_DEFAULT_CLIP,
                             ZONE_DEFAULT_TRENCH_ENDS)

TEMPLATE_FILES = [
    "step1_fetch_slab2.py",
    "step2_build_grid.py",
    "step3_convergence.py",
    "step4_fetch_gcmt.py",
    "step5_subset_gcmt.py",
    "step6_write_input.py",
    "step7_run.py",
    "step7b_stochastic_slip.py",
    "step7c_variable_mu.py",
    "step8_official.py",
    "step9_report.py",
    "step_total.py",
]


def unsegmented_zone_names():
    names = []
    with open(SZP, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("segment_name", "").strip():
                names.append(row["sourcename"].strip())
    return names


def zone_exists(name):
    return name in unsegmented_zone_names()


def zone_row_weight(name):
    """row_weight for `name`'s unsegmented row, or None if not found."""
    with open(SZP, newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if v else "") for k, v in raw_row.items()}
            if row["sourcename"] == name and not row.get("segment_name"):
                try:
                    return float(row["row_weight"])
                except (KeyError, ValueError):
                    return None
    return None


def resolve_region(name, explicit_id):
    if explicit_id:
        region = by_id(explicit_id)
        if region is None:
            raise SystemExit(
                f"'{explicit_id}' is not one of the 27 known SLAB2.0 "
                f"ScienceBase item ids.\nRun "
                f".venv/Scripts/python.exe from_scratch_v12/list_regions.py "
                f"to see the full list.")
        return region

    hits = find_region(name)
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise SystemExit(
            f"no SLAB2.0 region matches '{name}'.\nRun "
            f".venv/Scripts/python.exe from_scratch_v12/list_regions.py "
            f"to see the 27 available regions, then either use one of "
            f"their names/prefixes or pass --id directly.")
    lines = "\n".join(f"  {r['title']}  (prefix '{r['prefix']}', "
                      f"id {r['id']})" for r in hits)
    raise SystemExit(
        f"'{name}' matches more than one SLAB2.0 region:\n{lines}\n"
        f"Be more specific, or pass --id <scibase_item_id> directly.")


def warn_if_retired(zone_name):
    if zone_row_weight(zone_name) == 0.0:
        print(f"note: '{zone_name}' is a retired PTHA18 zone "
             f"(row_weight = 0) -- step 8 will find no logic tree to compare "
             f"against.")


def resolve_zone_from_berryman(explicit_zone, name, region_title):
    """--ptha false: the zone name, from this package's own Berryman et al.
    (2015) mapping only (lib/berryman_params.py); sourcezone_parameters.csv
    is never read."""
    if explicit_zone:
        if explicit_zone not in ZONE_TO_BERRYMAN_SEGMENTS:
            print(f"  note: '{explicit_zone}' has no Berryman et al. (2015) "
                  f"mapping in lib/berryman_params.py, so step 1 will stop. "
                  f"Add the zone to ZONE_TO_BERRYMAN_SEGMENTS and "
                  f"ZONE_DOMINANT_SEGMENT there first (all valid names: "
                  f"html/docs/zones.html).")
        return explicit_zone
    zone_names = sorted(ZONE_TO_BERRYMAN_SEGMENTS)
    squashed = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if squashed in zone_names:
        return squashed
    guesses = guess_ptha18_zones(region_title, zone_names)
    if len(guesses) == 1:
        return guesses[0]
    if not guesses:
        raise SystemExit(
            f"could not guess a zone name for '{region_title}' from "
            f"lib/berryman_params.py.\nPass --zone <name>.")
    raise SystemExit(
        f"'{region_title}' matches more than one zone:\n"
        + "\n".join(f"  {z}" for z in guesses) + "\nPass --zone <name> to pick one.")


def resolve_ptha18_zone(explicit_zone, name, region_title, allow_new_zone=False):
    if allow_new_zone:  # --ptha false
        return resolve_zone_from_berryman(explicit_zone, name, region_title)
    zone_names = unsegmented_zone_names()

    if explicit_zone:
        if not zone_exists(explicit_zone):
            if not allow_new_zone:
                raise SystemExit(
                    f"'{explicit_zone}' is not an unsegmented sourcename in "
                    f"sourcezone_parameters.csv.\nIf this is a real "
                    f"subduction zone SLAB2.0/SLAB1.0 covers but PTHA18 "
                    f"never modelled, pass --ptha false to build a run for "
                    f"it anyway (falls back to generic defaults for "
                    f"literature values PTHA18 does not have for this "
                    f"zone -- see report_geometry_inputs' output).")
            berry = ("it has a Berryman et al. (2015) mapping"
                     if explicit_zone in ZONE_TO_BERRYMAN_SEGMENTS else
                     "and it has NO Berryman et al. (2015) mapping either")
            print(f"  note: '{explicit_zone}' is not a PTHA18 zone (no row "
                  f"in sourcezone_parameters.csv; {berry}) -- proceeding "
                  f"with --ptha false. step 8 "
                  f"(official comparison) will not be available for this "
                  f"zone; see the geometry-inputs report below for which "
                  f"other numbers use a generic default vs. need to be set "
                  f"by hand.")
            return explicit_zone
        warn_if_retired(explicit_zone)
        return explicit_zone

    guesses = guess_ptha18_zones(region_title, zone_names)
    active = [z for z in guesses if zone_row_weight(z) != 0.0]

    squashed = name.lower().replace(" ", "").replace("-", "").replace("_", "")
    if zone_exists(squashed):
        if zone_row_weight(squashed) == 0.0 and len(active) == 1:
            print(f"note: '{squashed}' is retired -- using '{active[0]}'.")
            return active[0]
        warn_if_retired(squashed)
        return squashed

    if len(guesses) == 1:
        warn_if_retired(guesses[0])
        return guesses[0]
    if len(guesses) == 0:
        raise SystemExit(
            f"could not guess a PTHA18 zone name for '{region_title}'.\n"
            f"Pass --zone <exact_sourcename>.")
    if len(active) == 1:
        return active[0]

    lines = "\n".join(
        f"  {z}" + ("  (retired)" if zone_row_weight(z) == 0.0 else "")
        for z in guesses)
    raise SystemExit(
        f"'{region_title}' matches more than one PTHA18 zone:\n{lines}\n"
        f"Pass --zone <exact_sourcename> to pick one.")


CONVERGENCE_SOURCES_CSV = os.path.join(HERE, "data", "bird", "ptha18_convergence_sources.csv")


def convergence_source_note(zone, choice):
    """A note on where PTHA18 took `zone`'s convergence from, and what the
    two --convergence tables give on PTHA18's own mesh (measured by
    html/docs/figures_src/make_convergence_sources.py). Empty if the zone is
    not a PTHA18 subduction zone."""
    if not os.path.exists(CONVERGENCE_SOURCES_CSV):
        return ""
    with open(CONVERGENCE_SOURCES_CSV, newline="") as f:
        row = next((r for r in csv.DictReader(f) if r["zone"] == zone), None)
    if row is None:
        return ""
    both = (f"on PTHA18's mesh: bird {row['conv_bird']} mm/yr, bird-griffin "
            f"{row['conv_bird_griffin']} mm/yr")
    if row["use_bird_convergence"] == "0":
        return (f"  note: PTHA18 did not use Bird on '{zone}' but a constant "
                f"{row['constant_mm_per_yr']} mm/yr (set CONVERGENCE_OVERRIDE_MM_PER_YR "
                f"in step 3 for that); {both}")
    if row["source"] == "bird":
        return f"  note: PTHA18 took {zone}'s convergence from Bird's own steps; {both}"
    who = "Griffin's traces" if row["source"] == "griffin" else "Bird plus some of Griffin's or Davies' traces"
    hint = ("" if choice == "bird-griffin" else
            "; --convergence bird-griffin reproduces PTHA18's value")
    return (f"  NOTE: PTHA18 took {zone}'s convergence from {who} "
            f"({row['traces']}), not from Bird alone; {both}{hint}")


def pick_folder_name(requested):
    candidate = requested
    n = 1
    while os.path.exists(os.path.join(ROOT, candidate)):
        n += 1
        candidate = f"{requested}_{n}"
    return candidate


def render(text, mapping):
    for key, val in mapping.items():
        text = text.replace("{{" + key + "}}", val)
    leftover = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if leftover:
        raise SystemExit(
            f"template still has unresolved placeholders: {sorted(set(leftover))}")
    return text


def report_geometry_inputs(zone, use_ptha=True):
    """Print the recovered numbers before generating, so they are visible.

    If any of these looks wrong, it is better to find out now than after
    running the whole pipeline on it. With use_ptha False nothing of PTHA18
    is read.
    """
    print("\n  Geometry inputs recovered for this zone")
    print("  " + "-" * 62)
    cutoff, dominant_segment = berryman_cutoff_km(zone)
    if cutoff is None:
        print(f"  seismogenic cutoff   NOT AVAILABLE -- '{zone}' has no "
              f"Berryman et al. (2015) segment mapping: add the zone to ZONE_TO_BERRYMAN_SEGMENTS and ZONE_DOMINANT_SEGMENT in lib/berryman_params.py (step 1 stops without it).")
    else:
        print(f"  seismogenic cutoff   {cutoff:6.1f} km below trench")
        print(f"                       (Berryman et al. 2015, dominant segment "
              f"'{dominant_segment}')")
    if not use_ptha:
        print("  unit-source width      50.0 km  (--ptha false: the general "
              "default, see step2_build_grid.py)")
    else:
        try:
            width = width_km(zone)
            print(f"  unit-source width    {width:6.1f} km  "
                  f"(sourcezone_parameters.csv -- a resolution parameter, not a "
                  f"physical 'declared' value)")
        except SystemExit:
            print(f"  unit-source width    NOT AVAILABLE -- '{zone}' has no row "
                  f"in sourcezone_parameters.csv; falling back to the general "
                  f"default of 50 km (see step2_build_grid.py's own docstring)")
    has_berryman = zone in ZONE_TO_BERRYMAN_SEGMENTS
    coupling = berryman_coupling(zone)
    b_anchor = berryman_b_anchor(zone)
    mw_max_observed = berryman_mw_max_observed(zone)
    coupling_src = "Berryman et al. 2015" if has_berryman else \
        "generic default -- PTHA18's own uniform prior, no Berryman mapping for this zone"
    b_src = "Berryman et al. 2015" if has_berryman else \
        "generic default -- no Berryman mapping for this zone"
    print(f"  coupling (min/pref/max)  {coupling}  ({coupling_src})")
    print(f"  b (min/pref/max)         {b_anchor}  ({b_src})")
    if mw_max_observed is None:
        print(f"  mw_max_observed          NOT AVAILABLE -- '{zone}' has no "
              f"Berryman et al. (2015) segment mapping: add the zone to ZONE_TO_BERRYMAN_SEGMENTS and ZONE_DOMINANT_SEGMENT in lib/berryman_params.py (step 1 stops without it).")
    else:
        print(f"  mw_max_observed          {mw_max_observed:g}  (Berryman et al. 2015)")
    if not use_ptha:
        return  # no PTHA18 file is read with --ptha false
    off = official_area_and_dip(zone)
    if off:
        print(f"  official area        {off['area_km2']:10,.1f} km2   "
              f"<- step 2 will compare against this")
        print(f"  official mean dip    {off['mean_dip_deg']:6.3f} deg")
        print(f"  official mesh        {off['n_unit_sources']} unit sources, "
              f"depth {off['depth_min_km']:.1f}-{off['depth_max_km']:.1f} km")
    else:
        print(f"  official table not downloaded; step 2 will skip the "
              f"area/dip comparison.")
        print(f"  get it with: official_ptha_data/fetch_official_inputs.py "
              f"{zone}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate a from-scratch PTHA18 pipeline (v12) for any "
                    "SLAB2.0/SLAB1.0 zone.")
    ap.add_argument("name", help="zone name, e.g. cascadia, 'south america'")
    ap.add_argument("--zone", default=None,
                    help="zone name: with --ptha true an exact "
                         "sourcezone_parameters.csv sourcename, with --ptha "
                         "false a name in lib/berryman_params.py")
    ap.add_argument("--folder", default=None,
                    help="folder to create, relative to the repository root; "
                         "may be nested (e.g. examples/calabria2)")
    ap.add_argument("--id", default=None, help="ScienceBase item id")
    ap.add_argument("--ptha", default="true", choices=["true", "false"],
                    help="true (default): steps may read PTHA18's files -- "
                         "per-zone unit-source width and scaling "
                         "(sourcezone_parameters.csv) and official down-dip "
                         "row count (the .nc file) -- and requires the zone to "
                         "be one PTHA18 modelled. false: steps 1-7 and 9 read "
                         "no PTHA18 or rptha file (50 km width, "
                         "geometry-derived row count, Strasser/30 GPa); only "
                         "step 8, the official run, does. Also allows --zone "
                         "to be a zone PTHA18 never modelled, provided it "
                         "has a berryman_params.py mapping. The Bird table is "
                         "chosen by --convergence, not by this flag.")
    ap.add_argument("--discretizer", default="optimal",
                    choices=["optimal", "lm", "mid"],
                    help="optimal (default): rptha's own mesh, kept unchanged "
                         "whenever it has no folded, collapsed or overlapping "
                         "cell; only a broken mesh is repaired (nearest-point "
                         "start, then end-aligned contours). "
                         "lm: rptha's method exactly as PTHA18 runs it, no "
                         "repairs. mid: rptha's older, deprecated method "
                         "(eq_spacing cuts); PTHA18 does not use it.")
    ap.add_argument("--columns", default="trench", choices=["trench", "average"],
                    help="v8.2: how many unit sources along strike the optimal "
                         "mesh gets. trench (default): rptha's and PTHA18's "
                         "rule, trench length / 50 km. average: the average row "
                         "length / 50 km, for a zone whose mesh tapers far more "
                         "than any PTHA18 mesh (step 2 prints the taper; "
                         "PTHA18: at most 1.17), e.g. calabria2: 20 -> 12 "
                         "columns. Not PTHA18's procedure.")
    ap.add_argument("--clip", default=None,
                    help="optional LON_MIN,LON_MAX,LAT_MIN,LAT_MAX window: "
                         "step 1 uses only this part of the SLAB raster. For "
                         "a SLAB region that covers more than the source zone "
                         "(e.g. only the Lesser Antilles part of 'Caribbean'). "
                         "Longitudes in the raster's own convention.")
    ap.add_argument("--trench-ends", default=None,
                    help="optional LON1,LAT1,LON2,LAT2: the zone's two ends, for "
                         "a zone that is one part of a SLAB region. Step 1 "
                         "builds the trench, datum and contours on the whole "
                         "raster and cuts every contour along the down-dip "
                         "line through each point. Default for hellenic_west2, "
                         "hellenic_east2 and hellenic2: Berryman et al. (2015) "
                         "Table 3.1's segment end points "
                         "(berryman_params.ZONE_DEFAULT_TRENCH_ENDS).")
    ap.add_argument("--trench-ramp", default="on", choices=["on", "off"],
                    help="v10_q: on (default): an end of the trench that climbs in "
                         "depth like a ramp, more than 3 km deeper than the "
                         "trench's normal depth, is the edge of the SLAB data, "
                         "not trench, and is trimmed (at most 100 km) before "
                         "the local datum is built. Changes hellenic (71 km at "
                         "the Kefalonia end), caribbean/antilles (68 km) and "
                         "kermadectonga2 (11 km); the other zones are "
                         "identical. off: v9's trench.")
    ap.add_argument("--mesh-file", default=None,
                    help="v12: use this quadrilateral mesh instead of building "
                         "one from SLAB (steps 1-2). A text file with one line "
                         "per unit source and 12 numbers: its 4 corners as "
                         "lon lat depth, going round the cell (e.g. "
                         "inputs_meshes/alaskaaleutians_quadrilateral_coors.dat). It must be "
                         "a structured mesh (rows down dip x columns along "
                         "strike, cells sharing their corners). Copied into "
                         "the example's inputs/geometry/.")
    ap.add_argument("--mesh-depth-units", default="auto", choices=["auto", "m", "km"],
                    help="v12, with --mesh-file: depth units of the file. auto "
                         "(default): metres if any |depth| > 200, else km; the "
                         "sign (negative down or positive down) is detected.")
    ap.add_argument("--variable-mu", default="off", choices=["on", "off"],
                    help="v11: off (default): every rate uses the constant "
                         "rigidity of 30 GPa (PTHA18's rate_annual). on: "
                         "step_total also runs step 7c, which adds PTHA18's "
                         "variable shear modulus rates (variable_mu_*, the "
                         "ones its headline maps use with HS) next to the "
                         "constant ones. Needs step 7b.")
    ap.add_argument("--rupture-size", default="rptha", choices=["rptha", "local"],
                    help="v10: how many cells each uniform-slip rupture gets. "
                         "rptha (default): rptha's and PTHA18's rule, one "
                         "block of cells per magnitude from the zone's mean "
                         "cell size. local: a block per placement, from the "
                         "real km of the cells there, so the rupture area "
                         "follows the scaling relation on a mesh whose cells "
                         "differ a lot (calabria2: 377 to 4042 km2). Not "
                         "PTHA18's procedure.")
    ap.add_argument("--convergence", default="bird", choices=["bird", "bird-griffin"],
                    help="which plate-boundary table step 3 reads, both in "
                         "from_scratch_v12/data/bird/. bird (default): Bird "
                         "(2003)'s public catalogue. bird-griffin: the table "
                         "PTHA18 used, Bird plus Jonathan Griffin's source-zone "
                         "traces with their own plate rates. They agree on the "
                         "zones where PTHA18 used Bird's own steps and can "
                         "differ a lot where it used Griffin's traces (eastern "
                         "Indonesia, New Guinea): see "
                         "html/docs/convergence_sources.html.")
    ap.add_argument("--segmented", default="false", choices=["true", "false"],
                    help="v9 (LEVEL 0). false (default): run the zone "
                         "unsegmented, exactly as v8 did -- the generated "
                         "example is then v8's pipeline (with v9's Berryman zone values). true: also split "
                         "the zone into along-strike segments and run the full "
                         "LEVEL 0 tree (unsegmented branch 0.5, union of "
                         "segments 0.5, the segments summing within that "
                         "branch). Boundaries from --segment-boundaries, "
                         "public data either way, so this stays "
                         "public-data-only under --ptha false. See "
                         "lib/segmentation.py.")
    ap.add_argument("--segment-boundaries", default="berryman",
                    choices=["berryman", "bird"],
                    help="v9 (LEVEL 0), with --segmented true: where the "
                         "segments end. berryman (default): at the trench end "
                         "points Berryman et al. (2015) Table 3.1 gives every "
                         "segment, the segments PTHA18 used (on PTHA18's own "
                         "meshes the boundaries land within 0-2 columns of "
                         "PTHA18's). bird: where Bird (2003)'s plate pair "
                         "changes along the trench (misses a segment Bird has "
                         "no plate for, e.g. Hikurangi).")
    args = ap.parse_args()

    # v12: an external mesh (checked here, before anything is written)
    mesh_name = None
    if args.mesh_file:
        if not os.path.exists(args.mesh_file):
            raise SystemExit(f"--mesh-file: {args.mesh_file} not found")
        sys.path.insert(0, HERE)
        from pyptha_v12 import mesh_file as _mesh_file
        _mesh_file.read_quadrilateral_mesh(args.mesh_file, depth_units=args.mesh_depth_units)
        mesh_name = os.path.basename(args.mesh_file)
    segmented = args.segmented == "true"
    clip = None
    if args.clip:
        try:
            clip = tuple(float(v) for v in args.clip.split(","))
        except ValueError:
            clip = ()
        if len(clip) != 4 or clip[0] >= clip[1] or clip[2] >= clip[3]:
            raise SystemExit("--clip must be LON_MIN,LON_MAX,LAT_MIN,LAT_MAX "
                             "with LON_MIN < LON_MAX and LAT_MIN < LAT_MAX")

    trench_ends = None
    if args.trench_ends:
        try:
            v = [float(q) for q in args.trench_ends.split(",")]
        except ValueError:
            v = []
        if len(v) != 4:
            raise SystemExit("--trench-ends must be LON1,LAT1,LON2,LAT2")
        trench_ends = ((v[0], v[1]), (v[2], v[3]))

    use_ptha = args.ptha == "true"

    region = resolve_region(args.name, args.id)
    ptha18_zone = resolve_ptha18_zone(args.zone, args.name, region["title"],
                                      allow_new_zone=not use_ptha)
    if clip is None and ptha18_zone in ZONE_DEFAULT_CLIP:
        clip = ZONE_DEFAULT_CLIP[ptha18_zone]
    if trench_ends is None and ptha18_zone in ZONE_DEFAULT_TRENCH_ENDS:
        trench_ends = ZONE_DEFAULT_TRENCH_ENDS[ptha18_zone]

    requested = args.folder or (
        re.sub(r"[^a-z0-9_]+", "_", args.name.lower()).strip("_") + "_v12")
    # --folder may be nested (e.g. examples/calabria2): the steps then climb
    # one more level to find the package (ROOT_UP), and the run and figure
    # labels use the last component only (NAME)
    requested = requested.replace("\\", "/").strip("/")
    folder = pick_folder_name(requested)
    example_dir = os.path.join(ROOT, folder)

    print("=" * 70)
    print(f"Generating '{folder}/' (v8) for {region['title']}")
    print(f"  SLAB2.0 prefix '{region['prefix']}', PTHA18 zone '{ptha18_zone}'")
    print(f"  --ptha={args.ptha}  ({'may read PTHA18 files (width, row count, scaling, '
          'comparisons)' if use_ptha else 'steps 1-7 and 9 read no PTHA18 or rptha file; only step 8 does'})")
    discretizer_note = {
        "optimal": "rptha's own mesh; repaired only if it has folded, "
                   "collapsed or overlapping cells",
        "lm": "rptha's multigrid Levenberg-Marquardt method exactly as "
              "PTHA18 ran it, no repairs",
        "mid": "rptha's older, deprecated eq_spacing method",
    }[args.discretizer]
    print(f"  --discretizer={args.discretizer}  ({discretizer_note})")
    print(f"  --convergence={args.convergence}  ("
          + ("Bird (2003)'s public catalogue" if args.convergence == "bird" else
             "Bird + Griffin's traces, the table PTHA18 used") + ")")
    note = convergence_source_note(ptha18_zone, args.convergence)
    if note:
        print(note)
    if clip:
        print(f"  --clip={clip}  (step 1 uses only this lon/lat window of "
              f"the SLAB raster)")
    if trench_ends:
        print(f"  --trench-ends={trench_ends}  (step 1 cuts the zone along the "
              f"down-dip lines through these two points)")
    print("=" * 70)

    report_geometry_inputs(ptha18_zone, use_ptha)

    mapping = {
        "FOLDER": folder,
        "NAME": folder.rsplit("/", 1)[-1],
        # the exact command line, shown at the top of RUN.html
        "GENERATE_CMD": html.escape(
            f"python {PKG}/generate.py "
            + " ".join(shlex.quote(a) for a in sys.argv[1:])),
        "ROOT_UP": ", ".join(['".."'] * (folder.count("/") + 1)),
        "ZONE": ptha18_zone,
        "PTHA18_ZONE": ptha18_zone,
        # v9: the generated steps import the engine from the package that
        # generated them. v8 hardcoded "from_scratch_v8" in every template, so
        # a v9 example would have silently run v8's engine and none of the
        # segmentation below would ever have loaded. Keeping these as
        # placeholders also makes a future v10 a one-line change.
        "PKG": PKG,
        "PYPKG": PYPKG,
        "LTPKG": LTPKG,
        "SEGMENTED": "True" if segmented else "False",
        "SEGMENT_BOUNDARIES": args.segment_boundaries,
        "TITLE": region["title"],
        "PREFIX": region["prefix"],
        "SCIBASE_ITEM": region["id"],
        "USE_PTHA": "True" if use_ptha else "False",
        "DISCRETIZER": repr(args.discretizer),
        "COLUMN_RULE": repr(args.columns),
        "RUPTURE_SIZE": repr(args.rupture_size),
        "CONVERGENCE": repr(args.convergence),
        # Manual-download fallback links (see step1_fetch_slab2.py.tmpl's
        # "If the automatic download fails" section). region["title"] is
        # this region's full ScienceBase title (e.g. "Kamchatka-Kuril
        # Islands-Japan"), the same string download_depth_grid() itself
        # searches on, so the link actually finds this zone's item.
        "SLAB2_REGION_PHRASE": urllib.parse.quote(region["title"]),
        "SLAB1_REGION_PREFIX": region["prefix"],
        "CLIP_BBOX": repr(clip) if clip else "None",
        "TRENCH_ENDS": repr(trench_ends) if trench_ends else "None",
        "TRENCH_RAMP": "3.0" if args.trench_ramp == "on" else "None",
        "VARIABLE_MU": "True" if args.variable_mu == "on" else "False",
        "MESH_FILE": repr(mesh_name) if mesh_name else "None",
        "MESH_DEPTH_UNITS": repr(args.mesh_depth_units),
    }

    for sub in ("steps", "data/slab1", "data/slab2", "data/gcmt", "inputs/geometry", "outputs"):
        os.makedirs(os.path.join(example_dir, sub), exist_ok=True)

    if mesh_name:
        # v12: the external mesh, copied so the example is self-contained
        import shutil
        shutil.copyfile(args.mesh_file, os.path.join(example_dir, "inputs", "geometry", mesh_name))
        print(f"  copied the mesh -> inputs/geometry/{mesh_name}")

    print()
    for fname in TEMPLATE_FILES:
        with open(os.path.join(TEMPLATES_DIR, fname + ".tmpl"),
                  "r", encoding="utf-8") as f:
            text = f.read()
        with open(os.path.join(example_dir, "steps", fname),
                  "w", encoding="utf-8") as f:
            f.write(render(text, mapping))
        print(f"  wrote steps/{fname}")

    with open(os.path.join(TEMPLATES_DIR, "RUN.html.tmpl"),
              "r", encoding="utf-8") as f:
        run_html = render(f.read(), mapping)
    with open(os.path.join(example_dir, "RUN.html"), "w", encoding="utf-8") as f:
        f.write(run_html)
    print("  wrote RUN.html")

    cutoff, dominant_segment = berryman_cutoff_km(ptha18_zone)
    coupling = berryman_coupling(ptha18_zone)
    b_anchor = berryman_b_anchor(ptha18_zone)
    mw_max_observed = berryman_mw_max_observed(ptha18_zone)
    cutoff_str = (f"{cutoff:g} km, derived from Table 3.1 (dominant segment "
                 f"'{dominant_segment}')" if cutoff is not None else
                 "**NOT AVAILABLE** -- no Berryman et al. (2015) mapping "
                 "for this zone; add it to `lib/berryman_params.py` "
                 "(step 1 stops without it)")
    mw_max_str = (f"{mw_max_observed:g}" if mw_max_observed is not None else
                 "**NOT AVAILABLE** -- no Berryman et al. (2015) mapping "
                 "for this zone; add it to `lib/berryman_params.py` "
                 "(step 1 stops without it)")
    if not use_ptha:
        width_str = "50 km (--ptha false: the general default)"
    else:
        try:
            width_str = f"{width_km(ptha18_zone):g} km"
        except SystemExit:
            width_str = "50 km (generic default -- no row in sourcezone_parameters.csv for this zone)"
    readme = f"""# {region['title']} from scratch (v10_q)

Generated by from_scratch_v12/generate.py for SLAB2.0/SLAB1.0 region
"{region['title']}" (ScienceBase item {region['id']}, prefix
"{region['prefix']}"), matched to PTHA18 source zone "{ptha18_zone}".

{"**Segmented (LEVEL 0).** Segment boundaries from " + ("Berryman et al. (2015) Table 3.1" if args.segment_boundaries == "berryman" else "Bird (2003) plate pairs") + ", PTHA18" + chr(39) + "s segment method; step 8 runs PTHA18" + chr(39) + "s own segmented model. See from_scratch_v12/README.md, What v9 adds." if segmented else "Unsegmented (LEVEL 0 weight 1). Zone coupling and mw_max_observed follow v9" + chr(39) + "s Berryman rules (from_scratch_v12/README.md, What v9 adds)."}

{"**Rupture size: local (v10).** Each uniform-slip rupture gets its own block of cells, chosen from the real km of the cells where it sits, so its area follows the scaling relation even where the cells differ a lot in size. Not PTHA18" + chr(39) + "s procedure (rptha: one block per magnitude). v10_q: each rupture" + chr(39) + "s conditional probability is also multiplied by q, the rupture-overlap correction. See from_scratch_v12/README.md, What v10_q changes and What v10 changes." if args.rupture_size == "local" else "Rupture size: rptha" + chr(39) + "s rule (one block of cells per magnitude), as PTHA18."}

{"**Variable shear modulus: on (v11).** step_total runs step 7c, which adds PTHA18" + chr(39) + "s variable_mu rates next to the constant-rigidity ones. See from_scratch_v12/README.md, What v11 changes." if args.variable_mu == "on" else "Shear modulus: constant 30 GPa (generated with --variable-mu off; step 7c only runs with --force)."}

{"**Geometry: external mesh (v12).** Steps 1-2 do not use SLAB: the unit sources are the cells of inputs/geometry/" + mesh_name + ", used as given (generate.py --mesh-file). See from_scratch_v12/README.md, What v12 changes." if mesh_name else "Geometry: SLAB (steps 1-2)."}

{"Trench ramp rule: on (v10_q): a trench end that climbs in depth like a ramp is trimmed before the local datum is built (step 1, TRENCH_RAMP_RISE_KM)." if args.trench_ramp == "on" else "Trench ramp rule: off (v9" + chr(39) + "s trench)."}

## What v8 changes, relative to v7

v8 builds the fault surface the way PTHA18 built it: depth contours measured
below the NEARBY trench (a local datum, not one number for the whole zone),
the trench itself as the 0 km line, and one unbroken contour every 5 km down
to the seismogenic cutoff. The mesh optimiser now reproduces rptha's R call
step for step, and --discretizer optimal keeps rptha's mesh unless it is
actually broken. Full list and evidence: from_scratch_v12/html/docs/index.html.

## What v11 changes (2026-10-06)

v11 = v10_q plus **step 7c, the variable shear modulus**: PTHA18's
`variable_mu` rates, the ones its headline hazard maps use with the HS
scenarios. Rigidity grows with depth (10 GPa near the trench, 67 GPa from
35 km, PTHA18's fit to Bilek & Lay 1999); every scenario keeps its slip and
gets its real magnitude (`variable_mu_Mw`); the difference enters LEVEL 3 as
an error of the catalogue magnitudes, which changes the branch weights and
so the rates. Steps 1-7b are unchanged (identical files to v10_q); step 7c
only adds `outputs/*_variable_mu*.csv` and the `variable_mu_*` columns of
both HS/VAUS `summary.csv`. Checked against PTHA18's published files to
7e-13 (from_scratch_v12/validation/validate_v11.py). See
from_scratch_v12/README.md, "What v11 changes", and html/docs/v11.html.

## What v6 changes, relative to v5

v6 keeps v5's templates, mesh and engine (same `--discretizer optimal`
default, same three fidelity fixes) unchanged, and adds one new step:
**step 7b generates HETEROGENEOUS-slip (HS) AND variable-area-uniform-slip
(VAUS) variants** of every FAUS placement step 7 already built. HS uses the
SFFM generator (`pyptha_v12/stochastic_slip.py`, ported in v5 but never
wired into a step until now); VAUS is derived from each HS field exactly as
rptha's `make_all_earthquake_events.R` does. This is additive, not a
replacement:

- LEVELs 0-5 (the logic tree, rates, percentiles) still run on FAUS (Fixed
  Area Uniform Slip) only -- identical numbers to v5 for the same zone.
- step 7b's HS/VAUS fields are written to `outputs/hs_slip_fields/` and
  `outputs/vaus_slip_fields/`, and are never read back into the rate/logic-
  tree computation.
- step9_report.py labels every rupture figure/table FAUS, HS or VAUS
  explicitly.

## What v5 changes, relative to v4

v5 keeps v4's templates and mesh (same `--discretizer optimal` default)
unchanged, and fixes three engine bugs an independent fidelity audit found
against the rptha R source: spherical-vs-ellipsoidal geodesy in
unit_sources.py (feeds dip/width), `computational_increment` silently
defaulting to 0.01 instead of PTHA18's official 0.02, and a dormant
degeneracy-check baseline mismatch in the LEVEL 4 edge-multiplier fit. See
from_scratch_v12/html/docs/code_map.html for the full detail. v5 runs its own forked copy
of the engine (pyptha_v12/, python_logic_tree_v12/), so v1-v4 are unaffected.

## What v3 changes, relative to v2

v2 still read three "declared" literature numbers (coupling, b_anchor,
mw_max_observed) from PTHA18's own internal sourcezone_parameters.csv, and
derived the seismogenic depth cutoff + datum by measuring against PTHA18's
own published unit-source table. Both only work for a zone PTHA18 already
modelled. v3 reads all of these from the actual primary source PTHA18 itself
cites for them: Berryman et al. (2015), "The GEM Faulted Earth Subduction
Interface Characterisation Project, Version 2.0" (see
from_scratch_v12/lib/berryman_params.py for the full citation and table values).

| | v2 (PTHA18's own files) | v3 (Berryman et al. 2015 directly) |
|---|---|---|
| depth cutoff | read from PTHA18's published unit_source_statistics_*.nc | {cutoff_str} |
| coupling (min/pref/max) | sourcezone_parameters.csv | **{coupling}** |
| b (min/pref/max) | sourcezone_parameters.csv | **{b_anchor}** |
| mw_max_observed | sourcezone_parameters.csv | {mw_max_str} |
| unit-source width | {width_str}, from the CSV (unchanged in v3: a resolution parameter, not a physical "declared" value) | same |

These will not exactly match PTHA18's own declared numbers -- PTHA18
area-weights its segment merges and sometimes hand-adjusts mw_max_observed
against the GCMT catalogue in ways this module does not replicate. See
berryman_params.py's docstring for how close the match is, per zone.

## What v3 still cannot reproduce

PTHA18's along-strike cell boundaries. They come from a hand-edited
shapefile that was never published, and the parameter that would otherwise
control them (approx_unit_source_length) is documented by PTHA18 as ignored.
See from_scratch_v12/html/docs/official.html for the evidence.

So judge this mesh on **total area and mean dip**, which step 2 compares
directly against the published values (when available -- see below). Those
are the only two mesh quantities the moment balance consumes.

## Run one at a time, from ptha18_logic_tree_test/

    .venv/Scripts/python.exe {folder}/steps/step1_fetch_slab2.py
    .venv/Scripts/python.exe {folder}/steps/step2_build_grid.py
    .venv/Scripts/python.exe {folder}/steps/step3_convergence.py
    .venv/Scripts/python.exe {folder}/steps/step4_fetch_gcmt.py
    .venv/Scripts/python.exe {folder}/steps/step5_subset_gcmt.py
    .venv/Scripts/python.exe {folder}/steps/step6_write_input.py
    .venv/Scripts/python.exe {folder}/steps/step7_run.py
    .venv/Scripts/python.exe {folder}/steps/step7b_stochastic_slip.py  # HS variants, optional
    .venv/Scripts/python.exe {folder}/steps/step7c_variable_mu.py      # v11 variable shear modulus, after 7b
    .venv/Scripts/python.exe {folder}/steps/step8_official.py   # optional, needs R
    .venv/Scripts/python.exe {folder}/steps/step9_report.py     # -> report.html

## Or all in one go

    .venv/Scripts/python.exe {folder}/steps/step_total.py
"""
    with open(os.path.join(example_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print("  wrote README.md")

    print(f"\nDone. Start with:")
    print(f"  .venv/Scripts/python.exe {folder}/steps/step1_fetch_slab2.py")


if __name__ == "__main__":
    sys.exit(main())
