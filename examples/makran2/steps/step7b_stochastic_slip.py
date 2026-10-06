"""STEP 7b - Heterogeneous-slip (HS) AND variable-area-uniform-slip (VAUS)
variants of step 7's FAUS ruptures.

This step is ADDITIVE, not a replacement for step 7
-----------------------------------------------------
Step 7 already ran the full logic tree (LEVELs 0-5) on FAUS (Fixed Area
Uniform Slip) ruptures: every event in the moment balance and every number
in outputs/rate_curves.csv has ONE slip value spread uniformly over its
rupture footprint. That is what PTHA18 itself uses for LEVELs 0-5, and step
7b does not touch it -- rate_curves.csv, exceedance_rate_percentiles.csv
and logic_tree_branches_*.csv are unaffected by this step, before or after
it runs.

What this step does instead: pyptha_v12/stochastic_slip.py ports rptha's
SFFM generator (sffm_fit_simulate_earthquake.R, the S_NCF method of Davies
et al. 2015) -- PTHA18's own method for turning a uniform-slip rupture into
a HETEROGENEOUS one (slip concentrated in "asperities", the same total
slip and hence the same Mw). It was ported in v5 but never wired into a
step; this step wires it in as an illustrative, clearly-separated output:
more realistic tsunami initial conditions for a handful of sampled
ruptures, not a new rate model. See pyptha_v12/stochastic_slip.py's module
docstring for what was and was not ported (the sub_sample_size refinement
path and the full sffm_make_events_on_discretized_source wrapper are out of
scope; the spectral algorithm itself is faithfully ported).

v10 / v10_q: the FAUS "parents" are regenerated with the input's
events.rupture_size, the same call step 7 makes, so with "local" they are
the cell-by-cell ruptures and each HS field's peak is drawn inside its
cell-by-cell parent. The SFFM itself is rptha's, unchanged, and it still
assumes equal cells inside a footprint: it uses ONE cell spacing for the
whole footprint, the length and width of the peak cell (num_row/num_col =
Strasser W/L over the peak cell's size; reg_par = corner wavenumbers x the
peak cell's size). Where the cells inside a rupture differ (up to 2.3 times
in length on hellenic_west2_v9 and 2.0 on calabria2 inside a Mw 8.0
rupture; 1.17 on Kurils-Japan) the slip pattern is stretched or squeezed in
km. README, Known limitations (v10_q).

v10_q (2026-10-06): HS and VAUS rates, as PTHA18 computes them
(compute_rates_all_sources.R 1551-1600, fixed shear modulus; see
pyptha_v12/hs_vaus_rates.py). Each field shares the rate of the FAUS rupture
it was drawn from (step 7's outputs/scenario_rates_makran2.csv): a field
with peak slip above 7.5 times the scaling relation's mean slip for that Mw
gets rate 0; the others are weighted by their peak-slip rank in the family
(PTHA18's DART-calibrated table, recovered in
data/ptha18_peak_slip_quantile_weights.csv) and the weights of a family sum
to 1. Both summary.csv files gain faus_event_id, above_peak_slip_limit,
weight_in_family and the parent's rate columns times that weight. Without
step 7's file (step 7b run first) the rate columns are left out. These
rates feed nothing in LEVELs 0-5.

VAUS (Variable Area Uniform Slip) is a third PTHA18 rupture type -- uniform
slip, but with the rupture area itself varied rather than fixed by the
scaling relation alone. Read directly from rptha's own source
(make_all_earthquake_events.R lines ~255-345): VAUS is NOT an independent
stochastic model. rptha derives it from the ALREADY-GENERATED HS table --
same Mw, target_lon/target_lat, peak_slip_ind, kcx/kcy per row (that script
literally copies the whole stochastic_events_table and only overwrites
event_index_string/event_slip_string) -- by (1) taking the rectangular
region that "just contains" the HS footprint's own alongstrike/downdip
range, and (2) setting a single uniform slip on that rectangle so
sum(slip*area) matches the HS field's own sum(slip*area) exactly (same
seismic moment, same Mw). Confirmed against kermadectonga2's official
netCDF pair: VAUS has exactly the same 44,685 rows as HS, identical Mw/
target_lon/target_lat row-for-row, and the VAUS event_index_string is the
rectangular superset of the matching HS row's footprint (spot-checked: HS
row 5000's footprint {118, 124} -> VAUS {118, 121, 124}, the alongstrike
span that contains both). This step reproduces that exact derivation
inline, in the same loop that generates each HS field, using the array
that field's generator already holds in memory (see main()'s "--- VAUS ---"
block) -- a separate step would need to either re-run the stochastic
generator (different RNG draw order, no longer reproducible field-for-
field) or persist every HS field's full footprint to disk first (tens of
thousands of files), so it is computed here instead, right after its
matching HS field, and written to its own output.

Replicating rptha's per-event randomness, including footprint size
--------------------------------------------------------------------
The goal is to replicate what rptha itself does, in Python, on this run's
own mesh -- as closely as this package's scope allows, not to invent a new
roughness parameter. rptha does NOT reuse one fixed corner-wavenumber /
footprint-size pair for every HS realisation at a magnitude, and does NOT
confine an HS realisation to its parent FAUS rupture's own footprint. Its
wrapper (sffm_make_events_on_discretized_source) draws each event's OWN
random rupture length/width and corner wavenumbers kcx/kcy from a
log-normal regression against Mw (Davies et al. 2015, ported as
pyptha_v12.stochastic_slip.sffm_make_random_lwkc_function -- see that
module's docstring for the exact coefficients), converts L/W to a number of
unit sources, and places that rectangle on the FULL zone mesh (not just the
parent rupture's cells) via rectangle_on_grid (also ported, see
pyptha_v12.stochastic_slip.rectangle_on_grid), centred on the parent
rupture's peak-slip cell. A large random draw can spill well beyond the
parent footprint; a small one shrinks well inside it.

This step reproduces all three pieces of rptha's per-realisation
randomness: every HS realisation gets its own random footprint SIZE
(L/W), its own random corner wavenumber (kcx/kcy), and its own random
footprint CENTRE -- rptha's vary_peak_slip_location, sampled uniformly
from a window around the parent FAUS rupture's own peak-slip cell. The
window is NOT "+/- L/2, W/2 of this magnitude's typical rupture size" (a
formula sffm_fit_simulate_earthquake.R does contain, but only as a fallback
for when the caller leaves vary_peak_slip_alongstrike_downdip_extent NULL
-- rptha's real driver, make_all_earthquake_events.R lines ~195-198, NEVER
leaves it NULL: it always passes the parent FAUS rupture's own real
alongstrike/downdip index range explicitly, c(range(alongstrike_number
[usi]), range(downdip_number[usi])), so that fallback formula is dead code
in production and is NOT what this step ports). This step instead uses the
parent rupture's own bounding box directly, matching what R's driver
actually executes. Confirmed to matter incrementally: with a footprint
fixed at the parent FAUS rupture's own size and centre (this step's
earliest behaviour), mean peak slip was measured systematically below
PTHA18's own published catalogue, worsening at higher Mw; randomising the
footprint SIZE closed most of that gap; adding the random CENTRE on top
(first with the wrong Mw-only window, later corrected to the parent
rupture's own real extent) closed the remainder, landing this run's mean
peak slip within ordinary realisation-to-realisation scatter of the
official mean at nearly every magnitude (sometimes above, sometimes below,
as expected of two independent random samples of the same distribution).

PTHA18 also PUBLISHES the full official HS result:
``all_stochastic_slip_earthquake_events_<zone>.nc`` (and the matching VAUS
one). This step reads nothing of PTHA18: it only saves, per magnitude, the
few numbers a statistical comparison needs from each of its own fields
(outputs/hs_slip_fields/comparison_features.npz and the same for VAUS; see
lib/hs_official_compare.py's field_features). Step 8, the official step,
compares them with PTHA18's catalogues when those are downloaded
(step8_official.py --download-hs). It can only be a STATISTICAL comparison
(spectral shape, typical peak slip, spatial concentration), not a
field-by-field one: SFFM and the L/W/kcx/kcy draw are both random, and the
published table does not record the seed that produced any given row.

How many HS fields this generates: rptha's own real ratio, not a guess
-------------------------------------------------------------------------
Checked directly against kermadectonga2's official catalogue: rptha
generates EXACTLY 15 heterogeneous-slip realisations per FAUS rupture, at
every magnitude from Mw 7.2 to 9.5 without a single exception (verified: 27
magnitudes, ratio 15.00 at every one) -- except where a magnitude has very
few FAUS ruptures (Mw >= 9.6 at kermadectonga2, where 8/1/1 ruptures exist),
in which case rptha floors the total at 200 HS events for that magnitude
rather than only generating 15/30/15. This step reproduces that exact rule
(REALISATIONS_PER_FAUS_PLACEMENT=15, MIN_HS_EVENTS_PER_MAGNITUDE=200) by
default -- see main() -- so the total field count this run produces should
land close to the official catalogue's own size for a zone PTHA18 modelled
(kermadectonga2: ~39,400 here vs. 44,685 official; the gap is this run's
own from-scratch mesh having 213 unit sources against the official 216, not
a different rule).

What this script does, in order
--------------------------------
  1. rebuilds the SAME mesh step 7 ran on (reads input_makran2_scratch.json,
     same geometry.mode="shapefile" block, same discretizer -- so the unit
     sources here are identical to step 7's, not independently re-meshed)
  2. regenerates the FAUS event table with pyptha_v12.events, the same call
     run_logic_tree.py makes internally (the engine does not persist the
     per-event unit-source list, only aggregated rate outputs, so this is
     a re-derivation from the same geometry + Mmin/Mmax/dMw, not a re-read)
  3. for EVERY magnitude in the event table's range, and EVERY one of that
     magnitude's actual FAUS rupture placements (there can be from ~1 up to
     ~200+, depending on magnitude and mesh size), generates
     REALISATIONS_PER_FAUS_PLACEMENT heterogeneous-slip realisations,
     preserving total slip (so Mw is unchanged) -- floored at
     MIN_HS_EVENTS_PER_MAGNITUDE total events for that magnitude, exactly
     as rptha does. --max-placements-per-mw / --max-fields-per-mw (see
     --help) cap this for a faster, approximate run; the default is
     uncapped, matching rptha's own volume.
  4. derives the matching VAUS field from that same HS field (see the VAUS
     section above), in the same loop iteration
  5. writes outputs/hs_slip_fields/summary.csv and
     outputs/vaus_slip_fields/summary.csv (one row per field each, same
     schema as rptha's own published netCDF columns: Mw, target_lon/lat,
     peak_slip_downdip_ind/alongstrike_ind, physical_corner_wavenumber_x/y,
     event_index_string, event_slip_string -- the packed footprint/slip
     format get_unit_source_indices_in_event decodes, "-"-joined 1-based
     subfault_number / "_"-joined slip -- plus a few convenience columns
     rptha computes on demand rather than storing, like peak_slip_m and
     n_unit_sources). Every field generated gets a full row (no sampling);
     this replaces the earlier version's sample_fields/ directory, which
     only kept 30 fields' full footprints on disk.
  6. writes outputs/hs_slip_fields/comparison_features.npz and
     outputs/vaus_slip_fields/comparison_features.npz: for every field, its
     peak slip, active cells, concentration and 16-bin radial spectrum,
     grouped by magnitude. Step 8 compares ALL of them with ALL of PTHA18's
     official events at each magnitude, for HS and for VAUS separately. The
     VAUS comparison is not redundant with the HS one: VAUS being a
     deterministic function of the HS field next to it means an HS
     comparison passing does NOT by itself prove the DERIVATION RULE (the
     bounding-box superset + moment-preserving uniform slip) is correctly
     implemented -- only a direct comparison against the official VAUS
     catalogue can catch a bug in that rule specifically (e.g. an off-by-one
     in the bounding box, or an area computed with the wrong cells).

Known measured bias in VAUS's official comparison (mesh, not a rule bug)
--------------------------------------------------------------------------
Run against kermadectonga2's official VAUS catalogue (44,685 rows):
footprint size (mean_active_cells) matches the official catalogue closely
at every magnitude (within a few percent, both directions) -- confirming
the bounding-box "just contains" rule itself is implemented correctly.
Mean peak slip (= the single uniform value spread over that footprint),
however, runs CONSISTENTLY above the official mean at all 25 magnitudes:
+1% to +14%, growing with Mw (worse than the ~+3% average bias already
present in the HS comparison feeding it). Since VAUS's slip is exactly
sum(HS slip*area) / sum(footprint area), and the footprint areas match,
the extra inflation traces back to moment computed on THIS RUN'S OWN mesh
(213 unit sources for kermadectonga2) rather than the official one (216
unit sources) -- a from-scratch mesh built from the same SLAB2.0 contours
via this package's own discretizer will not have bit-identical cell areas
to PTHA18's own mesh, and that difference compounds through VAUS's own
division by total footprint area on top of whatever it already was in the
HS input. This is a mesh-fidelity limitation (same category as this
package's other from-scratch-vs-official mesh differences -- see
from_scratch_v12/html/docs/code_map.html), not a bug in the VAUS derivation rule itself: the rule was
verified structurally (bounding-box superset, moment-preserving slip) in
this module's own test, independent of which mesh feeds it.

Run from ptha18_logic_tree_test/, after step7_run.py. By default this
generates tens of thousands of fields (matching rptha's own volume, see
above), which can take 15-30+ minutes on a large zone -- progress is
printed per magnitude. Use the CLI flags below for a faster, capped run:
    .venv/Scripts/python.exe examples/makran2/steps/step7b_stochastic_slip.py
    .venv/Scripts/python.exe examples/makran2/steps/step7b_stochastic_slip.py --max-placements-per-mw 20 --max-fields-per-mw 300
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))  # ptha18_logic_tree_test/

sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))  # helper modules
from pyptha_v12 import events, unit_sources as us  # noqa: E402
from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import grid_cache  # noqa: E402
from pyptha_v12 import stochastic_slip as ss  # noqa: E402
from pyptha_v12.scaling import M0_2_Mw  # noqa: E402
from pyptha_v12 import hs_vaus_rates  # noqa: E402
import hs_official_compare as hoc  # noqa: E402
import official_geometry_params as ogp  # noqa: E402

INPUT_JSON = os.path.join(EXAMPLE, "inputs", "input_makran2_scratch.json")
OUT_DIR = os.path.join(EXAMPLE, "outputs", "hs_slip_fields")
VAUS_OUT_DIR = os.path.join(EXAMPLE, "outputs", "vaus_slip_fields")

ZONE = "makran2"
PTHA18_ZONE = "makran2"
# What step 8 needs to compare these fields with PTHA18's catalogues.
FEATURES_NAME = "comparison_features.npz"

# rptha's OWN rule, verified against kermadectonga2's official catalogue
# (see the module docstring's "How many HS fields" section): 15 HS
# realisations per FAUS rupture placement, floored at 200 total events for
# any magnitude with too few placements to reach that on its own. This is
# not a guess -- every one of 27 checked magnitudes matched exactly.
REALISATIONS_PER_FAUS_PLACEMENT = 15
MIN_HS_EVENTS_PER_MAGNITUDE = 200

# Scaling relation (for sffm_make_random_lwkc_function's L/W/kcx/kcy draw)
# and shear modulus (for the per-realisation moment rescale, slip * cell
# area * mu) are read PER ZONE from sourcezone_parameters.csv inside main()
# -- make_all_earthquake_events.R:98-105/config.R:15,19 do the same, and a
# hardcoded 'Strasser'/3e10 here would silently diverge from R on any zone
# whose CSV row differs (none of the zones this project currently runs
# do, but outerrisesunda/arutrough elsewhere in the catalogue would).

RNG_SEED = 1234


def _column_rule_fp(column_rule, discretizer):
    """Fingerprint field for the v8.2 column rule: none for rptha's own
    "trench" rule (or a discretiser that ignores the rule), so such a mesh
    keeps exactly the v8 fingerprint. Must match step 2's."""
    if discretizer != "optimal" or column_rule == "trench":
        return {}
    return {"column_rule": column_rule}


def build_grid_from_geometry(geo):
    """Reproduce run_logic_tree.py's build_grid() for mode='shapefile'.

    Duplicated rather than imported: importing run_logic_tree.py as a
    module would re-run its own top-level matplotlib/Agg setup and argparse
    CLI, for one function. This mirrors it exactly (same discretizer
    dispatch, same defaults) so the mesh here matches step 7's.
    """
    if geo.get("mode") == "mesh_file":
        # v12: the external mesh, read as the engine reads it
        from pyptha_v12 import mesh_file
        path = geo["mesh_file"]
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        return mesh_file.read_quadrilateral_mesh(
            path, depth_units=geo.get("mesh_depth_units", "auto"))

    import geopandas as gpd

    shp = geo["shapefile"]
    if not os.path.isabs(shp):
        shp = os.path.join(ROOT, shp)
    gdf = gpd.read_file(shp)
    gdf["level"] = gdf["level"].astype(float)
    contours = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "LineString":
            coords = np.array(geom.coords)
        else:
            parts = sorted(geom.geoms, key=lambda g: len(g.coords))
            coords = np.array(parts[-1].coords)
        contours.append((row["level"], coords[:, :2]))

    kwargs = dict(
        desired_unit_source_length=geo["desired_unit_source_length"],
        desired_unit_source_width=geo.get("desired_unit_source_width"),
        n_downdip=geo.get("n_downdip"), seed=geo.get("seed", 1234),
        min_downdip=geo.get("min_downdip"))
    discretizer = geo.get("discretizer", "lm")
    # v8.2: how many columns the "optimal" mesh gets, "trench" (rptha's
    # rule, default) or "average" (see contour_discretisation.
    # discretized_source_from_contours_optimal). Inputs without the key
    # (v8.1 and older) keep rptha's trench rule.
    column_rule = geo.get("column_rule", "trench")

    # Reuse the grid step 2 built and checked when it was built with these
    # same settings, exactly as run_logic_tree.build_grid() does.
    cache_npy = geo.get("unit_source_grid_npy")
    if cache_npy:
        if not os.path.isabs(cache_npy):
            cache_npy = os.path.join(ROOT, cache_npy)
        fp = grid_cache.fingerprint(
            shp, method=discretizer,
            desired_unit_source_length=kwargs["desired_unit_source_length"],
            desired_unit_source_width=kwargs["desired_unit_source_width"],
            n_downdip=kwargs["n_downdip"], min_downdip=kwargs["min_downdip"],
            seed=kwargs["seed"], **_column_rule_fp(column_rule, discretizer))
        cached = grid_cache.load_if_match(cache_npy, fp)
        if cached is not None:
            print(f"      mesh: reusing step 2's grid "
                  f"({os.path.relpath(cache_npy, ROOT)}, fingerprint matches)")
            return cached
        print("      mesh: step 2's saved grid does not match this input's "
              "meshing fields (or is missing) -- rebuilding")

    if discretizer == "optimal":
        return cd.discretized_source_from_contours_optimal(
            contours, column_rule=column_rule, **kwargs)
    elif discretizer == "mid":
        return cd.discretized_source_from_contours_mid(
            contours, desired_unit_source_length=kwargs["desired_unit_source_length"],
            desired_unit_source_width=kwargs["desired_unit_source_width"],
            n_downdip=kwargs["n_downdip"], min_downdip=kwargs["min_downdip"])
    return cd.discretized_source_from_contours_orthogonal(contours, **kwargs)


def full_mesh_grids(stats):
    """Dense (n_downdip x n_alongstrike) arrays over the WHOLE source zone
    mesh -- length_km, width_km, subfault_number -- indexed by
    (downdip_number-1, alongstrike_number-1). This is rptha's own
    ``dx``/``dy`` matrices in sffm_make_events_on_discretized_source (there:
    dx=length, dy=width, built the same way, one entry per unit source,
    NaN/0 where the mesh has no cell -- a rectangular source zone has none,
    so this package's meshes fill every cell).

    Needed because rptha's HS footprint is NOT confined to the parent FAUS
    rupture's own placement: sffm_make_events_on_discretized_source draws
    each event's own rupture length/width and places it via
    rectangle_on_grid against the FULL zone grid, centred on the FAUS
    rupture's peak-slip cell -- so a large random draw can spill well
    beyond that one rupture's original footprint, and a small one can
    shrink well inside it.

    subfault_number_grid holds rptha's own 1-based unit-source id at each
    (row, col) -- the id get_unit_source_indices_in_event decodes out of
    event_index_string (rupture_events.R line ~400: dash-separated
    unit_source_statistics row numbers). 0 marks a mesh cell with no unit
    source. Used to WRITE event_index_string in the same format.
    """
    dd = stats["downdip_number"].astype(int)
    ask = stats["alongstrike_number"].astype(int)
    n_dd = int(dd.max())
    n_as = int(ask.max())
    length_km = np.zeros((n_dd, n_as))
    width_km = np.zeros((n_dd, n_as))
    subfault_number_grid = np.zeros((n_dd, n_as), dtype=int)
    length_km[dd - 1, ask - 1] = stats["length"]
    width_km[dd - 1, ask - 1] = stats["width"]
    subfault_number_grid[dd - 1, ask - 1] = stats["subfault_number"].astype(int)
    return length_km, width_km, subfault_number_grid


_DEG2RAD = np.pi / 180.0


def _spherical_centroid_lonlat(lon_deg, lat_deg):
    """Geographic (spherical) centroid of a set of lon/lat points -- rptha's
    ``geomean`` (make_all_earthquake_events.R:171-180: ``event_location =
    geomean(unit_source_locations, ...)``, imported from geosphere, not a
    coordinate-wise arithmetic mean). geosphere's geomean averages the
    points as UNIT VECTORS on the sphere and converts the mean vector back
    to lon/lat, which is the standard way to average geographic coordinates
    correctly (handles antimeridian wraparound; an arithmetic mean of raw
    lon/lat does not, and is wrong in general even away from the
    antimeridian since lon/lat is not a flat coordinate system)."""
    lon = np.asarray(lon_deg, dtype=float) * _DEG2RAD
    lat = np.asarray(lat_deg, dtype=float) * _DEG2RAD
    x = (np.cos(lat) * np.cos(lon)).mean()
    y = (np.cos(lat) * np.sin(lon)).mean()
    z = np.sin(lat).mean()
    lon_c = np.arctan2(y, x)
    lat_c = np.arctan2(z, np.hypot(x, y))
    return float(lon_c / _DEG2RAD), float(lat_c / _DEG2RAD)


def _nearest_unit_source_to(lon_deg, lat_deg, stats):
    """Index (into `stats`) of the unit source whose centroid is closest,
    great-circle distance, to (lon_deg, lat_deg) -- rptha's
    ``target_unit_source_index = which.min(distHaversine(..., event_
    location))`` (sffm_fit_simulate_earthquake.R:1127-1131)."""
    lon = stats["lon_c"] * _DEG2RAD
    lat = stats["lat_c"] * _DEG2RAD
    lon0, lat0 = lon_deg * _DEG2RAD, lat_deg * _DEG2RAD
    # Haversine great-circle distance (radius cancels for argmin).
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    return int(np.argmin(a))


def rupture_to_template(all_eq, ev_i, stats):
    """The (down-dip x along-strike) uniform-slip template for one FAUS
    rupture, PLUS its peak-slip cell in full-mesh (0-based) coordinates --
    the centre rptha's random footprint gets placed around.

    Same footprint construction as
    pyptha_v12/examples/stochastic_slip_ruptures.py's rupture_to_template
    (applied to this zone's real mesh instead of a demo planar one), with
    the full-mesh peak location added since step 7b's HS footprint is no
    longer confined to this same block -- see full_mesh_grids's docstring.

    Peak-slip cell: NOT the array bounding-box midpoint of the rupture's
    (downdip, alongstrike) index range -- that can land on a cell with no
    real unit source at all when the footprint is not a filled rectangle
    (common near zone/segment edges). rptha instead takes the SPHERICAL
    centroid of the rupture's own unit sources (geomean) and picks the
    real unit source nearest that centroid
    (sffm_fit_simulate_earthquake.R:1127-1131) -- reproduced here via
    _spherical_centroid_lonlat + _nearest_unit_source_to.
    """
    idx = all_eq["event_indices"][ev_i]
    uniform_slip = float(all_eq["slip"][ev_i])
    dd = stats["downdip_number"][idx].astype(int)
    ask = stats["alongstrike_number"][idx].astype(int)
    dd0, as0 = dd.min(), ask.min()
    nd, na = dd.max() - dd0 + 1, ask.max() - as0 + 1
    tg = np.zeros((nd, na))
    tg[dd - dd0, ask - as0] = uniform_slip

    cen_lon, cen_lat = _spherical_centroid_lonlat(
        stats["lon_c"][idx], stats["lat_c"][idx])
    nearest = _nearest_unit_source_to(cen_lon, cen_lat, stats)
    peak_row_full = int(stats["downdip_number"][nearest]) - 1
    peak_col_full = int(stats["alongstrike_number"][nearest]) - 1

    return tg, uniform_slip, idx, (dd0, as0, nd, na), (peak_row_full, peak_col_full)


def add_rates(rows, kind, parent, peak_slip, parent_Mw, all_eq,
              scaling_relation, shear_modulus):
    """v10_q: PTHA18's rate of each HS/VAUS field (hs_vaus_rates), added to
    its summary row in place. Returns a one-line note for the log.

    `parent` is each field's 0-based row in all_eq, `peak_slip` its exact
    peak slip (the CSV keeps 3 decimals), in generation order."""
    path = os.path.join(EXAMPLE, "outputs", f"scenario_rates_{ZONE}.csv")
    if not os.path.exists(path):
        return (f"{kind}: no rates ({os.path.basename(path)} not found; "
                f"run step7_run.py first)")
    with open(path, newline="") as fh:
        faus = list(csv.DictReader(fh))
    area = np.array([float(r["area_km2"]) for r in faus])
    if area.size != all_eq["area"].size or not np.allclose(area, all_eq["area"], rtol=1e-6):
        return (f"{kind}: no rates (step 7's ruptures are not the ones "
                f"regenerated here; re-run step7_run.py)")
    rate_cols = [c for c in faus[0] if c.startswith("rate_")]
    weight, above = hs_vaus_rates.family_weights(
        parent, peak_slip, parent_Mw, kind, mu=shear_modulus,
        relation=scaling_relation)
    for row, ev_i, w, a in zip(rows, parent, weight, above):
        row["above_peak_slip_limit"] = int(a)
        row["weight_in_family"] = f"{w:.6g}"
        for c in rate_cols:
            row[c] = f"{w * float(faus[ev_i][c]):.6g}"
    lost = sum(float(faus[i]["rate_mean"]) for i in set(parent)
               if weight[np.asarray(parent) == i].sum() == 0)
    return (f"{kind}: {int(above.sum())} of {len(rows)} field(s) above "
            f"PTHA18's peak-slip limit (rate 0); FAUS rate left with no "
            f"field under the limit: {lost:.3g}/yr")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate heterogeneous-slip (HS) realisations for "
                    "every magnitude's FAUS ruptures, using rptha's own "
                    "15-per-placement rule by default (uncapped).")
    ap.add_argument("--max-placements-per-mw", type=int, default=None,
                     help="cap on how many of a magnitude's real FAUS "
                          "placements get HS realisations at all (default: "
                          "no cap -- every placement). Lowering this speeds "
                          "up a run at the cost of matching rptha's own "
                          "field count less closely.")
    ap.add_argument("--max-fields-per-mw", type=int, default=None,
                     help="cap on total HS fields generated per magnitude, "
                          "applied AFTER the 15-per-placement/200-floor "
                          "rule (default: no cap). Use this for a quick, "
                          "approximate run without touching the placement "
                          "cap above.")
    return ap.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("STEP 7b - Heterogeneous-slip (HS): every magnitude, every FAUS placement")
    print("=" * 70)
    if args.max_placements_per_mw is not None or args.max_fields_per_mw is not None:
        print(f"  CAPPED run: max_placements_per_mw="
              f"{args.max_placements_per_mw}, max_fields_per_mw="
              f"{args.max_fields_per_mw} (approximate -- will not match "
              f"rptha's own field count as closely as an uncapped run)")

    if not os.path.exists(INPUT_JSON):
        raise SystemExit(f"missing {INPUT_JSON}\nRun step6_write_input.py first.")

    with open(INPUT_JSON) as f:
        cfg = json.load(f)

    geo = cfg["geometry"]
    if geo["mode"] not in ("shapefile", "mesh_file"):
        raise SystemExit(
            f"step 7b only supports geometry.mode='shapefile' (this run's "
            f"from-scratch mesh) or 'mesh_file' (v12, an external mesh); got "
            f"{geo['mode']!r}")

    print("\n  [1/4] rebuilding step 7's mesh")
    grid = build_grid_from_geometry(geo)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    print(f"      n_unit_sources = {stats['subfault_number'].size}")

    # The FULL zone mesh, dense -- rptha's own dx/dy matrices (see
    # full_mesh_grids's docstring for why an HS footprint needs the whole
    # zone, not just one FAUS rupture's own placement).
    length_grid, width_grid, subfault_number_grid = full_mesh_grids(stats)
    area_grid = length_grid * width_grid
    n_dd_full, n_as_full = length_grid.shape

    print("\n  [2/4] regenerating the FAUS event table (same call step 7's "
          "engine makes internally)")
    ev = cfg["events"]
    # mu/relation: from the input's events block (step 6 writes both), the
    # same values step 7's engine uses. An older input without them falls
    # back to sourcezone_parameters.csv, as the engine does.
    if "scaling_relation" in ev and "shear_modulus_Pa" in ev:
        scaling_relation, shear_modulus = ev["scaling_relation"], float(ev["shear_modulus_Pa"])
    else:
        print("  note: the input has no events.scaling_relation/shear_modulus_Pa, "
              "so they are read from PTHA18's sourcezone_parameters.csv; step 6 "
              "always writes both")
        scaling_relation, shear_modulus = ogp.scaling_relation_and_shear_modulus(ZONE)
    # Built here (not at module level) because it needs scaling_relation,
    # which is only known once ZONE is resolved inside main().
    # clip at 2 sd: sffm_make_events_on_discretized_source's default
    # (clip_random_parameters_at_2sd = TRUE, sffm_fit_simulate_earthquake.R
    # 1119, 1199-1201), which PTHA18's make_all_earthquake_events.R keeps
    lwkc_fun = ss.sffm_make_random_lwkc_function(
        clip_random_parameter_ranges_to_2sd=True, relation=scaling_relation)
    # v10: the same rupture-size rule as step 7 (rptha's when unset)
    all_eq = events.get_all_earthquake_events(
        stats, Mmin=ev["Mmin"], Mmax=ev["Mmax"], dMw=ev["dMw"],
        source_zone_name=ZONE, mu=shear_modulus, relation=scaling_relation,
        rupture_size=ev.get("rupture_size", "rptha"))
    all_eq["Mw"] = np.round(all_eq["Mw"], 3)
    n_events = all_eq["Mw"].size
    print(f"      n_events (FAUS) = {n_events}")

    uniq_mw = np.unique(all_eq["Mw"])
    if uniq_mw.size == 0:
        raise SystemExit("no FAUS events generated; nothing to generate HS for")
    print(f"\n  [3/4] {uniq_mw.size} magnitude(s), "
          f"{uniq_mw.min():.2f}-{uniq_mw.max():.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VAUS_OUT_DIR, exist_ok=True)
    summary_rows = []
    summary_rows_vaus = []
    synthetic_grids_by_mw = {}
    vaus_grids_by_mw = {}
    # v10_q, for the rates: each field's parent row in all_eq and its peak
    # slip, in generation order
    field_parent, hs_peak, vaus_peak = [], [], []

    print(f"  corner wavenumbers: rptha's own random L/W/kcx/kcy draw per "
          f"HS realisation (relation={scaling_relation!r}) -- see "
          f"pyptha_v12/stochastic_slip.py's sffm_make_random_lwkc_function")

    n_fields_total_estimate = 0
    print(f"\n  [4/4] generating HS realisations "
          f"({REALISATIONS_PER_FAUS_PLACEMENT} per FAUS placement, floored "
          f"at {MIN_HS_EVENTS_PER_MAGNITUDE}/magnitude -- rptha's own rule)")
    for mw in uniq_mw:
        idx_for_mw = np.where(all_eq["Mw"] == mw)[0]
        n_placements_total = idx_for_mw.size
        if args.max_placements_per_mw is not None \
                and n_placements_total > args.max_placements_per_mw:
            # Evenly spaced through the placement list (not random and not
            # just the first N) so the kept subset still spans the full
            # along-strike/down-dip range of placements at this magnitude.
            keep = np.round(np.linspace(
                0, n_placements_total - 1, args.max_placements_per_mw)).astype(int)
            idx_for_mw = idx_for_mw[keep]

        # rptha's own rule (make_all_earthquake_events.R 186-194): every
        # placement of this magnitude gets max(15, ceiling(200 / n)) fields,
        # n = the number of placements at this magnitude (all of them, even
        # when --max-placements-per-mw keeps only a subset here).
        base_each = max(REALISATIONS_PER_FAUS_PLACEMENT,
                        int(np.ceil(MIN_HS_EVENTS_PER_MAGNITUDE / max(n_placements_total, 1))))
        realisations_per_placement = np.full(idx_for_mw.size, base_each, dtype=int)

        if args.max_fields_per_mw is not None:
            planned_total = int(realisations_per_placement.sum())
            if planned_total > args.max_fields_per_mw:
                scale = args.max_fields_per_mw / planned_total
                realisations_per_placement = np.maximum(
                    1, np.round(realisations_per_placement * scale)).astype(int)

        # rptha rescales every generated field to match Mw's seismic moment
        # EXACTLY (sffm_fit_simulate_earthquake.R lines ~1395-1403:
        # initial_moment = sum(slip*dx*dy*1e6*mu); slip = slip/initial_moment
        # * desired_M0) -- not the plain sum sffm_simulate itself preserves
        # (which only equals the moment when every cell has equal area, not
        # true on a real mesh). desired_M0 depends only on Mw, so it is
        # computed once per magnitude here, same as rptha's caller does.
        desired_M0 = M0_2_Mw(float(mw), inverse=True)

        synthetic_grids_by_mw.setdefault(mw, [])
        vaus_grids_by_mw.setdefault(mw, [])
        n_fields_this_mw = 0
        kc_note = ""
        for p, ev_i in enumerate(idx_for_mw):
            tg, uniform_slip, unit_idx, (dd0, as0, nd, na), \
                (parent_peak_row, parent_peak_col) = rupture_to_template(
                    all_eq, int(ev_i), stats)

            # target_lon/target_lat: written straight from x$target_location
            # (sffm_fit_simulate_earthquake.R:1489-1490), which IS event_
            # location, the SPHERICAL centroid (geomean) of the parent FAUS
            # placement's own unit sources -- make_all_earthquake_events.R:
            # 171-180 computes event_location = geomean(...), then passes it
            # as target_location = event_location into
            # sffm_make_events_on_discretized_source (line 200). Not a
            # coordinate-wise arithmetic mean of lon_c/lat_c (wrong in
            # general for lon/lat, which is not a flat coordinate system --
            # see _spherical_centroid_lonlat's own docstring). Constant
            # across every HS/VAUS realisation drawn from this placement --
            # confirmed against the official netCDF (same target_lon/lat for
            # every row sharing one uniform_event_row, even as the HS
            # footprint itself varies row to row).
            target_lon, target_lat = _spherical_centroid_lonlat(
                stats["lon_c"][unit_idx], stats["lat_c"][unit_idx])

            # Allowed range for the realisation's own peak-slip cell.
            # rptha's real driver ALWAYS passes vary_peak_slip_alongstrike_
            # downdip_extent explicitly (make_all_earthquake_events.R:195-
            # 198: c(range(alongstrike_number[usi]), range(downdip_number
            # [usi])) -- the exact index span of THIS parent FAUS rupture's
            # own unit sources), so the "L/2, W/2 of the scaling relation's
            # typical size" fallback formula (sffm_fit_simulate_earthquake.R
            # lines ~1136-1156) is a branch R's own production code never
            # takes -- confirmed by direct reading, since the argument is
            # never left NULL at the call site. The window here is simply
            # the parent rupture's own bounding box (dd0, as0, nd, na from
            # rupture_to_template), already computed above; no separate
            # per-magnitude "typical size" window is needed. 0-based here;
            # R's range() is 1-based but the span is identical.
            row_lo, row_hi = int(dd0 - 1), int(dd0 - 1 + nd - 1)
            col_lo, col_hi = int(as0 - 1), int(as0 - 1 + na - 1)

            for r_i in range(int(realisations_per_placement[p])):
                # one stream per (magnitude, placement, realisation): the
                # old seed RNG_SEED + p*1000 + r_i + int(mw*100000) repeated
                # across magnitudes (placement p at Mw m = p+10 at m-0.1)
                rng = np.random.default_rng(
                    [RNG_SEED, int(round(mw * 100)), int(p), int(r_i)])

                # rptha's vary_peak_slip_location (sffm_fit_simulate
                # _earthquake.R lines ~1235-1250): each realisation's own
                # peak-slip cell is drawn uniformly from the window computed
                # above, not fixed at the parent FAUS rupture's peak cell
                # every time.
                peak_row_full = int(rng.integers(row_lo, row_hi + 1))
                peak_col_full = int(rng.integers(col_lo, col_hi + 1))

                # rptha's own per-EVENT random draw (Davies et al. 2015):
                # every realisation gets its own rupture length/width and
                # corner wavenumber, not a value shared across the whole
                # magnitude or reused from the parent FAUS footprint -- see
                # pyptha_v12/stochastic_slip.py's sffm_make_random_lwkc_function
                # and this module's docstring for why this matters.
                lwkc = lwkc_fun(np.array([float(mw)]), rng=rng)
                slip_L, slip_W = float(lwkc["L"][0]), float(lwkc["W"][0])
                phys_kcx, phys_kcy = float(lwkc["kcx"][0]), float(lwkc["kcy"][0])

                peak_length_km = float(length_grid[peak_row_full, peak_col_full])
                peak_width_km = float(width_grid[peak_row_full, peak_col_full])

                # rptha's num_L/num_W: desired physical L/W -> number of
                # unit sources, clamped to the FULL zone's own extent (not
                # the parent FAUS footprint) at the peak-slip cell's own
                # length/width (sffm_fit_simulate_earthquake.R lines
                # ~1273-1276).
                num_col = int(min(n_as_full, max(1, round(slip_L / peak_length_km))))
                num_row = int(min(n_dd_full, max(1, round(slip_W / peak_width_km))))

                # "expand_length_if_width_limited='random'" (rptha's own
                # default): if the desired width exceeds what the zone can
                # ever hold, trade some of that missing width for more
                # length, on a coin flip -- see this module's docstring.
                max_available_width = n_dd_full * peak_width_km
                if slip_W > max_available_width:
                    width_deficit = num_row * peak_width_km / slip_W
                    if width_deficit < 1.0 and rng.uniform() >= 0.5:
                        alt_length = slip_L / width_deficit
                        num_col = int(min(n_as_full, max(
                            1, round(alt_length / peak_length_km))))

                # rptha's rectangle_on_grid: place (num_row x num_col) on
                # the FULL zone grid, centred on THIS REALISATION's own
                # randomly-drawn peak-slip cell (see peak_row_full/
                # peak_col_full above -- rptha's vary_peak_slip_location),
                # clamped at the zone boundary -- may spill well beyond (or
                # shrink well inside) the parent FAUS footprint, AND be
                # centred somewhere other than the parent rupture's own
                # peak cell.
                s_row, e_row, s_col, e_col = ss.rectangle_on_grid(
                    n_dd_full, n_as_full, num_row, num_col,
                    peak_row_full, peak_col_full,
                    randomly_vary_around_target_centre=True, rng=rng)

                window_area_km2 = area_grid[s_row:e_row + 1, s_col:e_col + 1]
                template = np.zeros((e_row - s_row + 1, e_col - s_col + 1))
                template[peak_row_full - s_row, peak_col_full - s_col] = 1.0

                reg_par = (phys_kcx * peak_length_km, phys_kcy * peak_width_km)
                if r_i == 0:
                    kc_note = (f"e.g. L/W {slip_L:.0f}/{slip_W:.0f} km -> "
                               f"{num_row}x{num_col} cells, kcx/kcy="
                               f"{phys_kcx:.5f}/{phys_kcy:.5f} -> reg_par "
                               f"{reg_par[0]:.3f}/{reg_par[1]:.3f}")

                # rptha redraws until the first and last row and column of
                # the non-zero-slip window all have some slip
                # (sffm_fit_simulate_earthquake.R 1347-1364)
                for _try in range(10000):
                    fake = ss.sffm_simulate(reg_par, template, rng=rng)
                    if (np.any(fake[0, :] > 0) and np.any(fake[-1, :] > 0)
                            and np.any(fake[:, 0] > 0) and np.any(fake[:, -1] > 0)):
                        break

                # Moment rescale, replacing sffm_simulate's own sum-preserving
                # rescale with rptha's area-weighted one (see desired_M0
                # comment above). window_area_km2 * 1e6 = area in m^2,
                # matching rptha's dx*dy*1e+06 (dx, dy there are also in km).
                initial_moment = float((fake * window_area_km2 * 1e6 * shear_modulus).sum())
                if initial_moment <= 0:
                    # Degenerate draw (all-zero field): fall back to a flat
                    # field over the window rather than dividing by zero.
                    het = np.where(window_area_km2 > 0, 1.0, 0.0)
                    initial_moment = float((het * window_area_km2 * 1e6 * shear_modulus).sum())
                    het = het / initial_moment * desired_M0
                else:
                    het = fake / initial_moment * desired_M0

                synthetic_grids_by_mw[mw].append(het)
                n_fields_this_mw += 1

                # rptha's own event_index_string/event_slip_string format
                # (rupture_events.R's get_unit_source_indices_in_event,
                # reversed): "-"-joined 1-based subfault_number per active
                # cell, "_"-joined slip (4 significant figures, matching
                # make_all_earthquake_events.R line ~317's signif(slip, 4)).
                # This is what rptha itself writes to the HS/VAUS netCDF --
                # not a summary derived FROM it -- so this is now the
                # authoritative record of each field's footprint, in place
                # of the small illustrative sample_fields/ CSVs the earlier
                # version of this step wrote for only 30 fields.
                active = het > 0
                active_subfaults = subfault_number_grid[s_row:e_row + 1, s_col:e_col + 1][active]
                active_slip = het[active]
                hs_index_string = "-".join(str(int(v)) for v in active_subfaults) + "-"
                hs_slip_string = "_".join(f"{v:.4g}" for v in active_slip) + "_"

                # peak_slip_downdip_ind/alongstrike_ind: THIS realisation's
                # own randomly-drawn peak cell (1-based) -- confirmed
                # against the official netCDF to vary row-to-row within one
                # FAUS placement's group of realisations, unlike target_lon/
                # target_lat which stays fixed at the parent's centroid.
                peak_dd_ind = peak_row_full + 1
                peak_as_ind = peak_col_full + 1

                field_parent.append(int(ev_i))
                # v11: peak slip as PTHA18 reads it, the largest value of
                # event_slip_string (4 significant digits), so that step 7c
                # ranks the fields exactly as here
                hs_peak.append(max(float(f"{v:.4g}") for v in active_slip))
                summary_rows.append({
                    "rupture_type": "HS", "Mw": f"{mw:.2f}",
                    "placement": p + 1, "realisation": r_i + 1,
                    "faus_event_id": int(ev_i) + 1,
                    "target_lon": f"{target_lon:.6f}", "target_lat": f"{target_lat:.6f}",
                    "peak_slip_downdip_ind": peak_dd_ind,
                    "peak_slip_alongstrike_ind": peak_as_ind,
                    "physical_corner_wavenumber_x": f"{phys_kcx:.6g}",
                    "physical_corner_wavenumber_y": f"{phys_kcy:.6g}",
                    "n_unit_sources": int(active.sum()),
                    "peak_slip_m": f"{het.max():.3f}",
                    "total_slip_m": f"{het.sum():.3f}",
                    "uniform_slip_m_FAUS_equivalent": f"{uniform_slip:.3f}",
                    "event_index_string": hs_index_string,
                    "event_slip_string": hs_slip_string,
                })

                # --- VAUS (Variable Area Uniform Slip) ---
                # rptha derives VAUS from the ALREADY-GENERATED HS table
                # (make_all_earthquake_events.R lines ~285-325), not from an
                # independent stochastic model: same Mw/target_lon/target_
                # lat/peak_slip_ind/kcx/kcy as the HS row it comes from
                # (that script copies the whole stochastic_events_table and
                # only overwrites event_index_string/event_slip_string --
                # confirmed against the official netCDF: identical Mw and
                # target_lon/lat row-for-row between the two published
                # tables). Only the footprint and slip differ:
                #   1. the RECTANGULAR region that "just contains" the HS
                #      footprint's own alongstrike/downdip range (R lines
                #      ~285-299) -- may include cells the HS field itself
                #      left at zero slip, but NEVER a cell with no real unit
                #      source at all (see the vaus_subfaults filter below --
                #      R's own vari_unif_uss can't include one either, since
                #      it filters the real unit_source_statistics table,
                #      never slices a dense array with placeholder zeros);
                #   2. a single uniform slip value on that whole rectangle,
                #      set so sum(slip*area) equals the HS field's own
                #      sum(slip*area) exactly (R lines ~301-309) -- same
                #      seismic moment, same Mw, by construction.
                vaus_dd = s_row + np.where(active)[0]
                vaus_as = s_col + np.where(active)[1]
                vaus_dd_lo, vaus_dd_hi = int(vaus_dd.min()), int(vaus_dd.max())
                vaus_as_lo, vaus_as_hi = int(vaus_as.min()), int(vaus_as.max())
                vaus_sub_block = subfault_number_grid[
                    vaus_dd_lo:vaus_dd_hi + 1, vaus_as_lo:vaus_as_hi + 1]
                vaus_area_block = area_grid[
                    vaus_dd_lo:vaus_dd_hi + 1, vaus_as_lo:vaus_as_hi + 1]
                hs_slip_x_area = float((het * area_grid[
                    s_row:e_row + 1, s_col:e_col + 1]).sum())
                vaus_uniform_slip = hs_slip_x_area / float(vaus_area_block.sum())

                # subfault_number_grid is 0 at any (downdip, alongstrike)
                # mesh cell with no real unit source (full_mesh_grids's own
                # docstring) -- true at zone edges where the mesh is not a
                # dense rectangle (this run: 213 real unit sources against
                # 216 officially, so some bounding boxes DO cross a gap).
                # R's own vari_unif_uss (make_all_earthquake_events.R lines
                # ~293-298) can never include such a cell: it is built by
                # filtering unit_source_statistics (which only ever has rows
                # for real unit sources), not by slicing a dense array, so
                # it has no placeholder-zero concept to filter in the first
                # place. Filtering here reproduces that same result on this
                # port's dense-grid representation -- without it, a bounding
                # box crossing a mesh gap would write a fabricated "0" into
                # event_index_string and inflate n_unit_sources/total_slip_m
                # by the number of such gaps.
                vaus_subfaults = vaus_sub_block[vaus_sub_block > 0]
                vaus_index_string = "-".join(str(int(v)) for v in vaus_subfaults) + "-"
                vaus_slip_string = "_".join(
                    f"{vaus_uniform_slip:.4g}" for _ in vaus_subfaults) + "_"

                vaus_peak.append(float(f"{vaus_uniform_slip:.4g}"))
                summary_rows_vaus.append({
                    "rupture_type": "VAUS", "Mw": f"{mw:.2f}",
                    "placement": p + 1, "realisation": r_i + 1,
                    "faus_event_id": int(ev_i) + 1,
                    "target_lon": f"{target_lon:.6f}", "target_lat": f"{target_lat:.6f}",
                    "peak_slip_downdip_ind": peak_dd_ind,
                    "peak_slip_alongstrike_ind": peak_as_ind,
                    "physical_corner_wavenumber_x": f"{phys_kcx:.6g}",
                    "physical_corner_wavenumber_y": f"{phys_kcy:.6g}",
                    "n_unit_sources": int(vaus_subfaults.size),
                    "peak_slip_m": f"{vaus_uniform_slip:.3f}",
                    "total_slip_m": f"{vaus_uniform_slip * vaus_subfaults.size:.3f}",
                    "uniform_slip_m_FAUS_equivalent": f"{uniform_slip:.3f}",
                    "event_index_string": vaus_index_string,
                    "event_slip_string": vaus_slip_string,
                })

                # Dense grid for the official comparison below, same shape
                # convention as het (only cells with an actual unit source
                # count as "active" -- subfault_number_grid is 0 off-mesh).
                vaus_grid = np.where(vaus_sub_block > 0, vaus_uniform_slip, 0.0)
                vaus_grids_by_mw[mw].append(vaus_grid)

        n_fields_total_estimate += n_fields_this_mw
        print(f"      Mw {mw:.2f}: {idx_for_mw.size}/{n_placements_total} "
              f"FAUS placement(s) -> {n_fields_this_mw} HS field(s) ({kc_note})")

    # v10_q: PTHA18's HS/VAUS rates from step 7's FAUS rates
    parent_mw = all_eq["Mw"][np.asarray(field_parent, dtype=int)]
    print("\n  HS/VAUS rates (PTHA18's rule, pyptha_v12/hs_vaus_rates.py):")
    for rows, kind, peaks in ((summary_rows, "HS", hs_peak),
                              (summary_rows_vaus, "VAUS", vaus_peak)):
        print("    " + add_rates(rows, kind, field_parent, peaks, parent_mw,
                                 all_eq, scaling_relation, shear_modulus))

    summary_path = os.path.join(OUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    vaus_summary_path = os.path.join(VAUS_OUT_DIR, "summary.csv")
    with open(vaus_summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows_vaus[0].keys()))
        w.writeheader()
        w.writerows(summary_rows_vaus)

    print(f"\n  generated {len(summary_rows)} HS field(s) and "
          f"{len(summary_rows_vaus)} VAUS field(s) across "
          f"{len(synthetic_grids_by_mw)} magnitude(s)")
    print(f"  HS summary (every field, full footprint as event_index_string/"
          f"event_slip_string) -> {summary_path}")
    print(f"  VAUS summary (derived from the HS table above, same rptha "
          f"rule as make_all_earthquake_events.R) -> {vaus_summary_path}")

    # What step 8 needs to compare these fields with PTHA18's own HS and
    # VAUS catalogues (see lib/hs_official_compare.py's field_features):
    # one row per field, grouped by magnitude. Nothing of PTHA18 is read
    # here; the comparison itself is step 8's.
    for label, grids_by_mw, out_dir in (("HS", synthetic_grids_by_mw, OUT_DIR),
                                        ("VAUS", vaus_grids_by_mw, VAUS_OUT_DIR)):
        mw_col, feats = [], []
        for mw, grids in grids_by_mw.items():
            if grids:
                feats.append(hoc.field_features(grids))
                mw_col.extend([float(mw)] * len(grids))
        path = os.path.join(out_dir, FEATURES_NAME)
        np.savez_compressed(path, mw=np.asarray(mw_col, dtype=float),
                            features=np.vstack(feats))
        print(f"  {label} comparison features ({len(mw_col)} fields) -> {path}")
        stale = os.path.join(out_dir, "official_comparison.csv")
        if os.path.exists(stale):
            os.remove(stale)  # an earlier layout wrote the comparison here

    print("\n  Reminder: outputs/rate_curves.csv and every LEVEL 0-5 number")
    print("  remain FAUS-only, exactly as step 7 wrote them. These HS/VAUS")
    print("  fields are an illustrative addition, not read back into the "
          "logic tree.")
    print("\nNext:  step8_official.py (optional) or step9_report.py")


if __name__ == "__main__":
    sys.exit(main())
