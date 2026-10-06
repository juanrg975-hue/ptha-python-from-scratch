"""LEVEL 0 segmentation: split a source zone into along-strike segments.

This is the one part of the PTHA18 logic tree that from_scratch_v12 did not
reproduce ("Segmented zones (LEVEL 0 with segments) are not reproduced; every
zone is run unsegmented"). The ENGINE could already do it -- pyptha_v12/
logic_tree.py has the LEVEL 0 weights and the LEVEL 5 cross-segment copula,
and python_logic_tree_v12/run_logic_tree.py accepts an `alongstrike_slice` per
segment. What was missing was a step that decides WHERE the segment boundaries
are, so step6 always wrote `"segments": {}`.

This module supplies that decision, from public data only.

---------------------------------------------------------------------------
The --ptha false rule
---------------------------------------------------------------------------

The whole point of `--ptha false` is that nothing PTHA18 produced is allowed to
feed the result; PTHA18's own files are read only by step 8, which COMPARES.
PTHA18's own boundary indices

    rptha/.../DATA/SOURCEZONE_PARAMETERS/sourcezone_parameters.csv
        segment_boundary_alongstrike_index_lower / _upper

are therefore never used to run a segment. Instead (lib/segmentation.py
``zone_segments``, generate.py --segment-boundaries):

  * ``berryman_segments()`` -- PUBLIC, the DEFAULT. Berryman et al. (2015)
                             Table 3.1 gives the trench end points of every
                             segment (Left_E_LONG, Left_N_LAT, Right_E_LONG,
                             Right_N_LAT; berryman_params.
                             BERRYMAN_SEGMENT_TRENCH). Each end point is placed
                             on the mesh's top edge and a boundary falls at
                             the column edge nearest to the end point two
                             neighbours share. These are the segments PTHA18
                             used: on PTHA18's own meshes the rule lands within
                             0 to 2 columns of PTHA18's indices on the 18
                             boundaries the two share (validate_v9.py
                             berryman_segments).
  * ``bird_segments()``    -- PUBLIC, the alternative (--segment-boundaries
                             bird). Derives the boundaries from Bird (2003)
                             PB2002_steps.dat, which labels every step of every
                             plate boundary with the PLATE PAIR it separates.
  * ``ptha18_segments()``  -- PTHA18's own indices, read from
                             sourcezone_parameters.csv and rescaled onto this
                             mesh, for the comparisons printed by
                             validate_v9.py. Never used to run a segment.
  * ``ptha18_official_segments_block()`` -- PTHA18's own segments on PTHA18's
                             own mesh, for step 8's official segmented run
                             (the comparison target).

Segment PARAMETERS (coupling, b-value, Mw_max_observed) come from the same
Berryman table: a Berryman segment IS a Table 3.1 row. So under --ptha false
a segment gets Berryman for both its extent and its physics, and reads
nothing of PTHA18's.

(Until 2026-09-30 the boundaries came from Bird, on the belief that Table 3.1
had names but no geography. It has both; Appendix A defines the end-point
columns.)

---------------------------------------------------------------------------
How the Bird derivation works (--segment-boundaries bird)
---------------------------------------------------------------------------

Bird (2003) splits each plate boundary into ~5800 short steps. Every step
carries the two plates it separates ("TO/PA" = Tonga plate under-thrust by
Pacific) and a boundary class ("SUB" = subduction). Berryman placed most of
its segment ends where these plate pairs change (Appendix A), so the two
usually agree.

    1. Keep Bird's subduction steps (class SUB) near the zone.
    2. Assign every along-strike COLUMN of the mesh to the plate pair of the
       nearest such step.
    3. A run of consecutive columns sharing a plate pair is one segment.
    4. Drop runs shorter than ``min_columns`` (stray matches at the ends of the
       trench, where a neighbouring boundary can be the closest thing).

Its limit: a segment Berryman separates by a plate Bird does not have gets no
boundary. kermadectonga2's Hikurangi is PA/HF, the Hikurangi forearc of
Wallace et al. (2004); Bird has only TO/PA and KE/PA along that trench, so
Bird gives two segments where Berryman and PTHA18 give three.

---------------------------------------------------------------------------
Why fractional boundaries, never raw indices
---------------------------------------------------------------------------

PTHA18's indices are columns of PTHA18's mesh. A `--ptha false` run builds its
OWN mesh, which generally has a different column count (kermadectonga2: 73
scratch columns vs 72 official). Reusing the raw integers would silently shift
every boundary. ``ptha18_segments()`` therefore converts each boundary to a
FRACTIONAL along-strike position and re-applies it to the mesh actually in
hand, reporting the shift that conversion caused.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BIRD_RAW_ZIP = os.path.join(_HERE, os.pardir, "data", "bird",
                             "PB2002_steps.dat.txt.zip")

# Bird's class for a subduction step. PTHA18's segmented zones are all
# subduction interfaces, so the other convergent classes (OCB/CCB) would only
# add noise from neighbouring structures.
_SUBDUCTION_CLASS = "SUB"


# ---------------------------------------------------------------------------
# Bird (2003) steps, with the plate pair kept
# ---------------------------------------------------------------------------

def load_bird_subduction_steps():
    """Bird (2003) subduction steps as (plate_pair, mid_lon, mid_lat) arrays.

    lib/bird_convergence.load_bird_raw_segments() parses the same file but
    discards the plate-pair field, which is the single thing this module needs,
    so the few lines are repeated here rather than changing a v8 function that
    the convergence step depends on.

    Longitudes come back in [0, 360) to match the meshes, which cross the
    dateline on several zones.
    """
    with zipfile.ZipFile(_BIRD_RAW_ZIP) as zf:
        with zf.open(zf.namelist()[0]) as f:
            lines = f.read().decode("utf-8", "replace").splitlines()

    pairs, lons, lats = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) < 15:
            continue
        if parts[-1].strip(":*") != _SUBDUCTION_CLASS:
            continue
        try:
            lon1, lat1, lon2, lat2 = (float(parts[2]), float(parts[3]),
                                      float(parts[4]), float(parts[5]))
        except ValueError:
            continue
        # parts[1] is the plate pair, with a leading ':' on continuation lines.
        pairs.append(parts[1].lstrip(":"))
        lons.append(((lon1 + lon2) / 2.0) % 360.0)
        lats.append((lat1 + lat2) / 2.0)

    return (np.asarray(pairs, dtype=object),
            np.asarray(lons, dtype=float),
            np.asarray(lats, dtype=float))


# ---------------------------------------------------------------------------
# mesh -> per-column plate pair
# ---------------------------------------------------------------------------

def column_centres(grid):
    """Top-edge centre (lon in [0,360), lat) of every along-strike column.

    ``grid`` is the unit-source lattice used everywhere else in this package,
    shaped (ndip+1, 3, nstrike+1). Row 0 is the top (trench) edge, which is
    what Bird's steps trace.
    """
    top = np.asarray(grid)[0]
    lon = ((top[0, :-1] + top[0, 1:]) / 2.0) % 360.0
    lat = (top[1, :-1] + top[1, 1:]) / 2.0
    return lon, lat


def assign_columns_to_plate_pairs(grid, max_distance_deg=3.0):
    """Label each along-strike column with the plate pair of its nearest step.

    Distances are computed in degrees with the longitude difference scaled by
    cos(latitude), the same cheap local approximation lib/bird_convergence.py
    uses for matching; the trench is a curve and we only need the NEAREST step,
    so a full great-circle distance would not change the assignment.

    Columns whose nearest subduction step is further than ``max_distance_deg``
    are labelled None: that is a stretch of the mesh Bird does not call a
    subduction boundary at all.
    """
    pairs, blon, blat = load_bird_subduction_steps()
    clon, clat = column_centres(grid)

    labels, distances = [], []
    for i in range(clon.size):
        # wrap the longitude difference into [-180, 180] so a column just east
        # of the dateline is not reported as 359 degrees from a step just west.
        dlon = (blon - clon[i] + 180.0) % 360.0 - 180.0
        dlon *= np.cos(np.radians(clat[i]))
        d = np.hypot(dlon, blat - clat[i])
        j = int(np.argmin(d))
        labels.append(pairs[j] if d[j] <= max_distance_deg else None)
        distances.append(float(d[j]))

    return labels, np.asarray(distances)


def _safe_name(plate_pair):
    """Bird's plate pair as a name usable in a JSON key, path and NetCDF name.

    Bird writes the pair with a slash whose direction says which plate goes
    under ("TO/PA", "NZ\\SA"). That is meaningful but unusable in a filename,
    and run_logic_tree.py builds source names as ``<zone>_<segment>``, so the
    separator is dropped and the two plate codes are simply joined in order:
    "TO/PA" -> "to_pa". The direction is preserved in ``plate_pair`` alongside.
    """
    return plate_pair.replace("\\", "/").replace("/", "_").lower()


def _runs(labels):
    """Consecutive runs of an equal label, as (label, first, last) 1-based."""
    out = []
    for i, lab in enumerate(labels):
        if out and out[-1][0] == lab:
            out[-1][2] = i + 1
        else:
            out.append([lab, i + 1, i + 1])
    return [(lab, a, b) for lab, a, b in out]


def bird_segments(grid, min_columns=5, max_distance_deg=3.0, log=print):
    """Segment a mesh from Bird (2003) plate pairs alone. PUBLIC DATA ONLY.

    Returns a list of dicts with ``plate_pair`` and ``alongstrike_slice``
    ([first, last], 1-based inclusive, matching run_logic_tree.py).

    Runs shorter than ``min_columns`` are merged into the longer neighbour
    rather than kept. Two reasons: at the ends of a trench the nearest Bird
    step can belong to an adjoining boundary, and a very short run is usually a
    sliver plate rather than a seismological segment. The default 5 columns is
    about 250 km at the 50 km unit-source length used throughout -- below that a
    segment has too few earthquakes of its own for the LEVEL 3 Bayesian update
    to say anything, which is the point at which segmenting stops being
    informative. On sunda2 this merges Bird's 3-column BU/AU into SU/AU and
    yields 4 segments, the same count PTHA18 uses.

    A zone Bird does not split returns a single segment spanning the mesh; the
    caller should then run it unsegmented (PTHA18 gives the unsegmented branch
    weight 1.0 when there are no segments).
    """
    labels, distances = assign_columns_to_plate_pairs(
        grid, max_distance_deg=max_distance_deg)
    n = len(labels)

    runs = [r for r in _runs(labels) if r[0] is not None]
    if not runs:
        log("    Bird has no subduction step near this mesh: no segmentation")
        return []

    # absorb short runs into the longer neighbour
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, (lab, a, b) in enumerate(runs):
            if (b - a + 1) >= min_columns:
                continue
            prev_len = (runs[k - 1][2] - runs[k - 1][1] + 1) if k > 0 else -1
            next_len = (runs[k + 1][2] - runs[k + 1][1] + 1) if k + 1 < len(runs) else -1
            if prev_len >= next_len and k > 0:
                runs[k - 1] = (runs[k - 1][0], runs[k - 1][1], b)
            elif k + 1 < len(runs):
                runs[k + 1] = (runs[k + 1][0], a, runs[k + 1][2])
            else:
                break
            runs.pop(k)
            changed = True
            break

    # a mesh column that matched nothing still has to belong somewhere, or the
    # segments would not tile the zone and the union-of-segments branch would
    # be missing moment. Stretch the outermost runs to the ends of the mesh.
    runs[0] = (runs[0][0], 1, runs[0][2])
    runs[-1] = (runs[-1][0], runs[-1][1], n)
    for k in range(len(runs) - 1):
        # close any gap between neighbours at their midpoint
        if runs[k][2] + 1 < runs[k + 1][1]:
            mid = (runs[k][2] + runs[k + 1][1]) // 2
            runs[k] = (runs[k][0], runs[k][1], mid)
            runs[k + 1] = (runs[k + 1][0], mid + 1, runs[k + 1][2])

    # A plate pair can reappear further along the trench with a different pair
    # in between: on southamerica, Bird's NZ\SA runs from lat -45 to -3 but the
    # Altiplano sliver NZ\AP is embedded inside it around lat -21..-16. Those
    # are two DISJOINT stretches of mesh, so they must stay two segments -- a
    # segment in PTHA18 is a contiguous along-strike slice, and
    # run_logic_tree.py's alongstrike_slice cannot express a gap. Give the
    # repeats distinct names so nothing downstream collides on the key.
    seen = {}
    segs = []
    for lab, a, b in runs:
        seen[lab] = seen.get(lab, 0) + 1
        name = _safe_name(lab)
        if seen[lab] > 1:
            name = f"{name}{seen[lab]}"
        segs.append({"plate_pair": lab, "segment_key": name,
                     "alongstrike_slice": [a, b]})

    log(f"    Bird (2003) plate pairs split {n} columns into {len(segs)} "
        f"segment(s), median match distance {np.median(distances):.2f} deg")
    for s in segs:
        a, b = s["alongstrike_slice"]
        log(f"      {s['segment_key']:<10} columns {a}-{b}  ({b - a + 1} columns)")
    if len(segs) == 1:
        log("    -> one plate pair only: this zone is NOT segmented from Bird")
    return segs


# ---------------------------------------------------------------------------
# Berryman et al. (2015)'s segmentation (the default, public)
# ---------------------------------------------------------------------------

def _haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def trench_position(grid, lon, lat, per_column=20):
    """Where a point lies along the mesh's top edge, in columns.

    Returns (t, distance_km): t runs from 0 at the first node of the top edge
    to n_columns at the last, and is the position of the top-edge point
    nearest to (lon, lat), sampled ``per_column`` times per column.
    """
    top = np.asarray(grid)[0]
    tlon = np.degrees(np.unwrap(np.radians(top[0])))
    tlat = top[1]
    n = tlon.size - 1
    t = np.linspace(0.0, n, n * per_column + 1)
    dlon = np.interp(t, np.arange(n + 1), tlon)
    dlat = np.interp(t, np.arange(n + 1), tlat)
    d = _haversine_km(dlon, dlat, lon, lat)
    j = int(np.argmin(d))
    return float(t[j]), float(d[j])


def column_edge_position(grid, lon, lat):
    """Which column edge (a down-dip line of the mesh, top to bottom) passes
    closest to a point. Returns (edge index 0..n_columns, distance_km).

    v9: for a zone whose ends were cut along the down-dip lines through
    Berryman et al. (2015)'s points (step 1 TRENCH_ENDS, hellenic2), the
    boundary between its segments is placed the same way. Berryman's points
    can lie far down dip of the mesh's top edge (hellenic2: 165-211 km), and
    the nearest TOP-EDGE point (trench_position) then lands up to ~100 km
    along strike from the down-dip line through the point."""
    g = np.asarray(grid, dtype=float)
    k = np.cos(np.radians(lat))
    best = (np.inf, 0)
    for i in range(g.shape[2]):
        px = (np.degrees(np.unwrap(np.radians(g[:, 0, i]))) - lon) * k * 111.195
        py = (g[:, 1, i] - lat) * 111.195
        ax, ay, ux, uy = px[:-1], py[:-1], np.diff(px), np.diff(py)
        t = np.clip(-(ax * ux + ay * uy) / np.maximum(ux * ux + uy * uy, 1e-12), 0, 1)
        d = float(np.min(np.hypot(ax + t * ux, ay + t * uy)))
        if d < best[0]:
            best = (d, i)
    return float(best[1]), best[0]


def berryman_segments(grid, zone, max_distance_km=300.0, log=print):
    """Segment a mesh at Berryman et al. (2015)'s segment ends. PUBLIC DATA.

    Table 3.1 gives the trench end points of every segment
    (berryman_params.BERRYMAN_SEGMENT_TRENCH). Each end is placed on this
    mesh's top edge (its nearest point, ``trench_position``), so a segment
    covers the columns between its two ends; the boundary between two
    neighbours falls at the column edge nearest to their shared end point.
    The first and last segments on the mesh are stretched to its ends (the
    mesh and Berryman's trench end a few km apart), a segment that gets no
    column (its trench lies beyond this mesh, e.g. Patagonia on a SLAB1.0
    southamerica) is dropped, and so is one whose ends are both further
    than ``max_distance_km`` from the mesh.

    Returns the same list of dicts as ``bird_segments``, plus
    ``berryman_rows`` (the segment's own Table 3.1 row). A zone Berryman
    does not divide returns [].
    """
    import berryman_params as bp

    table = bp.ZONE_BERRYMAN_SEGMENTS.get(zone)
    if not table:
        log(f"    Berryman et al. (2015) do not divide {zone} into segments")
        return []
    n = np.asarray(grid).shape[2] - 1
    # a zone cut at Berryman's points along down-dip lines (step 1
    # TRENCH_ENDS) gets its internal boundaries the same way
    downdip = zone in getattr(bp, "ZONE_DEFAULT_TRENCH_ENDS", {})
    position = column_edge_position if downdip else trench_position
    if downdip:
        log("    boundaries on the column edge (down-dip line) passing closest to "
            "each Berryman point, as step 1 cut this zone's ends")

    placed = []
    for key, row in table:
        (l_lon, l_lat), (r_lon, r_lat) = bp.BERRYMAN_SEGMENT_TRENCH[row]
        t1, d1 = position(grid, l_lon, l_lat)
        t2, d2 = position(grid, r_lon, r_lat)
        if min(d1, d2) > max_distance_km:
            log(f"      {key:<12} both ends > {max_distance_km:.0f} km from the mesh: left out")
            continue
        placed.append((min(t1, t2), max(t1, t2), key, row, d1, d2))
    placed.sort(key=lambda p: (p[0] + p[1]) / 2.0)

    # column edges: between neighbours at the rounded mean of the shared end
    edges = [0]
    for (lo_a, hi_a, *_), (lo_b, hi_b, *_) in zip(placed[:-1], placed[1:]):
        edges.append(int(round((hi_a + lo_b) / 2.0)))
    edges.append(n)
    edges = [min(max(e, 0), n) for e in edges]

    segs = []
    for (lo, hi, key, row, d1, d2), a, b in zip(placed, edges[:-1], edges[1:]):
        if b <= a:
            log(f"      {key:<12} gets no column of this mesh (its trench lies "
                f"beyond it): left out")
            continue
        segs.append({"plate_pair": bp.BERRYMAN_SEGMENT_PLATES[row],
                     "segment_key": key, "berryman_rows": [row],
                     "alongstrike_slice": [a + 1, b],
                     "_end_distance_km": [round(d1, 1), round(d2, 1)]})

    log(f"    Berryman et al. (2015) Table 3.1 split {n} columns into "
        f"{len(segs)} segment(s):")
    for s in segs:
        a, b = s["alongstrike_slice"]
        log(f"      {s['segment_key']:<12} columns {a}-{b}  ({b - a + 1} columns; "
            f"Berryman '{s['berryman_rows'][0]}', its ends "
            f"{s['_end_distance_km'][0]:g} and {s['_end_distance_km'][1]:g} km "
            f"from {'the nearest column edge' if downdip else 'this mesh' + chr(39) + 's trench'})")
    if len(segs) == 1:
        log("    -> only one Berryman segment on this mesh: NOT segmented")
    return segs


def zone_segments(grid, zone, boundaries="berryman", log=print):
    """v9: the segments of ``zone`` on this mesh, from Berryman et al. (2015)
    Table 3.1 (``boundaries="berryman"``, the default) or from Bird (2003)'s
    plate pairs (``"bird"``). Both are public; steps 5 and 6 call this."""
    if boundaries == "berryman":
        return berryman_segments(grid, zone, log=log)
    if boundaries == "bird":
        return bird_segments(grid, log=log)
    raise ValueError(f"unknown segment boundaries {boundaries!r}: 'berryman' or 'bird'")


# ---------------------------------------------------------------------------
# PTHA18's own segmentation (comparison target only)
# ---------------------------------------------------------------------------

def ptha18_segments(zone, n_columns, sourcezone_parameters_csv, log=print):
    """PTHA18's segment boundaries, rescaled onto a mesh of ``n_columns``.

    READS PTHA18 DATA. Callers must keep this out of a --ptha false pipeline
    except in step 8, where comparing against PTHA18 is the entire point.

    PTHA18's indices count columns of PTHA18's OWN mesh. A from-scratch mesh
    usually has a different count (kermadectonga2: 73 vs 72), so the boundaries
    are converted to fractional along-strike position and re-applied here. The
    shift that causes is logged, not swallowed.

    Returns [] for an unsegmented zone.
    """
    import csv

    rows = []
    with open(sourcezone_parameters_csv, newline="") as f:
        for r in csv.DictReader(f):
            r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            if r.get("sourcename") == zone and r.get("segment_name"):
                rows.append(r)
    if not rows:
        return []

    lowers = [int(r["segment_boundary_alongstrike_index_lower"]) for r in rows]
    uppers = [int(r["segment_boundary_alongstrike_index_upper"]) for r in rows]
    n_official = max(uppers)

    segs = []
    for r, lo, up in zip(rows, lowers, uppers):
        # fractional span of this segment on PTHA18's mesh, mapped onto ours
        a = int(round((lo - 1) / n_official * n_columns)) + 1
        b = int(round(up / n_official * n_columns))
        segs.append({
            "name": r["segment_name"].lstrip("_"),
            "alongstrike_slice": [a, b],
            "ptha18_slice": [lo, up],
        })

    # rounding can leave a one-column gap or overlap; make the segments tile
    segs[0]["alongstrike_slice"][0] = 1
    segs[-1]["alongstrike_slice"][1] = n_columns
    for k in range(len(segs) - 1):
        segs[k + 1]["alongstrike_slice"][0] = segs[k]["alongstrike_slice"][1] + 1

    log(f"    PTHA18 segments {zone} into {len(segs)} "
        f"({n_official} columns official -> {n_columns} here)")
    for s in segs:
        a, b = s["alongstrike_slice"]
        lo, up = s["ptha18_slice"]
        log(f"      _{s['name']:<15} official {lo}-{up} -> {a}-{b}")
    return segs


def ptha18_official_segments_block(zone, zone_rates, sourcezone_parameters_csv,
                                   read_gcmt, log=print):
    """The ``segments`` block of PTHA18's OWN segmented run of ``zone``.

    READS PTHA18 DATA: step 8 only (the official comparison target).

    Unlike ``ptha18_segments()`` the indices stay PTHA18's own: this block is
    for the official input, which runs on PTHA18's own mesh table. Each
    segment takes what compute_rates_all_sources.R takes from its row of
    sourcezone_parameters.csv: columns, coupling (cmin/cpref/cmax), b range,
    mw_max_observed and prob_Mmax_below_Mmin. Its GCMT events come from
    ``read_gcmt(<zone><segment_name>)`` (the session's own, per segment), or
    None if the session gave none. Its convergence is not set here: the
    engine averages the Bird profile over the segment, as R does.

    ``zone_rates`` is the official input's zone-level rates block, whose
    coupling settings (prior type, uniform range) the segments share.
    Returns {} for a zone PTHA18 does not segment.
    """
    import copy
    import csv

    rows = []
    with open(sourcezone_parameters_csv, newline="") as f:
        for r in csv.DictReader(f):
            r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            if r.get("sourcename") == zone and r.get("segment_name"):
                rows.append(r)
    out = {}
    for r in rows:
        name = r["segment_name"].lstrip("_")
        lo = int(r["segment_boundary_alongstrike_index_lower"])
        up = int(r["segment_boundary_alongstrike_index_upper"])
        coupling = copy.deepcopy(zone_rates["coupling"])
        coupling["spreadsheet_values"] = [float(r["cmin"]), float(r["cpref"]),
                                          float(r["cmax"])]
        coupling["prob_zero_coupling"] = float(r["prob_Mmax_below_Mmin"] or 0.0)
        seg_rates = {
            "coupling": coupling,
            "b_anchor": [float(r["bmin"]), float(r["bmax"])],
            "mw_max_observed": float(r["mw_max_observed"]),
            "scaling_relation": r["scaling_relation"],
        }
        if zone_rates.get("update_logic_tree_weights_with_data"):
            gcmt = read_gcmt(zone + r["segment_name"])
            if gcmt is None:
                raise ValueError(f"no official GCMT observations for "
                                 f"{zone}{r['segment_name']}")
            seg_rates["observed_seismicity"] = gcmt
        out[name] = {
            "alongstrike_slice": [lo, up],
            "rates": seg_rates,
            "_provenance": (
                f"OFFICIAL: PTHA18's segment {zone}{r['segment_name']}, row of "
                f"sourcezone_parameters.csv (columns {lo}-{up} of PTHA18's own "
                f"mesh, coupling {r['cmin']}/{r['cpref']}/{r['cmax']}, "
                f"mw_max_observed {r['mw_max_observed']}); GCMT events from "
                f"the saved session."),
        }
        log(f"    PTHA18 segment _{name:<12} columns {lo}-{up}, coupling "
            f"{r['cmin']}/{r['cpref']}/{r['cmax']}, mw_max_observed "
            f"{r['mw_max_observed']}"
            + (f", {seg_rates['observed_seismicity']['count']} GCMT event(s)"
               if "observed_seismicity" in seg_rates else ""))
    return out


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare_with_ptha18(bird_segs, ptha_segs, n_columns, log=print):
    """Report how a Bird-derived segmentation differs from PTHA18's.

    Prints a per-column agreement figure and, when the segment COUNTS differ,
    says so plainly. This exists so the difference described in the module
    docstring (kermadectonga2: Bird 2, PTHA18 3) is visible in the run log and
    the report instead of being quietly absorbed.
    """
    if not ptha_segs:
        log("    PTHA18 does not segment this zone; nothing to compare")
        return None

    def boundaries(segs):
        return [s["alongstrike_slice"][1] for s in segs[:-1]]

    bb, pb = boundaries(bird_segs), boundaries(ptha_segs)
    log(f"    segment count: Bird {len(bird_segs)}, PTHA18 {len(ptha_segs)}")
    log(f"    interior boundaries: Bird {bb}, PTHA18 {pb}")

    matched = []
    for p in pb:
        if bb:
            nearest = min(bb, key=lambda x: abs(x - p))
            matched.append((p, nearest, abs(nearest - p)))
    for p, b, d in matched:
        log(f"      PTHA18 boundary at column {p}: nearest Bird boundary "
            f"{b} ({d} column{'s' if d != 1 else ''} away)")

    if len(bird_segs) != len(ptha_segs):
        log("    NOTE: the two source models genuinely differ in segment "
            "count. Bird's plate pairs do not break where PTHA18 does "
            "(see lib/segmentation.py docstring). Not an error.")

    return {"n_bird": len(bird_segs), "n_ptha18": len(ptha_segs),
            "bird_boundaries": bb, "ptha18_boundaries": pb,
            "matched": matched}
