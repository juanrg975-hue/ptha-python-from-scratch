"""Bird (2003) plate convergence on a unit-source table, as PTHA18 computes it.

Port of two R files in rptha/R/examples/austptha_template/EVENT_RATES/:

  make_spatially_variable_source_zone_convergence_rates.R
      event_conditional_probability_bird2003_factory(): every top-edge unit
      source (downdip_number == 1) is matched to the nearest Bird segment, and
      each unit source takes the match of its own along-strike column.
  compute_rates_all_sources.R, lines 371-415 (use_bird_convergence == 1)
      the matched velocities become the convergent component div_vec, the
      capped lateral component rl_vec, and the zone's convergence, the
      area-weighted mean of sqrt(div_vec^2 + rl_vec^2).

Step 3 runs this on the run's own mesh; step 8 runs it on PTHA18's official
mesh, so the official reference run gets the same per-column input R used.

Two segment tables (generate.py --convergence), both in from_scratch_v12/data/bird/
---------------------------------------------------------------------------------
  table="bird"          Bird (2003)'s public catalogue as published
                        (PB2002_steps.dat.txt.zip), every step of a convergent
                        boundary type (BIRD_CONVERGENT_TYPES). The default.
  table="bird-griffin"  The table PTHA18 used (bird_griffin_traces_table.csv.zip,
                        a copy of rptha's sourcezone_traces_table_merged.csv):
                        Bird's steps that PTHA18 kept plus Jonathan Griffin's
                        source-zone traces with their own plate rates. What R
                        reads; step 8 always uses it.
Nothing is read from the rptha folder. Which PTHA18 zone took its convergence
from which source: html/docs/convergence_sources.html.

Geodesy, as geosphere computes it in the R code
-----------------------------------------------
  top-edge point   destPoint(centroid, strike - 90, width/2*cos(dip))
                   ellipsoidal (WGS84), geosphere's default
  Bird centroid    midPoint(p1, p2, f = 0): great-circle midpoint
  distance         distHaversine, sphere of radius 6378137 m
"""

import os
import zipfile

import numpy as np
import pandas as pd
from pyproj import Geod

HERE = os.path.dirname(os.path.abspath(__file__))
# v11: rptha/ sits next to the package, or one level up when the package is
# in V9/
_RPTHA = next((p for p in (os.path.join(HERE, "..", "..", "rptha"),
                           os.path.join(HERE, "..", "..", "..", "rptha"))
               if os.path.isdir(p)), os.path.join(HERE, "..", "..", "rptha"))
_DATA_DIR = os.path.abspath(os.path.join(
    _RPTHA, "R", "examples", "austptha_template", "DATA"))
SOURCEZONE_PARAMETERS_CSV = os.path.join(
    _DATA_DIR, "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")
_BIRD_DATA = os.path.abspath(os.path.join(HERE, "..", "data", "bird"))
# The table R reads (EVENT_RATES/config.R line 49): Bird's steps merged with
# Jonathan Griffin's traces, trimmed to the zones PTHA18 modelled. A copy of
# rptha's sourcezone_traces_table_merged.csv.zip (see data/bird/README.txt).
BIRD_ZIP = os.path.join(_BIRD_DATA, "bird_griffin_traces_table.csv.zip")
# Bird (2003)'s own public catalogue (see from_scratch_v12/data/bird/README.txt).
BIRD_RAW_ZIP = os.path.join(_BIRD_DATA, "PB2002_steps.dat.txt.zip")
# generate.py --convergence values
TABLES = ("bird", "bird-griffin")
# Bird's convergent boundary types: subduction zone, oceanic convergent
# boundary, continental convergent boundary. Transforms, ridges and rifts
# are left out. PTHA18 itself kept OCB steps where they carry the interface
# (Makran), so SUB alone would miss real zones.
BIRD_CONVERGENT_TYPES = ("SUB", "OCB", "CCB")

# EVENT_RATES/config.R line 60: how far (degrees) the lateral component may
# rotate the slip vector away from pure thrust or pure normal.
RAKE_DEVIATION_DEG = 50.0

_WGS84 = Geod(ellps="WGS84")
_SPHERE = Geod(a=6378137.0, f=0.0)
_R_HAVERSINE = 6378137.0


def spherical_midpoint(lon1, lat1, lon2, lat2):
    """geosphere::midPoint(p1, p2, f = 0): half way along the great circle."""
    az, _, dist = _SPHERE.inv(lon1, lat1, lon2, lat2)
    lon, lat, _ = _SPHERE.fwd(lon1, lat1, az, np.asarray(dist) / 2.0)
    return np.asarray(lon, float), np.asarray(lat, float)


def dist_haversine(lon1, lat1, lon2, lat2):
    """geosphere::distHaversine, in metres (same formula and radius)."""
    p1x, p1y = np.radians(lon1), np.radians(lat1)
    p2x, p2y = np.radians(lon2), np.radians(lat2)
    a = (np.sin((p2y - p1y) / 2) ** 2
         + np.cos(p1y) * np.cos(p2y) * np.sin((p2x - p1x) / 2) ** 2)
    a = np.minimum(a, 1.0)
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * _R_HAVERSINE


def top_edge_point(lon_c, lat_c, strike_deg, dip_deg, width_km):
    """A unit source's centroid moved half a cell up-dip, onto its top edge:
    destPoint(centroid, strike - 90, width/2 * 1000 * cos(dip)). rptha's dip
    direction is strike + 90, so strike - 90 points up-dip."""
    half_width_surface_m = width_km / 2.0 * 1000.0 * np.cos(np.radians(dip_deg))
    lon2, lat2, _ = _WGS84.fwd(lon_c, lat_c, (strike_deg - 90.0) % 360.0,
                               half_width_surface_m)
    return lon2, lat2


def bounding_box(lon, lat, margin_deg=3.0):
    """(lon_range, lat_range) around the given points, plus a margin.

    Longitudes may be unwrapped to [0, 360) (step 1 does this so a zone
    crossing the antimeridian meshes cleanly). A span that sits entirely at or
    beyond 180 is recast to -180..180; any other span is kept as is, because
    for a zone straddling 180 (e.g. Kermadec's 165-195) the unwrapped form is
    the one without a seam. load_bird_segments tests both conventions.
    """
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    if lon_min >= 180.0:
        lon_min -= 360.0
        lon_max -= 360.0
    m = margin_deg
    return (lon_min - m, lon_max + m), (lat_min - m, lat_max + m)


def _in_box(df, lon_range, lat_range):
    mid_lon_std = np.where(df["mid_lon"] > 180.0, df["mid_lon"] - 360.0, df["mid_lon"])
    mid_lon_wrapped = np.where(df["mid_lon"] < 0.0, df["mid_lon"] + 360.0, df["mid_lon"])
    return (
        (pd.Series(mid_lon_std, index=df.index).between(*lon_range)
         | pd.Series(mid_lon_wrapped, index=df.index).between(*lon_range))
        & df["mid_lat"].between(*lat_range))


def _add_midpoints(df):
    df["mid_lon"], df["mid_lat"] = spherical_midpoint(
        df["Long1"].to_numpy(float), df["Lat1"].to_numpy(float),
        df["Long2"].to_numpy(float), df["Lat2"].to_numpy(float))
    return df


def load_bird_raw_segments():
    """Parse PB2002_steps.dat.txt (Bird 2003's own, untrimmed catalogue) into
    the schema of sourcezone_traces_table_merged.csv: Long1, Lat1, Long2,
    Lat2, RL_vel, Div_vel, name, class, Vel_L2R, Azi_Vel, collator.

    Column layout checked against the trimmed CSV: the Kermadec segment
    (183.661,-29.881)-(183.847,-29.469) with RL_vel=11.9, Div_vel=-99.8,
    Vel_L2R=100.5, Azi_Vel=105.0 there is the raw line
    "1484 :KE/PA -176.339 -29.881 -176.153 -29.469 49.2 21 100.5 105 -99.8
    11.9 -7696 220 :SUB", so columns 9/10/11/12 are Vel_L2R/Azi_Vel/Div_vel/
    RL_vel and the last word is Bird's boundary type. Only the convergent
    types (BIRD_CONVERGENT_TYPES) are kept; `class` stays blank, like every
    Bird2003_subset row of the trimmed CSV (PTHA18 only ever excludes
    Griffin's 'Normal' class).
    """
    with zipfile.ZipFile(BIRD_RAW_ZIP) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw_lines = f.read().decode("utf-8", "replace").splitlines()

    rows = []
    for line in raw_lines:
        parts = line.replace(":", " ").split()
        if len(parts) < 15:
            continue
        try:
            lon1, lat1, lon2, lat2 = (float(parts[2]), float(parts[3]),
                                      float(parts[4]), float(parts[5]))
            vel_l2r, azi_vel, div_vel, rl_vel = (
                float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11]))
        except ValueError:
            continue
        if parts[-1].strip(":*") not in BIRD_CONVERGENT_TYPES:
            continue
        rows.append((lon1, lat1, lon2, lat2, rl_vel, div_vel, "", "",
                     vel_l2r, azi_vel, "Bird2003_raw", parts[1]))

    return pd.DataFrame(rows, columns=[
        "Long1", "Lat1", "Long2", "Lat2", "RL_vel", "Div_vel", "name",
        "class", "Vel_L2R", "Azi_Vel", "collator", "plate_pair"])


def plate_codes(pair):
    """The two plate codes of a Bird ('AS/AF', 'EU-AF') or Berryman
    ('AF' backslash 'AS') plate pair, as an unordered set."""
    import re
    return frozenset(c for c in re.split(r"[/\\-]", str(pair)) if c)


def load_bird_segments(lon_range, lat_range, exclude_normal, log=print,
                       table="bird-griffin"):
    """Bird segments near a zone, as R's nearest_bird_point() sees them.

    table="bird": Bird (2003)'s public catalogue only (see the module
    docstring). table="bird-griffin": PTHA18's Bird+Griffin table, as below.

    R searches the whole trimmed table. Here only segments whose centroid is
    inside the box are kept: the nearest segment to a zone's trench is always
    well inside a 3 degree margin, so this changes nothing for a zone PTHA18
    modelled, and for a zone it did not model it stops the search from
    matching a segment of some other zone thousands of km away.

    exclude_normal: a thrust source (rake 90) ignores class 'Normal' rows
    (outer-rise bending), and a normal source (rake -90) uses only those;
    R does this by adding 1e12 m to the excluded rows' distances.

    No velocity plausibility filter, like R: an earlier version filtered
    and so dropped northern Tonga's real interface segments (they carry an
    anomalous Vel_L2R ~260 mm/yr), sending the search ~1700 km away.

    If the trimmed table has nothing in the box (a zone PTHA18 never
    modelled), fall back to Bird's untrimmed catalogue.
    """
    if table == "bird":
        raw = _add_midpoints(load_bird_raw_segments())
        log(f"  {len(raw)} convergent-boundary steps ({'/'.join(BIRD_CONVERGENT_TYPES)}) "
            f"in Bird (2003)'s public catalogue")
        df = raw[_in_box(raw, lon_range, lat_range)].copy()
        log(f"  {len(df)} steps in the bounding box lon {lon_range}, lat {lat_range}")
        if len(df) == 0:
            raise SystemExit(
                "no Bird (2003) convergent-boundary steps found near this zone; "
                "the zone may sit outside Bird's plate-boundary coverage, or "
                "the box margin may need widening")
        return df
    if table != "bird-griffin":
        raise ValueError(f"table must be one of {TABLES}, got {table!r}")

    with zipfile.ZipFile(BIRD_ZIP) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    log(f"  {len(df)} Bird/Griffin boundary segments in the merged table")

    df = _add_midpoints(df)
    df = df[_in_box(df, lon_range, lat_range)].copy()
    log(f"  {len(df)} segments in the bounding box lon {lon_range}, lat {lat_range}")

    is_normal = df["class"].astype(str) == "Normal"
    if exclude_normal:
        df = df[~is_normal].copy()
        log(f"  {len(df)} remain after excluding class=='Normal' rows "
            f"(outer-rise/bending, not interface convergence)")
    else:
        df = df[is_normal].copy()
        log(f"  {len(df)} remain after keeping only class=='Normal' rows "
            f"(this is a normal-faulting source)")

    if len(df) == 0:
        log("  no segments in the curated table near this zone -- falling back "
            "to Bird (2003)'s own untrimmed catalogue (PTHA18's table was "
            "trimmed to the zones it modelled)")
        raw = _add_midpoints(load_bird_raw_segments())
        log(f"  {len(raw)} convergent-boundary steps in Bird's original catalogue")
        df = raw[_in_box(raw, lon_range, lat_range)].copy()
        log(f"  {len(df)} segments in the bounding box (Bird raw catalogue)")
        if len(df) == 0:
            raise SystemExit(
                "no Bird segments found near this zone, even in Bird (2003)'s "
                "own untrimmed catalogue; the zone may sit outside Bird's "
                "plate-boundary coverage, or the box margin may need widening")
    return df


def convergence_components(div_vel, rl_vel, rake_deg):
    """compute_rates_all_sources.R lines 386-405, per unit source:

        thrust (rake 90):  div_vec = max(0, -Div_vel)   (only convergence counts)
        normal (rake -90): div_vec = max(0,  Div_vel)
        rl_vec = sign(RL_vel) * min(|RL_vel|, div_vec * tan(rake_deviation))

    Returns (div_vec, rl_vec) in mm/yr.
    """
    div_vel = np.asarray(div_vel, float)
    rl_vel = np.asarray(rl_vel, float)
    if rake_deg == 90.0:
        div_vec = np.maximum(0.0, -div_vel)
    elif rake_deg == -90.0:
        div_vec = np.maximum(0.0, div_vel)
    else:
        raise ValueError(f"unsupported rake {rake_deg}; PTHA18 only uses pure "
                         f"thrust (90) or pure normal (-90)")
    rl_vec = np.sign(rl_vel) * np.minimum(
        np.abs(rl_vel), div_vec * np.tan(np.radians(RAKE_DEVIATION_DEG)))
    return div_vec, rl_vec


def column_convergence(stats, log=print, margin_deg=3.0, table="bird-griffin",
                       plate_pairs=None):
    """Bird convergence per along-strike column of a unit-source table.

    stats: dict of arrays with lon_c, lat_c, strike, dip, width, length,
    rake, downdip_number, alongstrike_number (the unit-source statistics).
    table: which Bird segments to match against, "bird" or "bird-griffin"
    (see the module docstring).
    plate_pairs: optional list of plate pairs (any notation, see
    plate_codes): match only Bird steps between those plates. v9: for a
    zone defined by Berryman et al. (2015)'s segments (hellenic_west2 /
    hellenic_east2 / hellenic2, Berryman's AF-AS = Bird AS/AF). SLAB2's
    trench there lies 120-300 km from Bird's, so the nearest step of ANY
    pair can be another boundary: hellenic_west2's north-west columns
    matched Calabria's EU/AF (7 mm/yr) and the EU-AF boundary north of
    Kefalonia (4 mm/yr) instead of Bird's own Hellenic AS/AF (28 mm/yr),
    which runs to 37.45N. Only with table="bird" (the raw catalogue carries
    the pair codes).

    Returns a dict:
      alongstrike_number      1..n
      div_mm_per_yr           convergent component, the LEVEL 4 target shape
      convergent_slip_mm_per_yr  sqrt(div^2 + rl^2), the event weights
      distance_km             top edge to the matched Bird centroid
      per_unit_source_convergent_slip  the column value on every unit source
      area_weighted_mean_mm_per_yr     the zone's convergence (horizontal,
                                       BEFORE the 1/cos(dip) correction,
                                       which the engine applies once)
    """
    rake = np.asarray(stats["rake"], float)
    if not np.all(rake == rake[0]) or rake[0] not in (90.0, -90.0):
        raise ValueError(f"expected one pure-thrust or pure-normal rake, got "
                         f"{np.unique(rake)}")

    lon_c = np.asarray(stats["lon_c"], float)
    lat_c = np.asarray(stats["lat_c"], float)
    lon_range, lat_range = bounding_box(lon_c, lat_c, margin_deg)
    bird = load_bird_segments(lon_range, lat_range, exclude_normal=(rake[0] == 90.0),
                              log=log, table=table)
    if plate_pairs:
        if "plate_pair" not in bird.columns:
            raise ValueError("plate_pairs needs table='bird' (Bird's own catalogue)")
        want = {plate_codes(p) for p in plate_pairs}
        keep = bird["plate_pair"].map(plate_codes).isin(want)
        log(f"  {int(keep.sum())} of {len(bird)} steps are between the zone's own "
            f"plates ({', '.join(sorted(map(str, plate_pairs)))}); only these are matched")
        bird = bird[keep].copy()
        if len(bird) == 0:
            raise SystemExit(f"no Bird step between {plate_pairs} near this zone")
    b_lon = bird["mid_lon"].to_numpy(float)
    b_lat = bird["mid_lat"].to_numpy(float)

    along = np.asarray(stats["alongstrike_number"]).astype(int)
    top = np.asarray(stats["downdip_number"]).astype(int) == 1
    n_col = int(along.max())
    if sorted(along[top]) != list(range(1, n_col + 1)):
        raise ValueError("expected exactly one top-edge unit source per column")

    div_vel = np.empty(n_col)
    rl_vel = np.empty(n_col)
    dist_km = np.empty(n_col)
    for k in np.where(top)[0]:
        e_lon, e_lat = top_edge_point(lon_c[k], lat_c[k], float(stats["strike"][k]),
                                      float(stats["dip"][k]), float(stats["width"][k]))
        d = dist_haversine(e_lon, e_lat, b_lon, b_lat)
        i = int(np.argmin(d))                       # which.min: first minimum
        j = along[k] - 1
        div_vel[j] = float(bird["Div_vel"].iloc[i])
        rl_vel[j] = float(bird["RL_vel"].iloc[i])
        dist_km[j] = d[i] / 1000.0

    div_vec, rl_vec = convergence_components(div_vel, rl_vel, float(rake[0]))
    slip = np.sqrt(div_vec ** 2 + rl_vec ** 2)
    per_us = slip[along - 1]
    area = np.asarray(stats["length"], float) * np.asarray(stats["width"], float)
    return {
        "alongstrike_number": np.arange(1, n_col + 1),
        "div_mm_per_yr": div_vec,
        "convergent_slip_mm_per_yr": slip,
        "distance_km": dist_km,
        "per_unit_source_convergent_slip": per_us,
        "area_weighted_mean_mm_per_yr": float(np.sum(per_us * area) / np.sum(area)),
    }


def rates_entries(col):
    """The input-JSON 'rates' entries that select R's Bird model in the
    engine, from column_convergence()'s result (or its JSON copy):

      conditional_probability_model  "convergent_slip_weighted": within a
                                     magnitude, an event's probability is
                                     proportional to its area times the mean
                                     convergent slip of its unit sources
      convergent_slip_profile        per column, those weights (mm/yr)
      convergence_profile            per column, div_vec, the shape the
                                     LEVEL 4 edge fit aims for
    """
    return {
        "conditional_probability_model": "convergent_slip_weighted",
        "convergent_slip_profile": [float(v) for v in col["convergent_slip_mm_per_yr"]],
        "convergence_profile": [float(v) for v in col["div_mm_per_yr"]],
    }


def ptha18_uses_bird_convergence(zone):
    """PTHA18's use_bird_convergence flag for a zone (sourcezone_parameters.csv),
    or None if PTHA18 has no row for it. Every PTHA18 zone uses Bird except
    puysegur/puysegur2, which use a constant 35 mm/yr."""
    df = pd.read_csv(SOURCEZONE_PARAMETERS_CSV, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    rows = df[df["sourcename"].astype(str).str.strip() == zone]
    if rows.empty:
        return None
    return int(rows["use_bird_convergence"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# Plate-boundary change detector (v8.1)
# ---------------------------------------------------------------------------
# Why this exists
# ---------------
# column_convergence() matches every along-strike column to its NEAREST Bird
# segment, and that match always succeeds: there is no "no data" outcome. So
# a mesh that runs past the end of its own subduction zone does not fail, it
# quietly starts reading the convergence of the NEXT plate boundary along, at
# a perfectly normal match distance.
#
# kurilsjapan is the case this was written for. Its SLAB2 raster continues
# south past the Japan Trench onto the Izu-Bonin Trench, a different plate
# pair that PTHA18 models separately as izumariana:
#
#   Bird rows 435-448  Japan Trench (Pacific/Okhotsk)  Div_vel -80..-93, ends 34.44N
#   Bird rows 781-789  Izu-Bonin Trench                Div_vel -44..-60, starts 33.96N
#
# PTHA18 cuts kurilsjapan at 34.3N and starts izumariana at exactly 34.3N.
# The v8 mesh ran to 32.27N, so its last 5 of 65 columns sat on Izu-Bonin and
# pulled the zone convergence from 87.71 down to 86.18 mm/yr. (The value
# recovered from PTHA18's own session is 87.7244, so dropping those 5 columns
# is what reproduces the official number.)
#
# What it detects, and what it deliberately does not
# --------------------------------------------------
# A step change in Div_vel between along-strike NEIGHBOURS is the signature:
# within one boundary Div_vel drifts smoothly (kurilsjapan columns 1-60 move
# by at most ~13 mm/yr between neighbours), while crossing onto another plate
# pair jumps it (column 60 -> 61: 90.9 -> 60.2). Only a jump that is followed
# by a RUN of columns that all stay on the far side counts, and only near an
# END of the zone -- a one-column spike mid-arc is Bird's own segment noise,
# not a boundary crossing, and a mesh cannot cross onto another plate in its
# middle and come back.
#
# This only ever REPORTS. It never trims anything by itself: which columns are
# really yours is a geological judgement, and on a zone PTHA18 did not model
# there may be no right answer. step3 prints it, step9 puts it in the report,
# and --auto-clip acts on it only when you ask.

# Between-neighbour jump in the convergent component (mm/yr) that marks a
# different plate pair rather than drift along one. kurilsjapan's real
# crossing is 30.7; its largest within-boundary step is 12.9.
PLATE_CHANGE_JUMP_MM_PER_YR = 25.0
# A crossing must be sustained over at least this many columns. Below it the
# jump is treated as Bird segment noise.
PLATE_CHANGE_MIN_RUN = 2
# A crossing is only looked for within this fraction of the zone from either
# end: the far side of a real crossing is, by construction, a tail.
PLATE_CHANGE_END_FRACTION = 0.25


def detect_plate_boundary_change(col, jump_mm_per_yr=PLATE_CHANGE_JUMP_MM_PER_YR,
                                 min_run=PLATE_CHANGE_MIN_RUN,
                                 end_fraction=PLATE_CHANGE_END_FRACTION):
    """Columns that appear to sit on a DIFFERENT plate boundary than the zone.

    `col` is column_convergence()'s result (or its JSON copy). Returns None if
    nothing is found, otherwise a dict:

      end               "start" (low column numbers) or "end" (high ones):
                        which tail of the zone is on the other boundary
      columns           1-based alongstrike_number of every suspect column
      n_suspect         how many
      n_total           columns in the mesh
      boundary_column   the last column that still looks like the zone itself
      jump_mm_per_yr    the Div_vel step across the crossing
      div_inside        mean div of the kept side
      div_outside       mean div of the suspect side
      mean_inside       area-unweighted mean convergent slip of the kept side,
                        i.e. roughly what the zone's convergence becomes if
                        the tail is trimmed (the exact value needs step 3 to
                        be re-run on the trimmed mesh)
    """
    div = np.asarray(col["div_mm_per_yr"], float)
    slip = np.asarray(col["convergent_slip_mm_per_yr"], float)
    along = np.asarray(col["alongstrike_number"], int)
    n = div.size
    if n < 2 * min_run + 2:
        return None

    d = np.diff(div)
    window = max(int(np.ceil(end_fraction * n)), min_run + 1)

    best = None
    for k in range(n - 1):                      # crossing sits between k, k+1
        if abs(d[k]) < jump_mm_per_yr:
            continue
        n_low, n_high = k + 1, n - k - 1        # columns each side
        # Only an END tail, and only one long enough to be a real run.
        if n_low <= window and n_low >= min_run:
            side, susp = "start", np.arange(0, k + 1)
        elif n_high <= window and n_high >= min_run:
            side, susp = "end", np.arange(k + 1, n)
        else:
            continue
        keep = np.setdiff1d(np.arange(n), susp)
        # The far side must STAY on the far side: every suspect column has to
        # remain past the step, not come straight back. Compared against the
        # MEDIAN of the kept side rather than its extremes -- a single
        # within-zone dip (kurilsjapan column 57 drops to 77.3 among
        # neighbours near 91) is ordinary Bird noise and must not veto a real
        # crossing five columns away.
        ref = float(np.median(div[keep]))
        half = jump_mm_per_yr * 0.5
        if d[k] < 0:                            # suspect side is the LOWER one
            if not np.all(div[susp] < ref - half):
                continue
        else:                                   # suspect side is the HIGHER one
            if not np.all(div[susp] > ref + half):
                continue
        if best is None or abs(d[k]) > best["jump_mm_per_yr"]:
            best = {
                "end": side,
                "columns": [int(v) for v in along[susp]],
                "n_suspect": int(susp.size),
                "n_total": int(n),
                "boundary_column": int(along[keep[0] if side == "start" else keep[-1]]),
                "jump_mm_per_yr": float(abs(d[k])),
                "div_inside": float(div[keep].mean()),
                "div_outside": float(div[susp].mean()),
                "mean_inside": float(slip[keep].mean()),
                "mean_outside": float(slip[susp].mean()),
                "mean_all": float(slip.mean()),
            }
    return best


def plate_boundary_change_message(found, zone=""):
    """The detector's finding as plain lines, for step 3's log and the report."""
    if not found:
        return []
    z = f"{zone} " if zone else ""
    tail = "first" if found["end"] == "start" else "last"
    return [
        f"{z}mesh appears to run past this zone's plate boundary: its {tail} "
        f"{found['n_suspect']} of {found['n_total']} along-strike columns "
        f"(numbers {found['columns'][0]}-{found['columns'][-1]}) match Bird "
        f"segments of a different plate pair.",
        f"Convergent component steps by {found['jump_mm_per_yr']:.1f} mm/yr "
        f"across column {found['boundary_column']}: "
        f"{found['div_inside']:.1f} mm/yr on this zone's side, "
        f"{found['div_outside']:.1f} mm/yr on the far side.",
        f"Trimming those columns would move the mean convergent slip from "
        f"{found['mean_all']:.1f} to about {found['mean_inside']:.1f} mm/yr "
        f"(unweighted column means; the exact zone convergence needs step 3 "
        f"re-run on the trimmed mesh).",
        f"To cut them automatically, re-run step 1 with --auto-clip (or "
        f"step_total.py --auto-clip): it reads this finding and clips the "
        f"raster to the columns that stay on this zone's own boundary. "
        f"Nothing has been trimmed by this run.",
    ]


def auto_clip_bbox(found, grid, margin_deg=0.35):
    """A --clip window that drops the columns detect_plate_boundary_change()
    flagged, derived from the mesh itself rather than typed by hand.

    `grid` is step 2's unit_source_grid, shape (n_downdip+1, 3,
    n_alongstrike+1). The suspect columns are cut off the flagged END, and
    the window is the lon/lat bounding box of what is left, plus a margin.

    The margin is DIRECTIONAL, and that is not a detail
    ---------------------------------------------------
    Step 1 needs a little room around the mesh it keeps: the trench detector
    works on the raster's outer edge, and a box drawn exactly on the mesh
    makes the raster's own cut edge look like a trench. So a margin is
    added -- but NOT on the side the tail was cut from, where it would
    simply re-admit part of what was just removed.

    Measured on kurilsjapan, where an all-sides margin was the bug: the kept
    columns reach down to lat 34.2277 (PTHA18's own southern limit for this
    zone is 34.30). A 0.35 deg margin on every side pushed the clip to
    33.8777, and step 1 re-contoured that strip into ONE extra along-strike
    column at lat 34.078 -- still on the Izu-Bonin side, still reading
    60.2 mm/yr, and still visible as a dark cell at the tip of the mesh.
    Cutting the margin on the flagged side alone removes it.

    The margin is kept on the other three sides, and on BOTH ends when the
    cut is in longitude but the tail is in latitude (or the reverse): only
    the axis and direction the tail actually lies in is tightened.

    Returns (lon_min, lon_max, lat_min, lat_max), or None if nothing to cut.

    This is deliberately a BOX, the same shape --clip already takes, so
    nothing downstream needs a new code path. For a zone whose kept part and
    cut tail overlap in both lon and lat a box cannot separate them; the
    caller checks that and says so rather than clipping wrongly.
    """
    if not found:
        return None
    n_along = grid.shape[2] - 1
    n_cut = found["n_suspect"]
    if n_cut >= n_along:
        return None
    # Grid NODES to keep, and the ones being cut. Columns are 1-based;
    # column i spans nodes i-1, i.
    if found["end"] == "end":
        keep = slice(0, n_along - n_cut + 1)
        cut = slice(n_along - n_cut, n_along + 1)
    else:
        keep = slice(n_cut, n_along + 1)
        cut = slice(0, n_cut + 1)
    lon, lat = grid[:, 0, keep], grid[:, 1, keep]
    lon_c, lat_c = grid[:, 0, cut], grid[:, 1, cut]

    lo0, lo1 = float(np.nanmin(lon)), float(np.nanmax(lon))
    la0, la1 = float(np.nanmin(lat)), float(np.nanmax(lat))
    m = margin_deg

    # Which way does the cut tail lie from the part we keep? Only the edges
    # it lies beyond are left unpadded; every other edge gets the margin.
    kept_lon_mid, kept_lat_mid = 0.5 * (lo0 + lo1), 0.5 * (la0 + la1)
    tail_lon, tail_lat = float(np.nanmean(lon_c)), float(np.nanmean(lat_c))

    pad_lo0 = pad_lo1 = pad_la0 = pad_la1 = m
    # Longitude: is the tail clearly to the west or the east?
    if tail_lon < lo0:
        pad_lo0 = 0.0
    elif tail_lon > lo1:
        pad_lo1 = 0.0
    # Latitude: clearly to the south or the north?
    if tail_lat < la0:
        pad_la0 = 0.0
    elif tail_lat > la1:
        pad_la1 = 0.0
    return (lo0 - pad_lo0, lo1 + pad_lo1, la0 - pad_la0, la1 + pad_la1)


def auto_clip_is_separable(found, grid, bbox):
    """True if `bbox` actually excludes the flagged columns.

    A lon/lat box can only cut a tail that sticks out in lon or lat. On a
    zone that doubles back on itself the tail can sit inside the kept part's
    own bounding box, and clipping to it would change nothing (or cut the
    wrong thing). Checked explicitly so the caller can say so instead of
    producing a silently useless clip.
    """
    if not found or bbox is None:
        return False
    n_along = grid.shape[2] - 1
    n_cut = found["n_suspect"]
    if found["end"] == "end":
        cut = slice(n_along - n_cut, n_along + 1)
    else:
        cut = slice(0, n_cut + 1)
    lon, lat = grid[:, 0, cut], grid[:, 1, cut]
    lo0, lo1, la0, la1 = bbox
    # The cut tail's own centre must fall outside the kept window.
    c_lon, c_lat = float(np.nanmean(lon)), float(np.nanmean(lat))
    return not (lo0 <= c_lon <= lo1 and la0 <= c_lat <= la1)
