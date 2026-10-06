"""STEP 5 - Filter the GCMT catalogue down to this source zone.

LEVEL 3 input, part 2 of 2. This reimplements PTHA18's gcmt_subsetter.R,
which does not exist anywhere else in this repository: every other place in
the codebase that uses observed seismicity reads it back out of PTHA18's own
saved session (see official_gcmt_observations.csv), rather than re-deriving
it. This script is the one place that actually applies the selection rule.

The rule (report Section 3.7.3 and gcmt_subsetter.R)
------------------------------------------------------
An event is attributed to this source zone's interface if ALL of:

  1. its hypocentre OR its centroid lies inside the unit-source cells, each
     cell polygon buffered by 0.4 deg (gcmt_subsetter.R:144-150 with
     rptha::lonlat_in_poly, which buffers every cell with
     rgeos::gBuffer(width = 0.4, byid = TRUE) in plain lon/lat degrees and
     tests point-in-polygon; longitudes are shifted by 360 to the mesh's
     side first);
  2. |rake - 90| <= 50 deg              (thrust-like, either nodal plane)
  3. |strike - unit_source_strike| <= 50 deg on the SAME nodal plane, where
     unit_source_strike is that of the unit source nearest the CENTROID by
     great-circle distance (gcmt_subsetter.R:175-185, distHaversine);
  4. depth <= 71 km, the NDK's hypocentral (PDE) depth;
  5. Mw >= 7.15.

v8 replaced v5-v7's test (1) "within 0.4 deg of the nearest unit-source
CENTROID, measured as plain lon/lat distance" and their plain lon/lat
nearest-unit-source search in (3). Checked on PTHA18's own meshes against
PTHA18's own selected events (official_gcmt_observations.csv): this rule
reproduces the official event list EXACTLY on all 8 zones that have one
(kermadectonga2 9/9, kurilsjapan 21/21, southamerica 25/25, philippine 6/6,
puysegur2 4/4, cascadia 1/1, makran2 0, izumariana 0), where v7's rule got
4 of the 8 (kermadectonga2 8/9, kurilsjapan 19/21, philippine 5/6,
cascadia 0/1). The centroid depth instead of the hypocentral one adds a
spurious event on kurilsjapan, so (4) is the hypocentral depth.

Both nodal planes are checked for (2)-(3) because a moment tensor does not by
itself say which of the two planes is the fault: if EITHER plane is thrust-like
and strike-aligned, the event qualifies.

Magnitude threshold: Mw >= 7.15, i.e. Mw_min - dMw/2 with Mw_min=7.2,
dMw=0.1 -- see run_logic_tree.py's own note on why the half-bin shift belongs
on the threshold, not the other way round. This is the report's global
default; it does not vary by zone.

Getting Mw from GCMT: M0 = sqrt(sum(M_ij^2)/2) of the moment tensor on NDK
line 4 (times 10^exponent), Mw = 2/3 (log10 M0[N m] - 9.05). This
reproduces all 122 magnitudes in official_gcmt_observations.csv to machine
precision (v8 check; the NDK's own scalar moment, or the best-double-couple
moment, do not).

v9, segmented runs (--segmented true)
-------------------------------------
Each segment (Berryman et al. 2015 Table 3.1 by default, lib/segmentation.py
zone_segments) gets its own catalogue from the SAME rule applied to its
own columns: its cells for the buffered test and its unit sources for the
strike test (gcmt_subsetter.R 106-142, as compute_rates_all_sources.R
183-195 calls it per segment). The 0.4 deg buffers of neighbours overlap,
so an event near a boundary can count in both segments, as in PTHA18
(kermadectonga2: _tonga 4 + _kermadec 6 = 10 against 9 for the zone).
validate_v9.py gcmt checks this against PTHA18's own list of every segment.

What to expect
---------------
With PTHA18's own mesh this rule gives PTHA18's own event list exactly (see
above). With this run's from-scratch mesh the list can still differ by an
event or two, wherever the two meshes' outlines differ (along-strike extent,
the shallow edge).

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/calabria2/steps/step5_subset_gcmt.py
"""

import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))
# v5's own forked pyptha (see step2's comment / from_scratch_v12/html/docs/code_map.html).
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))  # helper modules

from pyptha_v12 import unit_sources as us  # noqa: E402

GRID_NPY = os.path.join(EXAMPLE, "data", "slab2", "unit_source_grid.npy")
GCMT_DIR = os.path.join(EXAMPLE, "data", "gcmt")
OUT_CSV = os.path.join(GCMT_DIR, "calabria2_gcmt_subset.csv")

PTHA18_ZONE_NAME = "calabria2"
# --ptha (see generate.py): with False, PTHA18's own event count is never
# read here (the report compares it from step 8's outputs).
USE_PTHA = False
# --segmented flag from generate.py (v9, LEVEL 0). With False this step
# behaves exactly as v8's did.
SEGMENTED = False
# v9: where the segments end, "berryman" (Table 3.1 end points) or "bird"
SEGMENT_BOUNDARIES = "berryman"
OFFICIAL_GCMT_CSV = os.path.join(
    ROOT, "official_ptha_data", "trees", "official_gcmt_observations.csv")

# The selection rule, verbatim from the PTHA18 report / gcmt_subsetter.R.
BUFFER_DEG = 0.4
RAKE_TOL_DEG = 50.0
STRIKE_TOL_DEG = 50.0
MAX_DEPTH_KM = 71.0
THRESHOLD_MW = 7.15  # = Mw_min - dMw/2, Mw_min=7.2, dMw=0.1

# PTHA18's own observation window; matches step 4. See that script's module
# docstring for why v2 uses the official end date rather than today.
WINDOW_START = "1976-01-01"
WINDOW_END = "2017-03-01"


def parse_ndk_full(gcmt_dir):
    """Re-parse every cached NDK file, this time keeping line 4 (moment
    tensor) too.

    Step 4 kept only line 1 (hypocentre) and line 5 (nodal planes) fields; Mw
    needs line 4's exponent, so this is parsed again here rather than adding
    a Mw column to step 4's generic catalogue export. Step 4 caches one
    cumulative file plus one file per month since 2021, all as *.ndk in the
    same directory; they are concatenated here and deduplicated below.
    """
    ndk_files = sorted(glob.glob(os.path.join(gcmt_dir, "*.ndk")))
    if not ndk_files:
        raise SystemExit(f"no *.ndk files in {gcmt_dir}\n"
                         f"Run step4_fetch_gcmt.py first.")

    line1_re = re.compile(
        r"^\S+\s+(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):([\d.]+)\s+"
        r"(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)")

    lines = []
    for path in ndk_files:
        with open(path, "r", encoding="latin-1") as f:
            lines.extend(f.read().splitlines())

    rows = []
    n_records = len(lines) // 5
    for i in range(n_records):
        block = lines[i * 5:(i + 1) * 5]
        if len(block) < 5 or not block[0].strip():
            continue
        m = line1_re.match(block[0])
        if not m:
            continue
        yr, mo, da, hh, mm, ss, hypo_lat, hypo_lon, depth = m.groups()

        # Line 3 of the 5-line record (block index 2): "CENTROID: time_shift
        # sd lat sd lon sd depth sd type source_id" -- whitespace-tokenised
        # the same way line 1 is (field widths vary, but no field itself
        # contains whitespace). NDK layout is PDE-hypocentre (index 0), CMT
        # identifier/tensor-header line (index 1, NOT this one -- confirmed
        # directly against a real cached .ndk record; reading index 1 here
        # instead of 2 was an earlier bug in this script that silently
        # produced garbage cent_lat/cent_lon, e.g. cent_lon>360, from that
        # unrelated header line's own numeric-looking tokens), CENTROID
        # (index 2), moment tensor (index 3), nodal planes (index 4).
        # gcmt_subsetter.R tests BOTH hypocentre and centroid for spatial
        # inclusion (inside_events_hypo | inside_events_centroid), and uses
        # the CENTROID specifically for the nearest-unit-source strike match
        # (gcmt$cent_lon[i], gcmt$cent_lat[i]) -- this script previously used
        # only the hypocentre for everything, which measurably under-counted
        # real events (kermadectonga2: 5/9 of PTHA18's own official count).
        parts2 = block[2].split()
        try:
            cent_lat = float(parts2[3])
            cent_lon = float(parts2[5])
        except (ValueError, IndexError):
            continue

        # Line 4: "EX Mrr sMrr Mtt sMtt Mpp sMpp Mrt sMrt Mrp sMrp Mtp sMtp"
        # (13 whitespace tokens). EX is the power-of-ten exponent (dyne-cm)
        # shared by all six moments; no literal "EX" label is present in this
        # compact record form (only in GCMT's verbose web format).
        parts4 = block[3].split()
        try:
            exponent = float(parts4[0])
            moments = [float(x) for x in parts4[1:13:2]]
        except (ValueError, IndexError):
            continue
        if len(moments) < 6:
            continue

        # Scalar moment from the tensor norm (dyne-cm), then to Mw (Hanks &
        # Mw = (2/3)(log10(M0_Nm) - 9.05), M0_Nm = M0_dyncm*1e-7. The constant
        # is 9.05, not the more commonly cited 9.1: verified directly against
        # rptha::M0_2_Mw's source (default argument `constant = 9.05`), the
        # exact function PTHA18's own R pipeline uses for this conversion.
        # Using 9.1 here previously shifted every from-scratch Mw high by a
        # constant (2/3)*(9.1-9.05) = 0.0333, confirmed by diffing this
        # script's magnitudes against PTHA18's officially saved ones for
        # matching events.
        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp = moments
        m0_norm = np.sqrt(0.5 * (m_rr**2 + m_tt**2 + m_pp**2)
                          + m_rt**2 + m_rp**2 + m_tp**2)
        m0_dyncm = m0_norm * (10.0 ** exponent)
        m0_nm = m0_dyncm * 1e-7
        if m0_nm <= 0:
            continue
        mw = (2.0 / 3.0) * (np.log10(m0_nm) - 9.05)

        parts5 = block[4].split()
        if len(parts5) < 6:
            continue
        try:
            strike1, dip1, rake1, strike2, dip2, rake2 = (
                float(x) for x in parts5[-6:])
        except ValueError:
            continue

        rows.append({
            "date": f"{yr}-{mo}-{da}",
            "hypo_lat": float(hypo_lat), "hypo_lon": float(hypo_lon),
            "cent_lat": cent_lat, "cent_lon": cent_lon,
            "depth_km": float(depth), "Mw": mw,
            "strike1": strike1, "dip1": dip1, "rake1": rake1,
            "strike2": strike2, "dip2": dip2, "rake2": rake2,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # [start, end): PTHA18's window ends at 2017-03-01 00:00
    df = df[(df["date"] >= WINDOW_START) & (df["date"] < WINDOW_END)]
    df = df.drop_duplicates(subset=["date", "hypo_lat", "hypo_lon", "depth_km"])
    return df.reset_index(drop=True)


def load_unit_source_table():
    grid = np.load(GRID_NPY)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    return grid, pd.DataFrame({
        "lon": stats["lon_c"], "lat": stats["lat_c"], "strike": stats["strike"],
    })


def buffered_cells(grid):
    """Union of every unit-source cell polygon buffered by BUFFER_DEG, in
    plain lon/lat degrees: rptha::lonlat_in_poly's gBuffer(byid = TRUE).
    quad_segs=5 is rgeos' default arc resolution."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    nr, _, nc = grid.shape
    cells = []
    for r in range(nr - 1):
        for j in range(nc - 1):
            q = [(grid[r, 0, j], grid[r, 1, j]),
                 (grid[r, 0, j + 1], grid[r, 1, j + 1]),
                 (grid[r + 1, 0, j + 1], grid[r + 1, 1, j + 1]),
                 (grid[r + 1, 0, j], grid[r + 1, 1, j])]
            cells.append(Polygon(q).buffer(BUFFER_DEG, quad_segs=5))
    return unary_union(cells)


def near_lon(lon, ref_lon):
    """lon shifted by a multiple of 360 to the mesh's side
    (rptha::adjust_longitude_by_360_deg)."""
    return lon + 360.0 * np.round((ref_lon - lon) / 360.0)


def select_events(cat, grid, ref_lon):
    """The rule of the module docstring applied to the unit sources of
    `grid`: the whole zone, or (v9) one segment's columns of it.

    gcmt_subsetter.R's get_gcmt_events_in_poly does the same for a segment
    (lines 106-142): it keeps only the segment's cells (alongstrike_index
    min..max) for the buffered-polygon test AND only the segment's unit
    sources for the nearest-strike test, then applies the unchanged rule."""
    from shapely.geometry import Point
    stats = us.discretized_source_approximate_summary_statistics(grid)
    us_df = pd.DataFrame({"lon": stats["lon_c"], "lat": stats["lat_c"],
                          "strike": stats["strike"]})
    region = buffered_cells(grid)
    keep = []
    for _, row in cat.iterrows():
        if row["depth_km"] > MAX_DEPTH_KM:
            continue
        if row["Mw"] < THRESHOLD_MW:
            continue
        # gcmt_subsetter.R:144-150 -- spatial inclusion is judged on EITHER
        # point (inside_events_hypo | inside_events_centroid): inside the
        # 0.4-deg-buffered unit-source cells (rptha::lonlat_in_poly).
        inside = (region.contains(Point(near_lon(row["hypo_lon"], ref_lon), row["hypo_lat"]))
                  or region.contains(Point(near_lon(row["cent_lon"], ref_lon), row["cent_lat"])))
        if not inside:
            continue
        # gcmt_subsetter.R:179-185 -- the nearest-unit-source strike search
        # (used for the alignment test below) is built from the CENTROID
        # specifically, not the hypocentre.
        strike = nearest_unit_source_strike(row["cent_lon"], row["cent_lat"], us_df)
        if not qualifies(row, strike):
            continue
        keep.append(row)
    # An empty `keep` gives pd.DataFrame([]) no columns at all, which writes
    # a CSV with no header row -- step6 (or anyone else) reading it back then
    # hits pandas' EmptyDataError. cat.columns keeps the real column names
    # either way, so a zero-event zone (a legitimate result: some zones
    # genuinely have no qualifying GCMT event) still writes a readable,
    # header-only CSV instead of failing downstream.
    subset = pd.DataFrame(keep, columns=cat.columns).reset_index(drop=True)
    if len(subset):
        subset = subset.sort_values("Mw").reset_index(drop=True)
    return subset


def _haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6378137.0 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def nearest_unit_source_strike(lon, lat, us_df):
    """Strike (degrees) of the unit source whose centroid is nearest to
    (lon, lat) by great-circle distance (gcmt_subsetter.R: distHaversine)."""
    d = _haversine_m(lon, lat, us_df["lon"].to_numpy(), us_df["lat"].to_numpy())
    return float(us_df["strike"].iloc[int(np.argmin(d))]) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two angles in degrees, mod 360."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def qualifies(row, unit_source_strike):
    """Either nodal plane counts if it is thrust-like and strike-aligned."""
    for rake, strike in ((row["rake1"], row["strike1"]),
                        (row["rake2"], row["strike2"])):
        if angle_diff(rake, 90.0) <= RAKE_TOL_DEG \
                and angle_diff(strike, unit_source_strike) <= STRIKE_TOL_DEG:
            return True
    return False


def read_official_gcmt_count():
    """If step 8 has already extracted the official session, read its
    event count and duration for this zone. Returns (count, duration) or
    (None, None) if step 8 has not run, or the zone had no LEVEL 3 data."""
    if not os.path.exists(OFFICIAL_GCMT_CSV):
        return None, None
    with open(OFFICIAL_GCMT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["source"] == PTHA18_ZONE_NAME:
                return int(float(row["count"])), float(row["duration_years"])
    return None, None


def main():
    print("=" * 70)
    print("STEP 5 - Filter GCMT to the Calabria interface")
    print("=" * 70)

    print("\n  re-parsing NDK for moment-tensor magnitudes ...")
    cat = parse_ndk_full(GCMT_DIR)
    print(f"  {len(cat)} events in the catalogue window "
          f"[{WINDOW_START}, {WINDOW_END}]")

    grid, us_df = load_unit_source_table()
    print(f"  {len(us_df)} unit sources to match against")
    ref_lon = float(np.mean(grid[:, 0, :]))

    print(f"\n  applying: buffer<={BUFFER_DEG} deg, |rake-90|<={RAKE_TOL_DEG} deg, "
          f"|strike diff|<={STRIKE_TOL_DEG} deg, depth<={MAX_DEPTH_KM} km, "
          f"Mw>={THRESHOLD_MW}")
    subset = select_events(cat, grid, ref_lon)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    subset.to_csv(OUT_CSV, index=False)

    # --- v9: LEVEL 0, one catalogue per segment -------------------------
    # PTHA18's rule (compute_rates_all_sources.R 183-195 ->
    # gcmt_subsetter.R 106-142): each segment gets the events the SAME rule
    # selects on its own columns. The 0.4 deg buffers of neighbouring
    # segments overlap, so an event near a boundary can belong to both
    # (PTHA18 flags it double_counted and keeps it in both: kermadectonga2's
    # _tonga 4 + _kermadec 6 = 10 against 9 for the zone), and a segment's
    # own nearest-strike test can keep or drop an event the zone's did not.
    if SEGMENTED:
        import json
        sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))
        import segmentation as sg

        print("\n  LEVEL 0: the same rule on each segment's own columns")
        segments = sg.zone_segments(grid, PTHA18_ZONE_NAME, SEGMENT_BOUNDARIES,
                                    log=lambda m: print(m))
        if len(segments) < 2:
            print("  -> a single segment here (" + SEGMENT_BOUNDARIES + "), so "
                  "there is nothing to segment; step 6 will run this zone "
                  "unsegmented.")
            segments = []
        else:
            per_seg = {}
            for seg in segments:
                a, b = seg["alongstrike_slice"]
                per_seg[seg["segment_key"]] = select_events(
                    cat, grid[:, :, (a - 1):(b + 1)], ref_lon)
            key_of = lambda r: (r["date"], r["hypo_lat"], r["hypo_lon"])  # noqa: E731
            counts = {}
            for v in per_seg.values():
                for _, r in v.iterrows():
                    counts[key_of(r)] = counts.get(key_of(r), 0) + 1
            for seg in segments:
                k = seg["segment_key"]
                path = OUT_CSV.replace(".csv", "_" + k + ".csv")
                per_seg[k].to_csv(path, index=False)
                a, b = seg["alongstrike_slice"]
                n_dbl = sum(counts[key_of(r)] > 1 for _, r in per_seg[k].iterrows())
                print(f"    {k:<10} columns {a}-{b}: {len(per_seg[k])} event(s)"
                      + (f", {n_dbl} also in a neighbouring segment" if n_dbl else ""))
            total = sum(len(v) for v in per_seg.values())
            print(f"    segments hold {total} event(s), the zone {len(subset)} "
                  f"(an event near a boundary counts in both, as in PTHA18)")

            with open(os.path.join(GCMT_DIR, "segments.json"), "w") as f:
                json.dump([{**s, "n_events": len(per_seg[s["segment_key"]])}
                           for s in segments], f, indent=2)

    duration_years = (pd.Timestamp(WINDOW_END) - pd.Timestamp(WINDOW_START)).days / 365.25
    print(f"\n  {len(subset)} event(s) above Mw {THRESHOLD_MW:g} qualify")
    if len(subset):
        print("  magnitudes:", ", ".join(f"{m:.3f}" for m in subset["Mw"]))
    print(f"  duration: {duration_years:.4f} years")

    off_count, off_duration = read_official_gcmt_count() if USE_PTHA else (None, None)
    if off_count is not None:
        print(f"\n  official PTHA18 count for this zone (from step 8): "
             f"{off_count} events in {off_duration:.4f} years")
        print("  (see the note above on why an exact match is not the bar "
             "here, and step4_fetch_gcmt.py's docstring on why the window "
             "above may not match the official one either)")
    elif USE_PTHA:
        print("\n  (run step8_official.py first for a comparison against "
             "PTHA18's own GCMT count)")

    print(f"\n  saved -> {OUT_CSV}")
    print("\nNext:  step6_write_input.py")


if __name__ == "__main__":
    sys.exit(main())
