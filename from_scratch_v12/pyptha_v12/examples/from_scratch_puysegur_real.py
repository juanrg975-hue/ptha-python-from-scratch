"""FROM SCRATCH with a REAL zone: Puysegur, starting from its fault contours.

Same from-zero workflow as ``from_scratch_new_zone.py``, but the geometry now
comes from the REAL Puysegur depth contours shipped with rptha
(``rptha/R/rptha/inst/extdata/puysegur.shp``) instead of an idealised planar patch.

This shows the closest thing to the full R workflow that pyptha can do today:
we read the actual contour shapefile (as R does) and discretise it into a
unit-source grid with a SIMPLIFIED discretiser
(``discretized_source_from_contours``). That discretiser does not reproduce
rptha's down-dip-line orthogonalisation, so the grid is an approximation of
rptha's, but it is built from the same real fault geometry and drives the
identical rupture + rate pipeline.

    R Phase 1:  readOGR('puysegur.shp')  ->  discretized_source_from_source_contours()
    here:       geopandas.read_file(...)  ->  discretized_source_from_contours()   [simplified]
    R Phase 2:  summary stats -> events -> conditional probs -> rate model   (identical)

Run:  .venv/Scripts/python from_scratch_v12/pyptha_v12/examples/from_scratch_puysegur_real.py

Requires geopandas (already in requirements.txt).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import events, rates, unit_sources as us  # noqa: E402

CONTOURS_SHP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "rptha", "R", "rptha", "inst", "extdata", "puysegur.shp"))

# ---------------------------------------------------------------------------
# INPUT -- geometry now comes from the shapefile; only the discretisation
# resolution and the seismicity parameters are analyst choices.
# ---------------------------------------------------------------------------

ZONE_NAME = "puysegur"
N_ALONGSTRIKE = 20        # unit sources along strike (R uses ~50 km spacing)
N_DOWNDIP = 8             # unit sources down dip (contours span 0..40 km)

# Seismicity inputs (Puysegur-flavoured; adjust to your best estimates).
slip_rate = (1 / 1000) * np.array([25.0, 30.0, 35.0])   # m/yr, coupled
slip_rate_prob = [1 / 3, 1 / 3, 1 / 3]
b = [0.7, 0.95, 1.2]
b_prob = [1 / 3, 1 / 3, 1 / 3]
Mw_min = [7.5]
Mw_min_prob = [1.0]
Mw_max = [8.7, 9.0]
Mw_max_prob = [0.5, 0.5]
Mfd = ["truncated_gutenberg_richter", "characteristic_gutenberg_richter"]
Mfd_prob = [0.5, 0.5]
dMw = 0.1
Mw_count_duration = (7.5, 1, 38)   # ~1 event Mw>=7.5 in 38 yr (illustrative)


def load_contours():
    """Read the Puysegur depth contours as a list of (depth_km, coords)."""
    import geopandas as gpd
    gdf = gpd.read_file(CONTOURS_SHP)
    gdf["level"] = gdf["level"].astype(float)
    out = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "LineString":
            coords = np.array(geom.coords)
        else:  # MultiLineString: take the longest part
            parts = sorted(geom.geoms, key=lambda g: len(g.coords))
            coords = np.array(parts[-1].coords)
        out.append((row["level"], coords[:, :2]))
    return out


def main():
    print(f"=== Building REAL zone '{ZONE_NAME}' from its fault contours ===\n")

    # STEP 0 - read the real contour shapefile (R: readOGR('puysegur.shp')).
    contours = load_contours()
    depths = sorted(d for d, _ in contours)
    print(f"[0] Read {len(contours)} depth contours: {depths[0]:.0f}..{depths[-1]:.0f} km")

    # STEP 1 - discretise contours into a unit-source grid (simplified).
    # R: discretized_source_from_source_contours('puysegur.shp', 50, 50, downdip_lines)
    grid = us.discretized_source_from_contours(
        contours, n_alongstrike=N_ALONGSTRIKE, n_downdip=N_DOWNDIP)
    print(f"[1] Grid: {grid.shape[0]-1} x {grid.shape[2]-1} unit sources "
          f"(down-dip x along-strike) from real geometry")

    # STEP 2 - summary statistics.
    stats = us.discretized_source_approximate_summary_statistics(grid)
    source_area = float(np.sum(stats["length"] * stats["width"]))
    print(f"[2] {stats['subfault_number'].size} unit sources, "
          f"dip {stats['dip'].min():.0f}..{stats['dip'].max():.0f} deg, "
          f"mean length {stats['length'].mean():.0f} km, "
          f"width {stats['width'].mean():.0f} km, area {source_area:,.0f} km^2")

    # STEP 3 - all uniform-slip ruptures.
    all_eq = events.get_all_earthquake_events(
        stats, Mmin=min(Mw_min), Mmax=max(Mw_max), dMw=dMw,
        source_zone_name=ZONE_NAME)
    print(f"[3] {all_eq['Mw'].size} ruptures (Mw "
          f"{all_eq['Mw'].min():.2f}..{all_eq['Mw'].max():.2f})")

    # STEP 4 - conditional probabilities.
    event_table = {"Mw": all_eq["Mw"], "slip": all_eq["slip"], "area": all_eq["area"]}
    ecp = rates.get_event_probabilities_conditional_on_Mw(event_table, "inverse_slip")

    # STEP 5 - rate model over the logic tree, with Bayesian weight update.
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=slip_rate, slip_rate_prob=slip_rate_prob,
        b=b, b_prob=b_prob, Mw_min=Mw_min, Mw_min_prob=Mw_min_prob,
        Mw_max=Mw_max, Mw_max_prob=Mw_max_prob, sourcezone_total_area=source_area,
        event_table=event_table, event_conditional_probabilities=ecp,
        Mw_frequency_distribution=Mfd, Mw_frequency_distribution_prob=Mfd_prob,
        update_logic_tree_weights_with_data=True,
        Mw_count_duration=Mw_count_duration, account_for_moment_below_mwmin=True)
    branches = rate_fn(return_all_logic_tree_branches=True)
    print(f"[5] Rate model: {branches.all_par_prob.size} logic-tree branches")

    # STEP 6 - Mw-exceedance curve.
    print("\n[6] Mw-exceedance rates (events / year):")
    for mw in (7.5, 8.0, 8.5):
        r = float(rate_fn(mw))
        print(f"      Mw > {mw:.1f}:  {r:.3e} / yr   (return period {1/r:,.0f} yr)")

    # STEP 7 - individual scenario rates.
    mw = all_eq["Mw"]
    scenario_rate = ecp * (rate_fn(mw - dMw / 2) - rate_fn(mw + dMw / 2))
    print(f"\n[7] Individual scenario rates for all {scenario_rate.size} scenarios.")
    m0 = np.unique(np.round(mw, 4))[3]
    got = scenario_rate[np.isclose(mw, m0)].sum()
    expect = float(rate_fn(m0 - dMw / 2) - rate_fn(m0 + dMw / 2))
    print(f"      Check bin Mw={m0:.2f}: scenario-rate sum {got:.3e} "
          f"== bin rate {expect:.3e}  ({np.isclose(got, expect)})")

    print("\n=== Done: real-geometry ruptures + scenario rates for Puysegur ===")
    return {"all_eq": all_eq, "scenario_rate": scenario_rate,
            "rate_fn": rate_fn, "stats": stats, "grid": grid}


if __name__ == "__main__":
    main()
