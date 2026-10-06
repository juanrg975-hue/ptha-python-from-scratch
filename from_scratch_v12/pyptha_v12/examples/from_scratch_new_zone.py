"""FROM SCRATCH: ruptures + individual scenario rates for a NEW subduction zone.

This is a fully-commented, from-zero template. Every step shows the R
(rptha) line it mirrors, so you can see exactly how pyptha maps onto the
R workflow. Adapt the INPUT block to your own zone and run it.

============================================================================
HOW THE R WORKFLOW MAPS ONTO THIS SCRIPT
============================================================================

In R the job is done in two phases with two scripts:

  Phase 1 - geometry  (source_contours_2_unit_sources/produce_unit_sources.R)
      contours.shp  ->  downdip lines  ->  3D unit-source grid
  Phase 2 - rates     (event_rates/single_source_rate_computation.R)
      grid -> summary stats -> ruptures -> conditional probs -> rate model

pyptha reproduces Phase 2 exactly (steps 2-7 below). For Phase 1, the full
"read a fault-contour shapefile and discretise it" is the heavy spherical-
geometry code that is NOT ported; here we build the grid directly from a few
numbers with make_planar_unit_source_grid (step 1). If you already have a grid
from R, load it instead (see the note at step 1).

Run:  .venv/Scripts/python from_scratch_v12/pyptha_v12/examples/from_scratch_new_zone.py
============================================================================
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import events, rates, unit_sources as us  # noqa: E402


# ===========================================================================
# INPUT  -- everything you, the analyst, decide for the new zone.
# In R these are the parameters at the top of the two driver scripts.
# ===========================================================================

# --- Geometry of the interface (Phase 1 inputs) ---------------------------
# In R this comes from a contour shapefile + desired unit-source size. Here we
# describe an idealised planar interface with the same size parameters.
ZONE_NAME = "my_new_zone"
TRENCH_LON = 165.0        # longitude of the shallow (trench) corner, deg
TRENCH_LAT = -45.0        # latitude  of the shallow (trench) corner, deg
STRIKE = 25.0             # strike, deg clockwise from north
DIP = 15.0                # dip, deg below horizontal
SUBFAULT_LENGTH = 50.0    # km  (R: desired_subfault_length)
SUBFAULT_WIDTH = 50.0     # km  (R: desired_subfault_width)
N_ALONGSTRIKE = 10        # number of unit sources along strike
N_DOWNDIP = 5             # number of unit sources down dip
TOP_DEPTH = 5.0           # depth of the shallowest (trench) edge, km

# --- Seismicity / rate-model inputs (Phase 2 inputs) ----------------------
# These are exactly the INPUT PARAMETERS block of single_source_rate_computation.R.

# Seismically-coupled convergence (m/yr), dip-adjusted, as a small logic tree.
# R: slip_rate = (1/1000)*c(40,45,50)*1/cos(dip_rad)
slip_rate = (1 / 1000) * np.array([40.0, 45.0, 50.0]) / np.cos(np.radians(DIP))
slip_rate_prob = [1 / 3, 1 / 3, 1 / 3]

# Gutenberg-Richter b-value with weights.  R: b = c(0.7, 0.95, 1.2)
b = [0.7, 0.95, 1.2]
b_prob = [1 / 3, 1 / 3, 1 / 3]

# Smallest modelled magnitude (analyst choice).  R: Mw_min = 7.5
Mw_min = [7.5]
Mw_min_prob = [1.0]

# Maximum magnitude, weighted.  R: Mw_max = c(9.3, 9.3, 9.4)
Mw_max = [9.3, 9.3, 9.4]
Mw_max_prob = [0.1, 0.45, 0.45]

# Magnitude-frequency distribution choices.  R: Mw_frequency_dists = c(...)
Mfd = ["truncated_gutenberg_richter", "characteristic_gutenberg_richter"]
Mfd_prob = [0.5, 0.5]

# Magnitude increment for the synthetic event set.  R: dMw = 0.1
dMw = 0.1

# Historical data to update the logic-tree weights, as (threshold Mw, count,
# years). Use None to skip the update.  R: Mw_count_durations_CMT = c(7.5, 0, 38)
Mw_count_duration = (7.5, 0, 38)


# ===========================================================================
# WORKFLOW
# ===========================================================================

def main():
    print(f"=== Building source zone '{ZONE_NAME}' from scratch ===\n")

    # -----------------------------------------------------------------------
    # STEP 1 - Geometry: build the unit-source grid.
    #
    # R (Phase 1):
    #     discrete_source = discretized_source_from_source_contours(
    #         'CONTOURS/zone.shp', desired_subfault_length, desired_subfault_width,
    #         downdip_lines = ds1)
    #
    # pyptha: build the grid directly from parameters. (To use a grid that R
    # already produced, load its array instead of calling this, e.g. from a
    # saved .npy, and skip straight to step 2.)
    # -----------------------------------------------------------------------
    grid = us.make_planar_unit_source_grid(
        lon0=TRENCH_LON, lat0=TRENCH_LAT, strike=STRIKE, dip=DIP,
        n_alongstrike=N_ALONGSTRIKE, n_downdip=N_DOWNDIP,
        subfault_length=SUBFAULT_LENGTH, subfault_width=SUBFAULT_WIDTH,
        top_depth=TOP_DEPTH)
    print(f"[1] Grid built: {grid.shape[0]-1} x {grid.shape[2]-1} unit sources "
          f"(down-dip x along-strike)")

    # -----------------------------------------------------------------------
    # STEP 2 - Per-unit-source summary statistics.
    #
    # R:  stats = discretized_source_approximate_summary_statistics(discrete_source)
    # -----------------------------------------------------------------------
    stats = us.discretized_source_approximate_summary_statistics(grid)
    source_area = float(np.sum(stats["length"] * stats["width"]))
    print(f"[2] Summary stats: {stats['subfault_number'].size} unit sources, "
          f"mean dip {stats['dip'].mean():.1f} deg, total area {source_area:,.0f} km^2")

    # -----------------------------------------------------------------------
    # STEP 3 - Generate ALL uniform-slip ruptures over the magnitude range.
    #
    # R:  all_eq = get_all_earthquake_events(
    #         unit_source_statistics = stats, Mmin = min(Mw_min),
    #         Mmax = max(Mw_max), dMw = dMw)
    # -----------------------------------------------------------------------
    all_eq = events.get_all_earthquake_events(
        stats, Mmin=min(Mw_min), Mmax=max(Mw_max), dMw=dMw,
        source_zone_name=ZONE_NAME)
    print(f"[3] Ruptures generated: {all_eq['Mw'].size} events "
          f"(Mw {all_eq['Mw'].min():.2f}..{all_eq['Mw'].max():.2f})")

    # -----------------------------------------------------------------------
    # STEP 4 - Conditional probability of each event within its magnitude bin.
    #
    # R:  ecp = get_event_probabilities_conditional_on_Mw(all_eq, 'inverse_slip')
    # -----------------------------------------------------------------------
    event_table = {"Mw": all_eq["Mw"], "slip": all_eq["slip"], "area": all_eq["area"]}
    ecp = rates.get_event_probabilities_conditional_on_Mw(event_table, "inverse_slip")
    print(f"[4] Conditional probabilities assigned (sum per Mw bin = 1)")

    # -----------------------------------------------------------------------
    # STEP 5 - The Mw-exceedance-rate model over the full logic tree.
    #
    # R:  rate_fn = rate_of_earthquakes_greater_than_Mw_function(
    #         slip_rate, slip_rate_prob, b, b_prob, Mw_min, Mw_min_prob,
    #         Mw_max, Mw_max_prob, sourcezone_total_area = source_area,
    #         event_table = all_eq, event_conditional_probabilities = ecp,
    #         Mw_frequency_distribution = Mfd, ...,
    #         update_logic_tree_weights_with_data = TRUE,
    #         Mw_count_duration = Mw_count_durations_CMT,
    #         account_for_moment_below_mwmin = TRUE)
    # -----------------------------------------------------------------------
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=slip_rate, slip_rate_prob=slip_rate_prob,
        b=b, b_prob=b_prob,
        Mw_min=Mw_min, Mw_min_prob=Mw_min_prob,
        Mw_max=Mw_max, Mw_max_prob=Mw_max_prob,
        sourcezone_total_area=source_area,
        event_table=event_table, event_conditional_probabilities=ecp,
        Mw_frequency_distribution=Mfd, Mw_frequency_distribution_prob=Mfd_prob,
        update_logic_tree_weights_with_data=(Mw_count_duration is not None),
        Mw_count_duration=Mw_count_duration,
        account_for_moment_below_mwmin=True)
    branches = rate_fn(return_all_logic_tree_branches=True)
    print(f"[5] Rate model built: {branches.all_par_prob.size} logic-tree branches")

    # -----------------------------------------------------------------------
    # STEP 6 - Query the source-zone Mw-exceedance curve.
    #
    # R:  rate_of_events_gt_9 = rate_fn(9.0)          # mean
    #     rate_fn(9.0, bounds=TRUE)                    # + percentiles
    # -----------------------------------------------------------------------
    print("\n[6] Mw-exceedance rates (events / year):")
    for mw in (7.5, 8.0, 8.5, 9.0):
        r = float(rate_fn(mw))
        rp = (1.0 / r) if r > 0 else np.inf
        print(f"      Mw > {mw:.1f}:  {r:.3e} / yr   (return period {rp:,.0f} yr)")
    band = rate_fn(np.array([9.0]), bounds=True)
    print(f"      Mw > 9.0 uncertainty:  lower {float(band['lower'][0]):.2e}  "
          f"mean {float(band['rate'][0]):.2e}  upper {float(band['upper'][0]):.2e}")

    # -----------------------------------------------------------------------
    # STEP 7 - Individual scenario rates.
    #
    # R:  all_eq_event_rates = event_conditional_prob * (
    #         rate_fn(all_eq$Mw - dMw/2) - rate_fn(all_eq$Mw + dMw/2))
    # -----------------------------------------------------------------------
    mw = all_eq["Mw"]
    bin_rate = rate_fn(mw - dMw / 2) - rate_fn(mw + dMw / 2)
    scenario_rate = ecp * bin_rate
    print(f"\n[7] Individual scenario rates computed for all "
          f"{scenario_rate.size} scenarios.")
    # Defining property (PTHA18): within each magnitude bin, the scenario rates
    # sum to that bin's rate. Verify it for one bin.
    m0 = np.unique(np.round(mw, 4))[3]           # some magnitude bin
    got = scenario_rate[np.isclose(mw, m0)].sum()
    expect = float(rate_fn(m0 - dMw / 2) - rate_fn(m0 + dMw / 2))
    print(f"      Check bin Mw={m0:.2f}: sum of its scenario rates {got:.3e} "
          f"== bin rate {expect:.3e}  ({np.isclose(got, expect)})")

    # -----------------------------------------------------------------------
    # What you now have (adapt / save as you need):
    #   all_eq          - the rupture table (Mw, area, slip, event_index_string)
    #   scenario_rate   - annual rate of each individual scenario
    #   rate_fn         - callable Mw-exceedance-rate model (+ logic tree)
    #   stats           - unit-source geometry
    # -----------------------------------------------------------------------
    print("\n=== Done. Outputs available: all_eq, scenario_rate, rate_fn, stats ===")
    return {"all_eq": all_eq, "scenario_rate": scenario_rate,
            "rate_fn": rate_fn, "stats": stats}


if __name__ == "__main__":
    main()
