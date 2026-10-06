"""End-to-end example: ruptures + individual scenario rates for one source.

Python equivalent of rptha's
``R/examples/event_rates/single_source_rate_computation.R``. It shows the full
new-zone workflow using only the ported pyptha modules:

    1. build a unit-source grid for the interface        (unit_sources)
    2. summary statistics per unit source                (unit_sources)
    3. generate all uniform-slip ruptures                (events)
    4. conditional probabilities within each Mw bin      (rates)
    5. the Mw-exceedance-rate model over the logic tree  (rates)
    6. individual scenario rates                          (rates + events)

The R example uses real Alaska unit sources and quotes rate(Mw>9) ~ 1/540.
Here we use an idealised planar interface, so the numbers differ, but the
procedure and every function called are the direct analogue of the R script.

Run:  python from_scratch_v12/pyptha_v12/examples/single_source_rate_computation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import events, rates, unit_sources as us  # noqa: E402

# --------------------------------------------------------------------------
# INPUT PARAMETERS (the analyst supplies these per new source zone)
# --------------------------------------------------------------------------

SOURCENAME = "example_zone"

# Interface geometry (idealised planar patch). In a real study this comes from
# discretising fault contours; here we build it directly.
DIP = 15.0            # degrees
STRIKE = 20.0         # degrees
N_ALONGSTRIKE = 12    # unit sources along strike
N_DOWNDIP = 6         # unit sources down dip
SUBFAULT_L = 50.0     # km
SUBFAULT_W = 45.0     # km

# Seismically-coupled convergence (m/yr), dip-adjusted, with a small logic tree.
DIP_FOR_SLIP = DIP
slip_rate = (1 / 1000) * np.array([40.0, 45.0, 50.0]) / np.cos(np.radians(DIP_FOR_SLIP))
slip_rate_prob = [1 / 3, 1 / 3, 1 / 3]

# Gutenberg-Richter b, with weights (Berryman et al. 2015 values as in the R).
b = [0.7, 0.95, 1.2]
b_prob = [1 / 3, 1 / 3, 1 / 3]

Mw_min = [7.5]
Mw_min_prob = [1.0]
Mw_max = [9.3, 9.3, 9.4]
Mw_max_prob = [0.1, 0.45, 0.45]

Mw_frequency_dists = ["truncated_gutenberg_richter", "characteristic_gutenberg_richter"]
Mw_frequency_dists_prob = [0.5, 0.5]

dMw = 0.1

# Observed count of Mw>=7.5 events over 38 years (CMT-like), to update weights.
Mw_count_duration = (7.5, 0, 38)

# --------------------------------------------------------------------------
# WORKFLOW
# --------------------------------------------------------------------------


def main():
    # 1-2. Geometry and unit-source summary statistics.
    grid = us.make_planar_unit_source_grid(
        lon0=200.0, lat0=54.0, strike=STRIKE, dip=DIP,
        n_alongstrike=N_ALONGSTRIKE, n_downdip=N_DOWNDIP,
        subfault_length=SUBFAULT_L, subfault_width=SUBFAULT_W)
    stats = us.discretized_source_approximate_summary_statistics(grid)
    source_area = float(np.sum(stats["length"] * stats["width"]))
    print(f"Source zone '{SOURCENAME}': {stats['subfault_number'].size} unit "
          f"sources, total area {source_area:,.0f} km^2")

    # 3. All uniform-slip ruptures over the magnitude range.
    all_eq = events.get_all_earthquake_events(
        stats, Mmin=min(Mw_min), Mmax=max(Mw_max), dMw=dMw,
        source_zone_name=SOURCENAME)
    n_events = all_eq["Mw"].size
    print(f"Generated {n_events} uniform-slip ruptures "
          f"(Mw {all_eq['Mw'].min():.2f} to {all_eq['Mw'].max():.2f})")

    # 4. Conditional probabilities within each magnitude bin (inverse-slip).
    event_table = {"Mw": all_eq["Mw"], "slip": all_eq["slip"], "area": all_eq["area"]}
    ecp = rates.get_event_probabilities_conditional_on_Mw(event_table, "inverse_slip")

    # 5. Mw-exceedance-rate model over the full logic tree, with the Bayesian
    #    weight update from the observed event count.
    rate_fn = rates.rate_of_earthquakes_greater_than_Mw_function(
        slip_rate=slip_rate, slip_rate_prob=slip_rate_prob,
        b=b, b_prob=b_prob,
        Mw_min=Mw_min, Mw_min_prob=Mw_min_prob,
        Mw_max=Mw_max, Mw_max_prob=Mw_max_prob,
        sourcezone_total_area=source_area,
        event_table=event_table, event_conditional_probabilities=ecp,
        Mw_frequency_distribution=Mw_frequency_dists,
        Mw_frequency_distribution_prob=Mw_frequency_dists_prob,
        update_logic_tree_weights_with_data=True,
        Mw_count_duration=Mw_count_duration,
        account_for_moment_below_mwmin=True)

    # Example query: rate and return period of Mw > 9.0.
    r9 = float(rate_fn(9.0))
    print(f"Rate of Mw > 9.0: {r9:.3e} / yr  (return period {1 / r9:,.0f} yr)")

    # 6. Individual scenario rates: spread each magnitude bin's rate over its
    #    events using the conditional probabilities.
    mw = all_eq["Mw"]
    bin_rate = rate_fn(mw - dMw / 2) - rate_fn(mw + dMw / 2)
    scenario_rate = ecp * bin_rate
    print(f"Summed scenario rate (Mw > 7.85): "
          f"{scenario_rate[mw > 7.85].sum():.3e} / yr")
    print(f"Total individual scenarios with a rate: {np.sum(scenario_rate > 0)}")

    # Logic-tree branches are available for uncertainty work.
    branches = rate_fn(return_all_logic_tree_branches=True)
    print(f"Logic-tree branches: {branches.all_par_prob.size} "
          f"(weights sum to {branches.all_par_prob.sum():.3f})")

    return {"rate_fn": rate_fn, "event_table": all_eq,
            "scenario_rate": scenario_rate, "stats": stats}


if __name__ == "__main__":
    main()
