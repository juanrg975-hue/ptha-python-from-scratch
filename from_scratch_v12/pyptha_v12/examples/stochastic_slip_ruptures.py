"""EXAMPLE: heterogeneous (stochastic) slip ruptures for a new zone.

The rate pipeline (from_scratch_new_zone.py) uses UNIFORM slip: every unit
source in a rupture gets the same slip. Real earthquakes are patchy — slip
concentrates in "asperities". This example shows how to turn a uniform-slip
rupture into many HETEROGENEOUS-slip variants with the SFFM generator
(pyptha.stochastic_slip), the same S_{NCF} method as rptha / PTHA18.

Why you'd want this:
  * more realistic tsunami initial conditions (peak slip drives peak uplift),
  * an ensemble of slip patterns for one magnitude, to sample variability.

Each generated field keeps the SAME total slip as the uniform rupture, so the
seismic moment (and hence Mw) is preserved — only the spatial pattern changes.

Run:  .venv/Scripts/python from_scratch_v12/pyptha_v12/examples/stochastic_slip_ruptures.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyptha_v12 import events, stochastic_slip as ss, unit_sources as us  # noqa: E402


# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------

# Geometry of the new zone (planar, as in from_scratch_new_zone.py).
STRIKE, DIP = 25.0, 15.0
N_ALONGSTRIKE, N_DOWNDIP = 12, 6
SUBFAULT_L, SUBFAULT_W = 50.0, 45.0

TARGET_MW = 8.6            # magnitude of the rupture we want slip variants of
N_VARIANTS = 3            # how many heterogeneous realisations to generate

# Corner-wavenumber control on roughness. Lower -> smoother/broader asperity,
# higher -> more concentrated peak slip. ~0.15-0.3 (numerical units) is a
# reasonable range; this is the (kcx*dx, kcy*dy) pair sffm expects.
CORNER_WAVENUMBER = 0.18


def ascii_heatmap(mat, title):
    """Print a slip field as a compact ASCII heat map (no plotting deps)."""
    chars = " .:-=+*#%@"
    m = np.asarray(mat, dtype=float)
    hi = m.max() if m.max() > 0 else 1.0
    print(f"\n  {title}  (peak {m.max():.1f} m, mean {m.mean():.1f} m)")
    for row in m:
        line = "".join(chars[min(len(chars) - 1, int(v / hi * (len(chars) - 1)))]
                        for v in row)
        print(f"    |{line}|")


def rupture_to_template(all_eq, ev_i, stats):
    """Build the (down-dip x along-strike) slip template for one rupture.

    Returns the template matrix (uniform slip inside the rupture footprint,
    zero outside) plus the grid indices it occupies, so results can be mapped
    back to unit sources.
    """
    idx = all_eq["event_indices"][ev_i]
    uniform_slip = all_eq["slip"][ev_i]
    dd = stats["downdip_number"][idx].astype(int)
    ask = stats["alongstrike_number"][idx].astype(int)
    dd0, as0 = dd.min(), ask.min()
    nd, na = dd.max() - dd0 + 1, ask.max() - as0 + 1
    tg = np.zeros((nd, na))
    tg[dd - dd0, ask - as0] = uniform_slip
    return tg, uniform_slip, (dd0, as0, nd, na)


def main():
    # 1. Geometry + unit-source statistics (same as the rate workflow).
    grid = us.make_planar_unit_source_grid(
        165.0, -45.0, STRIKE, DIP, N_ALONGSTRIKE, N_DOWNDIP,
        SUBFAULT_L, SUBFAULT_W)
    stats = us.discretized_source_approximate_summary_statistics(grid)

    # 2. Get the uniform-slip ruptures at the target magnitude, pick one.
    all_eq = events.get_all_earthquake_events(
        stats, Mmin=TARGET_MW, Mmax=TARGET_MW, dMw=0.1, source_zone_name="demo")
    ev_i = len(all_eq["Mw"]) // 2                      # a central placement
    tg, uniform_slip, (dd0, as0, nd, na) = rupture_to_template(all_eq, ev_i, stats)
    print(f"Rupture: Mw {all_eq['Mw'][ev_i]:.2f}, {tg.size} unit sources, "
          f"uniform slip {uniform_slip:.2f} m, total slip {tg.sum():.1f} m")

    ascii_heatmap(tg, "UNIFORM slip (starting point)")

    # 3. Generate several heterogeneous realisations.
    #    Give the template a peak location so recentre_slip has a target; the
    #    generator preserves the total slip, so Mw is unchanged.
    tg_peak = tg.copy()
    tg_peak[nd // 2, na // 2] = uniform_slip * 1.5
    reg_par = (CORNER_WAVENUMBER, CORNER_WAVENUMBER)

    for v in range(N_VARIANTS):
        het = ss.sffm_simulate(reg_par, tg_peak, rng=np.random.default_rng(v))
        assert np.isclose(het.sum(), tg_peak.sum())    # moment preserved
        ascii_heatmap(het, f"HETEROGENEOUS realisation #{v + 1}")

    print(f"\nEach realisation keeps total slip = {tg_peak.sum():.1f} m "
          f"(same Mw), only the pattern differs.")
    print("Map a field back to unit sources with the (downdip, alongstrike) "
          "offsets: unit source at grid row dd0+r, col as0+c gets het[r, c].")

    # 4. (Optional) turn peak-slip location into a tsunami source:
    #    feed each unit source's slip + geometry to okada.okada_tsunami to get
    #    the seafloor deformation. See the guide's "Deformation" section.
    return {"template": tg, "stats": stats, "rupture_index": ev_i}


if __name__ == "__main__":
    main()
