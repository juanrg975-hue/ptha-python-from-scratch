"""PTHA18 logic tree for one source zone -> CSVs (pyptha / Python).

Version 3. This supersedes individual_codes_v2, fixing the defects found when
the pipeline was audited against the primary source (Davies, G. & Griffin, J.,
2018, "The 2018 Australian probabilistic tsunami hazard assessment: hazard from
earthquake generated tsunamis", Geoscience Australia Record 2018/41) and against
the PTHA18 driver scripts in R/examples/austptha_template/EVENT_RATES.

What changed relative to v2, and why:

  FIXED  Segment weighting. v2 gave each segment `union_of_segments_weight /
         n_segments`, which treats segments as mutually exclusive. They are
         not: the union-of-segments branch SUMS its segments (report Section
         3.7.6). Each segment now carries the full 0.5.

  FIXED  Dip correction. Tectonic convergence is horizontal; earthquake slip
         lies in the fault plane. PTHA18 divides the convergence rate by
         cos(mean_dip) before the moment balance. v2 omitted this, which
         under-estimated the GR 'a' by 1/cos(dip): 15% at 30 deg dip, 41% at
         45 deg. Now applied, and reported.

  NEW    Seismic coupling is a logic-tree axis in its own right (report Section
         3.7.2.3), with the composite 50/50 prior: half from the Berryman et
         al. (2015) lower/preferred/upper triple read as a CDF, half uniform
         over a wider range. v2 folded coupling into a hand-written slip rate,
         which cannot separate "how fast the plates converge" from "what
         fraction is released seismically".

  NEW    Parameter grids at PTHA18 resolution. The report uses 20 b-values, 20
         coupling values, 40 Mw_max values and 2 GR types = 32000 branches per
         zone. v2 ran 27. Grids are now sub-sampled from anchor values to any
         requested resolution.

  NEW    The characteristic GR branch (70/30 truncated/characteristic, report
         Section 3.7.1). Implemented in pyptha all along, never exercised.

  NEW    LEVEL 4 is fitted, not typed. The report specifies the edge multiplier
         as "determined numerically to give the best agreement" between
         integrated slip and the convergence pattern (Section 3.7.1.2). v2 made
         it a hand-entered constant, left at 0. Now fitted by least squares over
         [0, 30], with segments inheriting the unsegmented zone's value exactly
         as PTHA18 does.

  NEW    Integrated slip per unit source is computed and written out. This is
         the central check that seismic moment conservation holds spatially
         (report Figures 38, 39, 45). v2 computed it nowhere.

  NEW    Individual scenario rates r_j (report Equation 3) are written out, at
         the logic-tree mean and at each requested percentile. This is the
         actual end product of a rate calculation and v2 never emitted it.

  NEW    LEVEL 3 is exercised with real GCMT-style data (count, duration and
         observed magnitudes), instead of being switched off. The observation
         threshold defaults to Mw_min - dMw/2 = 7.15, as the report specifies,
         rather than Mw_min.

No tsunami component is involved at any point: nothing here reads wave heights,
rasters or NetCDF. In PTHA18 the LEVEL 5 percentiles are tabulated against
tsunami wave-height thresholds; that axis is kept free ("threshold_stages" in
the R original) and this pipeline feeds it magnitude thresholds, so the whole
tree runs end to end from a JSON file alone.

The R twin (run_logic_tree.R) reads the SAME JSON and writes the same files
under out_r/<run_name>/.

Usage:
    .venv/Scripts/python.exe individual_codes_v3/run_logic_tree.py inputs/input_puysegur_segmented.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# This copy lives at from_scratch_v12/python_logic_tree_v12/, one level
# deeper than the shared python_logic_tree/ it was forked from, so ROOT
# (ptha18_logic_tree_test/, where every path in the input JSON and every
# runs/python/ output is resolved relative to) needs one extra ".." to
# match. See from_scratch_v12/html/docs/code_map.html.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V5_DIR = os.path.join(ROOT, "from_scratch_v12")
sys.path.insert(0, V5_DIR)
sys.path.insert(0, os.path.join(V5_DIR, "lib"))  # helper modules

from pyptha_v12 import events, logic_tree as lt, moment_balance as mb  # noqa: E402
from pyptha_v12 import rates, unit_sources as us  # noqa: E402
from pyptha_v12 import contour_discretisation as cd  # noqa: E402
from pyptha_v12 import grid_cache  # noqa: E402
from pyptha_v12 import variable_mu as vmu  # noqa: E402
import official_geometry_params as ogp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

PY_AMBER = "#B35C00"
PRIOR_GREY = "#8A8580"
TARGET_BLUE = "#1F4E79"
plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25,
})


def save_csv(path, header, cols):
    """Save equal-length columns as a full-double-precision CSV."""
    arr = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    np.savetxt(path, arr, delimiter=",", header=",".join(header),
               comments="", fmt="%.17g")


# ---------------------------------------------------------------------------
# Logic-tree parameter axes (LEVEL 1)
# ---------------------------------------------------------------------------
def build_parameter_axes(rc, mean_dip_deg, stats=None, source_area=None):
    """Turn the JSON rate config into the five logic-tree axes.

    Returns a dict with, for each axis, its values and prior probabilities,
    plus the diagnostics needed to report how the slip rate was built.

    The slip-rate axis is the product of a FIXED tectonic convergence rate and
    the UNCERTAIN seismic coupling, divided by cos(mean dip):

        slip_rate = convergence * coupling / cos(dip)

    which is exactly what compute_rates_all_sources.R does. Coupling carries
    the uncertainty; convergence is a plate-model input.
    """
    nb = rc.get("n_logic_tree_bins", {})

    # --- seismic coupling -> slip rate ---
    conv_mm = float(rc["tectonic_convergence_mm_per_yr"])
    dip_factor, mean_dip = mb.mean_dip_cos_factor(mean_dip_deg)

    cpl_cfg = rc["coupling"]
    n_coupling = int(nb.get("coupling", 20))
    if cpl_cfg["prior_type"] == "spreadsheet_and_uniform_50_50":
        coupling, coupling_p = mb.coupling_prior_spreadsheet_and_uniform_50_50(
            uniform_range=cpl_cfg["uniform_range"],
            spreadsheet_values=cpl_cfg["spreadsheet_values"],
            n=n_coupling,
            prob_zero_coupling=cpl_cfg.get("prob_zero_coupling", 0.0))
    elif cpl_cfg["prior_type"] == "explicit":
        coupling = np.asarray(cpl_cfg["values"], dtype=float)
        coupling_p = np.asarray(cpl_cfg["probs"], dtype=float)
        coupling_p = coupling_p / coupling_p.sum()
    else:
        raise ValueError(f"unknown coupling prior_type {cpl_cfg['prior_type']!r}")

    # mm/yr -> m/yr, then the dip correction.
    slip_rate = (conv_mm / 1000.0) * coupling * dip_factor

    # --- b value ---
    b = mb.interpolate_logic_tree_parameter(rc["b_anchor"], int(nb.get("b", 20)))
    b_prob = np.full(b.size, 1.0 / b.size)

    # --- Mw_max ---
    # "derive" computes the anchors from the geometry the way PTHA18 does,
    # rather than taking hand-typed numbers that silently drift from the
    # official tree. See mb.mw_max_anchor.
    anchor = rc["Mw_max_anchor"]
    if isinstance(anchor, str):
        if anchor != "derive":
            raise ValueError(f"Mw_max_anchor must be a pair or 'derive', got {anchor!r}")
        if stats is None or source_area is None:
            raise ValueError("Mw_max_anchor='derive' needs the source geometry")
        anchor = mb.mw_max_anchor(
            source_area, stats["alongstrike_number"], stats["width"],
            mw_max_observed=float(rc["mw_max_observed"]),
            scaling_relation=rc.get("scaling_relation", "Strasser"))
        print(f"      Mw_max anchor derived from geometry: "
              f"[{anchor[0]:.4f}, {anchor[1]:.4f}]")
    mwmax = mb.interpolate_logic_tree_parameter(
        anchor, int(nb.get("Mw_max", 40)))
    # Clip AFTER interpolation, matching compute_rates_all_sources.R line 358-363.
    # See mb.mw_max_anchor: doing it in the other order changes which values the
    # axis holds whenever the raw anchor exceeds the cap.
    #
    # R branches this cap on rake (compute_rates_all_sources.R:358-363):
    # normal/outer-rise sources (target_rake==-90) get the LOWER cap
    # MAXIMUM_ALLOWED_MW_MAX_NORMAL=9.0; every other rake (thrust, this
    # project's only case -- stats["rake"] defaults to 90.0 everywhere and
    # is never overridden by any template) gets MAXIMUM_ALLOWED_MW_MAX=9.6.
    # Dormant today (no normal-fault zone is discretised by this pipeline),
    # but without this branch a future normal-fault run would silently use
    # the thrust cap and inflate Mw_max/the exceedance-rate tail.
    default_cap = 9.0 if (stats is not None and float(stats["rake"][0]) == -90.0) else 9.6
    mwmax = np.minimum(mwmax, float(rc.get("maximum_allowed_mw_max", default_cap)))
    mwmax_prob = np.full(mwmax.size, 1.0 / mwmax.size)

    # --- Mw_min (single value in PTHA18) ---
    mwmin = np.atleast_1d(np.asarray(rc["Mw_min"], dtype=float))
    mwmin_prob = np.atleast_1d(np.asarray(rc["Mw_min_prob"], dtype=float))

    # --- GR type ---
    mfd = list(rc["Mw_frequency_distribution"])
    mfd_p = np.asarray(rc["Mw_frequency_distribution_prob"], dtype=float)

    return {
        "slip_rate": slip_rate, "slip_rate_prob": coupling_p,
        "coupling": coupling, "coupling_prob": coupling_p,
        "b": b, "b_prob": b_prob,
        "Mw_min": mwmin, "Mw_min_prob": mwmin_prob,
        "Mw_max": mwmax, "Mw_max_prob": mwmax_prob,
        "Mfd": mfd, "Mfd_prob": mfd_p,
        "convergence_mm_per_yr": conv_mm,
        "mean_dip_deg": mean_dip, "dip_factor": dip_factor,
        "n_branches": (slip_rate.size * b.size * mwmin.size * mwmax.size
                       * len(mfd)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_branch_fan(fig_dir, name, br, rate_fn=None,
                      observed_mw=None, observed_rate=None):
    """Report Figures 43 ("before weight update") and 44 ("after weight
    update"), side by side, axes matched to the report (Mw 7.15-9.6,
    exceedance rate 1e-4 to 1e0).

    Both panels draw the same background of all logic-tree branch curves
    (grey), exactly as the report does. The left panel ("before") overlays
    the PRIOR mean curve, matching Figure 43. The right panel ("after")
    overlays the POSTERIOR mean, median, and the 16/84th and 2.5/97.5th
    percentile bands, matching Figure 44 exactly -- pass ``rate_fn`` (the
    branch rate function, still queryable for quantiles after LEVEL 3) to
    get those bands; without it the right panel falls back to the posterior
    mean curve only.

    When ``update_logic_tree_weights_with_data`` is off the two panels are
    identical (posterior = prior, no percentile spread to show); the
    comparison becomes informative once LEVEL 3 actually updates the
    weights.

    Pass ``observed_mw``/``observed_rate`` to overlay the empirical
    (GCMT-style) exceedance rate as the green line/markers the report also
    shows in both figures.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 6), sharex=True, sharey=True)

    # One LineCollection per axis instead of one Line2D per branch (up to
    # 32,000): same rendered image (every branch is still drawn, nothing is
    # subsampled), but matplotlib does not have to manage tens of thousands
    # of individual artists to get there.
    from matplotlib.collections import LineCollection
    segments = [np.column_stack([br.Mw_seq, br.all_rate_matrix[i, :]])
                for i in range(br.all_rate_matrix.shape[0])]
    for ax in axs:
        ax.set_yscale("log")
        ax.add_collection(LineCollection(
            segments, colors="0.6", linewidths=0.3, alpha=0.35, zorder=1))
        if observed_mw is not None and observed_rate is not None:
            ax.semilogy(observed_mw, observed_rate, color="darkgreen", lw=1.8,
                        marker="o", ms=5, label="Data", zorder=4)
        ax.set_xlabel("Magnitude")
        # loc="best" scans every artist's bounding box (including the tens of
        # thousands of branch-curve segments above) to avoid overlap, which is
        # what makes this legend call slow; a fixed corner is exactly as
        # correct here since the branch fan always occupies the same region.
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", ls=":", color="orange", alpha=0.4)
        ax.set_xlim(br.Mw_seq.min(), br.Mw_seq.max())

    # Left: before weight update (prior mean), report Figure 43.
    mean_prior = br.all_par_prob_prior @ br.all_rate_matrix
    axs[0].semilogy(br.Mw_seq, mean_prior, color="red", lw=1.8, marker="o",
                    ms=3.5, markerfacecolor="none", label="prior mean curve",
                    zorder=3)
    axs[0].set_title(f"Mw vs exceedance-rate (before weight update)\n{name}")
    axs[0].legend(fontsize=9, loc="upper right")

    # Right: after weight update (posterior mean + median + percentile
    # bands), report Figure 44.
    mean_post = br.all_par_prob @ br.all_rate_matrix
    if rate_fn is not None:
        pct = rate_fn(br.Mw_seq, quantiles=[0.025, 0.16, 0.5, 0.84, 0.975])
        axs[1].semilogy(br.Mw_seq, pct[2, :], color="black", lw=1.5, ls="--",
                        label="posterior median", zorder=3)
        axs[1].semilogy(br.Mw_seq, pct[1, :], color=PY_AMBER, lw=1.3, ls="--",
                        label="posterior 16/84 percentile", zorder=3)
        axs[1].semilogy(br.Mw_seq, pct[3, :], color=PY_AMBER, lw=1.3, ls="--",
                        zorder=3)
        axs[1].semilogy(br.Mw_seq, pct[0, :], color="purple", lw=1.1, ls="--",
                        label="posterior 2.5/97.5 percentile", zorder=3)
        axs[1].semilogy(br.Mw_seq, pct[4, :], color="purple", lw=1.1, ls="--",
                        zorder=3)
    axs[1].semilogy(br.Mw_seq, mean_post, color="red", lw=1.8, marker="o",
                    ms=3.5, markerfacecolor="none", label="posterior mean curve",
                    zorder=4)
    axs[1].set_title(f"Mw vs exceedance-rate (after weight update)\n{name}")
    axs[1].legend(fontsize=8, loc="upper right")

    axs[0].set_ylabel("Rate of exceedance (events/year)")
    # Match the report's axis (Figures 43/44 label only 1e-4, 1e-2, 1e0, but
    # their mean/percentile curves visibly reach the right edge at Mw 9.6):
    # floor the range on the curves actually drawn, not the full 32000-branch
    # matrix, whose extreme low-b tails can run far below the mean and would
    # otherwise stretch the axis out of proportion to the report's.
    curves = [mean_prior, mean_post]
    if rate_fn is not None:
        curves.append(pct)
    floor_candidates = np.concatenate([np.atleast_1d(c).ravel() for c in curves])
    floor_candidates = floor_candidates[floor_candidates > 0]
    ymin = min(1e-4, np.nanmin(floor_candidates)) if floor_candidates.size else 1e-4
    axs[0].set_ylim(ymin, 1e0)
    fig.suptitle(f"All {br.all_rate_matrix.shape[0]:,} logic-tree branches",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"fig_branch_fan_{name}.png"),
                bbox_inches="tight")
    plt.close(fig)


def figure_branch_weights(fig_dir, name, br, axes):
    """LEVEL 3: how the data moved the weight on each parameter axis."""
    prior = br.all_par_prob_prior
    post = br.all_par_prob

    # Marginal weight per value, for the three axes that carry uncertainty.
    specs = [("b", "b"), ("slip_rate", "coupling"), ("Mw_max", "Mw_max")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, label) in zip(axs, specs):
        vals = np.array([p[key] for p in br.all_par_combo])
        uniq = np.unique(vals)
        pri = np.array([prior[vals == v].sum() for v in uniq])
        pos = np.array([post[vals == v].sum() for v in uniq])
        if label == "coupling":
            # Show coupling, not the derived slip rate: it is the physical axis.
            uniq_plot = uniq / (axes["convergence_mm_per_yr"] / 1000.0
                                * axes["dip_factor"])
        else:
            uniq_plot = uniq
        ax.plot(uniq_plot, pri, "o-", color=PRIOR_GREY, ms=3.5, lw=1.3,
                label="prior")
        ax.plot(uniq_plot, pos, "o-", color=PY_AMBER, ms=3.5, lw=1.7,
                label="posterior")
        ax.set_xlabel(label)
        ax.set_ylabel("Marginal weight")
        ax.set_title(f"Marginal over {label}")
        ax.legend(fontsize=8.5)

    tvd = 0.5 * np.abs(post - prior).sum()
    fig.suptitle(f"LEVEL 3: Bayesian update ({name}) - "
                 f"total-variation distance = {tvd:.4g}",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"fig_branch_weights_{name}.png"),
                bbox_inches="tight")
    plt.close(fig)


def figure_moment_balance(fig_dir, name, stats, integrated_slip_raw,
                          integrated_slip_fit, target, edge_mult):
    """LEVEL 4: integrated slip vs the convergence pattern it should match.

    This is the report's Figure 38/39 check: does the rate model reproduce the
    spatial pattern of tectonic convergence?
    """
    asn = stats["alongstrike_number"]
    n_as = int(asn.max())
    # Collapse down-dip: sum the integrated slip within each along-strike column.
    def by_column(v):
        return np.array([v[asn == i].sum() for i in range(1, n_as + 1)])

    col = np.arange(1, n_as + 1)
    raw = by_column(integrated_slip_raw)
    fit = by_column(integrated_slip_fit)
    tgt = by_column(target)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Normalised shapes: coupling sets the amplitude, only the shape is fitted.
    ax1.plot(col, tgt / tgt.sum(), "o-", color=TARGET_BLUE, lw=2.2, ms=5,
             label="target: tectonic convergence")
    ax1.plot(col, raw / raw.sum(), "s--", color=PRIOR_GREY, lw=1.6, ms=4,
             label="modelled, no edge correction")
    ax1.plot(col, fit / fit.sum(), "^-", color=PY_AMBER, lw=2.0, ms=5,
             label=f"modelled, edge multiplier = {edge_mult:.3f}")
    ax1.set_xlabel("Along-strike unit-source index")
    ax1.set_ylabel("Normalised integrated slip rate")
    ax1.set_title(f"LEVEL 4: seismic moment conservation ({name})\n"
                  "the edge correction lifts the sagging zone ends")
    ax1.legend(fontsize=8.5)

    # Residual against the target shape, the quantity actually minimised.
    ax2.axhline(0, color="k", lw=1)
    ax2.plot(col, raw / raw.sum() - tgt / tgt.sum(), "s--", color=PRIOR_GREY,
             lw=1.6, ms=4, label="no edge correction")
    ax2.plot(col, fit / fit.sum() - tgt / tgt.sum(), "^-", color=PY_AMBER,
             lw=2.0, ms=5, label="fitted")
    ax2.set_xlabel("Along-strike unit-source index")
    ax2.set_ylabel("modelled - target (normalised)")
    ax2.set_title("Residual: what the fit minimises")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"fig_moment_balance_{name}.png"),
                bbox_inches="tight")
    plt.close(fig)


def figure_scenario_rates(fig_dir, name, mw, r_j, is_edge):
    """The individual scenario rates r_j (report Equation 3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    pos = r_j > 0
    ax1.scatter(mw[pos & ~is_edge], r_j[pos & ~is_edge], s=9,
                color=PRIOR_GREY, alpha=0.5, label="interior ruptures")
    ax1.scatter(mw[pos & is_edge], r_j[pos & is_edge], s=11, color=PY_AMBER,
                alpha=0.7, label="edge-touching ruptures")
    ax1.set_yscale("log")
    ax1.set_xlabel("Moment magnitude Mw")
    ax1.set_ylabel("Scenario rate r_j (events / yr)")
    ax1.set_title(f"Individual scenario rates ({name})\n"
                  "Equation 3: conditional probability x magnitude-bin rate")
    ax1.legend(fontsize=8.5)

    # Summing r_j above each magnitude must reproduce the exceedance curve.
    uniq = np.unique(mw)
    binned = np.array([r_j[mw == m].sum() for m in uniq])
    ax2.semilogy(uniq, binned, "o-", color=PY_AMBER, lw=1.8, ms=4,
                 label="rate per magnitude bin")
    ax2.semilogy(uniq, np.cumsum(binned[::-1])[::-1], "s-", color=TARGET_BLUE,
                 lw=1.8, ms=4, label="exceedance rate (summed)")
    ax2.set_xlabel("Moment magnitude Mw")
    ax2.set_ylabel("Rate (events / yr)")
    ax2.set_title("Scenario rates aggregate to the source-zone curve")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"fig_scenario_rates_{name}.png"),
                bbox_inches="tight")
    plt.close(fig)


def figure_geometry(fig_dir, name, grid, stats):
    """Map view of this representation's unit-source grid."""
    lon, lat = grid[:, 0, :], grid[:, 1, :]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for j in range(lon.shape[0]):
        ax.plot(lon[j], lat[j], color="0.6", lw=0.8, zorder=1)
    for i in range(lon.shape[1]):
        ax.plot(lon[:, i], lat[:, i], color="0.6", lw=0.8, zorder=1)
    sc = ax.scatter(stats["lon_c"], stats["lat_c"], c=stats["depth"],
                    cmap="viridis", s=45, zorder=3, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.85, label="centroid depth (km)")
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(lat)))))
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title(f"Unit-source grid: {name}\n"
                 f"{stats['subfault_number'].size} unit sources, "
                 f"mean dip {np.mean(stats['dip']):.1f} deg")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"fig_geometry_{name}.png"),
                bbox_inches="tight")
    plt.close(fig)


def figure_segmentation(fig_dir, reps, seg_info):
    """LEVEL 0: the source representations and their (corrected) weights."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    colours = ["#1F4E79", "#B35C00", "#2F6B4F", "#7B3F8C", "#8C4A2F"]
    for k, rep in enumerate(reps):
        g = rep["grid"]
        c = colours[k % len(colours)]
        n_src = rep["stats"]["subfault_number"].size
        lbl = f"{rep['name']} ({n_src} sources)"
        is_whole = (k == 0)
        lw = 2.6 if is_whole else 1.8
        ls = "--" if is_whole else "-"
        if g is None:
            # Official statistics carry cell centres but no corner lattice, so
            # outline the zone by its centres instead of its edges.
            ax1.scatter(rep["stats"]["lon_c"], rep["stats"]["lat_c"], s=26,
                        color=c, alpha=0.75, label=lbl)
        else:
            lon, lat = g[:, 0, :], g[:, 1, :]
            ax1.plot(lon[0], lat[0], color=c, lw=lw, ls=ls)
            ax1.plot(lon[-1], lat[-1], color=c, lw=lw, ls=ls)
            ax1.plot(lon[:, 0], lat[:, 0], color=c, lw=lw, ls=ls)
            ax1.plot(lon[:, -1], lat[:, -1], color=c, lw=lw, ls=ls, label=lbl)
            if not is_whole:
                ax1.scatter(rep["stats"]["lon_c"], rep["stats"]["lat_c"], s=20,
                            color=c, alpha=0.6)
    ref_lat = (float(np.mean(reps[0]["stats"]["lat_c"]))
               if reps[0]["grid"] is None
               else float(np.mean(reps[0]["grid"][:, 1, :])))
    ax1.set_aspect(1.0 / np.cos(np.radians(ref_lat)))
    ax1.set_xlabel("Longitude (deg E)")
    ax1.set_ylabel("Latitude (deg N)")
    ax1.set_title("LEVEL 0: source representations")
    ax1.legend(fontsize=8.5)

    names = [seg_info["unsegmented_name"]] + list(seg_info["segments"])
    # Each segment carries the FULL union weight: the union is a sum, not a
    # choice. This is the v2 bug that is fixed here.
    wts = [seg_info["unsegmented_weight"]] + \
          [seg_info["per_segment_weight"]] * len(seg_info["segments"])
    y = np.arange(len(names))
    ax2.barh(y, wts, color=[colours[k % len(colours)] for k in range(len(names))])
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Logic-tree weight")
    ax2.set_xlim(0, 1)
    ax2.set_title("Weight per representation\n"
                  "(segments SUM within the union branch)")
    for yi, wv in zip(y, wts):
        ax2.text(wv + 0.02, yi, f"{wv:g}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_segmentation.png"), bbox_inches="tight")
    plt.close(fig)


def figure_percentiles(fig_dir, res, copula):
    """LEVEL 5: the epistemic percentile bands."""
    th = res["threshold_stages"]
    pe = res["percentile_exrate"]
    probs = res["percentile_probs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    n_pair = len(probs) // 2
    for k in range(n_pair):
        ax1.fill_between(th, pe[k, :], pe[-(k + 1), :], color=PY_AMBER,
                         alpha=0.16 + 0.12 * k,
                         label=f"p{probs[k]:g} - p{probs[-(k+1)]:g}")
    ax1.semilogy(th, res["mean_exrate"], color="k", lw=2.2, label="mean")
    if len(probs) % 2 == 1:
        ax1.semilogy(th, pe[n_pair, :], color=PY_AMBER, lw=1.6, ls="--",
                     label=f"median (p{probs[n_pair]:g})")
    ax1.set_xlabel("Mw threshold")
    ax1.set_ylabel("Exceedance rate (events / yr)")
    ax1.set_title(f"LEVEL 5: epistemic uncertainty\n({copula} copula across segments)")
    ax1.legend(fontsize=8.5)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(res["mean_exrate"] > 0,
                             (pe[-1, :] - pe[0, :]) / res["mean_exrate"], np.nan)
    ax2.plot(th, rel_width, "o-", color=PY_AMBER, lw=1.8, ms=5)
    ax2.set_xlabel("Mw threshold")
    ax2.set_ylabel(f"(p{probs[-1]:g} - p{probs[0]:g}) / mean")
    ax2.set_title("Relative width of the uncertainty band")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_percentiles.png"), bbox_inches="tight")
    plt.close(fig)


def figure_representation_curves(fig_dir, reps, mw_q, seg_info):
    """Mean curve per representation, plus the union-of-segments sum."""
    colours = ["#1F4E79", "#B35C00", "#2F6B4F", "#7B3F8C", "#8C4A2F"]
    fig, ax = plt.subplots(figsize=(9, 6))
    seg_total = np.zeros_like(mw_q)
    for k, rep in enumerate(reps):
        curve = np.array([float(rep["rate_fn"](m)) for m in mw_q])
        pos = curve > 0
        ax.semilogy(mw_q[pos], curve[pos], lw=2.0,
                    color=colours[k % len(colours)], label=rep["name"])
        if k > 0:
            seg_total = seg_total + curve
    if len(reps) > 1:
        pos = seg_total > 0
        ax.semilogy(mw_q[pos], seg_total[pos], lw=2.6, ls="--", color="k",
                    label="union of segments (sum)")
    ax.set_xlabel("Moment magnitude Mw")
    ax.set_ylabel("Exceedance rate (events / yr)")
    ax.set_title("Logic-tree mean curve per source representation\n"
                 "the union-of-segments branch is the SUM of its segments")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_representation_curves.png"),
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _column_rule_fp(column_rule, discretizer, cell_size="mean"):
    """Fingerprint field for the v8.2 column rule: none for rptha's own
    "trench" rule (or a discretiser that ignores the rule, or a
    --cell-size strasser mesh, whose columns the cap sets), so such a mesh
    keeps exactly the v8 fingerprint. Must match step 2's."""
    if discretizer != "optimal" or column_rule == "trench" or cell_size == "strasser":
        return {}
    return {"column_rule": column_rule}


def _cell_size_fp(geo):
    """v12 (2026-10-07): fingerprint fields of --cell-size strasser; none for
    rptha's "mean" rule, so every earlier mesh keeps its fingerprint. Must
    match step 2's."""
    if geo.get("cell_size", "mean") == "mean":
        return {}
    return {k: geo[k] for k in ("cell_size", "cell_k", "cell_cap_mw", "cell_cap_relation")}


def build_grid(geo):
    if geo["mode"] == "planar":
        return us.make_planar_unit_source_grid(
            lon0=geo["lon0"], lat0=geo["lat0"], strike=geo["strike"],
            dip=geo["dip"], n_alongstrike=geo["n_alongstrike"],
            n_downdip=geo["n_downdip"], subfault_length=geo["subfault_length"],
            subfault_width=geo["subfault_width"],
            top_depth=geo.get("top_depth", 0.0))

    if geo["mode"] == "shapefile":
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
        # "discretizer" is only present in input JSONs step6_write_input.py
        # (from_scratch_v4) writes; v2/v3 inputs have no such key, and
        # defaulting to "lm" here reproduces this function's ORIGINAL
        # behaviour exactly for them, and for any v4 input generated before
        # this field existed.
        discretizer = geo.get("discretizer", "lm")
        # v8.2: how many columns the "optimal" mesh gets, "trench" (rptha's
        # rule, default) or "average" (see contour_discretisation.
        # discretized_source_from_contours_optimal). Inputs without the key
        # (v8.1 and older) keep rptha's trench rule.
        column_rule = geo.get("column_rule", "trench")
        # v12: "mean" (rptha's rule, inputs without the key) or "strasser"
        # (no cell larger than the scaling relation's rupture at
        # cell_cap_mw / cell_k; see contour_discretisation.
        # discretized_source_from_contours_bounded).
        cell_size = geo.get("cell_size", "mean")

        # v8: reuse the mesh step 2 built and checked, if it was built from
        # the same contour file with the same parameters (see
        # pyptha_v12/grid_cache.py); otherwise build it here.
        cache_npy = geo.get("unit_source_grid_npy")
        if cache_npy and not os.path.isabs(cache_npy):
            cache_npy = os.path.join(ROOT, cache_npy)
        fp = grid_cache.fingerprint(
            shp, method=discretizer,
            desired_unit_source_length=kwargs["desired_unit_source_length"],
            desired_unit_source_width=kwargs["desired_unit_source_width"],
            n_downdip=kwargs["n_downdip"], min_downdip=kwargs["min_downdip"],
            seed=kwargs["seed"], **_column_rule_fp(column_rule, discretizer, cell_size),
            **_cell_size_fp(geo))
        if cache_npy:
            cached = grid_cache.load_if_match(cache_npy, fp)
            if cached is not None:
                print(f"      mesh: reusing step 2's grid "
                      f"({os.path.relpath(cache_npy, ROOT)}, fingerprint matches)")
                return cached
            print("      mesh: step 2's saved grid does not match this input's "
                  "meshing fields (or is missing) -- rebuilding")
        if cell_size == "strasser":
            if discretizer != "optimal":
                raise ValueError("cell_size 'strasser' needs discretizer 'optimal'")
            cap_l, cap_w = cd.strasser_cell_cap(geo["cell_cap_mw"], geo["cell_k"],
                                                geo["cell_cap_relation"])
            return cd.discretized_source_from_contours_bounded(
                contours, cap_l, cap_w, seed=kwargs["seed"], min_downdip=kwargs["min_downdip"])
        if cell_size != "mean":
            raise ValueError(f"unknown cell_size {cell_size!r} (mean or strasser)")
        if discretizer == "optimal":
            return cd.discretized_source_from_contours_optimal(
                contours, column_rule=column_rule, **kwargs)
        elif discretizer == "mid":
            return cd.discretized_source_from_contours_mid(
                contours, desired_unit_source_length=kwargs["desired_unit_source_length"],
                desired_unit_source_width=kwargs["desired_unit_source_width"],
                n_downdip=kwargs["n_downdip"], min_downdip=kwargs["min_downdip"])
        return cd.discretized_source_from_contours_orthogonal(contours, **kwargs)

    if geo["mode"] == "official_grid":
        return build_grid_from_official(geo)

    if geo["mode"] == "mesh_file":
        # v12: an external quadrilateral mesh, used as given (pyptha_v12/
        # mesh_file.py finds its rows, columns and orientation)
        from pyptha_v12 import mesh_file
        path = geo["mesh_file"]
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        return mesh_file.read_quadrilateral_mesh(
            path, depth_units=geo.get("mesh_depth_units", "auto"))

    if geo["mode"] == "official_statistics":
        raise ValueError("mode 'official_statistics' bypasses build_grid; "
                         "it is handled in build_source_representation")

    raise ValueError(f"unknown geometry mode: {geo['mode']!r}")


def load_official_statistics(geo):
    """Read PTHA18's own per-unit-source table instead of deriving one.

    ``unit_source_statistics_<zone>.nc`` on the NCI THREDDS server holds the
    length, width, dip, strike and depth of every unit source PTHA18 computed
    with. Taking it directly removes the last modelling step between v3 and the
    official result: the ``official_grid`` mode still had to infer corner depths
    from SLAB2.0, which left the mean dip 0.36 degrees shallow. Here nothing is
    inferred.

    The netCDF variable names already match the keys
    ``discretized_source_approximate_summary_statistics`` returns, so the table
    substitutes for it directly.
    """
    import netCDF4

    path = geo["official_statistics_nc"]
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)

    fields = ["lon_c", "lat_c", "depth", "strike", "dip", "rake", "slip",
              "length", "width", "downdip_number", "alongstrike_number",
              "subfault_number", "max_depth"]
    with netCDF4.Dataset(path) as ds:
        missing = [f for f in fields if f not in ds.variables]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        stats = {f: np.asarray(ds.variables[f][:], dtype=float) for f in fields}

    stats["area"] = stats["length"] * stats["width"]
    n_as = int(stats["alongstrike_number"].max())
    n_dd = int(stats["downdip_number"].max())
    print(f"      official PTHA18 unit-source statistics: {n_as} along-strike "
          f"x {n_dd} down-dip = {stats['dip'].size} unit sources, "
          f"depths {stats['depth'].min():.1f}-{stats['depth'].max():.1f} km")
    return stats


def build_grid_from_official(geo):
    """Use PTHA18's own unit-source grid instead of meshing contours.

    The published grid (SOURCE_ZONES/<zone>/EQ_SOURCE/unit_source_grid/<zone>.shp
    on the NCI THREDDS server) is the mesh PTHA18 actually computed with, after
    the manual editing of down-dip lines that its README describes and that no
    parameter records. Meshing it from contours therefore cannot reproduce it,
    which is the whole reason this mode exists.

    Those polygons are 2D, so the depth of each corner is read off the published
    SLAB2.0 depth model for the same region. Nothing here is tuned: the
    horizontal geometry is PTHA18's, the depths are SLAB2.0's.
    """
    import geopandas as gpd
    from scipy.interpolate import griddata

    shp = geo["official_grid_shapefile"]
    if not os.path.isabs(shp):
        shp = os.path.join(ROOT, shp)
    gdf = gpd.read_file(shp)

    dd_col = geo.get("downdip_column", "dwndp_n")
    as_col = geo.get("alongstrike_column", "alngst_")
    n_dd = int(gdf[dd_col].max())
    n_as = int(gdf[as_col].max())
    if len(gdf) != n_dd * n_as:
        raise ValueError(
            f"official grid has {len(gdf)} cells but {n_as} x {n_dd} indices; "
            f"the grid is not complete")

    # Ring order, established from the corners that neighbouring cells share:
    # c0=(dd i, as j), c1=(dd i, as j+1), c2=(dd i+1, as j+1), c3=(dd i+1, as j).
    lon = np.full((n_dd + 1, n_as + 1), np.nan)
    lat = np.full((n_dd + 1, n_as + 1), np.nan)
    for _, row in gdf.iterrows():
        i, j = int(row[dd_col]) - 1, int(row[as_col]) - 1
        c = np.asarray(row.geometry.exterior.coords)[:4]
        lon[i, j], lat[i, j] = c[0]
        lon[i, j + 1], lat[i, j + 1] = c[1]
        lon[i + 1, j + 1], lat[i + 1, j + 1] = c[2]
        lon[i + 1, j], lat[i + 1, j] = c[3]
    if np.isnan(lon).any():
        raise ValueError("official grid left holes in the corner lattice")

    xyz_path = geo["slab2_depth_xyz"]
    if not os.path.isabs(xyz_path):
        xyz_path = os.path.join(ROOT, xyz_path)
    xyz = pd.read_csv(xyz_path, header=None, names=["lon", "lat", "dep"]).dropna()
    xyz["lon"] = np.where(xyz["lon"] > 180.0, xyz["lon"] - 360.0, xyz["lon"])
    pts = np.column_stack([xyz["lon"].to_numpy(), xyz["lat"].to_numpy()])
    vals = xyz["dep"].to_numpy()

    # SLAB2.0 depths are negative-down; the grid wants positive-down km.
    depth = np.abs(griddata(pts, vals, (lon, lat), method="linear"))
    # Corners just outside the slab model's convex hull come back NaN; fall
    # back to nearest-neighbour there rather than dropping the cell.
    nan = np.isnan(depth)
    if nan.any():
        depth[nan] = np.abs(griddata(pts, vals, (lon[nan], lat[nan]),
                                     method="nearest"))
        print(f"      {int(nan.sum())} corner(s) outside the SLAB2.0 hull, "
              f"filled by nearest neighbour")

    grid = np.empty((n_dd + 1, 3, n_as + 1))
    grid[:, 0, :] = lon
    grid[:, 1, :] = lat
    grid[:, 2, :] = depth
    print(f"      official PTHA18 grid: {n_as} along-strike x {n_dd} down-dip "
          f"= {len(gdf)} unit sources, depths {depth.min():.1f}-"
          f"{depth.max():.1f} km from SLAB2.0")
    return grid


# ---------------------------------------------------------------------------
# One source representation (the unsegmented branch, or one segment)
# ---------------------------------------------------------------------------
def load_hs_events(path):
    """v11: the HS scenarios the variable shear modulus model is built from:
    (Mw, event_index_string, event_slip_string). Either step 7b's
    outputs/hs_slip_fields/summary.csv or PTHA18's own
    all_stochastic_slip_earthquake_events_<zone>.nc (validation)."""
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if path.endswith(".nc"):
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            mw = np.asarray(ds["Mw"][:], dtype=float)
            eis = [s.strip() for s in netCDF4.chartostring(ds["event_index_string"][:])]
            ess = [s.strip() for s in netCDF4.chartostring(ds["event_slip_string"][:])]
    else:
        df = pd.read_csv(path, usecols=["Mw", "event_index_string", "event_slip_string"])
        mw = df["Mw"].to_numpy(dtype=float)
        eis = df["event_index_string"].astype(str).tolist()
        ess = df["event_slip_string"].astype(str).tolist()
    return np.round(mw, 3), eis, ess


def variable_mu_error_cdf(vsm, stats, in_seg):
    """v11: PTHA18's distribution of (variable - constant shear modulus Mw)
    given the constant one, from the HS scenarios that touch this source
    representation (compute_rates_all_sources.R 476-516). Returns (cdf, the
    deviations, their constant Mw) or None if the zone keeps the constant
    rigidity (normal faults, as in PTHA18)."""
    rake = stats.get("rake")
    if rake is not None and np.all(np.asarray(rake, dtype=float) == -90):
        print("      variable shear modulus: normal-fault zone, constant "
              "rigidity only (as PTHA18)")
        return None
    mw, eis, ess = load_hs_events(vsm["hs_events"])
    area = np.asarray(stats["length"], dtype=float) * np.asarray(stats["width"], dtype=float)
    depth = np.asarray(stats["depth"], dtype=float)
    curve = vsm.get("curve", "default")
    if in_seg is not None and np.any(in_seg == 0):
        keep = [bool(np.any(in_seg[vmu.parse_index_string(s) - 1] > 0)) for s in eis]
        keep = np.asarray(keep)
        mw = mw[keep]
        eis = [s for s, k in zip(eis, keep) if k]
        ess = [s for s, k in zip(ess, keep) if k]
        if mw.size == 0:
            raise ValueError("no HS scenario touches this segment")
    dev = vmu.variable_mu_Mw(eis, ess, area, depth, kind=curve) - mw
    print(f"      variable shear modulus: {mw.size:,} HS scenarios, Mw "
          f"(variable - constant) {dev.min():+.3f} .. {dev.max():+.3f} "
          f"(curve {curve!r})")
    return vmu.make_conditional_ecdf(dev, mw), dev, mw


def build_source_representation(name, cfg, rep_cfg, out_dir, fig_dir=None,
                                inherited_edge_multiplier=None):
    """Run LEVELs 1-4 for a single source representation."""
    print(f"\n  --- source representation '{name}' ---")

    geo = dict(cfg["geometry"])
    geo.update(rep_cfg.get("geometry", {}))

    if geo["mode"] == "official_statistics":
        # PTHA18's own per-unit-source table, so there is no mesh to build and
        # no depth to infer. See load_official_statistics. There is no corner
        # lattice in that table, so grid stays None and the map view is skipped.
        grid = None
        stats = load_official_statistics(geo)
    else:
        grid = build_grid(geo)
        stats = us.discretized_source_approximate_summary_statistics(grid)

    # v9 (LEVEL 0): a segment is a stretch of the zone, not a source of its
    # own, exactly as in compute_rates_all_sources.R (198-211, 492-574,
    # 761-774; report Section 3.7.6): it keeps the WHOLE zone's unit sources
    # and scenarios, and is_in_segment marks the unit sources whose column is
    # in [i0, i1]. Its area, mean dip, Mw_max anchor and convergence come
    # from those unit sources only; a scenario's conditional probability is
    # weighted by the part of it inside the segment ("Scenarios which cross
    # rupture barriers are assigned an individual rate from all segments
    # that they touch, in proportion to the fraction ... contained in the
    # segment"), so every segment's scenario rates are on the zone's own
    # scenario table and add up scenario by scenario.
    slice_cfg = rep_cfg.get("alongstrike_slice")
    asn_full = np.asarray(stats["alongstrike_number"]).astype(int)
    if slice_cfg is not None:
        i0, i1 = int(slice_cfg[0]), int(slice_cfg[1])
        n_cols = int(asn_full.max())
        if not (1 <= i0 <= i1 <= n_cols):
            raise ValueError(f"alongstrike_slice {slice_cfg} outside 1..{n_cols}")
        in_seg = ((asn_full >= i0) & (asn_full <= i1)).astype(float)
        print(f"      segment: columns {i0}..{i1} of {n_cols}, on the zone's "
              f"full scenario table (rptha's is_in_segment)")
    else:
        in_seg = np.ones(asn_full.size)
    seg_stats = {k: v[in_seg > 0] for k, v in stats.items()}
    source_area = float(np.sum(stats["length"] * stats["width"] * in_seg))
    print(f"      n_unit_sources = {int(in_seg.sum())}   "
          f"area = {source_area:,.0f} km^2")

    # --- events ---
    ev = cfg["events"]
    # mu/relation are PER ZONE (config.R:15,19, reading sourcezone_
    # parameters.csv's scaling_relation/shear_modulus for source_rows =
    # which(sourcename==site_name)[1] -- the WHOLE zone's row, never a
    # per-segment one, even when make_all_earthquake_events.R runs on a
    # segment: R's site_name always comes from the working directory,
    # which is one level per SOURCE ZONE, not per segment). Read here via
    # cfg["source_zone"] (the base zone, e.g. "kermadectonga2") rather than
    # `name` (which carries a "_tonga"-style segment suffix when this call
    # is for a segment) for the same reason.
    # A from-scratch input carries both in its events block (step 6), so
    # nothing of PTHA18 is read; PTHA18's official inputs do not, and take
    # them from sourcezone_parameters.csv as R does.
    if "scaling_relation" in ev and "shear_modulus_Pa" in ev:
        scaling_relation, shear_modulus = ev["scaling_relation"], float(ev["shear_modulus_Pa"])
    else:
        print("      note: the input has no events.scaling_relation/shear_modulus_Pa, "
              "so they are read from PTHA18's sourcezone_parameters.csv (an "
              "official input's behaviour; a from-scratch input from step 6 "
              "always carries both)")
        scaling_relation, shear_modulus = ogp.scaling_relation_and_shear_modulus(
            cfg["source_zone"])
    # v10: "local" sizes each rupture from the cells where it sits (events.py,
    # _local_blocks); inputs without the key keep rptha's rule.
    rupture_size = ev.get("rupture_size", "rptha")
    all_eq = events.get_all_earthquake_events(
        stats, Mmin=ev["Mmin"], Mmax=ev["Mmax"], dMw=ev["dMw"],
        source_zone_name=name, mu=shear_modulus, relation=scaling_relation,
        rupture_size=rupture_size)
    dMw = float(ev["dMw"])

    # Round away finite-precision noise in the generated magnitudes, exactly as
    # compute_rates_all_sources.R does on the published event table (line 445,
    # "Round away any finite-precision issues in the netcdf file").
    #
    # This is NOT cosmetic. The moment balance decides whether a magnitude bin
    # contributes with the exact test lower_Mw <= Mw_max, where
    # lower_Mw = Mw - dMw/2, so a bin whose lower edge lands exactly on Mw_max
    # can fall on either side of that test depending on the last bit. Python
    # happened to land on the correct side here while R did not, but that is
    # luck rather than correctness: without the rounding the answer depends on
    # how the magnitude sequence was generated. See the matching note in
    # run_logic_tree.R.
    all_eq["Mw"] = np.round(all_eq["Mw"], 3)
    d = np.diff(all_eq["Mw"])
    assert np.all(all_eq["Mw"] == np.sort(all_eq["Mw"]))
    assert np.all((d == 0) | (np.abs(d - dMw) < 1.0e-12))

    print(f"      n_events = {all_eq['Mw'].size} (rupture size: {rupture_size})")

    event_table = {"Mw": all_eq["Mw"], "slip": all_eq["slip"],
                   "area": all_eq["area"]}

    rc = dict(cfg["rates"])
    rc.update(rep_cfg.get("rates", {}))

    # v9 (LEVEL 0): a segment's convergence is its OWN, the area-weighted
    # mean of the convergent slip of its unit sources
    # (compute_rates_all_sources.R 406-412, "If we have segmentation, then
    # this source-zone averaged slip value will be a localised value").
    # Inheriting the zone's scalar gave kermadectonga2's Tonga segment the
    # zone's ~99 mm/yr instead of its own ~168 (PTHA18: slip 0.176 m/yr on
    # _tonga, 0.103 on the zone). Zones without a Bird profile (constant
    # convergence, puysegur2) keep the zone's value, as R does (line 417). A
    # segment that sets its own tectonic_convergence_mm_per_yr keeps it. The
    # per-column profiles stay whole: the segment keeps the zone's columns.
    if (slice_cfg is not None and "convergent_slip_profile" in rc
            and "tectonic_convergence_mm_per_yr" not in rep_cfg.get("rates", {})):
        _cs = np.asarray(rc["convergent_slip_profile"], dtype=float)
        if _cs.size == int(asn_full.max()):
            _a = stats["length"] * stats["width"] * in_seg
            _conv = float(np.sum(_cs[asn_full - 1] * _a) / np.sum(_a))
            print(f"      segment convergence {_conv:.4f} mm/yr (area-weighted "
                  f"over its own unit sources; zone "
                  f"{float(cfg['rates']['tectonic_convergence_mm_per_yr']):.4f})")
            rc["tectonic_convergence_mm_per_yr"] = _conv

    # A segment hosts only the earthquakes that happened inside it, so it must
    # declare its own observed seismicity. Silently reusing the whole zone's
    # catalogue would tell every segment it hosted all of the zone's events,
    # inflating each segment's rate. Refuse to do that by accident.
    is_segment = rep_cfg.get("alongstrike_slice") is not None
    if (is_segment and rc.get("update_logic_tree_weights_with_data")
            and "observed_seismicity" not in rep_cfg.get("rates", {})):
        raise ValueError(
            f"segment {name!r} inherits the whole zone's observed_seismicity, "
            f"which would credit it with earthquakes that happened outside it. "
            f"Give this segment its own 'observed_seismicity' block, or set "
            f"'update_logic_tree_weights_with_data': false for it.")

    # integer column numbers (the official statistics table stores them as
    # floats; indexing with those raised IndexError, so the profile branch
    # below could never run before v8)
    asn = np.asarray(stats["alongstrike_number"]).astype(int)

    def column_profile(key):
        """A per-column rates entry (one value per along-strike column),
        broadcast to every unit source of its column."""
        prof = np.asarray(rc[key], dtype=float)
        if prof.size != int(asn.max()):
            raise ValueError(
                f"{key} has {prof.size} values but the source has "
                f"{int(asn.max())} along-strike columns (was the mesh "
                f"changed after step 3 computed the profile?)")
        return prof[asn - 1]

    cp_model = rc["conditional_probability_model"]
    if cp_model == "convergent_slip_weighted":
        # v8. PTHA18's model for every zone with use_bird_convergence == 1
        # (all but puysegur): compute_rates_all_sources.R:537-543 builds it
        # with make_conditional_probability_function_uniform_slip
        # (make_spatially_variable_source_zone_convergence_rates.R:196-275).
        # Within one magnitude bin an event's probability is proportional to
        #     event area * area-weighted mean convergent slip of its unit
        #     sources,
        # so ruptures on fast-converging parts of the zone are more likely.
        # 'inverse_slip' is R's model for constant-convergence zones only.
        # On a segment the convergence is zeroed outside it (R 236-255:
        # div_vec and rl_vec times local_is_in_segment), while the area it
        # is averaged over stays the whole event's.
        cs = column_profile("convergent_slip_profile") * in_seg
        us_area = stats["length"] * stats["width"]
        slip_near = np.empty(all_eq["Mw"].size)
        for i, s in enumerate(all_eq["event_index_string"]):
            ui = events.get_unit_source_indices_in_event(s)
            slip_near[i] = np.sum(us_area[ui] * cs[ui]) / np.sum(us_area[ui])
        weight = np.asarray(all_eq["area"], dtype=float) * slip_near
        print(f"      conditional probabilities: event area x convergent slip "
              f"(area-weighted mean of the profile "
              f"{np.sum(us_area * cs) / np.sum(us_area * in_seg):.4f} mm/yr)")
        cp_model = lambda idx: weight[idx] / weight[idx].sum()  # noqa: E731
    elif cp_model == "inverse_slip" and slice_cfg is not None:
        # R 549-572: 1/slip times the fraction of the event inside the segment
        frac = np.array([in_seg[events.get_unit_source_indices_in_event(s)].mean()
                         for s in all_eq["event_index_string"]])
        weight = frac / np.asarray(all_eq["slip"], dtype=float)
        cp_model = lambda idx: weight[idx] / weight[idx].sum()  # noqa: E731
    if rupture_size == "local":
        # v10_q: with one rupture per starting cell (rupture_size "local"),
        # a part of the zone with small cells is covered by more ruptures per
        # km2 and would get more rate for the same area and convergence.
        # Multiply whichever weight the model above gives by q, which
        # removes that overlap effect (events.coverage_weights). rptha's
        # rule never gets here, so its probabilities are PTHA18's.
        if cp_model == "all_equal":
            weight = np.ones(all_eq["Mw"].size)
        elif cp_model == "inverse_slip":
            weight = 1.0 / np.asarray(all_eq["slip"], dtype=float)
        elif not callable(cp_model):
            raise ValueError(f"conditional_probability_model {cp_model!r} "
                             f"not recognized")
        q = events.coverage_weights(all_eq["Mw"], all_eq["event_indices"],
                                    stats["length"].size)
        weight = weight * q
        print(f"      v10_q: conditional probabilities x q (rupture overlap "
              f"correction, q from {q.min():.3f} to {q.max():.3f})")
        cp_model = lambda idx: weight[idx] / weight[idx].sum()  # noqa: E731
    ecp_raw = rates.get_event_probabilities_conditional_on_Mw(
        event_table, cp_model)

    # --- LEVEL 1: parameter axes, including coupling and the dip correction ---
    # (the segment's own unit sources for the mean dip and the Mw_max anchor,
    # R 301-319 and 426; the whole zone when unsegmented)
    axes = build_parameter_axes(rc, seg_stats["dip"], stats=seg_stats,
                                source_area=source_area)
    print(f"      convergence = {axes['convergence_mm_per_yr']:g} mm/yr, "
          f"mean dip = {axes['mean_dip_deg']:.2f} deg, "
          f"1/cos(dip) = {axes['dip_factor']:.4f}")
    print(f"      coupling: {axes['coupling'].size} values in "
          f"[{axes['coupling'].min():.3f}, {axes['coupling'].max():.3f}], "
          f"prior mean = {np.sum(axes['coupling'] * axes['coupling_prob']):.3f}")
    print(f"      LEVEL 1: {axes['n_branches']:,} branches "
          f"({axes['slip_rate'].size} coupling x {axes['b'].size} b x "
          f"{axes['Mw_max'].size} Mw_max x {len(axes['Mfd'])} GR type)")

    # --- LEVEL 3 inputs: observed seismicity ---
    obs = rc.get("observed_seismicity")
    update_weights = bool(rc.get("update_logic_tree_weights_with_data", False))
    Mw_obs_data = None
    Mw_count_duration = (np.nan, np.nan, np.nan)
    if update_weights:
        if obs is None:
            raise ValueError("update_logic_tree_weights_with_data is true but "
                             "no 'observed_seismicity' block was given")
        # PTHA18 counts events above MW_MIN, where MW_MIN is itself the
        # half-bin-shifted value 7.2 - dMw/2 = 7.15 (EVENT_RATES/config.R) and
        # the GCMT threshold is that same number (gcmt_subsetter.R). So the
        # default threshold is Mw_min as supplied, NOT Mw_min - dMw/2: the
        # offset already lives in Mw_min. rptha rejects a threshold outside
        # [Mw_min, Mw_max], which is the check that pins this down.
        thresh = obs.get("threshold_Mw")
        if thresh is None:
            thresh = float(axes["Mw_min"].min())
        if thresh < float(axes["Mw_min"].min()):
            raise ValueError(
                f"observation threshold {thresh:g} is below Mw_min "
                f"{float(axes['Mw_min'].min()):g}. In PTHA18 the half-bin "
                f"offset belongs on Mw_min itself (MW_MIN = 7.2 - dMw/2), so "
                f"set Mw_min to the shifted value rather than lowering the "
                f"threshold below it.")
        obs_mws = obs.get("observed_Mw")
        count = int(obs["count"]) if "count" in obs else len(obs_mws or [])
        Mw_count_duration = (float(thresh), count, float(obs["duration_years"]))
        if obs_mws:
            Mw_obs_data = {"Mw": np.asarray(obs_mws, dtype=float)}
        scope = ("whole zone" if rep_cfg.get("alongstrike_slice") is None
                 else "this segment only")
        print(f"      LEVEL 3 data: {count} events above Mw {thresh:g} "
              f"in {obs['duration_years']:g} yr ({scope})"
              + (", magnitudes supplied" if obs_mws else ""))

    err_cdf = None
    err_hw = rc.get("mw_observation_error_halfwidth")
    if err_hw:
        def err_cdf(x, mw_true, _h=float(err_hw)):
            return np.clip((np.asarray(x, dtype=float) + _h) / (2 * _h), 0.0, 1.0)
    # v11: PTHA18's variable shear modulus (report Section 3.7.5). Its
    # magnitude difference is treated as an observation error of the
    # catalogue magnitudes in LEVEL 3; the constant-rigidity results are
    # unchanged.
    vsm = cfg.get("variable_shear_modulus")
    vsm_info = None
    if vsm:
        if err_hw:
            raise ValueError("variable_shear_modulus and "
                             "mw_observation_error_halfwidth cannot both be set")
        vsm_info = variable_mu_error_cdf(vsm, stats, in_seg)
        if vsm_info is not None:
            err_cdf = vsm_info[0]

    def make_rate_fn(conditional_probabilities):
        return rates.rate_of_earthquakes_greater_than_Mw_function(
            slip_rate=axes["slip_rate"], slip_rate_prob=axes["slip_rate_prob"],
            b=axes["b"], b_prob=axes["b_prob"],
            Mw_min=axes["Mw_min"], Mw_min_prob=axes["Mw_min_prob"],
            Mw_max=axes["Mw_max"], Mw_max_prob=axes["Mw_max_prob"],
            sourcezone_total_area=source_area,
            event_table=event_table,
            event_conditional_probabilities=conditional_probabilities,
            # rptha/pyptha's own default is 0.01, but PTHA18's official run
            # (run_logic_tree.R:1176-1179) fixes 0.02 for the Mw grid used to
            # integrate seismic moment (and, via integration_dy =
            # computational_increment/2, the LEVEL 3 Mw-observation-error
            # smearing too). v4 never passed this, so it silently used 0.01 --
            # fixed here to match the official value, while still honouring
            # an explicit override in the input JSON, exactly like R does.
            computational_increment=rc.get("computational_increment", 0.02),
            Mw_frequency_distribution=axes["Mfd"],
            Mw_frequency_distribution_prob=axes["Mfd_prob"],
            update_logic_tree_weights_with_data=update_weights,
            Mw_count_duration=Mw_count_duration,
            account_for_moment_below_mwmin=rc.get(
                "account_for_moment_below_mwmin", False),
            mw_max_posterior_equals_mw_max_prior=rc.get(
                "mw_max_posterior_equals_mw_max_prior", False),
            Mw_obs_data=Mw_obs_data,
            mw_observation_error_cdf=err_cdf)

    # First pass with the uncorrected conditional probabilities. The edge fit
    # needs scenario rates, and scenario rates need a rate function, so PTHA18
    # runs the rate model once, fits the multiplier, then rebuilds.
    rate_fn = make_rate_fn(ecp_raw)

    # --- LEVEL 4: fit the edge multiplier against the convergence pattern ---
    edge_cfg = rc.get("edge_correction", {})
    edge_mode = edge_cfg.get("mode", "fit")

    r_j_raw = rates.individual_scenario_rates(
        all_eq["Mw"], ecp_raw, rate_fn, dMw)

    # The target convergence shape -- compared directly (unweighted) against
    # `model` (back_calculate_convergence's integrated_slip, itself a per-
    # unit-source SLIP RATE, not an area or a moment) in fun_to_optimize's
    # objective, sum(is_in_segment*(model/sum(model) - div_vec/sum(div_vec))
    # **2) (compute_rates_all_sources.R:680). div_vec is NEVER weighted by
    # unit-source area anywhere in R -- confirmed by reading both of its
    # branches directly:
    #   - constant convergence (compute_rates_all_sources.R:420, used here
    #     when the JSON has no convergence_profile): div_vec = rep(sourcepar$
    #     slip, n) * is_in_segment -- a single scalar repeated across every
    #     unit source, i.e. FLAT, not proportional to area.
    #   - Bird convergence (compute_rates_all_sources.R:388/391): div_vec =
    #     pmax(0, +-bird_vel_div) * is_in_segment -- a raw per-unit-source
    #     velocity, again with no area weighting. v8's step 3 computes it
    #     per column (lib/bird_convergence.py) and step 6 writes it here as
    #     convergence_profile.
    # An earlier version of this file multiplied by stats["length"]*stats
    # ["width"] in both branches, which is wrong: it changes the fitted
    # edge_multiplier (and hence every scenario's conditional probability)
    # whenever unit-source area varies along strike, which it does on any
    # real mesh.
    target_profile = rc.get("convergence_profile")
    if target_profile is None:
        # R's scalar here is sourcepar$slip = tectonic_slip*convergent_
        # fraction (compute_rates_all_sources.R:417-418) -- the RAW
        # convergence rate, before the dip correction (line 432) and before
        # coupling (line 613's slip_rate = sourcepar$slip*sourcepar$
        # coupling). This project's equivalent raw rate is the same
        # tectonic_convergence_mm_per_yr axes["slip_rate"] is itself built
        # from (see build_parameter_axes, line ~133/152) -- rc["tectonic_
        # convergence_mm_per_yr"], not axes["slip_rate"] (which already has
        # both dip and coupling folded in and is an array, one value per
        # coupling branch, not the single scalar R's fit uses here).
        target = np.full(stats["subfault_number"].size,
                         float(rc["tectonic_convergence_mm_per_yr"]))
    else:
        # One value per along-strike column, broadcast down dip -- still no
        # area weighting, matching R's per-unit-source div_vec exactly.
        target = column_profile("convergence_profile")
        if target.max() <= 0:
            # compute_rates_all_sources.R:393
            raise ValueError("convergence_profile has no positive value: no "
                             "tectonic moment on this source")
    # div_vec * is_in_segment (R 388-391, 420)
    target = target * in_seg
    if slice_cfg is not None and edge_mode == "fit":
        # R never fits the multiplier on a segment (lines 716-735): it takes
        # the unsegmented zone's, which is the engine's default for segments.
        raise ValueError(f"segment {name!r}: edge_correction mode 'fit' is not "
                         f"PTHA18's method on a segment; use 'inherit'")

    env_raw = mb.back_calculate_convergence(
        r_j_raw, all_eq["Mw"], all_eq["event_index_string"], all_eq["slip"],
        stats["alongstrike_number"],
        n_unit_sources=stats["subfault_number"].size, edge_multiplier=0.0)

    if edge_mode == "off":
        edge_multiplier = 0.0
        ecp = ecp_raw
        integrated_slip = env_raw["integrated_slip"]
        is_edge = env_raw["is_on_edge"]
        print("      LEVEL 4 edge correction: disabled")
    else:
        if edge_mode == "inherit":
            if inherited_edge_multiplier is None:
                raise ValueError("edge mode 'inherit' needs an unsegmented fit")
            edge_multiplier = float(inherited_edge_multiplier)
            env = mb.back_calculate_convergence(
                r_j_raw, all_eq["Mw"], all_eq["event_index_string"],
                all_eq["slip"], stats["alongstrike_number"],
                n_unit_sources=stats["subfault_number"].size,
                edge_multiplier=edge_multiplier)
            ecp = env["new_conditional_probability"]
            integrated_slip = env["integrated_slip"]
            is_edge = env["is_on_edge"]
            print(f"      LEVEL 4 edge correction: inherited "
                  f"{edge_multiplier:.4f} from the unsegmented zone")
        elif edge_mode == "fixed":
            edge_multiplier = float(edge_cfg["value"])
            env = mb.back_calculate_convergence(
                r_j_raw, all_eq["Mw"], all_eq["event_index_string"],
                all_eq["slip"], stats["alongstrike_number"],
                n_unit_sources=stats["subfault_number"].size,
                edge_multiplier=edge_multiplier)
            ecp = env["new_conditional_probability"]
            integrated_slip = env["integrated_slip"]
            is_edge = env["is_on_edge"]
            print(f"      LEVEL 4 edge correction: fixed at {edge_multiplier:g}")
        elif edge_mode == "fit":
            fit = mb.fit_edge_multiplier(
                r_j_raw, all_eq["Mw"], all_eq["event_index_string"],
                all_eq["slip"], stats["alongstrike_number"], target,
                n_unit_sources=stats["subfault_number"].size,
                lower=float(edge_cfg.get("lower", 0.0)),
                upper=float(edge_cfg.get("upper", 30.0)))
            edge_multiplier = fit["edge_multiplier"]
            ecp = fit["conditional_probability"]
            integrated_slip = fit["integrated_slip"]
            is_edge = fit["is_on_edge"]
            improvement = (1 - fit["objective"] / fit["objective_at_zero"]) * 100
            print(f"      LEVEL 4 edge correction: FITTED "
                  f"multiplier = {edge_multiplier:.4f}, "
                  f"misfit {fit['objective_at_zero']:.4g} -> "
                  f"{fit['objective']:.4g} ({improvement:.1f}% better)")
            if fit["hit_bound"]:
                print("        WARNING: the fit hit a search bound")
        else:
            raise ValueError(f"unknown edge correction mode {edge_mode!r}")

        # Rebuild the rate model with the corrected conditional probabilities:
        # they enter the moment balance, so 'a' changes.
        rate_fn = make_rate_fn(ecp)

    br = rate_fn(return_all_logic_tree_branches=True)
    n_branch = len(br.all_par_combo)
    print(f"      LEVEL 2: a in [{br.a_parameter.min():.4f}, "
          f"{br.a_parameter.max():.4f}]")
    if update_weights:
        tvd = 0.5 * np.abs(br.all_par_prob - br.all_par_prob_prior).sum()
        prior_c = np.array([p["slip_rate"] for p in br.all_par_combo])
        scale = axes["convergence_mm_per_yr"] / 1000.0 * axes["dip_factor"]
        mean_c_prior = np.sum(prior_c / scale * br.all_par_prob_prior)
        mean_c_post = np.sum(prior_c / scale * br.all_par_prob)
        print(f"      LEVEL 3: posterior applied, TVD = {tvd:.4g}, "
              f"mean coupling {mean_c_prior:.3f} -> {mean_c_post:.3f}")
    else:
        print("      LEVEL 3: disabled (posterior = prior)")

    # --- final scenario rates, with the corrected conditional probabilities ---
    pct = cfg.get("percentiles", {}).get("percentile_probs",
                                         [0.025, 0.16, 0.5, 0.84, 0.975])
    r_j = rates.individual_scenario_rates(all_eq["Mw"], ecp, rate_fn, dMw)
    r_j_pct = rates.individual_scenario_rates(all_eq["Mw"], ecp, rate_fn, dMw,
                                              quantiles=pct)

    # v11: the variable shear modulus rates: the same branch curves weighted
    # with the LEVEL 3 weights that treat the catalogue's magnitudes as
    # variable-rigidity ones (compute_rates_all_sources.R 749-869)
    rate_fn_mu = r_j_mu = None
    if vsm_info is not None:
        rate_fn_mu = rate_fn.with_mw_error()
        r_j_mu = rates.individual_scenario_rates(all_eq["Mw"], ecp, rate_fn_mu, dMw)
        r_j_mu_pct = rates.individual_scenario_rates(all_eq["Mw"], ecp, rate_fn_mu,
                                                     dMw, quantiles=pct)
        mw_mu = vmu.variable_mu_Mw(
            all_eq["event_index_string"], all_eq["slip"],
            np.asarray(stats["length"], dtype=float) * np.asarray(stats["width"], dtype=float),
            np.asarray(stats["depth"], dtype=float), kind=vsm.get("curve", "default"))
        path = os.path.join(out_dir, f"scenario_rates_{name}_variable_mu.csv")
        with open(path, "w") as f:
            f.write("event_id,Mw,variable_mu_Mw,rate_mean,"
                    + ",".join(f"rate_p{p}" for p in pct) + "\n")
            for i in range(all_eq["Mw"].size):
                f.write("%d,%.17g,%.17g,%.17g,%s\n" % (
                    i + 1, all_eq["Mw"][i], mw_mu[i], r_j_mu[i],
                    ",".join("%.17g" % r_j_mu_pct[k, i] for k in range(len(pct)))))
        print(f"      v11: variable shear modulus scenario rates -> "
              f"{os.path.basename(path)}")
        dev, dev_mw = vsm_info[1], vsm_info[2]
        save_csv(os.path.join(out_dir, f"variable_mu_deviation_{name}.csv"),
                 ["Mw_constant_mu", "Mw_variable_minus_constant"], [dev_mw, dev])

    # --- CSVs -------------------------------------------------------------
    save_csv(os.path.join(out_dir, f"conditional_prob_{name}.csv"),
             ["Mw", "ecp_raw", "ecp_edge_corrected", "is_edge_touching"],
             [all_eq["Mw"], ecp_raw, ecp, is_edge.astype(float)])

    # Individual scenario rates: the end product (report Equation 3).
    path = os.path.join(out_dir, f"scenario_rates_{name}.csv")
    with open(path, "w") as f:
        f.write("event_id,Mw,area_km2,slip_m,mean_depth_km,is_edge_touching,"
                "conditional_probability,rate_mean,"
                + ",".join(f"rate_p{p}" for p in pct) + "\n")
        for i in range(all_eq["Mw"].size):
            f.write("%d,%.17g,%.17g,%.17g,%.17g,%d,%.17g,%.17g,%s\n" % (
                i + 1, all_eq["Mw"][i], all_eq["area"][i], all_eq["slip"][i],
                all_eq["mean_depth"][i], int(is_edge[i]), ecp[i], r_j[i],
                ",".join("%.17g" % r_j_pct[k, i] for k in range(len(pct)))))
    print(f"      wrote {all_eq['Mw'].size:,} scenario rates -> "
          f"{os.path.basename(path)}")

    # Integrated slip: the moment-conservation check (report Figures 38/39/45).
    save_csv(os.path.join(out_dir, f"integrated_slip_{name}.csv"),
             ["unit_source", "alongstrike_number", "downdip_number",
              "lon_c", "lat_c", "depth_km", "area_km2",
              "target_convergence_shape", "integrated_slip_no_edge_correction",
              "integrated_slip"],
             [stats["subfault_number"], stats["alongstrike_number"],
              stats["downdip_number"], stats["lon_c"], stats["lat_c"],
              stats["depth"], stats["length"] * stats["width"],
              target / target.sum(), env_raw["integrated_slip"],
              integrated_slip])

    # Branch table.
    path = os.path.join(out_dir, f"logic_tree_branches_{name}.csv")
    scale = axes["convergence_mm_per_yr"] / 1000.0 * axes["dip_factor"]
    with open(path, "w") as f:
        f.write("branch_id,coupling,slip_rate,b,Mw_min,Mw_max,"
                "Mw_frequency_distribution,a,prior_prob,posterior_prob,"
                "posterior_prob_with_Mw_error\n")
        pwe = br.all_par_prob_with_Mw_error
        if pwe is None:
            pwe = br.all_par_prob
        for i, par in enumerate(br.all_par_combo):
            f.write("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%.17g,%.17g,%.17g,%.17g\n"
                    % (i + 1, par["slip_rate"] / scale, par["slip_rate"],
                       par["b"], par["Mw_min"], par["Mw_max"],
                       par["Mw_frequency_distribution"], br.a_parameter[i],
                       br.all_par_prob_prior[i], br.all_par_prob[i], pwe[i]))
    print(f"      wrote {n_branch:,} branches -> {os.path.basename(path)}")

    # Every branch's exceedance curve. With 32000 branches this file is large,
    # so it is written only when asked for.
    if cfg.get("outputs", {}).get("write_branch_rate_curves", True):
        path = os.path.join(out_dir, f"branch_rate_curves_{name}.csv")
        with open(path, "w") as f:
            f.write("Mw," + ",".join(f"branch_{i+1}" for i in range(n_branch))
                    + "\n")
            for j, mw in enumerate(br.Mw_seq):
                f.write("%.17g," % mw)
                f.write(",".join("%.17g" % v for v in br.all_rate_matrix[:, j]))
                f.write("\n")
        print(f"      wrote branch curves -> {os.path.basename(path)}")

    # --- figures ---
    if fig_dir is not None:
        obs_mw_curve = obs_rate_curve = None
        if Mw_obs_data is not None:
            duration = Mw_count_duration[2]
            obs_mws_arr = Mw_obs_data["Mw"]
            # Stop at the largest observed magnitude: beyond it the empirical
            # rate is exactly zero, which is a hole in the log scale, not a
            # point to plot. The report's Figures 43/44 truncate here too.
            in_range = br.Mw_seq <= obs_mws_arr.max()
            obs_mw_curve = br.Mw_seq[in_range]
            obs_rate_curve = np.array(
                [(obs_mws_arr >= m).sum() / duration for m in obs_mw_curve])

        # The official statistics table gives cell centres and sizes but not
        # the corner lattice this map view draws, so it is skipped there.
        if grid is not None:
            figure_geometry(fig_dir, name, grid, stats)
        figure_branch_fan(fig_dir, name, br, rate_fn=rate_fn,
                          observed_mw=obs_mw_curve, observed_rate=obs_rate_curve)
        figure_branch_weights(fig_dir, name, br, axes)
        figure_scenario_rates(fig_dir, name, all_eq["Mw"], r_j, is_edge)
        if edge_mode != "off":
            figure_moment_balance(fig_dir, name, stats,
                                  env_raw["integrated_slip"], integrated_slip,
                                  target, edge_multiplier)
        print(f"      figures written for {name}")

    return {"name": name, "rate_fn": rate_fn, "all_eq": all_eq, "ecp": ecp,
            "branches": br, "source_area": source_area, "grid": grid,
            "stats": stats, "axes": axes, "edge_multiplier": edge_multiplier,
            "integrated_slip": integrated_slip, "scenario_rates": r_j,
            "rate_fn_mu": rate_fn_mu, "scenario_rates_mu": r_j_mu}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inp_path = (sys.argv[1] if len(sys.argv) > 1
                else "inputs/input_puysegur2.json")
    if not os.path.isabs(inp_path):
        # Inputs are shared by both engines and live at the package root, so a
        # relative path is resolved from there, not from this folder.
        inp_path = os.path.join(ROOT, inp_path)
    with open(inp_path) as f:
        cfg = json.load(f)

    run_name = cfg["run_name"]
    out_dir = os.path.join(ROOT, "runs", "python", run_name)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    # remove what an earlier run of the same run_name wrote: files named after
    # sources that no longer exist (segments of an older segmentation) would
    # otherwise be copied into the example's outputs/ with the new ones
    import glob
    for _old in glob.glob(os.path.join(out_dir, "*.csv")) + glob.glob(os.path.join(fig_dir, "*.png")):
        os.remove(_old)
    print(f"pyptha v3 logic-tree run '{run_name}'  ->  {out_dir}")

    # --- LEVEL 0: segmentation ---
    source_zone = cfg["source_zone"]
    segment_cfgs = cfg.get("segments", {})
    all_names = [source_zone] + [f"{source_zone}_{s}" for s in segment_cfgs]
    seg_info = lt.get_unsegmented_and_segmented_source_names_on_source_zone(
        source_zone, all_names)
    print(f"\n  LEVEL 0: unsegmented '{seg_info['unsegmented_name']}' "
          f"weight {seg_info['unsegmented_weight']}, "
          f"{len(seg_info['segments'])} segment(s), union weight "
          f"{seg_info['union_of_segments_weight']} "
          f"(each segment carries {seg_info['per_segment_weight']}, "
          f"they sum within the union branch)")

    with open(os.path.join(out_dir, "segmentation.csv"), "w") as f:
        f.write("source_representation,role,weight\n")
        f.write("%s,unsegmented,%.17g\n"
                % (seg_info["unsegmented_name"], seg_info["unsegmented_weight"]))
        for s in seg_info["segments"]:
            # Full union weight per segment: the union is a SUM.
            f.write("%s,segment,%.17g\n" % (s, seg_info["per_segment_weight"]))

    # --- LEVELs 1-4 per representation ---
    unsegmented = build_source_representation(
        source_zone, cfg, cfg.get("unsegmented", {}), out_dir, fig_dir)

    segments = []
    for seg_short, seg_cfg in segment_cfgs.items():
        # PTHA18 gives segments the unsegmented zone's edge multiplier, because
        # edge events have too little leverage on a segment to fit it stably.
        seg_cfg = dict(seg_cfg)
        seg_rates = dict(seg_cfg.get("rates", {}))
        if "edge_correction" not in seg_rates:
            seg_rates["edge_correction"] = {"mode": "inherit"}
        seg_cfg["rates"] = seg_rates
        segments.append(build_source_representation(
            f"{source_zone}_{seg_short}", cfg, seg_cfg, out_dir, fig_dir,
            inherited_edge_multiplier=unsegmented["edge_multiplier"]))

    # --- mean rate curves ---
    ev = cfg["events"]
    mw_q = np.round(np.arange(ev["Mmin"], ev["Mmax"] + ev["dMw"] / 10, 0.01), 6)
    mean_unseg = np.array([float(unsegmented["rate_fn"](m)) for m in mw_q])
    cols = [mw_q, mean_unseg]
    header = ["Mw", "exceedance_rate_unsegmented"]
    if segments:
        seg_sum = np.zeros_like(mw_q)
        for s in segments:
            curve = np.array([float(s["rate_fn"](m)) for m in mw_q])
            cols.append(curve)
            header.append(f"exceedance_rate_{s['name']}")
            seg_sum += curve
        cols.append(seg_sum)
        header.append("exceedance_rate_union_of_segments")
        # The zone curve: weighted mix of the two representations.
        cols.append(seg_info["unsegmented_weight"] * mean_unseg
                    + seg_info["union_of_segments_weight"] * seg_sum)
        header.append("exceedance_rate_source_zone")
    save_csv(os.path.join(out_dir, "rate_curves.csv"), header, cols)

    # v11: the same curves with the variable shear modulus weights
    has_mu = unsegmented["rate_fn_mu"] is not None
    if has_mu:
        mean_unseg_mu = np.array([float(unsegmented["rate_fn_mu"](m)) for m in mw_q])
        cols_mu = [mw_q, mean_unseg_mu]
        if segments:
            seg_sum_mu = np.zeros_like(mw_q)
            for s in segments:
                curve = np.array([float(s["rate_fn_mu"](m)) for m in mw_q])
                cols_mu.append(curve)
                seg_sum_mu += curve
            cols_mu.append(seg_sum_mu)
            cols_mu.append(seg_info["unsegmented_weight"] * mean_unseg_mu
                           + seg_info["union_of_segments_weight"] * seg_sum_mu)
        save_csv(os.path.join(out_dir, "rate_curves_variable_mu.csv"), header, cols_mu)

    # --- v9 (LEVEL 0): the zone's own scenario rates ---
    # What PTHA18 publishes for a segmented zone (compute_rates_all_sources.R
    # 1883-1907): every scenario's rate is row_weight x its unsegmented rate
    # plus row_weight x its rate on each segment it touches, all on the zone's
    # one scenario table. The percentiles are not weighted sums of the
    # sources' percentiles; they come from R's partial-segmentation step
    # (lines 998-1409, lt.scenario_rate_percentiles_partial_segmentation).
    if segments:
        eq = unsegmented["all_eq"]
        for s in segments:
            if not np.array_equal(s["all_eq"]["Mw"], eq["Mw"]):
                raise ValueError(f"{s['name']} is not on the zone's scenario table")
        w_u = seg_info["unsegmented_weight"]
        w_s = seg_info["per_segment_weight"]
        zone_mean = w_u * unsegmented["scenario_rates"] + w_s * np.sum(
            [s["scenario_rates"] for s in segments], axis=0)
        pct = cfg.get("percentiles", {}).get("percentile_probs",
                                             [0.025, 0.16, 0.5, 0.84, 0.975])
        per_source = lt.scenario_rate_percentiles_partial_segmentation(
            unsegmented["rate_fn"], [s["rate_fn"] for s in segments],
            eq["Mw"], [unsegmented["ecp"]] + [s["ecp"] for s in segments],
            float(ev["dMw"]), w_u, seg_info["union_of_segments_weight"], pct)
        zone_pct = w_u * per_source[0] + w_s * np.sum(per_source[1:], axis=0)
        path = os.path.join(out_dir, f"scenario_rates_{source_zone}_source_zone.csv")
        with open(path, "w") as f:
            f.write("event_id,Mw,area_km2,slip_m,mean_depth_km,rate_mean,"
                    + ",".join(f"rate_p{p}" for p in pct) + "\n")
            for i in range(eq["Mw"].size):
                f.write("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s\n" % (
                    i + 1, eq["Mw"][i], eq["area"][i], eq["slip"][i],
                    eq["mean_depth"][i], zone_mean[i],
                    ",".join("%.17g" % zone_pct[k, i] for k in range(len(pct)))))
        print(f"\n  LEVEL 0: the zone's {eq['Mw'].size:,} scenario rates "
              f"(unsegmented + segments) -> {os.path.basename(path)}")
        if has_mu:
            zone_mean_mu = w_u * unsegmented["scenario_rates_mu"] + w_s * np.sum(
                [s["scenario_rates_mu"] for s in segments], axis=0)
            per_source_mu = lt.scenario_rate_percentiles_partial_segmentation(
                unsegmented["rate_fn_mu"], [s["rate_fn_mu"] for s in segments],
                eq["Mw"], [unsegmented["ecp"]] + [s["ecp"] for s in segments],
                float(ev["dMw"]), w_u, seg_info["union_of_segments_weight"], pct)
            zone_pct_mu = w_u * per_source_mu[0] + w_s * np.sum(per_source_mu[1:], axis=0)
            path = os.path.join(out_dir,
                                f"scenario_rates_{source_zone}_source_zone_variable_mu.csv")
            with open(path, "w") as f:
                f.write("event_id,Mw,rate_mean,"
                        + ",".join(f"rate_p{p}" for p in pct) + "\n")
                for i in range(eq["Mw"].size):
                    f.write("%d,%.17g,%.17g,%s\n" % (
                        i + 1, eq["Mw"][i], zone_mean_mu[i],
                        ",".join("%.17g" % zone_pct_mu[k, i] for k in range(len(pct)))))
            print(f"  v11: the zone's variable shear modulus scenario rates -> "
                  f"{os.path.basename(path)}")

    # --- LEVEL 5: epistemic percentiles ---
    lt_cfg = cfg.get("percentiles", {})
    thresholds = np.asarray(
        lt_cfg.get("threshold_Mw",
                   list(np.round(np.arange(ev["Mmin"], ev["Mmax"] + 1e-9, 0.1), 6))),
        dtype=float)

    unseg_mat = lt.branch_exceedance_rates_on_thresholds(
        unsegmented["rate_fn"], thresholds)
    seg_mats = [lt.branch_exceedance_rates_on_thresholds(s["rate_fn"], thresholds)
                for s in segments]

    copula = lt_cfg.get("copula", "comonotonic")
    N = int(lt_cfg.get("N", 40000))
    pct_probs = lt_cfg.get("percentile_probs", [0.025, 0.16, 0.5, 0.84, 0.975])
    seed = int(lt_cfg.get("seed", 123))

    print(f"\n  LEVEL 5: {copula} copula, N = {N:,}, "
          f"{len(seg_mats)} segment(s), {thresholds.size} Mw thresholds")

    res = lt.compute_exceedance_rate_percentiles_with_random_sampling(
        unseg_mat, seg_mats, N=N,
        unsegmented_wt=seg_info["unsegmented_weight"],
        union_of_segments_wt=seg_info["union_of_segments_weight"],
        segments_copula_type=copula,
        percentile_probs=pct_probs,
        rng=np.random.default_rng(seed))

    header = ["threshold_Mw", "mean_exrate"] + \
             [f"p{p}" for p in res["percentile_probs"]]
    cols = [res["threshold_stages"], res["mean_exrate"]] + \
           [res["percentile_exrate"][k, :] for k in range(res["percentile_probs"].size)]
    save_csv(os.path.join(out_dir, "exceedance_rate_percentiles.csv"), header, cols)

    # v11: LEVEL 5 with the variable shear modulus weights (same seed)
    if has_mu:
        res_mu = lt.compute_exceedance_rate_percentiles_with_random_sampling(
            lt.branch_exceedance_rates_on_thresholds(unsegmented["rate_fn_mu"], thresholds),
            [lt.branch_exceedance_rates_on_thresholds(s["rate_fn_mu"], thresholds)
             for s in segments], N=N,
            unsegmented_wt=seg_info["unsegmented_weight"],
            union_of_segments_wt=seg_info["union_of_segments_weight"],
            segments_copula_type=copula,
            percentile_probs=pct_probs,
            rng=np.random.default_rng(seed))
        cols_mu = [res_mu["threshold_stages"], res_mu["mean_exrate"]] + \
                  [res_mu["percentile_exrate"][k, :] for k in range(res_mu["percentile_probs"].size)]
        save_csv(os.path.join(out_dir, "exceedance_rate_percentiles_variable_mu.csv"),
                 header, cols_mu)

    # A segmented run also writes the unsegmented branch's own band, with the
    # exact call an unsegmented run makes (weight 1, no segments, same seed),
    # so step 9 can show the unsegmented report without a second run.
    if segments:
        res_u = lt.compute_exceedance_rate_percentiles_with_random_sampling(
            unseg_mat, [], N=N,
            unsegmented_wt=1.0, union_of_segments_wt=0.0,
            segments_copula_type=copula,
            percentile_probs=pct_probs,
            rng=np.random.default_rng(seed))
        cols = [res_u["threshold_stages"], res_u["mean_exrate"]] + \
               [res_u["percentile_exrate"][k, :]
                for k in range(res_u["percentile_probs"].size)]
        save_csv(os.path.join(out_dir, "exceedance_rate_percentiles_unsegmented.csv"),
                 header, cols)

    # --- summary ---
    with open(os.path.join(out_dir, "logic_tree_summary.csv"), "w") as f:
        f.write("level,item,value\n")
        f.write("0,unsegmented_weight,%.17g\n" % seg_info["unsegmented_weight"])
        f.write("0,union_of_segments_weight,%.17g\n"
                % seg_info["union_of_segments_weight"])
        f.write("0,per_segment_weight,%.17g\n" % seg_info["per_segment_weight"])
        f.write("0,n_segments,%d\n" % len(seg_info["segments"]))
        for rep in [unsegmented] + segments:
            b = rep["branches"]
            ax = rep["axes"]
            nm = rep["name"]
            f.write("1,n_branches_%s,%d\n" % (nm, len(b.all_par_combo)))
            f.write("1,mean_dip_deg_%s,%.17g\n" % (nm, ax["mean_dip_deg"]))
            f.write("1,dip_factor_%s,%.17g\n" % (nm, ax["dip_factor"]))
            f.write("1,convergence_mm_per_yr_%s,%.17g\n"
                    % (nm, ax["convergence_mm_per_yr"]))
            f.write("1,prior_mean_coupling_%s,%.17g\n"
                    % (nm, float(np.sum(ax["coupling"] * ax["coupling_prob"]))))
            f.write("2,a_min_%s,%.17g\n" % (nm, b.a_parameter.min()))
            f.write("2,a_max_%s,%.17g\n" % (nm, b.a_parameter.max()))
            tvd = 0.5 * np.abs(b.all_par_prob - b.all_par_prob_prior).sum()
            f.write("3,prior_posterior_tvd_%s,%.17g\n" % (nm, tvd))
            if rep.get("rate_fn_mu") is not None:
                # v11: how far the variable shear modulus weights are from
                # the prior and from the constant-rigidity posterior
                wmu = b.all_par_prob_with_Mw_error
                f.write("3,prior_posterior_tvd_variable_mu_%s,%.17g\n"
                        % (nm, 0.5 * np.abs(wmu - b.all_par_prob_prior).sum()))
                f.write("3,posterior_vs_variable_mu_tvd_%s,%.17g\n"
                        % (nm, 0.5 * np.abs(wmu - b.all_par_prob).sum()))
                f.write("R,total_scenario_rate_variable_mu_%s,%.17g\n"
                        % (nm, float(rep["scenario_rates_mu"].sum())))
            f.write("4,edge_multiplier_%s,%.17g\n" % (nm, rep["edge_multiplier"]))
            f.write("4,total_integrated_slip_%s,%.17g\n"
                    % (nm, float(rep["integrated_slip"].sum())))
            f.write("R,total_scenario_rate_%s,%.17g\n"
                    % (nm, float(rep["scenario_rates"].sum())))
        f.write("5,copula,%s\n" % copula)
        f.write("5,N_samples,%d\n" % N)

    # --- run-level figures ---
    all_reps = [unsegmented] + segments
    figure_segmentation(fig_dir, all_reps, seg_info)
    figure_representation_curves(fig_dir, all_reps, mw_q, seg_info)
    figure_percentiles(fig_dir, res, copula)

    n_fig = len([f for f in os.listdir(fig_dir) if f.endswith(".png")])
    print(f"\n  {n_fig} figures written to {fig_dir}")
    print(f"  all outputs written to {out_dir}")


if __name__ == "__main__":
    main()
