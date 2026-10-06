"""Posterior-mean Gutenberg-Richter curve + tectonic convergence, per zone.

Consolidates, for each of the 4 verified source zones, the two quantities
Davies & Griffin's (2018) rate model ties together in the seismic-moment
balance:

  - the logic-tree POSTERIOR MEAN Mw-exceedance-rate curve
    (rate_curves.csv, column 'exceedance_rate_unsegmented', which
    run_logic_tree.py computes as all_par_prob @ all_rate_matrix)
  - the tectonic convergence rate that, after the dip correction and the
    coupling logic tree, produced the slip rates behind that curve
    (logic_tree_summary.csv, item 'convergence_mm_per_yr_<zone>')

IMPORTANT: on all 4 verified zones LEVEL 3 (the Bayesian update of branch
weights against observed seismicity) is DISABLED in the shipped inputs
(update_logic_tree_weights_with_data: false). So here posterior == prior by
construction (prior_posterior_tvd = 0 in every logic_tree_summary.csv) -- this
is not a limitation of this script, it reproduces PTHA18's own published
configuration for these zones. The column is still labelled "posterior mean"
because that is the quantity the rate-function API returns; it happens to
equal the prior mean here. See docs/GR_curves_and_convergence.html.

This script does not recompute anything: it only reads the CSVs that
run_logic_tree.py already wrote under runs/python/<zone>_official/. Run that
first for any zone whose run folder is missing.

Usage:
    python python_logic_tree/gr_curve_and_convergence.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ZONES = [
    ("puysegur2", "puysegur2_official", "Puysegur"),
    ("kermadectonga2", "kermadectonga2_official", "Kermadec-Tonga"),
    ("kurilsjapan", "kurilsjapan_official", "Kurils-Japan"),
    ("southamerica", "southamerica_official", "South America"),
]

OUT_DIR = os.path.join(ROOT, "docs", "figures", "gr_convergence")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_BLUE = "#1F4E79"
PY_AMBER = "#B35C00"
plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25,
})


def load_summary(run_dir):
    """logic_tree_summary.csv -> {item: value}, numeric where possible.

    Most rows are numeric (weights, rates, dip...); a few carry text (e.g.
    'copula' = 'comonotonic'), so values are converted to float only when
    that succeeds and left as strings otherwise.
    """
    path = os.path.join(run_dir, "logic_tree_summary.csv")
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        v = row["value"]
        try:
            v = float(v)
        except ValueError:
            pass
        out[row["item"]] = v
    return out


def main():
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (short, run_name, label) in zip(axes, ZONES):
        run_dir = os.path.join(ROOT, "runs", "python", run_name)
        curves_path = os.path.join(run_dir, "rate_curves.csv")
        if not os.path.isdir(run_dir) or not os.path.isfile(curves_path):
            raise FileNotFoundError(
                f"missing run output for {label}: {curves_path}. "
                f"Run: python python_logic_tree/run_logic_tree.py "
                f"inputs/input_{short}.json")

        summary = load_summary(run_dir)
        curves = pd.read_csv(curves_path)
        mw = curves["Mw"].to_numpy()
        gr_mean = curves["exceedance_rate_unsegmented"].to_numpy()

        conv_mm_yr = summary[f"convergence_mm_per_yr_{short}"]
        dip_deg = summary[f"mean_dip_deg_{short}"]
        dip_factor = summary[f"dip_factor_{short}"]
        coupling = summary[f"prior_mean_coupling_{short}"]
        tvd = summary[f"prior_posterior_tvd_{short}"]
        slip_rate_mm_yr = conv_mm_yr * coupling * dip_factor

        rows.append({
            "zone": label,
            "convergence_mm_per_yr": conv_mm_yr,
            "mean_dip_deg": dip_deg,
            "dip_factor_1_over_cos_dip": dip_factor,
            "prior_mean_coupling": coupling,
            "mean_seismic_slip_rate_mm_per_yr": slip_rate_mm_yr,
            "posterior_equals_prior": (tvd == 0.0),
            "rate_Mw7.5_per_yr": float(np.interp(7.5, mw, gr_mean)),
            "rate_Mw8.0_per_yr": float(np.interp(8.0, mw, gr_mean)),
            "rate_Mw9.0_per_yr": float(np.interp(9.0, mw, gr_mean)),
        })

        # --- per-zone figure panel ---
        pos = gr_mean > 0
        ax.semilogy(mw[pos], gr_mean[pos], color=TARGET_BLUE, lw=2.2,
                    label="posterior mean GR curve")
        ax.set_title(f"{label}\nconvergence = {conv_mm_yr:.1f} mm/yr, "
                     f"dip = {dip_deg:.1f}°, coupling = {coupling:.2f}")
        ax.set_xlabel("Moment magnitude Mw")
        ax.set_ylabel("Exceedance rate (events/yr)")
        ax.axhline(1.0, color="0.85", lw=0.8, zorder=0)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.annotate(
            f"seismic slip rate\n= convergence × coupling / cos(dip)\n"
            f"= {slip_rate_mm_yr:.1f} mm/yr",
            xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8,
            color="0.35", va="bottom")

    fig.suptitle(
        "Posterior-mean Gutenberg-Richter curve and tectonic convergence, "
        "per source zone\n"
        "(LEVEL 3 disabled on all 4 zones: posterior mean = prior mean)",
        fontweight="bold", y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "gr_curve_and_convergence.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")

    out = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, "runs", "python", "gr_curve_and_convergence_summary.csv")
    out.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
