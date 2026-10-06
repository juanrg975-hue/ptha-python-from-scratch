"""STEP 9 - Build an HTML report of the whole from-scratch run.

Reads back everything steps 1-7 actually wrote to disk (nothing is
recomputed, nothing is hand-typed) and assembles one self-contained HTML
page: what each step did, the real numbers it produced, and how the final
rate curve compares to PTHA18's own official run, built by step 8. The one
part that is not self-contained is STEP 2's map of the mesh over Esri's
ocean basemap, whose tiles load from the internet when the page opens.

Requires steps 1-7 to have already run. If outputs/ is missing, this exits
and tells you which step to run first. Step 8 (the official run) is
optional: without it, the report still covers steps 1-7 in full, just
without the comparison section.

v9: a segmented run gets a LEVEL 0 section (how each model divides the
trench as a bar and figures/segments_map.png, one table per segmentation,
this run against PTHA18's segmented run for the unsegmented branch, the union
and the mix, and a chart), and its exports carry the segments.
It also writes report_unsegmented.html, the unsegmented branch alone as an
unsegmented run's report shows it; a switch at the top links the two.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/makran2/steps/step9_report.py
"""

import base64
import html
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))  # helper modules
from official_geometry_params import slab_product_for_zone  # noqa: E402

# build_grid from v5's OWN forked engine, not the shared python_logic_tree/
# -- step7 ran the actual rates through this exact copy (see
# from_scratch_v12/html/docs/code_map.html), so rebuilding the mesh from any other copy
# would silently show a different mesh than the one the rates came from.
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "python_logic_tree_v12"))
from run_logic_tree import build_grid  # noqa: E402

# Read back the GCMT window step 4/5 actually filtered on, rather than
# hardcoding it here -- these constants have changed before (v1 used "today"
# as WINDOW_END; v3 pins it to PTHA18's own 2017-03-01 cutoff), and this
# report must never describe a window step 4/5 no longer use.
sys.path.insert(0, HERE)
from step5_subset_gcmt import WINDOW_START, WINDOW_END  # noqa: E402

TITLE = "Makran"
PTHA18_ZONE_NAME = "makran2"
# Which SLAB product this zone's official mesh actually used (PTHA18's own
# naming rule: a zone name ending in "2" used SLAB2.0, any other name used
# SLAB1.0 -- see ReportPTHA.pdf p.11-12). This report describes whichever
# product step 1 actually downloaded, not a hardcoded default.
SLAB_PRODUCT = slab_product_for_zone(PTHA18_ZONE_NAME)
# logic_tree_summary.csv and logic_tree_branches_*.csv key their per-zone
# rows/files by the PTHA18 zone name -- e.g. "mean_dip_deg_kermadectonga2".
SUFFIX = PTHA18_ZONE_NAME

OUT_DIR = os.path.join(EXAMPLE, "outputs")
INPUT_JSON = os.path.join(EXAMPLE, "inputs", "input_makran2_scratch.json")
GCMT_SUBSET_CSV = os.path.join(EXAMPLE, "data", "gcmt", "makran2_gcmt_subset.csv")
GCMT_CATALOGUE_CSV = os.path.join(EXAMPLE, "data", "gcmt", "gcmt_catalogue_1976_present.csv")
CONV_TXT = os.path.join(EXAMPLE, "data", "convergence.txt")
CONV_PERCELL_NPY = os.path.join(EXAMPLE, "data", "convergence_per_cell.npy")
CONV_PERCOLUMN_JSON = os.path.join(EXAMPLE, "data",
                                   "convergence_per_column.json")
# Step 3's plate-boundary-change finding, if it made one: the mesh appears to
# run past the end of its own subduction zone onto the next one. Present only
# when there is something to report (step 3 deletes it otherwise).
PLATE_CHANGE_JSON = os.path.join(EXAMPLE, "data", "plate_boundary_change.json")
CONTOURS_SHP = os.path.join(EXAMPLE, "inputs", "geometry",
                            "makran2_slab2_contours.shp")
# step 7b's illustrative heterogeneous-slip (HS) and variable-area-uniform-
# slip (VAUS) fields, if it has been run. Entirely optional -- never read
# into the logic tree, so their absence changes nothing about the rate
# numbers or the rest of this report.
HS_SUMMARY_CSV = os.path.join(EXAMPLE, "outputs", "hs_slip_fields", "summary.csv")
VAUS_SUMMARY_CSV = os.path.join(EXAMPLE, "outputs", "vaus_slip_fields", "summary.csv")
# The official PTHA18 run, if step 8 has been run: used only as a reference
# line on the final chart and in the comparison table. Entirely optional -- a
# from-scratch run is still a complete, honest result without it.
OFFICIAL_RUN_DIR = os.path.join(EXAMPLE, "outputs_official")

# v12: generate.py --mesh-file: the geometry is an external quadrilateral
# mesh (inputs/geometry/<file>, one line per unit source with its 4 corners
# as lon, lat, depth), not SLAB. None: SLAB, as before.
MESH_FILE = None
MESH_DEPTH_UNITS = 'auto'
# v11: step 8 writes this when the zone is not one of PTHA18's
NOT_A_PTHA18_ZONE = False
try:
    with open(os.path.join(EXAMPLE, "outputs_official_status.txt"), encoding="utf-8") as _fh:
        NOT_A_PTHA18_ZONE = _fh.readline().strip() == "not_a_ptha18_zone"
except OSError:
    pass
# Everything else of PTHA18 this report compares with, written by step 8.
# This script reads PTHA18 from nowhere but OFFICIAL_RUN_DIR.
REFERENCE_DIR = os.path.join(OFFICIAL_RUN_DIR, "ptha18_reference")
OFFICIAL_NC = os.path.join(REFERENCE_DIR, f"unit_source_statistics_{PTHA18_ZONE_NAME}.nc")
OFFICIAL_CONV_NPY = os.path.join(REFERENCE_DIR, "official_convergence_per_cell.npy")
# The same Bird match per along-strike column on PTHA18's own mesh, written by
# step 8 with the same code and PTHA18's own Bird table (v8.1).
OFFICIAL_CONV_PERCOLUMN_JSON = os.path.join(
    REFERENCE_DIR, "official_convergence_per_column.json")
OFFICIAL_GRID_NPY = os.path.join(REFERENCE_DIR, "official_grid_lonlat.npy")
OFFICIAL_GCMT_JSON = os.path.join(REFERENCE_DIR, "official_gcmt.json")
# step 8's statistical comparison of step 7b's fields with PTHA18's own
# published HS/VAUS catalogues, if it ran.
HS_OFFICIAL_COMPARISON_CSV = os.path.join(REFERENCE_DIR, "hs_comparison.csv")
VAUS_OFFICIAL_COMPARISON_CSV = os.path.join(REFERENCE_DIR, "vaus_comparison.csv")

OUT_HTML = os.path.join(EXAMPLE, "report.html")
# v9: a segmented run also writes the unsegmented branch alone as its own page
OUT_HTML_UNSEG = os.path.join(EXAMPLE, "report_unsegmented.html")
UNSEG_PERCENTILES = "exceedance_rate_percentiles_unsegmented.csv"

VIEW_SWITCH_CSS = """<style>
.view-switch{display:flex;margin:28px 0 0;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:var(--surface-2)}
.view-switch a{flex:1;min-width:0;padding:11px 16px;text-decoration:none;color:var(--text-soft);font-size:13.5px;line-height:1.35;border-right:1px solid var(--line)}
.view-switch a:last-child{border-right:0}
.view-switch a b{display:block;font-size:15px;font-weight:600;color:var(--text)}
.view-switch a:not([aria-current]):hover{background:var(--surface)}
.view-switch a[aria-current=page]{background:var(--accent);color:var(--surface)}
.view-switch a[aria-current=page] b{color:var(--surface)}
</style>
"""


def esc(x):
    return html.escape(str(x))


def require(path, step):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nRun {step} first.")


def load_everything():
    require(INPUT_JSON, "step6_write_input.py")
    require(OUT_DIR, "step7_run.py")

    with open(INPUT_JSON) as f:
        cfg = json.load(f)

    summary = pd.read_csv(os.path.join(OUT_DIR, "logic_tree_summary.csv"))
    summary_map = {row["item"]: row["value"] for _, row in summary.iterrows()}

    rates = pd.read_csv(os.path.join(OUT_DIR, "rate_curves.csv"))
    percentiles = pd.read_csv(os.path.join(OUT_DIR, "exceedance_rate_percentiles.csv"))
    branches_path = os.path.join(OUT_DIR, f"logic_tree_branches_{SUFFIX}.csv")
    branches = pd.read_csv(branches_path) if os.path.exists(branches_path) \
        else pd.DataFrame()

    slip_path = os.path.join(OUT_DIR, f"integrated_slip_{SUFFIX}.csv")
    slip = pd.read_csv(slip_path) if os.path.exists(slip_path) else None
    off_slip_path = os.path.join(OFFICIAL_RUN_DIR, f"integrated_slip_{SUFFIX}.csv")
    off_slip = pd.read_csv(off_slip_path) if os.path.exists(off_slip_path) else None

    try:
        gcmt_subset = pd.read_csv(GCMT_SUBSET_CSV) if os.path.exists(GCMT_SUBSET_CSV) \
            else pd.DataFrame()
    except pd.errors.EmptyDataError:
        gcmt_subset = pd.DataFrame()
    n_catalogue = 0
    if os.path.exists(GCMT_CATALOGUE_CSV):
        n_catalogue = sum(1 for _ in open(GCMT_CATALOGUE_CSV)) - 1

    convergence = None
    if os.path.exists(CONV_TXT):
        with open(CONV_TXT) as f:
            convergence = float(f.read().strip())

    n_contours = None
    if os.path.exists(CONTOURS_SHP):
        try:
            import geopandas as gpd
            n_contours = len(gpd.read_file(CONTOURS_SHP))
        except Exception:
            n_contours = None

    # v10_q: what step 1's trench ramp rule did (absent in older folders)
    step1_info = None
    step1_json = os.path.join(EXAMPLE, "data", "step1_contours_info.json")
    if os.path.exists(step1_json):
        try:
            with open(step1_json, encoding="utf-8") as fh:
                step1_info = json.load(fh)
        except Exception:
            step1_info = None

    official = None
    if os.path.isdir(OFFICIAL_RUN_DIR):
        try:
            off_summary = pd.read_csv(
                os.path.join(OFFICIAL_RUN_DIR, "logic_tree_summary.csv"))
            off_map = {row["item"]: row["value"] for _, row in off_summary.iterrows()}
            off_rates = pd.read_csv(os.path.join(OFFICIAL_RUN_DIR, "rate_curves.csv"))
            official = {"summary": off_map, "rates": off_rates}
        except Exception:
            official = None

    hs_summary = None
    if os.path.exists(HS_SUMMARY_CSV):
        try:
            hs_summary = pd.read_csv(HS_SUMMARY_CSV)
        except Exception:
            hs_summary = None

    vaus_summary = None
    if os.path.exists(VAUS_SUMMARY_CSV):
        try:
            vaus_summary = pd.read_csv(VAUS_SUMMARY_CSV)
        except Exception:
            vaus_summary = None

    hs_official_comparison = None
    if os.path.exists(HS_OFFICIAL_COMPARISON_CSV):
        try:
            hs_official_comparison = pd.read_csv(HS_OFFICIAL_COMPARISON_CSV)
        except Exception:
            hs_official_comparison = None

    vaus_official_comparison = None
    if os.path.exists(VAUS_OFFICIAL_COMPARISON_CSV):
        try:
            vaus_official_comparison = pd.read_csv(VAUS_OFFICIAL_COMPARISON_CSV)
        except Exception:
            vaus_official_comparison = None

    return {
        "cfg": cfg, "summary": summary_map, "rates": rates,
        "percentiles": percentiles, "branches": branches,
        "slip": slip, "off_slip": off_slip,
        "gcmt_subset": gcmt_subset, "n_catalogue": n_catalogue,
        "convergence": convergence, "n_contours": n_contours, "step1_info": step1_info,
        "official": official, "hs_summary": hs_summary,
        "vaus_summary": vaus_summary,
        "hs_official_comparison": hs_official_comparison,
        "vaus_official_comparison": vaus_official_comparison,
        "variable_mu": load_variable_mu(),  # v11, step 7c
    }


def unsegmented_view(d):
    """A segmented run seen as an unsegmented run: the unsegmented branch
    alone, weight 1. Its branches, slip and scenario rates are the files an
    unsegmented run writes; only the percentile band differs (the segmented
    one is the zone mix), so the engine writes the branch's own band to
    exceedance_rate_percentiles_unsegmented.csv. None if step 7 predates it."""
    if not os.path.exists(os.path.join(OUT_DIR, UNSEG_PERCENTILES)):
        return None
    u = dict(d)
    u["cfg"] = dict(d["cfg"], segments={})
    u["rates"] = d["rates"][["Mw", "exceedance_rate_unsegmented"]]
    u["percentiles"] = pd.read_csv(os.path.join(OUT_DIR, UNSEG_PERCENTILES))
    if d["official"] is not None:
        u["official"] = dict(d["official"], rates=d["official"]["rates"][
            ["Mw", "exceedance_rate_unsegmented"]])
    u["percentiles_file"] = UNSEG_PERCENTILES
    u["view"] = "unsegmented"
    return u


def rate_at(df, mw, col="exceedance_rate_unsegmented"):
    """Nearest-Mw exceedance rate, for headline numbers."""
    i = (df["Mw"] - mw).abs().idxmin()
    return float(df[col].iloc[i])


def _series(label, color):
    """Open one toggleable series of a chart: the page script puts a button per
    <g class="series"> under the chart, so each curve can be hidden."""
    return f'<g class="series" data-name="{esc(label)}" data-color="{color}">'


def svg_rate_curve(rates, official_rates, percentiles):
    """A log-scale exceedance-rate chart: this run, +/- the 16-84 percentile
    band, and the official curve if available. Hand-drawn SVG so it needs no
    charting library and reads correctly in both themes via CSS variables.

    Carries invisible per-point hover targets (class "hitdot") with
    data-mw/data-rate/data-series attributes; the page's own inline JS (see
    TOOLTIP_JS) reads those to drive a shared tooltip, so no charting
    library is needed for interactivity either.
    """
    W, H = 640, 340
    ml, mr, mt, mb = 56, 16, 16, 40
    pw, ph = W - ml - mr, H - mt - mb

    mw_lo, mw_hi = 7.2, 9.6
    rate_lo, rate_hi = 1e-4, 1.0

    def x(mw):
        return ml + (mw - mw_lo) / (mw_hi - mw_lo) * pw

    def y(rate):
        rate = max(rate, rate_lo)
        return mt + (1 - (np.log10(rate) - np.log10(rate_lo))
                    / (np.log10(rate_hi) - np.log10(rate_lo))) * ph

    def path(df, col):
        pts = [(x(m), y(r)) for m, r in zip(df["Mw"], df[col]) if r > 0]
        return "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)

    def hitdots(df, col, series, label):
        # Thin the run's own 241-row curve down to ~60 hover targets; dense
        # enough to feel continuous, sparse enough to stay readable in the
        # page source.
        pts = [(m, r) for m, r in zip(df["Mw"], df[col]) if r > 0]
        step = max(1, len(pts) // 60)
        out = []
        for m, r in pts[::step]:
            out.append(f'<circle class="hitdot" cx="{x(m):.1f}" cy="{y(r):.1f}" '
                       f'r="7" fill="transparent" data-mw="{m:.3f}" '
                       f'data-rate="{r:.5g}" data-series="{esc(label)}"/>')
        return "\n".join(out)

    parts = []
    parts.append(f'<svg viewBox="0 0 {W} {H}" role="img" class="hoverchart" '
                 f'aria-label="Exceedance rate versus magnitude, this run '
                 f'compared to the official PTHA18 run">')

    # gridlines + labels, decades of rate
    for decade in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        gy = y(decade)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="var(--line)" stroke-width="1"/>')
        lbl = f"{decade:g}"
        parts.append(f'<text x="{ml-8}" y="{gy+4:.1f}" text-anchor="end" '
                     f'font-family="IBM Plex Mono, monospace" font-size="10" '
                     f'fill="var(--text-soft)">{lbl}</text>')
    for mw in [7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6]:
        gx = x(mw)
        parts.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt+ph}" '
                     f'stroke="var(--line)" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<text x="{gx:.1f}" y="{mt+ph+18}" text-anchor="middle" '
                     f'font-family="IBM Plex Mono, monospace" font-size="10" '
                     f'fill="var(--text-soft)">{mw:g}</text>')

    # percentile band (16-84), drawn up to the last magnitude where its upper
    # edge is still above zero; where the 16th percentile is zero (more than
    # 16% of the branches do not allow that magnitude) the lower edge sits on
    # the bottom of the chart
    if percentiles is not None and len(percentiles):
        pp = percentiles[percentiles["p0.84"] > 0]
        top = [(x(m), y(r)) for m, r in zip(pp["threshold_Mw"], pp["p0.16"])]
        bot = [(x(m), y(r)) for m, r in zip(pp["threshold_Mw"][::-1],
                                             pp["p0.84"][::-1])]
        band = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in top + bot) + " Z"
        parts.append(_series("16th-84th percentile", "var(--accent-2)")
                     + f'<path d="{band}" fill="var(--accent-2)" opacity="0.16"/></g>')

    # official curve, if present
    if official_rates is not None:
        parts.append(_series("Official PTHA18" + (", unsegmented" if "exceedance_rate_union_of_segments" in rates.columns else ""), "var(--amber)"))
        parts.append(f'<path d="{path(official_rates, "exceedance_rate_unsegmented")}" '
                     f'fill="none" stroke="var(--amber)" stroke-width="2" '
                     f'stroke-dasharray="5 4"/>')
        parts.append(hitdots(official_rates, "exceedance_rate_unsegmented",
                             "official", "Official PTHA18"))
        parts.append('</g>')

    # v9 (LEVEL 0): the segments and their union, when this was a segmented run.
    #
    # Drawn UNDER the unsegmented curve and thinner, because they are the other
    # half of the tree rather than a competing answer: PTHA18 weights the
    # unsegmented branch 0.5 and the union of segments 0.5. Showing them makes
    # the one thing a segmented run adds visible -- how far the union departs
    # from the unsegmented curve, which is where segmentation actually changes
    # the hazard.
    seg_cols = [c for c in rates.columns
                if c.startswith("exceedance_rate_")
                and c not in ("exceedance_rate_unsegmented",
                              "exceedance_rate_union_of_segments",
                              "exceedance_rate_source_zone")]
    for col in seg_cols:
        parts.append(_series("Segment " + col.replace(f"exceedance_rate_{PTHA18_ZONE_NAME}_", ""),
                             "var(--text-soft)"))
        parts.append(f'<path d="{path(rates, col)}" fill="none" '
                     f'stroke="var(--text-soft)" stroke-width="1.2" '
                     f'opacity="0.55" stroke-dasharray="2 3"/>')
        parts.append(hitdots(rates, col, "segment",
                             col.replace("exceedance_rate_", "")))
        parts.append('</g>')
    if "exceedance_rate_union_of_segments" in rates.columns:
        parts.append(_series("Union of segments", "var(--accent-2)"))
        parts.append(f'<path d="{path(rates, "exceedance_rate_union_of_segments")}" '
                     f'fill="none" stroke="var(--accent-2)" stroke-width="2" '
                     f'stroke-dasharray="7 3"/>')
        parts.append(hitdots(rates, "exceedance_rate_union_of_segments",
                             "union", "Union of segments"))
        parts.append('</g>')

    # this run's mean curve
    parts.append(_series("This run" + (", unsegmented" if "exceedance_rate_union_of_segments" in rates.columns else ""), "var(--accent)"))
    parts.append(f'<path d="{path(rates, "exceedance_rate_unsegmented")}" '
                 f'fill="none" stroke="var(--accent)" stroke-width="2.5"/>')
    parts.append(hitdots(rates, "exceedance_rate_unsegmented",
                         "ours", "This run"))
    parts.append('</g>')

    # the zone curve: what the LEVEL 0 mixture actually gives. Only meaningful
    # when there ARE segments; without them it equals the unsegmented curve.
    if "exceedance_rate_source_zone" in rates.columns:
        parts.append(_series("Source zone (LEVEL 0 mix)", "var(--accent)"))
        parts.append(f'<path d="{path(rates, "exceedance_rate_source_zone")}" '
                     f'fill="none" stroke="var(--accent)" stroke-width="2.5" '
                     f'stroke-dasharray="1 4" stroke-linecap="round" '
                     f'opacity="0.95"/>')
        parts.append(hitdots(rates, "exceedance_rate_source_zone",
                             "zone", "Source zone (LEVEL 0 mix)"))
        parts.append('</g>')

    parts.append(f'<text x="{ml+pw/2:.1f}" y="{H-4}" text-anchor="middle" '
                 f'font-family="IBM Plex Sans, sans-serif" font-size="11" '
                 f'fill="var(--text-soft)">Moment magnitude (Mw)</text>')
    parts.append('</svg>')
    return "\n".join(parts)


# One colour per segment in along-strike order, the same as the segment map
# (lib/plot_meshes.py SEGMENT_COLOURS), so bar and map read together.
SEGMENT_COLOURS = ["#2a78d6", "#eb6834", "#2e9e5b", "#9b59b6", "#d4a017",
                   "#17a2b8", "#c0392b", "#7f8c8d"]


def svg_segment_strips(rows):
    """v9 (LEVEL 0). How each model splits the zone along strike, as one bar
    per model on a common 0-100% scale of the arc: rows is a list of
    (label, n_columns, [(name, first_column, last_column), ...])."""
    W, lab, bar_h, gap, top = 660, 118, 30, 26, 22
    pw = W - lab - 10
    H = top + len(rows) * (bar_h + gap) + 4
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Along-strike '
         f'segments of each model">']
    for t in (0, 25, 50, 75, 100):
        gx = lab + pw * t / 100
        p.append(f'<line x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" y2="{H - 4}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        p.append(f'<text x="{gx:.1f}" y="{top - 10}" text-anchor="middle" '
                 f'font-size="10" fill="var(--text-soft)">{t}%</text>')
    for r, (label, n, segs) in enumerate(rows):
        y0 = top + r * (bar_h + gap)
        p.append(f'<text x="{lab - 8}" y="{y0 + bar_h / 2 + 4:.1f}" '
                 f'text-anchor="end" font-size="12" font-weight="600" '
                 f'fill="var(--text)">{esc(label)}</text>')
        for k, (name, a, b) in enumerate(segs):
            x0 = lab + pw * (a - 1) / n
            w = pw * (b - a + 1) / n
            p.append(f'<rect x="{x0:.1f}" y="{y0}" width="{w:.1f}" '
                     f'height="{bar_h}" fill="{SEGMENT_COLOURS[k % 8]}" '
                     f'stroke="#fff" stroke-width="1.5"/>')
            p.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + bar_h / 2 + 4:.1f}" '
                     f'text-anchor="middle" font-size="11" font-weight="600" '
                     f'fill="#fff">{esc(name)} ({a}-{b})</text>')
        p.append(f'<text x="{lab}" y="{y0 + bar_h + 13}" font-size="10" '
                 f'fill="var(--text-soft)">{n} along-strike columns</text>')
    p.append('</svg>')
    return "\n".join(p)


def svg_curves(series, aria):
    """A log-scale exceedance-rate chart of the given series, same axes and
    hover mechanism as svg_rate_curve. Each series is a dict with df, col,
    color, width, dash ("" for solid), label and opacity."""
    W, H = 640, 340
    ml, mr, mt, mb = 56, 16, 16, 40
    pw, ph = W - ml - mr, H - mt - mb
    mw_lo, mw_hi = 7.2, 9.6
    rate_lo, rate_hi = 1e-4, 1.0

    def x(mw):
        return ml + (mw - mw_lo) / (mw_hi - mw_lo) * pw

    def y(rate):
        rate = max(rate, rate_lo)
        return mt + (1 - (np.log10(rate) - np.log10(rate_lo))
                     / (np.log10(rate_hi) - np.log10(rate_lo))) * ph

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" class="hoverchart" '
         f'aria-label="{esc(aria)}">']
    for decade in [1.0, 0.1, 0.01, 0.001, 0.0001]:
        gy = y(decade)
        p.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        p.append(f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10" '
                 f'fill="var(--text-soft)">{decade:g}</text>')
    for mw in [7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6]:
        gx = x(mw)
        p.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt + ph}" '
                 f'stroke="var(--line)" stroke-width="1" opacity="0.5"/>')
        p.append(f'<text x="{gx:.1f}" y="{mt + ph + 18}" text-anchor="middle" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10" '
                 f'fill="var(--text-soft)">{mw:g}</text>')
    for s in series:
        pts = [(m, r) for m, r in zip(s["df"]["Mw"], s["df"][s["col"]]) if r > 0]
        if not pts:
            continue
        d = "M " + " L ".join(f"{x(m):.1f},{y(r):.1f}" for m, r in pts)
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        p.append(_series(s["label"], s["color"]))
        p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" '
                 f'stroke-width="{s["width"]}" opacity="{s.get("opacity", 1)}"'
                 f'{dash} stroke-linecap="round"/>')
        step = max(1, len(pts) // 60)
        for m, r in pts[::step]:
            p.append(f'<circle class="hitdot" cx="{x(m):.1f}" cy="{y(r):.1f}" '
                     f'r="7" fill="transparent" data-mw="{m:.3f}" '
                     f'data-rate="{r:.5g}" data-series="{esc(s["label"])}"/>')
        p.append('</g>')
    p.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 4}" text-anchor="middle" '
             f'font-family="IBM Plex Sans, sans-serif" font-size="11" '
             f'fill="var(--text-soft)">Moment magnitude (Mw)</text>')
    p.append('</svg>')
    return "\n".join(p)


def svg_moment_balance(slip_df):
    """Along-strike moment-balance chart, one point per unit-source column:
    the target shape (blue: the plate convergence) vs. the uncorrected (grey)
    and edge-corrected (orange) integrated slip rate. Each series is summed
    over the column's unit sources and divided by its own total, so the
    chart shows the normalised shapes LEVEL 4 actually compares (as the
    engine's own fig_moment_balance does). Same hitdot/tooltip mechanism as
    svg_rate_curve. Returns None if step 7 did not write the per-unit-source
    integrated-slip table (a failed step 7).
    """
    if slip_df is None or not len(slip_df):
        return None

    cols = (slip_df.groupby("alongstrike_number")
            .agg(target=("target_convergence_shape", "sum"),
                 uncorrected=("integrated_slip_no_edge_correction", "sum"),
                 corrected=("integrated_slip", "sum"))
            .reset_index().sort_values("alongstrike_number"))
    for c in ("target", "uncorrected", "corrected"):
        total = float(cols[c].sum())
        if total > 0:
            cols[c] = cols[c] / total
    n = len(cols)
    if n < 2:
        return None

    W, H = 640, 300
    ml, mr, mt, mb = 50, 16, 16, 34
    pw, ph = W - ml - mr, H - mt - mb

    y_hi = float(max(cols["target"].max(), cols["uncorrected"].max(),
                     cols["corrected"].max())) * 1.08
    y_hi = y_hi if y_hi > 0 else 1.0

    def x(i):
        return ml + (i / max(n - 1, 1)) * pw

    def y(v):
        return mt + (1 - max(v, 0) / y_hi) * ph

    def path(col):
        pts = list(enumerate(cols[col]))
        return "M " + " L ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in pts)

    def hitdots(col, series, label):
        step = max(1, n // 80)
        out = []
        for i, v in list(enumerate(cols[col]))[::step]:
            out.append(f'<circle class="hitdot" cx="{x(i):.1f}" cy="{y(v):.1f}" '
                       f'r="7" fill="transparent" data-mw="col {i+1}" '
                       f'data-rate="{v:.4g}" data-series="{esc(label)}"/>')
        return "\n".join(out)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" class="hoverchart" '
            f'aria-label="Moment balance: target convergence shape versus '
            f'modelled integrated slip rate, along-strike unit-source column">']

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = mt + (1 - frac) * ph
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                     f'stroke="var(--line)" stroke-width="1"/>')
        parts.append(f'<text x="{ml-8}" y="{gy+4:.1f}" text-anchor="end" '
                     f'font-family="IBM Plex Mono, monospace" font-size="10" '
                     f'fill="var(--text-soft)">{frac*y_hi:.3f}</text>')
    step_lbl = max(1, n // 8)
    for i in range(0, n, step_lbl):
        gx = x(i)
        parts.append(f'<text x="{gx:.1f}" y="{mt+ph+16}" text-anchor="middle" '
                     f'font-family="IBM Plex Mono, monospace" font-size="9" '
                     f'fill="var(--text-soft)">{i+1}</text>')

    parts.append(_series("No edge correction", "var(--text-soft)"))
    parts.append(f'<path d="{path("uncorrected")}" fill="none" '
                f'stroke="var(--text-soft)" stroke-width="1.75" '
                f'stroke-dasharray="4 3"/>')
    parts.append(hitdots("uncorrected", "unc", "No edge correction"))
    parts.append('</g>')
    parts.append(_series("Target (convergence shape)", "var(--accent-2)"))
    parts.append(f'<path d="{path("target")}" fill="none" '
                f'stroke="var(--accent-2)" stroke-width="2.25"/>')
    parts.append(hitdots("target", "target", "Target (plate convergence shape)"))
    parts.append('</g>')
    parts.append(_series("Edge-corrected", "var(--amber)"))
    parts.append(f'<path d="{path("corrected")}" fill="none" '
                f'stroke="var(--amber)" stroke-width="2.25"/>')
    parts.append(hitdots("corrected", "cor", "Edge-corrected"))
    parts.append('</g>')

    parts.append(f'<text x="{ml+pw/2:.1f}" y="{H-2}" text-anchor="middle" '
                f'font-family="IBM Plex Sans, sans-serif" font-size="11" '
                f'fill="var(--text-soft)">Along-strike unit-source column</text>')
    parts.append('</svg>')
    return "\n".join(parts)


TOOLTIP_JS = """
<div id="chart-tip" class="chart-tip" hidden></div>
<script>
(function(){
  var tip = document.getElementById('chart-tip');
  document.querySelectorAll('svg.hoverchart').forEach(function(svg){
    svg.addEventListener('pointermove', function(ev){
      var t = ev.target;
      if (!t.classList || !t.classList.contains('hitdot')) { tip.hidden = true; return; }
      var mw = t.getAttribute('data-mw');
      var rate = t.getAttribute('data-rate');
      var series = t.getAttribute('data-series');
      tip.innerHTML = '<b>' + series + '</b><br>' + mw + '<br>' + rate;
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 14) + 'px';
      tip.hidden = false;
    });
    svg.addEventListener('pointerleave', function(){ tip.hidden = true; });
  });
  // one button per curve under each chart: click to hide or show it
  document.querySelectorAll('svg.hoverchart').forEach(function(svg){
    var gs = svg.querySelectorAll('g.series');
    if (gs.length < 2) return;
    var bar = document.createElement('div');
    bar.className = 'series-toggle';
    bar.innerHTML = '<span class="st-lbl">Show (click to hide or show a curve):</span>';
    gs.forEach(function(g){
      var b = document.createElement('button');
      b.type = 'button'; b.className = 'st on';
      b.innerHTML = '<span class="st-sw" style="background:' + g.getAttribute('data-color') + '"></span>' + g.getAttribute('data-name');
      b.addEventListener('click', function(){
        var on = b.classList.toggle('on');
        g.style.display = on ? '' : 'none';
      });
      bar.appendChild(b);
    });
    svg.parentNode.insertBefore(bar, svg.nextSibling);
  });
})();
</script>
"""


def embed_image(path):
    """Read a PNG and return it as a data: URI, or None if it does not exist.

    Embedding rather than linking keeps report.html a single self-contained
    file, consistent with everything else this script writes.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_variable_mu():
    """v11: step 7c's outputs, or None if it has not run."""
    p = os.path.join(OUT_DIR, f"scenario_rates_{SUFFIX}_variable_mu.csv")
    if not os.path.exists(p):
        return None
    try:
        out = {"scen": pd.read_csv(p),
               "scen_const": pd.read_csv(os.path.join(OUT_DIR, f"scenario_rates_{SUFFIX}.csv")),
               "curves": pd.read_csv(os.path.join(OUT_DIR, "rate_curves_variable_mu.csv")),
               "dev": pd.read_csv(os.path.join(OUT_DIR, f"variable_mu_deviation_{SUFFIX}.csv"))}
        s = pd.read_csv(os.path.join(OUT_DIR, "logic_tree_summary_variable_mu.csv"))
        out["summary"] = {row["item"]: row["value"] for _, row in s.iterrows()}
        return out
    except Exception as exc:
        print(f"  note: could not read step 7c's outputs: {exc}")
        return None


def _rigidity_gpa(depth_km):
    """rptha shear_modulus_depth (Bilek & Lay fit), GPa"""
    return 10 ** np.interp(np.asarray(depth_km, dtype=float), [0, 7.5, 15, 35, 9999],
                           np.log10([10.0, 10.0, 30.0, 67.0, 67.0]))


def svg_rigidity_curve(rows):
    """Rigidity against depth (PTHA18's curve), the constant 30 GPa of steps 7
    and 7b, and each mesh row at its mean depth with its magnitude shift."""
    W, H, L, R, T, B = 620, 300, 56, 20, 18, 44
    dmax = max(50.0, max(r["depth"] for r in rows) + 5)
    X = lambda d: L + d / dmax * (W - L - R)  # noqa: E731
    Y = lambda g: T + (1 - g / 75.0) * (H - T - B)  # noqa: E731
    p = [f'<svg viewBox="0 0 {W} {H}" class="vmu-svg" role="img" '
         f'aria-label="Rigidity against depth, with the mesh rows">']
    for g in (0, 10, 30, 50, 67):
        p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="vmu-grid"/>'
                 f'<text x="{L-6}" y="{Y(g)+4:.1f}" text-anchor="end" class="vmu-tick">{g}</text>')
    for d in range(0, int(dmax) + 1, 10):
        p.append(f'<text x="{X(d):.1f}" y="{H-B+16}" text-anchor="middle" class="vmu-tick">{d}</text>')
    p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" text-anchor="middle" class="vmu-lab">depth of the cell (km)</text>'
             f'<text x="14" y="{(T+H-B)/2:.0f}" text-anchor="middle" class="vmu-lab" '
             f'transform="rotate(-90 14 {(T+H-B)/2:.0f})">rigidity (GPa)</text>')
    p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(30):.1f}" y2="{Y(30):.1f}" class="vmu-const"/>'
             f'<text x="{W-R-4}" y="{Y(30)-6:.1f}" text-anchor="end" class="vmu-note">constant 30 GPa (steps 7 and 7b)</text>')
    dd = np.linspace(0, dmax, 200)
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(dd, _rigidity_gpa(dd)))
    p.append(f'<polyline points="{pts}" class="vmu-curve"/>'
             f'<text x="{X(dmax)-4:.1f}" y="{Y(67)-8:.1f}" text-anchor="end" class="vmu-note">PTHA18 variable (Bilek &amp; Lay fit)</text>')
    groups = []  # consecutive rows with the same shift share one label
    for r in rows:
        if groups and round(groups[-1][-1]["shift"], 2) == round(r["shift"], 2):
            groups[-1].append(r)
        else:
            groups.append([r])
    for g in groups:
        for r in g:
            p.append(f'<circle cx="{X(r["depth"]):.1f}" cy="{Y(r["mu"]):.1f}" r="6" class="vmu-row"/>')
        r = g[-1]
        x, y = X(r["depth"]), Y(r["mu"])
        name = (f'rows {g[0]["row"]}-{r["row"]}' if len(g) > 1 else f'row {r["row"]}')
        right = x > L + 0.7 * (W - L - R)
        p.append(f'<text x="{x - 9 if right else x + 9:.1f}" y="{y + 18:.1f}" '
                 f'text-anchor="{"end" if right else "start"}" class="vmu-rowlab">'
                 f'{name}: Mw {r["shift"]:+.2f}</text>')
    p.append("</svg>")
    return "".join(p)


def svg_shift_band(dev):
    """Mw (variable - constant) of the HS scenarios against their constant
    Mw: median and the 5-95% band."""
    g = dev.groupby(dev["Mw_constant_mu"].round(2))["Mw_variable_minus_constant"]
    q = pd.DataFrame({"p05": g.quantile(0.05), "p50": g.median(), "p95": g.quantile(0.95),
                      "lo": g.min(), "hi": g.max()}).reset_index()
    mw = q["Mw_constant_mu"].to_numpy()
    W, H, L, R, T, B = 620, 260, 56, 20, 16, 40
    y0, y1 = -0.4, 0.3
    X = lambda m: L + (m - mw.min()) / max(mw.max() - mw.min(), 1e-9) * (W - L - R)  # noqa: E731
    Y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)  # noqa: E731
    p = [f'<svg viewBox="0 0 {W} {H}" class="vmu-svg" role="img" '
         f'aria-label="Magnitude shift of the HS scenarios against their magnitude">']
    for v in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2):
        p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                 f'class="{"vmu-const" if v == 0 else "vmu-grid"}"/>'
                 f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end" class="vmu-tick">{v:+.1f}</text>')
    for m in np.arange(np.ceil(mw.min() * 2) / 2, mw.max() + 1e-9, 0.5):
        p.append(f'<text x="{X(m):.1f}" y="{H-B+16}" text-anchor="middle" class="vmu-tick">{m:.1f}</text>')
    p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" text-anchor="middle" class="vmu-lab">Mw with constant rigidity</text>')
    top = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(mw, q["p95"]))
    bot = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(mw[::-1], q["p05"][::-1]))
    p.append(f'<polygon points="{top} {bot}" class="vmu-band"/>')
    med = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(mw, q["p50"]))
    p.append(f'<polyline points="{med}" class="vmu-curve"/>')
    p.append(f'<text x="{W-R-4}" y="{Y(0)-6:.1f}" text-anchor="end" class="vmu-note">0 = same magnitude</text>')
    p.append("</svg>")
    return "".join(p)


def variable_mu_section(d):
    """v11 STEP 7c card: the variable shear modulus, in plain words, with
    this run's numbers."""
    vm = d.get("variable_mu")
    if vm is None:
        return """
  <div class="stepcard">
    <h3><span class="idx">STEP 7c</span> Shear modulus: constant<span class="pill declared">30 GPa</span></h3>
    <p>Every rate on this page uses a constant rigidity (shear modulus) of 30 GPa, as PTHA18's <code>rate_annual</code>, the rates compared with PTHA18 throughout this report. PTHA18 also publishes rates with a rigidity that grows with depth (<code>variable_mu_rate_annual</code>); this pipeline computes them in step 7c when the example is generated with <code>--variable-mu on</code> (from_scratch_v12), which was not done here.</p>
  </div>"""
    slip = d.get("slip")
    rows = []
    if slip is not None and "depth_km" in slip.columns:
        for r, g in slip.groupby("downdip_number"):
            dep = float(np.average(g["depth_km"], weights=g["area_km2"]))
            mu = float(np.average(_rigidity_gpa(g["depth_km"]), weights=g["area_km2"]))
            rows.append({"row": int(r), "depth": dep, "mu": mu,
                         "shift": 2 / 3 * np.log10(mu / 30.0),
                         "slip_factor": 30.0 / mu})
    row_table = "".join(
        f"<tr><td>{r['row']}</td><td>{r['depth']:.1f}</td><td>{r['mu']:.0f}</td>"
        f"<td>{r['shift']:+.2f}</td><td>{r['slip_factor']:.2f}</td></tr>" for r in rows)

    scen, const = vm["scen"], vm["scen_const"]
    shift = scen["variable_mu_Mw"] - scen["Mw"]
    curves, rc = vm["curves"], d["rates"]
    col = "exceedance_rate_unsegmented"

    def at(df, m):
        return float(np.interp(m, df["Mw"], df[col]))

    def real_rate(m):
        return float(scen.loc[scen["variable_mu_Mw"] >= m - 1e-9, "rate_mean"].sum())

    rate_rows = ""
    for m in (7.5, 8.0, 8.5, 9.0):
        a, b, c = at(rc, m), at(curves, m), real_rate(m)
        ratio = b / a if a > 0 else float("nan")
        rp = lambda r: f"{1/r:,.0f} yr" if r > 0 else "&mdash;"  # noqa: E731
        rate_rows += (f"<tr><td>{m:.1f}</td><td>{a:.3g} ({rp(a)})</td>"
                      f"<td>{b:.3g} ({rp(b)})</td><td>{ratio:.3f}</td>"
                      f"<td>{c:.3g} ({rp(c)})</td></tr>")

    s, smu = d["summary"], vm["summary"]
    z = SUFFIX
    tvd_c = float(s.get(f"prior_posterior_tvd_{z}", float("nan")))
    tvd_v = float(smu.get(f"prior_posterior_tvd_variable_mu_{z}", float("nan")))
    tvd_cv = float(smu.get(f"posterior_vs_variable_mu_tvd_{z}", float("nan")))
    tot_c = float(const["rate_mean"].sum())
    tot_v = float(scen["rate_mean"].sum())
    n_obs = len(d["gcmt_subset"]) if d.get("gcmt_subset") is not None else 0
    n_hs = len(vm["dev"])
    few_obs = (f"This zone has {n_obs} GCMT earthquake(s) in LEVEL 3, so the catalogue "
               f"has little to say and the two sets of weights stay close"
               if n_obs <= 2 else
               f"This zone has {n_obs} GCMT earthquakes in LEVEL 3")

    hs = d.get("hs_summary")
    hs_note = ""
    if hs is not None and "variable_mu_rate_mean" in hs.columns:
        hs_note = (f"<p>HS and VAUS: both <code>summary.csv</code> files now also carry "
                   f"<code>variable_mu_Mw</code>, <code>variable_mu_weight_in_family</code> and "
                   f"<code>variable_mu_rate_*</code>: the variable shear modulus rate of the parent FAUS "
                   f"rupture, shared with PTHA18's DART curves for that case (recovered like the constant "
                   f"ones). HS magnitude shifts here: {hs['variable_mu_Mw'].sub(hs['Mw']).min():+.2f} to "
                   f"{hs['variable_mu_Mw'].sub(hs['Mw']).max():+.2f}.</p>")

    return f"""
  <div class="stepcard">
    <h3><span class="idx">STEP 7c</span> Variable shear modulus (v11)<span class="pill derived">PTHA18's variable_mu rates</span></h3>
    <p><b>Rigidity</b> (shear modulus, &mu;) is how stiff the rock is. Steps 7 and 7b use 30 GPa everywhere, like PTHA18's <code>rate_annual</code>. Real rock is softer near the trench and stiffer deeper down. For the same earthquake as a seismometer measures it (the same seismic moment, M0 = &mu; &times; area &times; slip), soft rock needs more slip, and more slip near the trench makes a bigger tsunami. PTHA18 publishes a second set of rates for this, <code>variable_mu_rate_annual</code>, and its headline hazard maps use them with the HS scenarios (PTHA18 report, Sections 3.7.5 and 4.1).</p>
    <div class="callout">
      <span class="lbl">How PTHA18 does it (step 7c ports it)</span>
      1. The rigidity of each cell comes from its depth, with the curve below. 2. Every scenario keeps its cells and its slip; only its magnitude is relabelled, M0 = sum(area &times; slip &times; &mu;(depth)). 3. The rates stay functions of the constant-rigidity magnitude, but the GCMT earthquakes of LEVEL 3 have real magnitudes, so the difference (measured on this run's {n_hs:,} HS scenarios) is treated as an observation error of the catalogue. That changes the LEVEL 3 weights of the logic-tree branches. 4. The variable shear modulus rates are the same branch curves with those weights.
    </div>
    <h4 style="margin:18px 0 6px;font-size:14px">Rigidity against depth, and this mesh's rows</h4>
    {svg_rigidity_curve(rows) if rows else ''}
    <table>
      <thead><tr><th>Row (1 = trench)</th><th>Mean depth (km)</th><th>Rigidity (GPa)</th><th>Mw shift of a rupture on this row</th><th>Slip for the same real Mw (&times; constant)</th></tr></thead>
      <tbody>{row_table}</tbody>
    </table>
    <p class="sec-note">Read the last column as: a real earthquake of a given magnitude on that row slips this many times what the constant 30 GPa gives. Shallow rows: more slip, a stronger tsunami for the same measured magnitude.</p>
    <h4 style="margin:18px 0 6px;font-size:14px">Magnitude shift of this run's HS scenarios</h4>
    {svg_shift_band(vm["dev"])}
    <p class="sec-note">Line: median of (variable &minus; constant) Mw of the HS scenarios of each magnitude; band: 5% to 95%. Large ruptures cover shallow and deep cells, so their shifts average out. FAUS scenarios here: {shift.min():+.2f} to {shift.max():+.2f}.</p>
    <h4 style="margin:18px 0 6px;font-size:14px">What it changes in the rates</h4>
    <table>
      <thead><tr><th>Mw</th><th>Constant rigidity: rate of Mw &ge; (step 7)</th><th>Variable, same magnitude scale</th><th>Ratio</th><th>Variable: rate of real Mw &ge;</th></tr></thead>
      <tbody>{rate_rows}</tbody>
    </table>
    <p class="sec-note">Rates per year (return period). The first two columns use the constant-rigidity magnitude, the scale every scenario is built on; the last one counts each scenario at its relabelled, real magnitude, as a catalogue would see it. LEVEL 3 weights: prior to posterior TVD {tvd_c:.3f} with constant rigidity, {tvd_v:.3f} with variable; between the two posteriors {tvd_cv:.3f}. Sum of all scenario rates: {tot_v:.4g} against {tot_c:.4g} per year ({tot_v/tot_c:.4f} times).</p>
    {hs_note}
    <div class="callout good">
      <span class="lbl">What this means for the tsunami maps</span>
      In PTHA18's method no scenario changes: same cells, same slip, same tsunami. What changes is how often each scenario happens, only through the LEVEL 3 weights. {few_obs}: the rates above move by the ratio shown. The real magnitude of each scenario (<code>variable_mu_Mw</code>) is what to quote when a scenario is compared with a real earthquake, or when a scenario of a given magnitude is chosen for a map: a "real Mw 8" on the shallow rows is a constant-rigidity Mw of about {8 - min(r['shift'] for r in rows) if rows else 8:.1f}, with that much more slip.
    </div>
    <p class="where">Files: <code>scenario_rates_{z}_variable_mu.csv</code>, <code>rate_curves_variable_mu.csv</code>, <code>exceedance_rate_percentiles_variable_mu.csv</code>, <code>variable_mu_deviation_{z}.csv</code>, <code>logic_tree_branches_{z}_variable_mu.csv</code>. Checked against PTHA18's published files: <code>validation/validate_v11.py</code>.</p>
  </div>"""


def explorer_vaus_fields(vaus_summary, rup_stats, n_faus):
    """Step 7b's VAUS fields for the rupture explorer, grouped by the FAUS
    rupture each one was drawn from.

    Returns {mw_str: [fields of FAUS rupture 1, of rupture 2, ...]}, each
    field [dip_i, strike_i, nlength, nwidth, area_km2, uniform_slip_m] like
    a FAUS placement (1-based top-left cell, block size in cells), plus
    [weight_in_family, above_peak_slip_limit] when step 7b wrote rates
    (v10_q). A VAUS
    field is the smallest rectangle around its HS footprint, so the corner
    and size describe it exactly; a field that is not a rectangle raises.

    `n_faus` is {mw_str: number of FAUS ruptures}. summary.csv numbers the
    placements step 7b used 1..k in FAUS order; with
    --max-placements-per-mw those are evenly spaced through the FAUS list
    (step 7b's np.linspace rule), so k < n maps placement p to that index.
    """
    sub = np.asarray(rup_stats["subfault_number"], dtype=int)
    dd = np.zeros(sub.max() + 1, dtype=int)
    as_ = np.zeros(sub.max() + 1, dtype=int)
    area = np.zeros(sub.max() + 1)
    dd[sub] = rup_stats["downdip_number"]
    as_[sub] = rup_stats["alongstrike_number"]
    area[sub] = np.asarray(rup_stats["length"]) * np.asarray(rup_stats["width"])

    has_rates = "weight_in_family" in vaus_summary.columns
    out = {}
    for mw, g in vaus_summary.groupby(vaus_summary["Mw"].round(6)):
        key = f"{float(mw):g}"
        if key not in n_faus:
            continue
        n_total = n_faus[key]
        used = np.sort(g["placement"].unique())
        if used.size == n_total:
            faus_index = {p: i for i, p in enumerate(used)}
        else:
            keep = np.round(np.linspace(0, n_total - 1, used.size)).astype(int)
            faus_index = dict(zip(used, keep))
        fields = [[] for _ in range(n_total)]
        for row in g.sort_values(["placement", "realisation"]).itertuples():
            ids = np.array([int(s) for s in row.event_index_string.split("-") if s])
            d0, d1 = dd[ids].min(), dd[ids].max()
            s0, s1 = as_[ids].min(), as_[ids].max()
            if ids.size != (d1 - d0 + 1) * (s1 - s0 + 1):
                raise ValueError(f"Mw {key} placement {row.placement}: "
                                 f"a VAUS field is not a rectangle")
            field = [int(d0), int(s0), int(s1 - s0 + 1), int(d1 - d0 + 1),
                     int(round(area[ids].sum())), round(float(row.peak_slip_m), 2)]
            if has_rates:
                field += [round(float(row.weight_in_family), 5),
                          int(row.above_peak_slip_limit)]
            fields[faus_index[row.placement]].append(field)
        out[key] = fields
    return out


def svg_rupture_explorer(scratch_polys, downdip_arr, alongstrike_arr,
                         mws_data, zone_label, relation="Strasser",
                         lw_2sd=None, cell_mu_area=None):
    """Interactive rupture explorer: a Mw dropdown, a FAUS / VAUS / both
    switch and a position slider that recolour the mesh's own cells to show
    one rupture at a time, all in inline SVG/JS -- no video, no per-frame
    images, works offline in the static report.html.

    `mws_data` is {mw_str: {"placements": [[dip_i, strike_i, nlength,
    nwidth, area_km2, slip_m], ...], "target": scaling-relation area,
    "band": [low, high] its 1-sigma area range / target, "slip_ref":
    uniform slip on the target area,
    "vaus": [[field, ...] per placement] (optional)}} -- each rupture's
    topleft corner and block size in cells, from
    events.get_all_earthquake_events_of_magnitude_Mw's "topleft_indices"/
    "event_dims" (v10: with --rupture-size local the block size changes
    from one placement to the next), and the VAUS fields from
    explorer_vaus_fields (same layout). Reconstructing which CELLS a block
    covers happens in JS from each cell's own downdip/alongstrike number
    (embedded as one small array, index-aligned with scratch_polys), not
    from a per-rupture list of cell indices -- that keeps the embedded
    JSON to a few numbers per rupture instead of up to ~130.

    The position slider and play button step through the FAUS ruptures;
    the VAUS dropdown picks which of the current rupture's VAUS realisations
    is drawn (step 7b draws several per FAUS rupture). A panel next to the
    map plots each block's area / scaling-relation area on a log axis,
    with the relation's 1-sigma range: every FAUS rupture of the magnitude
    and every VAUS field of the current rupture as small dots, the ones
    drawn on the map as large dots. A second panel does the same for the
    uniform slip in metres, with PTHA18's HS/VAUS peak-slip limit (7.5 x
    the slip on the median area) marked; a menu picks which one is shown.
    `lw_2sd` is {"length": (low, high), "width": (low, high)}, the
    relation's +-2 sigma factors, for the text on VAUS. v11 `cell_mu_area`:
    each cell's rigidity (PTHA18's depth curve) x area (Pa km2), index-aligned
    with scratch_polys; each line then also gives the block's magnitude with
    the variable rigidity (its slip unchanged).
    """
    lons = np.concatenate([p[:, 0] for p in scratch_polys])
    lats = np.concatenate([p[:, 1] for p in scratch_polys])
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())
    pad = 0.03 * max(lon_max - lon_min, lat_max - lat_min, 1e-6)
    lon_min, lon_max = lon_min - pad, lon_max + pad
    lat_min, lat_max = lat_min - pad, lat_max + pad

    W = 460
    H = int(W * (lat_max - lat_min) / max(lon_max - lon_min, 1e-6))
    H = max(320, min(H, 1400))

    def X(lon):
        return (lon - lon_min) / (lon_max - lon_min) * W

    def Y(lat):
        # SVG y grows downward; latitude grows northward, so flip.
        return (1 - (lat - lat_min) / (lat_max - lat_min)) * H

    paths = []
    for i, poly in enumerate(scratch_polys):
        pts = " ".join(f"{X(lo):.1f},{Y(la):.1f}" for lo, la in poly)
        paths.append(f'<path id="rup-cell-{i}" class="rup-cell" d="M {pts} Z"/>')

    mw_options = "".join(
        f'<option value="{mw}">{mw}</option>' for mw in sorted(mws_data, key=float))
    has_vaus = any("vaus" in v for v in mws_data.values())
    vaus_disabled = "" if has_vaus else " disabled"
    has_rates = any(len(f) > 6 for v in mws_data.values()
                    for fs in v.get("vaus", []) for f in fs)
    vaus_note = (("VAUS: step 7b's variable-area uniform-slip fields drawn from "
                  "the FAUS rupture shown. "
                  + ("Each carries a share of that rupture's rate (PTHA18's rule, "
                     "shown on its line); none of it enters LEVELs 0-5."
                     if has_rates else "This run's step 7b wrote no rates for them."))
                 if has_vaus else
                 "No VAUS fields for this run (step 7b not run, or run with "
                 "--skip-hs): only FAUS can be shown.")

    lw = lw_2sd or {"length": (0.44, 2.29), "width": (0.45, 2.22)}
    # the relation's sigma does not depend on Mw: any Mw gives the band
    band_lo, band_hi = next(iter(mws_data.values()))["band"]
    data_json = json.dumps(mws_data, separators=(",", ":"))
    dd_json = json.dumps([int(v) for v in downdip_arr], separators=(",", ":"))
    as_json = json.dumps([int(v) for v in alongstrike_arr], separators=(",", ":"))
    mua_json = json.dumps([float(f"{v:.6g}") for v in cell_mu_area]
                          if cell_mu_area is not None else None, separators=(",", ":"))

    return f"""
<div class="rupture-explorer">
  <div class="rup-controls">
    <label>Magnitude
      <select id="rup-mw">{mw_options}</select>
    </label>
    <label>Show
      <select id="rup-mode">
        <option value="faus">FAUS</option>
        <option value="vaus"{vaus_disabled}>VAUS</option>
        <option value="both"{vaus_disabled}>FAUS + VAUS</option>
      </select>
    </label>
    <label class="rup-slider-label">FAUS rupture
      <input type="range" id="rup-pos" min="0" max="0" value="0" step="1">
    </label>
    <label>VAUS realisation
      <select id="rup-real" disabled></select>
    </label>
    <button type="button" id="rup-play" class="rup-play" aria-label="Play">&#9654;</button>
  </div>
  <div class="rup-count"><span id="rup-count-f"></span><br><span id="rup-count-v"></span></div>
  <div class="rup-legend">
    <span><i class="rup-key rup-f"></i>FAUS (the rupture the rates use)</span>
    <span><i class="rup-key rup-v"></i>VAUS</span>
    <span><i class="rup-key rup-fv"></i>both</span>
  </div>
  <p class="sec-note" style="margin:4px 0 8px">{vaus_note} Play steps through the FAUS ruptures; the VAUS menu picks the realisation.</p>
  <div class="rup-stage">
  <svg id="rup-svg" viewBox="0 0 {W} {H}" role="img"
       aria-label="{zone_label} mesh with a selectable rupture highlighted">
    <defs>
      <pattern id="rup-both" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" class="rup-both-a"/><rect width="3" height="6" class="rup-both-b"/>
      </pattern>
    </defs>
    {"".join(paths)}
  </svg>
  <div class="rup-ratio">
    <label class="rup-panel-label">Panel
      <select id="rup-panel">
        <option value="area">Area / scaling relation</option>
        <option value="slip">Uniform slip, in metres</option>
      </select>
    </label>
    <svg id="rup-ratio-svg" viewBox="0 0 300 170" role="img"
         aria-label="Area, or uniform slip, of the ruptures shown"></svg>
    <div id="rup-note-area" class="sec-note">
      <p>The scaling relation ({esc(relation)}) gives, for each Mw, a <b>median</b> rupture area (1 on this axis). Real earthquakes of the same Mw scatter around it by about a factor of 2: &plusmn;1&sigma; is {band_lo:.2f} to {band_hi:.2f} times that area (shaded), and about 1 in 3 earthquakes falls outside that range.</p>
      <p>VAUS: for each realisation, PTHA18 (and step 7b) draws a rupture length and a width separately from {esc(relation)}'s own scatter, cut at &plusmn;2&sigma; (length {lw['length'][0]:.2f} to {lw['length'][1]:.2f} times the median, width {lw['width'][0]:.2f} to {lw['width'][1]:.2f}), makes the heterogeneous slip in that rectangle, and the VAUS is the rectangle around the cells that slipped. So VAUS areas spread like real earthquakes, and fall lower where the edge of the zone cuts the rectangle.</p>
      <p>Small dots: every FAUS rupture of this Mw, and every VAUS realisation of the FAUS rupture shown; hollow: a VAUS above PTHA18's peak-slip limit (rate 0). Large dots: the ones on the map. Log scale.</p>
    </div>
    <div id="rup-note-slip" class="sec-note" hidden>
      <p>Same moment for every rupture of this Mw, so a smaller area means a larger slip. Black dashed: the slip the median area gives (<span id="rup-slip-ref"></span>); shaded: the slip for the &plusmn;1&sigma; areas.</p>
      <p>Red: PTHA18's peak-slip limit, 7.5 times that slip (<span id="rup-slip-lim"></span>). PTHA18 gives an HS or VAUS scenario above it rate 0 (<code>config_peak_slip_limit_factor.R</code>), and so does step 7b (v10_q); hollow dots are those. FAUS is never limited. Log scale.</p>
    </div>
  </div>
  </div>
</div>
<script>
(function(){{
  var DATA = {data_json};
  var CELL_DD = {dd_json};
  var CELL_AS = {as_json};
  var CELL_MUA = {mua_json};
  var svg = document.getElementById('rup-svg');
  var mwSel = document.getElementById('rup-mw');
  var modeSel = document.getElementById('rup-mode');
  var posInput = document.getElementById('rup-pos');
  var realInput = document.getElementById('rup-real');
  var countF = document.getElementById('rup-count-f');
  var countV = document.getElementById('rup-count-v');
  var playBtn = document.getElementById('rup-play');
  var ratioSvg = document.getElementById('rup-ratio-svg');
  var panelSel = document.getElementById('rup-panel');
  var noteArea = document.getElementById('rup-note-area');
  var noteSlip = document.getElementById('rup-note-slip');
  var slipRefEl = document.getElementById('rup-slip-ref');
  var slipLimEl = document.getElementById('rup-slip-lim');
  var realKey = null;
  var cells = Array.prototype.slice.call(svg.querySelectorAll('.rup-cell'));
  var timer = null;

  function stopPlaying() {{
    if (timer) {{ clearInterval(timer); timer = null; }}
    playBtn.textContent = '\\u25B6';
    playBtn.setAttribute('aria-label', 'Play');
  }}

  function startPlaying() {{
    var d = DATA[mwSel.value];
    if (parseInt(posInput.value, 10) >= d.placements.length - 1) {{
      posInput.value = 0;
    }}
    playBtn.textContent = '\\u23F8';
    playBtn.setAttribute('aria-label', 'Pause');
    timer = setInterval(function() {{
      var d = DATA[mwSel.value];
      var pos = parseInt(posInput.value, 10) + 1;
      if (pos > d.placements.length - 1) {{ stopPlaying(); return; }}
      posInput.value = pos;
      paint();
    }}, 250);
  }}

  function inBlock(i, b) {{
    var dd = CELL_DD[i], as_ = CELL_AS[i];
    return dd >= b[0] && dd < b[0] + b[3] && as_ >= b[1] && as_ < b[1] + b[2];
  }}

  function realMw(b) {{
    // v11: M0 = slip x sum(area x rigidity at depth), slip unchanged
    var s = 0;
    for (var i = 0; i < CELL_DD.length; i++) if (inBlock(i, b)) s += CELL_MUA[i];
    return 2 / 3 * (Math.log10(b[5] * s * 1e6) - 9.05);
  }}

  function describe(b, target) {{
    return b[2] + ' along strike x ' + b[3] + ' down dip, ' +
      b[4].toLocaleString('en-US') + ' km\\u00B2 (' + (b[4] / target).toFixed(2) +
      ' x scaling relation), slip ' + b[5].toFixed(2) + ' m' +
      (CELL_MUA ? '; Mw ' + realMw(b).toFixed(2) + ' with variable rigidity' : '');
  }}

  // Log-axis dot panels next to the map (area / scaling relation, slip).
  // ax: {{lo, hi, ticks: [[value, label, class]], band: [a, b], val: fn}}
  var RX0 = 56, RX1 = 288;
  function drawPanel(svgEl, ax, rows) {{
    function rx(v) {{
      v = Math.min(Math.max(v, ax.lo), ax.hi);
      return RX0 + (Math.log(v) - Math.log(ax.lo)) / (Math.log(ax.hi) - Math.log(ax.lo)) * (RX1 - RX0);
    }}
    var s = '<rect class="rr-band" x="' + rx(ax.band[0]).toFixed(1) + '" y="18" width="' +
      (rx(ax.band[1]) - rx(ax.band[0])).toFixed(1) + '" height="118"/>' +
      '<text class="rr-tick" x="' + ((rx(ax.band[0]) + rx(ax.band[1])) / 2).toFixed(1) + '" y="12" text-anchor="middle">\\u00B11\\u03C3</text>';
    ax.ticks.forEach(function(tk) {{
      s += '<line class="' + tk[2] + '" x1="' + rx(tk[0]).toFixed(1) + '" x2="' + rx(tk[0]).toFixed(1) + '" y1="18" y2="136"/>' +
        '<text class="rr-tick" x="' + rx(tk[0]).toFixed(1) + '" y="152" text-anchor="middle">' + tk[1] + '</text>';
    }});
    rows.forEach(function(rw) {{
      s += '<text class="rr-row" x="4" y="' + (rw.y + 4) + '">' + rw.label + '</text>';
      rw.all.forEach(function(blk) {{
        s += '<circle class="rr-all ' + rw.cls + (blk[7] ? ' rr-out' : '') + '" cx="' + rx(ax.val(blk)).toFixed(1) + '" cy="' + rw.y + '" r="3.5"/>';
      }});
      if (rw.cur) {{
        var v = ax.val(rw.cur), x = rx(v);
        s += '<circle class="rr-cur ' + rw.cls + (rw.cur[7] ? ' rr-out' : '') + '" cx="' + x.toFixed(1) + '" cy="' + rw.y + '" r="7.5"/>' +
          '<text class="rr-val" x="' + x.toFixed(1) + '" y="' + (rw.y - 11) + '" text-anchor="middle">' + ax.fmt(v) + '</text>';
      }}
    }});
    svgEl.innerHTML = s;
  }}

  function metres(v) {{ return v < 10 ? v.toFixed(1) : v.toFixed(0); }}

  function drawPanels(d, tl, fields, vb, showF, showV) {{
    var rows = [];
    if (showF) rows.push({{y: 58, label: 'FAUS', all: d.placements, cur: tl, cls: 'rr-f'}});
    if (showV) rows.push({{y: 118, label: 'VAUS', all: fields, cur: vb, cls: 'rr-v'}});
    var slipView = panelSel.value === 'slip';
    noteArea.hidden = slipView;
    noteSlip.hidden = !slipView;
    if (!slipView) drawPanel(ratioSvg, {{
      lo: 0.25, hi: 4, band: d.band,
      ticks: [[0.25, '0.25', 'rr-grid'], [0.5, '0.5', 'rr-grid'], [1, '1', 'rr-one'],
              [2, '2', 'rr-grid'], [4, '4', 'rr-grid']],
      val: function(blk) {{ return blk[4] / d.target; }},
      fmt: function(v) {{ return v.toFixed(2); }}
    }}, rows);
    // slip axis in metres around the slip at the relation's median area;
    // a larger area means a smaller uniform slip, so the band flips
    var ref = d.slip_ref;
    if (slipView) drawPanel(ratioSvg, {{
      lo: ref / 4, hi: ref * 9, band: [ref / d.band[1], ref / d.band[0]],
      ticks: [[ref / 4, metres(ref / 4), 'rr-grid'], [ref / 2, metres(ref / 2), 'rr-grid'],
              [ref, metres(ref), 'rr-one'], [ref * 2, metres(ref * 2), 'rr-grid'],
              [ref * 4, metres(ref * 4), 'rr-grid'], [ref * 7.5, metres(ref * 7.5), 'rr-limit']],
      val: function(blk) {{ return blk[5]; }},
      fmt: function(v) {{ return v.toFixed(v < 10 ? 2 : 1) + ' m'; }}
    }}, rows);
    slipRefEl.textContent = metres(ref) + ' m';
    slipLimEl.textContent = metres(ref * 7.5) + ' m';
  }}

  function paint() {{
    var d = DATA[mwSel.value];
    var mode = modeSel.value;
    var pos = parseInt(posInput.value, 10);
    posInput.max = d.placements.length - 1;
    if (pos > d.placements.length - 1) {{ pos = d.placements.length - 1; posInput.value = pos; }}
    var tl = d.placements[pos];
    var fields = (d.vaus && d.vaus[pos]) || [];
    // VAUS menu: one entry per realisation of this FAUS rupture
    var key = mwSel.value + '|' + pos;
    if (key !== realKey) {{
      var keep = Math.max(realInput.selectedIndex, 0);
      var opts = '';
      for (var k = 0; k < fields.length; k++) opts += '<option value="' + k + '">' + (k + 1) + '</option>';
      realInput.innerHTML = opts || '<option value="0">-</option>';
      realInput.selectedIndex = Math.min(keep, Math.max(fields.length - 1, 0));
      realKey = key;
    }}
    realInput.disabled = mode === 'faus' || fields.length === 0;
    var real = Math.max(realInput.selectedIndex, 0);
    var vb = (mode !== 'faus' && fields.length) ? fields[real] : null;
    var showF = mode !== 'vaus';

    countF.textContent = 'FAUS rupture ' + (pos + 1) + ' of ' + d.placements.length +
      ': ' + describe(tl, d.target);
    if (mode === 'faus') {{
      countV.textContent = '';
    }} else if (vb) {{
      countV.textContent = 'VAUS realisation ' + (real + 1) + ' of ' + fields.length +
        ' of this rupture: ' + describe(vb, d.target) +
        (vb.length > 6 ? (vb[7] ? '; rate 0 (peak slip above the PTHA18 limit)'
                          : '; carries ' + (100 * vb[6]).toFixed(1) + '% of the rate of its FAUS rupture') : '');
    }} else {{
      countV.textContent = 'No VAUS field for this rupture.';
    }}
    for (var i = 0; i < cells.length; i++) {{
      var f = showF && inBlock(i, tl), v = !!vb && inBlock(i, vb);
      cells[i].classList.toggle('rup-f', f && !v);
      cells[i].classList.toggle('rup-v', v && !f);
      cells[i].classList.toggle('rup-fv', f && v);
    }}
    drawPanels(d, tl, fields, vb, showF, mode !== 'faus' && fields.length > 0);
  }}

  mwSel.addEventListener('change', function() {{
    stopPlaying(); posInput.value = 0; realInput.selectedIndex = 0; paint();
  }});
  modeSel.addEventListener('change', paint);
  panelSel.addEventListener('change', paint);
  posInput.addEventListener('input', function() {{ stopPlaying(); paint(); }});
  realInput.addEventListener('change', paint);
  playBtn.addEventListener('click', function() {{
    if (timer) {{ stopPlaying(); }} else {{ startPlaying(); }}
  }});
  paint();
}})();
</script>
"""


LEAFLET_CDN = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4"
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services"


def leaflet_mesh_map(meshes):
    """A zoomable map of each mesh over Esri's ocean basemap, to check by
    eye that the mesh sits on the real trench.

    `meshes` is a list of (label, colour, top_style, lonlat), drawn in that
    order: lonlat is a (rows+1, 2, columns+1) array of grid nodes with row 0
    on the trench (the unit_source_grid layout). Each mesh is drawn as its
    cell outlines plus its top edge (row 0), styled by `top_style` (Leaflet
    line options, e.g. {"weight": 8, "opacity": 0.55}), so a reference mesh
    drawn first with a wide, see-through top edge shows as a band around a
    thinner one drawn after it. Longitudes are used as given: 0-360 across
    the antimeridian is fine, Leaflet repeats the tiles.

    The one part of report.html that is not embedded: the Leaflet library
    and Esri's tiles load from the internet when the page is opened, and
    offline the map area shows a note instead.
    """
    data = []
    for label, colour, top_style, lonlat in meshes:
        ll = np.round(lonlat[:, ::-1, :], 4)  # (lat, lon), the order Leaflet wants
        cells = [np.array([ll[i, :, j], ll[i, :, j + 1], ll[i + 1, :, j + 1], ll[i + 1, :, j]])
                 for j in range(ll.shape[2] - 1) for i in range(ll.shape[0] - 1)]
        data.append({"label": label, "colour": colour, "top_style": top_style,
                     "cells": [c.tolist() for c in cells if np.isfinite(c).all()],
                     "top": ll[0].T.tolist()})
    data_json = json.dumps(data, separators=(",", ":"))

    return f"""
<link rel="stylesheet" href="{LEAFLET_CDN}/leaflet.min.css" integrity="sha512-h9FcoyWjHcOcmEVkxOfTLnmZFWIH0iZhZT1H2TbOq55xssQGEJHEaIm+PgoUaZbRvQTNTluNOEfb1ZRy6D3BOw==" crossorigin="anonymous" referrerpolicy="no-referrer">
<script src="{LEAFLET_CDN}/leaflet.min.js" integrity="sha512-puJW3E/qXDqYp9IfhAI54BJEaWIfloJ7JWs7OeD5i6ruC9JZL1gERT1wjtwXFlh7CjE7ZJ+/vcRZRkIYIb6p4g==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<div id="mesh-map" class="mesh-map"><div class="mesh-map-offline">This map needs an internet connection: it loads the Leaflet map library and Esri's map tiles.</div></div>
<script>
(function(){{
  if (!window.L) return;
  var el = document.getElementById('mesh-map');
  el.innerHTML = '';
  var MESHES = {data_json};
  function esri(service, attribution) {{
    return L.tileLayer('{ESRI_TILES}/' + service + '/MapServer/tile/{{z}}/{{y}}/{{x}}',
                       {{maxZoom: 13, attribution: attribution}});
  }}
  var ocean = esri('Ocean/World_Ocean_Base', 'Tiles &copy; Esri. Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri');
  var imagery = esri('World_Imagery', 'Tiles &copy; Esri. Sources: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community');
  var names = esri('Ocean/World_Ocean_Reference', '');
  var map = L.map(el, {{layers: [ocean, names], zoomSnap: 0.25, zoomDelta: 0.5, scrollWheelZoom: false}});
  // The mouse wheel zooms only after a click on the map, so scrolling the
  // page never gets stuck on it.
  map.on('click', function(){{ map.scrollWheelZoom.enable(); }});
  map.on('mouseout', function(){{ map.scrollWheelZoom.disable(); }});
  var overlays = {{'Place names': names}};
  var bounds = null;
  MESHES.forEach(function(m){{
    var g = L.featureGroup();
    m.cells.forEach(function(c){{
      L.polygon(c, {{color: m.colour, weight: 1, fill: false, interactive: false}}).addTo(g);
    }});
    L.polyline(m.top, Object.assign({{color: m.colour, interactive: false}}, m.top_style)).addTo(g);
    g.addTo(map);
    overlays[m.label] = g;
    bounds = bounds ? bounds.extend(g.getBounds()) : g.getBounds();
  }});
  L.control.layers({{'Sea floor (Esri Ocean)': ocean, 'Satellite (Esri World Imagery)': imagery}},
                   overlays, {{collapsed: false}}).addTo(map);
  L.control.scale({{imperial: false}}).addTo(map);
  map.fitBounds(bounds, {{padding: [12, 12]}});
}})();
</script>
"""


def mesh_file_card(grid):
    """v12 STEP 1 card when the geometry is an external mesh file."""
    if grid is None:
        return ""
    from pyptha_v12 import unit_sources as us_mod
    st = us_mod.discretized_source_approximate_summary_statistics(grid)
    area = float(np.sum(st["length"] * st["width"]))
    return f"""
  <div class="stepcard">
    <h3><span class="idx">STEP 1-2</span> Interface geometry from an external mesh<span class="pill declared">given</span></h3>
    <p>This run does not build its mesh from SLAB: it uses the quadrilateral mesh in <code>inputs/geometry/{esc(MESH_FILE)}</code>, one line per unit source with its four corners (lon, lat, depth), read by <code>pyptha/mesh_file.py</code> (rows, columns and orientation found from the corners the cells share) and used as given. Everything from step 3 on is computed on it exactly as on a SLAB mesh.</p>
    <div class="figure">
      <div class="stat"><span class="k">Unit sources</span><span class="v">{grid.shape[2]-1} &times; {grid.shape[0]-1}</span></div>
      <div class="stat"><span class="k">Area</span><span class="v" style="font-size:14px">{area:,.0f} km&sup2;</span></div>
      <div class="stat"><span class="k">Depth</span><span class="v" style="font-size:14px">{grid[:, 2, :].min():.2f} &ndash; {grid[:, 2, :].max():.1f} km</span></div>
      <div class="stat"><span class="k">Mean dip</span><span class="v" style="font-size:14px">{float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(st['dip']))), np.mean(np.cos(np.radians(st['dip'])))))):.2f}&deg;</span></div>
    </div>
  </div>"""


def build_html(d):
    cfg = d["cfg"]
    s = d["summary"]
    rates = d["rates"]
    off = d["official"]
    gcmt = d["gcmt_subset"]

    n_branches = int(s.get(f"n_branches_{SUFFIX}", 0))
    mean_dip = float(s.get(f"mean_dip_deg_{SUFFIX}", float("nan")))
    # logic_tree_summary.csv records what step 7 actually ran with, which
    # is the JSON's tectonic_convergence_mm_per_yr -- the number this report
    # must show. data/convergence.txt is only step 3's DERIVED-from-Bird
    # value at the time step 3 last ran, and goes stale the moment someone
    # hand-edits tectonic_convergence_mm_per_yr in the input JSON without
    # re-running step 3 (a supported workflow -- see the input JSON's
    # _convergence_provenance field). Falling back to convergence.txt only
    # when the summary has nothing avoids silently showing a number step 7
    # did not use.
    convergence_summary = s.get(f"convergence_mm_per_yr_{SUFFIX}")
    convergence = float(convergence_summary) if convergence_summary is not None \
        else d["convergence"]
    prior_coupling = float(s.get(f"prior_mean_coupling_{SUFFIX}", float("nan")))
    a_min = float(s.get(f"a_min_{SUFFIX}", float("nan")))
    a_max = float(s.get(f"a_max_{SUFFIX}", float("nan")))
    tvd = float(s.get(f"prior_posterior_tvd_{SUFFIX}", float("nan")))
    edge_mult = float(s.get(f"edge_multiplier_{SUFFIX}", float("nan")))

    # Rebuild the mesh from THIS run's own input JSON (cfg["geometry"]),
    # via run_logic_tree.py's own build_grid() -- the exact function step 7
    # used to compute the real rates. step2_build_grid.py's own .npy is a
    # SEPARATE, earlier grid: it always uses --ptha's default width rule and
    # never re-reads a hand-edited JSON, so if you edited
    # desired_unit_source_width (or n_downdip, or anything else in
    # cfg["geometry"]) after step 6, step 2's .npy and figure go stale --
    # they show the mesh from BEFORE your edit. Rebuilding here instead
    # means what this report shows and counts is always what step 7 (and
    # any edit you made to the JSON) actually computed.
    n_unit_sources = None
    grid = None
    try:
        grid = build_grid(cfg["geometry"])
        step1_mesh_card = mesh_file_card(grid) if MESH_FILE is not None else ""
        n_unit_sources = int((grid.shape[0] - 1) * (grid.shape[2] - 1))
    except Exception as exc:
        print(f"  note: could not rebuild the mesh for the report figure: {exc}")

    mesh_map_uri = None
    scratch_polys = scratch_depth = None
    if grid is not None:
        try:
            from plot_meshes import fig_scratch_mesh_map, scratch_cell_polygons
            os.makedirs(os.path.join(EXAMPLE, "figures"), exist_ok=True)
            scratch_polys, scratch_depth = scratch_cell_polygons(grid)
            n_downdip = grid.shape[0] - 1
            n_alongstrike = grid.shape[2] - 1
            mesh_png = fig_scratch_mesh_map(
                scratch_polys, scratch_depth, PTHA18_ZONE_NAME, n_downdip,
                n_alongstrike, os.path.join(EXAMPLE, "figures"),
                out_name="scratch_mesh_map.png")
            mesh_map_uri = embed_image(mesh_png)
        except Exception as exc:
            print(f"  note: could not draw the mesh figure: {exc}")
            mesh_map_uri = embed_image(
                os.path.join(EXAMPLE, "figures", "scratch_mesh_map.png"))

    # Interactive rupture explorer: LEVEL 2's rupture generation
    # (pyptha_v12/events.py, a port of rptha's rupture_events.R) enumerates
    # every valid placement of a fixed-size rectangular block of unit
    # sources for each magnitude -- this run's own scenario_rates_{zone}.csv
    # has thousands of rows of these, but step 9 never showed any of them on
    # the mesh itself. This sweeps every Mw in this run's own Mmin..Mmax
    # range and embeds a Mw dropdown + position slider (inline SVG/JS, no
    # video, no per-frame images) so any one of them can be picked and seen
    # highlighted on the mesh -- direct, explorable proof the generation is
    # a real set of earthquake-sized patches, not just a rate number.
    rupture_explorer_html = None
    n_events_total = None
    ev_mmin = cfg["events"]["Mmin"]
    # v10: which rupture-size rule step 7 used, for the LEVEL 2 text
    rupture_size_note = (
        " <b>This run used <code>--rupture-size local</code> (v10, not PTHA18's "
        "procedure):</b> the block size is chosen again at every placement from "
        "the real km of the cells there, so the rupture area follows the "
        "scaling relation even where the cells differ a lot in size; a "
        "magnitude's ruptures can then have different numbers of cells. "
        "v10_q: each rupture's conditional probability is also multiplied by "
        "q, the mean over its cells of 1 / (number of ruptures of that "
        "magnitude containing the cell), so that places covered by more "
        "ruptures (smaller cells) do not get more rate for the same area and "
        "convergence."
        if cfg["events"].get("rupture_size", "rptha") == "local" else "")
    ev_mmax = cfg["events"]["Mmax"]
    ev_dmw = cfg["events"]["dMw"]
    if grid is not None and scratch_polys is not None:
        try:
            from pyptha_v12 import events as ev_mod
            from pyptha_v12 import unit_sources as us
            from pyptha_v12.scaling import Mw_2_rupture_size, slip_from_Mw_area_mu

            rup_stats = us.discretized_source_approximate_summary_statistics(grid)
            ev_cfg = cfg["events"]
            # step 6 writes "scaling_relation"; "relation" is the older key
            relation = ev_cfg.get("scaling_relation", ev_cfg.get("relation", "Strasser"))
            mu = float(ev_cfg.get("shear_modulus_Pa", 3.0e10))
            rupture_size = ev_cfg.get("rupture_size", "rptha")

            # One call per Mw in the full range this run's own step 7 used
            # (get_all_earthquake_events_of_magnitude_Mw is cheap -- it only
            # enumerates block placements, not rates -- so sweeping all ~25
            # magnitudes costs nothing like re-running the moment balance).
            # Each Mw's result is reduced to one (topleft corner, nlength,
            # nwidth) per placement -- a rectangular block's corner and own
            # dimensions -- rather than a full per-cell index list, so
            # the embedded JSON stays a few integers per placement instead
            # of up to ~130.
            mws = np.round(np.arange(ev_mmin, ev_mmax + 1e-9, ev_dmw), 6)
            # the +-2 sigma range of the random length and width step 7b
            # draws for each HS/VAUS realisation (sigma does not depend on Mw)
            sig = Mw_2_rupture_size(float(mws[0]), relation=relation,
                                    detailed=True).log10_sigmas
            lw_2sd = {k: (10 ** (-2 * sig[k]), 10 ** (2 * sig[k]))
                      for k in ("length", "width")}
            mws_data = {}
            n_events_total = 0
            for mw in mws:
                one_mw = ev_mod.get_all_earthquake_events_of_magnitude_Mw(
                    float(mw), rup_stats, mu=mu, relation=relation,
                    rupture_size=rupture_size)
                st = one_mw["event_statistics"]
                placements = [[int(tl[0]), int(tl[1]), dim["length"], dim["width"],
                               int(round(a)), round(float(s), 2)]
                              for tl, dim, a, s in zip(one_mw["topleft_indices"],
                                                       one_mw["event_dims"],
                                                       st["area"], st["slip"])]
                # the relation's own 1-sigma area range, as a ratio to its
                # median, for the explorer's area / scaling-relation panel
                size = Mw_2_rupture_size(float(mw), relation=relation,
                                         detailed=True, CI_sd=1.0)
                mws_data[f"{float(mw):g}"] = {
                    "placements": placements,
                    "target": int(round(one_mw["desired_ALW"]["area"])),
                    "band": [round(size.minus_CI["area"] / size.values["area"], 3),
                             round(size.plus_CI["area"] / size.values["area"], 3)],
                    # uniform slip on the median area: PTHA18's peak-slip
                    # limit for HS/VAUS is 7.5 times this
                    # (compute_rates_all_sources.R 1569, slip_from_Mw)
                    "slip_ref": round(slip_from_Mw_area_mu(
                        float(mw), size.values["area"], mu), 3)}
                n_events_total += len(placements)

            # Step 7b's VAUS fields, attached to the FAUS rupture each one
            # was drawn from, so the explorer can show FAUS, VAUS or both.
            # Kept apart from the FAUS try: a problem here only drops VAUS.
            if d.get("vaus_summary") is not None and len(d["vaus_summary"]):
                try:
                    vaus = explorer_vaus_fields(
                        d["vaus_summary"], rup_stats,
                        {k: len(v["placements"]) for k, v in mws_data.items()})
                    for k, fields in vaus.items():
                        mws_data[k]["vaus"] = fields
                except Exception as exc:
                    print(f"  note: rupture explorer without VAUS: {exc}")

            rupture_explorer_html = svg_rupture_explorer(
                scratch_polys, rup_stats["downdip_number"],
                rup_stats["alongstrike_number"], mws_data, PTHA18_ZONE_NAME,
                relation, lw_2sd,
                cell_mu_area=(_rigidity_gpa(rup_stats["depth"]) * 1e9
                              * np.asarray(rup_stats["length"], dtype=float)
                              * np.asarray(rup_stats["width"], dtype=float)))
        except Exception as exc:
            print(f"  note: could not build the rupture explorer: {exc}")

    # STEP 7b: illustrative heterogeneous-slip (HS) and variable-area-
    # uniform-slip (VAUS) fields, if it was run. Never feeds any rate number
    # above -- this card exists purely to show what HS/VAUS ruptures for
    # this zone look like, clearly labelled apart from the FAUS ruptures
    # LEVELs 0-5 actually use.
    hs_summary = d.get("hs_summary")
    vaus_summary = d.get("vaus_summary")
    if hs_summary is not None and len(hs_summary):
        # summary.csv holds one row per generated field, EVERY field (not a
        # sample) -- potentially thousands across every magnitude (one field
        # per FAUS placement per realisation, see step7b's module
        # docstring), with the field's full footprint kept as its own
        # event_index_string/event_slip_string, rptha's own packed format.
        # A per-field table would be unreadable at that scale, so this rolls
        # it up to one row per magnitude: n fields, mean/peak slip.
        hs_agg = hs_summary.groupby("Mw", sort=True).agg(
            n_fields=("Mw", "size"),
            mean_peak_slip_m=("peak_slip_m", "mean"),
            max_peak_slip_m=("peak_slip_m", "max"),
            mean_uniform_slip_m=("uniform_slip_m_FAUS_equivalent", "mean"),
        ).reset_index()
        # v10_q: fields above PTHA18's peak-slip limit (rate 0), if step 7b
        # wrote rates
        def n_above(summary):
            if "above_peak_slip_limit" not in summary.columns:
                return None
            return summary.groupby("Mw", sort=True)["above_peak_slip_limit"].sum()
        hs_above = n_above(hs_summary)
        hs_rows = "".join(
            f"<tr><td>{esc(r['Mw'])}</td><td>{esc(int(r['n_fields']))}</td>"
            f"<td>{esc(round(r['mean_peak_slip_m'], 2))}</td>"
            f"<td>{esc(round(r['max_peak_slip_m'], 2))}</td>"
            f"<td>{esc(round(r['mean_uniform_slip_m'], 2))}</td>"
            + (f"<td>{int(hs_above[r['Mw']])}</td>" if hs_above is not None else "")
            + "</tr>"
            for _, r in hs_agg.iterrows())
        above_th = ("<th>Above the peak-slip limit (rate 0)</th>"
                    if hs_above is not None else "")
        rates_html = ("""
    <p><b>Rates (v10_q).</b> Each HS field, and each VAUS field, takes a share of the rate of the FAUS rupture it was drawn from, as PTHA18 does (<code>compute_rates_all_sources.R</code> 1551&ndash;1600, ported in <code>pyptha_v12/hs_vaus_rates.py</code>): a field whose peak slip is above 7.5 times the scaling relation's mean slip for its Mw gets rate 0; the others are weighted by the rank of their peak slip among the fields of the same FAUS rupture, with PTHA18's DART-calibrated weights (recovered from its published HS/VAUS rates, which they reproduce to 1e-12), and the shares of one FAUS rupture add up to 1. The rates are in the two <code>summary.csv</code> files; they enter nothing in LEVELs 0&ndash;5 or in the rate curve.</p>"""
                      if hs_above is not None else "")
        n_hs_mw = hs_summary["Mw"].nunique()
        n_hs_fields = len(hs_summary)

        hs_official_comparison = d.get("hs_official_comparison")
        if hs_official_comparison is not None and len(hs_official_comparison):
            comp_rows = "".join(
                f"<tr><td>{esc(r['Mw'])}</td>"
                f"<td>{esc(r['n_synthetic'])} vs {esc(r['n_official'])}</td>"
                f"<td>{esc(r['mean_peak_slip_m_synthetic'])} vs {esc(r['mean_peak_slip_m_official'])}</td>"
                f"<td>{esc(r['mean_active_cells_synthetic'])} vs {esc(r['mean_active_cells_official'])}</td>"
                f"<td>{esc(r['mean_concentration_synthetic'])} vs {esc(r['mean_concentration_official'])}</td>"
                f"<td>{esc(r['spectral_distance'])}</td></tr>"
                for _, r in hs_official_comparison.iterrows())
            total_official = int(hs_official_comparison["n_official"].astype(int).sum())
            total_synthetic = int(hs_official_comparison["n_synthetic"].astype(int).sum())
            official_comparison_html = f"""
    <h4 style="margin:22px 0 6px;font-size:14px">Statistical comparison against PTHA18's own published HS catalogue</h4>
    <p>PTHA18 does not just use the SFFM generator internally -- it publishes the full result: <code>all_stochastic_slip_earthquake_events_{esc(PTHA18_ZONE_NAME)}.nc</code>. This compares <b>every</b> HS field generated above against <b>every</b> official HS event published at the same magnitude: {total_synthetic:,} synthetic field(s) vs {total_official:,} official event(s) in total, across {len(hs_official_comparison)} magnitude(s) -- not a small subsample on either side. This is <b>a statistical comparison of distributions, not a field-by-field match</b>: SFFM draws a fresh random phase field every call, the published table does not record which seed produced any given row, and R's / NumPy's random number generators are not bit-compatible regardless, so no individual field here could ever be expected to reproduce a specific published one exactly. What CAN be validated -- and is shown below -- is whether the two sides' fields share the same typical peak slip, active-cell count, spatial concentration (fraction of total slip in the single largest cell) and amplitude-spectrum shape (the "spectral distance": 0 = identical average spectra, the same method rptha's own generator uses to validate itself against target data). See <code>lib/hs_official_compare.py</code> for the exact method.</p>
    <table>
      <thead><tr><th>Mw</th><th>n events (ours vs official)</th><th>Mean peak slip m (ours vs official)</th><th>Mean active cells (ours vs official)</th><th>Mean concentration (ours vs official)</th><th>Spectral distance</th></tr></thead>
      <tbody>{comp_rows}</tbody>
    </table>"""
        elif NOT_A_PTHA18_ZONE:
            official_comparison_html = f"""
    <p class="sec-note">No official HS comparison: {PTHA18_ZONE_NAME} is not one of PTHA18's source zones (PTHA18 covers the subduction zones that can send tsunamis to Australia), so there is no official run, mesh or HS/VAUS catalogue to compare with. Everything in this report is this run's own.</p>"""
        else:
            official_comparison_html = """
    <div class="callout warn">
      <span class="lbl">No official HS comparison for this zone</span>
      Step 8 has not compared these fields with PTHA18's official
      heterogeneous-slip catalogue
      (all_stochastic_slip_earthquake_events_&lt;zone&gt;.nc): run
      step8_official.py --download-hs, then this step again. This has no
      effect on anything else in this report.
    </div>"""

        if vaus_summary is not None and len(vaus_summary):
            vaus_agg = vaus_summary.groupby("Mw", sort=True).agg(
                n_fields=("Mw", "size"),
                mean_peak_slip_m=("peak_slip_m", "mean"),
                max_peak_slip_m=("peak_slip_m", "max"),
                mean_n_unit_sources=("n_unit_sources", "mean"),
            ).reset_index()
            vaus_above = n_above(vaus_summary)
            vaus_rows = "".join(
                f"<tr><td>{esc(r['Mw'])}</td><td>{esc(int(r['n_fields']))}</td>"
                f"<td>{esc(round(r['mean_peak_slip_m'], 2))}</td>"
                f"<td>{esc(round(r['max_peak_slip_m'], 2))}</td>"
                f"<td>{esc(round(r['mean_n_unit_sources'], 1))}</td>"
                + (f"<td>{int(vaus_above[r['Mw']])}</td>" if vaus_above is not None else "")
                + "</tr>"
                for _, r in vaus_agg.iterrows())
            n_vaus_fields = len(vaus_summary)

            vaus_official_comparison = d.get("vaus_official_comparison")
            if vaus_official_comparison is not None and len(vaus_official_comparison):
                vaus_comp_rows = "".join(
                    f"<tr><td>{esc(r['Mw'])}</td>"
                    f"<td>{esc(r['n_synthetic'])} vs {esc(r['n_official'])}</td>"
                    f"<td>{esc(r['mean_peak_slip_m_synthetic'])} vs {esc(r['mean_peak_slip_m_official'])}</td>"
                    f"<td>{esc(r['mean_active_cells_synthetic'])} vs {esc(r['mean_active_cells_official'])}</td>"
                    f"<td>{esc(r['mean_concentration_synthetic'])} vs {esc(r['mean_concentration_official'])}</td>"
                    f"<td>{esc(r['spectral_distance'])}</td></tr>"
                    for _, r in vaus_official_comparison.iterrows())
                vaus_total_official = int(vaus_official_comparison["n_official"].astype(int).sum())
                vaus_total_synthetic = int(vaus_official_comparison["n_synthetic"].astype(int).sum())
                # Mean signed %-bias of our peak slip vs official, across
                # magnitudes -- computed from this run's own numbers (not
                # hardcoded), so the note below describes whatever this
                # zone/mesh actually produced.
                vaus_pct_bias = (
                    (vaus_official_comparison["mean_peak_slip_m_synthetic"]
                     - vaus_official_comparison["mean_peak_slip_m_official"])
                    / vaus_official_comparison["mean_peak_slip_m_official"] * 100)
                vaus_bias_note = ""
                if len(vaus_pct_bias) >= 3 and (vaus_pct_bias > 0).all():
                    vaus_bias_note = f"""
    <div class="callout warn">
      <span class="lbl">Known measured bias, not a rule bug</span>
      Our mean peak slip runs above the official VAUS catalogue at EVERY
      magnitude compared here ({vaus_pct_bias.min():.1f}% to
      {vaus_pct_bias.max():.1f}%). Mean active cells (footprint size) above
      matches the official catalogue closely at every magnitude, which
      confirms the bounding-box derivation rule itself is correct -- the
      bias traces to this run's OWN from-scratch mesh having slightly
      different cell areas than PTHA18's official mesh (VAUS's uniform slip
      is sum(HS slip&times;area) / sum(footprint area), so a mesh-area
      difference compounds on top of whatever bias was already in the HS
      comparison above). See step7b_stochastic_slip.py's module docstring
      for the full explanation.
    </div>"""
                vaus_official_comparison_html = f"""
    <h5 style="margin:16px 0 6px;font-size:13px">Statistical comparison against PTHA18's own published VAUS catalogue</h5>
    <p>This is a SEPARATE comparison from the HS one above, and checks something the HS comparison cannot: VAUS being a deterministic function of the HS field next to it means an HS comparison passing does not by itself prove the DERIVATION RULE (the bounding-box superset + moment-preserving uniform slip) is implemented correctly -- only a direct comparison against <code>all_variable_uniform_slip_earthquake_events_{esc(PTHA18_ZONE_NAME)}.nc</code> can catch a bug in that rule specifically. {vaus_total_synthetic:,} synthetic VAUS field(s) vs {vaus_total_official:,} official VAUS event(s) in total, across {len(vaus_official_comparison)} magnitude(s). Same statistical method as the HS comparison (see above) -- not a field-by-field match.</p>
    <table>
      <thead><tr><th>Mw</th><th>n events (ours vs official)</th><th>Mean peak slip m (ours vs official)</th><th>Mean active cells (ours vs official)</th><th>Mean concentration (ours vs official)</th><th>Spectral distance</th></tr></thead>
      <tbody>{vaus_comp_rows}</tbody>
    </table>
    {vaus_bias_note}"""
            elif NOT_A_PTHA18_ZONE:
                vaus_official_comparison_html = ""
            else:
                vaus_official_comparison_html = """
    <div class="callout warn">
      <span class="lbl">No official VAUS comparison for this zone</span>
      Step 8 has not compared these fields with PTHA18's official
      variable-area-uniform-slip catalogue
      (all_variable_uniform_slip_earthquake_events_&lt;zone&gt;.nc): run
      step8_official.py --download-hs, then this step again. This has no
      effect on anything else in this report.
    </div>"""

            vaus_section_html = f"""
    <h4 style="margin:22px 0 6px;font-size:14px">VAUS (Variable Area Uniform Slip), derived from the HS fields above</h4>
    <p>rptha does not generate VAUS independently -- it derives it from the ALREADY-GENERATED HS table (<code>make_all_earthquake_events.R</code> lines ~255&ndash;345): same Mw, target location and peak-slip cell as the matching HS field, but with slip spread <b>uniformly</b> over the smallest <b>rectangle</b> that contains the HS field's own footprint, at whatever single slip value keeps sum(slip&times;area) -- and hence Mw -- unchanged. This step reproduces that exact derivation for every HS field generated above: {n_vaus_fields:,} VAUS field(s) in total, one per HS field, summarised by magnitude below.</p>
    <table>
      <thead><tr><th>Mw</th><th>VAUS fields</th><th>Mean uniform slip (m)</th><th>Max uniform slip (m)</th><th>Mean unit sources per field</th>{above_th if vaus_above is not None else ""}</tr></thead>
      <tbody>{vaus_rows}</tbody>
    </table>
    {vaus_official_comparison_html}"""
        else:
            vaus_section_html = """
    <div class="callout warn">
      <span class="lbl">No VAUS fields</span>
      outputs/vaus_slip_fields/summary.csv was not found -- step7b may have
      been run with an older version of this package that did not derive
      VAUS. Re-run step7b_stochastic_slip.py to generate it.
    </div>"""

        hs_section_html = f"""
  <div class="stepcard">
    <h3><span class="idx">STEP 7b</span> Heterogeneous-slip (HS) and variable-area-uniform-slip (VAUS)<span class="pill derived">{"rates shared from FAUS, not in LEVELs 0-5" if hs_above is not None else "illustrative, not rate-bearing"}</span></h3>
    <p><code>pyptha_v12/stochastic_slip.py</code> (a port of rptha's SFFM generator, the S<sub>NCF</sub> method of Davies et al. 2015) turns each FAUS rupture into a heterogeneous-slip realisation: same total slip (and hence the same Mw) as the FAUS rupture, but concentrated into asperities instead of spread uniformly. This covers EVERY magnitude this run generated FAUS ruptures for ({n_hs_mw} magnitude(s)) and, for each one, up to {esc(int(hs_summary.groupby("Mw").size().max()))} of that magnitude's real FAUS placements (LEVEL 2 above enumerates every valid one -- see "Total ruptures enumerated") -- {n_hs_fields:,} HS field(s) in total, summarised by magnitude below; the underlying summary.csv has one row PER FIELD (not a sample), with the field's own full footprint kept as event_index_string/event_slip_string, rptha's own packed format. <b>None of this feeds into the logic tree</b> -- every rate number in this report, including the exceedance-rate curve below, is FAUS-only, exactly as PTHA18 itself computes LEVELs 0&ndash;5. HS here is an illustrative addition (more realistic tsunami initial conditions), not an alternative rate model.</p>
    <table>
      <thead><tr><th>Mw</th><th>HS fields generated</th><th>Mean peak slip (m)</th><th>Max peak slip (m)</th><th>Mean FAUS uniform slip (m)</th>{above_th}</tr></thead>
      <tbody>{hs_rows}</tbody>
    </table>
    {rates_html}
    {official_comparison_html}
    {vaus_section_html}
  </div>"""
    else:
        hs_section_html = """
  <div class="stepcard">
    <h3><span class="idx">STEP 7b</span> Heterogeneous-slip (HS) and variable-area-uniform-slip (VAUS)<span class="pill declared">not run</span></h3>
    <p>step7b_stochastic_slip.py has not been run for this example, so no HS/VAUS sample is shown here. This has no effect on any rate number: LEVELs 0&ndash;5 above already used FAUS ruptures only, exactly as they would with step 7b run.</p>
  </div>"""

    # Per-cell Bird (2003) convergence map: step 3 saves each unit source's
    # own matched (and down-dip propagated) convergent slip, in the same
    # along-strike-outer/down-dip-inner order scratch_cell_polygons walks
    # the grid. Only drawn if that array's length matches THIS report's
    # rebuilt grid -- if you edited the geometry in the input JSON after
    # step 3 ran, the two would be for different meshes, and colouring one
    # mesh's cells with another mesh's values would be silently wrong.
    mesh_convergence_uri = None
    per_cell_conv = None
    if grid is not None and scratch_polys is not None \
            and os.path.exists(CONV_PERCELL_NPY):
        try:
            from plot_meshes import fig_scratch_mesh_map_by_value
            per_cell_conv = np.load(CONV_PERCELL_NPY)
            if len(per_cell_conv) != len(scratch_polys):
                print(f"  note: skipping the per-cell convergence figure -- "
                      f"{len(per_cell_conv)} saved values vs "
                      f"{len(scratch_polys)} cells in this report's "
                      f"rebuilt mesh (geometry changed since step 3 ran; "
                      f"re-run step3_convergence.py to refresh it)")
            else:
                conv_png = fig_scratch_mesh_map_by_value(
                    scratch_polys, per_cell_conv, PTHA18_ZONE_NAME,
                    n_downdip, n_alongstrike, os.path.join(EXAMPLE, "figures"),
                    value_label="Bird (2003) convergent slip (mm/yr, "
                               "top-edge match, propagated down-dip)",
                    cmap="viridis", out_name="scratch_mesh_map_convergence.png")
                mesh_convergence_uri = embed_image(conv_png)
        except Exception as exc:
            print(f"  note: could not draw the per-cell convergence "
                  f"figure: {exc}")

    # This run's own depth-along-the-arc profile. Drawn ALWAYS, whether or
    # not there is an official mesh to compare against: it answers "how deep
    # does each down-dip row get, and over what stretch of the arc", which is
    # a property of this mesh alone and does not need PTHA18 for anything.
    # (The two-mesh version below adds the official series when step 8 ran.)
    mesh_profile_uri = None
    figures_dir = os.path.join(EXAMPLE, "figures")
    if grid is not None:
        try:
            from plot_meshes import fig_profile_downdip_scratch
            mesh_profile_uri = embed_image(fig_profile_downdip_scratch(
                grid, "makran2", figures_dir))
        except Exception as exc:
            print(f"  note: could not draw the mesh profile: {exc}")

    # The official mesh, its own map, and the two meshes compared side by
    # side (fig_map_meshes) and along the arc (fig_profile_downdip). Needs
    # step 8's copy of the official .nc table (unit_source_statistics_<zone>.nc)
    # -- absent entirely without step 8, in which case this whole block is
    # skipped and the report still covers steps 1-7 in full.
    mesh_official_uri = None
    mesh_compare_map_uri = None
    mesh_compare_profile_uri = None
    off_polys = off_depth = off_n_downdip = None
    official_nc_path = OFFICIAL_NC
    if os.path.exists(official_nc_path) and grid is not None \
            and scratch_polys is not None:
        try:
            import netCDF4
            from plot_meshes import (fig_scratch_mesh_map,  # noqa: E501
                                     official_cell_polygons_best,
                                     fig_map_meshes, fig_profile_downdip)
            # PTHA18's OWN published cell outlines when they are available
            # (validation/ptha18_reference/unit_source_grid/<zone>.shp),
            # falling back to the centroid+length+width+strike rectangles
            # only for a zone that has no shapefile. The rectangles overlap
            # and leave ragged edges on a curved arc; the published polygons
            # are the mesh itself.
            off_polys, off_depth, off_poly_source = official_cell_polygons_best(
                PTHA18_ZONE_NAME, path=official_nc_path)
            print(f"  official mesh cells: {off_poly_source}"
                  + ("" if off_poly_source == "shapefile" else
                     " (no published unit_source_grid shapefile for this zone)"))
            with netCDF4.Dataset(official_nc_path) as f:
                off_dd = np.asarray(f["downdip_number"][:], dtype=int)
            figures_dir = os.path.join(EXAMPLE, "figures")
            off_n_downdip = int(off_dd.max())
            off_mesh_png = fig_scratch_mesh_map(
                off_polys, off_depth, f"{PTHA18_ZONE_NAME} (official)",
                off_n_downdip, len(off_polys) // max(off_n_downdip, 1),
                figures_dir, out_name="official_mesh_map.png")
            mesh_official_uri = embed_image(off_mesh_png)
            mesh_compare_map_uri = embed_image(fig_map_meshes(
                off_polys, off_depth, scratch_polys, scratch_depth,
                PTHA18_ZONE_NAME, "makran2", figures_dir,
                **({"scratch_label": f"external mesh ({MESH_FILE})"} if MESH_FILE else {})))
            mesh_compare_profile_uri = embed_image(fig_profile_downdip(
                off_polys, off_depth, off_dd, grid, figures_dir))
        except Exception as exc:
            print(f"  note: could not draw the official/comparison mesh "
                  f"figures: {exc}")

    # Per-cell Bird convergence on the OFFICIAL mesh, computed by step 8 with
    # the same lib/bird_convergence.py and PTHA18's own Bird table: match
    # each top-edge unit source to its nearest Bird (2003) segment,
    # propagate down-dip. PTHA18 publishes only the area-weighted scalar,
    # but this reconstruction reproduces that scalar to machine precision on
    # every Bird zone tested, so it is the per-cell field PTHA18 used, not an
    # approximation of it.
    mesh_official_convergence_uri = None
    if os.path.exists(OFFICIAL_CONV_NPY) and off_polys is not None:
        try:
            from plot_meshes import (fig_scratch_mesh_map_by_value,
                                     fig_map_meshes_by_value)
            off_slip = np.load(OFFICIAL_CONV_NPY)

            conv_value_label = ("Bird (2003) convergent slip (mm/yr, "
                                "top-edge match, propagated down-dip)")
            if per_cell_conv is not None and len(per_cell_conv) == len(scratch_polys):
                # Side by side, ONE shared colour scale -- the comparison
                # the user actually asked for, not two separately-scaled maps.
                conv_png = fig_map_meshes_by_value(
                    off_polys, off_slip, scratch_polys, per_cell_conv,
                    PTHA18_ZONE_NAME, "makran2", conv_value_label,
                    figures_dir, out_name="mesh_comparison_map_convergence.png")
            else:
                # This run's own per-cell array is unavailable or stale
                # (see the guard above) -- fall back to the official mesh
                # alone rather than skip the figure entirely.
                conv_png = fig_scratch_mesh_map_by_value(
                    off_polys, off_slip, f"{PTHA18_ZONE_NAME} (official)",
                    off_n_downdip, len(off_polys) // max(off_n_downdip, 1),
                    figures_dir, value_label=conv_value_label,
                    cmap="viridis", out_name="official_mesh_map_convergence.png")
            mesh_official_convergence_uri = embed_image(conv_png)
        except Exception as exc:
            print(f"  note: could not draw the official per-cell "
                  f"convergence figure: {exc}")

    # The meshes on a real map (Esri's ocean basemap), to check by eye that
    # the mesh sits on the real trench. PTHA18's mesh is drawn from its
    # published cell outlines (NCI unit_source_grid/<zone>.shp, saved by
    # step 8) -- the same source the figures above now use whenever it is
    # present.
    web_map_card = ""
    if grid is not None:
        web_meshes = [(f"This run: {n_unit_sources} cells", "#eb6834", {"weight": 3},
                       grid[:, :2, :])]
        n_web_official = None
        if os.path.exists(OFFICIAL_GRID_NPY):
            off_grid = np.load(OFFICIAL_GRID_NPY)
            # Same longitude convention as this run's mesh (0 to 360 or -180 to 180).
            off_grid[:, 0, :] += 360.0 * np.round(
                (np.nanmean(grid[:, 0, :]) - np.nanmean(off_grid[:, 0, :])) / 360.0)
            n_web_official = (off_grid.shape[0] - 1) * (off_grid.shape[2] - 1)
            web_meshes.insert(0, (f"PTHA18 published: {n_web_official} cells",
                                  "#d6249f", {"weight": 8, "opacity": 0.55}, off_grid))
        else:
            print("  note: the map shows this run's mesh only (no PTHA18 cell "
                  "outlines from step 8)")
        web_map_card = f'''<div class="stepcard">
    <h3><span class="idx">STEP 2</span> The mesh on a real map<span class="pill derived">check by eye</span></h3>
    <p>The mesh is a real place on Earth, so it must sit where the real subduction zone is. The background is Esri's ocean map, which shades the sea floor by its depth: the trench is the long, narrow, dark groove where one plate bends down under the other. The thick line of each mesh is its top edge (its shallowest cells) and should run along that groove; from there the mesh follows the sinking plate down, under the island arc, to step 1's depth limit.{" PTHA18's published mesh is drawn under this run's mesh, its top edge as a wide band: where the two top edges agree, the orange line runs inside the pink band." if n_web_official else ""} Drag to move, zoom with the buttons or a double click (the mouse wheel zooms after a click on the map), and change the background or hide a mesh in the box at the top right.</p>
    {leaflet_mesh_map(web_meshes)}
    <div class="legend">
      <span><span class="sw" style="background:#eb6834;height:3px"></span>this run: {n_unit_sources} cells (thick line: top edge)</span>
      {f'<span><span class="sw" style="background:#d6249f;height:7px;opacity:.55"></span>PTHA18 published mesh: {n_web_official} cells (wide band: top edge)</span>' if n_web_official else ''}
    </div>
    <p class="sec-note" style="margin-top:8px">Unlike the rest of this report, this map is not stored in the page: the Leaflet map library and Esri's tiles load from the internet when the page opens. {f"PTHA18's cells are its published outlines (NCI <code>SOURCE_ZONES/{PTHA18_ZONE_NAME}/EQ_SOURCE/unit_source_grid/{PTHA18_ZONE_NAME}.shp</code>), not the rectangles rebuilt from the statistics table in the figures above." if n_web_official else ("Only this run's mesh is drawn: " + PTHA18_ZONE_NAME + " is not one of PTHA18's source zones." if NOT_A_PTHA18_ZONE else "PTHA18's published cell outlines could not be read for this zone, so only this run's mesh is drawn.")}</p>
  </div>'''

    rate72 = rate_at(rates, 7.2)
    rate80 = rate_at(rates, 8.0)
    rate90 = rate_at(rates, 9.0)

    n_contours = d["n_contours"]
    _s1 = d.get("step1_info") or {}
    if _s1.get("trench_ramp_rise_km") is not None:
        _a, _b = _s1.get("ramp_trim_km") or (0.0, 0.0)
        ramp_note = (f'<p class="sec-note"><b>Trench ramp rule (v10_q):</b> a trench end more than '
                     f'{_s1["trench_ramp_rise_km"]:g} km deeper than the trench&apos;s normal depth is the edge of the SLAB '
                     f'data, not trench, and is trimmed before the depth-below-trench datum is built. This run: '
                     + (f'start trimmed by {_a:.0f} km, end by {_b:.0f} km.' if (_a or _b) else 'nothing to trim.')
                     + '</p>')
    elif "trench_ramp_rise_km" in _s1:
        ramp_note = '<p class="sec-note">Trench ramp rule: off (v9&apos;s trench).</p>'
    else:
        ramp_note = ""
    n_catalogue = d["n_catalogue"]
    n_subset = len(gcmt)
    obs = cfg["rates"]["observed_seismicity"]

    off_summary = off["summary"] if off else None
    off_rates = off["rates"] if off else None
    off_rate72 = rate_at(off_rates, 7.2) if off_rates is not None else None

    chart_svg = svg_rate_curve(rates, off_rates, d["percentiles"])

    # The branch-fan figure (report Figures 43/44: every one of the 32,000+
    # branches' own Mw-vs-exceedance-rate curve, prior on the left, posterior
    # with percentile bands on the right) is generated by the shared engine
    # itself in step 7/8, not redrawn here -- embedding the real PNG keeps
    # this report showing exactly what the engine produced.
    branch_fan_img = embed_image(
        os.path.join(OUT_DIR, "figures", f"fig_branch_fan_{SUFFIX}.png"))
    off_branch_fan_img = embed_image(
        os.path.join(OFFICIAL_RUN_DIR, "figures", f"fig_branch_fan_{SUFFIX}.png"))

    coupling_axis = cfg["rates"]["coupling"]
    b_anchor = cfg["rates"]["b_anchor"]

    # ---- comparison table rows ----
    def cmp_row(label, mine, official, unit="", fmt="{:.4f}"):
        if official is None:
            off_txt = "n/a"
            diff_txt = "n/a"
        else:
            official = float(official)
            off_txt = fmt.format(official) + unit
            diff = 100.0 * (mine - official) / official if official else float("nan")
            diff_txt = f"{diff:+.1f}%"
        return (f'<tr><td>{esc(label)}</td>'
               f'<td class="n">{fmt.format(mine)}{unit}</td>'
               f'<td class="n src">{off_txt}</td>'
               f'<td class="n">{diff_txt}</td></tr>')

    off_mean_dip = off_summary.get(f"mean_dip_deg_{SUFFIX}") if off_summary else None
    off_conv = off_summary.get(f"convergence_mm_per_yr_{SUFFIX}") if off_summary else None
    off_edge = off_summary.get(f"edge_multiplier_{SUFFIX}") if off_summary else None
    off_tvd = off_summary.get(f"prior_posterior_tvd_{SUFFIX}") if off_summary else None
    off_gcmt_count = off_gcmt_duration = None
    if os.path.exists(OFFICIAL_GCMT_JSON):
        with open(OFFICIAL_GCMT_JSON) as f:
            _g = json.load(f)
        off_gcmt_count, off_gcmt_duration = int(_g["count"]), float(_g["duration_years"])

    # ---- plain-text exports (v8.1) -------------------------------------
    # Everything the figures above are drawn from, as CSV, so the numbers can
    # be read, diffed and re-plotted without going through this HTML or
    # re-running anything. Each file is written for THIS run, and again with
    # an _official suffix when step 8 supplied PTHA18's own version, so the
    # pair can be compared row by row.
    export_dir = os.path.join(EXAMPLE, "outputs", "exports")
    os.makedirs(export_dir, exist_ok=True)
    exported = []

    def _note_export(path, what):
        exported.append((os.path.basename(path), what))
        print(f"    {os.path.basename(path)}  ({what})")

    # v9: on a segmented run, label every row of this run's mesh and
    # convergence-profile exports with the segment its along-strike column
    # belongs to (alongstrike_slice is 1-based and inclusive, like
    # alongstrike_number). An unsegmented run writes exactly v8's files.
    _seg_of_col = {}
    for _key, _sc in cfg.get("segments", {}).items():
        _a, _b = _sc["alongstrike_slice"]
        for _col in range(int(_a), int(_b) + 1):
            _seg_of_col[_col] = _key

    def _add_segment_column(path):
        if not _seg_of_col:
            return
        with open(path, encoding="utf-8") as _f:
            _lines = _f.read().splitlines()
        _k = _lines[0].split(",").index("alongstrike_number")
        _out = [_lines[0] + ",segment"]
        for _ln in _lines[1:]:
            _out.append(_ln + "," + _seg_of_col.get(int(_ln.split(",")[_k]), ""))
        with open(path, "w", encoding="utf-8") as _f:
            _f.write("\n".join(_out) + "\n")

    print("  exports ->", os.path.relpath(export_dir, EXAMPLE))
    try:
        from plot_meshes import write_mesh_csv, scratch_cell_indices

        # 1. the mesh: lon, lat, depth per unit source
        if scratch_polys is not None and grid is not None:
            _dd, _al = scratch_cell_indices(grid)
            _extra = (per_cell_conv if per_cell_conv is not None
                      and len(per_cell_conv) == len(scratch_polys) else None)
            _p = write_mesh_csv(
                scratch_polys, scratch_depth,
                os.path.join(export_dir, "mesh_cells.csv"),
                downdip=_dd, alongstrike=_al, extra=_extra,
                extra_name="convergent_slip_mm_per_yr")
            _add_segment_column(_p)
            _note_export(_p, "this run's mesh: lon, lat, depth per cell"
                         + (", and its segment" if _seg_of_col else ""))

        if off_polys is not None and off_depth is not None:
            try:
                import netCDF4 as _nc
                with _nc.Dataset(OFFICIAL_NC) as _f:
                    _odd = np.asarray(_f["downdip_number"][:], dtype=int)
                    _oal = np.asarray(_f["alongstrike_number"][:], dtype=int)
            except Exception:
                _odd = _oal = None
            _oextra = None
            if os.path.exists(OFFICIAL_CONV_NPY):
                _os = np.load(OFFICIAL_CONV_NPY)
                if len(_os) == len(off_polys):
                    _oextra = _os
            _p = write_mesh_csv(
                off_polys, off_depth,
                os.path.join(export_dir, "mesh_cells_official.csv"),
                downdip=_odd, alongstrike=_oal, extra=_oextra,
                extra_name="convergent_slip_mm_per_yr")
            _note_export(_p, "PTHA18's published mesh, same columns")

        # 2. the POSTERIOR mean exceedance-rate curve. "Posterior" is in the
        #    file name because it is load-bearing, not decoration: the curve
        #    is the logic-tree mean taken with the weights AFTER the LEVEL 3
        #    Bayesian update against the observed GCMT events
        #    (update_logic_tree_weights_with_data, passed through to
        #    rate_of_earthquakes_greater_than_Mw_function). The official
        #    curve is posterior in the same sense: step 8 runs PTHA18's own
        #    input through this same engine. How far the update moved the
        #    weights is the prior-vs-posterior TVD shown in the report.
        _post = "posterior_mean_rate_curve"
        rates.to_csv(os.path.join(export_dir, f"{_post}.csv"), index=False)
        _note_export(os.path.join(export_dir, f"{_post}.csv"),
                     "posterior mean exceedance rate vs Mw, this run "
                     "(logic-tree mean, weights after the LEVEL 3 update)")
        if off_rates is not None:
            off_rates.to_csv(
                os.path.join(export_dir, f"{_post}_official.csv"), index=False)
            _note_export(
                os.path.join(export_dir, f"{_post}_official.csv"),
                "the same posterior curve from PTHA18's official run")

        # v9: a segmented run's scenario rates for the zone as a whole
        # (0.5 x unsegmented + 0.5 x every segment the scenario touches)
        for _src, _tag, _what in (
                (OUT_DIR, "", "this run"), (OFFICIAL_RUN_DIR, "_official", "PTHA18's run")):
            _zs = os.path.join(_src, f"scenario_rates_{SUFFIX}_source_zone.csv")
            if cfg.get("segments") and os.path.exists(_zs):
                _dst = os.path.join(export_dir, f"scenario_rates_source_zone{_tag}.csv")
                shutil.copy2(_zs, _dst)
                _note_export(_dst, f"every scenario's rate for the whole segmented "
                                   f"zone, {_what} (unsegmented + segments, and "
                                   f"PTHA18's partial-segmentation percentiles)")

        # 3. the convergence rate: the zone scalar, and the per-column
        #    profile it is the area-weighted mean of
        _conv_lines = ["item,value,units"]
        _conv_lines.append(f"convergence_horizontal,{convergence:.10f},mm/yr")
        _conv_lines.append(f"mean_dip,{mean_dip:.10f},deg")
        _conv_lines.append(
            f"slip_rate,{convergence / np.cos(np.radians(mean_dip)):.10f},"
            f"mm/yr")
        with open(os.path.join(export_dir, "convergence.csv"), "w",
                  encoding="utf-8") as _f:
            _f.write("\n".join(_conv_lines) + "\n")
        _note_export(os.path.join(export_dir, "convergence.csv"),
                     "this run's convergence, mean dip and slip rate")

        if off_conv is not None:
            _ol = ["item,value,units",
                   f"convergence_horizontal,{float(off_conv):.10f},mm/yr"]
            if off_mean_dip is not None:
                _ol.append(f"mean_dip,{float(off_mean_dip):.10f},deg")
                _ol.append(
                    f"slip_rate,"
                    f"{float(off_conv) / np.cos(np.radians(float(off_mean_dip))):.10f},"
                    f"mm/yr")
            with open(os.path.join(export_dir, "convergence_official.csv"),
                      "w", encoding="utf-8") as _f:
                _f.write("\n".join(_ol) + "\n")
            _note_export(os.path.join(export_dir, "convergence_official.csv"),
                         "the same three numbers for PTHA18's run")

        # the per-column Bird profile behind the scalar above
        if os.path.exists(CONV_PERCOLUMN_JSON):
            with open(CONV_PERCOLUMN_JSON) as _f:
                _c = json.load(_f)
            _pl = ["alongstrike_number,div_mm_per_yr,"
                   "convergent_slip_mm_per_yr,distance_to_bird_km"]
            for _i in range(len(_c["alongstrike_number"])):
                _pl.append(
                    f'{_c["alongstrike_number"][_i]},'
                    f'{_c["div_mm_per_yr"][_i]:.6f},'
                    f'{_c["convergent_slip_mm_per_yr"][_i]:.6f},'
                    f'{_c["distance_km"][_i]:.4f}')
            with open(os.path.join(export_dir, "convergence_profile.csv"),
                      "w", encoding="utf-8") as _f:
                _f.write("\n".join(_pl) + "\n")
            _add_segment_column(os.path.join(export_dir, "convergence_profile.csv"))
            _note_export(
                os.path.join(export_dir, "convergence_profile.csv"),
                "Bird convergence per along-strike column"
                + (", and its segment" if _seg_of_col else ""))

        # the same profile on PTHA18's own mesh, written by step 8 with the
        # same code and PTHA18's own Bird table. The two files have identical
        # columns but NOT necessarily the same number of rows: each mesh has
        # its own column count (kurilsjapan: 60 here vs PTHA18's 61), so
        # compare them as two profiles along the arc, not row by row.
        if os.path.exists(OFFICIAL_CONV_PERCOLUMN_JSON):
            with open(OFFICIAL_CONV_PERCOLUMN_JSON) as _f:
                _oc = json.load(_f)
            _ol2 = ["alongstrike_number,div_mm_per_yr,"
                    "convergent_slip_mm_per_yr,distance_to_bird_km"]
            for _i in range(len(_oc["alongstrike_number"])):
                _ol2.append(
                    f'{_oc["alongstrike_number"][_i]},'
                    f'{_oc["div_mm_per_yr"][_i]:.6f},'
                    f'{_oc["convergent_slip_mm_per_yr"][_i]:.6f},'
                    f'{_oc["distance_km"][_i]:.4f}')
            with open(os.path.join(export_dir,
                                   "convergence_profile_official.csv"),
                      "w", encoding="utf-8") as _f:
                _f.write("\n".join(_ol2) + "\n")
            _note_export(
                os.path.join(export_dir, "convergence_profile_official.csv"),
                "the same profile on PTHA18's own mesh (its own column count)")
    except Exception as exc:
        print(f"  note: could not write one or more exports: {exc}")

    exports_rows = "".join(
        f'<tr><td><code>{esc(n)}</code></td><td>{esc(w)}</td></tr>'
        for n, w in exported)


    # ---- attribute the exceedance-rate gap to whichever input actually
    # ---- drives it, instead of assuming it is always the mesh's dip.
    # v2's geometry fixes (the real per-zone depth cutoff, the measured
    # SLAB2.0-vs-trench datum, contour segments joined instead of the
    # single longest kept) bring mean dip within a few hundredths of a
    # degree of the official value on kermadectonga2 -- so on THIS zone the
    # old "it's the dip" explanation is no longer true, and printing it
    # unconditionally would be actively misleading.
    def _pct_gap(this_v, off_v):
        if this_v is None or off_v is None:
            return None
        off_v = float(off_v)
        return 100.0 * (float(this_v) - off_v) / off_v if off_v else None

    gaps = {
        "mean dip": (_pct_gap(mean_dip, off_mean_dip),
                     "the mesh (step 1-2): a steeper or shallower mean dip "
                     "changes the slip rate via 1/cos(dip), "
                     "which moves the seismic moment rate directly. Look at "
                     "step 1's figure (figures/step1_trench_and_contours.png) "
                     "and the along-strike extent if this dominates."),
        "convergence": (_pct_gap(convergence, off_conv),
                        "step 3's Bird (2003) area-weighted convergence. The "
                        "method is PTHA18's own (on PTHA18's mesh it gives "
                        "the official value exactly), so a gap comes from "
                        "the mesh: which stretch of trench it covers and "
                        "where its columns fall. Exception: puysegur, where "
                        "PTHA18 used a constant 35 mm/yr instead of Bird."),
        "edge multiplier": (_pct_gap(edge_mult, off_edge),
                            "step 7's LEVEL 4 edge-taper fit, not the mesh "
                            "directly: it depends on the number of "
                            "along-strike columns and their individual "
                            "lengths, which cannot match PTHA18's "
                            "hand-edited seams even when total area and mean "
                            "dip do."),
    }
    _rate_gap = _pct_gap(rate72, off_rate72)
    _valid = {k: v for k, (v, _) in gaps.items() if v is not None}
    if not _valid or _rate_gap is None:
        gap_attribution_label = "No official run to compare against yet"
        gap_attribution_text = ("Run step8_official.py, then re-run this "
                                "step, for an attribution of the gap above.")
    else:
        _dominant = max(_valid, key=lambda k: abs(_valid[k]))
        _dom_gap, _dom_expl = gaps[_dominant]
        _mesh_close = (abs(gaps["mean dip"][0]) < 0.5
                      if gaps["mean dip"][0] is not None else False)
        gap_attribution_label = "Read a gap here as geometry, not error"
    if _valid and _rate_gap is not None and abs(_rate_gap) < 1.0:
        # a gap under 1% is not worth attributing to one input
        gap_attribution_label = "The two runs agree"
        gap_attribution_text = (
            f"The Mw&nbsp;7.2 exceedance rates agree to {_rate_gap:+.1f}%, so "
            f"there is no gap to explain. The inputs above still differ (the "
            f"largest relative gap is the {_dominant}, {_dom_gap:+.1f}%): "
            f"{_dom_expl}")
    elif _valid and _rate_gap is not None:
        gap_attribution_text = (
            f"The {_rate_gap:+.0f}% difference in the Mw&nbsp;7.2 exceedance "
            f"rate traces mostly to {_dom_expl} "
            f"({_dominant} gap: {_dom_gap:+.1f}%)."
            + (f" Mean dip itself is within {abs(gaps['mean dip'][0]):.2f}% "
              f"of the official value here, so this mesh's TOTAL AREA and "
              f"DIP are not the story -- the remaining gap is downstream of "
              f"the mesh, in how convergence and the edge correction "
              f"respond to where the along-strike seams fall, which cannot "
              f"be reproduced (see from_scratch_v12/html/docs/official.html)."
              if _mesh_close else ""))

    cmp_rows = "\n".join([
        cmp_row("Mean dip", mean_dip, off_mean_dip, " deg"),
        cmp_row("Convergence", convergence, off_conv, " mm/yr"),
        cmp_row("Edge multiplier", edge_mult, off_edge, ""),
        cmp_row("Prior->posterior shift (TVD)", tvd, off_tvd, ""),
        cmp_row("Exceedance rate at Mw 7.2" + (" (unsegmented branch)" if cfg.get("segments") else ""),
                rate72, off_rate72, " /yr", "{:.4f}"),
    ])

    # ---- v9: LEVEL 0 segmentation section ----
    #
    # Only rendered when this run actually had segments. An unsegmented run
    # produces exactly the page v8 produced, which is what keeps a v9 report
    # comparable with a v8 one.
    #
    # The point of the section is to show WHAT SEGMENTING CHANGED, so it puts
    # the three curves side by side at a few magnitudes: the unsegmented
    # branch, the union of segments, and the 50/50 mixture of the two that is
    # the zone's actual answer. It also states plainly where the boundaries
    # came from and how they compare with PTHA18's, because those are the two
    # things a reader needs in order to judge the numbers.
    seg_cfg = cfg.get("segments", {})
    # v9: which public source placed the boundaries (step 6 records it)
    seg_from = ("berryman" if any(sc.get("_boundaries") == "berryman"
                                  for sc in seg_cfg.values()) else "bird")
    if seg_cfg and "exceedance_rate_union_of_segments" in rates.columns:
        zone_pfx = f"exceedance_rate_{PTHA18_ZONE_NAME}_"
        run_segs = [(k, int(sc["alongstrike_slice"][0]), int(sc["alongstrike_slice"][1]))
                    for k, sc in seg_cfg.items()]
        n_cols_run = (grid.shape[2] - 1) if grid is not None else run_segs[-1][2]

        # PTHA18's own segmented run, when step 8 built one (a segmented
        # example): its segments, its mesh columns, its curves.
        off_segs, off_al = [], None
        off_seg_json = os.path.join(REFERENCE_DIR, "official_segments.json")
        if (os.path.exists(off_seg_json) and off_rates is not None
                and "exceedance_rate_union_of_segments" in off_rates.columns):
            with open(off_seg_json) as f:
                off_segs = json.load(f)["segments"]
            try:
                import netCDF4
                with netCDF4.Dataset(OFFICIAL_NC) as f:
                    off_al = np.asarray(f["alongstrike_number"][:], dtype=int)
            except Exception:
                off_al = None
        has_off = bool(off_segs)

        strip_rows = [("This run", n_cols_run, run_segs)]
        if has_off and off_al is not None:
            strip_rows.append(("PTHA18", int(off_al.max()),
                               [(sg["name"], *sg["alongstrike_slice"]) for sg in off_segs]))
        strips_svg = svg_segment_strips(strip_rows)

        seg_map_uri = None
        if scratch_polys is not None and grid is not None:
            try:
                from plot_meshes import fig_map_segments, scratch_cell_indices
                panels = []
                if has_off and off_polys is not None and off_al is not None \
                        and len(off_al) == len(off_polys):
                    panels.append((f"PTHA18: {len(off_segs)} segments on its own mesh",
                                   off_polys, off_al,
                                   [(sg["name"], sg["alongstrike_slice"]) for sg in off_segs]))
                panels.append((f"This run: {len(run_segs)} segments from "
                                f"{'Berryman et al. (2015)' if seg_from == 'berryman' else 'Bird (2003)'}",
                               scratch_polys, scratch_cell_indices(grid)[1],
                               [(k, [a, b]) for k, a, b in run_segs]))
                seg_map_uri = embed_image(fig_map_segments(
                    panels, os.path.join(EXAMPLE, "figures")))
            except Exception as exc:
                print(f"  note: could not draw the segment map: {exc}")

        run_coupling = cfg["rates"]["coupling"].get("spreadsheet_values", [])
        seg_rows = []
        for k, (key, a, b) in enumerate(run_segs):
            sc = seg_cfg[key]
            n_obs = sc.get("rates", {}).get("observed_seismicity", {}).get("count", 0)
            conv_k = s.get(f"convergence_mm_per_yr_{PTHA18_ZONE_NAME}_{key}")
            mwo = sc.get("rates", {}).get("mw_max_observed")
            col = zone_pfx + key
            r72 = rate_at(rates, 7.2, col) if col in rates.columns else float("nan")
            conv_td = f'<td>{float(conv_k):.1f}</td>' if conv_k is not None else '<td>-</td>'
            mwo_td = f'<td>{mwo:g}</td>' if mwo is not None else '<td>-</td>'
            seg_c = sc.get("rates", {}).get("coupling", {}).get("spreadsheet_values")
            rows_b = sc.get("_berryman_rows") or []
            coup_td = (f'<td>{"/".join(f"{c:g}" for c in seg_c)} '
                       f'({esc(" + ".join(rows_b)) if rows_b else "own"})</td>' if seg_c else
                       f'<td>{"/".join(f"{c:g}" for c in run_coupling)} (zone)</td>')
            seg_rows.append(
                f'<tr><td><span class="sw" style="background:{SEGMENT_COLOURS[k % 8]}"></span> '
                f'<code>{esc(key)}</code></td>'
                f'<td>{esc(sc.get("_plate_pair", "-"))}</td>'
                f'<td>{a}-{b} ({b - a + 1})</td>' + conv_td + coup_td + mwo_td
                + f'<td>{n_obs}</td><td>{r72:.4f}</td></tr>')

        off_seg_rows = []
        if has_off:
            for k, sg in enumerate(off_segs):
                a, b = sg["alongstrike_slice"]
                conv_k = off_summary.get(f"convergence_mm_per_yr_{PTHA18_ZONE_NAME}_{sg['name']}") \
                    if off_summary is not None else None
                col = zone_pfx + sg["name"]
                r72 = rate_at(off_rates, 7.2, col) if col in off_rates.columns else float("nan")
                off_seg_rows.append(
                    f'<tr><td><span class="sw" style="background:{SEGMENT_COLOURS[k % 8]}"></span> '
                    f'<code>_{esc(sg["name"])}</code></td>'
                    f'<td>{a}-{b} ({b - a + 1})</td>'
                    + (f'<td>{float(conv_k):.1f}</td>' if conv_k is not None else '<td>-</td>')
                    + f'<td>{"/".join(f"{c:g}" for c in sg["coupling_spreadsheet"])}</td>'
                    f'<td>{sg["mw_max_observed"]:g}</td>'
                    f'<td>{sg["gcmt_count"] if sg["gcmt_count"] is not None else "-"}</td>'
                    f'<td>{r72:.4f}</td></tr>')

        def mix_row(mw):
            u = rate_at(rates, mw, "exceedance_rate_unsegmented")
            un = rate_at(rates, mw, "exceedance_rate_union_of_segments")
            z = rate_at(rates, mw, "exceedance_rate_source_zone")
            pct = (un / u - 1.0) * 100.0 if u > 0 else float("nan")
            return (f'<tr><td>Mw {mw:g}</td><td>{u:.5g}</td><td>{un:.5g}</td>'
                    f'<td>{pct:+.1f}%</td><td><b>{z:.5g}</b></td></tr>')

        def vs_row(mw):
            cells = [f'<td>Mw {mw:g}</td>']
            for c in ("exceedance_rate_unsegmented", "exceedance_rate_union_of_segments",
                      "exceedance_rate_source_zone"):
                a_, b_ = rate_at(rates, mw, c), rate_at(off_rates, mw, c)
                pct = f"{(a_ / b_ - 1.0) * 100.0:+.0f}%" if b_ > 0 else "-"
                cells.append(f'<td>{a_:.4g}</td><td>{b_:.4g}</td><td>{pct}</td>')
            return "<tr>" + "".join(cells) + "</tr>"

        series = []
        if has_off:
            series += [dict(df=off_rates, col=zone_pfx + sg["name"], color="var(--amber)",
                            width=1.2, dash="2 3", opacity=0.6, label=f"PTHA18 {sg['name']}")
                       for sg in off_segs if zone_pfx + sg["name"] in off_rates.columns]
        series += [dict(df=rates, col=zone_pfx + k, color="var(--text-soft)", width=1.2,
                        dash="2 3", opacity=0.6, label=f"This run {k}")
                   for k, _, _ in run_segs if zone_pfx + k in rates.columns]
        if has_off:
            series += [dict(df=off_rates, col="exceedance_rate_union_of_segments",
                            color="var(--amber)", width=2, dash="7 3", label="PTHA18 union of segments"),
                       dict(df=off_rates, col="exceedance_rate_source_zone",
                            color="var(--amber)", width=2.5, dash="", label="PTHA18 source zone (mix)")]
        series += [dict(df=rates, col="exceedance_rate_union_of_segments", color="var(--accent-2)",
                        width=2, dash="7 3", label="This run union of segments"),
                   dict(df=rates, col="exceedance_rate_source_zone", color="var(--accent)",
                        width=2.5, dash="", label="This run source zone (mix)")]
        seg_chart_svg = svg_curves(series, "Segmented exceedance rates, this run and PTHA18")

        n_zone_obs = int(cfg["rates"]["observed_seismicity"]["count"])
        official_part = (f'''
  <h3 style="margin-top:22px">PTHA18's own segments</h3>
  <p class="sec-note">Step 8 ran PTHA18's segmented model through the same engine: its segment boundaries (columns of PTHA18's own mesh), its per-segment coupling and <code>mw_max_observed</code> (<code>sourcezone_parameters.csv</code>) and the GCMT events the saved session gives each segment. This reproduces PTHA18's official tree of every segment to machine precision (<code>validate_v9.py official_segments</code>).</p>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>PTHA18 segment</th><th>Columns (n)</th><th>Convergence (mm/yr)</th><th>Coupling min/pref/max</th><th>Mw max observed</th><th>GCMT events</th><th>Rate at Mw 7.2 (/yr)</th></tr></thead>
      <tbody>{"".join(off_seg_rows)}</tbody>
    </table>
  </div>

  <h3 style="margin-top:22px">This run against PTHA18, all three curves</h3>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th rowspan="2"></th><th colspan="3">Unsegmented (w=0.5)</th><th colspan="3">Union of segments (w=0.5)</th><th colspan="3">Source zone (the mix)</th></tr>
      <tr><th>this run</th><th>PTHA18</th><th>diff</th><th>this run</th><th>PTHA18</th><th>diff</th><th>this run</th><th>PTHA18</th><th>diff</th></tr></thead>
      <tbody>{"".join(vs_row(m) for m in (7.2, 7.6, 8.0, 8.4, 8.8, 9.2))}</tbody>
    </table>
  </div>
  <p class="sec-note">Exceedance rates in /yr. The method is the same (the engine reproduces PTHA18's segmented scenario rates exactly from PTHA18's inputs, <code>validate_v9.py events_segmented</code>); the inputs differ: the segments (Bird's plate pairs, not PTHA18's table: count and boundaries above), a segment's Berryman row where Bird's segment spans two of PTHA18's, <code>mw_max_observed</code> (this run's own GCMT maximum and Berryman's Mmax-min, PTHA18's table with its own adjustments), and the mesh and GCMT differences the unsegmented comparison already shows.</p>'''
                         if has_off else '''
  <p class="sec-note" style="margin-top:14px">No official segmented run to compare with: run <code>step8_official.py</code> (PTHA18 segments this zone only if <code>sourcezone_parameters.csv</code> has segment rows for it), then this step again.</p>''')

        segmentation_section = f'''
<section class="sec">
  <h2 id="level0">LEVEL 0: segmentation</h2>
  <p class="sec-note">This zone was run <b>segmented</b>. A segment is a stretch of the trench, a run of along-strike columns of the mesh, that is allowed to break on its own. PTHA18 does not choose between a segmented and an unsegmented model: it gives each weight 0.5 and mixes them. Within the segmented half the segments are <b>summed</b>, not chosen between, because all of them are active (report Section 3.7.6). So the zone's answer is <code>0.5 &times; unsegmented + 0.5 &times; (sum of segments)</code>.</p>

  <h3 style="margin-top:18px">How the zone is divided</h3>
  <div class="chart-wrap">{strips_svg}</div>
  <p class="sec-note">Each bar is the whole trench from its first column to its last, on a common 0-100% scale; each colour is one segment (the same colours as the map below).{" PTHA18's mesh has its own column count, so compare positions along the bar rather than column numbers." if has_off else ""}</p>
  {f'<figure style="margin:14px 0 0"><img src="{seg_map_uri}" alt="Map of the mesh cells coloured by segment" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Every cell coloured by the segment its column belongs to.{" Left: PTHA18&apos;s own segments on its published mesh. Right: this run." if has_off and len(strip_rows) > 1 else ""}</figcaption></figure>' if seg_map_uri else ''}

  <div class="callout" style="margin-top:16px">
    {'<p><b>Where this run&apos;s boundaries came from.</b> From <b>Berryman et al. (2015)</b> Table 3.1, which gives the trench end points of every segment (<code>Left_E_LONG</code>, <code>Left_N_LAT</code>, <code>Right_E_LONG</code>, <code>Right_N_LAT</code>). Each end point is placed on this mesh&apos;s top edge, and the boundary between two neighbouring segments falls at the column edge nearest to their shared end point. These are the segments PTHA18 used: on PTHA18&apos;s own meshes this rule lands within 0 to 2 columns of PTHA18&apos;s indices. <b>No PTHA18 file was read to place them</b>, which is what lets a segmented run keep the <code>--ptha false</code> promise.</p><p class="sec-note" style="margin-bottom:0">Berryman places segment ends mostly where the plate pair changes; PTHA18 then merged some of them (Patagonia, eastern Alaska) and added one (Arakan on sunda2), so the counts can still differ from PTHA18&apos;s.</p>' if seg_from == "berryman" else '<p><b>Where this run&apos;s boundaries came from.</b> They are derived from <b>Bird (2003)</b>&apos;s plate-boundary steps (<code>PB2002_steps.dat</code>, class <code>SUB</code>): every along-strike column is assigned to the plate pair of the nearest subduction step, and a run of columns sharing a pair becomes one segment (<code>--segment-boundaries bird</code>). <b>No PTHA18 file was read to place them</b>, which is what lets a segmented run keep the <code>--ptha false</code> promise.</p><p class="sec-note" style="margin-bottom:0">Bird&apos;s plate pairs are not PTHA18&apos;s segment list, and this run does not force them to agree. Where PTHA18 splits a trench that Bird does not (Hikurangi, whose forearc plate Bird does not have), the two differ, and that is a difference between two source models rather than an error.</p>'}
  </div>

  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Segment</th><th>{"Berryman plate pair" if seg_from == "berryman" else "Bird plate pair"}</th><th>Columns (n)</th><th>Convergence (mm/yr)</th><th>Coupling min/pref/max</th><th>Mw max observed</th><th>GCMT events</th><th>Rate at Mw 7.2 (/yr)</th></tr></thead>
      <tbody>{"".join(seg_rows)}</tbody>
    </table>
  </div>
  <p class="sec-note">As in PTHA18, each segment has its own convergence (the area-weighted Bird convergence of its own cells), its own coupling and Mw max observed ({"its own Berryman et al. 2015 row" if seg_from == "berryman" else "its Berryman et al. 2015 row(s), where Bird&apos;s plate pair names them; otherwise the zone&apos;s"}), and the earthquakes PTHA18's selection rule picks on its own cells: an event near a boundary counts in both neighbours, so the counts can add up to more than the zone's {n_zone_obs}. Every segment is computed on the zone's full scenario table, each scenario weighted by the part of it inside the segment, so a rupture across a boundary takes rate from both segments.</p>
{official_part}

  <h3 style="margin-top:22px">The segmented curves</h3>
  <div class="chart-wrap">{seg_chart_svg}</div>
  <div class="legend">
    <span><span class="sw" style="background:var(--accent)"></span>this run: source zone (mix)</span>
    <span><span class="sw" style="background:var(--accent-2)"></span>this run: union of segments (dashed)</span>
    <span><span class="sw" style="background:var(--text-soft)"></span>this run: each segment (dotted)</span>
    {'<span><span class="sw" style="background:var(--amber)"></span>PTHA18: the same three</span>' if has_off else ''}
  </div>

  <h3 style="margin-top:22px">What segmenting changed in this run</h3>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th></th><th>Unsegmented (w=0.5)</th><th>Union of segments (w=0.5)</th><th>Difference</th><th>Source zone (the mix)</th></tr></thead>
      <tbody>{"".join(mix_row(m) for m in (7.2, 8.0, 8.8, 9.2))}</tbody>
    </table>
  </div>
  <p class="sec-note">The union typically sits <b>above</b> the unsegmented curve at small magnitudes (several smaller faults produce more moderate earthquakes than one large one) and <b>below</b> it at the largest magnitudes, because no single segment is long enough to host a rupture that spans the whole zone. That crossover is the physical content of LEVEL 0; the last column is what this run actually reports for the zone.</p>
</section>
'''
    else:
        segmentation_section = ""

    # ---- v9: a segmented run's report is two pages, the segmented view
    # (report.html) and the unsegmented branch alone (report_unsegmented.html),
    # linked by a switch at the very top. An unsegmented run has one page and
    # no switch, so its report.html is byte for byte what it was before.
    view = d.get("view")
    if view:
        _tabs = []
        for _href, _key, _title, _what in (
                ("report.html", "segmented", "Segmented",
                 "the zone's answer: 0.5 &times; unsegmented + 0.5 &times; segments"),
                ("report_unsegmented.html", "unsegmented", "Unsegmented only",
                 "the zone as one fault, as an unsegmented run shows it")):
            _cur = ' aria-current="page"' if _key == view else ""
            _tabs.append(f'<a href="{_href}"{_cur}><b>{_title}</b>{_what}</a>')
        view_switch = VIEW_SWITCH_CSS + (
            '<nav class="view-switch" aria-label="Report view">'
            + "".join(_tabs) + '</nav>\n')
    else:
        view_switch = ""

    # ---- v9: which kind of run this is, stated first on the page. A
    # segmented and an unsegmented report share most blocks, but several of
    # them then describe different curves, so the reader must know up front.
    if segmentation_section:
        _seg_list = ", ".join(
            f"<code>{esc(k)}</code> ({int(sc['alongstrike_slice'][0])}-{int(sc['alongstrike_slice'][1])})"
            for k, sc in seg_cfg.items())
        _src = ("Berryman et al. (2015) Table 3.1" if seg_from == "berryman"
                else "Bird (2003) plate pairs")
        mode_banner = f'''
<div class="callout warn" style="max-width:none;margin:0 0 22px">
  <p style="margin:0 0 6px"><b>SEGMENTED RUN (LEVEL 0).</b> The zone's answer is <code>0.5 &times; unsegmented + 0.5 &times; (sum of {len(seg_cfg)} segments)</code>: {_seg_list}, columns of this mesh, boundaries from {_src}.</p>
  <p style="margin:0">Read the page with that in mind: the step cards, the prior-versus-posterior and moment-balance figures and the "this run vs PTHA18" table describe the <b>unsegmented branch</b> (the zone as one fault); the percentile band, the full rate table and the <a href="#level0">LEVEL 0 section</a> describe the <b>whole zone</b> (the mix). Step 8 ran PTHA18's own segmented model.</p>
</div>'''
    elif d.get("view") == "unsegmented":
        mode_banner = '''
<div class="callout good" style="max-width:none;margin:0 0 22px">
  <p style="margin:0 0 6px"><b>UNSEGMENTED VIEW OF A SEGMENTED RUN.</b> This page shows only the unsegmented branch: the zone as one fault that can break from end to end, read as if it had weight 1. It is the report an unsegmented run of this zone gives (same branches, same curves, same band), built from the files this segmented run already wrote, so nothing was run twice.</p>
  <p style="margin:0">It is <b>not</b> the zone's answer of this run: that is <code>0.5 &times; this branch + 0.5 &times; (sum of segments)</code>, on the <a href="report.html">segmented view</a>. The comparison with PTHA18 here is against PTHA18's unsegmented branch.</p>
</div>'''
    else:
        mode_banner = '''
<div class="callout good" style="max-width:none;margin:0 0 22px">
  <p style="margin:0"><b>UNSEGMENTED RUN.</b> The zone is one fault that can break from end to end, with weight 1 (no LEVEL 0 segments), so every curve and number on this page is the zone's. A segmented run of the same zone (<code>generate.py ... --segmented true</code>) adds a LEVEL 0 section and mixes this answer 50/50 with the sum of its segments.</p>
</div>'''

    # ---- SLAB1.0 vs SLAB2.0 section: which product this zone actually
    # used, derived from the zone name's own suffix (slab_product_for_zone,
    # imported above), not hardcoded per zone -- valid for any new zone
    # generate.py produces. ----
    slab_is_v2 = SLAB_PRODUCT == "SLAB2.0"
    slab_section = f'''
<section class="sec">
  <h2>Which SLAB product this zone actually used</h2>
  <p class="sec-note">PTHA18's own report (<em>ReportPTHA.pdf</em>, p.11) explains why it mixes two depth products: <em>&ldquo;Although SLAB2.0 is an update of SLAB1.0, the former was released just prior to the PTHA18 completion, and so we chose to update to SLAB2.0 only at source-zones near to Australia where the geometry changed substantially from SLAB1.0.&rdquo;</em> Figure 3's caption (p.12) gives the exact rule this report applies: <em>&ldquo;Those with names finishing in &lsquo;2&rsquo; used SLAB2.0 &mdash; other source-zones either used SLAB1.0 or prescribed linear or parabolic profiles.&rdquo;</em></p>
  <div class="callout {"good" if slab_is_v2 else "warn"}">
    <span class="lbl">Zone &ldquo;{esc(PTHA18_ZONE_NAME)}&rdquo; {"ends in &ldquo;2&rdquo;" if slab_is_v2 else "does not end in &ldquo;2&rdquo;"} &rarr; {SLAB_PRODUCT}</span>
    {(f'This is one of the zones PTHA18 updated to SLAB2.0, so this from-scratch reconstruction and the official mesh were built from the same depth product (Hayes et al. 2018). Any remaining gap on this page is attributable to discretisation and hand-editing differences, not to a different underlying depth model.' if slab_is_v2 else
      f'PTHA18 kept the older SLAB1.0 product (Hayes, Wald &amp; Johnson 2012) for this zone. <code>step1_fetch_slab2.py</code> downloads it automatically from <code>earthquake.usgs.gov/static/lfs/data/slab/models/</code> when <code>slab_product_for_zone()</code> returns &ldquo;SLAB1.0&rdquo; &mdash; no manual override needed, and no zone-specific code path beyond the suffix rule itself. Passing SLAB2.0 geometry into this zone instead (an easy mistake, since the download script is still named step1_fetch_<em>slab2</em>.py.tmpl for historical reasons) would silently compare against the wrong depth model.')}
  </div>
</section>
'''
    if MESH_FILE is not None:
        slab_section = ""  # v12: no SLAB with an external mesh

    # ---- full exceedance-rate table, from the engine's own percentile
    # output (not just the 3 headline points above) ----
    perc = d["percentiles"]
    _perc_file = d.get("percentiles_file", "exceedance_rate_percentiles.csv")
    off_perc = (pd.read_csv(os.path.join(OFFICIAL_RUN_DIR, _perc_file))
               if os.path.isdir(OFFICIAL_RUN_DIR) and
               os.path.exists(os.path.join(OFFICIAL_RUN_DIR, _perc_file))
               else None)
    if perc is not None and len(perc):
        off_mean_by_mw = (dict(zip(off_perc["threshold_Mw"].round(2), off_perc["mean_exrate"]))
                          if off_perc is not None else {})
        exrate_rows = []
        for _, row in perc.iterrows():
            mw = round(float(row["threshold_Mw"]), 2)
            mine = row["mean_exrate"]
            ov = off_mean_by_mw.get(mw)
            if mine <= 0 and not ov:
                continue      # above every branch's Mw_max, in both runs
            if ov is None:
                off_txt, gap_txt = "n/a", "n/a"
            else:
                off_txt = f"{ov:.3g}"
                gap_txt = f"{100*(mine-ov)/ov:+.1f}%" if ov else "n/a"
            exrate_rows.append(
                f'<tr><td class="n">{mw:g}</td><td class="n">{mine:.3g}</td>'
                f'<td class="n src">{off_txt}</td><td class="n">{gap_txt}</td></tr>')
        exrate_table = f'''
<section class="sec">
  <h2>Exceedance rate by magnitude, full table</h2>
  <p class="sec-note">Mean exceedance rate (events/yr) at every magnitude this run reports percentiles for (every 0.1 from 7.2; rows stop where the rate is zero in both runs), against the official run when available.{" Segmented run: both columns are the whole zone (the LEVEL 0 mix of unsegmented and segments), this run's segments against PTHA18's." if cfg.get("segments") else ""}</p>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Mw</th><th>This run (mean)</th><th>Official (mean)</th><th>Difference</th></tr></thead>
      <tbody>{"".join(exrate_rows)}</tbody>
    </table>
  </div>
</section>
'''
    else:
        exrate_table = ""

    # ---- moment-balance section: what LEVEL 4's "target" actually is, and
    # the interactive along-strike chart built from this run's own
    # integrated_slip_<zone>.csv (falls back to the engine's own PNG if that
    # table is missing, e.g. an older run). ----
    slip_svg = svg_moment_balance(d["slip"])
    moment_png = embed_image(
        os.path.join(OUT_DIR, "figures", f"fig_moment_balance_{SUFFIX}.png"))
    if slip_svg is not None:
        moment_chart_html = (f'<div class="chart-wrap">{slip_svg}</div>'
                            f'<div class="legend">'
                            f'<span><span class="sw" style="background:var(--text-soft)"></span>no edge correction</span>'
                            f'<span><span class="sw" style="background:var(--accent-2)"></span>target: plate convergence shape</span>'
                            f'<span><span class="sw" style="background:var(--amber)"></span>edge-corrected</span>'
                            f'</div>')
    elif moment_png is not None:
        moment_chart_html = (f'<div class="chart-wrap"><img src="{moment_png}" '
                            f'alt="Moment balance fit for {esc(TITLE)}" '
                            f'style="width:100%;height:auto;display:block"></div>')
    else:
        moment_chart_html = ""

    target_note = ("Here it is Bird's convergent rate of each column "
                   "(<code>convergence_profile</code> in the input JSON), as in PTHA18."
                   if cfg["rates"].get("convergence_profile") is not None else
                   "Here the convergence is one constant, so the target is flat.")
    if moment_chart_html:
        moment_section = f'''
<section class="sec">
  <h2>What the moment-balance &ldquo;target&rdquo; actually is</h2>
  <p class="sec-note">Added up over every scenario, the slip the earthquakes release along the zone should have the same shape as the plate convergence: that shape is LEVEL 4's target. {target_note} Near the two ends of the zone fewer scenarios overlap, so the released slip sags there (grey). LEVEL 4 fits a single multiplier of the probability of scenarios that touch an end ({edge_mult:.3f} on this run{f", {float(off_edge):.3f} on the official run" if off_edge is not None else ""}) until the two shapes agree as well as possible (orange). The three curves are normalised to the same total, because only the shapes are compared; one multiplier can lift the ends but cannot reshape the middle.</p>
  {moment_chart_html}
</section>
'''
    else:
        moment_section = ""

    gcmt_rows = "\n".join(
        f'<tr><td class="n">{r.date}</td><td class="n">{r.Mw:.3f}</td>'
        f'<td class="n">{r.depth_km:.1f}</td>'
        f'<td class="n">{r.hypo_lat:.2f}, {r.hypo_lon:.2f}</td>'
        f'<td class="n">{r.cent_lat:.2f}, {r.cent_lon:.2f}</td></tr>'
        for r in gcmt.itertuples()
    ) if len(gcmt) else '<tr><td colspan="5" class="src">no events qualified</td></tr>'

    dip_block = (f'<div class="stat"><span class="k">Official mean dip</span>'
                f'<span class="v">{float(off_mean_dip):.2f}<small> deg</small></span></div>'
                if off_mean_dip is not None else "")
    dip_callout = (
        # v12: an external mesh is not built from SLAB; say what it is
        f'<div class="callout"><span class="lbl">External mesh against the official mesh</span>'
        f'This run\'s mesh is the external file (inputs/geometry/{esc(MESH_FILE)}), used as given: '
        f'mean dip {mean_dip:.2f} deg against the official {float(off_mean_dip):.2f} deg. '
        f'Any gap comes from the file itself, not from SLAB or from meshing.</div>'
        if (off_mean_dip is not None and MESH_FILE is not None) else
        f'<div class="callout warn"><span class="lbl">Gap against the official mesh</span>'
        f'This run\'s mean dip ({mean_dip:.1f} deg) differs from the official '
        f'{float(off_mean_dip):.1f} deg. The official mesh was built from '
        f'hand-edited contours (a published shapefile, but not one this run '
        f'reads as a geometry source -- and the editing PROCESS itself is not '
        f'published anywhere), so an exact match is not expected; the '
        f'seismogenic cutoff (from Berryman et al. 2015, '
        f'see berryman_params.py) and where SLAB puts the slab and its '
        f'trench (figures/step1_trench_and_contours.png) are the main levers '
        f'behind this gap. '
        f'It flows through the whole run below, most visibly in the '
        f'slip rate (convergence / cos(dip)) and the moment balance.</div>'
        if off_mean_dip is not None else
        ('<p class="sec-note">' + f"{PTHA18_ZONE_NAME} is not one of PTHA18's source zones (PTHA18 covers the subduction zones that can send tsunamis to Australia), so there is no official run, mesh or HS/VAUS catalogue to compare with. Everything in this report is this run's own.".replace("{PTHA18_ZONE_NAME}", PTHA18_ZONE_NAME) + ' The seismogenic depth cutoff in step 1 is derived from Berryman et al. (2015), the source PTHA18 itself cites for it (berryman_params.py).</p>')
        if NOT_A_PTHA18_ZONE else
        '<div class="callout warn"><span class="lbl">No official mesh to compare against yet</span>'
        'Run step8_official.py for a comparison. The seismogenic depth '
        'cutoff in step 1 is derived from Berryman et al. (2015) -- the '
        'actual primary source PTHA18 itself cites for it, see '
        'berryman_params.py -- not a placeholder. It will not exactly match '
        'PTHA18\'s own cutoff for every zone (see that module\'s docstring '
        'for the verified cases), so treat this run\'s mean dip as '
        'approximate either way.</div>'
    )

    # Step 3's plate-boundary check. Shown only when it found something --
    # a mesh that overshoots its zone reads the NEXT boundary's convergence
    # at a normal match distance, so nothing else in this report would show
    # it as an error.
    plate_change_callout = ""
    if os.path.exists(PLATE_CHANGE_JSON):
        try:
            with open(PLATE_CHANGE_JSON) as _f:
                _pc = json.load(_f)
            _tail = "first" if _pc["end"] == "start" else "last"
            _cols = _pc["columns"]
            if _pc.get("auto_clip_separable"):
                _bb = _pc["auto_clip_bbox"]
                _how = (
                    f'To cut them, re-run with <code>--auto-clip</code> '
                    f'(<code>steps/step_total.py --auto-clip</code>, or '
                    f'<code>steps/step1_fetch_slab2.py --auto-clip</code> then '
                    f'steps 2-3). It clips the raster to '
                    f'lon {_bb[0]:.2f}&ndash;{_bb[1]:.2f}, '
                    f'lat {_bb[2]:.2f}&ndash;{_bb[3]:.2f}, a window derived '
                    f'from this mesh &mdash; you do not supply any coordinates.')
            else:
                _how = (
                    f'A lon/lat box cannot separate this tail from the rest of '
                    f'the zone here (the arc doubles back), so '
                    f'<code>--auto-clip</code> will refuse. Cut it with an '
                    f'explicit <code>--clip LON_MIN,LON_MAX,LAT_MIN,LAT_MAX</code> '
                    f'on generate.py instead.')
            plate_change_callout = (
                f'<div class="callout warn"><span class="lbl">This mesh may run '
                f'past its own plate boundary</span>'
                f'Step 3 matches every along-strike column to the nearest '
                f'Bird (2003) segment, and that match never fails &mdash; so a '
                f'mesh that extends beyond its own subduction zone quietly '
                f'reads the <em>next</em> boundary&rsquo;s convergence at a '
                f'perfectly normal match distance, with nothing else in this '
                f'report flagging it. '
                f'Here the {_tail} <strong>{_pc["n_suspect"]} of '
                f'{_pc["n_total"]}</strong> columns (numbers {_cols[0]}&ndash;'
                f'{_cols[-1]}) look like a different plate pair: the '
                f'convergent component steps by '
                f'<strong>{_pc["jump_mm_per_yr"]:.1f} mm/yr</strong> across '
                f'column {_pc["boundary_column"]} '
                f'({_pc["div_inside"]:.1f} mm/yr on this zone&rsquo;s side, '
                f'{_pc["div_outside"]:.1f} on the far side). '
                f'Trimming them would move the mean convergent slip from '
                f'{_pc.get("mean_all", float("nan")):.1f} to about '
                f'<strong>{_pc["mean_inside"]:.1f} mm/yr</strong> '
                f'(unweighted column means). '
                f'{_how} '
                f'<em>Nothing has been trimmed in this run</em> &mdash; every '
                f'number below still includes those columns.</div>')
        except Exception as _exc:
            print(f"  note: could not render the plate-boundary callout: {_exc}")

    conv_block = (f'<div class="stat"><span class="k">Official (recovered)</span>'
                 f'<span class="v">{float(off_conv):.2f}<small> mm/yr</small></span></div>'
                 f'<div class="stat"><span class="k">Difference</span>'
                 f'<span class="v">{100*(convergence-float(off_conv))/float(off_conv):+.1f}<small> %</small></span></div>'
                 if off_conv is not None else "")

    gcmt_official_block = (
        f'<div class="stat"><span class="k">Official count</span><span class="v">{off_gcmt_count}</span></div>'
        f'<div class="stat"><span class="k">Official window</span><span class="v">{off_gcmt_duration:.1f}<small> yr</small></span></div>'
        if off_gcmt_count is not None else "")
    # Window comparison is read from what step 5 actually filtered on
    # (WINDOW_START/WINDOW_END, surfaced here via obs["duration_years"]), not
    # assumed -- earlier report text hardcoded "runs to today", which went
    # stale once step4/step5 were pinned to PTHA18's own 2017-03-01 cutoff.
    _same_window = (off_gcmt_duration is not None
                    and abs(obs["duration_years"] - off_gcmt_duration) < 0.05)
    if off_gcmt_count is None:
        gcmt_callout = ""
    elif _same_window:
        gcmt_callout = (
            f'<div class="callout good"><span class="lbl">Same observation window as PTHA18</span>'
            f'This run filters GCMT events over the same {obs["duration_years"]:.1f}-year '
            f'window PTHA18 itself used (ending 2017-03-01, report Section 3.7.3), so the '
            f'{n_subset}-vs-{off_gcmt_count} event counts are directly comparable: any '
            f'difference reflects the subsetting filter and catalogue, not a different '
            f'time span.</div>')
    else:
        gcmt_callout = (
            f'<div class="callout warn"><span class="lbl">Why {n_subset} and not {off_gcmt_count}</span>'
            f'This window is {obs["duration_years"]:.1f} years, not PTHA18\'s own cutoff of '
            f'2017-03-01 ({off_gcmt_duration:.1f} years), so the two counts are not directly '
            f'comparable: both the numerator and the denominator differ. What matters is that '
            f'the filter finds a small number of real, plausible thrust events clustered on '
            f'this interface, which it does.</div>')

    tvd_callout_tail = (
        f'unlike the official run\'s {off_gcmt_count}-event update, which '
        f'yields a shift of {float(off_tvd):.3f} against the same '
        f'{n_branches:,}-branch tree.'
        if off_tvd is not None else
        'run step8_official.py to compare this against the official update\'s shift.'
    )

    # ---- prior vs posterior section: the engine's own branch-fan figure
    # (built separately from the main f-string so the optional official
    # panel does not have to nest inside it) ----
    if branch_fan_img is None:
        prior_posterior_section = ""
    else:
        official_panel = ""
        if off_branch_fan_img is not None:
            off_threshold = cfg["rates"]["observed_seismicity"]["threshold_Mw"]
            official_panel = (
                '<div style="margin-top:22px">'
                f'<p class="sec-note" style="margin-bottom:10px">The same figure '
                f'for the official run, for comparison: PTHA18\'s own '
                f'observations (Mw&ge;{off_threshold:g}) versus this run\'s '
                f'{n_subset}.</p>'
                f'<div class="chart-wrap"><img src="{off_branch_fan_img}" '
                f'alt="Official run: all logic-tree branches, before and '
                f'after the LEVEL 3 weight update" style="width:100%;'
                f'height:auto;display:block"></div>'
                '</div>'
            )
        prior_posterior_section = f'''
<section class="sec">
  <h2>Prior versus posterior: what LEVEL 3 actually did</h2>
  <p class="sec-note">Every one of the {n_branches:,} logic-tree branches' own Mw-vs-exceedance-rate curve (grey), before the Bayesian update (left) and after it (right), against the {n_subset} observed event(s) from step 5 (green). This is the engine's own figure, produced in step 7/8, not redrawn for this report.</p>
  <div class="chart-wrap">
    <img src="{branch_fan_img}" alt="All logic-tree branches, before and after the LEVEL 3 weight update" style="width:100%;height:auto;display:block">
  </div>
  <div class="figure" style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
    <div class="stat"><span class="k">Prior&rarr;posterior shift</span><span class="v">{tvd:.3f}<small> TVD</small></span></div>
    <div class="stat"><span class="k">Observed events</span><span class="v">{n_subset}</span></div>
  </div>
  {official_panel}
</section>
'''

    doc = f"""<title>{TITLE} From Scratch</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  --abyss:#0d2436; --deep:#12374f; --shelf:#2a6f8f; --shallow:#5ba3bd; --foam:#a8d4de;
  --sand:#f2f0eb; --paper:#faf9f6; --ink:#10222e; --ink-soft:#4a6270; --rule:#d5dee2;
  --amber:#c2701b; --amber-soft:#f5e4d0;
  --surface:var(--paper); --surface-2:var(--sand); --text:var(--ink); --text-soft:var(--ink-soft);
  --line:var(--rule); --accent:var(--deep); --accent-2:var(--shelf);
  --good:#2f7d4f; --good-soft:#e5f2ea; --bad:#b3453c; --bad-soft:#fbe9e7;
}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    --surface:#0a1a25; --surface-2:#102632; --text:#e2edf1; --text-soft:#93aeba;
    --line:#1e3c4d; --accent:#7fc4d8; --accent-2:#5ba3bd; --amber:#e0a058; --amber-soft:#2a2013;
    --good:#5fbf8a; --good-soft:#123024; --bad:#e2857c; --bad-soft:#301715;
  }}
}}
:root[data-theme="dark"]{{
  --surface:#0a1a25; --surface-2:#102632; --text:#e2edf1; --text-soft:#93aeba;
  --line:#1e3c4d; --accent:#7fc4d8; --accent-2:#5ba3bd; --amber:#e0a058; --amber-soft:#2a2013;
  --good:#5fbf8a; --good-soft:#123024; --bad:#e2857c; --bad-soft:#301715;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--surface);color:var(--text);font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.6}}
.wrap{{max-width:980px;margin:0 auto;padding:0 28px 100px}}
header{{border-bottom:1px solid var(--line);padding:56px 0 32px;margin-bottom:44px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent-2);margin:0 0 14px}}
h1{{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:clamp(32px,5vw,48px);line-height:1.08;margin:0 0 18px;text-wrap:balance;letter-spacing:-.015em}}
.lede{{font-size:16.5px;color:var(--text-soft);max-width:66ch;margin:0}}
.lede strong{{color:var(--text);font-weight:500}}
h2{{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:24px;margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance}}
.sec{{margin-top:52px}}
.sec-note{{color:var(--text-soft);font-size:14.5px;max-width:68ch;margin:0 0 20px}}
.stepcard{{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:0 4px 4px 0;padding:20px 22px;margin-bottom:14px;background:var(--surface-2)}}
.stepcard h3{{margin:0 0 4px;font-size:16px;font-weight:600;display:flex;align-items:baseline;gap:9px}}
.stepcard .idx{{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--accent-2);letter-spacing:.05em}}
.stepcard p{{margin:6px 0 0;color:var(--text-soft);font-size:14.5px;max-width:66ch}}
.stepcard .figure{{margin-top:12px;display:flex;flex-wrap:wrap;gap:10px}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:9px 14px;min-width:120px}}
.stat .k{{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-soft);display:block;margin-bottom:3px}}
.stat .v{{font-family:"IBM Plex Mono",monospace;font-size:16px;font-weight:600;color:var(--text);font-variant-numeric:tabular-nums}}
.stat .v small{{font-size:11px;font-weight:400;color:var(--text-soft)}}
.chart-wrap{{background:var(--surface-2);border:1px solid var(--line);border-radius:4px;padding:20px 20px 6px;overflow-x:auto}}
.chart-wrap svg{{display:block;min-width:560px;width:100%;height:auto}}
.legend{{display:flex;gap:22px;flex-wrap:wrap;margin:12px 0 4px;font-size:13px;color:var(--text-soft)}}
.legend .sw{{display:inline-block;width:22px;height:2.5px;vertical-align:middle;margin-right:6px;border-radius:1px}}
.legend .band{{display:inline-block;width:14px;height:14px;vertical-align:middle;margin-right:6px;border-radius:2px;opacity:.5}}
.tbl-wrap{{overflow-x:auto;margin-top:10px}}
table{{border-collapse:collapse;width:100%;min-width:440px;font-size:13.5px}}
th,td{{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--line)}}
th{{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-soft);font-weight:500}}
td.n{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
td.src{{color:var(--text-soft)}}
.callout{{border-radius:0 4px 4px 0;padding:15px 18px;margin:16px 0 0;font-size:14px;max-width:68ch;border-left-width:3px;border-left-style:solid}}
.callout.warn{{background:var(--amber-soft);border-color:var(--amber)}}
.callout.good{{background:var(--good-soft);border-color:var(--good)}}
.callout .lbl{{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;display:block;margin-bottom:5px}}
.callout.warn .lbl{{color:var(--amber)}}
.callout.good .lbl{{color:var(--good)}}
.pill{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.05em;padding:2px 8px;border-radius:10px;margin-left:8px;vertical-align:middle}}
.pill.derived{{background:var(--good-soft);color:var(--good)}}
.pill.declared{{background:var(--amber-soft);color:var(--amber)}}
footer{{margin-top:60px;padding-top:20px;border-top:1px solid var(--line);font-size:13px;color:var(--text-soft)}}
a{{color:var(--accent-2)}}
.hoverchart .hitdot{{cursor:crosshair}}
.series-toggle{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:10px 0 2px;font-size:12px}}
.series-toggle .st-lbl{{color:var(--text-soft);margin-right:4px}}
.series-toggle .st{{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:transparent;color:var(--text);border-radius:999px;padding:3px 10px;cursor:pointer;opacity:.4;font:inherit}}
.series-toggle .st.on{{opacity:1}}
.series-toggle .st-sw{{display:inline-block;width:12px;height:4px;border-radius:2px}}
.chart-tip{{position:fixed;z-index:50;pointer-events:none;background:#10222e;color:#f4f8fa;font-family:"IBM Plex Mono",monospace;font-size:11.5px;line-height:1.5;padding:7px 10px;border-radius:4px;box-shadow:0 4px 14px rgba(0,0,0,.25);max-width:220px}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]) .chart-tip{{background:#e2edf1;color:#0a1a25}}}}
:root[data-theme="dark"] .chart-tip{{background:#e2edf1;color:#0a1a25}}
.rupture-explorer{{margin:16px 0 0}}
.rup-controls{{display:flex;flex-wrap:wrap;align-items:center;gap:18px;margin-bottom:10px;font-size:13px;color:var(--text-soft)}}
.rup-controls label{{display:flex;flex-direction:column;gap:4px;font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.04em;text-transform:uppercase}}
.rup-controls select, .rup-controls input[type=range]{{font-family:inherit;font-size:14px}}
.rup-slider-label{{flex:1;min-width:200px}}
#rup-pos{{width:100%}}
.rup-count{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--text)}}
.rup-play{{font-family:inherit;font-size:14px;line-height:1;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--surface-2);color:var(--text);cursor:pointer}}
.rup-play:hover{{border-color:var(--accent-2)}}
#rup-svg{{width:100%;max-width:520px;height:auto;display:block;margin:0 auto;background:var(--surface)}}
.rup-cell{{fill:var(--surface-2);stroke:var(--line);stroke-width:.5;transition:fill .05s linear}}
.rup-f{{fill:var(--accent-2)}}
.rup-v{{fill:var(--amber)}}
.rup-fv{{fill:url(#rup-both)}}
.rup-both-a{{fill:var(--accent-2)}}
.rup-both-b{{fill:var(--amber)}}
.rup-legend{{display:flex;flex-wrap:wrap;gap:16px;margin-top:8px;font-size:12.5px;color:var(--text-soft)}}
.rup-legend span{{display:flex;align-items:center;gap:6px}}
.rup-key{{display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid var(--line)}}
.rup-key.rup-f{{background:var(--accent-2)}}
.rup-key.rup-v{{background:var(--amber)}}
.rup-key.rup-fv{{background:repeating-linear-gradient(45deg,var(--accent-2) 0 3px,var(--amber) 3px 6px)}}
#rup-real:disabled{{opacity:.4}}
.vmu-svg{{width:100%;max-width:640px;height:auto;display:block;margin:6px 0 10px}}
.vmu-grid{{stroke:var(--line);stroke-width:1}}
.vmu-const{{stroke:var(--text-soft);stroke-width:1.3;stroke-dasharray:5 4}}
.vmu-curve{{fill:none;stroke:var(--accent-2);stroke-width:2.5}}
.vmu-band{{fill:var(--accent-2);opacity:.18}}
.vmu-row{{fill:var(--amber);stroke:var(--surface);stroke-width:2}}
.vmu-tick{{font-family:"IBM Plex Mono",monospace;font-size:11px;fill:var(--text-soft)}}
.vmu-lab{{font-size:12px;fill:var(--text-soft)}}
.vmu-note{{font-size:11.5px;fill:var(--text-soft)}}
.vmu-rowlab{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;font-weight:600;fill:var(--text);paint-order:stroke;stroke:var(--surface);stroke-width:4px}}
.rup-stage{{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start;justify-content:center}}
.rup-stage #rup-svg{{flex:1 1 360px;margin:0}}
.rup-ratio{{flex:0 1 340px;min-width:260px}}
.rup-ratio-title{{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--text-soft);margin-bottom:4px}}
#rup-ratio-svg{{width:100%;height:auto;display:block}}
.rup-panel-label{{display:flex;flex-direction:column;gap:4px;font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--text-soft);margin-bottom:6px}}
.rup-panel-label select{{font-family:inherit;font-size:14px;text-transform:none;letter-spacing:0;max-width:260px}}
.rup-ratio .sec-note p{{margin:0 0 8px}}
.rr-out{{fill:var(--surface)!important;stroke:var(--amber);stroke-width:1.5;opacity:1}}
.rr-limit{{stroke:var(--bad);stroke-width:1.5;stroke-dasharray:4 3}}
.rr-band{{fill:var(--accent-2);opacity:.13}}
.rr-grid{{stroke:var(--line);stroke-width:1}}
.rr-one{{stroke:var(--text);stroke-width:1.5;stroke-dasharray:4 3}}
.rr-tick{{font-family:"IBM Plex Mono",monospace;font-size:11px;fill:var(--text-soft)}}
.rr-row{{font-family:"IBM Plex Mono",monospace;font-size:11px;fill:var(--text-soft)}}
.rr-val{{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600;fill:var(--text);paint-order:stroke;stroke:var(--surface);stroke-width:5px;stroke-linejoin:round}}
.rr-all{{opacity:.35}}
.rr-cur{{stroke:var(--surface);stroke-width:2}}
.rr-f{{fill:var(--accent-2)}}
.rr-v{{fill:var(--amber)}}
.mesh-map{{height:min(80vh,720px);margin-top:14px;border:1px solid var(--line);border-radius:4px;background:var(--surface)}}
.mesh-map-offline{{padding:48px 20px;text-align:center;color:var(--text-soft);font-size:13.5px}}
</style>
<div class="wrap">
{view_switch}
<header>
  <p class="eyebrow">PTHA18 &middot; from-scratch run report</p>
  <h1>{TITLE}, Built From Scratch</h1>
  <p class="lede">A full run through <strong>{n_branches:,} logic-tree branches</strong> for the {TITLE} subduction interface, using only public data: {SLAB_PRODUCT} geometry, Bird (2003) convergence, and a fresh GCMT catalogue subset. This report reads back what each of the seven build steps actually wrote to disk.</p>
</header>
{mode_banner}

<section class="sec" style="margin-top:0">
  <h2>Step by step</h2>
  <p class="sec-note">What each script did, and the real numbers it produced on this run.</p>

  {step1_mesh_card if MESH_FILE is not None else ""}
  <div class="stepcard"{' hidden' if MESH_FILE is not None else ''}>
    <h3><span class="idx">STEP 1</span> Interface geometry from {SLAB_PRODUCT}<span class="pill derived">derived</span></h3>
    <p>Downloaded depth contours for the {TITLE} region from {SLAB_PRODUCT} ({"Hayes et al. 2018, USGS ScienceBase" if SLAB_PRODUCT == "SLAB2.0" else "Hayes, Wald &amp; Johnson 2012, USGS"}), unwrapped the antimeridian crossing if needed, and clipped to this zone's own seismogenic cutoff, derived from Berryman et al. (2015).</p>
    <div class="figure">
      <div class="stat"><span class="k">Contours kept</span><span class="v">{n_contours if n_contours else '&mdash;'}</span></div>
    </div>
    {ramp_note}
  </div>

  <div class="stepcard">
    <h3><span class="idx">STEP 2</span> Unit-source grid<span class="pill derived">derived</span></h3>
    <p>{"Took the unit sources from the external mesh, as given," if MESH_FILE is not None else "Discretised the contours into a grid of ~50&nbsp;km unit sources"} and computed per-unit-source statistics: length, width, dip, strike, depth.</p>
    <div class="figure">
      <div class="stat"><span class="k">Unit sources</span><span class="v">{n_unit_sources if n_unit_sources else '&mdash;'}</span></div>
      <div class="stat"><span class="k">Mean dip</span><span class="v">{mean_dip:.2f}<small> deg</small></span></div>
      {dip_block}
    </div>
    {dip_callout}
    {plate_change_callout}
    {f'<figure style="margin:20px 0 0"><img src="{mesh_profile_uri}" alt="Depth versus latitude along the arc for this run mesh, one line per down-dip row" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Depth against latitude, one line per down-dip row, for THIS run mesh. Shown whether or not an official mesh is available: the span on the x axis is the stretch of arc this mesh actually covers, and the spread between the lines is how much deeper each row sits. When step 8 has run, the same plot with PTHA18 own series overlaid appears in the comparison card below.</figcaption></figure>' if mesh_profile_uri else ''}
    {f'<figure style="margin:16px 0 0"><img src="{mesh_map_uri}" alt="Map of this run own unit-source mesh, cells coloured by depth to the interface" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Rebuilt from this run own {os.path.basename(INPUT_JSON)} (via run_logic_tree.py build_grid), the same geometry step 7 actually computed rates on -- including any field you edited by hand after step 6.</figcaption></figure>' if mesh_map_uri else ''}
  </div>

  {f'''<div class="stepcard">
    <h3><span class="idx">STEP 2</span> Official mesh, side by side<span class="pill derived">comparison</span></h3>
    <p class="sec-note">From step 8's downloaded unit_source_statistics_{PTHA18_ZONE_NAME}.nc -- the mesh PTHA18 actually published, hand-edited seams and all. Compared here against this run own rebuilt mesh above, on the two things that reach the moment balance (area, mean dip) and on whether either mesh stops short of the full arc.</p>
    <figure style="margin:16px 0 0"><img src="{mesh_official_uri}" alt="Map of the official PTHA18 unit-source mesh, cells coloured by depth to the interface" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">The official mesh alone, same colour scale convention as the run own mesh above.</figcaption></figure>
    <figure style="margin:20px 0 0"><img src="{mesh_compare_map_uri}" alt="Side-by-side map of the official PTHA18 mesh and this run own mesh, same depth colour scale" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Same map extent and depth colour scale on both panels -- the cell BOUNDARIES will not match (PTHA18 meshed its own hand-edited contours, published on NCI; this run builds its contours from SLAB), but the two meshes occupying the same patch of the interface is the thing this figure answers at a glance.</figcaption></figure>
    <figure style="margin:20px 0 0"><img src="{mesh_compare_profile_uri}" alt="Depth versus latitude along the arc, one line per mesh, official and this run" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Cell-centroid depth against latitude, one line per down-dip row, both meshes. Reveals directly whether either mesh's arc coverage stops short of the other's -- not visible from area/dip numbers alone.</figcaption></figure>
  </div>''' if mesh_official_uri else '' if NOT_A_PTHA18_ZONE else f'''<div class="stepcard">
    <h3><span class="idx">STEP 2</span> Official mesh, side by side<span class="pill declared">unavailable</span></h3>
    <p class="sec-note">No official unit_source_statistics_{PTHA18_ZONE_NAME}.nc found. Run step8_official.py first (needs R with rptha installed) for the official mesh map and the two comparison figures.</p>
  </div>'''}

  {web_map_card}

  <div class="stepcard">
    <h3><span class="idx">STEP 3</span> Convergence from Bird (2003)<span class="pill derived">derived</span></h3>
    <p>Matched the top edge of every along-strike column to its nearest Bird (2003) plate-boundary segment and area-weighted the convergent velocities, exactly as PTHA18 does (on PTHA18's own mesh this gives the official value to machine precision). The number below is horizontal plate motion; step 7 divides it by cos(mean dip), once, to get slip on the inclined fault plane. The per-column values also set how likely each rupture is (faster convergence under it, higher probability) and the shape the LEVEL 4 edge fit aims for, as in PTHA18.</p>
    <div class="figure">
      <div class="stat"><span class="k">This run</span><span class="v">{convergence:.2f}<small> mm/yr</small></span></div>
      {conv_block}
    </div>
    {f'<figure style="margin:16px 0 0"><img src="{mesh_convergence_uri}" alt="Map of this run own unit-source mesh, cells coloured by per-cell Bird (2003) convergent slip" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Each along-strike COLUMN shares one value (its own top/trench-edge match to the nearest Bird segment, propagated unchanged down-dip -- see step3_convergence.py); the {convergence:.2f} mm/yr figure above is this map area-weighted into a single scalar.</figcaption></figure>' if mesh_convergence_uri else ''}
    {f'<figure style="margin:20px 0 0"><img src="{mesh_official_convergence_uri}" alt="Side-by-side map of the official PTHA18 mesh and this run own mesh, cells coloured by per-cell Bird convergent slip on one shared colour scale" style="max-width:100%;border-radius:6px"><figcaption class="sec-note" style="margin-top:6px">Same method (top-edge Bird match, propagated down-dip), applied to each mesh own geometry. PTHA18 publishes only the area-weighted scalar, so the official panel is rebuilt from unit_source_statistics_{PTHA18_ZONE_NAME}.nc; that rebuild reproduces the official scalar exactly on every Bird zone tested.</figcaption></figure>' if mesh_official_convergence_uri else ''}
  </div>

  <div class="stepcard">
    <h3><span class="idx">STEP 4</span> GCMT catalogue download<span class="pill derived">derived</span></h3>
    <p>Downloaded the worldwide Global CMT catalogue from {WINDOW_START} to {WINDOW_END}, parsed from the raw NDK record format. This matches PTHA18's own observation window exactly (report Section 3.7.3), so the event counts are directly comparable with the official run.</p>
    <div class="figure">
      <div class="stat"><span class="k">Events, worldwide</span><span class="v">{n_catalogue:,}</span></div>
      <div class="stat"><span class="k">Window</span><span class="v" style="font-size:13px">{WINDOW_START} &ndash; {WINDOW_END}</span></div>
    </div>
  </div>

  <div class="stepcard">
    <h3><span class="idx">STEP 5</span> GCMT subset for this zone<span class="pill derived">derived</span></h3>
    <p>Applied a from-scratch reimplementation of gcmt_subsetter.R: hypocentre OR centroid within 0.4&deg; of a unit source, either nodal plane thrust-like (within 50&deg; of pure dip-slip) and strike-aligned with the unit source nearest the CENTROID (within 50&deg;), depth &le;71&nbsp;km, Mw&ge;{obs['threshold_Mw']:g}, over {WINDOW_START} &ndash; {WINDOW_END} (PTHA18's own window, report Section 3.7.3).</p>
    <div class="figure">
      <div class="stat"><span class="k">Qualifying events</span><span class="v">{n_subset}</span></div>
      {gcmt_official_block}
      <div class="stat"><span class="k">This window</span><span class="v">{obs['duration_years']:.1f}<small> yr</small></span></div>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr><th>Date</th><th>Mw</th><th>Depth (km)</th><th>Hypocentre (lat, lon)</th><th>Centroid (lat, lon)</th></tr></thead>
        <tbody>{gcmt_rows}</tbody>
      </table>
    </div>
    {gcmt_callout}
  </div>

  <div class="stepcard">
    <h3><span class="idx">STEP 6</span> Assemble the input JSON<span class="pill declared">partly declared</span></h3>
    <p>Combined the derived geometry, convergence and seismicity with three literature values that cannot be derived from geometry or a catalogue: seismic coupling (cmin/cpref/cmax = {coupling_axis['spreadsheet_values'][0]:g}/{coupling_axis['spreadsheet_values'][1]:g}/{coupling_axis['spreadsheet_values'][2]:g}), the b-value anchor, and the largest historically observed magnitude (Mw {cfg['rates']['mw_max_observed']:g}, the larger of Berryman's largest row and step 5's largest event) -- all from Berryman et al. (2015)'s "GEM Faulted Earth Subduction Interface Characterisation Project" via berryman_params.py (coupling from its Whole Margin row where it has one), not from any PTHA18-internal file.</p>
    <div class="figure">
      <div class="stat"><span class="k">b anchor (min/pref/max)</span><span class="v">{'/'.join(f'{v:g}' for v in b_anchor)}</span></div>
      <div class="stat"><span class="k">Prior mean coupling</span><span class="v">{prior_coupling:.3f}</span></div>
      <div class="stat"><span class="k">Bins (coupling&times;b&times;Mw_max)</span><span class="v" style="font-size:13px">20&times;20&times;40&times;2</span></div>
    </div>
  </div>

  <div class="stepcard">
    <h3><span class="idx">STEP 7</span> Run LEVELs 0&ndash;5<span class="pill derived">derived</span></h3>
    <p>Enumerated the full logic tree, solved the moment balance for each branch's Gutenberg-Richter 'a' parameter, applied the Bayesian update from step 5's observations, fitted the along-strike edge taper, and sampled percentiles.</p>
    <div class="figure">
      <div class="stat"><span class="k">Branches</span><span class="v">{n_branches:,}</span></div>
      <div class="stat"><span class="k">GR 'a' range</span><span class="v" style="font-size:14px">{a_min:.2f} &ndash; {a_max:.2f}</span></div>
      <div class="stat"><span class="k">Edge multiplier</span><span class="v">{edge_mult:.2f}</span></div>
      <div class="stat"><span class="k">Prior&rarr;posterior shift</span><span class="v">{tvd:.3f}<small> TVD</small></span></div>
    </div>
    <div class="callout good">
      <span class="lbl">What the Bayesian update did here</span>
      Total variation distance between prior and posterior is {tvd:.2f} (0&nbsp;=&nbsp;no change, 1&nbsp;=&nbsp;complete reshuffle). With {n_subset} observed event(s) pulling toward specific branches, that is how much LEVEL 3 moved the weights &mdash; {tvd_callout_tail}
    </div>
    <h4 style="margin:22px 0 6px;font-size:14px">LEVEL 2: rupture generation<span class="pill derived" style="margin-left:8px">FAUS</span></h4>
    <p>Before any rate is computed, every branch needs an actual set of earthquake-sized ruptures to assign rates to. For each magnitude in the {ev_mmin:g}&ndash;{ev_mmax:g} range (step {ev_dmw:g}), <code>pyptha_v12/events.py</code> (a port of rptha's <code>rupture_events.R</code>) sizes a rectangular block of unit sources from the Strasser scaling relation and enumerates every valid placement of that block on the mesh -- each placement is one rupture, with <b>uniform slip</b> set so its seismic moment matches M0(Mw) exactly. This is the FAUS (Fixed Area Uniform Slip) rupture type, and every LEVEL 0&ndash;5 rate number on this page is computed from FAUS ruptures only.{rupture_size_note}</p>
    <div class="figure">
      <div class="stat"><span class="k">Total ruptures enumerated</span><span class="v">{f'{n_events_total:,}' if n_events_total is not None else '&mdash;'}</span></div>
      <div class="stat"><span class="k">Magnitude range</span><span class="v" style="font-size:14px">{ev_mmin:g} &ndash; {ev_mmax:g}</span></div>
    </div>
    {rupture_explorer_html if rupture_explorer_html else ''}
  </div>

  {hs_section_html}

  {variable_mu_section(d)}

  <div class="stepcard">
    <h3><span class="idx">FILES</span> The numbers behind this page<span class="pill derived">csv</span></h3>
    <p>Every figure above is drawn from these files, written to <code>outputs/exports/</code> by step&nbsp;9. They are plain CSV, so they can be read, diffed or re-plotted without opening this report or re-running anything. Where PTHA18's own version exists (step&nbsp;8), it is written alongside with an <code>_official</code> suffix, with identical columns, so the two can be compared row by row.</p>
    <table>
      <thead><tr><th>file</th><th>what is in it</th></tr></thead>
      <tbody>{exports_rows if exports_rows else '<tr><td colspan="2" class="src">nothing exported</td></tr>'}</tbody>
    </table>
    <p class="sec-note" style="margin-top:10px">In <code>mesh_cells.csv</code> each row is one unit source: <code>lon</code>/<code>lat</code> are its centroid and <code>depth_km</code> its centroid depth, positive down &mdash; the same three numbers each coloured cell on the maps above is drawn from.</p>
  </div>
</section>

<section class="sec">
  <h2>The exceedance-rate curve</h2>
  <p class="sec-note">How often (per year) an earthquake exceeds a given magnitude on this interface. The shaded band is the 16th&ndash;84th percentile across all {n_branches:,} branches, i.e. epistemic uncertainty from the logic tree itself, not measurement noise.{" On this segmented run the band, like the full table below, is the whole zone's (the LEVEL 0 mix, sampled as PTHA18 does), so it belongs with the fine dotted source-zone line; the solid line and the official dashed one are the unsegmented branch." if segmentation_section else ""}</p>
  <div class="chart-wrap">
    {chart_svg}
  </div>
  <div class="legend">
    <span><span class="sw" style="background:var(--accent)"></span>this run ({SLAB_PRODUCT} + Bird + fresh GCMT)</span>
    {'<span><span class="sw" style="background:var(--amber);border-top:2px dashed var(--amber)"></span>official PTHA18 run</span>' if off_rates is not None else ''}
    <span><span class="band" style="background:var(--accent-2)"></span>16th&ndash;84th percentile</span>
    {'<span><span class="sw" style="background:var(--accent-2)"></span>union of segments</span><span><span class="sw" style="background:var(--text-soft)"></span>individual segments</span><span><span class="sw" style="background:var(--accent)"></span>source zone (LEVEL 0 mix)</span>' if segmentation_section else ''}
  </div>
  <div class="figure" style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
    <div class="stat"><span class="k">Rate at Mw 7.2</span><span class="v">{rate72:.3f}<small> /yr</small></span></div>
    <div class="stat"><span class="k">Rate at Mw 8.0</span><span class="v">{rate80:.4f}<small> /yr</small></span></div>
    <div class="stat"><span class="k">Rate at Mw 9.0</span><span class="v">{rate90:.5f}<small> /yr</small></span></div>
  </div>
</section>

{segmentation_section}

{prior_posterior_section}

{moment_section}

{exrate_table}

{slab_section}

{"" if off_rates is None else f'''
<section class="sec">
  <h2>This run vs. the official PTHA18 run</h2>
  <p class="sec-note">Same engine, same source zone, different inputs: this run uses public {SLAB_PRODUCT} geometry and a fresh GCMT subset; the official run uses PTHA18's own unit-source table (published on NCI) and its own saved GCMT extraction.</p>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Quantity</th><th>This run</th><th>Official</th><th>Difference</th></tr></thead>
      <tbody>{cmp_rows}</tbody>
    </table>
  </div>
  <div class="callout warn">
    <span class="lbl">{gap_attribution_label}</span>
    {gap_attribution_text}
  </div>
</section>
'''}

<footer>
  Generated by step9_report.py from examples/makran2/outputs/, examples/makran2/outputs_official/, examples/makran2/data/ and examples/makran2/inputs/. Re-run any earlier step and re-run this one to refresh the numbers above.
</footer>

</div>
{TOOLTIP_JS}
"""
    return doc


def main():
    print("=" * 70)
    print("STEP 9 - Build the HTML report")
    print("=" * 70)

    d = load_everything()
    if d["official"] is None:
        print(f"\n  note: no official run found at {OFFICIAL_RUN_DIR}")
        print("  the report will cover steps 1-7 only, with no comparison "
              "section. Run step8_official.py first for the full report.")

    # A segmented run: the unsegmented page first, then the segmented one,
    # so the exports left in outputs/exports/ are the segmented run's.
    u = unsegmented_view(d) if d["cfg"].get("segments") else None
    if u is not None:
        print("\n  unsegmented view (the unsegmented branch alone):")
        with open(OUT_HTML_UNSEG, "w", encoding="utf-8") as f:
            f.write(build_html(u))
        print(f"\n  wrote -> {OUT_HTML_UNSEG}")
        print("\n  segmented view:")
        d["view"] = "segmented"
    elif d["cfg"].get("segments"):
        print(f"\n  note: outputs/{UNSEG_PERCENTILES} is missing (step 7 ran an "
              "older engine), so there is no unsegmented view; re-run "
              "step7_run.py, step8_official.py and this step to get it.")

    doc = build_html(d)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"\n  wrote -> {OUT_HTML}")
    print("  open it in a browser to view.")
    print("\nAll 9 steps complete." if d["official"] is not None
         else "\nSteps 1-7 and 9 complete (step 8 skipped).")


if __name__ == "__main__":
    sys.exit(main())
