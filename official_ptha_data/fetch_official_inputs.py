"""Get everything needed to run a PTHA18 source zone, from official sources only.

One command takes a zone name and produces a ready-to-run input file, having
downloaded what it needs, extracted the official answer to compare against, and
checked at every step that what it has really is what PTHA18 used:

    .venv/Scripts/python.exe official_ptha_data/fetch_official_inputs.py puysegur2

The rule this script exists to enforce is the one from the guide: **if a number
exists in an official file, read it from there; if a rule computes it, implement
the rule; never type it.** Of the five errors found while building the original
puysegur2 comparison, four were hand-typed inputs and only one was in code.

What it does, in order:

  1. Download the saved PTHA18 session (1.34 GB, once) and the zone's
     per-unit-source geometry table (~15-300 KB, once per zone).
  2. Call R to extract the official logic tree and sourcepar from the session.
  3. Read the zone's row from sourcezone_parameters.csv.
  4. Recover the tectonic convergence rate (see `convergence_from_session`).
  5. Write input_<zone>.json, every field carrying a `_provenance` note.
  6. Verify the geometry and the recovered convergence against the session.

Steps 1 and 2 are skipped when their outputs already exist, so re-running is
cheap. Pass --force to redo them.

Requires R with rptha installed for step 2 only. Everything else is pure Python.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))  # pyptha_v12

THREDDS = "https://thredds.nci.org.au/thredds/fileServer/fj6/PTHA/AustPTHA_1"
SESSION_URL = f"{THREDDS}/EVENT_RATES/compute_rates_all_sources_session.RData"

SZP = os.path.join(ROOT, "rptha", "R", "examples", "austptha_template", "DATA",
                   "SOURCEZONE_PARAMETERS", "sourcezone_parameters.csv")

# Report defaults from EVENT_RATES/config.R. Identical for every zone, which is
# why they are constants here rather than per-zone inputs.
UNIFORM_COUPLING_RANGE = [0.1, 1.3]
GR_WEIGHTS = [0.7, 0.3]
MW_MIN = 7.15         # MW_MIN; 7.15 not 7.2 because bins are 0.1 wide
DMW = 0.1
N_BINS = {"coupling": 20, "b": 20, "Mw_max": 40}
MW_MAX_CAP = 9.6      # MAXIMUM_ALLOWED_MW_MAX


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def download(url, dest, what):
    """Fetch to a temporary name, then rename, so an interrupted download never
    leaves a truncated file that later steps would silently trust."""
    if os.path.exists(dest):
        mb = os.path.getsize(dest) / 1e6
        print(f"   have {what} ({mb:,.1f} MB) - skipping")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"   downloading {what}\n      {url}")
    try:
        subprocess.run(["curl", "-fL", "--progress-bar", "-o", tmp, url], check=True)
    except FileNotFoundError:
        raise SystemExit("curl not found; it is needed to fetch official data")
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        hint = ("\n  curl exit 22 means the server returned an error, usually a "
                "404: check the zone name, or the file may not be published for "
                "this zone." if e.returncode == 22 else "")
        raise SystemExit(f"download failed for {what} (curl exit "
                         f"{e.returncode}){hint}\n  {url}")
    os.replace(tmp, dest)
    print(f"   saved {dest} ({os.path.getsize(dest) / 1e6:,.1f} MB)")
    return dest


def geometry_path(zone):
    return os.path.join(ROOT, "inputs", "geometry",
                        f"unit_source_statistics_{zone}.nc")


SESSION_NAME = "compute_rates_all_sources_session.RData"
# Smallest the real session can plausibly be. A truncated or partial copy is
# worse than none: R fails deep inside the load with an unhelpful error.
SESSION_MIN_BYTES = 1_000_000_000


def session_candidates():
    """Where a copy of the session usually lives, in preference order.

    The first one, official_ptha_data/session/, is also where it is downloaded
    to when no copy is found.
    """
    return [
        os.path.join(HERE, "session", SESSION_NAME),
        os.path.join(ROOT, SESSION_NAME),
        os.path.join(os.path.dirname(ROOT), SESSION_NAME),
    ]


def find_session(explicit):
    """Resolve the session path, preferring a copy that already exists.

    An explicit --session that does not exist is an error, not a download
    target: silently starting a 1.34 GB fetch because of a typo or a relative
    path resolved from the wrong directory is the one behaviour to avoid.
    """
    if explicit:
        p = os.path.abspath(explicit)
        if os.path.exists(p):
            return p, True
        found = [c for c in session_candidates() if os.path.exists(c)]
        hint = ""
        if found:
            hint = ("\n  but a copy does exist at:\n    " + found[0] +
                    "\n  pass that path, or omit --session to use it "
                    "automatically")
        raise SystemExit(
            f"--session points at a file that does not exist:\n  {p}{hint}\n"
            "  (refusing to start a 1.34 GB download for a path you named "
            "explicitly)")

    for c in session_candidates():
        if os.path.exists(c):
            return c, True
    # nothing found: fall back to the in-package location as the download target
    return session_candidates()[0], False


def check_session_file(path):
    """Reject a truncated copy before R spends minutes failing on it."""
    size = os.path.getsize(path)
    if size < SESSION_MIN_BYTES:
        raise SystemExit(
            f"the session file looks truncated:\n  {path}\n"
            f"  {size / 1e6:,.1f} MB, expected about 1,401 MB\n"
            "  delete it (and any .part beside it) and let it download again, "
            "or point --session at a complete copy")


# ---------------------------------------------------------------------------
# R extraction
# ---------------------------------------------------------------------------
def find_rscript():
    for c in (shutil.which("Rscript"),
              r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"):
        if c and os.path.exists(c):
            return c
    for base in (r"C:\Program Files\R",):
        if os.path.isdir(base):
            for d in sorted(os.listdir(base), reverse=True):
                p = os.path.join(base, d, "bin", "Rscript.exe")
                if os.path.exists(p):
                    return p
    return None


def extract_official(session, outdir, zones, force=False):
    """Pull the official tree and sourcepar out of the session, via R.

    R is unavoidable here: the branches live inside a closure attached to each
    source environment and are only produced by calling it, which no Python
    RData reader can do.
    """
    need = [z for z in zones
            if force or not os.path.exists(
                os.path.join(outdir, f"logic_tree_branches_{z}_OFFICIAL.csv"))]
    if not need:
        print("   official trees already extracted - skipping")
        return
    rscript = find_rscript()
    if not rscript:
        raise SystemExit("Rscript not found; needed to read the saved session")
    env = dict(os.environ)
    lib = os.path.expanduser("~/R/win-library/4.6")
    if os.path.isdir(lib):  # pointing R at a missing folder hides its default library
        env.setdefault("R_LIBS_USER", lib)
    print(f"   extracting {', '.join(need)} (loading 1.34 GB, takes minutes)")
    subprocess.run([rscript, os.path.join(HERE, "extract_official_tree.R"),
                    session, outdir] + need, check=True, cwd=ROOT, env=env)


# ---------------------------------------------------------------------------
# Official sources
# ---------------------------------------------------------------------------
def read_sourcepar(outdir):
    scal, vec = {}, {}
    with open(os.path.join(outdir, "official_sourcepar_scalars.csv"),
              newline="") as f:
        for r in csv.DictReader(f):
            if r["value"] not in ("", "NA"):
                scal.setdefault(r["source"], {})[r["item"]] = float(r["value"])
    p = os.path.join(outdir, "official_sourcepar_vectors.csv")
    if os.path.exists(p):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                vec.setdefault(r["source"], {}).setdefault(
                    r["item"], []).append(float(r["value"]))
    return scal, vec


def read_gcmt_observations(outdir, zone):
    """The observed seismicity LEVEL 3 was run on, as extracted from the session.

    Returns None when the file predates this feature or holds nothing for the
    zone, in which case LEVEL 3 stays off and the caller says so. A zone with a
    genuine count of zero is NOT None: it is a real observation (no qualifying
    earthquake in 41 years) and still moves the weights, so it is returned with
    an empty magnitude list.
    """
    p = os.path.join(outdir, "official_gcmt_observations.csv")
    if not os.path.exists(p):
        return None
    mws, head = [], None
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] != zone:
                continue
            head = r
            if r["Mw"] not in ("", "NA"):
                mws.append(float(r["Mw"]))
    if head is None:
        return None
    return {"threshold_Mw": float(head["threshold_Mw"]),
            "count": int(float(head["count"])),
            "duration_years": float(head["duration_years"]),
            "observed_Mw": sorted(mws)}


def zone_row(name):
    """The unsegmented row for a zone: the one with no segment_name."""
    with open(SZP, newline="") as f:
        for r in csv.DictReader(f):
            if r["sourcename"].strip() == name and not r.get("segment_name", "").strip():
                return {k.strip(): (v.strip() if v else "") for k, v in r.items()}
    raise KeyError(f"{name} has no unsegmented row in sourcezone_parameters.csv")


def convergence_from_session(sourcepar):
    """Recover the tectonic convergence rate in mm/yr.

    Zones with `use_bird_convergence = 1` (most of them) have no published
    convergence rate: it is an area-weighted average over per-unit-source Bird
    (2003) velocity vectors that PTHA18 never released, and it appears in no
    file in this repository.

    It is still recoverable exactly, because compute_rates_all_sources.R line
    432 is

        sourcepar$slip = sourcepar$slip/cos_dip * 1/1000

    and the session stores both `slip` (after that line) and `cos_dip`. So

        convergence_mm_per_yr = slip * 1000 * cos_dip

    inverts it. That is one algebraic step on two saved numbers, not a fit.

    What licenses trusting it is puysegur2, whose convergence is independently
    known from sourcezone_parameters.csv to be 35 mm/yr because it carries
    `use_bird_convergence = 0`. The same inversion returns 35.0000000000 there.
    `verify()` re-runs that control on every invocation.
    """
    return sourcepar["slip"] * 1000.0 * sourcepar["cos_dip"]


# ---------------------------------------------------------------------------
# Input file
# ---------------------------------------------------------------------------
def build_input(zone, row, sourcepar, geo_rel, conv, conv_note, gcmt=None):
    cfg = {
        "_what_this_is": (
            f"{zone} unsegmented, on PTHA18's own per-unit-source statistics "
            "table. Every rate parameter is read from an official file or "
            "derived by an official rule; none is typed by hand. Generated by "
            "official_ptha_data/fetch_official_inputs.py - re-run that rather "
            "than editing this."),
        "_geometry_provenance": (
            f"OFFICIAL, verbatim. unit_source_statistics_{zone}.nc from NCI "
            f"THREDDS SOURCE_ZONES/{zone}/TSUNAMI_EVENTS/. Summing length*width "
            f"reproduces the session's sourcepar$area = {sourcepar['area']:.4f} "
            f"km2 and the angle-averaged dip reproduces sourcepar$mean_dip = "
            f"{sourcepar['mean_dip']:.7f} deg, both to machine precision. That "
            "is what confirms it is the geometry the official rates were "
            "computed on. The mesh cannot be rebuilt from parameters: PTHA18's "
            "contour and down-dip inputs are unpublished and were hand-edited."),
        "_convergence_provenance": conv_note,
        "_coupling_provenance": (
            f"READ from the sourcezone_parameters.csv row for {zone}: "
            f"cmin/cpref/cmax = {row['cmin']}/{row['cpref']}/{row['cmax']}. Do "
            "not substitute the report's generic [0.3, 0.5, 0.7] triple from "
            "Section 3.7.2.3; on puysegur2 that cost -12.7% on the mean slip "
            "rate while leaving the slip-rate axis itself exact."),
        "run_name": f"{zone}_official",
        "source_zone": zone,
        "geometry": {"mode": "official_statistics",
                     "official_statistics_nc": geo_rel},
        "events": {"Mmin": 7.2, "Mmax": MW_MAX_CAP, "dMw": DMW,
                   "source_zone_name": zone},
        "rates": {
            "tectonic_convergence_mm_per_yr": conv,
            "coupling": {
                "prior_type": "spreadsheet_and_uniform_50_50",
                "uniform_range": UNIFORM_COUPLING_RANGE,
                "spreadsheet_values": [float(row["cmin"]), float(row["cpref"]),
                                       float(row["cmax"])],
                "prob_zero_coupling": float(row["prob_Mmax_below_Mmin"] or 0.0),
            },
            "b_anchor": [float(row["bmin"]), float(row["bmax"])],
            "Mw_max_anchor": "derive",
            "Mw_min": [MW_MIN],
            "Mw_min_prob": [1.0],
            "Mw_frequency_distribution": ["truncated_gutenberg_richter",
                                          "characteristic_gutenberg_richter"],
            "Mw_frequency_distribution_prob": GR_WEIGHTS,
            "n_logic_tree_bins": N_BINS,
            "update_logic_tree_weights_with_data": False,
            "mw_max_posterior_equals_mw_max_prior": False,
            "edge_correction": {"mode": "fit", "lower": 0.0, "upper": 30.0},
            "account_for_moment_below_mwmin": True,
            "conditional_probability_model": "inverse_slip",
            "mw_max_observed": float(row["mw_max_observed"]),
            "scaling_relation": row["scaling_relation"],
        },
        "segments": {},
        "percentiles": {
            "threshold_Mw": [7.2, 7.6, 8.0, 8.4, 8.8, 9.2],
            "copula": "comonotonic", "N": 40000,
            "percentile_probs": [0.025, 0.16, 0.5, 0.84, 0.975], "seed": 123,
        },
        "outputs": {"write_branch_rate_curves": False},
        "_mw_max_anchor_provenance": (
            "DERIVED, never typed. The literal string 'derive' makes the runner "
            "apply the rule in compute_rates_all_sources.R (lines ~293-363): "
            "lower = max(mw_max_observed + 0.05, 7.35); upper = min(Strasser "
            "area bound at -1 SD, Strasser width bound at -2 SD). The upper "
            "anchor is deliberately NOT capped here - the driver interpolates "
            "the 40 slots first and applies pmin(..., 9.6) afterwards, under "
            "its own comment 'Clip AFTER interpolation' (line 358). Doing it in "
            "the other order changes which values the axis holds on any zone "
            "whose raw anchor exceeds 9.6, and cost -0.57% to -1.12% on the "
            "weighted mean Mw_max before it was found."),
        "_moment_below_mwmin_provenance": (
            "TRUE, matching compute_rates_all_sources.R line 634. It controls "
            "how much seismic moment the balance attributes to earthquakes "
            "below Mw_min, so it moves the derived 'a' in a way that depends on "
            "b and Mw_max but not on slip_rate."),
    }

    # --- LEVEL 3, when the session gave us the data it was run on ---
    if gcmt is not None:
        cfg["rates"]["update_logic_tree_weights_with_data"] = True
        cfg["rates"]["observed_seismicity"] = {
            "threshold_Mw": gcmt["threshold_Mw"],
            "count": gcmt["count"],
            "duration_years": gcmt["duration_years"],
            "observed_Mw": gcmt["observed_Mw"],
        }
        cfg["_observed_seismicity_provenance"] = (
            f"OFFICIAL, read from the saved PTHA18 session, not re-derived from "
            f"a catalogue. compute_rates_all_sources.R passes the GCMT subset to "
            f"rate_of_earthquakes_greater_than_Mw_function as Mw_count_duration "
            f"and Mw_obs_data (line ~631), so both live in the closure of the "
            f"stored mw_rate_function and come back out by reading its "
            f"environment. This zone carries {gcmt['count']} event(s) above Mw "
            f"{gcmt['threshold_Mw']:g} in {gcmt['duration_years']:.4f} years, "
            f"with the individual magnitudes listed. Because these are the very "
            f"numbers the published posterior was computed from, the LEVEL 3 "
            f"update reproduces the official result exactly rather than "
            f"plausibly - which re-running the catalogue selection of "
            f"gcmt_subsetter.R (buffer 0.4 deg, rake and strike within 50 deg, "
            f"depth <= 71 km) could not guarantee on its own. The duration is "
            f"the GCMT window 1976-01-01 to 2017-03-01.")
    else:
        cfg["_level_3_is_off_here"] = (
            "update_logic_tree_weights_with_data is false because no official "
            "GCMT observations were found for this zone in "
            "official_gcmt_observations.csv. Re-run the extraction with --force "
            "if that file predates the LEVEL 3 extraction; otherwise the "
            "posterior equals the prior by construction and must not be "
            "compared against the official posterior_prob column.")

    return cfg


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(zone, cfg, sourcepar, vec, scal_all):
    """Check the inputs against the session before anyone runs on them.

    Everything here is re-derived from the .nc file and the session rather than
    taken from what was just written, so a bug in this very script still fails
    the check.
    """
    import netCDF4
    from pyptha_v12 import moment_balance as mb

    fails = []

    def ok(cond, msg, detail=""):
        print(f"      {'PASS' if cond else 'FAIL'}  {msg}{detail}")
        if not cond:
            fails.append(msg)

    f = netCDF4.Dataset(geometry_path(zone))
    L = np.array(f["length"][:], dtype=float)
    W = np.array(f["width"][:], dtype=float)
    dip = np.radians(np.array(f["dip"][:], dtype=float))
    ast = np.array(f["alongstrike_number"][:])

    area = float((L * W).sum())
    mean_dip = float(np.degrees(np.arctan2(np.sin(dip).mean(), np.cos(dip).mean())))
    ra = abs(area - sourcepar["area"]) / sourcepar["area"]
    rd = abs(mean_dip - sourcepar["mean_dip"])
    ok(ra < 1e-12, "geometry area matches the session", f"  (rel {ra:.1e})")
    ok(rd < 1e-9, "geometry mean dip matches the session", f"  (abs {rd:.1e})")

    conv = cfg["rates"]["tectonic_convergence_mm_per_yr"]
    slip = conv / 1000.0 / sourcepar["cos_dip"]
    rs = abs(slip - sourcepar["slip"]) / sourcepar["slip"]
    ok(rs < 1e-12, "convergence reproduces the session's in-plane slip",
       f"  (rel {rs:.1e})")

    # The control that licenses the recovery on Bird-convergence zones.
    if "puysegur2" in scal_all:
        c = convergence_from_session(scal_all["puysegur2"])
        ok(abs(c - 35.0) < 1e-9,
           "control: the same inversion returns 35 mm/yr on puysegur2",
           f"  (got {c:.10f})")

    # The Mw_max rule, against the axis the session actually stored.
    lo, hi = mb.mw_max_anchor(sourcepar["area"], ast, W,
                              mw_max_observed=cfg["rates"]["mw_max_observed"],
                              scaling_relation=cfg["rates"]["scaling_relation"])
    axis = np.minimum(
        mb.interpolate_logic_tree_parameter([lo, hi], N_BINS["Mw_max"]), MW_MAX_CAP)
    off_axis = np.sort(np.asarray(vec.get("Mw_max", []), dtype=float))
    if off_axis.size == axis.size:
        e = float(np.abs(axis - off_axis).max())
        ok(e < 1e-9, "derived Mw_max axis matches the official axis",
           f"  (max abs {e:.1e}, {np.unique(off_axis).size}/{axis.size} distinct)")
    else:
        print("      SKIP  Mw_max axis not in the extracted sourcepar")

    return fails


# ---------------------------------------------------------------------------
def check_zone_name(zone):
    """Reject an unknown or retired zone before downloading anything.

    Without this the first sign of a typo is a curl 404 on a URL built from it,
    which reads like a server problem rather than a spelling mistake.
    """
    names = []
    with open(SZP, newline="") as f:
        for r in csv.DictReader(f):
            n = r["sourcename"].strip()
            if n and n not in names:
                names.append(n)
    if zone in names:
        # Present in the CSV is not the same as usable. PTHA18 retires a zone by
        # setting row_weight = 0 rather than deleting its row, and
        # compute_rates_all_sources.R then skips building its rate function
        # altogether, so the session holds an environment with no logic tree in
        # it. Caught here, the user is told in seconds rather than after a
        # ten-minute 1.34 GB load ending in "attempt to apply a non-function".
        row = zone_row(zone)
        if (row.get("row_weight") or "").strip() == "0":
            successor = zone + "2"
            hint = (f"\n  use {successor} instead" if successor in names else "")
            raise SystemExit(
                f"'{zone}' is a retired source zone (row_weight = 0 in "
                f"sourcezone_parameters.csv).\n"
                f"  PTHA18 supersedes a zone by zeroing its weight rather than "
                f"removing it, so every\n"
                f"  event on it has rate zero and no logic tree was ever built "
                f"for it. There is\n"
                f"  nothing to compare against.{hint}")
        return
    near = [n for n in names if zone.lower() in n.lower()
            or n.lower() in zone.lower()]
    msg = f"'{zone}' is not a source zone in sourcezone_parameters.csv"
    if near:
        msg += f"\n  did you mean: {', '.join(near)}"
    else:
        msg += f"\n  {len(names)} zones available, e.g. " + ", ".join(names[:8])
    raise SystemExit(msg)


def process(zone, session, outdir, args):
    print(f"\n=== {zone}")
    check_zone_name(zone)

    print("   [1/6] geometry table")
    download(f"{THREDDS}/SOURCE_ZONES/{zone}/TSUNAMI_EVENTS/"
             f"unit_source_statistics_{zone}.nc",
             geometry_path(zone), f"unit_source_statistics_{zone}.nc")

    print("   [2/6] official tree from the session")
    extract_official(session, outdir, [zone], force=args.force)

    print("   [3/6] zone parameters")
    row = zone_row(zone)
    scal, vec = read_sourcepar(outdir)
    if zone not in scal:
        raise SystemExit(f"{zone} has no sourcepar in {outdir}")
    sp = scal[zone]
    print(f"      cmin/cpref/cmax = {row['cmin']}/{row['cpref']}/{row['cmax']}, "
          f"b {row['bmin']}-{row['bmax']}, mw_max_observed {row['mw_max_observed']}, "
          f"use_bird_convergence = {row['use_bird_convergence']}")

    print("   [4/6] convergence rate")
    conv = convergence_from_session(sp)
    if row["use_bird_convergence"] == "1":
        note = ("RECOVERED from the saved session by inverting "
                "compute_rates_all_sources.R line 432: convergence = "
                f"sourcepar$slip ({sp['slip']!r}) * 1000 * sourcepar$cos_dip "
                f"({sp['cos_dip']!r}). This zone has use_bird_convergence=1, so "
                "its convergence is an area-weighted average of unpublished "
                "per-unit-source Bird (2003) vectors and cannot be read from "
                "any file. The inversion is validated on puysegur2, where it "
                "returns 35.0000000000 against a CSV value of 35.")
        print(f"      use_bird_convergence=1 -> recovered {conv:.6f} mm/yr "
              f"from the session")
    else:
        csv_conv = float(row["tectonic_slip"]) * float(row["convergent_fraction"])
        note = (f"READ from sourcezone_parameters.csv: tectonic_slip "
                f"{row['tectonic_slip']} x convergent_fraction "
                f"{row['convergent_fraction']} = {csv_conv}. The session "
                f"inversion independently returns {conv!r}, agreeing to "
                f"{abs(conv - csv_conv):.2e}, which is what validates the "
                f"inversion used on Bird-convergence zones.")
        print(f"      use_bird_convergence=0 -> read {csv_conv} mm/yr from the "
              f"CSV (session inversion agrees to {abs(conv - csv_conv):.1e})")
        conv = csv_conv

    gcmt = read_gcmt_observations(outdir, zone)
    if args.no_level3:
        gcmt = None
        print("      LEVEL 3: disabled by --no-level3")
    elif gcmt is None:
        print("      LEVEL 3: no official GCMT observations found - leaving off")
    else:
        print(f"      LEVEL 3: {gcmt['count']} official event(s) above Mw "
              f"{gcmt['threshold_Mw']:g} in {gcmt['duration_years']:.4f} yr"
              + (f", magnitudes {gcmt['observed_Mw'][0]:.3f}-"
                 f"{gcmt['observed_Mw'][-1]:.3f}" if gcmt["observed_Mw"]
                 else " (none above threshold)"))

    print("   [5/6] writing the input file")
    geo_rel = ("inputs/geometry/"
               f"unit_source_statistics_{zone}.nc")
    cfg = build_input(zone, row, sp, geo_rel, conv, note, gcmt=gcmt)
    os.makedirs(args.inputs_dir, exist_ok=True)
    dest = os.path.join(args.inputs_dir, f"input_{zone}.json")
    with open(dest, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"      {os.path.relpath(dest, ROOT)}")

    print("   [6/6] verifying against the session")
    return dest, verify(zone, cfg, sp, vec.get(zone, {}), scal)


def main():
    ap = argparse.ArgumentParser(
        description="Assemble PTHA18 inputs for one or more source zones from "
                    "official sources only.")
    ap.add_argument("zones", nargs="+",
                    help="source zone names, e.g. puysegur2 kermadectonga2")
    ap.add_argument("--session", default=None,
                    help="path to compute_rates_all_sources_session.RData. "
                         "If omitted, the usual locations are searched (see "
                         "find_session); only if none holds it is it downloaded.")
    ap.add_argument("--outdir", default=os.path.join(HERE, "trees"),
                    help="where extracted official trees and sourcepar go")
    # The package-root inputs/ folder, which is where BOTH engines read from.
    # Writing anywhere else produces a file the drivers cannot find.
    ap.add_argument("--inputs-dir", default=os.path.join(ROOT, "inputs"),
                    help="where the generated input JSON files go "
                         "(default: the inputs/ folder both engines read)")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the official tree already exists")
    ap.add_argument("--no-level3", action="store_true",
                    help="leave the LEVEL 3 Bayesian update off even when the "
                         "session's GCMT observations are available. The "
                         "posterior then equals the prior, which is what the "
                         "four shipped comparisons were built on.")
    args = ap.parse_args()

    session_path, already_have = find_session(args.session)
    args.session = session_path

    print("PTHA18 official input assembly")
    print(f"  session : {args.session}"
          + ("  [found]" if already_have else "  [will download, 1.34 GB]"))
    print(f"  official: {args.outdir}")
    print(f"  inputs  : {args.inputs_dir}")

    print("\n[0] saved PTHA18 session")
    download(SESSION_URL, args.session,
             "compute_rates_all_sources_session.RData (1.34 GB, once)")
    check_session_file(args.session)

    written, all_fails = [], {}
    for z in args.zones:
        dest, fails = process(z, args.session, args.outdir, args)
        written.append(dest)
        if fails:
            all_fails[z] = fails

    print("\n" + "=" * 68)
    for d in written:
        print(f"  wrote {os.path.relpath(d, ROOT)}")
    if all_fails:
        print("\nVERIFICATION FAILED:")
        for z, fs in all_fails.items():
            for f in fs:
                print(f"   {z}: {f}")
        sys.exit(1)
    print("\nall checks passed. Run a zone with:")
    z = args.zones[0]
    rel = os.path.relpath(os.path.join(args.inputs_dir, f"input_{z}.json"),
                          os.path.join(ROOT, "python_logic_tree"))
    print(f"   .venv/Scripts/python.exe python_logic_tree/run_logic_tree.py "
          f"{rel.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
