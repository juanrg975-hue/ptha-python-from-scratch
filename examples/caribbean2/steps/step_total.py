"""STEP_TOTAL - Run the entire Caribbean pipeline in one go.

Runs steps 1-9 in order, each as its own subprocess, exactly as if you had
typed them one at a time. Nothing here duplicates their logic; this script
only sequences them and stops at the first failure so you can see exactly
which step needs attention.

Step 7b and step 8 are treated as OPTIONAL. Step 7b generates illustrative
heterogeneous-slip (HS) and variable-area-uniform-slip (VAUS) variants of
every one of step 7's FAUS placements -- neither feeds back into the logic
tree, so a failure there does not affect any rate number. Step 8 needs R
with rptha installed, and needs "antilles2" to actually be a zone PTHA18
modelled. If either fails, this script prints why and moves on -- step 9's
report is still complete without them (minus the HS/VAUS section and/or
the official comparison section). Every other step is required: if any of
them fails, this script stops there rather than pressing on with stale or
missing inputs.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/caribbean2/steps/step_total.py

Add --skip-official to skip step 8 outright, e.g. if you already know R is
not installed and do not want to see it fail:
    .venv/Scripts/python.exe examples/caribbean2/steps/step_total.py --skip-official

Add --skip-hs to skip step 7b outright:
    .venv/Scripts/python.exe examples/caribbean2/steps/step_total.py --skip-hs

Add --auto-clip if the report from a previous run warned that this mesh runs
past its own plate boundary (step 3's check). It re-runs the whole pipeline
with step 1 clipped to the part of the raster that belongs to this zone,
using the window step 3 derived from the mesh -- you do not supply any
lon/lat. It needs steps 1-3 to have run once first (a full run also does),
because the finding is made from the mesh (data/plate_boundary_change.json):
    .venv/Scripts/python.exe examples/caribbean2/steps/step1_fetch_slab2.py
    .venv/Scripts/python.exe examples/caribbean2/steps/step2_build_grid.py
    .venv/Scripts/python.exe examples/caribbean2/steps/step3_convergence.py
    .venv/Scripts/python.exe examples/caribbean2/steps/step_total.py --auto-clip

Because --auto-clip rebuilds the mesh with fewer columns, it also passes
--force to step 6, so the input JSON is regenerated for the new geometry.
Any hand edit you made to that JSON is therefore DISCARDED by an --auto-clip
run; re-apply it afterwards if you had one.

Add --download-official-hs-vaus to have step 8 fetch PTHA18's own
published HS and VAUS catalogues for this zone (from NCI THREDDS) and
compare step 7b's fields against them. Off by default -- these files can be
large and are not needed for anything except that one comparison section of
the report:
    .venv/Scripts/python.exe examples/caribbean2/steps/step_total.py --download-official-hs-vaus
"""

import argparse
import html
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# v11: generate.py --variable-mu on|off (default off). Off: step 7c (PTHA18's
# variable shear modulus rates) is not run; every rate uses 30 GPa.
VARIABLE_MU = False

STEPS = [
    ("step1_fetch_slab2.py", True),
    ("step2_build_grid.py", True),
    ("step3_convergence.py", True),
    ("step4_fetch_gcmt.py", True),
    ("step5_subset_gcmt.py", True),
    ("step6_write_input.py", True),
    ("step7_run.py", True),
    ("step7b_stochastic_slip.py", False),  # optional: illustrative HS + VAUS
    ("step7c_variable_mu.py", False),  # v11, optional: needs 7b's HS scenarios
    ("step8_official.py", False),  # optional: needs R + rptha
    ("step9_report.py", True),
]


RUN_HTML = os.path.join(os.path.dirname(HERE), "RUN.html")


def record_run(elapsed_min):
    """Add this run's command line to the "Commands used" part of RUN.html, so
    the page lists the flags every full run was really given."""
    marker = "<!-- RUNS:END -->"
    try:
        with open(RUN_HTML, encoding="utf-8") as fh:
            text = fh.read()
        if marker not in text:
            return
        cmd = " ".join(["python", "examples/caribbean2/steps/step_total.py"] + sys.argv[1:])
        line = ('<div class="cmd"><code><span class="p">$ </span>' + html.escape(cmd)
                + '</code></div>\n  <p class="sec-note" style="margin-bottom:10px">'
                + time.strftime("%Y-%m-%d %H:%M") + ", " + f"{elapsed_min:.1f}"
                + " min</p>\n  ")
        with open(RUN_HTML, "w", encoding="utf-8") as fh:
            fh.write(text.replace(marker, line + marker))
    except OSError as exc:
        print(f"  (could not record this run in RUN.html: {exc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-official", action="store_true",
                    help="skip step8_official.py entirely, without trying it")
    ap.add_argument("--skip-hs", action="store_true",
                    help="skip step7b_stochastic_slip.py (and v11's "
                         "step7c_variable_mu.py, which needs its HS "
                         "scenarios) entirely, without trying them")
    ap.add_argument("--auto-clip", action="store_true",
                    help="pass --auto-clip to step 1, so it cuts the "
                         "along-strike columns a previous run's step 3 "
                         "flagged as sitting on a DIFFERENT plate boundary. "
                         "Needs one full run first: step 3 makes the finding "
                         "from the mesh, and this re-runs the whole pipeline "
                         "on the clipped geometry.")
    ap.add_argument("--download-official-hs-vaus", action="store_true",
                    help="pass --download-hs through to step8_official.py, "
                         "so it fetches PTHA18's own published HS/VAUS "
                         "catalogues for this zone and compares step 7b's "
                         "fields with them (off by default)")
    args = ap.parse_args()

    print("#" * 70)
    print(f"# Caribbean -- full pipeline (steps 1-9)")
    print("#" * 70)

    t_start = time.time()
    for fname, required in STEPS:
        if fname == "step8_official.py" and args.skip_official:
            print(f"\n>>> skipping {fname} (--skip-official)")
            continue
        if fname == "step7c_variable_mu.py" and not VARIABLE_MU:
            print(f"\n>>> skipping {fname} (generated with --variable-mu off: "
                  f"constant rigidity only)")
            continue
        if fname in ("step7b_stochastic_slip.py", "step7c_variable_mu.py") and args.skip_hs:
            print(f"\n>>> skipping {fname} (--skip-hs)")
            continue

        path = os.path.join(HERE, fname)
        cmd = [sys.executable, path]
        if fname == "step8_official.py" and args.download_official_hs_vaus:
            cmd.append("--download-hs")
        if fname == "step1_fetch_slab2.py" and args.auto_clip:
            cmd.append("--auto-clip")
        # --auto-clip deliberately REBUILDS the mesh with fewer along-strike
        # columns, so the input JSON written for the old mesh no longer fits
        # it: its convergent_slip_profile still has the old column count and
        # step 7 refuses to run. Step 6 never overwrites an existing JSON on
        # its own (that is what protects hand edits), so it has to be told
        # to here -- the geometry changing IS the instruction to regenerate.
        if fname == "step6_write_input.py" and args.auto_clip:
            cmd.append("--force")
        print(f"\n>>> {fname}")
        print("-" * 70, flush=True)
        # Unbuffered (v8), so that when the output goes to a log file each
        # step's own lines and those of the programs it starts (the engine,
        # R) appear in the order they were printed.
        result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})

        if result.returncode != 0:
            if required:
                raise SystemExit(
                    f"\n{fname} failed (exit {result.returncode}); stopping "
                    f"here. Fix the error above, then either re-run "
                    f"step_total.py (it will redo earlier steps too, which "
                    f"is harmless since they cache their downloads) or run "
                    f"the remaining steps one at a time from {fname} on.")
            if fname == "step7c_variable_mu.py":
                print(f"\n{fname} failed (exit {result.returncode}) -- this "
                      f"step is optional (v11 variable shear modulus rates, "
                      f"added next to the constant ones), so continuing. The "
                      f"report will show the constant shear modulus results "
                      f"only.")
            elif fname == "step7b_stochastic_slip.py":
                print(f"\n{fname} failed (exit {result.returncode}) -- this "
                     f"step is optional (illustrative HS/VAUS variants only, "
                     f"never fed back into the logic tree), so continuing "
                     f"to the next step. The final report will be complete "
                     f"for LEVELs 0-5, just without an HS/VAUS section.")
            else:
                print(f"\n{fname} failed (exit {result.returncode}) -- this step "
                     f"is optional (needs R + rptha, and needs this zone to be "
                     f"one PTHA18 modelled), so continuing to the next step. "
                     f"The final report will be complete for steps 1-7, just "
                     f"without an official-run comparison.")

    elapsed = time.time() - t_start
    print("\n" + "#" * 70)
    print(f"# Done in {elapsed/60:.1f} min. See examples/caribbean2/report.html")
    print("#" * 70)
    record_run(elapsed / 60)


if __name__ == "__main__":
    sys.exit(main())
