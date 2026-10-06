"""STEP 7 - Run the full logic tree: LEVELs 0 through 5.

Everything built in steps 1-6 becomes one call into
from_scratch_v12/python_logic_tree_v12/run_logic_tree.py, the same engine step 8 runs on
PTHA18's official input (see from_scratch_v12/html/docs/code_map.html).
Nothing else is computed by this script; it exists to invoke the runner
correctly and then point you at what it wrote.

What the runner does with the input from step 6, in order
-----------------------------------------------------------
  LEVEL 0  segmentation           unsegmented only, or (--segmented true)
                                   0.5 unsegmented + 0.5 union of segments,
                                   each segment on the zone's scenario table
  LEVEL 1  logic-tree axes        coupling x b x Mw_max x GR-type
  LEVEL 2  moment balance         solves the GR 'a' parameter per branch --
                                   THIS is where the rate curves are actually
                                   born, from geometry + convergence + coupling
  LEVEL 3  Bayesian update        re-weights the branches using step 5's GCMT
                                   subset (prior -> posterior); does not touch
                                   the rate curves themselves
  LEVEL 4  edge correction        fits how rupture probability tapers at the
                                   along-strike edges of the zone
  LEVEL 5  epistemic percentiles  Monte Carlo over branch weights to get
                                   exceedance-rate percentiles, not just a mean

v10 / v10_q: the input's events.rupture_size (step 6, generate.py
--rupture-size) picks how the uniform-slip ruptures are built before LEVEL 2:
"rptha" (rptha's rule, one cell count per magnitude) or "local" (each
rupture sized from the real km of its own cells, pyptha_v12/events.py
_local_blocks). With "local" the engine also multiplies every rupture's
conditional-probability weight by q (pyptha_v12/events.py coverage_weights),
so places covered by more ruptures do not get more rate. Neither changes the
rate of each magnitude (rate_curves.csv, the exceedance tables, the branches:
each rupture's slip x area x rigidity is its magnitude's moment, so the
moment balance cannot see the shapes); they change every scenario's area,
slip and rate, the conditional probabilities, the integrated slip and the
fitted edge multiplier. The log line "n_events = N (rupture size: ...)" says
which rule ran. Not PTHA18's procedure with "local" (README, Known
limitations (v10_q)).

Output
------
run_logic_tree.py always writes to its own runs/python/<run_name>/, relative
to ptha18_logic_tree_test/ (both the shared engine and v5's fork resolve
that same root, two directories up from run_logic_tree.py) -- that path is
not an argument, it is fixed in the engine, and this script does not patch
it to keep it identical to every official run. So this step runs the
engine, then COPIES the result into examples/makran2/outputs/, which is where
everything else in this example lives:

  outputs/
    rate_curves.csv                 mean exceedance rate vs Mw
    exceedance_rate_percentiles.csv mean + percentile curves vs Mw
    logic_tree_branches_*.csv       every branch: its parameters, prior and
                                     posterior weight
    logic_tree_summary.csv          per-level summary numbers (area, mean dip,
                                     convergence, edge multiplier, ...)
    segmentation.csv                LEVEL 0 weights
    scenario_rates_*.csv            every scenario's rate (mean, percentiles)
  and, on a segmented run, the same per-representation files for every
  segment (<zone>_<segment>) plus scenario_rates_<zone>_source_zone.csv:
  every scenario's rate for the zone as a whole (0.5 x unsegmented + 0.5 x
  each segment it touches, PTHA18's partial-segmentation percentiles).

The engine's own copy in runs/python/<run_name>/ (its fixed location; v8
names it <zone>_scratch_<folder>, so two example folders of the same zone no
longer overwrite each other there) is only a staging area: once outputs/ holds
the same files this step deletes it, and runs/ with it when nothing else is in
there. Re-running this step replaces outputs/ as a whole, including step 7b's
results (re-run step 7b after it).

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/makran2/steps/step7_run.py
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))  # ptha18_logic_tree_test/

# v5's own forked engine (with the 3 fidelity fixes -- see
# from_scratch_v12/html/docs/code_map.html), NOT the shared python_logic_tree/ every other
# version and step8's official comparison use.
RUNNER = os.path.join(ROOT, "from_scratch_v12", "python_logic_tree_v12", "run_logic_tree.py")
INPUT_JSON = os.path.join(EXAMPLE, "inputs", "input_makran2_scratch.json")
EXAMPLE_OUT_DIR = os.path.join(EXAMPLE, "outputs")


def engine_out_dir():
    """Where run_logic_tree.py writes: runs/python/<run_name>, with run_name
    read from the input JSON the engine itself reads (v8; v7 hard-coded
    "<zone>_scratch" here, so a hand-edited run_name made this step copy the
    wrong folder)."""
    import json
    with open(INPUT_JSON, encoding="utf-8") as fh:
        run_name = json.load(fh)["run_name"]
    return os.path.join(ROOT, "runs", "python", run_name)


def remove_engine_copy(out_dir):
    """Drop the engine's staging folder (outputs/ has the same files now), and
    runs/python/ and runs/ too if that leaves them empty (step 8 puts the
    official run there, so they stay when it has run)."""
    shutil.rmtree(out_dir)
    for parent in (os.path.dirname(out_dir), os.path.join(ROOT, "runs")):
        try:
            os.rmdir(parent)
        except OSError:
            pass


def main():
    print("=" * 70)
    print("STEP 7 - Run the full logic tree (LEVELs 0-5)")
    print("=" * 70)

    if not os.path.exists(INPUT_JSON):
        raise SystemExit(f"missing {INPUT_JSON}\nRun step6_write_input.py first.")
    if not os.path.exists(RUNNER):
        raise SystemExit(f"missing {RUNNER}")

    # run_logic_tree.py resolves a relative input path from its own directory
    # (ROOT in its own code, i.e. ptha18_logic_tree_test/), and everything the
    # input JSON itself references (the shapefile) is written relative to that
    # same root by step6 -- so pass the path relative to ROOT, not an absolute
    # one, to match exactly how the official runs are invoked.
    input_rel = os.path.relpath(INPUT_JSON, ROOT).replace(os.sep, "/")

    print(f"\n  invoking: {os.path.relpath(RUNNER, ROOT)} {input_rel}\n")
    result = subprocess.run(
        [sys.executable, RUNNER, input_rel], cwd=ROOT)

    if result.returncode != 0:
        raise SystemExit(f"\nrun_logic_tree.py exited with code "
                         f"{result.returncode}")

    out_dir = engine_out_dir()
    print(f"\n  engine wrote -> {out_dir}  (its own fixed location)")

    # outputs/ is replaced as a whole, including any step 7b (HS/VAUS)
    # results: those were built on the previous run's event table, which
    # may no longer be this run's. Re-run step7b_stochastic_slip.py after
    # this step if you want them.
    if os.path.isdir(EXAMPLE_OUT_DIR):
        shutil.rmtree(EXAMPLE_OUT_DIR)
    shutil.copytree(out_dir, EXAMPLE_OUT_DIR)
    print(f"  copied      -> {EXAMPLE_OUT_DIR}  (this example's own copy)")
    remove_engine_copy(out_dir)
    print(f"  removed the engine's staging folder {out_dir}")

    print("\n  For a comparison against the official run, see step8_official.py")
    print("  (it needs R with rptha installed; run it, then re-run step9_report.py)")
    print("\nAll 7 steps complete.")


if __name__ == "__main__":
    sys.exit(main())
