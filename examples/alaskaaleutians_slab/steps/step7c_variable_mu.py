"""STEP 7c (v11) - Variable shear modulus: PTHA18's second set of rates.

What it is
----------
Rigidity (shear modulus, mu) is how stiff the rock is. Steps 7 and 7b use
one value everywhere (30 GPa), like PTHA18's "constant shear modulus"
results. Real rock near the trench is softer and deeper rock stiffer, and
for the same earthquake (the same seismic moment, which is what
seismometers measure) soft rock needs more slip, so a shallow rupture makes
a bigger tsunami than its constant-rigidity magnitude suggests. PTHA18
publishes a second set of rates for that ("variable_mu_rate_annual"), and
its headline hazard maps use it with the HS scenarios (PTHA18 report
Section 4.1).

How PTHA18 does it (report Section 3.7.5), ported in
pyptha_v12/variable_mu.py and from_scratch_v12/python_logic_tree_v12/run_logic_tree.py:

1. Rigidity at each unit source's depth: rptha's shear_modulus_depth, a fit
   to Bilek & Lay (1999): 10 GPa down to 7.5 km, 30 GPa at 15 km, 67 GPa
   from 35 km.
2. Every scenario keeps its ruptures and slip; only its magnitude is
   "relabelled": M0 = sum(area x slip x mu(depth)). Shallow scenarios get a
   lower magnitude, deep ones a higher one.
3. The rates stay functions of the constant-rigidity magnitude, but the
   GCMT earthquakes of LEVEL 3 have real (variable-rigidity) magnitudes.
   The difference, measured on this run's own HS scenarios (step 7b), is
   treated as an observation error of the catalogue magnitudes, which
   changes the LEVEL 3 weights of the logic-tree branches.
4. The variable shear modulus scenario rates are the same branch curves
   averaged with those weights. HS and VAUS take a share of them with
   PTHA18's DART curves for the variable shear modulus case.

Checked against PTHA18's published files (validation/validate_v11.py):
every published variable_mu_Mw of puysegur2 and kermadectonga2 (FAUS, HS,
VAUS) to 4e-15; the engine on PTHA18's puysegur2 input with PTHA18's HS
catalogue gives its published variable_mu_rate_annual and 5 percentiles
to 7e-13; HS and VAUS variable_mu_rate_annual of both zones to 4e-12.

What it does
------------
- needs step 7 (outputs/scenario_rates_alaskaaleutians.csv) and step 7b
  (outputs/hs_slip_fields/summary.csv, the HS scenarios)
- writes inputs/input_alaskaaleutians_scratch_variable_mu.json (step 6's input plus
  a variable_shear_modulus block) and runs the engine on it
- checks that the engine's constant shear modulus outputs are identical to
  step 7's (they must be: only extra outputs are added)
- copies into outputs/ the variable shear modulus files:
    scenario_rates_alaskaaleutians_variable_mu.csv      every FAUS scenario: its
                                                variable_mu_Mw and rates
    rate_curves_variable_mu.csv, exceedance_rate_percentiles_variable_mu.csv
    variable_mu_deviation_alaskaaleutians.csv           the HS magnitude differences
    logic_tree_branches_alaskaaleutians_variable_mu.csv  (posterior_prob_with_Mw_error
                                                = the variable shear modulus
                                                weights)
    logic_tree_summary_variable_mu.csv
- adds to both outputs/hs_slip_fields/summary.csv and
  outputs/vaus_slip_fields/summary.csv: variable_mu_Mw,
  variable_mu_weight_in_family and variable_mu_rate_* (the parent FAUS
  rupture's variable shear modulus rates times that weight)

Nothing step 7 or 7b wrote changes. Zones whose unit sources are normal
faults keep the constant rigidity, as in PTHA18.

Run from ptha18_logic_tree_test/, after step7b_stochastic_slip.py:
    .venv/Scripts/python.exe examples/alaskaaleutians_slab/steps/step7c_variable_mu.py
"""

import csv
import filecmp
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXAMPLE, "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "lib"))
sys.path.insert(0, os.path.join(ROOT, "from_scratch_v12", "python_logic_tree_v12"))
from pyptha_v12 import hs_vaus_rates  # noqa: E402
from pyptha_v12 import unit_sources as us  # noqa: E402
from pyptha_v12 import variable_mu as vmu  # noqa: E402
from run_logic_tree import build_grid  # noqa: E402

ZONE = "alaskaaleutians"
# generate.py --variable-mu on|off; off: this step only runs with --force
VARIABLE_MU = False
RUNNER = os.path.join(ROOT, "from_scratch_v12", "python_logic_tree_v12", "run_logic_tree.py")
INPUT_JSON = os.path.join(EXAMPLE, "inputs", f"input_{ZONE}_scratch.json")
INPUT_MU_JSON = os.path.join(EXAMPLE, "inputs", f"input_{ZONE}_scratch_variable_mu.json")
OUT = os.path.join(EXAMPLE, "outputs")
HS_SUMMARY = os.path.join(OUT, "hs_slip_fields", "summary.csv")
VAUS_SUMMARY = os.path.join(OUT, "vaus_slip_fields", "summary.csv")


def add_variable_mu_rates(path, kind, faus_mu, stats, mu, relation):
    """Variable shear modulus Mw and rates of every HS or VAUS field, added
    to its summary.csv in place (rows and existing columns untouched)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return f"{kind}: no fields"
    if "faus_event_id" not in rows[0]:
        return (f"{kind}: no rates (summary.csv has no faus_event_id; re-run "
                f"step7b_stochastic_slip.py)")
    parent = np.array([int(r["faus_event_id"]) - 1 for r in rows])
    # peak slip as PTHA18 reads it: the largest value of event_slip_string
    peak = np.array([vmu.parse_slip_string(r["event_slip_string"]).max() for r in rows])
    parent_mw = np.array([faus_mu["Mw"][i] for i in parent])
    weight, _ = hs_vaus_rates.family_weights(
        parent, peak, parent_mw, kind, mu=mu, relation=relation, variable_mu=True)
    area = np.asarray(stats["length"], dtype=float) * np.asarray(stats["width"], dtype=float)
    mw_mu = vmu.variable_mu_Mw([r["event_index_string"] for r in rows],
                               [r["event_slip_string"] for r in rows],
                               area, np.asarray(stats["depth"], dtype=float))
    rate_cols = [c for c in faus_mu["columns"] if c.startswith("rate_")]
    for r, i, w, m in zip(rows, parent, weight, mw_mu):
        r["variable_mu_Mw"] = f"{m:.4f}"
        r["variable_mu_weight_in_family"] = f"{w:.6g}"
        for c in rate_cols:
            r[f"variable_mu_{c}"] = f"{w * faus_mu[c][i]:.6g}"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    shift = mw_mu - np.array([float(r["Mw"]) for r in rows])
    return (f"{kind}: {len(rows):,} fields, Mw (variable - constant) "
            f"{shift.min():+.2f} .. {shift.max():+.2f}")


def main():
    print("=" * 70)
    print("STEP 7c (v11) - Variable shear modulus (PTHA18's variable_mu rates)")
    print("=" * 70)
    if not VARIABLE_MU and "--force" not in sys.argv[1:]:
        print("\n  This example was generated with --variable-mu off (constant "
              "rigidity only), so nothing is done. Pass --force to compute the "
              "variable shear modulus rates anyway.")
        return 0
    need = {INPUT_JSON: "step6_write_input.py",
            os.path.join(OUT, f"scenario_rates_{ZONE}.csv"): "step7_run.py",
            HS_SUMMARY: "step7b_stochastic_slip.py"}
    for path, step in need.items():
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}\nRun {step} first.")

    with open(INPUT_JSON, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["run_name"] = cfg["run_name"] + "_variable_mu"
    cfg["variable_shear_modulus"] = {
        "hs_events": os.path.relpath(HS_SUMMARY, ROOT).replace(os.sep, "/"),
        "curve": "default",
    }
    with open(INPUT_MU_JSON, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    input_rel = os.path.relpath(INPUT_MU_JSON, ROOT).replace(os.sep, "/")
    print(f"\n  [1/3] engine with the variable shear modulus: "
          f"{os.path.relpath(RUNNER, ROOT)} {input_rel}\n")
    result = subprocess.run([sys.executable, RUNNER, input_rel], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"\nrun_logic_tree.py exited with code {result.returncode}")
    run_dir = os.path.join(ROOT, "runs", "python", cfg["run_name"])

    print("\n  [2/3] outputs")
    same = filecmp.cmp(os.path.join(run_dir, f"scenario_rates_{ZONE}.csv"),
                       os.path.join(OUT, f"scenario_rates_{ZONE}.csv"), shallow=False)
    print(f"      constant shear modulus scenario rates identical to step 7's: {same}")
    if not same:
        raise SystemExit("the engine's constant shear modulus results differ from "
                         "step 7's: re-run step7_run.py and step 7b first")
    copies = {f: f for f in os.listdir(run_dir) if "variable_mu" in f and f.endswith(".csv")}
    copies[f"logic_tree_branches_{ZONE}.csv"] = f"logic_tree_branches_{ZONE}_variable_mu.csv"
    copies["logic_tree_summary.csv"] = "logic_tree_summary_variable_mu.csv"
    for src, dst in sorted(copies.items()):
        shutil.copyfile(os.path.join(run_dir, src), os.path.join(OUT, dst))
        print(f"      {dst}")
    # the engine's staging folder is no longer needed (and runs/ with it, when
    # nothing else is in there: step 8 keeps the official run in runs/python/)
    shutil.rmtree(run_dir)
    for parent in (os.path.dirname(run_dir), os.path.join(ROOT, "runs")):
        try:
            os.rmdir(parent)
        except OSError:
            pass

    print("\n  [3/3] HS and VAUS variable shear modulus rates")
    with open(os.path.join(OUT, f"scenario_rates_{ZONE}_variable_mu.csv"), newline="") as fh:
        faus_rows = list(csv.DictReader(fh))
    faus_mu = {"columns": list(faus_rows[0].keys())}
    for c in faus_mu["columns"]:
        faus_mu[c] = np.array([float(r[c]) for r in faus_rows])
    faus_mu["Mw"] = np.round(faus_mu["Mw"], 3)
    stats = us.discretized_source_approximate_summary_statistics(build_grid(cfg["geometry"]))
    ev = cfg["events"]
    mu, relation = float(ev.get("shear_modulus_Pa", 3e10)), ev.get("scaling_relation", "Strasser")
    for path, kind in ((HS_SUMMARY, "HS"), (VAUS_SUMMARY, "VAUS")):
        if os.path.exists(path):
            print("      " + add_variable_mu_rates(path, kind, faus_mu, stats, mu, relation))

    print("\n  Steps 7 and 7b's constant shear modulus results are unchanged.")
    print("\nNext:  step8_official.py (optional) or step9_report.py")


if __name__ == "__main__":
    sys.exit(main())
