"""v11 checks of the variable shear modulus port against PTHA18's own files.

  mu_Mw     every published variable_mu_Mw of puysegur2 and kermadectonga2
            (FAUS, HS and VAUS) from PTHA18's unit-source table with
            pyptha_v12.variable_mu.variable_mu_Mw (the Bilek & Lay curve on
            each unit source's depth).
  engine    the v11 engine on PTHA18's official puysegur2 input, with the
            variable shear modulus built from PTHA18's own HS catalogue,
            against PTHA18's published variable_mu_rate_annual and its 5
            percentiles (and the constant rate_annual, unchanged).
  hs_vaus   recover_peak_slip_weights.py: PTHA18's 4 DART weight curves
            (constant / variable shear modulus x HS / VAUS) recovered from
            its published rates, and every published HS and VAUS rate of
            both zones, both shear modulus treatments, recomputed.

Needs, from ptha18_logic_tree_test/: official_ptha_data/public_nc/ (PTHA18's
all_{uniform,stochastic,variable_uniform}_slip_earthquake_events_<zone>.nc),
inputs/geometry/unit_source_statistics_<zone>.nc and inputs/input_puysegur2.json
(PTHA18's official input, extracted once with
official_ptha_data/fetch_official_inputs.py). Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe V9/from_scratch_v12/validation/validate_v11.py [mu_Mw engine hs_vaus]
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
from pyptha_v12 import variable_mu as vmu  # noqa: E402

# the package may sit in ptha18_logic_tree_test/ or one level down (V9/)
BASES = [os.path.abspath(os.path.join(PKG, os.pardir)),
         os.path.abspath(os.path.join(PKG, os.pardir, os.pardir))]


def find(rel):
    for b in BASES:
        p = os.path.join(b, rel)
        if os.path.exists(p):
            return p
    raise SystemExit(f"not found in {BASES}: {rel}")


def check_mu_Mw(zones=("puysegur2", "kermadectonga2")):
    import netCDF4 as nc
    print("\nMU_MW: PTHA18's published variable_mu_Mw from its unit-source table")
    ok = True
    for z in zones:
        with nc.Dataset(find(f"inputs/geometry/unit_source_statistics_{z}.nc")) as u:
            u.set_auto_mask(False)
            area = u["length"][:] * u["width"][:]
            depth = u["depth"][:].astype(float)
        for kind in ("uniform", "stochastic", "variable_uniform"):
            with nc.Dataset(find(f"official_ptha_data/public_nc/all_{kind}_slip_earthquake_events_{z}.nc")) as f:
                f.set_auto_mask(False)
                eis = [s.strip() for s in nc.chartostring(f["event_index_string"][:])]
                slips = (f["slip"][:].astype(float) if kind == "uniform" else
                         [s.strip() for s in nc.chartostring(f["event_slip_string"][:])])
                off = f["variable_mu_Mw"][:].astype(float)
            d = float(np.max(np.abs(vmu.variable_mu_Mw(eis, slips, area, depth) - off)))
            good = d < 1e-9
            ok &= good
            print(f"  {z:15s} {kind:17s} {len(eis):6d} scenarios  max|diff| {d:.1e}  "
                  f"{'PASS' if good else 'FAIL'}")
    return ok


def check_engine(z="puysegur2"):
    import netCDF4 as nc
    import pandas as pd
    print(f"\nENGINE: v11 variable shear modulus rates on PTHA18's {z} input vs its "
          f"published variable_mu_rate_annual")
    root = os.path.dirname(os.path.dirname(find(f"inputs/input_{z}.json")))
    cfg = json.load(open(os.path.join(root, "inputs", f"input_{z}.json")))
    cfg["run_name"] = f"{z}_v11_variable_mu_validation"
    cfg["geometry"]["official_statistics_nc"] = os.path.join(
        root, cfg["geometry"]["official_statistics_nc"])
    # sourcezone_parameters.csv: Strasser, 30 GPa for puysegur2
    cfg["events"].update({"scaling_relation": "Strasser", "shear_modulus_Pa": 3.0e10})
    cfg["variable_shear_modulus"] = {"hs_events": find(
        f"official_ptha_data/public_nc/all_stochastic_slip_earthquake_events_{z}.nc")}
    tmp = os.path.join(HERE, f"_input_{z}_v11_mu.json")
    json.dump(cfg, open(tmp, "w"), indent=1)
    runner = os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py")
    try:
        subprocess.run([sys.executable, runner, tmp], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)
    finally:
        os.remove(tmp)
    engine_root = os.path.abspath(os.path.join(PKG, os.pardir))  # the engine's ROOT
    out = os.path.join(engine_root, "runs", "python", cfg["run_name"])
    p = pd.read_csv(os.path.join(out, f"scenario_rates_{z}_variable_mu.csv"))
    pc = pd.read_csv(os.path.join(out, f"scenario_rates_{z}.csv"))
    cols = {"variable_mu_rate_annual": "rate_mean",
            "variable_mu_rate_annual_lower_ci": "rate_p0.025",
            "variable_mu_rate_annual_16pc": "rate_p0.16",
            "variable_mu_rate_annual_median": "rate_p0.5",
            "variable_mu_rate_annual_84pc": "rate_p0.84",
            "variable_mu_rate_annual_upper_ci": "rate_p0.975"}
    with nc.Dataset(find(f"official_ptha_data/public_nc/all_uniform_slip_earthquake_events_{z}.nc")) as ds:
        ds.set_auto_mask(False)
        off = {k: np.asarray(ds[k][:], float) for k in list(cols) + ["rate_annual", "variable_mu_Mw"]}
    n = len(p)
    ok = True
    a = off["rate_annual"][:n]
    rel_c = float(np.max(np.abs(pc.rate_mean - a)[a > 0] / a[a > 0]))
    ok &= rel_c < 1e-9
    print(f"  constant rate_annual (unchanged)       max rel diff {rel_c:.1e}")
    d_mw = float(np.max(np.abs(p.variable_mu_Mw - off["variable_mu_Mw"][:n])))
    ok &= d_mw < 1e-9
    print(f"  variable_mu_Mw                         max |diff|   {d_mw:.1e}")
    for k, c in cols.items():
        a, b = off[k][:n], p[c].to_numpy()
        nz = a > 0
        rel = float(np.max(np.abs(b - a)[nz] / a[nz]))
        zm = int(np.sum((a == 0) != (b == 0)))
        good = rel < 1e-9 and zm == 0
        ok &= good
        print(f"  {k:38s} max rel diff {rel:.1e}, zero mismatches {zm}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"  ({n} scenarios: the engine's Mw range stops at 9.6, PTHA18's table at "
          f"{off['rate_annual'].size})")
    return ok


def check_hs_vaus():
    import recover_peak_slip_weights as rpw
    print("\nHS_VAUS:", end="")
    return rpw.main() == 0


def main():
    checks = sys.argv[1:] or ["mu_Mw", "engine", "hs_vaus"]
    fns = {"mu_Mw": check_mu_Mw, "engine": check_engine, "hs_vaus": check_hs_vaus}
    results = {c: fns[c]() for c in checks}
    print("\nSUMMARY: " + ", ".join(f"{c} {'PASS' if r else 'FAIL'}" for c, r in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
