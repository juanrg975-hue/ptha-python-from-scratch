"""v12 checks of --mesh-file (an external quadrilateral mesh) on alaskaaleutians.

The colleagues' file ``alaskaaleutians_quadrilateral_coors.dat`` has 312
quadrilaterals, as many as PTHA18's alaskaaleutians unit sources. Two checks:

  geometry  the mesh read by pyptha_v12/mesh_file.py against PTHA18's own
            unit-source table (unit_source_statistics_alaskaaleutians.nc):
            centre, strike, length, width, depth and dip of every cell.
  rates     the v12 engine on PTHA18's official alaskaaleutians input three
            times, everything identical but the geometry:
              table      PTHA18's unit-source table (what step 8 runs),
              mesh       the .dat as sent (trench at 0.1 km),
              mesh_0km   the .dat with its trench moved to 0 km, as PTHA18's;
            mesh_0km must give the table's scenario rates (it is the same
            mesh), and mesh shows what the 0.1 km trench changes.

Needs, from ptha18_logic_tree_test/: inputs/input_alaskaaleutians.json and
inputs/geometry/unit_source_statistics_alaskaaleutians.nc (PTHA18's official
input, extracted with official_ptha_data/fetch_official_inputs.py) and the
.dat in V9/. Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe V9/from_scratch_v12/validation/validate_v12_mesh.py
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
V9 = os.path.dirname(PKG)
BASE = os.path.dirname(V9)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "lib"))
from pyptha_v12 import mesh_file, unit_sources as us  # noqa: E402

Z = "alaskaaleutians"
DAT = os.path.join(V9, "alaskaaleutians_quadrilateral_coors.dat")
USS = os.path.join(BASE, "inputs", "geometry", f"unit_source_statistics_{Z}.nc")
INPUT = os.path.join(BASE, "inputs", f"input_{Z}.json")


def official_stats():
    import netCDF4
    with netCDF4.Dataset(USS) as d:
        d.set_auto_mask(False)
        return {k: np.asarray(d[k][:], float) for k in (
            "lon_c", "lat_c", "depth", "dip", "strike", "length", "width", "rake",
            "downdip_number", "alongstrike_number")}


def check_geometry():
    print("\nGEOMETRY: the .dat read by mesh_file vs PTHA18's unit-source table")
    st = us.discretized_source_approximate_summary_statistics(
        mesh_file.read_quadrilateral_mesh(DAT, log=lambda *a: None))
    o = official_stats()
    ko = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(o["alongstrike_number"], o["downdip_number"]))}
    idx = np.array([ko[(int(a), int(b))] for a, b in zip(st["alongstrike_number"], st["downdip_number"])])
    trench = np.asarray(st["downdip_number"]) == 1
    ok = True
    for f in ("lon_c", "lat_c", "strike", "length", "width", "depth", "dip"):
        d = np.asarray(st[f], float) - o[f][idx]
        if f in ("lon_c", "strike"):
            d = (d + 180) % 360 - 180
        d = np.abs(d)
        deep, top = d[~trench].max(), d[trench].max()
        good = deep < 1e-6
        ok &= good
        print(f"  {f:7s} rows 2-4 max|diff| {deep:.1e}   trench row {top:.3g}  {'PASS' if good else 'FAIL'}")
    a, b = np.sum(st["length"] * st["width"]), np.sum(o["length"] * o["width"])
    print(f"  area {a:,.0f} vs {b:,.0f} km2 ({100 * (a / b - 1):+.4f}%). The trench row differs "
          f"only because the file puts the trench at 0.1 km (PTHA18: 0 km).")
    return ok


def run(name, geometry):
    import bird_convergence as bc
    cfg = json.load(open(INPUT))
    cfg["run_name"] = f"{Z}_v12_mesh_check_{name}"
    cfg["events"].update({"scaling_relation": "Strasser", "shear_modulus_Pa": 3.0e10})
    if bc.ptha18_uses_bird_convergence(Z):
        # PTHA18's Bird model on its own mesh, the same for all three runs
        o = official_stats()
        st = {k: o[k] for k in ("lon_c", "lat_c", "strike", "dip", "rake", "downdip_number", "alongstrike_number")}
        st["length"], st["width"] = o["length"], o["width"]
        cfg["rates"].update(bc.rates_entries(bc.column_convergence(st, log=lambda *a: None)))
    cfg["geometry"] = geometry
    tmp = os.path.join(HERE, f"_input_{name}.json")
    json.dump(cfg, open(tmp, "w"), indent=1)
    try:
        subprocess.run([sys.executable, os.path.join(PKG, "python_logic_tree_v12", "run_logic_tree.py"), tmp],
                       cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    finally:
        os.remove(tmp)
    import pandas as pd
    out = os.path.join(V9, "runs", "python", cfg["run_name"])
    return (pd.read_csv(os.path.join(out, f"scenario_rates_{Z}.csv")),
            pd.read_csv(os.path.join(out, "rate_curves.csv")))


def check_rates():
    print("\nRATES: the engine on PTHA18's official input, PTHA18's table vs the .dat")
    zero = os.path.join(HERE, "_alaska_trench_0km.dat")
    a = np.loadtxt(DAT)
    dep = a[:, 2::3]
    dep[dep == -100.0] = 0.0
    np.savetxt(zero, a, fmt="%.12f")
    try:
        t_s, t_c = run("table", {"mode": "official_statistics", "official_statistics_nc": USS})
        m_s, m_c = run("mesh", {"mode": "mesh_file", "mesh_file": DAT, "mesh_depth_units": "m"})
        z_s, z_c = run("mesh_0km", {"mode": "mesh_file", "mesh_file": zero, "mesh_depth_units": "m"})
    finally:
        os.remove(zero)
    ok = True
    for name, s, c in (("mesh_0km", z_s, z_c), ("mesh", m_s, m_c)):
        same_rows = len(s) == len(t_s) and np.allclose(s.Mw, t_s.Mw)
        nz = t_s.rate_mean > 0
        rel = float(np.max(np.abs(s.rate_mean[nz] - t_s.rate_mean[nz]) / t_s.rate_mean[nz]))
        rc = float(np.max(np.abs(c.exceedance_rate_unsegmented - t_c.exceedance_rate_unsegmented)
                          / t_c.exceedance_rate_unsegmented.where(t_c.exceedance_rate_unsegmented > 0)))
        r72 = float(c.exceedance_rate_unsegmented.iloc[0] / t_c.exceedance_rate_unsegmented.iloc[0])
        line = (f"  {name:8s} scenarios {len(s)}/{len(t_s)}; max rel diff: scenario rates {rel:.1e}, "
                f"rate curve {rc:.1e}; Mw>=7.2 rate x{r72:.5f}")
        if name == "mesh_0km":
            good = same_rows and rel < 1e-6
            ok &= good
            line += f"  {'PASS' if good else 'FAIL'}"
        print(line)
    print("  (mesh_0km = the .dat with PTHA18's trench depth: same scenarios and rates as PTHA18's "
          "table; mesh = the .dat as sent: the 0.1 km trench changes the trench row's dip slightly)")
    return ok


def main():
    for p in (DAT, USS, INPUT):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    res = {"geometry": check_geometry(), "rates": check_rates()}
    print("\nSUMMARY: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in res.items()))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
