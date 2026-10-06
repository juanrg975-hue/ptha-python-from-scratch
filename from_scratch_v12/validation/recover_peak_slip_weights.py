"""Recover PTHA18's peak-slip quantile weights for HS and VAUS scenarios.

PTHA18 shares a FAUS rupture's rate among the HS (and the VAUS) scenarios
drawn from it with a weight f(q), q the scenario's peak-slip rank in its
family (see pyptha_v12/hs_vaus_rates.py). f is tabulated at 51 knots
(q = 0, 0.02, ..., 1; event_properties_and_GOF.R) in
peak_slip_quantile_adjustment_factors.csv (constant shear modulus) and
peak_slip_quantile_adjustment_factors_varyMu.csv (variable shear modulus,
v11), which PTHA18 did not publish.

Each published family gives linear equations on the knots: with s_i the
share of the parent's rate scenario i carries,
f(q_i) - s_i * sum_j f(q_j) = 0. Stacked over every family of a zone, they
fix f up to one constant factor (which cancels when a family is
normalised). The constant shear modulus curves use the published
rate_annual columns; the variable ones the variable_mu_rate_annual columns
of the same files (same families, same peak-slip limit). This script

1. fits each curve on puysegur2 alone and checks it on kermadectonga2 (to
   ~1e-14). The other way round is printed but is not a test:
   kermadectonga2's families never reach q near 0 or 1 (rank 46 of 50), so
   some knots stay free;
2. fits each curve on both zones and writes
   data/ptha18_peak_slip_quantile_weights.csv (normalised to integrate to 1
   over q, as PTHA18's densities do);
3. checks the method of the PTHA18 report (Section 3.6): the integral of
   each curve is the inverse of a cubic through (0, 0) and (1, 1);
4. recomputes every published HS and VAUS rate of both zones, constant and
   variable shear modulus, from the published FAUS rates with
   hs_vaus_rates.family_weights and that file.

Needs official_ptha_data/public_nc/ (PTHA18's all_uniform/stochastic/
variable_uniform_slip_earthquake_events_<zone>.nc). Run from
ptha18_logic_tree_test/:
    .venv/Scripts/python.exe V9/from_scratch_v12/validation/recover_peak_slip_weights.py
"""

import os
import sys

import netCDF4 as nc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from pyptha_v12 import hs_vaus_rates as hvr  # noqa: E402

NC_DIRS = [os.path.join(PKG, os.pardir, os.pardir, "official_ptha_data", "public_nc"),
           os.path.join(PKG, os.pardir, "official_ptha_data", "public_nc")]
ZONES = ("puysegur2", "kermadectonga2")
KINDS = {"HS": "stochastic", "VAUS": "variable_uniform"}
MU = {False: "rate_annual", True: "variable_mu_rate_annual"}
KNOTS = np.linspace(0, 1, 51)
OUT_CSV = hvr.DEFAULT_WEIGHTS_CSV


def nc_dir():
    for d in NC_DIRS:
        if os.path.exists(os.path.join(d, "all_uniform_slip_earthquake_events_puysegur2.nc")):
            return os.path.abspath(d)
    raise SystemExit("official_ptha_data/public_nc/ not found")


def load(d, zone, kind):
    """parent rates/Mw, and per scenario: parent row, peak slip, rates"""
    with nc.Dataset(os.path.join(d, f"all_uniform_slip_earthquake_events_{zone}.nc")) as u, \
            nc.Dataset(os.path.join(d, f"all_{KINDS[kind]}_slip_earthquake_events_{zone}.nc")) as f:
        u.set_auto_mask(False)
        f.set_auto_mask(False)
        slip = nc.chartostring(f["event_slip_string"][:])
        return {
            "parent_rate": {m: u[c][:].astype(float) for m, c in MU.items()},
            "parent_Mw": u["Mw"][:].astype(float),
            "row": f["uniform_event_row"][:].astype(int),  # 1-based
            "peak": np.array([max(float(x) for x in s.split("_") if x) for s in slip]),
            "rate": {m: f[c][:].astype(float) for m, c in MU.items()},
        }


def families(z, variable_mu):
    """(q, share) of the scenarios under the limit, per family with a rate"""
    _, above = hvr.family_weights(z["row"], z["peak"], z["parent_Mw"][z["row"] - 1],
                                  "HS", weights={"HS": (KNOTS, np.ones(51))})
    out = []
    for r in np.unique(z["row"]):
        pr = z["parent_rate"][variable_mu][r - 1]
        ok = np.where((z["row"] == r) & ~above)[0]
        if pr <= 0 or ok.size == 0:
            continue
        rank = np.empty(ok.size)
        rank[np.argsort(z["peak"][ok], kind="stable")] = np.arange(1, ok.size + 1)
        out.append((rank / (ok.size + 1), z["rate"][variable_mu][ok] / pr))
    return out


def basis(q):
    """linear-interpolation weights of each q on the 51 knots"""
    B = np.zeros((q.size, KNOTS.size))
    j = np.clip(np.floor(q / 0.02).astype(int), 0, 49)
    t = (q - KNOTS[j]) / 0.02
    B[np.arange(q.size), j] = 1 - t
    B[np.arange(q.size), j + 1] = t
    return B


def fit(fams):
    A = np.vstack([basis(q) - s[:, None] * basis(q).sum(0)[None, :] for q, s in fams])
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-10))
    # fix the free factor with mean(f) = 1, then normalise the integral
    A = np.vstack([A, np.full((1, KNOTS.size), 1e3 / KNOTS.size)])
    b = np.zeros(A.shape[0])
    b[-1] = 1e3
    f, *_ = np.linalg.lstsq(A, b, rcond=None)
    return f / np.trapezoid(f, KNOTS), rank


def worst(fams, f):
    err = 0.0
    for q, s in fams:
        g = np.interp(q, KNOTS, f)
        err = max(err, float(np.max(np.abs(g / g.sum() - s) / s)))
    return err


def cubic_check(f):
    """PTHA18 report 3.6: q = cubic(F), F the integral of f; max residual"""
    F = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * 0.02)])
    F /= F[-1]
    c = np.polyfit(F, KNOTS, 3)
    return c, float(np.max(np.abs(np.polyval(c, F) - KNOTS)))


def main():
    d = nc_dir()
    data = {(z, k): load(d, z, k) for z in ZONES for k in KINDS}
    ok = True
    table = {}
    for variable_mu in (False, True):
        for kind in KINDS:
            label = kind + (" variable shear modulus" if variable_mu else "")
            fams = {z: families(data[(z, kind)], variable_mu) for z in ZONES}
            print(f"\n{label}")
            for train, test in (ZONES, ZONES[::-1]):
                f, rank = fit(fams[train])
                e = worst(fams[test], f)
                if rank == 50:
                    ok &= e < 1e-9
                    note = ""
                else:
                    note = " (not a test: knots near q = 0 or 1 left free)"
                print(f"  fitted on {train} (rank {rank} of 50 free): "
                      f"worst relative error on {test} {e:.1e}{note}")
            f, rank = fit(fams[ZONES[0]] + fams[ZONES[1]])
            ok &= rank == 50
            c, res = cubic_check(f)
            print(f"  report 3.6 check: q = {c[0]:.4f} F^3 {c[1]:+.4f} F^2 {c[2]:+.4f} F "
                  f"{c[3]:+.4f}, max residual {res:.1e}")
            table[(kind, variable_mu)] = f

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        fh.write("peak_slip_quantile,weight_stochastic,weight_variable_uniform,"
                 "weight_stochastic_variable_mu,weight_variable_uniform_variable_mu\n")
        for i, q in enumerate(KNOTS):
            fh.write(f"{q:.2f},{table[('HS', False)][i]:.12g},{table[('VAUS', False)][i]:.12g},"
                     f"{table[('HS', True)][i]:.12g},{table[('VAUS', True)][i]:.12g}\n")
    print(f"\nwrote {OUT_CSV}")

    # end to end, through the module and the file just written
    weights = hvr.load_peak_slip_weights(OUT_CSV)
    print("\nevery published rate from the published FAUS rates (hs_vaus_rates):")
    for (zone, kind), z in data.items():
        for variable_mu in (False, True):
            w, above = hvr.family_weights(z["row"], z["peak"], z["parent_Mw"][z["row"] - 1],
                                          kind, weights=weights, variable_mu=variable_mu)
            pred = w * z["parent_rate"][variable_mu][z["row"] - 1]
            obs = z["rate"][variable_mu]
            nz = obs > 0
            rel = float(np.max(np.abs(pred[nz] - obs[nz]) / obs[nz]))
            zero_ok = bool(np.all(pred[~nz] == 0))
            ok &= rel < 1e-9 and zero_ok
            print(f"  {zone:15s} {kind:4s} {MU[variable_mu]:24s} {obs.size:6d} scenarios, "
                  f"{int(above.sum()):5d} above the 7.5x limit; worst relative error "
                  f"{rel:.1e}; zero rates reproduced: {zero_ok}")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
