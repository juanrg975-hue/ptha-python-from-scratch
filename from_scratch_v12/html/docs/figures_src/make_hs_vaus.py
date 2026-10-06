"""Figures for hs_vaus.html (step 7b: heterogeneous and variable-area slip).

Two kinds of figure, both read-only with respect to the example folders:

1. Illustrations on kermadectonga2_v8's real mesh (data/slab2/unit_source_grid.npy):
   one FAUS scenario, heterogeneous-slip (HS) fields drawn from it and the VAUS
   field derived from each HS field. They are made HERE with the same recipe
   step 7b uses (templates/step7b_stochastic_slip.py.tmpl, main()'s inner
   loop), calling the same pyptha_v12.stochastic_slip functions. They are an
   illustration, not rows of any step 7b output file.
2. The comparison with PTHA18's published HS/VAUS catalogues, from the files
   step 8 wrote: kermadectonga2_v8/outputs_official/ptha18_reference/*_comparison.csv.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_hs_vaus.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.abspath(os.path.join(HERE, ".."))
PKG = os.path.abspath(os.path.join(DOCS, "..", ".."))
ROOT = os.path.abspath(os.path.join(PKG, ".."))
IMG = os.path.join(DOCS, "img")
os.makedirs(IMG, exist_ok=True)
sys.path.insert(0, PKG)

from pyptha_v12 import events, unit_sources as us  # noqa: E402
from pyptha_v12 import stochastic_slip as ss  # noqa: E402
from pyptha_v12.scaling import M0_2_Mw  # noqa: E402

INK = "#1d2433"
MUTED = "#5b6475"
ENG = "#2f6fdf"
SCR = "#eb6834"
OFFC = "#2a78d6"
SEIS = "#8a5cd1"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#b8bfca", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": False, "legend.edgecolor": "#dde1e8",
})
DPI = 120
MU = 3.0e10


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1e3:.0f} KB)")


# ---------------------------------------------------------------------------
# The step 7b recipe, for one parent FAUS scenario (see the template's main())
# ---------------------------------------------------------------------------
grid = np.load(os.path.join(ROOT, "kermadectonga2_v8", "data", "slab2", "unit_source_grid.npy"))
stats = us.discretized_source_approximate_summary_statistics(grid)
dd = stats["downdip_number"].astype(int)
ask = stats["alongstrike_number"].astype(int)
n_dd, n_as = int(dd.max()), int(ask.max())
length_g = np.zeros((n_dd, n_as))
width_g = np.zeros((n_dd, n_as))
length_g[dd - 1, ask - 1] = stats["length"]
width_g[dd - 1, ask - 1] = stats["width"]
area_g = length_g * width_g
lwkc = ss.sffm_make_random_lwkc_function(relation="Strasser")

MW = 9.0
all_eq = events.get_all_earthquake_events(stats, Mmin=MW, Mmax=MW, dMw=0.1)
# the parent: the placement whose first column is 30, up-dip position
first = np.array([ask[ix].min() for ix in all_eq["event_indices"]])
top = np.array([dd[ix].min() for ix in all_eq["event_indices"]])
parent = int(np.where((first == 30) & (top == 1))[0][0])
pidx = all_eq["event_indices"][parent]
faus_slip = float(all_eq["slip"][parent])
dd0, as0 = dd[pidx].min(), ask[pidx].min()
nd, na = dd[pidx].max() - dd0 + 1, ask[pidx].max() - as0 + 1
faus = np.zeros((n_dd, n_as))
faus[dd[pidx] - 1, ask[pidx] - 1] = faus_slip
desired_M0 = M0_2_Mw(MW, inverse=True)


def one_realisation(seed):
    """Steps of step 7b's inner loop for one HS field and its VAUS field."""
    rng = np.random.default_rng(seed)
    pr = int(rng.integers(dd0 - 1, dd0 - 1 + nd))         # random peak cell, inside the parent's box
    pc = int(rng.integers(as0 - 1, as0 - 1 + na))
    d = lwkc(np.array([MW]), rng=rng)                      # random L, W, kcx, kcy
    L, W, kcx, kcy = (float(d[k][0]) for k in ("L", "W", "kcx", "kcy"))
    plen, pwid = length_g[pr, pc], width_g[pr, pc]
    ncol = int(min(n_as, max(1, round(L / plen))))
    nrow = int(min(n_dd, max(1, round(W / pwid))))
    if W > n_dd * pwid:                                   # expand_length_if_width_limited = 'random'
        deficit = nrow * pwid / W
        if deficit < 1.0 and rng.uniform() >= 0.5:
            ncol = int(min(n_as, max(1, round(L / deficit / plen))))
    sr_, er_, sc_, ec_ = ss.rectangle_on_grid(n_dd, n_as, nrow, ncol, pr, pc,
                                              randomly_vary_around_target_centre=True, rng=rng)
    tmpl = np.zeros((er_ - sr_ + 1, ec_ - sc_ + 1))
    tmpl[pr - sr_, pc - sc_] = 1.0
    fake = ss.sffm_simulate((kcx * plen, kcy * pwid), tmpl, rng=rng)
    win_area = area_g[sr_:er_ + 1, sc_:ec_ + 1]
    fake = fake / float((fake * win_area * 1e6 * MU).sum()) * desired_M0   # exact moment of Mw
    hs = np.zeros((n_dd, n_as))
    hs[sr_:er_ + 1, sc_:ec_ + 1] = fake
    on = np.where(hs > 0)
    r0, r1, c0, c1 = on[0].min(), on[0].max(), on[1].min(), on[1].max()
    vaus_slip = float((hs * area_g).sum()) / float(area_g[r0:r1 + 1, c0:c1 + 1].sum())
    vaus = np.zeros((n_dd, n_as))
    vaus[r0:r1 + 1, c0:c1 + 1] = vaus_slip
    return hs, vaus, dict(L=L, W=W, kcx=kcx, kcy=kcy, peak=(pr, pc), win=(sr_, er_, sc_, ec_))


def moment(f):
    return float((f * area_g * 1e6 * MU).sum())


C0, C1 = 12, 62        # columns shown (1-based, inclusive)


def draw(ax, f, title, vmax, box=None):
    sub = f[:, C0 - 1:C1]
    im = ax.imshow(np.where(sub > 0, sub, np.nan), cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto",
                   extent=(C0 - 0.5, C1 + 0.5, n_dd + 0.5, 0.5), interpolation="nearest")
    ax.set_facecolor("#f0f2f5")
    for c in range(C0, C1 + 2):
        ax.axvline(c - 0.5, color="white", lw=0.6)
    for r in range(1, n_dd + 2):
        ax.axhline(r - 0.5, color="white", lw=0.6)
    if box is not None:
        r0, r1, c0, c1 = box
        ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, fill=False,
                                   ec=ENG, lw=2.2, ls="--"))
    ax.set_yticks(range(1, n_dd + 1))
    ax.set_ylabel("down-dip row\n(1 = trench)", fontsize=10)
    ax.set_title(title, fontsize=11.5, loc="left")
    return im


# ---------------------------------------------------------------------------
# Figure 1: FAUS -> HS -> VAUS, one realisation
# ---------------------------------------------------------------------------
hs, vaus, info = one_realisation(20260927)
m_f, m_h, m_v = moment(faus), moment(hs), moment(vaus)
vmax = float(np.ceil(hs.max()))
fig, axs = plt.subplots(3, 1, figsize=(12, 7.4), sharex=True)
draw(axs[0], faus, f"FAUS (step 7): {nd} x {na} cells, uniform {faus_slip:.2f} m", vmax,
     box=(dd0, dd0 + nd - 1, as0, as0 + na - 1))
draw(axs[1], hs, f"HS (step 7b): random L {info['L']:.0f} km, W {info['W']:.0f} km; "
     f"peak {hs.max():.1f} m; {int((hs > 0).sum())} cells with slip", vmax,
     box=(dd0, dd0 + nd - 1, as0, as0 + na - 1))
on = np.where(vaus > 0)
im = draw(axs[2], vaus, f"VAUS (from the HS field above): rectangle around its cells, "
          f"uniform {vaus.max():.2f} m", vmax,
          box=(on[0].min() + 1, on[0].max() + 1, on[1].min() + 1, on[1].max() + 1))
pr, pc = info["peak"]
axs[1].plot(pc + 1, pr + 1, marker="*", ms=16, color=ENG, mec="white")
axs[2].set_xlabel("along-strike column of the kermadectonga2_v8 mesh (columns 12 to 62 of 73 shown)")
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.015)
cb.set_label("slip (m)")
fig.text(0.01, -0.04,
         f"Mw {MW}. Dashed box in the top two panels: the parent FAUS scenario (columns {as0}-{as0 + na - 1}). "
         f"Star: the HS field's randomly drawn peak cell, always inside that box.\n"
         f"Seismic moment (slip x area x 3e10 Pa): FAUS {m_f:.4e}, HS {m_h:.4e}, VAUS {m_v:.4e} N m, "
         f"all equal to M0(Mw {MW}) = {desired_M0:.4e}.",
         fontsize=10, color=MUTED)
save(fig, "hs_vaus_kermadectonga2_three_types.png")
print("moments", m_f, m_h, m_v, desired_M0, info)

# ---------------------------------------------------------------------------
# Figure 2: six HS realisations of the same parent
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(6, 1, figsize=(12, 9.6), sharex=True)
fields = [one_realisation(1000 + k) for k in range(6)]
vmax2 = float(np.ceil(max(f[0].max() for f in fields)))
for k, (ax, (h, v, inf)) in enumerate(zip(axs, fields)):
    im = draw(ax, h, f"realisation {k + 1}: L {inf['L']:.0f} km, W {inf['W']:.0f} km, "
              f"peak {h.max():.1f} m, {int((h > 0).sum())} cells", vmax2,
              box=(dd0, dd0 + nd - 1, as0, as0 + na - 1))
    ax.set_ylabel("")
    ax.plot(inf["peak"][1] + 1, inf["peak"][0] + 1, marker="*", ms=12, color=ENG, mec="white")
axs[-1].set_xlabel("along-strike column (dashed box: the parent FAUS scenario; star: random peak cell)")
cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.015)
cb.set_label("slip (m)")
save(fig, "hs_vaus_kermadectonga2_six_realisations.png")

# ---------------------------------------------------------------------------
# Figure 3: comparison with PTHA18's published catalogues
# ---------------------------------------------------------------------------
k8 = os.path.join(ROOT, "kermadectonga2_v8", "outputs_official", "ptha18_reference")
hs8 = pd.read_csv(os.path.join(k8, "hs_comparison.csv"))
va8 = pd.read_csv(os.path.join(k8, "vaus_comparison.csv"))
print("HS peak-slip ratio range:", round(float((hs8.mean_peak_slip_m_synthetic / hs8.mean_peak_slip_m_official).min()), 3),
      round(float((hs8.mean_peak_slip_m_synthetic / hs8.mean_peak_slip_m_official).max()), 3))
print("VAUS slip ratio range:", round(float((va8.mean_peak_slip_m_synthetic / va8.mean_peak_slip_m_official).min()), 3),
      round(float((va8.mean_peak_slip_m_synthetic / va8.mean_peak_slip_m_official).max()), 3))
print("HS cells ratio range:", round(float((hs8.mean_active_cells_synthetic / hs8.mean_active_cells_official).min()), 3),
      round(float((hs8.mean_active_cells_synthetic / hs8.mean_active_cells_official).max()), 3))
print("fields", int(hs8.n_synthetic.sum()), "published", int(hs8.n_official.sum()), "magnitudes", len(hs8))

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.axhline(1, color=MUTED, lw=1)
ax.plot(hs8.Mw, hs8.mean_peak_slip_m_synthetic / hs8.mean_peak_slip_m_official, "o-", color=SCR, ms=4,
        label="HS: mean peak slip, this run / PTHA18")
ax.plot(va8.Mw, va8.mean_peak_slip_m_synthetic / va8.mean_peak_slip_m_official, "s-", color=SEIS, ms=4,
        label="VAUS: mean (uniform) slip, this run / PTHA18")
ax.plot(hs8.Mw, hs8.mean_active_cells_synthetic / hs8.mean_active_cells_official, "^--", color=OFFC, ms=4,
        label="HS: mean number of cells with slip, ratio")
ax.set_title("kermadectonga2_v8, step 7b: "
             f"{int(hs8.n_synthetic.sum()):,} fields vs {int(hs8.n_official.sum()):,} published, {len(hs8)} magnitudes",
             fontsize=11.5)
ax.set_xlabel("Magnitude Mw")
ax.set_ylabel("this run / PTHA18 (1 = identical)")
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, color="#e3e6ec")
fig.tight_layout()
save(fig, "hs_vaus_official_comparison.png")
