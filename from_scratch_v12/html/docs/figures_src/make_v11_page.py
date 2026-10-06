"""Write html/docs/v11.html: what v11 adds (variable shear modulus, step 7c).

Numbers and diagrams come from calabria2_v11_varmu's own outputs, so the page can
be regenerated after a re-run. Run from V9/:
    ../.venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_v11_page.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
V9 = os.path.abspath(os.path.join(DOCS, os.pardir, os.pardir, os.pardir))
EX = os.path.join(V9, "calabria2_v11_varmu", "outputs")
Z = "calabria2"


def rigidity(d):
    return 10 ** np.interp(np.asarray(d, float), [0, 7.5, 15, 35, 9999],
                           np.log10([10.0, 10.0, 30.0, 67.0, 67.0]))


# ------------------------------------------------------------------ numbers
slip = pd.read_csv(os.path.join(EX, f"integrated_slip_{Z}.csv"))
rows = []
for r, g in slip.groupby("downdip_number"):
    dep = float(np.average(g["depth_km"], weights=g["area_km2"]))
    mu = float(np.average(rigidity(g["depth_km"]), weights=g["area_km2"]))
    rows.append((int(r), dep, mu, 2 / 3 * np.log10(mu / 30), 30 / mu))
c = pd.read_csv(os.path.join(EX, "rate_curves.csv"))
v = pd.read_csv(os.path.join(EX, "rate_curves_variable_mu.csv"))
s = pd.read_csv(os.path.join(EX, f"scenario_rates_{Z}_variable_mu.csv"))
summ = pd.read_csv(os.path.join(EX, "logic_tree_summary_variable_mu.csv"))
sm = dict(zip(summ["item"], summ["value"]))
col = "exceedance_rate_unsegmented"
rate_rows = []
for m in (7.5, 8.0, 8.5, 9.0):
    a = float(np.interp(m, c["Mw"], c[col]))
    b = float(np.interp(m, v["Mw"], v[col]))
    r = float(s.loc[s["variable_mu_Mw"] >= m - 1e-9, "rate_mean"].sum())
    rate_rows.append((m, a, b, r))
tvd = float(sm[f"posterior_vs_variable_mu_tvd_{Z}"])

# ------------------------------------------------------------------ diagrams
def flow_svg():
    """Where variable rigidity enters: steps 7, 7b, 7c."""
    box = lambda x, y, w, h, cls, t1, t2="": (  # noqa: E731
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>'
        f'<text x="{x+w/2}" y="{y+h/2-(6 if t2 else -4)}" text-anchor="middle" class="t1">{t1}</text>'
        + (f'<text x="{x+w/2}" y="{y+h/2+12}" text-anchor="middle" class="t2">{t2}</text>' if t2 else ""))
    arrow = lambda x1, y1, x2, y2: (  # noqa: E731
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ar" marker-end="url(#ah)"/>')
    p = ['<svg viewBox="0 0 860 300" role="img" aria-label="How the variable shear modulus enters the pipeline">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         '<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>',
         box(10, 20, 190, 60, "b0", "step 7: FAUS rates", "constant rigidity 30 GPa"),
         box(10, 120, 190, 60, "b0", "step 7b: HS and VAUS", "fields + their rates"),
         box(250, 20, 190, 60, "b1", "Mw relabelled", "M0 = sum area x slip x mu(depth)"),
         box(250, 120, 190, 60, "b1", "difference distribution", "(variable - constant) Mw | Mw"),
         box(490, 70, 170, 60, "b1", "LEVEL 3 weights", "catalogue Mw has an error"),
         box(700, 20, 150, 60, "b2", "variable_mu rates", "FAUS, curves, percentiles"),
         box(700, 120, 150, 60, "b2", "HS / VAUS shares", "DART curves, variable mu"),
         box(250, 220, 410, 56, "b3", "unchanged: cells, slip, tsunami of every scenario",
             "only how often each one happens changes"),
         arrow(200, 150, 250, 150), arrow(200, 50, 250, 50), arrow(345, 80, 345, 120),
         arrow(440, 150, 490, 110), arrow(660, 100, 700, 50), arrow(660, 100, 700, 150),
         '<text x="12" y="210" class="t2">existing steps</text>'
         '<text x="252" y="205" class="t2">step 7c (v11)</text>',
         '</svg>']
    return "".join(p)


def curve_svg():
    W, H, L, R, T, B = 640, 300, 56, 24, 18, 44
    dmax = 50.0
    X = lambda d: L + d / dmax * (W - L - R)  # noqa: E731
    Y = lambda g: T + (1 - g / 75.0) * (H - T - B)  # noqa: E731
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rigidity against depth with the calabria2 rows">']
    for g in (0, 10, 30, 50, 67):
        p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="gr"/>'
                 f'<text x="{L-6}" y="{Y(g)+4:.1f}" text-anchor="end" class="tk">{g}</text>')
    for d in range(0, 51, 10):
        p.append(f'<text x="{X(d):.1f}" y="{H-B+16}" text-anchor="middle" class="tk">{d}</text>')
    p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-6}" text-anchor="middle" class="t2">depth of the cell (km)</text>'
             f'<text x="14" y="{(T+H-B)/2:.0f}" text-anchor="middle" class="t2" transform="rotate(-90 14 {(T+H-B)/2:.0f})">rigidity (GPa)</text>')
    p.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(30):.1f}" y2="{Y(30):.1f}" class="cst"/>'
             f'<text x="{W-R-4}" y="{Y(30)-6:.1f}" text-anchor="end" class="t2">constant 30 GPa (steps 7, 7b)</text>')
    dd = np.linspace(0, dmax, 200)
    p.append('<polyline points="' + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(dd, rigidity(dd)))
             + '" class="cv"/>')
    p.append(f'<text x="{X(dmax)-4:.1f}" y="{Y(67)-8:.1f}" text-anchor="end" class="t2">PTHA18 (Bilek &amp; Lay 1999 fit)</text>')
    groups = []
    for r in rows:
        if groups and round(groups[-1][-1][3], 2) == round(r[3], 2):
            groups[-1].append(r)
        else:
            groups.append([r])
    for g in groups:
        for r in g:
            p.append(f'<circle cx="{X(r[1]):.1f}" cy="{Y(r[2]):.1f}" r="6" class="pt"/>')
        r = g[-1]
        x, y = X(r[1]), Y(r[2])
        name = f"rows {g[0][0]}-{r[0]}" if len(g) > 1 else f"row {r[0]}"
        right = x > L + 0.7 * (W - L - R)
        p.append(f'<text x="{x - 9 if right else x + 9:.1f}" y="{y + 18:.1f}" '
                 f'text-anchor="{"end" if right else "start"}" class="lb">{name}: Mw {r[3]:+.2f}</text>')
    p.append("</svg>")
    return "".join(p)


row_html = "".join(f"<tr><td>{r}</td><td>{d:.1f}</td><td>{m:.0f}</td><td>{sh:+.2f}</td><td>{f:.2f}</td></tr>"
                   for r, d, m, sh, f in rows)
rp = lambda x: f"{1/x:,.0f} yr" if x > 0 else "-"  # noqa: E731
rate_html = "".join(f"<tr><td>{m:.1f}</td><td>{a:.3g} ({rp(a)})</td><td>{b:.3g} ({rp(b)})</td>"
                    f"<td>{b/a:.3f}</td><td>{r:.3g} ({rp(r)})</td><td>{r/a:.2f}</td></tr>"
                    for m, a, b, r in rate_rows)

STYLE = """<style>
figure.fig svg .b0{fill:var(--card);stroke:var(--line);stroke-width:1.5}
figure.fig svg .b1{fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5}
figure.fig svg .b2{fill:var(--good-soft);stroke:var(--good);stroke-width:1.5}
figure.fig svg .b3{fill:var(--warn-soft);stroke:var(--warn);stroke-width:1.5}
figure.fig svg .t1{font-size:14px;font-weight:600;fill:var(--ink)}
figure.fig svg .t2{font-size:12px;fill:var(--muted)}
figure.fig svg .ar{stroke:var(--muted);stroke-width:1.6}
figure.fig svg .ah{fill:var(--muted)}
figure.fig svg .gr{stroke:var(--line);stroke-width:1}
figure.fig svg .cst{stroke:var(--muted);stroke-width:1.3;stroke-dasharray:5 4}
figure.fig svg .cv{fill:none;stroke:var(--accent);stroke-width:2.5}
figure.fig svg .pt{fill:var(--warn);stroke:var(--card);stroke-width:2}
figure.fig svg .tk{font-family:var(--mono);font-size:11px;fill:var(--muted)}
figure.fig svg .lb{font-family:var(--mono);font-size:11.5px;font-weight:600;fill:var(--ink);paint-order:stroke;stroke:var(--card);stroke-width:4px}
</style>"""

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>What is new in v11 | from_scratch_v12</title>
<link rel="stylesheet" href="assets/style.css">
{STYLE}
</head>
<body>
<div class="layout">
<nav class="side" id="side"></nav>
<main>

<section id="top">
  <div class="crumb">Start</div>
  <h1>What is new in v11: variable shear modulus</h1>
  <p class="lead">v11 (2026-10-06) is v10_q plus PTHA18's second set of rates, the ones with a rigidity that grows with depth (<code>variable_mu_rate_annual</code>). PTHA18's headline hazard maps use them with the HS scenarios. A new step 7c, switched on with <code>generate.py --variable-mu on</code> (default off), computes them next to the constant-rigidity ones, which do not change: every file steps 1-7b write is byte for byte v10_q's.</p>
  <div class="grid g4">
    <div class="card kpi"><div class="num">1</div><div class="lbl">new step, 7c, after 7b; on with <code>--variable-mu on</code></div></div>
    <div class="card kpi"><div class="num">7e-13</div><div class="lbl">largest relative difference to PTHA18's published variable_mu rates (puysegur2, mean and 5 percentiles)</div></div>
    <div class="card kpi"><div class="num">identical</div><div class="lbl">constant-rigidity results to v10_q (calabria2: every output file)</div></div>
    <div class="card kpi"><div class="num">179</div><div class="lbl">tests passing (7 new), plus validation/validate_v11.py</div></div>
  </div>
</section>

<section id="what-sec">
  <h2 id="what">What the shear modulus is, and why it matters for tsunamis</h2>
  <p>The <b>shear modulus</b> or <b>rigidity</b> (&mu;) is how stiff the rock is. An earthquake's size as a seismometer measures it is its seismic moment, M0 = &mu; &times; area &times; slip, and its magnitude Mw follows from M0. For the same M0, softer rock needs more slip. Near the trench the rock is soft (about 10 GPa) and deep down it is stiff (up to 67 GPa), so a shallow earthquake of a given measured magnitude slips more than one at depth, and more slip near the trench makes a bigger tsunami. Steps 7 and 7b use 30 GPa everywhere (PTHA18's <code>rate_annual</code>); step 7c adds the depth-varying case (PTHA18's <code>variable_mu_rate_annual</code>, report Section 3.7.5).</p>
  <figure class="fig">{curve_svg()}<figcaption>PTHA18's rigidity curve (rptha <code>shear_modulus_depth</code>: 10 GPa down to 7.5 km, 30 GPa at 15 km, 67 GPa from 35 km, log-linear between) and the rows of calabria2's mesh at their mean depth, with the magnitude change of a rupture on each row.</figcaption></figure>
  <div class="tbl"><table>
    <tr><th>calabria2 row (1 = trench)</th><th>mean depth (km)</th><th>rigidity (GPa)</th><th>Mw shift of a rupture there</th><th>slip of a real earthquake there (&times; constant)</th></tr>
    {row_html}
  </table></div>
</section>

<section id="how-sec">
  <h2 id="how">How PTHA18 does it, and what step 7c ports</h2>
  <figure class="fig">{flow_svg()}<figcaption>Step 7c reads what steps 7 and 7b wrote and adds the variable shear modulus files. No scenario is rebuilt.</figcaption></figure>
  <ol>
    <li><b>Relabel the magnitude.</b> Every scenario keeps its cells and its slip (computed with 30 GPa). Its magnitude is recomputed with the rigidity of each cell: M0 = sum(area &times; slip &times; &mu;(depth)). Shallow scenarios get a lower magnitude, deep ones a higher one. <code>pyptha_v12/variable_mu.py</code>: <code>shear_modulus_depth</code>, <code>variable_mu_Mw</code> (rptha <code>append_variable_mu_variables_to_event_netcdf.R</code> 90-114).</li>
    <li><b>Measure the difference</b> (variable &minus; constant Mw) on the HS scenarios of each magnitude and keep it as a distribution that depends on the magnitude (<code>make_conditional_ecdf</code>, rptha <code>rupture_probabilities.R</code> 1481; <code>compute_rates_all_sources.R</code> 476-516). On a segment only the HS scenarios touching it count.</li>
    <li><b>LEVEL 3 with an observation error.</b> The rates stay functions of the constant-rigidity magnitude, the scale every scenario is built on. The GCMT earthquakes have real magnitudes, so the difference is treated as an error of the catalogue magnitudes when the branch weights are updated. That gives a second set of weights (<code>posterior_prob_with_Mw_error</code>, already in v9's engine but unused).</li>
    <li><b>The variable rates</b> are the same branch curves averaged with those weights: scenario rates, their 5 percentiles, the rate curves, LEVEL 5 (<code>MwRateFunction.with_mw_error</code>; <code>compute_rates_all_sources.R</code> 749-869).</li>
    <li><b>HS and VAUS</b> share the variable FAUS rates with PTHA18's DART curves for that case (a second pair, recovered from PTHA18's published rates like the constant ones). The peak-slip limit (7.5 &times;) is the same.</li>
  </ol>
  <div class="callout good"><span class="lbl">What does not change</span>No scenario changes: same cells, same slip, same sea-floor deformation, same tsunami. Only how often each scenario happens, and only through the LEVEL 3 weights. A zone with many GCMT earthquakes (and deep or shallow ones) feels it more; a zone without, like calabria2, little.</div>
</section>

<section id="calabria-sec">
  <h2 id="calabria">What it changes in calabria2</h2>
  <div class="tbl"><table>
    <tr><th>Mw</th><th>constant: rate of Mw &ge;</th><th>variable, same Mw scale</th><th>ratio</th><th>variable: rate of real Mw &ge;</th><th>ratio</th></tr>
    {rate_html}
  </table></div>
  <p>Rates per year (return period). calabria2 has no GCMT earthquake in LEVEL 3, so the two sets of weights are close (total variation distance {tvd:.3f}): on the constant-magnitude scale the rates move between {min(b/a for _, a, b, _ in rate_rows)*100-100:+.0f}% and {max(b/a for _, a, b, _ in rate_rows)*100-100:+.0f}% at these magnitudes. Counted by their real magnitude the rates are lower, because most of the rate sits on the shallow rows, whose real magnitude is about 0.3 lower. Said the other way round: an earthquake of a given real magnitude in Calabria has more slip, and a bigger tsunami, than the constant rigidity gives it.</p>
  <div class="callout"><span class="lbl">For tsunami maps</span>Use the variable rates with the HS scenarios (<code>variable_mu_rate_*</code> in <code>outputs/hs_slip_fields/summary.csv</code>), as PTHA18's headline maps do. Quote <code>variable_mu_Mw</code> when a scenario is compared with a real earthquake or picked as "the Mw 8" for a map: on calabria2's shallow rows a real Mw 8 is a constant-rigidity Mw of about 8.3.</div>
</section>

<section id="files-sec">
  <h2 id="files">Files step 7c writes</h2>
  <div class="tbl"><table>
    <tr><th>file (in <code>outputs/</code>)</th><th>what it holds</th></tr>
    <tr><td><code>scenario_rates_&lt;zone&gt;_variable_mu.csv</code></td><td>every FAUS scenario: <code>Mw</code> (constant), <code>variable_mu_Mw</code>, <code>rate_mean</code> and 5 percentiles</td></tr>
    <tr><td><code>rate_curves_variable_mu.csv</code>, <code>exceedance_rate_percentiles_variable_mu.csv</code></td><td>as the constant files, with the variable weights</td></tr>
    <tr><td><code>variable_mu_deviation_&lt;zone&gt;.csv</code></td><td>the HS magnitude differences the distribution is built from</td></tr>
    <tr><td><code>logic_tree_branches_&lt;zone&gt;_variable_mu.csv</code></td><td><code>posterior_prob_with_Mw_error</code> = the variable weights of every branch</td></tr>
    <tr><td><code>logic_tree_summary_variable_mu.csv</code></td><td>adds the TVDs (prior to variable posterior, constant to variable posterior) and the summed rate</td></tr>
    <tr><td><code>hs_slip_fields/summary.csv</code>, <code>vaus_slip_fields/summary.csv</code></td><td>new columns <code>variable_mu_Mw</code>, <code>variable_mu_weight_in_family</code>, <code>variable_mu_rate_*</code></td></tr>
    <tr><td><code>inputs/input_&lt;zone&gt;_scratch_variable_mu.json</code></td><td>the engine's input: step 6's plus a <code>variable_shear_modulus</code> block</td></tr>
  </table></div>
  <p>In <code>report.html</code>: a STEP 7c card (this explanation, the curve with the zone's rows, the shift of the HS scenarios, the rate table), and the rupture explorer gives every FAUS and VAUS block its magnitude with the variable rigidity.</p>
</section>

<section id="use-sec">
  <h2 id="use">How to use it</h2>
<pre class="code"># from V9/ (venv ../.venv)
../.venv/Scripts/python.exe from_scratch_v12/generate.py calabria --zone calabria2 --folder calabria2_v11_varmu --ptha false --rupture-size local --variable-mu on
../.venv/Scripts/python.exe calabria2_v11_varmu/steps/step_total.py --skip-official
# or, on an existing v11 folder after steps 7 and 7b:
../.venv/Scripts/python.exe &lt;folder&gt;/steps/step7c_variable_mu.py --force</pre>
  <p>It is off by default: <code>generate.py --variable-mu on</code> makes <code>step_total.py</code> run step 7c after 7b (<code>--skip-hs</code> skips both); in a folder generated with <code>off</code>, <code>step7c_variable_mu.py --force</code> runs it by hand. Zones whose unit sources are normal faults keep the constant rigidity, as in PTHA18.</p>
</section>

<section id="checks-sec">
  <h2 id="checks">How it was checked</h2>
  <ul>
    <li><b>Against PTHA18's published files</b> (<code>validation/validate_v11.py</code>): every published <code>variable_mu_Mw</code> of puysegur2 and kermadectonga2 (FAUS, HS and VAUS, 101,000+ scenarios) to 4e-15; the v11 engine on PTHA18's official puysegur2 input, with PTHA18's own HS catalogue, gives its <code>variable_mu_rate_annual</code> and all 5 percentiles to 7e-13 with every zero rate in place (and the constant <code>rate_annual</code> unchanged, 8e-13); every published HS and VAUS rate of both zones, both rigidities, to 4e-12.</li>
    <li><b>The four DART curves</b> (HS and VAUS, constant and variable rigidity) recovered from PTHA18's published rates: fitted on puysegur2 alone they reproduce kermadectonga2 to 2e-14; each is the derivative of the inverse of a cubic through (0,0) and (1,1), the method of PTHA18's report Section 3.6 (largest residual 4e-4).</li>
    <li><b>calabria2_v11_varmu</b>: every file of steps 1-7b identical to calabria2_v10_q's; step 7c checks that the engine's constant results equal step 7's before copying anything.</li>
    <li><b>Tests</b> (179 passed, 15 skipped): <code>test_variable_mu.py</code> (7: the rigidity curve and its log interpolation, the relabelled magnitude, the conditional ECDF against R's semantics, the rate function with the variable weights, the four weight curves, and PTHA18's published variable_mu_Mw when the files are present).</li>
  </ul>
</section>

<section id="limits-sec">
  <h2 id="limits">Known limitations</h2>
  <ul>
    <li><b>Only PTHA18's curve</b> (Bilek &amp; Lay fit, or PREM through <code>"curve": "prem"</code> in the input's <code>variable_shear_modulus</code> block). A Mediterranean-specific rigidity profile would be a new curve, not PTHA18's.</li>
    <li><b>The depth is PTHA18's:</b> the curve is applied to each unit source's <code>depth</code>, measured below the nearby trench (PTHA18's convention), while Bilek &amp; Lay's depths include the water. PTHA18 does exactly this (its published <code>variable_mu_Mw</code> is reproduced to 4e-15), so v11 does too.</li>
    <li><b>The DART curves come from 18 Pacific and Indian Ocean tsunamis</b> (2006-2016); there are no DART buoys in the Mediterranean. PTHA18 applies them to every zone, because they correct the slip generator, not a place; using them for Calabria is the same assumption.</li>
    <li><b>The segmented official check</b> (kermadectonga2) was not run for the variable rates: the code path is the constant one's (validated to 1e-10) with the variable weights, and each segment uses the HS scenarios touching it, as rptha does.</li>
    <li>Everything in <a href="v10_q.html#limits">v10_q's limitations</a> still applies.</li>
  </ul>
</section>

<nav class="pager" id="pager"></nav>
<footer>Generated by <code>html/docs/figures_src/make_v11_page.py</code> from <code>calabria2_v11_varmu/outputs/</code> (run from V9/). README sections "What v11 changes" and "step7c_variable_mu.py".</footer>
</main>
<aside class="toc" id="toc"></aside>
</div>
<script src="assets/nav.js"></script>
</body>
</html>
"""
open(os.path.join(DOCS, "v11.html"), "w", encoding="utf-8").write(page)
print("wrote", os.path.join(DOCS, "v11.html"))
