"""Write html/docs/zones.html: every zone name generate.py --ptha false accepts
(the keys of lib/berryman_params.ZONE_TO_BERRYMAN_SEGMENTS), with its SLAB
region, the ready-to-copy command and what is known about running it.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_zones_page.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
PKG = os.path.abspath(os.path.join(DOCS, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "lib"))
from berryman_params import (ZONE_TO_BERRYMAN_SEGMENTS, berryman_b_anchor,  # noqa: E402
                             berryman_coupling, berryman_cutoff_km,
                             berryman_mw_max_observed)
from official_geometry_params import slab_product_for_zone  # noqa: E402
from slab2_catalog import REGIONS  # noqa: E402

CONV = {r["zone"]: r for r in json.load(open(os.path.join(HERE, "convergence_sources.json")))}
TITLE = {p: t for _, t, p in REGIONS}

# Zone -> SLAB region prefix (the first argument of generate.py). None: no
# SLAB region covers the structure (PTHA18 built it from its own traces).
ZONE_PREFIX = {
    "alaskaaleutians": "alu", "antilles2": "car", "calabria": "cal", "calabria2": "cal",
    "cascadia": "cas", "hjort": None, "izumariana": "izu", "kermadectonga": "ker",
    "kermadectonga2": "ker", "kurilsjapan": "kur", "makran": "mak", "makran2": "mak",
    "manokwari": None, "manus": None, "mexico": "cam", "newguinea": "png", "newguinea2": "png",
    "newhebrides": "van", "newhebrides2": "van", "philippine": "phi", "puysegur": "puy",
    "puysegur2": "puy", "ryuku": "ryu", "sandwich": "sco", "sangihe": None, "solomon": "sol",
    "solomon2": "sol", "southamerica": "sam", "sunda": "sum", "sunda2": "sum",
    "timor": None, "timortrough": None,
}
# Run end to end (steps 1 to 9) while writing these docs, on 2026-09-28.
RUN = {"alaskaaleutians", "antilles2", "calabria2", "cascadia", "izumariana",
       "kermadectonga2", "kurilsjapan", "makran2", "newguinea2", "philippine",
       "puysegur2", "ryuku", "sandwich", "southamerica", "sunda2"}
# Tried, and stopped for a reason outside this package.
KNOWN = {
    "mexico": "step 1: earthquake.usgs.gov answered <code>AccessDenied</code> for "
              "<code>cam_slab1.0_clip.grd</code>, in the browser too (<a href=\"troubleshooting.html#step1\">what to do</a>)",
}
# Not tried, but with a known reason to expect trouble.
CAUTION = {
    "puysegur": "SLAB1.0 has no <code>puy</code> grid (PTHA18 used a hand-built profile there); use <code>puysegur2</code>",
    "calabria": "the tested version is <code>calabria2</code> (SLAB2.0)",
}

ZONES = sorted(ZONE_TO_BERRYMAN_SEGMENTS)
assert set(ZONES) == set(ZONE_PREFIX), set(ZONES) ^ set(ZONE_PREFIX)


def convergence(z):
    r = CONV.get(z)
    if r is None:
        return '<span class="where">not a PTHA18 zone: both tables give the same</span>'
    if int(r["use_bird_convergence"]) == 0:
        return f'PTHA18 used a constant {float(r["constant_mm_per_yr"]):g} mm/yr (<a href="step3.html#puysegur">step 3</a>)'
    a, b = float(r["conv_bird_griffin"]), float(r["conv_bird"])
    d = abs(100.0 * (b / a - 1.0))
    if d < 1.0:
        return "same with both tables"
    if d < 10.0:
        return f"tables differ by {d:.0f}%"
    return f'<b>use <code>--convergence bird-griffin</code></b> to match PTHA18 ({b:.0f} vs {a:.0f} mm/yr)'


def status(z):
    if ZONE_PREFIX[z] is None:
        return '<span class="badge bad">no SLAB region</span>'
    if z in RUN:
        return '<span class="badge good">run end to end</span>'
    if z in KNOWN:
        return '<span class="badge warn">blocked download</span><br><span class="where">' + KNOWN[z] + "</span>"
    if z in CAUTION:
        return '<span class="badge warn">not tried</span><br><span class="where">' + CAUTION[z] + "</span>"
    return '<span class="badge grey">not tried</span>'


def command(z):
    p = ZONE_PREFIX[z]
    if p is None:
        return '<span class="where">none: no SLAB2.0 or SLAB1.0 region covers this structure</span>'
    first = (f'.venv/Scripts/python.exe from_scratch_v12/generate.py {p} --zone {z} '
             f'--folder {z}_v9 --ptha false\n')
    if z == 'kurilsjapan':
        # step 3 flags the Izu-Bonin tail: steps 1-3 first, then --auto-clip
        return (f'<div class="code"><pre>{first}'
                + ''.join(f'.venv/Scripts/python.exe {z}_v9/steps/{s}\n' for s in (
                    'step1_fetch_slab2.py', 'step2_build_grid.py', 'step3_convergence.py'))
                + f'.venv/Scripts/python.exe {z}_v9/steps/step_total.py --skip-hs --auto-clip</pre></div>')
    return (f'<div class="code"><pre>{first}'
            f'.venv/Scripts/python.exe {z}_v9/steps/step_total.py --skip-hs</pre></div>')


rows = []
order = sorted(ZONES, key=lambda z: (ZONE_PREFIX[z] is None, TITLE.get(ZONE_PREFIX[z] or "", "~"), z))
for z in order:
    p = ZONE_PREFIX[z]
    cutoff, seg = berryman_cutoff_km(z)
    cmin, cpref, cmax = berryman_coupling(z)
    mw = berryman_mw_max_observed(z)
    b = berryman_b_anchor(z)
    region = f'{TITLE[p]} <code>{p}</code>' if p else '<span class="muted">none</span>'
    ptha = "yes" if z in CONV else '<span class="muted">no</span>'
    rows.append(
        f'<tr id="{z}"><td><code><b>{z}</b></code></td><td>{region}</td><td>{slab_product_for_zone(z)}</td>'
        f'<td>{ptha}</td><td>{seg}<br><span class="where">cutoff {cutoff:g} km, coupling {cpref:g} '
        f'({cmin:g} to {cmax:g}), b {b[1]:g}, Mw<sub>max,obs</sub> {mw:g}</span></td>'
        f'<td>{convergence(z)}</td><td>{status(z)}</td></tr>\n'
        f'<tr class="cmd"><td colspan="7">{command(z)}</td></tr>')

n_region = sum(p is not None for p in ZONE_PREFIX.values())
n_run = len(RUN)
n_none = len(ZONES) - n_region

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zones you can run | from_scratch_v12</title>
<link rel="stylesheet" href="assets/style.css">
<style>
  .muted{{color:#8a93a3}}
  tr.cmd td{{border-top:0;padding-top:0}}
  tr.cmd .code{{margin:0}}
  tr.cmd pre{{font-size:11px;padding-right:70px}}
  td code b{{white-space:nowrap}}
</style>
</head>
<body>
<div class="layout">
<nav class="side" id="side"></nav>
<main>

<section id="top">
  <div class="crumb">Your own runs</div>
  <h1>Zones you can run</h1>
  <p class="lead">Every zone name that <code>generate.py --ptha false</code> accepts, with the SLAB region it needs, the command ready to copy, and what is known about running it. The list is exactly the zones in <code>lib/berryman_params.py</code>: with <code>--ptha false</code>, each zone's seismogenic cutoff, coupling, b-value and largest observed magnitude come from Berryman et al. (2015) through that file, so a zone that is not there cannot run. <a href="#add">Adding one</a> takes two lines.</p>
  <div class="grid g4">
    <div class="card kpi"><div class="num">{len(ZONES)}</div><div class="lbl">zone names with a Berryman mapping</div></div>
    <div class="card kpi"><div class="num">{n_region}</div><div class="lbl">of them on a SLAB region, so <code>generate.py</code> can build them</div></div>
    <div class="card kpi"><div class="num">{n_run}</div><div class="lbl">run end to end, steps 1 to 9, while writing these docs</div></div>
    <div class="card kpi"><div class="num">{n_none}</div><div class="lbl">on no SLAB region (eastern Indonesia, south of New Zealand)</div></div>
  </div>
</section>

<section>
  <h2 id="at-a-glance">At a glance</h2>
  <figure class="fig">
    <svg viewBox="0 0 960 200" role="img" aria-label="The command has two names: the SLAB region picks the raster, --zone picks the parameters, and a zone name ending in 2 means SLAB2.0">
      <defs><marker id="zn-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8a93a3"/></marker></defs>
      <g font-size="12.5">
        <rect x="10" y="20" width="940" height="44" rx="10" fill="#f0f2f5" stroke="#8a93a3"/>
        <text x="26" y="47" font-family="monospace" font-size="13.5">generate.py <tspan fill="#2f6fdf" font-weight="700">sco</tspan> --zone <tspan fill="#2e9e6b" font-weight="700">sandwich</tspan> --folder sandwich_v9 --ptha false --discretizer optimal</text>
        <rect x="80" y="104" width="250" height="84" rx="10" fill="#e8f0fd" stroke="#2f6fdf"/>
        <text x="94" y="128" font-weight="700">SLAB region (first argument)</text>
        <text x="94" y="148" fill="#5b6475">which raster to download: one of</text>
        <text x="94" y="166" fill="#5b6475">the 27 of list_regions.py (Scotia Sea)</text>
        <rect x="360" y="104" width="270" height="84" rx="10" fill="#e4f5ec" stroke="#2e9e6b"/>
        <text x="374" y="128" font-weight="700">--zone: the zone name</text>
        <text x="374" y="148" fill="#5b6475">which Berryman row: cutoff, coupling,</text>
        <text x="374" y="166" fill="#5b6475">b, Mw_max (South Sandwich)</text>
        <rect x="660" y="104" width="290" height="84" rx="10" fill="#fdf6e3" stroke="#b7791f"/>
        <text x="674" y="128" font-weight="700">does the name end in "2"?</text>
        <text x="674" y="148" fill="#5b6475">yes: SLAB2.0 (ScienceBase)</text>
        <text x="674" y="166" fill="#5b6475">no: SLAB1.0 (earthquake.usgs.gov)</text>
      </g>
      <g stroke="#8a93a3" stroke-width="1.8" fill="none" marker-end="url(#zn-ar)">
        <path d="M130,64 L180,102"/><path d="M270,64 L470,102"/><path d="M330,64 C560,80 740,80 780,102"/>
      </g>
    </svg>
    <figcaption><b>Two names, two jobs.</b> The first argument chooses the SLAB raster, <code>--zone</code> chooses the zone's parameters, and the zone name's last character chooses SLAB1.0 or SLAB2.0 (PTHA18's own naming rule). The same region can hold several zones: <code>ker</code> holds kermadectonga (SLAB1.0) and kermadectonga2 (SLAB2.0).</figcaption>
  </figure>
</section>

<section>
  <h2 id="rules">Choosing the names</h2>
  <ul class="tight">
    <li><b>The first argument</b> is a SLAB region: its three-letter prefix or its title from <code>list_regions.py</code> (<code>sco</code> or <code>"scotia sea"</code>). It is never a zone name: <code>generate.py antilles2</code> fails with <code>no SLAB2.0 region matches 'antilles2'</code>.</li>
    <li><b><code>--zone</code></b> is the zone name from the table below, spelled exactly (<code>ryuku</code>, not ryukyu: that is PTHA18's spelling). Without it, <code>generate.py</code> guesses from the region's title, which works only when one zone name looks like it (<code>cascadia</code>); pass it always and nothing is left to chance.</li>
    <li><b>The trailing "2"</b> is part of the name, not a version you choose: <code>kermadectonga2</code> and <code>kermadectonga</code> are two different PTHA18 zones on two different SLAB products. A name that is not in the table (for example <code>cascadia2</code>) has no Berryman mapping, and step 1 stops.</li>
    <li><b><code>--folder</code></b> is any name you like; the commands below use <code>&lt;zone&gt;_v9</code> (add <code>--segmented true</code> for LEVEL 0 with segments). They leave out <code>--discretizer optimal</code> and <code>--convergence bird</code>, which are the defaults, and skip step 7b (<code>--skip-hs</code>) to keep a first run short.</li>
  </ul>
</section>

<section>
  <h2 id="zones">All {len(ZONES)} zones</h2>
  <p>Sorted by SLAB region. "PTHA18" says whether PTHA18 modelled the zone (then step 8 can compare with it). The Berryman column is the dominant segment and the values step 1 and step 6 use. "Convergence" compares the two plate tables on PTHA18's mesh (<a href="convergence_sources.html">details</a>). The status is what happened when the zone was run for these docs; "not tried" means <code>generate.py</code> works but steps 1 to 9 were not run.</p>
  <div class="tbl"><table class="fixed">
    <colgroup><col style="width:17%"><col style="width:14%"><col style="width:8%"><col style="width:7%"><col style="width:21%"><col style="width:17%"><col style="width:16%"></colgroup>
    <tr><th>Zone (<code>--zone</code>)</th><th>SLAB region (first argument)</th><th>SLAB</th><th>PTHA18</th><th>Berryman et al. (2015)</th><th>Convergence</th><th>Status</th></tr>
{chr(10).join(rows)}
  </table></div>
  <p class="where">Made by <code>figures_src/make_zones_page.py</code> from <code>lib/berryman_params.py</code>, <code>lib/slab2_catalog.py</code> and <code>figures_src/convergence_sources.json</code>. Every command was checked with <code>generate.py</code> while writing this page.</p>
</section>

<section>
  <h2 id="add">Adding a zone</h2>
  <p>A zone that is not in the table needs two lines in <code>lib/berryman_params.py</code>, pointing at its row of Berryman et al. (2015) (the rows are the dictionary <code>BERRYMAN_ROWS</code> in the same file). This is how <code>sandwich</code> was added:</p>
  <div class="code"><pre># in ZONE_TO_BERRYMAN_SEGMENTS: the Berryman row(s) the zone spans
"sandwich": ["South Sandwich"],
# in ZONE_DOMINANT_SEGMENT: the one row whose cutoff the zone uses
"sandwich": "South Sandwich",</pre></div>
  <p>Then run <code>generate.py</code> again: its report should show a cutoff, coupling and Mw<sub>max,obs</sub>, not <code>NOT AVAILABLE</code>. If the name ends in "2", step 1 uses SLAB2.0; otherwise SLAB1.0. The zone also needs a SLAB region that covers it: the six zones marked "no SLAB region" have a Berryman mapping but nothing to build a mesh from.</p>
</section>

<nav class="pager" id="pager"></nav>
<footer>from_scratch_v12 documentation. Checked on 2026-09-28: every command with <code>generate.py</code>; the zones marked "run end to end" through steps 1 to 9.</footer>
</main>
<aside class="toc" id="toc"></aside>
</div>
<script src="assets/nav.js"></script>
</body>
</html>
"""
out = os.path.join(DOCS, "zones.html")
open(out, "w", encoding="utf-8", newline="\n").write(html)
print(f"wrote zones.html: {len(ZONES)} zones, {n_region} on a SLAB region, {n_run} run end to end")
