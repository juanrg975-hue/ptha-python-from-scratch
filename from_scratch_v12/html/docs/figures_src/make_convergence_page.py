"""Write html/docs/convergence_sources.html from convergence_sources.json
(made by make_convergence_sources.py).

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_convergence_page.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
PKG = os.path.abspath(os.path.join(DOCS, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "lib"))
from slab2_catalog import REGIONS  # noqa: E402
from berryman_params import ZONE_TO_BERRYMAN_SEGMENTS  # noqa: E402

ROWS = {r["zone"]: r for r in json.load(open(os.path.join(HERE, "convergence_sources.json")))}

# PTHA18 zone -> (SLAB2.0 region title of list_regions.py, how it was assigned).
# "raster": the zone's mesh centre lies on that region's SLAB raster (checked on
# the rasters cached in this package). "position": the region that covers the
# zone by geography (its raster was not available to check). Zones not listed
# here are outside every one of the 27 regions, or on a structure (backthrust,
# trough) no SLAB2.0 region models.
ZONE_REGION = {
    "alaskaaleutians": ("Alaska", "raster"),
    "cascadia": ("Cascadia", "raster"),
    "mexico": ("Central America", "position"),
    "izumariana": ("Izu-Bonin", "raster"),
    "kurilsjapan": ("Kamchatka-Kuril Islands-Japan", "raster"),
    "kermadectonga": ("Kermadec", "raster"),
    "kermadectonga2": ("Kermadec", "raster"),
    "makran": ("Makran", "raster"),
    "makran2": ("Makran", "raster"),
    "newguinea": ("New Guinea", "position"),
    "newguinea2": ("New Guinea", "position"),
    "philippine": ("Philippines", "raster"),
    "puysegur": ("Puysegur", "raster"),
    "puysegur2": ("Puysegur", "raster"),
    "ryuku": ("Ryukyu", "position"),
    "sandwich": ("Scotia Sea", "position"),
    "solomon": ("Solomon Islands", "position"),
    "solomon2": ("Solomon Islands", "position"),
    "southamerica": ("South America", "raster"),
    "north_sulawesi": ("Sulawesi", "position"),
    "sunda": ("Sumatra-Java", "position"),
    "sunda2": ("Sumatra-Java", "position"),
    "newhebrides": ("Vanuatu", "position"),
    "newhebrides2": ("Vanuatu", "position"),
}
NOT_IN_ANY = {"hjort": "south of the Puysegur raster (which ends at 50.1 S)",
              "macquarieislandnorth": "south of the Puysegur raster (which ends at 50.1 S)",
              "macquarienorth": "south of the Puysegur raster (which ends at 50.1 S)"}

SOURCE_LABEL = {
    "bird": '<span class="badge good">Bird</span>',
    "mixed": '<span class="badge warn">Bird + some traces</span>',
    "griffin": '<span class="badge bad">Griffin traces</span>',
}


def fmt(v):
    return "" if v in ("", None) else f"{float(v):.1f}"


def diff(r):
    a, b = r["conv_bird_griffin"], r["conv_bird"]
    if b in ("", None) or not a:
        return ""
    d = 100.0 * (float(b) / float(a) - 1.0)
    cls = ' class="n hot"' if abs(d) >= 10 else ' class="n"'
    return f"<td{cls}>{d:+.0f}%</td>"


def source_cell(r):
    if int(r["use_bird_convergence"]) == 0:
        return (f'<span class="badge grey">constant {float(r["constant_mm_per_yr"]):g} mm/yr</span>'
                f'<br><span class="where">(table: {r["source"]})</span>')
    return SOURCE_LABEL[r["source"]]


def pct(r):
    a, b = r["conv_bird_griffin"], r["conv_bird"]
    return abs(100.0 * (float(b) / float(a) - 1.0)) if a and b not in ("", None) else 0.0


def advice(d):
    if d < 1.0:
        return "either (same result)"
    if d < 10.0:
        return f"either (differ by {d:.0f}%)"
    return "<b>bird-griffin</b> to match PTHA18"


def rec(r):
    if int(r["use_bird_convergence"]) == 0:
        text = "either; PTHA18 used neither (see note)"
    else:
        text = advice(pct(r))
    if r["zone"] not in ZONE_TO_BERRYMAN_SEGMENTS:
        text += '<br><span class="where">not runnable yet: no Berryman mapping (note 6)</span>'
    return text


# --- table 1: the 27 regions of list_regions.py ----------------------------
by_region = {}
for z, (reg, how) in ZONE_REGION.items():
    if z in ROWS:
        by_region.setdefault(reg, []).append((z, how))
t1 = []
for _, title, prefix in REGIONS:
    zs = sorted(by_region.get(title, []))
    if not zs:
        t1.append(f'<tr><td>{title}</td><td><code>{prefix}</code></td><td class="muted">no PTHA18 zone</td>'
                  f'<td><span class="badge grey">none</span></td><td>either (same result, note&nbsp;4)</td></tr>')
        continue
    names = ", ".join(f'<a href="#z-{z}">{z}</a>' + ("" if how == "raster" else "*")
                      + ("" if z in ZONE_TO_BERRYMAN_SEGMENTS else "&dagger;") for z, how in zs)
    srcs = sorted({ROWS[z]["source"] if int(ROWS[z]["use_bird_convergence"]) else "constant" for z, _ in zs})
    badges = " ".join(SOURCE_LABEL.get(s, '<span class="badge grey">constant 35 mm/yr</span>') for s in srcs)
    region_advice = ("either; PTHA18 used a constant rate (see note)" if srcs == ["constant"] else
                     advice(max(pct(ROWS[z]) for z, _ in zs if int(ROWS[z]["use_bird_convergence"]))))
    t1.append(f'<tr><td>{title}</td><td><code>{prefix}</code></td><td>{names}</td><td>{badges}</td><td>{region_advice}</td></tr>')

# --- table 2: every PTHA18 subduction zone ----------------------------------
order = sorted(ROWS.values(), key=lambda r: ({"bird": 0, "mixed": 1, "griffin": 2}[r["source"]], r["zone"]))
t2 = []
for r in order:
    z = r["zone"]
    if z in ZONE_REGION:
        reg, how = ZONE_REGION[z]
        region = reg + ("" if how == "raster" else "*")
    else:
        region = '<span class="muted">none' + (" (note 2)" if z in NOT_IN_ANY else " (note 5)") + "</span>"
    cols = f'{r["cols_bird"]} / {r["cols_griffin"]} / {r["cols_davies"]}'
    if r["traces"]:
        cols += '<br><span class="where">' + r["traces"].replace(";", ", ") + "</span>"
    retired = "" if float(r["row_weight"]) > 0 else ' <span class="where">(weight 0)</span>'
    t2.append(f'<tr id="z-{z}"><td><code>{z}</code>{retired}</td><td>{region}</td><td>{source_cell(r)}</td>'
              f'<td>{cols}</td><td class="n">{fmt(r["conv_bird_griffin"])}</td>'
              f'<td class="n">{fmt(r["conv_bird"])}</td>{diff(r)}<td>{rec(r)}</td></tr>')

n_bird = sum(r["source"] == "bird" for r in ROWS.values())
n_mixed = sum(r["source"] == "mixed" for r in ROWS.values())
n_griffin = sum(r["source"] == "griffin" for r in ROWS.values())

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3. Bird or Bird + Griffin, per zone | from_scratch_v12</title>
<link rel="stylesheet" href="assets/style.css">
<style>
  td.hot{{color:#a33030;font-weight:700}}
  .muted{{color:#8a93a3}}
</style>
</head>
<body>
<div class="layout">
<nav class="side" id="side"></nav>
<main>

<section id="top">
  <div class="crumb">Rates</div>
  <h1>3. Bird or Bird + Griffin, per zone</h1>
  <p class="lead">Step 3 takes the plate convergence from a table of plate-boundary pieces. Two tables are in the package, and <code>generate.py --convergence</code> picks one. On most zones they agree. On the zones where PTHA18 did not use Bird but Jonathan Griffin's own plate rates, they can differ by a factor of two or more. This page says which zone is which, measured on PTHA18's own meshes.</p>
  <div class="grid g4">
    <div class="card kpi"><div class="num">{len(ROWS)}</div><div class="lbl">PTHA18 subduction zones measured</div></div>
    <div class="card kpi"><div class="num">{n_bird}</div><div class="lbl">took their convergence from Bird's own steps</div></div>
    <div class="card kpi"><div class="num">{n_mixed}</div><div class="lbl">from Bird plus a few traces</div></div>
    <div class="card kpi"><div class="num">{n_griffin}</div><div class="lbl">from Griffin's (or Davies') traces only</div></div>
  </div>
</section>

<section>
  <h2 id="at-a-glance">At a glance</h2>
  <figure class="fig">
    <svg viewBox="0 0 960 250" role="img" aria-label="Two tables in from_scratch_v12/data/bird feed step 3; the flag --convergence chooses one">
      <defs><marker id="cs-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8a93a3"/></marker></defs>
      <g font-size="12.5">
        <rect x="10" y="20" width="360" height="92" rx="10" fill="#e4f5ec" stroke="#2e9e6b"/>
        <text x="24" y="44" font-weight="700">--convergence bird  (default)</text>
        <text x="24" y="66" fill="#5b6475">Bird (2003)'s public catalogue, PB2002_steps</text>
        <text x="24" y="84" fill="#5b6475">1869 steps of convergent boundary types</text>
        <text x="24" y="102" fill="#5b6475">(SUB, OCB, CCB), as published</text>
        <rect x="10" y="138" width="360" height="92" rx="10" fill="#fcf0e2" stroke="#d9822b"/>
        <text x="24" y="162" font-weight="700">--convergence bird-griffin</text>
        <text x="24" y="184" fill="#5b6475">the table PTHA18 used: 1060 of Bird's steps</text>
        <text x="24" y="202" fill="#5b6475">+ 2645 traces by J. Griffin + 100 by G. Davies</text>
        <text x="24" y="220" fill="#5b6475">(a copy of rptha's sourcezone_traces_table_merged)</text>
        <rect x="430" y="80" width="200" height="90" rx="10" fill="#fff" stroke="#2f6fdf" stroke-width="1.6"/>
        <text x="444" y="106" font-weight="700">step 3</text>
        <text x="444" y="128" fill="#5b6475">each column's top edge</text>
        <text x="444" y="146" fill="#5b6475">to the nearest piece</text>
        <rect x="690" y="20" width="260" height="92" rx="10" fill="#e4f5ec" stroke="#2e9e6b"/>
        <text x="704" y="44" font-weight="700">zones where PTHA18 used Bird</text>
        <text x="704" y="66" fill="#5b6475">both tables: the same or close</text>
        <text x="704" y="84" fill="#5b6475">(except near transforms, see notes)</text>
        <rect x="690" y="138" width="260" height="92" rx="10" fill="#fdecec" stroke="#c43d3d"/>
        <text x="704" y="162" font-weight="700">zones where PTHA18 used Griffin</text>
        <text x="704" y="184" fill="#5b6475">eastern Indonesia, New Guinea:</text>
        <text x="704" y="202" fill="#5b6475">very different; bird-griffin = PTHA18</text>
      </g>
      <g stroke="#8a93a3" stroke-width="1.8" fill="none" marker-end="url(#cs-ar)">
        <path d="M370,66 C400,66 400,110 428,112"/><path d="M370,184 C400,184 400,140 428,138"/>
        <path d="M630,112 C660,112 660,66 688,66"/><path d="M630,138 C660,138 660,184 688,184"/>
      </g>
    </svg>
    <figcaption><b>Where the convergence comes from.</b> Both tables are in <code>from_scratch_v12/data/bird/</code>, so nothing is downloaded and nothing is read from <code>rptha/</code>. The method is the same for both (<a href="step3.html#matching">matching</a>); only the pieces it matches against differ.</figcaption>
  </figure>
  <div class="code"><pre>.venv/Scripts/python.exe from_scratch_v12/generate.py "new guinea" --zone newguinea2 --folder newguinea2_v9 --ptha false                             <span class="cm"># bird (default): 38 mm/yr</span>
.venv/Scripts/python.exe from_scratch_v12/generate.py "new guinea" --zone newguinea2 --folder newguinea2_bg --ptha false --convergence bird-griffin <span class="cm"># PTHA18's table: 96 mm/yr</span></pre></div>
  <p><code>generate.py</code> prints, for any PTHA18 zone, where PTHA18 took its convergence from and what both tables give on PTHA18's mesh. In an existing folder, change <code>BIRD_TABLE</code> in <code>steps/step3_convergence.py</code> and re-run steps 3, 6 (<code>--force</code>), 7 and 9. The choice is recorded in <code>data/convergence_per_column.json</code> (<code>bird_table</code>). Step 8, the official run, always uses <code>bird-griffin</code>, as PTHA18 did.</p>
</section>

<section>
  <h2 id="griffin">What "Griffin's traces" are</h2>
  <p>PTHA18 did not read Bird's file as it is. Its script <code>rptha/R/examples/austptha_template/DATA/BIRD_PLATE_BOUNDARIES/reformat_edited_Bird_convergence.R</code> starts from "a shapefile containing Bird's (2003) plate model" that was edited (Bird's pieces were deleted where PTHA18 wanted other information) and combines it "with the plate motion rates put together by Jonathan Griffin for eastern Indonesia and other new source zones". <code>combine_traces.R</code> then merges four sources into the one table PTHA18's R code reads (<code>EVENT_RATES/config.R</code>): Bird's remaining steps, Griffin's traces, and two zones added by Gareth Davies (Arakan at 23 mm/yr from Socquet et al. 2006, Seram south at 1 mm/yr). The column <code>collator</code> of the table says who supplied each row: <code>Bird2003_subset</code>, <code>JG</code> or <code>GD</code>.</p>
  <p>Why the numbers differ so much where Griffin's traces are used: Bird's model divides the region between Australia and the Pacific into many small plates (Bird's Head, Caroline, Woodlark, Manus, Molucca Sea, Banda Sea ...), and each boundary carries only its own share of the motion. At the New Guinea trench, for example, Bird's steps between the Caroline and Bird's Head plates converge at 22 to 28 mm/yr. Griffin's trace <code>newguinea</code> gives the trench about 70 to 90 mm/yr, most of the Pacific-Australia motion. Neither is "wrong": they are two models of the same plates. <code>bird</code> is Bird's model as published; <code>bird-griffin</code> is PTHA18's.</p>
</section>

<section>
  <h2 id="regions">The 27 regions of list_regions.py</h2>
  <p>Which PTHA18 zones lie on each SLAB2.0 region, and where PTHA18 took their convergence from. A zone marked * is assigned by its position; the others were checked on the region's SLAB raster. A zone marked &dagger; has no Berryman et al. (2015) mapping yet, so it cannot be run with <code>--ptha false</code> (note 6). The last column compares the two tables' measured values: under 1% apart is the same result, 10% or more calls for <code>bird-griffin</code> to match PTHA18. Click a zone for its numbers.</p>
  <div class="tbl"><table class="fixed">
    <colgroup><col style="width:19%"><col style="width:7%"><col style="width:25%"><col style="width:20%"><col style="width:29%"></colgroup>
    <tr><th>Region</th><th>Prefix</th><th>PTHA18 zones</th><th>PTHA18's source</th><th>Which --convergence</th></tr>
    {chr(10).join("    " + r for r in t1)}
  </table></div>
</section>

<section>
  <h2 id="zones">Every PTHA18 subduction zone, measured</h2>
  <p>For each of PTHA18's {len(ROWS)} subduction zones (thrust interface, rake 90; PTHA18's outer-rise zones are normal faults, which this package does not build): which rows of PTHA18's table each column's top edge is matched to (Bird / Griffin / Davies), the traces used, and the zone's convergence with each table, all on PTHA18's own mesh. The <code>bird-griffin</code> value on PTHA18's mesh is PTHA18's own convergence (checked to 1e-15 on 8 zones, <a href="validation.html#tests">tests</a>). Differences of 10% or more are in red.</p>
  <div class="tbl"><table class="fixed">
    <colgroup><col style="width:17%"><col style="width:13%"><col style="width:13%"><col style="width:17%"><col style="width:9%"><col style="width:8%"><col style="width:8%"><col style="width:15%"></colgroup>
    <tr><th>Zone</th><th>SLAB2.0 region</th><th>PTHA18's source</th><th>columns matched: Bird / Griffin / Davies (traces)</th><th class="n">bird-griffin mm/yr</th><th class="n">bird mm/yr</th><th class="n">bird vs b-g</th><th>Which --convergence</th></tr>
    {chr(10).join("    " + r for r in t2)}
  </table></div>
  <p class="where">Made by <code>figures_src/make_convergence_sources.py</code> (the measurement, also written to <code>from_scratch_v12/data/bird/ptha18_convergence_sources.csv</code>, which <code>generate.py</code> reads for its note) and <code>figures_src/make_convergence_page.py</code> (this page). PTHA18's meshes: <code>unit_source_statistics_&lt;zone&gt;.nc</code> from NCI. "(weight 0)": a zone PTHA18 kept in its table but replaced by a newer version (for example <code>kermadectonga</code> by <code>kermadectonga2</code>).</p>
</section>

<section>
  <h2 id="notes">Notes</h2>
  <ul class="tight" style="list-style:none;padding-left:0">
    <li><b>1. puysegur and puysegur2</b>: PTHA18 did not use either table there but a constant 35 mm/yr (<code>use_bird_convergence = 0</code>). Both tables give the same there (21.9 mm/yr on puysegur's mesh, 24.9 on puysegur2's), since PTHA18's table has Bird's own steps there. To use PTHA18's value, set <code>CONVERGENCE_OVERRIDE_MM_PER_YR = 35</code> in step 3 (<a href="step3.html#puysegur">details</a>).</li>
    <li><b>2. Zones PTHA18 used Bird on but where the tables still differ</b> (hjort, the two Macquarie zones, sandwich, newhebrides2): PTHA18's table kept some of Bird's steps of other boundary types: oceanic transforms (OTF), and on the Macquarie Ridge also a spreading ridge (OSR). <code>bird</code> uses only the convergent types (SUB, OCB, CCB), so there it matches a farther convergent step. Measured on PTHA18's meshes: 12 of macquarieislandnorth's 13 columns, 5 of hjort's 13, 4 of newhebrides2's 38 and 1 of sandwich's 14 are matched to a transform in PTHA18's table. On the Macquarie Ridge, where the boundary is mostly a transform, that halves the value (7.2 against 17.0 mm/yr). hjort and the two Macquarie zones lie south of the Puysegur raster (which ends at 50.1 S), on no SLAB2.0 region, so this package cannot build them anyway.</li>
    <li><b>3. solomon, solomon2</b>: one column (of 46 on solomon, 42 on solomon2) is matched to Griffin's <code>trobriand</code> trace (1 mm/yr); <b>sunda, sunda2</b>: the northern 32 of 123 columns (sunda) and 31 of 120 (sunda2) to Davies' Arakan trace (23 mm/yr). The zone values differ by 3% or less.</li>
    <li><b>4. Zones PTHA18 never modelled</b> (Calabria, Caribbean, Hellenic Arc, ...): the Bird + Griffin table has nothing near them, so <code>bird-griffin</code> falls back to Bird's public catalogue and gives exactly the <code>bird</code> result.</li>
    <li><b>5. Eastern-Indonesia zones without a SLAB2.0 region</b> (timor, flores, seram, tolo, sangihe, manus, mussau, trobriand, moresby_trough ...): PTHA18 built them from its own traces and contours; the 27 SLAB2.0 regions do not model these structures, so this package cannot build them.</li>
    <li><b>6. Zones with no Berryman mapping</b> (marked &dagger;): step 1 needs the zone's seismogenic cutoff, which with <code>--ptha false</code> comes from Berryman et al. (2015) through <code>ZONE_TO_BERRYMAN_SEGMENTS</code> in <code>lib/berryman_params.py</code>. Without an entry, step 1 stops. To run one, add the zone there and in <code>ZONE_DOMINANT_SEGMENT</code>, pointing at its Berryman row (for example <code>"sandwich": ["South Sandwich"]</code>).</li>
  </ul>
</section>

<nav class="pager" id="pager"></nav>
<footer>from_scratch_v12 documentation. Numbers from PTHA18's meshes (NCI) and the two tables in <code>from_scratch_v12/data/bird/</code>; see <code>html/docs/figures_src/make_convergence_sources.py</code>.</footer>
</main>
<aside class="toc" id="toc"></aside>
</div>
<script src="assets/nav.js"></script>
</body>
</html>
"""
open(os.path.join(DOCS, "convergence_sources.html"), "w", encoding="utf-8").write(html)
print("wrote convergence_sources.html:", len(ROWS), "zones,", len(t1), "regions")
